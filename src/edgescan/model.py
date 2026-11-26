"""Probabilités calibrées des 3 issues d'un trade : P(stop), P(sortie à la clôture), P(objectif).

3.3 BaselineModel : fréquences historiques par setup et par sens. C'est la référence à battre :
    un modèle ML qui ne fait pas mieux n'apporte rien.
3.4 ProbModel : gradient boosting (HistGradientBoosting de scikit-learn, même famille d'algorithme
    que LightGBM, sans dépendance système) + calibration sur plis temporels :
      1. sur chaque pli, un modèle apprend sur le passé et prédit la période qui suit ;
      2. un calibrateur multinomial (régression logistique sur les log-probabilités) apprend à
         corriger ces prédictions « hors échantillon », pour les 3 issues ensemble ;
      3. le modèle final apprend sur tout l'historique, et ses sorties passent par le calibrateur.
3.5 explain : les 3 facteurs qui pèsent le plus sur P(objectif), valeurs SHAP.
3.6 similar_trades : les 25 trades passés les plus proches et ce qu'ils ont donné.

« Calibré » veut dire : parmi les trades auxquels le modèle donne 60 %, environ 60 % doivent
réussir. C'est ce qui rend les probabilités utilisables pour calculer une espérance (phase 5).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors

from .dataset import FEATURE_LABELS, OUTCOMES, TRADE_FEATURES

PROBA_COLUMNS = [f"p_{o}" for o in OUTCOMES]  # p_stop, p_close, p_target
TARGET = OUTCOMES.index("target")

# Volontairement prudent : peu de feuilles, beaucoup d'exemples par feuille, régularisation.
# Les marchés sont bruités ; un modèle trop souple apprend le bruit.
GBM_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def _codes(outcome: pd.Series) -> np.ndarray:
    return outcome.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()


def _full_proba(model, X: np.ndarray) -> np.ndarray:
    """Probabilités sur les 3 issues, même si le modèle n'en a vu que 2 à l'entraînement."""
    out = np.zeros((len(X), len(OUTCOMES)))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def _logit_inputs(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1.0))


# --------------------------------------------------------------------------- 3.3


class BaselineModel:
    """Fréquence de chaque issue par (setup, sens), lissée vers la fréquence globale."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing  # poids, en nombre de trades, de la fréquence globale

    def fit(self, trades: pd.DataFrame) -> BaselineModel:
        onehot = pd.get_dummies(trades["outcome"]).reindex(columns=OUTCOMES, fill_value=0).astype(float)
        self.global_ = onehot.mean()
        grouped = onehot.groupby([trades["setup"], trades["direction"]])
        counts, n = grouped.sum(), grouped.size()
        self.table_ = (counts + self.smoothing * self.global_).div(n + self.smoothing, axis=0)
        self.n_ = n
        return self

    def predict_proba(self, trades: pd.DataFrame) -> pd.DataFrame:
        keys = pd.MultiIndex.from_frame(trades[["setup", "direction"]])
        p = self.table_.reindex(keys).to_numpy()
        p = np.where(np.isnan(p), self.global_.to_numpy(), p)  # setup jamais vu : fréquence globale
        return pd.DataFrame(p, columns=PROBA_COLUMNS, index=trades.index)


# --------------------------------------------------------------------------- 3.4 / 3.5


class ProbModel:
    def __init__(self, cfg: dict):
        """cfg : section `model` de config.yaml."""
        self.cfg = cfg
        self.min_samples = int(cfg.get("min_train_samples", 400))
        self.folds = int(cfg.get("calibration_folds", 4))

    def fit(self, trades: pd.DataFrame) -> ProbModel:
        """trades : sortie de dataset.build_dataset, triée par date d'entrée."""
        if not trades["entry_date"].is_monotonic_increasing:
            raise ValueError("les trades doivent être triés par date d'entrée (plis temporels)")
        self.baseline_ = BaselineModel().fit(trades)
        self.n_train_ = len(trades)
        self.uses_baseline_ = len(trades) < self.min_samples or trades["outcome"].nunique() < len(OUTCOMES)
        if self.uses_baseline_:
            return self  # trop peu d'exemples : les fréquences historiques sont plus fiables

        X, y = trades[TRADE_FEATURES].to_numpy(float), _codes(trades["outcome"])
        seed = self.cfg.get("random_state", 42)

        oof, oof_y = [], []
        for past, nxt in TimeSeriesSplit(self.folds).split(X):
            fold = HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed).fit(X[past], y[past])
            oof.append(_full_proba(fold, X[nxt]))
            oof_y.append(y[nxt])
        self.calibrator_ = LogisticRegression(C=1.0).fit(_logit_inputs(np.vstack(oof)), np.concatenate(oof_y))

        self.gbm_ = HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed).fit(X, y)
        self.explainer_ = None
        return self

    def predict_proba(self, trades: pd.DataFrame) -> pd.DataFrame:
        if self.uses_baseline_:
            return self.baseline_.predict_proba(trades)
        raw = _full_proba(self.gbm_, trades[TRADE_FEATURES].to_numpy(float))
        p = _full_proba(self.calibrator_, _logit_inputs(raw))
        return pd.DataFrame(p, columns=PROBA_COLUMNS, index=trades.index)

    def explain(self, trades: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Pour chaque trade, les `top` features qui pèsent le plus sur P(objectif).

        Valeurs SHAP du modèle final : > 0 = rapproche de l'objectif, < 0 = en éloigne.
        Elles sont en log-cotes, pas en points de %, mais leur signe et leur ordre sont fiables.
        """
        if self.uses_baseline_:
            return [[] for _ in range(len(trades))]
        import shap  # import lent : seulement quand on en a besoin

        if self.explainer_ is None:
            self.explainer_ = shap.TreeExplainer(self.gbm_)
        X = trades[TRADE_FEATURES].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = np.asarray(self.explainer_.shap_values(X))[..., TARGET]

        out = []
        for row_values, row_x in zip(values, X, strict=True):
            order = np.argsort(-np.abs(row_values))[:top]
            out.append(
                [
                    {
                        "feature": TRADE_FEATURES[i],
                        "label": FEATURE_LABELS[TRADE_FEATURES[i]],
                        "value": float(row_x[i]),
                        "shap": float(row_values[i]),
                    }
                    for i in order
                ]
            )
        return out


# --------------------------------------------------------------------------- 3.6

SIMILARITY_FEATURES = [f for f in TRADE_FEATURES if not f.startswith("setup_")]


def similar_trades(history: pd.DataFrame, query: pd.Series, k: int = 25) -> pd.DataFrame:
    """Les k trades passés les plus proches de `query` (même setup, entrés avant la date du signal).

    Distance euclidienne sur les features standardisées. Renvoie les trades triés du plus proche
    au plus lointain, avec une colonne `distance`.
    """
    past = history[(history["setup"] == query["setup"]) & (history["entry_date"] <= query["date"])]
    if past.empty:
        return past.assign(distance=pd.Series(dtype=float))

    X = past[SIMILARITY_FEATURES].astype(float)
    center = X.median()
    scale = X.std().replace(0, 1).fillna(1)
    Z = ((X.fillna(center) - center) / scale).to_numpy()
    q = ((query[SIMILARITY_FEATURES].astype(float).fillna(center) - center) / scale).to_numpy()

    nn = NearestNeighbors(n_neighbors=min(k, len(past))).fit(Z)
    dist, idx = nn.kneighbors(q.reshape(1, -1))
    return past.iloc[idx[0]].assign(distance=dist[0])


def summarize_similar(similar: pd.DataFrame) -> dict:
    """Fréquence de chaque issue et R net moyen parmi les trades comparables."""
    if similar.empty:
        return {"n": 0}
    freq = similar["outcome"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0.0)
    return {"n": len(similar), **{f"freq_{o}": float(freq[o]) for o in OUTCOMES}, "mean_net_r": float(similar["net_r"].mean())}
