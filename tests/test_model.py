"""Tests du modèle de probabilités (3.3 à 3.6) sur des trades artificiels dont on connaît la vérité."""

import numpy as np
import pandas as pd
import pytest

from edgescan.dataset import OUTCOMES, TRADE_FEATURES
from edgescan.model import PROBA_COLUMNS, BaselineModel, ProbModel, similar_trades, summarize_similar

CFG = {"min_train_samples": 400, "calibration_folds": 4, "random_state": 0}


def make_trades(n=3000, seed=0):
    """Trades artificiels : P(objectif) dépend fortement de `mom5_atr`, le reste est du bruit."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, len(TRADE_FEATURES))), columns=TRADE_FEATURES)
    setups = rng.choice(["trend_pullback", "donchian_breakout"], n)
    for s in ["trend_pullback", "donchian_breakout", "nr7_breakout", "bollinger_reversion"]:
        x[f"setup_{s}"] = (setups == s).astype(float)
    p_target = 1 / (1 + np.exp(-(2.0 * x["mom5_atr"] - 1.0)))
    u = rng.uniform(size=n)
    outcome = np.where(u < p_target, "target", np.where(u < p_target + (1 - p_target) / 2, "stop", "close"))
    meta = pd.DataFrame(
        {
            "date": pd.bdate_range("2010-01-01", periods=n),
            "symbol": "ES",
            "setup": setups,
            "direction": rng.choice(["long", "short"], n),
            "outcome": outcome,
            "net_r": np.where(outcome == "target", 1.5, np.where(outcome == "stop", -1.0, 0.1)),
        }
    )
    meta["entry_date"] = meta["date"] + pd.Timedelta(days=1)
    return pd.concat([meta, x], axis=1), p_target


def brier(p: pd.DataFrame, outcome: pd.Series) -> float:
    y = pd.get_dummies(outcome).reindex(columns=OUTCOMES, fill_value=0).to_numpy(float)
    return float(np.mean(np.sum((p.to_numpy() - y) ** 2, axis=1)))


@pytest.fixture(scope="module")
def data():
    trades, truth = make_trades()
    train, test = trades.iloc[:2400], trades.iloc[2400:].reset_index(drop=True)
    model = ProbModel(CFG).fit(train)
    return train, test, truth.iloc[2400:].reset_index(drop=True), model


# --------------------------------------------------------------------------- 3.3


def test_baseline_frequencies_by_setup_and_direction():
    trades = pd.DataFrame(
        {
            "setup": ["a"] * 4 + ["b"] * 4,
            "direction": "long",
            "outcome": ["target", "target", "target", "stop", "stop", "stop", "close", "close"],
        }
    )
    p = BaselineModel(smoothing=0).fit(trades).predict_proba(trades)
    assert p.iloc[0].tolist() == [0.25, 0.0, 0.75]  # setup a : 3 objectifs sur 4
    assert p.iloc[4].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_baseline_smoothing_and_unknown_setup():
    trades = pd.DataFrame({"setup": ["a", "b"], "direction": "long", "outcome": ["target", "stop"]})
    model = BaselineModel(smoothing=1).fit(trades)
    p = model.predict_proba(pd.DataFrame({"setup": ["a", "zzz"], "direction": "long"}))
    assert p.iloc[0]["p_target"] == pytest.approx((1 + 0.5) / 2)  # tiré vers la moyenne globale
    assert p.iloc[1].tolist() == pytest.approx([0.5, 0.0, 0.5])  # inconnu : fréquence globale


# --------------------------------------------------------------------------- 3.4


def test_probabilities_are_valid(data):
    _, test, _, model = data
    p = model.predict_proba(test)
    assert list(p.columns) == PROBA_COLUMNS
    assert ((p >= 0) & (p <= 1)).all().all()
    assert np.allclose(p.sum(axis=1), 1)


def test_model_beats_baseline_when_there_is_signal(data):
    train, test, _, model = data
    baseline = BaselineModel().fit(train).predict_proba(test)
    assert brier(model.predict_proba(test), test["outcome"]) < brier(baseline, test["outcome"]) - 0.02


def test_model_is_calibrated(data):
    _, test, truth, model = data
    p = model.predict_proba(test)["p_target"]
    hit = (test["outcome"] == "target").astype(float)
    bins = pd.cut(p, [0, 0.2, 0.4, 0.6, 0.8, 1])
    grouped = pd.DataFrame({"p": p, "hit": hit}).groupby(bins, observed=True)
    table = grouped.agg(n=("hit", "size"), p=("p", "mean"), hit=("hit", "mean"))
    table = table[table["n"] >= 60]
    assert len(table) >= 3
    assert (table["p"] - table["hit"]).abs().max() < 0.12  # « 60 % » arrive ~60 % du temps (± bruit)
    assert np.corrcoef(p, truth)[0, 1] > 0.9  # et suit la vraie probabilité


def test_small_dataset_falls_back_to_baseline():
    trades, _ = make_trades(n=200)
    model = ProbModel(CFG).fit(trades)
    assert model.uses_baseline_
    pd.testing.assert_frame_equal(model.predict_proba(trades), BaselineModel().fit(trades).predict_proba(trades))
    assert model.explain(trades.iloc[:2]) == [[], []]


def test_unsorted_trades_are_rejected():
    trades, _ = make_trades(n=500)
    with pytest.raises(ValueError, match="triés"):
        ProbModel(CFG).fit(trades.iloc[::-1])


# --------------------------------------------------------------------------- 3.5


def test_explain_finds_the_driving_feature(data):
    _, test, _, model = data
    explanations = model.explain(test.iloc[:20])
    assert len(explanations) == 20
    assert all(len(e) == 3 for e in explanations)
    top = [e[0]["feature"] for e in explanations]
    assert top.count("mom5_atr") >= 18
    first = explanations[0][0]
    assert np.sign(first["shap"]) == np.sign(test.iloc[0]["mom5_atr"])  # momentum favorable => pousse vers l'objectif
    assert first["label"] == "momentum 5 jours"


# --------------------------------------------------------------------------- 3.6


def test_similar_trades(data):
    train, _, _, _ = data
    query = train.iloc[2000].copy()
    query["date"] = query["entry_date"]  # signal le lendemain : le trade 2000 est déjà connu
    similar = similar_trades(train, query, k=25)
    assert len(similar) == 25
    assert similar.index[0] == 2000 and similar["distance"].iloc[0] == pytest.approx(0)  # lui-même
    assert (similar["setup"] == query["setup"]).all()
    assert (similar["entry_date"] <= query["date"]).all()
    assert similar["distance"].is_monotonic_increasing


def test_similar_trades_never_look_ahead(data):
    train, _, _, _ = data
    query = train.iloc[100].copy()
    query["date"] = train["date"].iloc[50]
    similar = similar_trades(train, query)
    assert (similar["entry_date"] <= query["date"]).all()


def test_summarize_similar():
    similar = pd.DataFrame({"outcome": ["target", "stop", "stop", "close"], "net_r": [1.5, -1, -1, 0.5]})
    s = summarize_similar(similar)
    assert s == {"n": 4, "freq_stop": 0.5, "freq_close": 0.25, "freq_target": 0.25, "mean_net_r": 0.0}
    assert summarize_similar(similar.iloc[0:0]) == {"n": 0}
