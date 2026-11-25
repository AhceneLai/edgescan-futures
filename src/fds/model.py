"""Probabilités calibrées : P(objectif), P(stop), P(sortie à la clôture)."""
import pandas as pd


class ProbModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, trades: pd.DataFrame) -> "ProbModel":
        # TODO: gradient boosting + CalibratedClassifierCV (plis temporels)
        raise NotImplementedError

    def predict_proba(self, trades: pd.DataFrame) -> pd.DataFrame:
        # TODO: colonnes p_stop, p_timeout, p_target
        raise NotImplementedError
