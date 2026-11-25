"""Espérance nette, taille de position et Monte Carlo du P&L de la journée."""
import pandas as pd


def expected_r(probs: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    # TODO: P(obj)*R_obj - P(stop) + P(clôture)*E[R|clôture] - frais
    raise NotImplementedError


def position_size(trades: pd.DataFrame, cfg: dict) -> pd.Series:
    # TODO: contrats = floor(capital * risque% / (distance_stop * multiplicateur))
    raise NotImplementedError


def simulate_day(picked: pd.DataFrame, cfg: dict) -> dict:
    # TODO: Monte Carlo -> espérance, P(journée gagnante), VaR 95 %
    raise NotImplementedError
