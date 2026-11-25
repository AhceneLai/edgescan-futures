"""Signaux -> trades -> issues labellisées (stop / target / timeout), frais inclus."""
import pandas as pd

OUTCOMES = ["stop", "timeout", "target"]


def build_trades(prices: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    # TODO: entrée à l'ouverture de J+1, stop/objectif en ATR, stop si les deux touchés
    raise NotImplementedError
