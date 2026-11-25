"""Indicateurs calculés à la clôture de J, avec des données <= J uniquement."""
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # TODO: Average True Range
    raise NotImplementedError


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: SMA20/50/200, RSI, Bollinger, NR7, gap, momentum... normalisés par l'ATR
    raise NotImplementedError
