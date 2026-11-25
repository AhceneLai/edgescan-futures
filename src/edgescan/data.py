"""Chargement des données journalières OHLCV (yfinance, CSV ou synthétique)."""
import pandas as pd


def load_yfinance(symbol: str, years: int = 12) -> pd.DataFrame:
    # TODO: télécharger les barres journalières (ES=F, NQ=F...) et les nettoyer
    raise NotImplementedError


def load_csv(path: str) -> pd.DataFrame:
    # TODO: lire data/raw/<CLE>.csv (date, open, high, low, close, volume)
    raise NotImplementedError


def make_synthetic(key: str, n_days: int = 2600, seed: int = 0) -> pd.DataFrame:
    # TODO: générer des prix simulés pour tester sans internet
    raise NotImplementedError


def load_all(contracts: dict, source: str) -> dict[str, pd.DataFrame]:
    # TODO: charger chaque contrat de la config selon la source choisie
    raise NotImplementedError
