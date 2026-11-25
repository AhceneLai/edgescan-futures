"""Chargement et nettoyage des barres journalières OHLCV (yfinance, CSV ou synthétique).

Convention de sortie, quelle que soit la source :
    index   : DatetimeIndex nommé "date", sans fuseau, une ligne par séance, trié
    colonnes: open, high, low, close (float) et volume (float, NaN si inconnu)

La date d'une barre est le jour où la séance Globex se termine (17 h, heure de New York).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

PRICES = ["open", "high", "low", "close"]
OHLCV = [*PRICES, "volume"]

EXCHANGE_TZ = ZoneInfo("America/New_York")
SESSION_CLOSE = time(17, 0)  # fin de la séance Globex, heure de New York
REFRESH_OVERLAP_DAYS = 7  # jours re-téléchargés à chaque mise à jour (Yahoo corrige parfois les dernières barres)

# fetch(symbol, start, years) -> barres brutes au format Yahoo
Fetcher = Callable[[str, date | None, int], pd.DataFrame]


class DataError(RuntimeError):
    """Données absentes ou inutilisables."""


# --------------------------------------------------------------------------- yfinance


def _fetch_yahoo(symbol: str, start: date | None, years: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:  # dépendance optionnelle
        raise ImportError('yfinance n\'est pas installé : uv pip install -e ".[live]"') from e

    ticker = yf.Ticker(symbol)
    if start is None:
        return ticker.history(period=f"{years}y", interval="1d", auto_adjust=False)
    return ticker.history(start=start.isoformat(), interval="1d", auto_adjust=False)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Met des barres brutes (Yahoo ou CSV) au format standard, sans rien supprimer."""
    df = raw.rename(columns=str.lower)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"colonnes manquantes : {missing}")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(EXCHANGE_TZ).tz_localize(None)
    df = df[OHLCV].astype(float)
    df.index = idx.normalize().rename("date")
    return df


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Supprime les barres inutilisables. Renvoie (barres propres, nombre de barres retirées par motif)."""
    df = df.sort_index()
    report: dict[str, int] = {}

    def drop(name: str, bad: pd.Series) -> None:
        nonlocal df
        report[name] = int(bad.sum())
        df = df[~bad]

    drop("duplicates", df.index.duplicated(keep="last"))
    drop("missing_price", df[PRICES].isna().any(axis=1))
    # Ex. CL le 20/04/2020 (-37,63 $) : réel, mais inexploitable pour des ratios et des rendements.
    drop("non_positive_price", (df[PRICES] <= 0).any(axis=1))
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    drop("incoherent_range", (df["high"] < oc_max) | (df["low"] > oc_min))
    # O = H = L = C : prix de règlement sans vraie séance (fréquent sur GC=F entre 2016 et 2019).
    drop("flat_bar", df["high"] == df["low"])

    # Volume 0 sur une vraie séance = donnée manquante chez Yahoo, pas une séance sans échange.
    bad_volume = df["volume"] <= 0
    report["unknown_volume"] = int(bad_volume.sum())
    df = df.assign(volume=df["volume"].mask(bad_volume))
    return df, report


def drop_incomplete(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Retire la dernière barre si sa séance n'est pas encore terminée (17 h à New York)."""
    if df.empty:
        return df
    now = (now or datetime.now(EXCHANGE_TZ)).astimezone(EXCHANGE_TZ)
    last_close = datetime.combine(df.index[-1].date(), SESSION_CLOSE, tzinfo=EXCHANGE_TZ)
    return df.iloc[:-1] if now < last_close else df


def load_yfinance(
    symbol: str,
    years: int = 12,
    cache_dir: str | Path | None = None,
    key: str | None = None,
    fetch: Fetcher = _fetch_yahoo,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Barres journalières propres d'un contrat continu Yahoo (ex. "ES=F").

    Avec cache_dir, les barres sont gardées dans <cache_dir>/<key>.csv et seules les dernières
    séances sont re-téléchargées aux appels suivants.
    """
    cache = Path(cache_dir) / f"{key or symbol.replace('=', '_')}.csv" if cache_dir else None

    cached = load_csv(cache) if cache and cache.exists() else None
    start = cached.index[-1].date() - timedelta(days=REFRESH_OVERLAP_DAYS) if cached is not None else None

    raw = fetch(symbol, start, years)
    if raw is None or raw.empty:
        if cached is None:
            raise DataError(f"aucune donnée reçue de Yahoo pour {symbol}")
        log.warning("%s : aucune nouvelle donnée, utilisation du cache", symbol)
        return drop_incomplete(cached, now)

    df = normalize(raw)
    if cached is not None:
        df = pd.concat([cached, df])  # en cas de doublon, clean_ohlcv garde la version la plus récente
    df, report = clean_ohlcv(df)
    df = drop_incomplete(df, now)
    if df.empty:
        raise DataError(f"aucune barre exploitable pour {symbol}")

    removed = {k: v for k, v in report.items() if v and k != "duplicates"}
    if removed:
        log.info("%s : nettoyage %s", symbol, removed)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


# --------------------------------------------------------------------------- autres sources


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lit un CSV date, open, high, low, close, volume (format écrit par load_yfinance)."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"fichier introuvable : {path}")
    raw = pd.read_csv(path, index_col=0, parse_dates=[0])
    return normalize(raw)


def make_synthetic(key: str, n_days: int = 2600, seed: int = 0) -> pd.DataFrame:
    # TODO (tâche 1.7) : générer des prix simulés pour tester sans internet
    raise NotImplementedError


def load_all(contracts: dict, source: str, raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """Charge chaque contrat de la config selon la source : yfinance, csv ou synthetic."""
    out = {}
    for key, spec in contracts.items():
        if source == "yfinance":
            out[key] = load_yfinance(spec["symbol"], cache_dir=raw_dir, key=key)
        elif source == "csv":
            out[key] = load_csv(Path(raw_dir) / f"{key}.csv")
        elif source == "synthetic":
            out[key] = make_synthetic(key)
        else:
            raise ValueError(f"source inconnue : {source}")
    return out
