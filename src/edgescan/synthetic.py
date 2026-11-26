"""Barres journalières synthétiques, reproductibles, pour tester sans internet.

Le modèle reprend les traits des vrais futures qui comptent pour les setups :
* volatilité qui change par régimes (calme / agité) : log-volatilité autorégressive ;
* queues épaisses (loi de Student) : des journées extrêmes plus fréquentes qu'avec une loi normale ;
* un gap d'ouverture puis un mouvement en séance ;
* des plus hauts / plus bas au-delà de l'ouverture et de la clôture, des prix arrondis au tick ;
* un volume plus fort les jours de grands mouvements ;
* uniquement des jours de cotation CME (pas de week-ends ni de jours fériés).

Même (key, seed, n_days, start) => exactement les mêmes barres, sur toutes les machines.
Ces données n'ont aucun avantage exploitable : un modèle qui « gagne » dessus est en surapprentissage.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sessions import trading_days


@dataclass(frozen=True)
class SyntheticSpec:
    start_price: float
    annual_vol: float  # volatilité annuelle moyenne des rendements
    tick: float
    base_volume: float


SPECS = {
    "ES": SyntheticSpec(start_price=2050.0, annual_vol=0.17, tick=0.25, base_volume=1_500_000),
    "NQ": SyntheticSpec(start_price=4250.0, annual_vol=0.22, tick=0.25, base_volume=500_000),
    "CL": SyntheticSpec(start_price=55.0, annual_vol=0.35, tick=0.01, base_volume=600_000),
    "GC": SyntheticSpec(start_price=1200.0, annual_vol=0.17, tick=0.10, base_volume=200_000),
}
DEFAULT_SPEC = SyntheticSpec(start_price=100.0, annual_vol=0.20, tick=0.01, base_volume=100_000)

TRADING_DAYS_PER_YEAR = 252
VOL_PERSISTENCE = 0.97  # proche de 1 : les régimes de volatilité durent plusieurs semaines
VOL_OF_VOL = 0.12
STUDENT_DF = 4
GAP_SHARE = 0.05  # part de la variance quotidienne dans le gap : Globex rouvre 1 h après la clôture, gaps faibles


def make_synthetic(key: str, n_days: int = 2600, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    """n_days barres journalières au format standard (voir edgescan.data)."""
    if n_days < 1:
        raise ValueError("n_days doit être >= 1")
    spec = SPECS.get(key, DEFAULT_SPEC)
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])

    # Assez de jours calendaires pour obtenir n_days séances (~252 par an).
    days = trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=int(n_days * 1.5) + 30))[:n_days]
    n = len(days)

    # Volatilité quotidienne par régimes : log-vol AR(1) centrée sur la volatilité moyenne.
    daily_vol = spec.annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    log_vol = np.empty(n)
    log_vol[0] = 0.0
    shocks = rng.normal(0.0, VOL_OF_VOL, n)
    for t in range(1, n):
        log_vol[t] = VOL_PERSISTENCE * log_vol[t - 1] + shocks[t]
    vol = daily_vol * np.exp(log_vol - log_vol.var() / 2)

    def student(size: int) -> np.ndarray:  # variance 1
        return rng.standard_t(STUDENT_DF, size) / np.sqrt(STUDENT_DF / (STUDENT_DF - 2))

    gap = vol * np.sqrt(GAP_SHARE) * student(n)
    intraday = vol * np.sqrt(1 - GAP_SHARE) * student(n)
    drift = 0.04 / TRADING_DAYS_PER_YEAR - vol**2 / 2  # légère tendance haussière, comme les indices

    log_close = np.log(spec.start_price) + np.cumsum(drift + gap + intraday)
    close = np.exp(log_close)
    open_ = np.exp(log_close - intraday)

    # Mèches : excursions au-delà de l'ouverture et de la clôture, proportionnelles à la volatilité.
    upper = np.abs(rng.normal(0.0, 0.45, n)) * vol
    lower = np.abs(rng.normal(0.0, 0.45, n)) * vol
    high = np.maximum(open_, close) * np.exp(upper)
    low = np.minimum(open_, close) * np.exp(-lower)

    tick = spec.tick
    open_, close = _to_tick(open_, tick), _to_tick(close, tick)
    high = np.maximum(_to_tick(high, tick), np.maximum(open_, close))
    low = np.minimum(_to_tick(low, tick), np.minimum(open_, close))
    high = np.where(high == low, high + tick, high)  # jamais de barre plate
    low = np.maximum(low, tick)

    move = np.abs(gap + intraday) / vol
    volume = np.round(spec.base_volume * np.exp(rng.normal(0.0, 0.25, n)) * (0.7 + 0.3 * move))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume.astype(float)},
        index=days,
    )


def _to_tick(x: np.ndarray, tick: float) -> np.ndarray:
    return np.round(np.round(x / tick) * tick, 10)
