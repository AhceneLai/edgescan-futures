"""Indicateurs journaliers, calculés à la clôture de J avec des données <= J uniquement.

Les distances et les pentes sont exprimées en ATR : « le prix est à 1,5 ATR au-dessus de sa
moyenne » veut dire la même chose pour ES à 7 000 points et pour CL à 80 $. C'est ce qui permet
d'entraîner un seul modèle sur les quatre contrats.

Chaque fonction n'utilise que des calculs « causaux » (moyennes glissantes, lissages exponentiels,
décalages vers le passé) : une barre future ne peut jamais modifier un indicateur passé.
Le test tests/test_features.py le vérifie en modifiant le futur (tâche 2.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_N = 14
SMA_WINDOWS = (20, 50, 200)
BB_N, BB_K = 20, 2.0
DONCHIAN_N = 20
NR_N = 7
SLOPE_LAG = 5  # pente d'une moyenne = variation sur 5 séances

# Colonnes sans unité (ATR, %, σ, 0-1) : comparables d'un contrat à l'autre, utilisables par le modèle.
MODEL_FEATURES = [
    "atr_pct",
    "atr_ratio",
    "range_atr",
    "nr7",
    "close_location",
    "gap_atr",
    "ret1_atr",
    "mom5_atr",
    "mom20_atr",
    "dist_sma20_atr",
    "dist_sma50_atr",
    "dist_sma200_atr",
    "slope_sma20_atr",
    "slope_sma50_atr",
    "trend_up",
    "rsi2",
    "rsi14",
    "bb_z",
    "bb_width_atr",
    "dist_donchian_high_atr",
    "dist_donchian_low_atr",
    "rel_volume",
]


def true_range(df: pd.DataFrame) -> pd.Series:
    """Plus grand de : high - low, |high - clôture veille|, |low - clôture veille|."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1)
    return tr.max(axis=1)


def wilder(series: pd.Series, n: int) -> pd.Series:
    """Moyenne lissée de Wilder (utilisée par l'ATR et le RSI)."""
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    """Average True Range : amplitude moyenne d'une séance, gaps compris."""
    return wilder(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (0-100), version de Wilder."""
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder(-delta.clip(upper=0), n)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tous les indicateurs d'un contrat, une ligne par séance J.

    Contient les niveaux bruts (en points, utiles pour les setups et les rapports) et les
    colonnes normalisées listées dans MODEL_FEATURES. Les premières lignes valent NaN tant que
    l'historique est trop court (200 séances pour la SMA 200).
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]  # noqa: E741
    f = pd.DataFrame(index=df.index)
    f["close"] = c

    # Volatilité
    f["atr"] = atr(df)
    f["atr_pct"] = f["atr"] / c
    f["atr_ratio"] = atr(df, 5) / atr(df, 50)  # > 1 : la volatilité récente augmente
    rng = h - l
    f["range_atr"] = rng / f["atr"]
    f["nr7"] = (rng == rng.rolling(NR_N).min()) & rng.rolling(NR_N).min().notna()

    # La séance elle-même
    f["close_location"] = ((c - l) / rng).where(rng > 0, 0.5)  # 0 = clôture au plus bas, 1 = au plus haut
    f["gap_atr"] = (o - c.shift(1)) / f["atr"].shift(1)
    f["ret1_atr"] = c.diff() / f["atr"]
    f["mom5_atr"] = (c - c.shift(5)) / f["atr"]
    f["mom20_atr"] = (c - c.shift(20)) / f["atr"]

    # Tendance
    for n in SMA_WINDOWS:
        sma = c.rolling(n).mean()
        f[f"sma{n}"] = sma
        f[f"dist_sma{n}_atr"] = (c - sma) / f["atr"]
    for n in (20, 50):
        f[f"slope_sma{n}_atr"] = (f[f"sma{n}"] - f[f"sma{n}"].shift(SLOPE_LAG)) / f["atr"]
    f["trend_up"] = (f["sma50"] > f["sma200"]).astype(float).where(f["sma200"].notna())  # 1, 0 ou NaN

    # Oscillateurs
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)

    # Bandes de Bollinger (20 séances, 2 écarts-types)
    std = c.rolling(BB_N).std(ddof=0)
    f["bb_upper"] = f["sma20"] + BB_K * std
    f["bb_lower"] = f["sma20"] - BB_K * std
    f["bb_z"] = (c - f["sma20"]) / std  # position en écarts-types : -2 = sur la bande basse
    f["bb_width_atr"] = (f["bb_upper"] - f["bb_lower"]) / f["atr"]

    # Canal de Donchian des 20 séances *précédentes* (J exclu : sinon une cassure serait impossible)
    f["donchian_high"] = h.rolling(DONCHIAN_N).max().shift(1)
    f["donchian_low"] = l.rolling(DONCHIAN_N).min().shift(1)
    f["dist_donchian_high_atr"] = (c - f["donchian_high"]) / f["atr"]
    f["dist_donchian_low_atr"] = (c - f["donchian_low"]) / f["atr"]

    # Volume relatif (NaN si le volume est inconnu)
    vol = df["volume"].astype(float)
    f["rel_volume"] = vol / vol.rolling(20, min_periods=10).mean()

    f = f.replace([np.inf, -np.inf], np.nan)
    f["nr7"] = f["nr7"].astype(bool)
    return f
