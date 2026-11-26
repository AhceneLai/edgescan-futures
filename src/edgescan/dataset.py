"""Signaux -> trades -> issues labellisées, frais inclus, et features orientées dans le sens du trade.

3.1 Labellisation. Pour un signal à la clôture de J :
    entrée  = ouverture de J+1
    stop    = entrée -/+ stop_atr   x ATR(J)
    objectif= entrée +/- target_atr x ATR(J)
    issue   = "target" si seul l'objectif est touché pendant J+1,
              "stop"   si seul le stop est touché,
              "close"  si aucun des deux : sortie à la clôture de J+1.
    Si le stop ET l'objectif sont dans la fourchette de J+1, une barre journalière ne dit pas
    lequel a été touché en premier : on compte un stop (hypothèse prudente) et on le marque
    `ambiguous`. Avec les barres 1 minute (tâche 1.2), l'ordre réel sera connu.

    Résultat exprimé en R (1 R = la perte si le stop est touché), après commission et slippage
    à l'entrée et à la sortie.

3.2 Features orientées. Un modèle unique pour les achats et les ventes : chaque feature est
    exprimée du point de vue du trade. « +2 ATR de momentum » veut dire « le marché va dans mon
    sens », que je sois acheteur ou vendeur.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import compute_features
from .levels import daily_levels, trade_plan
from .sessions import next_trading_day
from .setups import SETUPS, detect_setups

log = logging.getLogger(__name__)

OUTCOMES = ["stop", "close", "target"]  # ordre des classes du modèle

# --------------------------------------------------------------------------- 3.2 : features

# Ne dépendent pas du sens du trade.
NEUTRAL = ["atr_pct", "atr_ratio", "range_atr", "nr7", "bb_width_atr", "rel_volume"]
# Changent de signe pour une vente.
SIGNED = [
    "gap_atr",
    "ret1_atr",
    "mom5_atr",
    "mom20_atr",
    "dist_sma20_atr",
    "dist_sma50_atr",
    "dist_sma200_atr",
    "slope_sma20_atr",
    "slope_sma50_atr",
    "bb_z",
]

TRADE_FEATURES = [
    *NEUTRAL,
    *SIGNED,
    "close_location",  # 1 = clôture du côté du trade (au plus haut pour un achat)
    "rsi2",  # élevé = le marché est déjà monté dans le sens du trade
    "rsi14",
    "trend_with",  # 1 = SMA 50 / SMA 200 dans le sens du trade
    "donchian_break",  # > 0 = au-delà de l'extrême 20 jours, dans le sens du trade
    "donchian_room",  # distance à l'extrême opposé (de combien le marché a déjà bougé)
    "stop_atr",
    "target_atr",
    "first_obstacle_atr",  # premier niveau clé avant l'objectif (= target_atr si aucun)
    "n_obstacles",
    "first_support_atr",  # premier niveau clé entre l'entrée et le stop (= stop_atr si aucun)
    *[f"setup_{s}" for s in SETUPS],
]

FEATURE_LABELS = {
    "atr_pct": "volatilité (ATR en % du prix)",
    "atr_ratio": "volatilité récente vs 50 jours",
    "range_atr": "amplitude de la séance",
    "nr7": "séance NR7",
    "bb_width_atr": "largeur des bandes de Bollinger",
    "rel_volume": "volume relatif",
    "gap_atr": "gap d'ouverture",
    "ret1_atr": "mouvement du jour",
    "mom5_atr": "momentum 5 jours",
    "mom20_atr": "momentum 20 jours",
    "dist_sma20_atr": "distance à la SMA 20",
    "dist_sma50_atr": "distance à la SMA 50",
    "dist_sma200_atr": "distance à la SMA 200",
    "slope_sma20_atr": "pente de la SMA 20",
    "slope_sma50_atr": "pente de la SMA 50",
    "bb_z": "position dans les bandes de Bollinger",
    "close_location": "position de la clôture dans la séance",
    "rsi2": "RSI(2)",
    "rsi14": "RSI(14)",
    "trend_with": "tendance de fond dans le sens du trade",
    "donchian_break": "dépassement du canal 20 jours",
    "donchian_room": "chemin parcouru depuis l'extrême opposé",
    "stop_atr": "distance du stop",
    "target_atr": "distance de l'objectif",
    "first_obstacle_atr": "premier obstacle avant l'objectif",
    "n_obstacles": "nombre d'obstacles avant l'objectif",
    "first_support_atr": "premier support avant le stop",
    **{f"setup_{s}": f"setup {s}" for s in SETUPS},
}


def trade_features(signals: pd.DataFrame, feat: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Features de chaque signal, orientées dans le sens du trade (même index que `signals`)."""
    f = feat.loc[signals["date"]].reset_index(drop=True)
    f.index = signals.index
    side = np.where(signals["direction"] == "long", 1.0, -1.0)
    is_long = side > 0

    x = pd.DataFrame(index=signals.index)
    for col in NEUTRAL:
        x[col] = f[col].astype(float)
    for col in SIGNED:
        x[col] = f[col] * side
    x["close_location"] = np.where(is_long, f["close_location"], 1 - f["close_location"])
    x["rsi2"] = np.where(is_long, f["rsi2"], 100 - f["rsi2"])
    x["rsi14"] = np.where(is_long, f["rsi14"], 100 - f["rsi14"])
    x["trend_with"] = np.where(is_long, f["trend_up"], 1 - f["trend_up"])
    x["donchian_break"] = np.where(is_long, f["dist_donchian_high_atr"], -f["dist_donchian_low_atr"])
    x["donchian_room"] = np.where(is_long, f["dist_donchian_low_atr"], -f["dist_donchian_high_atr"])
    x["stop_atr"] = signals["stop_atr"]
    x["target_atr"] = signals["target_atr"]

    # Niveaux clés sur le chemin (plan calculé avec la clôture de J comme référence).
    obstacles, n_obstacles, supports = [], [], []
    for _, s in signals.iterrows():
        plan = trade_plan(s, levels.loc[s["date"]])
        obstacles.append(plan["obstacles"][0][2] if plan["obstacles"] else s["target_atr"])
        n_obstacles.append(len(plan["obstacles"]))
        supports.append(plan["supports"][0][2] if plan["supports"] else s["stop_atr"])
    x["first_obstacle_atr"] = obstacles
    x["n_obstacles"] = n_obstacles
    x["first_support_atr"] = supports

    for s in SETUPS:
        x[f"setup_{s}"] = (signals["setup"] == s).astype(float)
    return x[TRADE_FEATURES]


# --------------------------------------------------------------------------- 3.1 : labels


def cost_per_contract(spec: dict, costs: dict) -> float:
    """Frais d'un aller-retour en $ : commission + slippage, à l'entrée et à la sortie."""
    slippage = costs["slippage_ticks_per_side"] * spec["tick"] * spec["multiplier"]
    return 2 * (costs["commission_per_side"] + slippage)


def label_trades(signals: pd.DataFrame, bars: pd.DataFrame, spec: dict, costs: dict) -> pd.DataFrame:
    """Ajoute à chaque signal le résultat de son trade en J+1. Les signaux sans séance J+1 sont retirés."""
    positions = bars.index.get_indexer(signals["date"])
    rows = []
    skipped = 0
    for (_, s), pos in zip(signals.iterrows(), positions, strict=True):
        if pos < 0 or pos + 1 >= len(bars):
            continue  # pas encore de séance J+1 (signal du soir) : rien à labelliser
        nxt = bars.iloc[pos + 1]
        entry_date = bars.index[pos + 1]
        if entry_date != next_trading_day(s["date"]):
            skipped += 1  # séance J+1 absente des données : on ne devine pas
            continue

        side = 1 if s["direction"] == "long" else -1
        entry = nxt["open"]
        risk = s["stop_atr"] * s["atr"]
        stop = entry - side * risk
        target = entry + side * s["target_atr"] * s["atr"]
        favorable, adverse = (nxt["high"], nxt["low"]) if side > 0 else (nxt["low"], nxt["high"])
        hit_target = (favorable - target) * side >= 0
        hit_stop = (adverse - stop) * side <= 0

        if hit_stop:
            outcome, exit_price = "stop", stop  # y compris si les deux sont touchés (prudent)
        elif hit_target:
            outcome, exit_price = "target", target
        else:
            outcome, exit_price = "close", nxt["close"]

        cost = cost_per_contract(spec, costs)
        gross_r = (exit_price - entry) * side / risk
        cost_r = cost / (risk * spec["multiplier"])
        rows.append(
            {
                **s.to_dict(),
                "entry_date": entry_date,
                "entry": entry,
                "stop": stop,
                "target": target,
                "exit_price": exit_price,
                "outcome": outcome,
                "ambiguous": bool(hit_stop and hit_target),
                "gross_r": gross_r,
                "cost_r": cost_r,
                "net_r": gross_r - cost_r,
                "pnl_usd": (exit_price - entry) * side * spec["multiplier"] - cost,
            }
        )
    if skipped:
        log.info("%s : %d signaux sans séance J+1 dans les données", signals["symbol"].iloc[0], skipped)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- assemblage


def build_dataset(bars_by_symbol: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tous les contrats -> (trades labellisés avec features, signaux du dernier soir avec features).

    Les trades sont triés par date d'entrée : indispensable pour les validations temporelles.
    """
    trades, pending = [], []
    for key, bars in bars_by_symbol.items():
        feat = compute_features(bars)
        levels = daily_levels(bars, feat)
        signals = detect_setups(feat, cfg["setups"], symbol=key)
        if signals.empty:
            continue
        x = trade_features(signals, feat, levels)
        signals = signals.join(x.drop(columns=[c for c in x.columns if c in signals.columns]))

        labelled = label_trades(signals, bars, cfg["contracts"][key], cfg["costs"])
        if not labelled.empty:
            trades.append(labelled)
        pending.append(signals[signals["date"] == bars.index[-1]])

    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if not all_trades.empty:
        all_trades = all_trades.sort_values(["entry_date", "symbol", "setup"], ignore_index=True)
    tonight = pd.concat(pending, ignore_index=True) if pending else pd.DataFrame()
    return all_trades, tonight
