"""Les 4 setups journaliers, définis AVANT de regarder les résultats.

Signal à la clôture de J (avec les indicateurs de J), trade pendant la séance J+1 : entrée à
l'ouverture, stop et objectif en multiples de l'ATR de J (config.yaml, section setups), sortie à
la clôture de J+1 au plus tard.

Les règles sont volontairement simples et classiques. Elles ne sont PAS optimisées : leur rôle
est de produire des situations comparables ; c'est le modèle (phase 3) qui estimera, situation
par situation, la probabilité que le trade marche. Changer une règle après avoir vu un backtest,
c'est ajuster le passé : toute modification doit être notée et justifiée.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- règles

PULLBACK_RSI2 = 10  # repli : RSI(2) < 10 (long) ou > 90 (short)
BOLLINGER_Z = 2.0  # excès : clôture à plus de 2 écarts-types de la moyenne 20 jours

SETUPS = ("trend_pullback", "donchian_breakout", "nr7_breakout", "bollinger_reversion")

SETUP_RATIONALE = {
    "trend_pullback": (
        "Acheter un repli dans une tendance haussière (ou vendre un rebond dans une tendance baissière). "
        "Les tendances de fond durent ; les excès de court terme contre elles se corrigent souvent vite."
    ),
    "donchian_breakout": (
        "Suivre la cassure du plus haut (ou du plus bas) des 20 dernières séances. "
        "Une sortie de range attire de nouveaux acheteurs (vendeurs) et déclenche des stops."
    ),
    "nr7_breakout": (
        "Après la séance la plus étroite des 7 dernières, jouer l'expansion dans le sens de la tendance. "
        "La volatilité alterne : une compression précède souvent un mouvement plus ample."
    ),
    "bollinger_reversion": (
        "Parier sur un retour vers la moyenne après une clôture hors des bandes de Bollinger. "
        "Un écart de plus de 2 écarts-types en 20 jours est statistiquement rare et souvent excessif."
    ),
}

COLUMNS = ["date", "symbol", "setup", "direction", "close", "atr", "stop_atr", "target_atr", "reason", "rationale"]


def fr(x: float, decimals: int = 1) -> str:
    """Nombre au format français : 1,5 au lieu de 1.5."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


# --------------------------------------------------------------------------- détection


def detect_setups(feat: pd.DataFrame, setup_cfg: dict, symbol: str = "") -> pd.DataFrame:
    """Une ligne par (date, setup, direction) déclenché. `feat` vient de features.compute_features.

    setup_cfg : section `setups` de config.yaml, ex. {"trend_pullback": {"stop_atr": 0.5, "target_atr": 0.75}, ...}
    """
    valid = feat["atr"].notna() & (feat["atr"] > 0)
    rules = {
        "trend_pullback": _trend_pullback(feat),
        "donchian_breakout": _donchian_breakout(feat),
        "nr7_breakout": _nr7_breakout(feat),
        "bollinger_reversion": _bollinger_reversion(feat),
    }

    rows = []
    for setup, (long_mask, short_mask, explain) in rules.items():
        if setup not in setup_cfg:
            continue
        for direction, mask in (("long", long_mask), ("short", short_mask)):
            hits = feat[mask.fillna(False).astype(bool) & valid]
            for date, f in hits.iterrows():
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "setup": setup,
                        "direction": direction,
                        "close": f["close"],
                        "atr": f["atr"],
                        "stop_atr": float(setup_cfg[setup]["stop_atr"]),
                        "target_atr": float(setup_cfg[setup]["target_atr"]),
                        "reason": explain(f, direction),
                        "rationale": SETUP_RATIONALE[setup],
                    }
                )
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["date", "setup", "direction"], ignore_index=True)


def _trend_pullback(f: pd.DataFrame):
    up = (f["trend_up"] == 1) & (f["close"] > f["sma200"])
    down = (f["trend_up"] == 0) & (f["close"] < f["sma200"])
    long_mask = up & (f["rsi2"] < PULLBACK_RSI2)
    short_mask = down & (f["rsi2"] > 100 - PULLBACK_RSI2)

    def explain(r, direction):
        if direction == "long":
            return (
                f"Tendance haussière : SMA 50 au-dessus de la SMA 200, prix {fr(r['dist_sma200_atr'])} ATR au-dessus "
                f"de la SMA 200. Repli marqué à court terme : RSI(2) = {fr(r['rsi2'], 0)}."
            )
        return (
            f"Tendance baissière : SMA 50 sous la SMA 200, prix {fr(-r['dist_sma200_atr'])} ATR sous "
            f"la SMA 200. Rebond marqué à court terme : RSI(2) = {fr(r['rsi2'], 0)}."
        )

    return long_mask, short_mask, explain


def _donchian_breakout(f: pd.DataFrame):
    long_mask = f["close"] > f["donchian_high"]
    short_mask = f["close"] < f["donchian_low"]

    def explain(r, direction):
        if direction == "long":
            return (
                f"Clôture à {fr(r['close'], 2)}, au-dessus du plus haut des 20 séances précédentes "
                f"({fr(r['donchian_high'], 2)}, +{fr(r['dist_donchian_high_atr'], 2)} ATR)."
            )
        return (
            f"Clôture à {fr(r['close'], 2)}, sous le plus bas des 20 séances précédentes "
            f"({fr(r['donchian_low'], 2)}, {fr(r['dist_donchian_low_atr'], 2)} ATR)."
        )

    return long_mask, short_mask, explain


def _nr7_breakout(f: pd.DataFrame):
    nr7 = f["nr7"] & f["sma50"].notna()
    long_mask = nr7 & (f["close"] > f["sma50"])
    short_mask = nr7 & (f["close"] < f["sma50"])

    def explain(r, direction):
        side = "au-dessus" if direction == "long" else "en dessous"
        return (
            f"Séance la plus étroite des 7 dernières (amplitude {fr(r['range_atr'], 2)} ATR). "
            f"Prix {side} de la SMA 50 ({fr(r['dist_sma50_atr'])} ATR) : on joue l'expansion dans ce sens."
        )

    return long_mask, short_mask, explain


def _bollinger_reversion(f: pd.DataFrame):
    long_mask = f["bb_z"] <= -BOLLINGER_Z
    short_mask = f["bb_z"] >= BOLLINGER_Z

    def explain(r, direction):
        band = "basse" if direction == "long" else "haute"
        return (
            f"Clôture à {fr(r['bb_z'], 2)} écarts-types de la moyenne 20 jours, hors de la bande de Bollinger {band} "
            f"({fr(r['bb_lower'] if direction == 'long' else r['bb_upper'], 2)}). "
            f"Retour visé vers la moyenne ({fr(r['sma20'], 2)})."
        )

    return long_mask, short_mask, explain
