"""Niveaux clés pour la séance J+1, connus dès la clôture de J, et plan de trade d'un signal.

Les niveaux servent à deux choses :
* placer l'entrée, le stop et l'objectif (en ATR, voir config.yaml) ;
* repérer ce qui se trouve sur le chemin : un ancien plus haut entre l'entrée et l'objectif
  peut bloquer le mouvement (obstacle), un support entre le stop et l'entrée le protège.

Niveaux journaliers uniquement. Plus haut / plus bas de la nuit et VWAP demandent des barres
1 minute : ils seront ajoutés avec la tâche 2.3.
"""

from __future__ import annotations

import pandas as pd

from .setups import fr

WEEK = 5  # séances

LEVEL_NAMES = {
    "prev_high": "plus haut de la veille",
    "prev_low": "plus bas de la veille",
    "prev_close": "clôture de la veille",
    "pivot": "pivot",
    "r1": "résistance R1",
    "s1": "support S1",
    "week_high": "plus haut 5 séances",
    "week_low": "plus bas 5 séances",
    "month_high": "plus haut 20 séances",
    "month_low": "plus bas 20 séances",
    "sma20": "SMA 20",
    "sma50": "SMA 50",
    "sma200": "SMA 200",
}


def daily_levels(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Niveaux valables pour la séance qui suit J (une ligne par J, données <= J)."""
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    lv = pd.DataFrame(index=df.index)
    lv["prev_high"] = h
    lv["prev_low"] = l
    lv["prev_close"] = c
    # Points pivots classiques (hérités de la criée) : repères encore très suivis par les intervenants.
    lv["pivot"] = (h + l + c) / 3
    lv["r1"] = 2 * lv["pivot"] - l
    lv["s1"] = 2 * lv["pivot"] - h
    lv["week_high"] = h.rolling(WEEK).max()
    lv["week_low"] = l.rolling(WEEK).min()
    lv["month_high"] = h.rolling(20).max()  # niveau à casser demain (J inclus)
    lv["month_low"] = l.rolling(20).min()
    for n in (20, 50, 200):
        lv[f"sma{n}"] = feat[f"sma{n}"]
    lv["atr"] = feat["atr"]
    return lv


def trade_plan(signal: pd.Series, levels: pd.Series, entry: float | None = None) -> dict:
    """Entrée, stop, objectif et niveaux sur le chemin pour un signal (ligne de detect_setups).

    entry : prix d'entrée réel (ouverture de J+1). Inconnu le soir de J : on prend alors la
    clôture de J comme référence, les distances en ATR restent les mêmes.
    """
    ref = float(signal["close"] if entry is None else entry)
    atr = float(signal["atr"])
    side = 1 if signal["direction"] == "long" else -1
    stop = ref - side * signal["stop_atr"] * atr
    target = ref + side * signal["target_atr"] * atr

    def between(a: float, b: float) -> list[tuple[str, float, float]]:
        lo, hi = min(a, b), max(a, b)
        found = []
        for name in LEVEL_NAMES:
            price = levels.get(name)
            if price is not None and pd.notna(price) and lo < price < hi:
                found.append((name, float(price), abs(price - ref) / atr))
        return sorted(found, key=lambda x: x[2])

    return {
        "entry": ref,
        "stop": stop,
        "target": target,
        "risk_points": abs(ref - stop),
        "reward_points": abs(target - ref),
        "reward_risk": signal["target_atr"] / signal["stop_atr"],
        "obstacles": between(ref, target),  # peuvent freiner le mouvement avant l'objectif
        "supports": between(stop, ref),  # protègent le stop (ou, cassés, l'accélèrent)
    }


def describe_plan(plan: dict, direction: str) -> str:
    """Plan de trade en une phrase lisible."""
    verb = "Achat" if direction == "long" else "Vente"
    text = (
        f"{verb} vers {fr(plan['entry'], 2)}, stop {fr(plan['stop'], 2)} (risque {fr(plan['risk_points'], 2)} pts), "
        f"objectif {fr(plan['target'], 2)} (gain {fr(plan['reward_points'], 2)} pts, ratio {fr(plan['reward_risk'], 2)})."
    )
    if plan["obstacles"]:
        # Plusieurs niveaux au même prix (ex. plus haut de la veille = plus haut 5 séances) : une seule mention.
        grouped: dict[float, list[str]] = {}
        dist: dict[float, float] = {}
        for n, p, d in plan["obstacles"]:
            grouped.setdefault(round(p, 8), []).append(LEVEL_NAMES[n])
            dist[round(p, 8)] = d
        names = ", ".join(f"{' / '.join(ns)} {fr(p, 2)} ({fr(dist[p], 2)} ATR)" for p, ns in grouped.items())
        text += f" Sur le chemin : {names}."
    else:
        text += " Aucun niveau clé avant l'objectif."
    return text
