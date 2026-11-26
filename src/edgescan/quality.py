"""Contrôles qualité des barres journalières et rapport par contrat.

Le chargeur (edgescan.data) corrige ce qu'il peut ; ce module *mesure* ce qui reste et ce qui
manque, quelle que soit la source (yfinance, CSV, fournisseur pro). Il ne modifie rien.

Deux niveaux :
* erreur (FAIL)  : données fausses qui faussent les calculs -> à corriger avant toute utilisation ;
* alerte (WARN)  : données suspectes ou incomplètes -> à connaître, souvent acceptables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PRICES
from .sessions import trading_days

OUTLIER_Z = 8.0  # écart au rendement « normal », en écarts-types robustes
OUTLIER_WINDOW = 63  # ~ un trimestre de séances
STALE_RUN = 3  # clôtures identiques consécutives
MAX_LISTED = 10  # dates affichées par problème dans le rapport

ERRORS = ("duplicates", "missing_price", "non_positive_price", "incoherent_range")
WARNINGS = ("missing_days", "unexpected_days", "flat_bar", "unknown_volume", "outliers", "stale_close")

LABELS = {
    "duplicates": "dates en double",
    "missing_price": "prix manquants",
    "non_positive_price": "prix nuls ou négatifs",
    "incoherent_range": "barres incohérentes (high/low)",
    "missing_days": "jours de cotation absents",
    "unexpected_days": "barres un week-end ou un jour férié",
    "flat_bar": "barres plates (O = H = L = C)",
    "unknown_volume": "volumes inconnus",
    "outliers": "mouvements extrêmes à vérifier",
    "stale_close": "clôtures figées (3 jours ou plus)",
}


@dataclass
class QualityReport:
    key: str
    n_bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    longest_gap: int  # plus longue suite de jours de cotation absents
    issues: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    def count(self, name: str) -> int:
        return len(self.issues.get(name, []))

    @property
    def status(self) -> str:
        if self.n_bars == 0 or any(self.count(n) for n in ERRORS):
            return "FAIL"
        if any(self.count(n) for n in WARNINGS):
            return "WARN"
        return "OK"


def check_bars(df: pd.DataFrame, key: str = "") -> QualityReport:
    """Analyse un DataFrame de barres journalières au format standard (voir edgescan.data)."""
    if df.empty:
        return QualityReport(key, 0, None, None, 0)

    idx = pd.DatetimeIndex(df.index)
    issues: dict[str, list[pd.Timestamp]] = {}

    def add(name: str, mask) -> None:
        dates = list(pd.DatetimeIndex(idx[np.asarray(mask)]).unique())
        if dates:
            issues[name] = dates

    add("duplicates", idx.duplicated(keep=False))
    add("missing_price", df[PRICES].isna().any(axis=1))
    add("non_positive_price", (df[PRICES] <= 0).any(axis=1))
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    add("incoherent_range", (df["high"] < oc_max) | (df["low"] > oc_min) | (df["high"] < df["low"]))
    add("flat_bar", (df["high"] == df["low"]) & df["high"].notna())
    add("unknown_volume", df["volume"].isna() | (df["volume"] <= 0))

    expected = trading_days(idx.min(), idx.max())
    unique_idx = idx.unique()
    missing = expected.difference(unique_idx)
    if len(missing):
        issues["missing_days"] = list(missing)
    unexpected = unique_idx.difference(expected)
    if len(unexpected):
        issues["unexpected_days"] = list(unexpected)

    bars = df[~idx.duplicated(keep="last")].sort_index()
    close = bars["close"].where(bars["close"] > 0)
    add_on = pd.DatetimeIndex(bars.index)
    outliers = _outliers(np.log(close).diff())
    if outliers.any():
        issues["outliers"] = list(add_on[outliers.to_numpy()])
    stale = _stale(bars["close"])
    if stale.any():
        issues["stale_close"] = list(add_on[stale.to_numpy()])

    return QualityReport(
        key=key,
        n_bars=len(df),
        first=idx.min(),
        last=idx.max(),
        longest_gap=_longest_run(expected.isin(unique_idx)),
        issues=issues,
    )


def _outliers(returns: pd.Series) -> pd.Series:
    """Rendements très éloignés de la médiane récente, mesurés en écarts-types robustes (MAD)."""
    med = returns.rolling(OUTLIER_WINDOW, min_periods=20).median().shift()
    mad = (returns - med).abs().rolling(OUTLIER_WINDOW, min_periods=20).median().shift()
    z = (returns - med).abs() / (1.4826 * mad)
    return (z > OUTLIER_Z).fillna(False)


def _stale(close: pd.Series) -> pd.Series:
    """Marque chaque barre d'une suite d'au moins STALE_RUN clôtures identiques."""
    same = close.ne(close.shift()).cumsum()
    run_length = close.groupby(same).transform("size")
    return run_length >= STALE_RUN


def _longest_run(present: np.ndarray) -> int:
    longest = current = 0
    for ok in present:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return longest


def to_markdown(reports: list[QualityReport]) -> str:
    """Rapport lisible : un tableau récapitulatif puis le détail par contrat."""
    lines = [
        "# Rapport qualité des données",
        "",
        "| Contrat | Statut | Barres | Du | Au | Jours absents | Plus long trou | Extrêmes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        first = f"{r.first:%Y-%m-%d}" if r.first is not None else "—"
        last = f"{r.last:%Y-%m-%d}" if r.last is not None else "—"
        lines.append(
            f"| {r.key} | {r.status} | {r.n_bars} | {first} | {last} | {r.count('missing_days')} "
            f"| {r.longest_gap} j | {r.count('outliers')} |"
        )

    for r in reports:
        lines += ["", f"## {r.key} — {r.status}", ""]
        if r.n_bars == 0:
            lines.append("- **erreur** : aucune barre")
            continue
        if not r.issues:
            lines.append("Aucun problème détecté.")
            continue
        for name in (*ERRORS, *WARNINGS):
            dates = r.issues.get(name)
            if not dates:
                continue
            level = "erreur" if name in ERRORS else "alerte"
            shown = ", ".join(f"{d:%Y-%m-%d}" for d in dates[:MAX_LISTED])
            more = f" … (+{len(dates) - MAX_LISTED})" if len(dates) > MAX_LISTED else ""
            lines.append(f"- **{level}** — {LABELS[name]} : {len(dates)} ({shown}{more})")
    return "\n".join(lines) + "\n"
