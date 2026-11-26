"""Calendrier des séances CME Globex : jours de cotation, heures d'ouverture, pauses et demi-séances.

Deux notions à ne pas confondre :

* **Jour de cotation** (trading day) : jour qui a sa propre barre journalière (un prix de règlement).
  Ce sont les jours ouvrés, moins les jours fériés américains. Les jours comme MLK ou Thanksgiving,
  Globex ouvre quelques heures mais ces échanges comptent pour la séance du lendemain : pas de barre.
* **Marché ouvert** : Globex accepte des ordres. Une séance va de 18 h (heure de New York) la veille
  à 17 h le jour J, avec une pause d'une heure chaque soir et une fermeture du vendredi 17 h au
  dimanche 18 h. Les heures exactes (fermetures anticipées) dépendent du produit.

Toutes les heures renvoyées sont dans le fuseau de New York.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from .data import EXCHANGE_TZ

# Calendrier pandas_market_calendars par contrat. Les contrats inconnus utilisent celui des indices.
PRODUCT_CALENDARS = {
    "ES": "CME_Equity",
    "NQ": "CME_Equity",
    "MES": "CME_Equity",
    "MNQ": "CME_Equity",
    "CL": "CMEGlobex_CL",
    "MCL": "CMEGlobex_CL",
    "GC": "CMEGlobex_GC",
    "MGC": "CMEGlobex_GC",
}
DEFAULT_CALENDAR = "CME_Equity"
NORMAL_CLOSE_HOUR = 17  # clôture normale de Globex, heure de New York

DateLike = str | date | datetime | pd.Timestamp


def _day(d: DateLike) -> pd.Timestamp:
    """Date calendaire (sans heure ni fuseau) ; une heure avec fuseau est d'abord ramenée à New York."""
    t = pd.Timestamp(d)
    if t.tz is not None:
        t = t.tz_convert(EXCHANGE_TZ).tz_localize(None)
    return t.normalize()


@lru_cache(maxsize=1)
def _us_holidays() -> pd.DatetimeIndex:
    # Jours fériés réguliers de la bourse de New York : ce sont aussi les jours sans règlement sur Globex.
    hol = mcal.get_calendar("NYSE").regular_holidays.holidays(start="2000-01-01", end="2040-12-31")
    return pd.DatetimeIndex(hol).normalize()


def holidays(start: DateLike, end: DateLike) -> pd.DatetimeIndex:
    """Jours fériés (en semaine) sans barre journalière entre start et end inclus."""
    hol = _us_holidays()
    return hol[(hol >= _day(start)) & (hol <= _day(end))]


def trading_days(start: DateLike, end: DateLike) -> pd.DatetimeIndex:
    """Jours qui ont une barre journalière, entre start et end inclus."""
    days = pd.bdate_range(_day(start), _day(end))
    return days.difference(_us_holidays()).rename("date")


def is_trading_day(d: DateLike) -> bool:
    return len(trading_days(d, d)) == 1


def previous_trading_day(d: DateLike) -> pd.Timestamp:
    """Dernier jour de cotation strictement avant d."""
    day = _day(d)
    return trading_days(day - pd.Timedelta(days=10), day - pd.Timedelta(days=1))[-1]


def next_trading_day(d: DateLike) -> pd.Timestamp:
    """Premier jour de cotation strictement après d."""
    day = _day(d)
    return trading_days(day + pd.Timedelta(days=1), day + pd.Timedelta(days=10))[0]


@lru_cache(maxsize=16)
def _calendar(key: str):
    return mcal.get_calendar(PRODUCT_CALENDARS.get(key, DEFAULT_CALENDAR))


def schedule(key: str, start: DateLike, end: DateLike) -> pd.DataFrame:
    """Séances Globex d'un contrat : colonnes open, close (heure de New York) et early_close.

    Inclut les séances raccourcies des jours fériés (ex. MLK), qui n'ont pas de barre journalière :
    la colonne trading_day permet de les distinguer.
    """
    raw = _calendar(key).schedule(_day(start), _day(end))
    out = pd.DataFrame(
        {
            "open": raw["market_open"].dt.tz_convert(EXCHANGE_TZ),
            "close": raw["market_close"].dt.tz_convert(EXCHANGE_TZ),
        },
        index=pd.DatetimeIndex(raw.index, name="date"),
    )
    normal_close = out.index + pd.Timedelta(hours=NORMAL_CLOSE_HOUR)
    out["early_close"] = out["close"].dt.tz_localize(None) < normal_close
    out["trading_day"] = out.index.isin(trading_days(start, end))
    return out


def early_closes(key: str, start: DateLike, end: DateLike) -> pd.DataFrame:
    """Demi-séances qui ont une barre journalière (ex. lendemain de Thanksgiving, 24 décembre)."""
    s = schedule(key, start, end)
    return s[s["early_close"] & s["trading_day"]]


def _as_ny(ts: DateLike) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(EXCHANGE_TZ) if t.tz is None else t.tz_convert(EXCHANGE_TZ)


def is_open(key: str, ts: DateLike) -> bool:
    """Globex est-il ouvert pour ce contrat à cet instant ? (heure naïve = heure de New York)"""
    t = _as_ny(ts)
    s = schedule(key, t - pd.Timedelta(days=1), t + pd.Timedelta(days=1))
    return bool(((s["open"] <= t) & (t < s["close"])).any())


def next_open(key: str, ts: DateLike) -> pd.Timestamp:
    """Prochaine ouverture de Globex à partir de ts (ts lui-même si le marché est ouvert)."""
    t = _as_ny(ts)
    if is_open(key, t):
        return t
    s = schedule(key, t - pd.Timedelta(days=1), t + pd.Timedelta(days=10))
    return s.loc[s["open"] > t, "open"].iloc[0]
