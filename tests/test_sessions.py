"""Tests du calendrier CME sur des dates connues de 2026."""

import pandas as pd
import pytest

from edgescan import sessions as s


def test_us_holidays_2026():
    expected = [
        "2026-01-01",  # Jour de l'an
        "2026-01-19",  # Martin Luther King
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Vendredi saint
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Fête nationale (observée, le 4 tombe un samedi)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Noël
    ]
    assert list(s.holidays("2026-01-01", "2026-12-31").strftime("%Y-%m-%d")) == expected


def test_trading_days_skip_weekends_and_holidays():
    days = s.trading_days("2026-11-23", "2026-11-30")
    assert list(days.strftime("%m-%d")) == ["11-23", "11-24", "11-25", "11-27", "11-30"]
    assert len(s.trading_days("2026-01-01", "2026-12-31")) == 251


@pytest.mark.parametrize(
    ("d", "expected"),
    [("2026-11-25", True), ("2026-11-26", False), ("2026-11-28", False), ("2026-11-27", True)],
)
def test_is_trading_day(d, expected):
    assert s.is_trading_day(d) is expected


def test_previous_and_next_trading_day():
    assert s.previous_trading_day("2026-01-20") == pd.Timestamp("2026-01-16")  # saute le week-end et MLK
    assert s.next_trading_day("2026-11-25") == pd.Timestamp("2026-11-27")  # saute Thanksgiving
    assert s.next_trading_day("2026-12-24") == pd.Timestamp("2026-12-28")  # saute Noël et le week-end


def test_timezone_aware_dates_use_new_york_day():
    # 2 h du matin à Paris le 26/11 = 20 h à New York le 25/11
    assert s.is_trading_day(pd.Timestamp("2026-11-26 02:00", tz="Europe/Paris"))


def test_early_closes_have_a_daily_bar():
    es = s.early_closes("ES", "2026-01-01", "2026-12-31")
    assert list(es.index.strftime("%m-%d")) == ["11-27", "12-24"]
    assert (es["close"].dt.hour < 17).all()


def test_holiday_sessions_are_not_trading_days():
    sched = s.schedule("ES", "2026-01-16", "2026-01-20")
    mlk = sched.loc["2026-01-19"]
    assert mlk["early_close"]
    assert not mlk["trading_day"]  # Globex ouvre le matin, mais pas de barre journalière


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        ("2026-11-24 10:00", True),  # séance normale
        ("2026-11-24 17:30", False),  # pause quotidienne 17 h - 18 h
        ("2026-11-24 18:00", True),  # réouverture
        ("2026-11-27 23:00", False),  # vendredi soir
        ("2026-11-29 17:59", False),  # dimanche avant l'ouverture
        ("2026-11-29 18:00", True),  # ouverture du dimanche soir
        ("2026-01-19 10:00", True),  # MLK : Globex ouvert le matin
        ("2026-01-19 15:00", False),  # MLK : arrêt anticipé
        ("2026-12-25 12:00", False),  # Noël
    ],
)
def test_is_open_es(ts, expected):
    assert s.is_open("ES", ts) is expected


def test_is_open_depends_on_product():
    # Lendemain de Thanksgiving à 14 h : les indices ont fermé à 13 h, le pétrole cote jusqu'à 14 h 45.
    assert not s.is_open("ES", "2026-11-27 14:00")
    assert s.is_open("CL", "2026-11-27 14:00")


def test_is_open_accepts_other_timezones():
    assert s.is_open("ES", pd.Timestamp("2026-11-24 16:00", tz="Europe/Paris"))  # 10 h à New York
    assert not s.is_open("ES", pd.Timestamp("2026-11-24 23:30", tz="Europe/Paris"))  # 17 h 30 à New York


def test_next_open():
    assert s.next_open("ES", "2026-11-27 23:00") == pd.Timestamp("2026-11-29 18:00", tz=s.EXCHANGE_TZ)
    assert s.next_open("ES", "2026-12-25 12:00") == pd.Timestamp("2026-12-27 18:00", tz=s.EXCHANGE_TZ)
    open_now = pd.Timestamp("2026-11-24 10:00", tz=s.EXCHANGE_TZ)
    assert s.next_open("ES", open_now) == open_now
