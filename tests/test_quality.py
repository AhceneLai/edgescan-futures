"""Tests des contrôles qualité : on part de données synthétiques propres et on y injecte des défauts."""

import numpy as np
import pandas as pd
import pytest

from edgescan.quality import check_bars, to_markdown
from edgescan.synthetic import make_synthetic


@pytest.fixture
def bars():
    # Série courte et calme : aucun mouvement extrême attendu.
    return make_synthetic("ES", n_days=120, seed=3)


def day(df, i):
    return df.index[i]


def test_clean_synthetic_data_is_ok(bars):
    report = check_bars(bars, "ES")
    assert report.issues == {}
    assert report.status == "OK"
    assert report.n_bars == 120
    assert report.longest_gap == 0


def test_errors_make_the_report_fail(bars):
    df = bars.copy()
    df.iloc[5, df.columns.get_loc("close")] = np.nan  # prix manquant
    df.iloc[6, df.columns.get_loc("low")] = -1.0  # prix négatif
    df.iloc[7, df.columns.get_loc("high")] = df.iloc[7]["low"] - 1  # high < low
    df = pd.concat([df, df.iloc[[8]]]).sort_index()  # doublon

    report = check_bars(df, "ES")

    assert report.status == "FAIL"
    assert report.issues["missing_price"] == [day(bars, 5)]
    assert report.issues["non_positive_price"] == [day(bars, 6)]
    assert report.issues["incoherent_range"] == [day(bars, 7)]
    assert report.issues["duplicates"] == [day(bars, 8)]


def test_missing_trading_days_and_longest_gap(bars):
    df = bars.drop(index=bars.index[[10, 30, 31, 32]])
    report = check_bars(df, "ES")
    assert report.status == "WARN"
    assert report.count("missing_days") == 4
    assert report.longest_gap == 3


def test_bars_on_weekends_or_holidays(bars):
    extra = bars.iloc[[0]].copy()
    extra.index = pd.DatetimeIndex([pd.Timestamp("2015-01-19")], name="date")  # MLK 2015
    report = check_bars(pd.concat([bars, extra]).sort_index(), "ES")
    assert report.issues["unexpected_days"] == [pd.Timestamp("2015-01-19")]


def test_flat_bars_and_unknown_volume(bars):
    df = bars.copy()
    i = df.index[20]
    df.loc[i, ["open", "high", "low", "close"]] = df.loc[i, "close"]
    df.loc[df.index[21], "volume"] = np.nan
    report = check_bars(df, "ES")
    assert report.issues["flat_bar"] == [i]
    assert report.issues["unknown_volume"] == [df.index[21]]


def test_extreme_move_is_flagged(bars):
    df = bars.copy()
    i = df.index[80]
    df.loc[i:, ["open", "high", "low", "close"]] *= 1.25  # saut de +25 % (ex. roll non ajusté)
    report = check_bars(df, "ES")
    assert report.issues["outliers"] == [i]


def test_stale_closes(bars):
    df = bars.copy()
    for i in (50, 51, 52):
        df.loc[df.index[i], "close"] = df.loc[df.index[49], "close"]
        df.loc[df.index[i], "high"] = max(df.loc[df.index[i], "high"], df.loc[df.index[i], "close"])
        df.loc[df.index[i], "low"] = min(df.loc[df.index[i], "low"], df.loc[df.index[i], "close"])
    report = check_bars(df, "ES")
    assert report.issues["stale_close"] == list(df.index[49:53])


def test_empty_data_fails():
    empty = make_synthetic("ES", 5).iloc[0:0]
    assert check_bars(empty, "ES").status == "FAIL"


def test_markdown_report(bars):
    broken = bars.drop(index=bars.index[[10]])
    text = to_markdown([check_bars(bars, "ES"), check_bars(broken, "NQ")])
    assert "| ES | OK | 120 |" in text
    assert "| NQ | WARN | 119 |" in text
    assert "jours de cotation absents : 1" in text
    assert "Aucun problème détecté." in text
