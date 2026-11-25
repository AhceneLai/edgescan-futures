"""Tests du chargeur yfinance, sans internet : Yahoo est remplacé par de fausses réponses."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from edgescan.data import (
    EXCHANGE_TZ,
    DataError,
    clean_ohlcv,
    drop_incomplete,
    load_csv,
    load_yfinance,
    normalize,
)


def yahoo_frame(rows: dict[str, tuple]) -> pd.DataFrame:
    """Barres au format renvoyé par yfinance : colonnes en majuscules, index à minuit heure de New York."""
    idx = pd.DatetimeIndex(list(rows), tz=EXCHANGE_TZ, name="Date")
    df = pd.DataFrame(list(rows.values()), index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    df["Adj Close"] = df["Close"]
    df["Dividends"] = 0.0
    return df


def ny(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=EXCHANGE_TZ)


GOOD = {
    "2024-01-02": (100.0, 102.0, 99.0, 101.0, 1000),
    "2024-01-03": (101.0, 103.0, 100.0, 102.0, 1100),
    "2024-01-04": (102.0, 104.0, 101.0, 103.0, 1200),
}
AFTER_CLOSE = ny("2024-01-04 18:00")


class FakeYahoo:
    """Remplace Yahoo : renvoie les barres demandées à partir de `start` et garde la trace des appels."""

    def __init__(self, rows: dict[str, tuple]):
        self.frame = yahoo_frame(rows)
        self.calls: list = []

    def __call__(self, symbol, start, years):
        self.calls.append(start)
        if start is None:
            return self.frame
        return self.frame[self.frame.index.date >= start]


def test_normalize_standard_format():
    df = normalize(yahoo_frame(GOOD))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_normalize_rejects_missing_columns():
    with pytest.raises(DataError, match="volume"):
        normalize(yahoo_frame(GOOD).drop(columns="Volume"))


def test_clean_removes_each_kind_of_bad_bar():
    rows = {
        **GOOD,
        "2024-01-05": (np.nan, 104.0, 101.0, 103.0, 1000),  # prix manquant
        "2024-01-08": (17.7, 17.8, -40.3, -37.6, 1000),  # prix négatif (CL avril 2020)
        "2024-01-09": (100.0, 99.0, 98.0, 98.5, 1000),  # high < open
        "2024-01-10": (100.0, 100.0, 100.0, 100.0, 50),  # barre plate
        "2024-01-11": (100.0, 101.0, 99.0, 100.5, 0),  # volume inconnu : barre gardée
    }
    raw = normalize(yahoo_frame(rows))
    raw = pd.concat([raw, raw.iloc[[0]] * 1.01])  # doublon du 02/01 : la dernière version gagne

    df, report = clean_ohlcv(raw)

    assert list(df.index.strftime("%m-%d")) == ["01-02", "01-03", "01-04", "01-11"]
    assert report == {
        "duplicates": 1,
        "missing_price": 1,
        "non_positive_price": 1,
        "incoherent_range": 1,
        "flat_bar": 1,
        "unknown_volume": 1,
    }
    assert df.loc["2024-01-02", "close"] == pytest.approx(101.0 * 1.01)
    assert np.isnan(df.loc["2024-01-11", "volume"])
    assert df.index.is_monotonic_increasing


def test_clean_keeps_good_data_untouched():
    raw = normalize(yahoo_frame(GOOD))
    df, report = clean_ohlcv(raw)
    pd.testing.assert_frame_equal(df, raw)
    assert sum(report.values()) == 0


@pytest.mark.parametrize(
    ("now", "kept"),
    [
        (ny("2024-01-04 10:00"), 2),  # séance du 04/01 en cours : barre retirée
        (ny("2024-01-04 16:59"), 2),
        (ny("2024-01-04 17:00"), 3),  # séance terminée
        (ny("2024-01-05 09:00"), 3),
        (datetime.fromisoformat("2024-01-04 23:30+01:00"), 3),  # 17 h 30 à New York, vu depuis Paris
    ],
)
def test_drop_incomplete_last_bar(now, kept):
    df = normalize(yahoo_frame(GOOD))
    assert len(drop_incomplete(df, now)) == kept


def test_load_yfinance_without_cache():
    fake = FakeYahoo(GOOD)
    df = load_yfinance("ES=F", fetch=fake, now=AFTER_CLOSE)
    assert len(df) == 3
    assert fake.calls == [None]


def test_load_yfinance_cache_downloads_only_recent_bars(tmp_path):
    fake = FakeYahoo(GOOD)
    load_yfinance("ES=F", cache_dir=tmp_path, key="ES", fetch=fake, now=AFTER_CLOSE)
    assert (tmp_path / "ES.csv").exists()

    # Le lendemain : une nouvelle barre, et Yahoo a corrigé la clôture du 04/01.
    fake.frame = yahoo_frame(
        {**GOOD, "2024-01-04": (102.0, 104.0, 101.0, 103.5, 1200), "2024-01-05": (103.5, 105.0, 103.0, 104.0, 900)}
    )
    df = load_yfinance("ES=F", cache_dir=tmp_path, key="ES", fetch=fake, now=ny("2024-01-05 18:00"))

    assert fake.calls[1] == datetime(2023, 12, 28).date()  # dernière date en cache - 7 jours
    assert len(df) == 4
    assert df.loc["2024-01-04", "close"] == 103.5
    pd.testing.assert_frame_equal(load_csv(tmp_path / "ES.csv"), df, check_freq=False)


def test_load_yfinance_never_caches_incomplete_bar(tmp_path):
    fake = FakeYahoo(GOOD)
    df = load_yfinance("ES=F", cache_dir=tmp_path, key="ES", fetch=fake, now=ny("2024-01-04 12:00"))
    assert df.index[-1] == pd.Timestamp("2024-01-03")
    assert load_csv(tmp_path / "ES.csv").index[-1] == pd.Timestamp("2024-01-03")


def test_load_yfinance_falls_back_to_cache_when_yahoo_is_empty(tmp_path):
    load_yfinance("ES=F", cache_dir=tmp_path, key="ES", fetch=FakeYahoo(GOOD), now=AFTER_CLOSE)
    df = load_yfinance("ES=F", cache_dir=tmp_path, key="ES", fetch=lambda *a: pd.DataFrame(), now=AFTER_CLOSE)
    assert len(df) == 3


def test_load_yfinance_fails_clearly_without_data():
    with pytest.raises(DataError, match="ES=F"):
        load_yfinance("ES=F", fetch=lambda *a: pd.DataFrame(), now=AFTER_CLOSE)
