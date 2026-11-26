"""Tests du générateur de données synthétiques."""

import numpy as np
import pandas as pd
import pytest

from edgescan.data import clean_ohlcv, load_all
from edgescan.sessions import trading_days
from edgescan.synthetic import SPECS, make_synthetic


@pytest.fixture(scope="module")
def es():
    return make_synthetic("ES", n_days=2600)


def test_reproducible():
    pd.testing.assert_frame_equal(make_synthetic("ES", 300, seed=1), make_synthetic("ES", 300, seed=1))


def test_seed_and_key_change_the_series():
    a = make_synthetic("ES", 300, seed=1)
    assert not a["close"].equals(make_synthetic("ES", 300, seed=2)["close"])
    assert not a["close"].equals(make_synthetic("NQ", 300, seed=1)["close"])


def test_standard_format(es):
    assert list(es.columns) == ["open", "high", "low", "close", "volume"]
    assert es.index.name == "date"
    assert len(es) == 2600
    assert es.index.is_monotonic_increasing
    assert (es.dtypes == "float64").all()


def test_only_cme_trading_days(es):
    expected = trading_days(es.index[0], es.index[-1])
    pd.testing.assert_index_equal(es.index, expected)


def test_passes_the_cleaning_untouched(es):
    cleaned, report = clean_ohlcv(es)
    assert sum(report.values()) == 0
    pd.testing.assert_frame_equal(cleaned, es)


def test_prices_on_tick_grid():
    for key in ["ES", "CL"]:
        df = make_synthetic(key, 500)
        ticks = df[["open", "high", "low", "close"]] / SPECS[key].tick
        assert np.allclose(ticks, ticks.round(), atol=1e-6)


@pytest.mark.parametrize("key", ["ES", "NQ", "CL", "GC"])
def test_realistic_volatility(key):
    df = make_synthetic(key, 2600)
    realised = np.log(df["close"]).diff().std() * np.sqrt(252)
    assert 0.6 * SPECS[key].annual_vol < realised < 1.5 * SPECS[key].annual_vol


def test_volatility_regimes_and_fat_tails(es):
    r = np.log(es["close"]).diff().dropna()
    rolling = r.rolling(63).std().dropna()
    assert rolling.max() / rolling.min() > 2  # des périodes calmes et des périodes agitées
    assert r.kurt() > 1  # plus de journées extrêmes qu'une loi normale (kurtosis 0)


def test_unknown_contract_uses_default():
    df = make_synthetic("ZZ", 50)
    assert len(df) == 50
    assert (df["low"] > 0).all()


def test_rejects_empty_request():
    with pytest.raises(ValueError):
        make_synthetic("ES", 0)


def test_load_all_synthetic_source():
    bars = load_all({"ES": {"symbol": "ES=F"}, "GC": {"symbol": "GC=F"}}, "synthetic")
    assert set(bars) == {"ES", "GC"}
    assert len(bars["ES"]) == 2600
