"""Tests du stockage Parquet + DuckDB (dans un dossier temporaire)."""

from datetime import date

import pandas as pd
import pytest

from edgescan.storage import Store
from edgescan.synthetic import make_synthetic


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


@pytest.fixture
def bars():
    return make_synthetic("ES", n_days=50)


def signal(signal_id="2026-11-25_ES_trend_pullback_long", **changes):
    row = {
        "signal_id": signal_id,
        "created_at": pd.Timestamp("2026-11-24 22:30", tz="UTC"),
        "trade_date": date(2026, 11, 25),
        "symbol": "ES",
        "setup": "trend_pullback",
        "direction": "long",
        "entry": 7200.0,
        "stop": 7180.0,
        "target": 7230.0,
        "p_target": 0.42,
        "p_stop": 0.33,
        "p_close": 0.25,
        "expected_r": 0.18,
        "contracts": 1,
        "rationale": "Tendance haussière, repli sur la SMA20.",
        "model_version": "v0",
        "status": "open",
    }
    return {**row, **changes}


def test_daily_bars_round_trip(store, bars):
    assert store.write_bars("ES", bars) == 50
    pd.testing.assert_frame_equal(store.read_bars("ES"), bars, check_freq=False, check_names=False)
    assert store.read_bars("ES").index.name == "date"


def test_daily_upsert_replaces_and_appends(store, bars):
    store.write_bars("ES", bars.iloc[:40])
    corrected = bars.iloc[35:].copy()
    corrected.loc[corrected.index[0], "close"] += 1  # barre corrigée par le fournisseur
    store.write_bars("ES", corrected)

    out = store.read_bars("ES")
    assert len(out) == 50
    assert out.loc[bars.index[35], "close"] == bars.loc[bars.index[35], "close"] + 1


def test_read_with_bounds(store, bars):
    store.write_bars("ES", bars)
    out = store.read_bars("ES", start=bars.index[10], end=bars.index[19])
    assert list(out.index) == list(bars.index[10:20])


def test_unknown_symbol_returns_empty(store):
    out = store.read_bars("ZZ")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_symbols(store, bars):
    store.write_bars("NQ", bars)
    store.write_bars("ES", bars)
    assert store.symbols() == ["ES", "NQ"]


def test_minute_bars_partitioned_by_month(store):
    idx = pd.date_range("2026-10-31 23:58", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)
    store.write_bars("ES", df, timeframe="1m")

    files = sorted(p.name for p in (store.root / "bars_1m" / "ES").glob("*.parquet"))
    assert files == ["2026-10.parquet", "2026-11.parquet"]
    out = store.read_bars("ES", "1m", start=pd.Timestamp("2026-11-01", tz="UTC"))
    assert len(out) == 2
    assert str(out.index.tz) == "UTC"


def test_minute_bars_require_timezone(store):
    idx = pd.date_range("2026-11-02 14:30", periods=2, freq="1min")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)
    with pytest.raises(ValueError, match="fuseau"):
        store.write_bars("ES", df, timeframe="1m")


def test_unknown_timeframe(store, bars):
    with pytest.raises(ValueError, match="timeframe"):
        store.write_bars("ES", bars, timeframe="5m")


def test_signals_lifecycle(store):
    store.write_signals(pd.DataFrame([signal(), signal("2026-11-25_NQ_nr7_breakout_short", symbol="NQ")]))
    # À la clôture, le signal ES est mis à jour avec son résultat.
    closed = signal(status="closed", outcome="target", exit_price=7230.0, pnl_r=1.5)
    closed["closed_at"] = pd.Timestamp("2026-11-25 20:10", tz="UTC")
    store.write_signals(pd.DataFrame([closed]))

    out = store.read_signals().set_index("signal_id")
    assert len(out) == 2
    es = out.loc["2026-11-25_ES_trend_pullback_long"]
    assert es["status"] == "closed"
    assert es["pnl_r"] == 1.5
    assert pd.isna(out.loc["2026-11-25_NQ_nr7_breakout_short", "outcome"])


def test_signals_validation(store):
    with pytest.raises(ValueError, match="statut"):
        store.write_signals(pd.DataFrame([signal(status="maybe")]))
    with pytest.raises(ValueError, match="double"):
        store.write_signals(pd.DataFrame([signal(), signal()]))
    bad = signal()
    del bad["entry"]
    with pytest.raises(ValueError, match="entry"):
        store.write_signals(pd.DataFrame([bad]))


def test_empty_signals_table(store):
    out = store.read_signals()
    assert out.empty
    assert "expected_r" in out.columns


def test_sql_queries(store, bars):
    store.write_bars("ES", bars)
    store.write_bars("NQ", make_synthetic("NQ", n_days=30))
    store.write_signals(pd.DataFrame([signal()]))

    counts = store.query("SELECT symbol, count(*) AS n FROM bars_1d GROUP BY symbol ORDER BY symbol")
    assert counts.to_dict("records") == [{"symbol": "ES", "n": 50}, {"symbol": "NQ", "n": 30}]
    assert store.query("SELECT count(*) AS n FROM signals WHERE expected_r > ?", [0.1])["n"][0] == 1


def test_sql_on_empty_store(store):
    with pytest.raises(Exception, match="bars_1d"):
        store.query("SELECT * FROM bars_1d")
