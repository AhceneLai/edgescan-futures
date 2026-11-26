"""Tests de la labellisation (3.1) et des features orientées (3.2)."""

import numpy as np
import pandas as pd
import pytest

from edgescan.dataset import TRADE_FEATURES, build_dataset, cost_per_contract, label_trades, trade_features
from edgescan.features import compute_features
from edgescan.levels import daily_levels
from edgescan.setups import detect_setups
from edgescan.synthetic import make_synthetic

SPEC = {"multiplier": 50, "tick": 0.25}
COSTS = {"commission_per_side": 2.5, "slippage_ticks_per_side": 1}
SETUPS_CFG = {
    "trend_pullback": {"stop_atr": 0.5, "target_atr": 0.75},
    "donchian_breakout": {"stop_atr": 0.5, "target_atr": 1.0},
    "nr7_breakout": {"stop_atr": 0.4, "target_atr": 0.8},
    "bollinger_reversion": {"stop_atr": 0.5, "target_atr": 0.5},
}


def bars_with_next_day(high, low, close, open_=100.0):
    """Deux séances : J (signal) puis J+1 (trade). Lundi 23 et mardi 24 novembre 2026."""
    idx = pd.DatetimeIndex(["2026-11-23", "2026-11-24"], name="date")
    return pd.DataFrame(
        {"open": [99.0, open_], "high": [101.0, high], "low": [98.0, low], "close": [100.0, close], "volume": 1.0},
        index=idx,
    )


def one_signal(direction="long", atr=4.0, stop_atr=0.5, target_atr=1.0):
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-11-23"),
                "symbol": "ES",
                "setup": "donchian_breakout",
                "direction": direction,
                "close": 100.0,
                "atr": atr,
                "stop_atr": stop_atr,
                "target_atr": target_atr,
            }
        ]
    )


# --------------------------------------------------------------------------- 3.1


def test_cost_per_contract():
    # 2 x (2,50 $ de commission + 1 tick x 0,25 x 50 $ de slippage) = 2 x 15 $
    assert cost_per_contract(SPEC, COSTS) == 30.0


@pytest.mark.parametrize(
    ("direction", "high", "low", "close", "outcome", "gross_r"),
    [
        # ATR 4 : stop à 2 points, objectif à 4 points de l'entrée (100).
        ("long", 104.5, 99.0, 103.0, "target", 2.0),
        ("long", 101.0, 97.5, 98.0, "stop", -1.0),
        ("long", 103.0, 98.5, 101.0, "close", 0.5),
        ("long", 104.5, 97.5, 101.0, "stop", -1.0),  # les deux touchés : prudent
        ("short", 101.0, 95.5, 97.0, "target", 2.0),
        ("short", 102.5, 99.0, 101.0, "stop", -1.0),
        ("short", 101.0, 98.0, 99.0, "close", 0.5),
    ],
)
def test_label_outcomes(direction, high, low, close, outcome, gross_r):
    t = label_trades(one_signal(direction), bars_with_next_day(high, low, close), SPEC, COSTS).iloc[0]
    assert t["outcome"] == outcome
    assert t["gross_r"] == pytest.approx(gross_r)
    assert t["entry"] == 100.0
    assert t["entry_date"] == pd.Timestamp("2026-11-24")


def test_both_levels_hit_is_marked_ambiguous():
    t = label_trades(one_signal(), bars_with_next_day(104.5, 97.5, 101.0), SPEC, COSTS).iloc[0]
    assert t["ambiguous"]
    t = label_trades(one_signal(), bars_with_next_day(104.5, 99.0, 103.0), SPEC, COSTS).iloc[0]
    assert not t["ambiguous"]


def test_costs_are_subtracted_in_r_and_dollars():
    t = label_trades(one_signal(), bars_with_next_day(104.5, 99.0, 103.0), SPEC, COSTS).iloc[0]
    # 1 R = 2 points x 50 $ = 100 $ ; frais 30 $ = 0,3 R
    assert t["cost_r"] == pytest.approx(0.3)
    assert t["net_r"] == pytest.approx(2.0 - 0.3)
    assert t["pnl_usd"] == pytest.approx(4 * 50 - 30)


def test_entry_is_next_open_not_signal_close():
    t = label_trades(one_signal(), bars_with_next_day(106.0, 101.5, 105.0, open_=102.0), SPEC, COSTS).iloc[0]
    assert (t["entry"], t["stop"], t["target"]) == (102.0, 100.0, 106.0)
    assert t["outcome"] == "target"


def test_signal_without_next_session_is_not_labelled():
    bars = bars_with_next_day(104.5, 99.0, 103.0).iloc[:1]
    assert label_trades(one_signal(), bars, SPEC, COSTS).empty


def test_missing_next_session_is_skipped():
    bars = bars_with_next_day(104.5, 99.0, 103.0)
    bars.index = pd.DatetimeIndex(["2026-11-23", "2026-11-25"], name="date")  # le 24 manque
    assert label_trades(one_signal(), bars, SPEC, COSTS).empty


# --------------------------------------------------------------------------- 3.2


@pytest.fixture(scope="module")
def synthetic_case():
    bars = make_synthetic("ES", n_days=900, seed=5)
    feat = compute_features(bars)
    return bars, feat, daily_levels(bars, feat), detect_setups(feat, SETUPS_CFG, "ES")


def test_features_are_mirrored_for_shorts(synthetic_case):
    _, feat, levels, signals = synthetic_case
    s = signals.iloc[[0]]
    long = trade_features(s.assign(direction="long"), feat, levels).iloc[0]
    short = trade_features(s.assign(direction="short"), feat, levels).iloc[0]
    assert short["mom5_atr"] == pytest.approx(-long["mom5_atr"])
    assert short["bb_z"] == pytest.approx(-long["bb_z"])
    assert short["rsi2"] == pytest.approx(100 - long["rsi2"])
    assert short["close_location"] == pytest.approx(1 - long["close_location"])
    assert short["atr_pct"] == long["atr_pct"]  # neutre
    f = feat.loc[s["date"].iloc[0]]
    assert long["donchian_break"] == pytest.approx(f["dist_donchian_high_atr"])
    assert short["donchian_break"] == pytest.approx(-f["dist_donchian_low_atr"])


def test_trade_features_columns(synthetic_case):
    _, feat, levels, signals = synthetic_case
    x = trade_features(signals, feat, levels)
    assert list(x.columns) == TRADE_FEATURES
    assert (x.filter(like="setup_").sum(axis=1) == 1).all()  # exactement un setup par ligne
    assert (x["first_obstacle_atr"] <= x["target_atr"] + 1e-9).all()
    assert (x["first_support_atr"] <= x["stop_atr"] + 1e-9).all()


def test_build_dataset(synthetic_case):
    bars = synthetic_case[0]
    cfg = {"setups": SETUPS_CFG, "contracts": {"ES": SPEC}, "costs": COSTS}
    trades, tonight = build_dataset({"ES": bars}, cfg)
    assert trades["entry_date"].is_monotonic_increasing
    assert set(trades["outcome"]) == {"stop", "close", "target"}
    assert (trades["entry_date"] > trades["date"]).all()  # on entre toujours APRÈS le signal
    assert set(TRADE_FEATURES) <= set(trades.columns)
    assert tonight.empty or (tonight["date"] == bars.index[-1]).all()
    assert np.isfinite(trades["net_r"]).all()
