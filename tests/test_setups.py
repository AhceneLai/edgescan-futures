"""Tests des setups (2.4) et des niveaux clés (2.5)."""

import numpy as np
import pandas as pd
import pytest

from edgescan.features import compute_features
from edgescan.levels import daily_levels, describe_plan, trade_plan
from edgescan.setups import SETUPS, detect_setups, fr
from edgescan.synthetic import make_synthetic

CFG = {
    "trend_pullback": {"stop_atr": 0.5, "target_atr": 0.75},
    "donchian_breakout": {"stop_atr": 0.5, "target_atr": 1.0},
    "nr7_breakout": {"stop_atr": 0.4, "target_atr": 0.8},
    "bollinger_reversion": {"stop_atr": 0.5, "target_atr": 0.5},
}


@pytest.fixture(scope="module")
def bars():
    return make_synthetic("ES", n_days=1500, seed=11)


@pytest.fixture(scope="module")
def feat(bars):
    return compute_features(bars)


@pytest.fixture(scope="module")
def signals(feat):
    return detect_setups(feat, CFG, symbol="ES")


# --------------------------------------------------------------------------- setups


def test_every_setup_fires_in_both_directions(signals):
    counts = signals.groupby(["setup", "direction"]).size()
    for setup in SETUPS:
        for direction in ("long", "short"):
            assert counts.get((setup, direction), 0) > 0, f"{setup} {direction} jamais déclenché"


def test_signals_match_the_written_rules(feat, signals):
    for _, s in signals.iterrows():
        f = feat.loc[s["date"]]
        long = s["direction"] == "long"
        if s["setup"] == "trend_pullback":
            assert (f["rsi2"] < 10) if long else (f["rsi2"] > 90)
            assert (f["trend_up"] == 1 and f["close"] > f["sma200"]) if long else (f["trend_up"] == 0)
        elif s["setup"] == "donchian_breakout":
            assert (f["close"] > f["donchian_high"]) if long else (f["close"] < f["donchian_low"])
        elif s["setup"] == "nr7_breakout":
            assert f["nr7"]
            assert (f["close"] > f["sma50"]) if long else (f["close"] < f["sma50"])
        else:
            assert (f["bb_z"] <= -2) if long else (f["bb_z"] >= 2)


def test_signal_columns_and_config(signals):
    assert set(signals["symbol"]) == {"ES"}
    pullback = signals[signals["setup"] == "nr7_breakout"].iloc[0]
    assert (pullback["stop_atr"], pullback["target_atr"]) == (0.4, 0.8)
    assert signals["date"].is_monotonic_increasing
    assert signals["atr"].gt(0).all()


def test_every_signal_has_a_specific_reason(signals):
    assert signals["reason"].str.len().gt(30).all()
    assert signals["rationale"].str.len().gt(30).all()
    nan_text = signals["reason"].str.contains("nan", case=False)
    assert not nan_text.any(), signals.loc[nan_text, "reason"].iloc[0]


def test_setup_missing_from_config_is_skipped(feat):
    only = detect_setups(feat, {"donchian_breakout": CFG["donchian_breakout"]})
    assert set(only["setup"]) == {"donchian_breakout"}


def test_no_signal_during_warmup(feat, signals):
    first_valid = feat["atr"].first_valid_index()
    assert signals["date"].min() >= first_valid


def test_setups_never_use_the_future(bars, signals):
    # 2.2 appliqué aux setups : les signaux jusqu'à J ne dépendent pas des barres après J.
    cut = 900
    truncated = detect_setups(compute_features(bars.iloc[: cut + 1]), CFG, symbol="ES")
    expected = signals[signals["date"] <= bars.index[cut]].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, expected)


def test_french_number_format():
    assert fr(1234.5, 2) == "1 234,50"
    assert fr(-0.25, 2) == "-0,25"


# --------------------------------------------------------------------------- niveaux


def test_daily_levels(bars, feat):
    lv = daily_levels(bars, feat)
    j = bars.index[300]
    row, bar = lv.loc[j], bars.loc[j]
    assert row["prev_high"] == bar["high"]
    assert row["pivot"] == pytest.approx((bar["high"] + bar["low"] + bar["close"]) / 3)
    assert row["s1"] < row["pivot"] < row["r1"]
    assert row["week_high"] == bars["high"].iloc[296:301].max()
    assert row["month_low"] <= row["week_low"]


def make_signal(direction, close=100.0, atr=2.0, stop_atr=0.5, target_atr=1.0):
    return pd.Series({"close": close, "atr": atr, "direction": direction, "stop_atr": stop_atr, "target_atr": target_atr})


def test_trade_plan_long_and_short():
    levels = pd.Series({"prev_high": 101.5, "prev_low": 99.5, "sma20": 103.0, "pivot": np.nan})
    long = trade_plan(make_signal("long"), levels)
    assert (long["stop"], long["target"]) == (99.0, 102.0)
    assert long["reward_risk"] == 2.0
    assert [n for n, _, _ in long["obstacles"]] == ["prev_high"]  # 101,5 entre 100 et 102 ; SMA 20 au-delà
    assert [n for n, _, _ in long["supports"]] == ["prev_low"]

    short = trade_plan(make_signal("short"), levels)
    assert (short["stop"], short["target"]) == (101.0, 98.0)
    assert [n for n, _, _ in short["obstacles"]] == ["prev_low"]


def test_trade_plan_uses_actual_entry_when_known():
    plan = trade_plan(make_signal("long"), pd.Series(dtype=float), entry=104.0)
    assert (plan["entry"], plan["stop"], plan["target"]) == (104.0, 103.0, 106.0)


def test_obstacles_sorted_by_distance():
    levels = pd.Series({"r1": 101.8, "prev_high": 100.6, "week_high": 101.2})
    plan = trade_plan(make_signal("long"), levels)
    assert [n for n, _, _ in plan["obstacles"]] == ["prev_high", "week_high", "r1"]
    assert plan["obstacles"][0][2] == pytest.approx(0.3)  # 0,6 point / ATR 2


def test_describe_plan():
    plan = trade_plan(make_signal("long"), pd.Series({"prev_high": 101.5}))
    text = describe_plan(plan, "long")
    assert text.startswith("Achat vers 100,00, stop 99,00")
    assert "plus haut de la veille 101,50" in text
    assert "Aucun niveau clé" in describe_plan(trade_plan(make_signal("long"), pd.Series(dtype=float)), "long")


def test_describe_plan_groups_levels_at_same_price():
    plan = trade_plan(make_signal("long"), pd.Series({"prev_high": 101.5, "week_high": 101.5, "r1": 101.8}))
    text = describe_plan(plan, "long")
    assert "plus haut de la veille / plus haut 5 séances 101,50 (0,75 ATR), résistance R1 101,80" in text
