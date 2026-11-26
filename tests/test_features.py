"""Tests des indicateurs journaliers, dont le test anti-biais d'anticipation (tâche 2.2)."""

import numpy as np
import pandas as pd
import pytest

from edgescan.features import MODEL_FEATURES, atr, compute_features, rsi, true_range
from edgescan.synthetic import make_synthetic


@pytest.fixture(scope="module")
def bars():
    return make_synthetic("ES", n_days=600, seed=7)


def frame(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows), name="date")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


# --------------------------------------------------------------------------- 2.2 : pas d'anticipation


def scramble_future(df: pd.DataFrame, cut: int, seed: int) -> pd.DataFrame:
    """Remplace toutes les barres après la position `cut` par des barres aléatoires (mais cohérentes)."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out) - cut - 1
    noise = np.exp(rng.normal(0, 0.05, n))
    for col in ["open", "high", "low", "close"]:
        out.iloc[cut + 1 :, out.columns.get_loc(col)] *= noise
    out.iloc[cut + 1 :, out.columns.get_loc("volume")] *= rng.uniform(0.1, 10, n)
    return out


@pytest.mark.parametrize("cut", [50, 250, 420, 598])
def test_changing_the_future_never_changes_the_past(bars, cut):
    before = compute_features(bars)
    after = compute_features(scramble_future(bars, cut, seed=cut))
    pd.testing.assert_frame_equal(before.iloc[: cut + 1], after.iloc[: cut + 1])
    # Contrôle du test lui-même : le futur a bien changé.
    assert not before["close"].iloc[cut + 1 :].equals(after["close"].iloc[cut + 1 :])


@pytest.mark.parametrize("cut", [210, 400])
def test_truncated_history_gives_same_features(bars, cut):
    # Ce que le scanner calcule le soir de J (sans futur) = ce que voit le backtest pour J.
    full = compute_features(bars)
    live = compute_features(bars.iloc[: cut + 1])
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], live)


# --------------------------------------------------------------------------- 2.1 : valeurs


def test_true_range_includes_gaps():
    df = frame([(100, 102, 99, 101, 1), (105, 106, 104, 105, 1), (98, 99, 97, 98, 1)])
    # J2 : gap haussier, TR = 106 - 101 ; J3 : gap baissier, TR = 105 - 97
    assert list(true_range(df)) == [3.0, 5.0, 8.0]


def test_atr_of_constant_range_is_that_range():
    rows = [(100 + i, 102 + i, 98 + i, 100 + i, 1) for i in range(40)]  # amplitude 4, pas de gap
    a = atr(frame(rows), 14)
    assert a.iloc[:13].isna().all()
    assert a.iloc[-1] == pytest.approx(4.0)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 40.0))
    down = up[::-1].reset_index(drop=True)
    assert rsi(up, 14).iloc[-1] == 100
    assert rsi(down, 14).iloc[-1] == pytest.approx(0)
    zigzag = pd.Series([100.0, 101.0] * 30)
    assert rsi(zigzag, 14).iloc[-1] == pytest.approx(50, abs=5)


def test_feature_columns_and_warmup(bars):
    f = compute_features(bars)
    assert set(MODEL_FEATURES) <= set(f.columns)
    assert f["sma200"].iloc[:199].isna().all()
    assert f["sma200"].iloc[199:].notna().all()
    # Après la période de chauffe, aucun indicateur ne manque (le volume synthétique est complet).
    assert f[MODEL_FEATURES].iloc[250:].notna().all().all()


def test_features_are_scale_free(bars):
    # Mêmes mouvements, prix 10 fois plus élevé : les colonnes normalisées ne changent pas.
    scaled = bars.copy()
    scaled[["open", "high", "low", "close"]] *= 10
    a = compute_features(bars)[MODEL_FEATURES]
    b = compute_features(scaled)[MODEL_FEATURES]
    pd.testing.assert_frame_equal(a, b, rtol=1e-9)


def test_nr7_marks_narrowest_range_of_seven():
    ranges = [5, 4, 6, 5, 4, 5, 2, 3, 6, 1]
    rows = [(100, 100 + r / 2, 100 - r / 2, 100, 1) for r in ranges]
    f = compute_features(frame(rows))
    assert list(f["nr7"]) == [False] * 6 + [True, False, False, True]


def test_donchian_uses_previous_days_only():
    rows = [(100, 101, 99, 100, 1)] * 20 + [(100, 105, 99, 104, 1)]
    f = compute_features(frame(rows))
    last = f.iloc[-1]
    assert last["donchian_high"] == 101  # le plus haut du jour (105) n'est pas inclus
    assert last["dist_donchian_high_atr"] > 0  # donc la clôture à 104 est une cassure


def test_bollinger_and_close_location():
    rows = [(100, 101, 99, 100, 1)] * 19 + [(100, 110, 99, 110, 1)]
    f = compute_features(frame(rows))
    last = f.iloc[-1]
    assert last["bb_z"] > 2  # clôture au-dessus de la bande haute
    assert last["close_location"] == 1.0  # clôture au plus haut
    assert last["bb_upper"] > last["sma20"] > last["bb_lower"]


def test_unknown_volume_stays_unknown(bars):
    df = bars.copy()
    df.loc[df.index[300], "volume"] = np.nan
    f = compute_features(df)
    assert np.isnan(f.loc[df.index[300], "rel_volume"])
    assert f.loc[df.index[301], "rel_volume"] > 0
