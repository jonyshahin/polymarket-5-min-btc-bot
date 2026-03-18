"""Tests for utils.math_utils."""

import pytest

from utils.candle_types import Candle
from utils.math_utils import (
    calc_acceleration,
    calc_atr,
    calc_body_ratio,
    calc_candle_velocity,
    calc_velocity,
    classify_return_velocity,
    find_equal_levels,
    is_marubozu,
)


def _candle(
    o: float, h: float, l: float, c: float, ts: float = 0.0
) -> Candle:
    """Shorthand candle constructor."""
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


# -----------------------------------------------------------------------
# calc_atr
# -----------------------------------------------------------------------


class TestCalcATR:
    """ATR with known True Range values."""

    def test_basic_atr(self) -> None:
        # Candle 0: TR = high-low = 10-5 = 5  (no prev)
        # Candle 1: TR = max(12-7, |12-8|, |7-8|) = max(5,4,1) = 5
        # Candle 2: TR = max(15-9, |15-11|, |9-11|) = max(6,4,2) = 6
        candles = [
            _candle(6, 10, 5, 8),
            _candle(9, 12, 7, 11),
            _candle(10, 15, 9, 13),
        ]
        atr = calc_atr(candles, period=3)
        assert atr == pytest.approx((5 + 5 + 6) / 3, abs=1e-6)

    def test_empty_list(self) -> None:
        assert calc_atr([]) == 0.0

    def test_single_candle(self) -> None:
        c = _candle(100, 110, 95, 105)
        assert calc_atr([c], period=14) == pytest.approx(15.0)

    def test_fewer_than_period(self) -> None:
        candles = [_candle(10, 15, 8, 12), _candle(11, 16, 9, 14)]
        atr = calc_atr(candles, period=14)
        # TR0 = 15-8=7, TR1 = max(16-9, |16-12|, |9-12|) = max(7,4,3) = 7
        assert atr == pytest.approx((7 + 7) / 2, abs=1e-6)


# -----------------------------------------------------------------------
# calc_velocity
# -----------------------------------------------------------------------


class TestCalcVelocity:
    """Rate of change tests."""

    def test_linear_trend(self) -> None:
        values = [100, 101, 102, 103, 104]
        assert calc_velocity(values, window=5) == pytest.approx(0.8)

    def test_single_value(self) -> None:
        assert calc_velocity([42.0]) == 0.0

    def test_empty(self) -> None:
        assert calc_velocity([]) == 0.0

    def test_window_larger_than_list(self) -> None:
        values = [10.0, 20.0, 30.0]
        # window=5 but only 3 values -> actual_window=3, (30-10)/3
        assert calc_velocity(values, window=5) == pytest.approx(20.0 / 3)


# -----------------------------------------------------------------------
# calc_acceleration
# -----------------------------------------------------------------------


class TestCalcAcceleration:
    """Velocity change tests."""

    def test_accelerating(self) -> None:
        # Prior 3 candles close: 100,101,102 -> vel = (102-100)/3
        # Recent 3 candles close: 102,105,110 -> vel = (110-102)/3
        # Acceleration > 0
        candles = [
            _candle(100, 101, 99, 100),
            _candle(100, 102, 100, 101),
            _candle(101, 103, 100, 102),
            _candle(102, 106, 101, 105),
            _candle(105, 111, 104, 110),
            _candle(110, 112, 109, 110),
        ]
        acc = calc_acceleration(candles, window=3)
        assert acc > 0

    def test_decelerating(self) -> None:
        # Prior: big moves, Recent: small moves
        candles = [
            _candle(100, 110, 99, 110),
            _candle(110, 120, 109, 120),
            _candle(120, 130, 119, 130),
            _candle(130, 132, 129, 131),
            _candle(131, 133, 130, 131.5),
            _candle(131.5, 133, 131, 132),
        ]
        acc = calc_acceleration(candles, window=3)
        assert acc < 0

    def test_not_enough_candles(self) -> None:
        candles = [_candle(100, 105, 95, 103)]
        assert calc_acceleration(candles, window=3) == 0.0


# -----------------------------------------------------------------------
# classify_return_velocity
# -----------------------------------------------------------------------


class TestClassifyReturnVelocity:
    """Velocity profile classification."""

    def test_v_shape(self) -> None:
        # Increasing body sizes, last one large
        candles = [
            _candle(100, 101, 99, 100.5),   # body=0.5
            _candle(100.5, 102, 100, 101),   # body=0.5
            _candle(101, 102, 100, 101.2),   # body=0.2
            _candle(101, 103, 100, 102),     # body=1.0
            _candle(102, 107, 101, 106),     # body=4.0
        ]
        assert classify_return_velocity(candles) == "v_shape"

    def test_rounded(self) -> None:
        # Decreasing body sizes
        candles = [
            _candle(100, 110, 99, 109),      # body=9
            _candle(109, 115, 108, 114),     # body=5
            _candle(114, 117, 113, 116),     # body=2
            _candle(116, 118, 115, 117),     # body=1
            _candle(117, 118, 116, 117.3),   # body=0.3
        ]
        assert classify_return_velocity(candles) == "rounded"

    def test_corrective(self) -> None:
        # Mixed body sizes
        candles = [
            _candle(100, 103, 99, 102),      # body=2
            _candle(102, 103, 101, 101),     # body=1
            _candle(101, 104, 100, 103),     # body=2
            _candle(103, 104, 102, 102),     # body=1
            _candle(102, 104, 101, 103),     # body=1
        ]
        assert classify_return_velocity(candles) == "corrective"

    def test_fewer_than_3_candles(self) -> None:
        candles = [_candle(100, 105, 95, 103)]
        assert classify_return_velocity(candles) == "corrective"


# -----------------------------------------------------------------------
# find_equal_levels
# -----------------------------------------------------------------------


class TestFindEqualLevels:
    """Equal price level detection."""

    def test_finds_equal_pair(self) -> None:
        prices = [100.0, 103.0, 100.02, 105.0]
        pairs = find_equal_levels(prices, tolerance=0.0005)
        assert (0, 2) in pairs

    def test_all_different(self) -> None:
        prices = [100.0, 200.0, 300.0, 400.0]
        assert find_equal_levels(prices) == []

    def test_adjacent_not_matched(self) -> None:
        # Indices 0 and 1 are adjacent — should be skipped
        prices = [100.0, 100.01]
        assert find_equal_levels(prices) == []


# -----------------------------------------------------------------------
# Wrapper / convenience functions
# -----------------------------------------------------------------------


class TestWrappers:
    """calc_body_ratio, is_marubozu, calc_candle_velocity."""

    def test_calc_body_ratio(self) -> None:
        c = _candle(100, 110, 100, 110)
        assert calc_body_ratio(c) == 1.0

    def test_is_marubozu(self) -> None:
        c = _candle(100, 110, 100, 110)
        assert is_marubozu(c) is True
        small = _candle(100, 110, 95, 101)
        assert is_marubozu(small) is False

    def test_calc_candle_velocity(self) -> None:
        candles = [
            _candle(100, 105, 95, 100, ts=0),
            _candle(100, 106, 94, 102, ts=60),
            _candle(102, 107, 101, 104, ts=120),
        ]
        # closes = [100, 102, 104], window=5 -> actual_window=3, (104-100)/3
        vel = calc_candle_velocity(candles, window=5)
        assert vel == pytest.approx(4.0 / 3)

    def test_calc_candle_velocity_empty(self) -> None:
        assert calc_candle_velocity([]) == 0.0
