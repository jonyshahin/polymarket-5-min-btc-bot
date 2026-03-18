"""Tests for engines.liquidity_engine."""

import pytest

from engines.liquidity_engine import LiquidityEngine, LiquidityLevel, LiquiditySweep
from utils.candle_types import Candle, Direction, SwingPoint, SwingType


def _c(ts: float, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def _sh(price: float, ts: float = 0.0) -> SwingPoint:
    return SwingPoint(timestamp=ts, price=price, type=SwingType.HH)


def _sl(price: float, ts: float = 0.0) -> SwingPoint:
    return SwingPoint(timestamp=ts, price=price, type=SwingType.LL)


# -----------------------------------------------------------------------
# Test 1: EQH detection
# -----------------------------------------------------------------------


class TestEQHDetection:
    def test_equal_highs_found(self) -> None:
        le = LiquidityEngine()
        highs = [_sh(100.0, 0), _sh(105.0, 60), _sh(100.02, 120), _sh(108.0, 180)]
        levels = le.detect_levels(highs, [])

        eqh = [lv for lv in levels if lv.level_type == "EQH"]
        assert len(eqh) == 1
        assert abs(eqh[0].price - 100.01) < 0.1


# -----------------------------------------------------------------------
# Test 2: EQL detection
# -----------------------------------------------------------------------


class TestEQLDetection:
    def test_equal_lows_found(self) -> None:
        le = LiquidityEngine()
        lows = [_sl(50.0, 0), _sl(45.0, 60), _sl(49.98, 120), _sl(42.0, 180)]
        levels = le.detect_levels([], lows)

        eql = [lv for lv in levels if lv.level_type == "EQL"]
        assert len(eql) == 1
        assert abs(eql[0].price - 49.99) < 0.1


# -----------------------------------------------------------------------
# Test 3: Bearish sweep (EQH swept)
# -----------------------------------------------------------------------


class TestBearishSweep:
    def test_eqh_swept(self) -> None:
        le = LiquidityEngine()
        le._levels.append(LiquidityLevel(price=100.0, level_type="EQH"))

        candle = _c(60, 99, 100.5, 98.5, 99.5)  # high > 100, close < 100
        sweep = le.check_sweep(candle)

        assert sweep is not None
        assert sweep.direction_after == Direction.BEARISH
        assert le._levels[0].is_swept is True


# -----------------------------------------------------------------------
# Test 4: Bullish sweep (EQL swept)
# -----------------------------------------------------------------------


class TestBullishSweep:
    def test_eql_swept(self) -> None:
        le = LiquidityEngine()
        le._levels.append(LiquidityLevel(price=50.0, level_type="EQL"))

        candle = _c(60, 50.5, 51, 49.5, 50.5)  # low < 50, close > 50
        sweep = le.check_sweep(candle)

        assert sweep is not None
        assert sweep.direction_after == Direction.BULLISH


# -----------------------------------------------------------------------
# Test 5: No sweep (break through without closing back)
# -----------------------------------------------------------------------


class TestNoSweep:
    def test_break_not_sweep(self) -> None:
        le = LiquidityEngine()
        le._levels.append(LiquidityLevel(price=100.0, level_type="EQH"))

        # Closes above — not a sweep (no reversal).
        candle = _c(60, 99, 101, 98.5, 101)
        sweep = le.check_sweep(candle)
        assert sweep is None


# -----------------------------------------------------------------------
# Test 6: Internal vs external classification
# -----------------------------------------------------------------------


class TestInternalExternal:
    def test_external_near_range_high(self) -> None:
        le = LiquidityEngine()
        le.update_range(110, 90)
        # 109.8 is within 0.2% of 110 (110 * 0.002 = 0.22).
        assert le._is_external(109.8) is True

    def test_internal_mid_range(self) -> None:
        le = LiquidityEngine()
        le.update_range(110, 90)
        assert le._is_external(100.0) is False

    def test_external_near_range_low(self) -> None:
        le = LiquidityEngine()
        le.update_range(110, 90)
        # 90.1 is within 0.2% of 90 (90 * 0.002 = 0.18).
        assert le._is_external(90.1) is True


# -----------------------------------------------------------------------
# Test 7: get_active_levels excludes swept
# -----------------------------------------------------------------------


class TestActiveLevels:
    def test_excludes_swept(self) -> None:
        le = LiquidityEngine()
        lv1 = LiquidityLevel(price=100.0, level_type="EQH")
        lv2 = LiquidityLevel(price=105.0, level_type="EQH", is_swept=True)
        le._levels.extend([lv1, lv2])

        active = le.get_active_levels()
        assert len(active) == 1
        assert active[0] is lv1


# -----------------------------------------------------------------------
# Test 8: get_sweep_within
# -----------------------------------------------------------------------


class TestSweepWithin:
    def test_within_window(self) -> None:
        le = LiquidityEngine()
        candle = _c(1000, 99, 100.5, 98.5, 99.5)
        sweep = LiquiditySweep(
            timestamp=1000,
            level=LiquidityLevel(price=100.0, level_type="EQH", is_swept=True),
            sweep_candle=candle,
            direction_after=Direction.BEARISH,
        )
        le._sweeps.append(sweep)

        # 1200 - 1000 = 200s < 5*60=300s → found.
        assert le.get_sweep_within(5, 1200) is sweep

    def test_outside_window(self) -> None:
        le = LiquidityEngine()
        candle = _c(1000, 99, 100.5, 98.5, 99.5)
        sweep = LiquiditySweep(
            timestamp=1000,
            level=LiquidityLevel(price=100.0, level_type="EQH", is_swept=True),
            sweep_candle=candle,
            direction_after=Direction.BEARISH,
        )
        le._sweeps.append(sweep)

        # 1400 - 1000 = 400s > 300s → not found.
        assert le.get_sweep_within(5, 1400) is None

    def test_no_sweeps(self) -> None:
        le = LiquidityEngine()
        assert le.get_sweep_within(5, 1000) is None
