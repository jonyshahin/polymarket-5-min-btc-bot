"""Tests for engines.confluence_engine."""

import pytest

from engines.confluence_engine import (
    ConfluenceEngine,
    FairValueGap,
    FlipPattern,
    QMPattern,
)
from utils.candle_types import (
    Candle,
    Direction,
    ReturnType,
    SwingPoint,
    SwingType,
    Zone,
    ZoneType,
)


def _c(ts: float, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


# -----------------------------------------------------------------------
# Test 1: Bullish FVG detection
# -----------------------------------------------------------------------


class TestBullishFVG:
    def test_bullish_fvg(self) -> None:
        ce = ConfluenceEngine()
        candles = [
            _c(0, 99, 100, 98, 99.5),    # c1.high = 100
            _c(60, 100, 105, 99, 104),    # big bullish
            _c(120, 103, 106, 102, 105),  # c3.low = 102
        ]
        fvg = ce._detect_fvg(candles)

        assert fvg is not None
        assert fvg.direction == Direction.BULLISH
        assert fvg.bottom == 100  # c1.high
        assert fvg.top == 102     # c3.low


# -----------------------------------------------------------------------
# Test 2: Bearish FVG detection
# -----------------------------------------------------------------------


class TestBearishFVG:
    def test_bearish_fvg(self) -> None:
        ce = ConfluenceEngine()
        candles = [
            _c(0, 106, 107, 105, 106),    # c1.low = 105
            _c(60, 105, 106, 100, 101),    # big bearish
            _c(120, 101, 103, 100, 102),   # c3.high = 103
        ]
        fvg = ce._detect_fvg(candles)

        assert fvg is not None
        assert fvg.direction == Direction.BEARISH
        assert fvg.top == 105    # c1.low
        assert fvg.bottom == 103  # c3.high


# -----------------------------------------------------------------------
# Test 3: FVG too small
# -----------------------------------------------------------------------


class TestFVGTooSmall:
    def test_no_fvg_when_tiny(self) -> None:
        ce = ConfluenceEngine()
        # c1.high=100.0, c3.low=100.05 → gap=0.05, threshold=100*0.001=0.1 → too small.
        candles = [
            _c(0, 99, 100.0, 98, 99.5),
            _c(60, 100, 101, 99.5, 100.5),
            _c(120, 100, 101, 100.05, 100.5),
        ]
        fvg = ce._detect_fvg(candles)
        assert fvg is None


# -----------------------------------------------------------------------
# Test 4: FVG fill tracking
# -----------------------------------------------------------------------


class TestFVGFill:
    def test_bullish_fvg_filled(self) -> None:
        ce = ConfluenceEngine()
        fvg = FairValueGap(
            timestamp=0, top=102, bottom=100, direction=Direction.BULLISH,
            candle_index=1,
        )
        ce._fvgs.append(fvg)

        # Candle dips into gap: low=101, fills 50% (102-101)/(102-100) = 0.5.
        ce._update_fvg_fills(_c(60, 103, 104, 101, 103))
        assert fvg.fill_pct == pytest.approx(0.5, abs=0.01)
        assert fvg.is_filled is True

    def test_bearish_fvg_filled(self) -> None:
        ce = ConfluenceEngine()
        fvg = FairValueGap(
            timestamp=0, top=105, bottom=103, direction=Direction.BEARISH,
            candle_index=1,
        )
        ce._fvgs.append(fvg)

        # Candle pushes up into gap: high=104, fills (104-103)/(105-103)=0.5.
        ce._update_fvg_fills(_c(60, 102, 104, 101, 103))
        assert fvg.fill_pct == pytest.approx(0.5, abs=0.01)
        assert fvg.is_filled is True


# -----------------------------------------------------------------------
# Test 5: Bearish QM (H-L-HH-LL)
# -----------------------------------------------------------------------


class TestBearishQM:
    def test_bearish_qm(self) -> None:
        ce = ConfluenceEngine()
        swings = [
            SwingPoint(timestamp=0, price=100, type=SwingType.HH),    # H
            SwingPoint(timestamp=60, price=90, type=SwingType.HL),    # L
            SwingPoint(timestamp=120, price=105, type=SwingType.HH),  # HH (>100)
            SwingPoint(timestamp=180, price=85, type=SwingType.LL),   # LL (<90)
        ]
        qm = ce._detect_qm(swings)

        assert qm is not None
        assert qm.direction == Direction.BEARISH
        assert qm.mpl_price == 90
        assert qm.over_price == 105
        assert qm.under_price == 85


# -----------------------------------------------------------------------
# Test 6: Bullish QM (L-H-LL-HH)
# -----------------------------------------------------------------------


class TestBullishQM:
    def test_bullish_qm(self) -> None:
        ce = ConfluenceEngine()
        swings = [
            SwingPoint(timestamp=0, price=100, type=SwingType.HL),    # L
            SwingPoint(timestamp=60, price=110, type=SwingType.HH),   # H
            SwingPoint(timestamp=120, price=95, type=SwingType.LL),   # LL (<100)
            SwingPoint(timestamp=180, price=115, type=SwingType.HH),  # HH (>110)
        ]
        qm = ce._detect_qm(swings)

        assert qm is not None
        assert qm.direction == Direction.BULLISH
        assert qm.mpl_price == 110


# -----------------------------------------------------------------------
# Test 7: No QM
# -----------------------------------------------------------------------


class TestNoQM:
    def test_no_qm_when_pattern_fails(self) -> None:
        ce = ConfluenceEngine()
        # H(100) - L(90) - LH(98) - HL(92) → 98 < 100 (not HH) → no QM.
        swings = [
            SwingPoint(timestamp=0, price=100, type=SwingType.HH),
            SwingPoint(timestamp=60, price=90, type=SwingType.HL),
            SwingPoint(timestamp=120, price=98, type=SwingType.LH),
            SwingPoint(timestamp=180, price=92, type=SwingType.HL),
        ]
        qm = ce._detect_qm(swings)
        assert qm is None


# -----------------------------------------------------------------------
# Test 8: Return type — V-shape
# -----------------------------------------------------------------------


class TestReturnTypeVShape:
    def test_v_shape(self) -> None:
        ce = ConfluenceEngine()
        # Increasing body sizes: 0.5, 0.5, 1, 2, 8.
        candles = [
            _c(0, 100, 101, 99, 100.5),      # body=0.5
            _c(60, 100.5, 102, 100, 101),     # body=0.5
            _c(120, 101, 103, 100, 102),      # body=1
            _c(180, 102, 105, 101, 104),      # body=2
            _c(240, 104, 114, 103, 112),      # body=8
        ]
        rt = ce.classify_return_type(candles)
        assert rt == ReturnType.V_SHAPE


# -----------------------------------------------------------------------
# Test 9: Return type — rounded
# -----------------------------------------------------------------------


class TestReturnTypeRounded:
    def test_rounded(self) -> None:
        ce = ConfluenceEngine()
        # Decreasing body sizes.
        candles = [
            _c(0, 100, 110, 99, 109),       # body=9
            _c(60, 109, 115, 108, 114),      # body=5
            _c(120, 114, 117, 113, 116),     # body=2
            _c(180, 116, 118, 115, 117),     # body=1
            _c(240, 117, 118, 116, 117.3),   # body=0.3
        ]
        rt = ce.classify_return_type(candles)
        assert rt == ReturnType.ROUNDED


# -----------------------------------------------------------------------
# Test 10: Strong engulfing — bearish
# -----------------------------------------------------------------------


class TestStrongEngulfing:
    def test_bearish_engulfing(self) -> None:
        ce = ConfluenceEngine()
        candles = [
            _c(0, 100, 103, 99, 102),       # small bullish (body 100-102)
            _c(60, 103, 104, 97, 98),        # large bearish engulfs (body 98-103)
            _c(120, 98, 99, 94, 95),         # bearish follow-through (body=3)
        ]
        result = ce.check_strong_engulfing(candles)
        assert result == Direction.BEARISH

    def test_bullish_engulfing(self) -> None:
        ce = ConfluenceEngine()
        candles = [
            _c(0, 102, 103, 99, 100),       # small bearish (body 100-102)
            _c(60, 99, 106, 98, 105),        # large bullish engulfs (body 99-105)
            _c(120, 105, 110, 104, 109),     # bullish follow-through (body=4)
        ]
        result = ce.check_strong_engulfing(candles)
        assert result == Direction.BULLISH


# -----------------------------------------------------------------------
# Test 11: Weak engulfing (no follow-through)
# -----------------------------------------------------------------------


class TestWeakEngulfing:
    def test_no_follow_through(self) -> None:
        ce = ConfluenceEngine()
        candles = [
            _c(0, 100, 103, 99, 102),       # small bullish
            _c(60, 103, 104, 97, 98),        # bearish engulfing
            _c(120, 98, 102, 97, 101),       # bullish follow — opposite direction
        ]
        result = ce.check_strong_engulfing(candles)
        assert result is None


# -----------------------------------------------------------------------
# Test 12: S/D flip detection
# -----------------------------------------------------------------------


class TestSDFlip:
    def test_demand_flip_to_bearish(self) -> None:
        ce = ConfluenceEngine()
        ce._candle_count = 10

        zone = Zone(
            timestamp=0, high=102, low=100, zone_type=ZoneType.DEMAND,
            is_mitigated=True, is_broken=True, candle_index=5,
        )
        flip = ce._detect_flip([zone], None)

        assert flip is not None
        assert flip.direction == Direction.BEARISH
        assert flip.broken_zone is zone

    def test_supply_flip_to_bullish(self) -> None:
        ce = ConfluenceEngine()
        ce._candle_count = 10

        zone = Zone(
            timestamp=0, high=110, low=108, zone_type=ZoneType.SUPPLY,
            is_mitigated=True, is_broken=True, candle_index=5,
        )
        flip = ce._detect_flip([zone], None)

        assert flip is not None
        assert flip.direction == Direction.BULLISH

    def test_no_flip_when_not_broken(self) -> None:
        ce = ConfluenceEngine()
        ce._candle_count = 10

        zone = Zone(
            timestamp=0, high=102, low=100, zone_type=ZoneType.DEMAND,
            is_mitigated=True, is_broken=False, candle_index=5,
        )
        flip = ce._detect_flip([zone], None)
        assert flip is None
