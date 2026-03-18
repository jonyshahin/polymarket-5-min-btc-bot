"""Comprehensive tests for utils.candle_types."""

import pytest

from utils.candle_types import (
    BOS,
    BOSType,
    BetDecision,
    Candle,
    ControlStateType,
    Direction,
    MarketPhase,
    ReturnType,
    SwingPoint,
    SwingStrength,
    SwingType,
    Zone,
    ZonePattern,
    ZonePosition,
    ZoneType,
)


# -----------------------------------------------------------------------
# Candle tests
# -----------------------------------------------------------------------


class TestCandleBullish:
    """Bullish candle: open=100, high=105, low=99, close=104."""

    @pytest.fixture()
    def candle(self) -> Candle:
        return Candle(timestamp=1.0, open=100, high=105, low=99, close=104)

    def test_body_size(self, candle: Candle) -> None:
        assert candle.body_size == 4

    def test_range_size(self, candle: Candle) -> None:
        assert candle.range_size == 6

    def test_body_ratio(self, candle: Candle) -> None:
        assert candle.body_ratio == pytest.approx(4 / 6, abs=1e-6)

    def test_is_bullish(self, candle: Candle) -> None:
        assert candle.is_bullish is True
        assert candle.is_bearish is False

    def test_direction(self, candle: Candle) -> None:
        assert candle.direction is Direction.BULLISH

    def test_upper_wick(self, candle: Candle) -> None:
        assert candle.upper_wick == 1

    def test_lower_wick(self, candle: Candle) -> None:
        assert candle.lower_wick == 1


class TestCandleBearish:
    """Bearish candle: open=104, high=105, low=99, close=100."""

    @pytest.fixture()
    def candle(self) -> Candle:
        return Candle(timestamp=1.0, open=104, high=105, low=99, close=100)

    def test_is_bearish(self, candle: Candle) -> None:
        assert candle.is_bearish is True
        assert candle.is_bullish is False

    def test_direction(self, candle: Candle) -> None:
        assert candle.direction is Direction.BEARISH

    def test_body_high_low(self, candle: Candle) -> None:
        assert candle.body_high == 104
        assert candle.body_low == 100


class TestCandleMarubozu:
    """Marubozu: open=100, high=110, low=100, close=110."""

    def test_marubozu(self) -> None:
        c = Candle(timestamp=1.0, open=100, high=110, low=100, close=110)
        assert c.body_ratio == 1.0
        assert c.is_marubozu() is True
        assert c.is_doji() is False


class TestCandleDoji:
    """Doji: open=100, high=105, low=95, close=100.01."""

    def test_doji(self) -> None:
        c = Candle(timestamp=1.0, open=100, high=105, low=95, close=100.01)
        assert c.body_ratio == pytest.approx(0.01 / 10, abs=1e-6)
        assert c.is_doji() is True
        assert c.is_marubozu() is False


class TestCandleZeroRange:
    """Zero-range candle: all values = 100."""

    def test_no_crash(self) -> None:
        c = Candle(timestamp=1.0, open=100, high=100, low=100, close=100)
        assert c.range_size == 1e-10
        assert c.body_ratio == 0.0


class TestCandleEngulfing:
    """Engulfing pattern detection."""

    def test_big_engulfs_small(self) -> None:
        big = Candle(timestamp=1.0, open=95, high=112, low=94, close=110)
        small = Candle(timestamp=2.0, open=98, high=104, low=97, close=103)
        assert big.engulfs(small) is True

    def test_small_does_not_engulf_big(self) -> None:
        big = Candle(timestamp=1.0, open=95, high=112, low=94, close=110)
        small = Candle(timestamp=2.0, open=98, high=104, low=97, close=103)
        assert small.engulfs(big) is False


class TestCandleMidpoints:
    """Midpoint and body_midpoint calculations."""

    def test_midpoint(self) -> None:
        c = Candle(timestamp=1.0, open=102, high=110, low=100, close=108)
        assert c.midpoint == 105.0

    def test_body_midpoint(self) -> None:
        c = Candle(timestamp=1.0, open=102, high=110, low=100, close=108)
        assert c.body_midpoint == 105.0


class TestCandleWickRatio:
    """Wick ratio calculation."""

    def test_wick_ratio(self) -> None:
        c = Candle(timestamp=1.0, open=100, high=105, low=99, close=104)
        # upper_wick=1, lower_wick=1, range=6 → wick_ratio=2/6
        assert c.wick_ratio == pytest.approx(2 / 6, abs=1e-6)


# -----------------------------------------------------------------------
# SwingPoint tests
# -----------------------------------------------------------------------


class TestSwingPoint:
    """SwingPoint is_high / is_low properties."""

    def test_hh_is_high(self) -> None:
        sp = SwingPoint(timestamp=1.0, price=100, type=SwingType.HH)
        assert sp.is_high is True
        assert sp.is_low is False

    def test_ll_is_low(self) -> None:
        sp = SwingPoint(timestamp=1.0, price=90, type=SwingType.LL)
        assert sp.is_high is False
        assert sp.is_low is True

    def test_hl_is_low(self) -> None:
        sp = SwingPoint(timestamp=1.0, price=95, type=SwingType.HL)
        assert sp.is_low is True
        assert sp.is_high is False

    def test_lh_is_high(self) -> None:
        sp = SwingPoint(timestamp=1.0, price=105, type=SwingType.LH)
        assert sp.is_high is True
        assert sp.is_low is False


# -----------------------------------------------------------------------
# Zone tests
# -----------------------------------------------------------------------


class TestZone:
    """Zone geometry and price containment."""

    @pytest.fixture()
    def zone(self) -> Zone:
        return Zone(timestamp=1.0, high=105, low=100, zone_type=ZoneType.DEMAND)

    def test_midpoint(self, zone: Zone) -> None:
        assert zone.midpoint == 102.5

    def test_size(self, zone: Zone) -> None:
        assert zone.size == 5

    def test_contains_price_inside(self, zone: Zone) -> None:
        assert zone.contains_price(103) is True

    def test_contains_price_outside(self, zone: Zone) -> None:
        assert zone.contains_price(106) is False

    def test_is_above_price_true(self, zone: Zone) -> None:
        assert zone.is_above_price(99) is True

    def test_is_above_price_false(self, zone: Zone) -> None:
        assert zone.is_above_price(103) is False

    def test_is_below_price_true(self, zone: Zone) -> None:
        assert zone.is_below_price(106) is True

    def test_is_below_price_false(self, zone: Zone) -> None:
        assert zone.is_below_price(103) is False


# -----------------------------------------------------------------------
# BetDecision tests
# -----------------------------------------------------------------------


class TestBetDecision:
    """BetDecision should_bet / is_skip logic."""

    def test_should_bet(self) -> None:
        bd = BetDecision(direction=Direction.BULLISH, confidence=0.6)
        assert bd.should_bet is True
        assert bd.is_skip is False

    def test_skip_when_no_direction(self) -> None:
        bd = BetDecision(direction=None, confidence=0.0)
        assert bd.should_bet is False
        assert bd.is_skip is True

    def test_skip_when_zero_confidence(self) -> None:
        bd = BetDecision(direction=Direction.BEARISH, confidence=0.0)
        assert bd.should_bet is False
        assert bd.is_skip is False  # direction is set, just no confidence


# -----------------------------------------------------------------------
# Enum validation tests
# -----------------------------------------------------------------------


class TestEnums:
    """Verify enum member counts and value types."""

    def test_direction_members(self) -> None:
        assert len(Direction) == 2

    def test_swing_type_members(self) -> None:
        assert len(SwingType) == 4

    def test_all_enum_values_are_strings(self) -> None:
        for enum_cls in (
            Direction,
            SwingType,
            SwingStrength,
            BOSType,
            ZoneType,
            ZonePattern,
            ZonePosition,
            ControlStateType,
            MarketPhase,
            ReturnType,
        ):
            for member in enum_cls:
                assert isinstance(member.value, str), f"{enum_cls.__name__}.{member.name} is not a string"
