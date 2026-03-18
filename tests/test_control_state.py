"""Tests for engines.control_state."""

import pytest

from engines.control_state import ControlState
from utils.candle_types import (
    Candle,
    ControlStateType,
    Direction,
    Zone,
    ZoneType,
)


def _c(ts: float, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c)


def _supply(high: float, low: float) -> Zone:
    return Zone(timestamp=0, high=high, low=low, zone_type=ZoneType.SUPPLY)


def _demand(high: float, low: float) -> Zone:
    return Zone(timestamp=0, high=high, low=low, zone_type=ZoneType.DEMAND)


# -----------------------------------------------------------------------
# Test 1: Initial state
# -----------------------------------------------------------------------


class TestInitialState:
    def test_neutral(self) -> None:
        cs = ControlState()
        assert cs.get_state() == ControlStateType.NEUTRAL
        assert cs.get_direction_bias() is None


# -----------------------------------------------------------------------
# Test 2: Supply tap -> SUPPLY_CONTROL
# -----------------------------------------------------------------------


class TestSupplyTap:
    def test_supply_tapped(self) -> None:
        cs = ControlState()
        zone = _supply(110, 108)
        # Candle wicks into zone but closes below zone.high.
        candle = _c(60, 107, 109, 106, 107)
        cs.update(candle, [zone])

        assert cs.get_state() == ControlStateType.SUPPLY_CONTROL
        assert cs.get_direction_bias() == Direction.BEARISH
        assert cs.last_event is not None
        assert "Tapped supply" in cs.last_event


# -----------------------------------------------------------------------
# Test 3: Supply break -> DEMAND_CONTROL
# -----------------------------------------------------------------------


class TestSupplyBreak:
    def test_supply_broken(self) -> None:
        cs = ControlState()
        zone = _supply(110, 108)
        # Candle closes above zone.high.
        candle = _c(60, 109, 112, 108, 111)
        cs.update(candle, [zone])

        assert cs.get_state() == ControlStateType.DEMAND_CONTROL
        assert cs.get_direction_bias() == Direction.BULLISH
        assert "Broke supply" in cs.last_event


# -----------------------------------------------------------------------
# Test 4: Demand tap -> DEMAND_CONTROL
# -----------------------------------------------------------------------


class TestDemandTap:
    def test_demand_tapped(self) -> None:
        cs = ControlState()
        zone = _demand(102, 100)
        # Candle low enters zone, closes above zone.low.
        candle = _c(60, 103, 104, 101, 103)
        cs.update(candle, [zone])

        assert cs.get_state() == ControlStateType.DEMAND_CONTROL
        assert cs.get_direction_bias() == Direction.BULLISH


# -----------------------------------------------------------------------
# Test 5: Demand break -> SUPPLY_CONTROL
# -----------------------------------------------------------------------


class TestDemandBreak:
    def test_demand_broken(self) -> None:
        cs = ControlState()
        zone = _demand(102, 100)
        # Candle closes below zone.low.
        candle = _c(60, 101, 101.5, 98, 99)
        cs.update(candle, [zone])

        assert cs.get_state() == ControlStateType.SUPPLY_CONTROL
        assert cs.get_direction_bias() == Direction.BEARISH


# -----------------------------------------------------------------------
# Test 6: matches_direction
# -----------------------------------------------------------------------


class TestMatchesDirection:
    def test_supply_control_bearish(self) -> None:
        cs = ControlState()
        cs._state = ControlStateType.SUPPLY_CONTROL
        assert cs.matches_direction(Direction.BEARISH) is True
        assert cs.matches_direction(Direction.BULLISH) is False

    def test_demand_control_bullish(self) -> None:
        cs = ControlState()
        cs._state = ControlStateType.DEMAND_CONTROL
        assert cs.matches_direction(Direction.BULLISH) is True
        assert cs.matches_direction(Direction.BEARISH) is False

    def test_neutral_matches_any(self) -> None:
        cs = ControlState()
        assert cs.matches_direction(Direction.BULLISH) is True
        assert cs.matches_direction(Direction.BEARISH) is True


# -----------------------------------------------------------------------
# Test 7: Break overrides tap
# -----------------------------------------------------------------------


class TestBreakOverridesTap:
    def test_break_wins(self) -> None:
        cs = ControlState()
        zone = _supply(110, 108)
        # Candle enters zone AND closes above it — a break.
        candle = _c(60, 109, 112, 108, 111)
        cs.update(candle, [zone])

        # Break should win -> DEMAND_CONTROL, not SUPPLY_CONTROL.
        assert cs.get_state() == ControlStateType.DEMAND_CONTROL


# -----------------------------------------------------------------------
# Test 8: State persists when no interaction
# -----------------------------------------------------------------------


class TestStatePersistence:
    def test_persists(self) -> None:
        cs = ControlState()
        zone = _supply(110, 108)
        # Tap supply.
        cs.update(_c(60, 107, 109, 106, 107), [zone])
        assert cs.get_state() == ControlStateType.SUPPLY_CONTROL

        # Feed candle far from any zone.
        zone.is_mitigated = True  # already tapped
        cs.update(_c(120, 90, 91, 89, 90), [zone])

        # State should persist.
        assert cs.get_state() == ControlStateType.SUPPLY_CONTROL
