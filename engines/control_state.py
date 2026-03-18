"""Control State: tracks whether supply or demand is currently in control.

Based on Book 14's "Who's In Control" concept:
- Price taps unmitigated supply -> supply in control
- Price breaks through supply -> demand in control
- Mirror for demand zones
"""

from __future__ import annotations

from typing import List, Optional

from utils.candle_types import (
    Candle,
    ControlStateType,
    Direction,
    Zone,
    ZoneType,
)


class ControlState:
    """State machine tracking whether supply or demand controls the market."""

    def __init__(self) -> None:
        self._state: ControlStateType = ControlStateType.NEUTRAL
        self._last_event: Optional[str] = None
        self._last_event_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, candle: Candle, zones: List[Zone]) -> ControlStateType:
        """Check candle interactions with *zones* and update the control state.

        Breaks override taps.  If multiple zones are affected, the most
        recent zone (highest index / latest timestamp) wins.
        """
        # Sort zones newest-first so the first match is the most recent.
        sorted_zones = sorted(zones, key=lambda z: z.timestamp, reverse=True)

        # --- Pass 1: check for BREAKS (higher priority) ---
        for z in sorted_zones:
            if z.is_broken:
                continue

            if z.zone_type == ZoneType.SUPPLY and candle.close > z.high:
                self._state = ControlStateType.DEMAND_CONTROL
                self._last_event = f"Broke supply at {z.high}"
                self._last_event_timestamp = candle.timestamp
                return self._state

            if z.zone_type == ZoneType.DEMAND and candle.close < z.low:
                self._state = ControlStateType.SUPPLY_CONTROL
                self._last_event = f"Broke demand at {z.low}"
                self._last_event_timestamp = candle.timestamp
                return self._state

        # --- Pass 2: check for TAPS (lower priority) ---
        for z in sorted_zones:
            if z.is_broken or z.is_mitigated:
                continue

            if z.zone_type == ZoneType.SUPPLY:
                if candle.high >= z.low and candle.close < z.high:
                    self._state = ControlStateType.SUPPLY_CONTROL
                    self._last_event = f"Tapped supply at {z.high}"
                    self._last_event_timestamp = candle.timestamp
                    return self._state

            if z.zone_type == ZoneType.DEMAND:
                if candle.low <= z.high and candle.close > z.low:
                    self._state = ControlStateType.DEMAND_CONTROL
                    self._last_event = f"Tapped demand at {z.low}"
                    self._last_event_timestamp = candle.timestamp
                    return self._state

        # No interaction — state persists.
        return self._state

    def get_state(self) -> ControlStateType:
        """Return the current control state."""
        return self._state

    def get_direction_bias(self) -> Optional[Direction]:
        """Map control state to a directional bias."""
        if self._state == ControlStateType.SUPPLY_CONTROL:
            return Direction.BEARISH
        if self._state == ControlStateType.DEMAND_CONTROL:
            return Direction.BULLISH
        return None

    def matches_direction(self, direction: Direction) -> bool:
        """True if *direction* aligns with (or doesn't conflict with) current state."""
        if self._state == ControlStateType.NEUTRAL:
            return True
        if self._state == ControlStateType.SUPPLY_CONTROL:
            return direction == Direction.BEARISH
        if self._state == ControlStateType.DEMAND_CONTROL:
            return direction == Direction.BULLISH
        return True

    @property
    def last_event(self) -> Optional[str]:
        """Description of what caused the last state change."""
        return self._last_event
