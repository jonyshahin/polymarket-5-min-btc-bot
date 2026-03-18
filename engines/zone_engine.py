"""Zone Engine: identifies, tracks, and scores supply/demand zones.

After MarketStructure detects a BOS, ZoneEngine finds the origin candle
that caused the move and marks it as a supply or demand zone.
Zones are tracked for freshness, mitigation, and quality.
"""

from __future__ import annotations

from typing import List, Optional

from utils.candle_types import (
    BOS,
    BOSType,
    Candle,
    Direction,
    Zone,
    ZonePattern,
    ZonePosition,
    ZoneType,
)
import smc_config as cfg


class ZoneEngine:
    """Identifies, tracks, and scores supply/demand zones from BOS events."""

    def __init__(self) -> None:
        self._zones: List[Zone] = []
        self._candle_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        candle: Candle,
        bos: Optional[BOS],
        candles_buffer: List[Candle],
    ) -> Optional[Zone]:
        """Process a new candle, update existing zones, optionally create a new one."""
        self._candle_count += 1

        # Update mitigation / broken status for all existing zones.
        self._update_zone_status(candle)

        # Expire freshness.
        for z in self._zones:
            if not z.is_broken and z.is_fresh:
                age = self._candle_count - z.candle_index
                if age > cfg.ZONE_FRESHNESS_MAX_AGE:
                    z.is_fresh = False

        # Create a new zone if a BOS was detected.
        if bos is not None:
            new_zone = self._create_zone_from_bos(bos, candles_buffer)
            if new_zone is not None:
                self._zones.append(new_zone)
                return new_zone

        return None

    def get_active_zones(self) -> List[Zone]:
        """Return all zones that are not broken, newest first."""
        return [z for z in reversed(self._zones) if not z.is_broken]

    def get_fresh_zones(self) -> List[Zone]:
        """Return all zones that are fresh and not broken."""
        return [z for z in reversed(self._zones) if z.is_fresh and not z.is_broken]

    def get_supply_zones(self) -> List[Zone]:
        """Return active supply zones."""
        return [z for z in self.get_active_zones() if z.zone_type == ZoneType.SUPPLY]

    def get_demand_zones(self) -> List[Zone]:
        """Return active demand zones."""
        return [z for z in self.get_active_zones() if z.zone_type == ZoneType.DEMAND]

    def get_nearest_zone(self, price: float, zone_type: ZoneType) -> Optional[Zone]:
        """Find the nearest active zone of *zone_type* to *price*."""
        candidates = [
            z for z in self.get_active_zones() if z.zone_type == zone_type
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda z: abs(z.midpoint - price))

    def classify_zone_position(
        self, zone: Zone, range_high: float, range_low: float,
    ) -> ZonePosition:
        """Classify where *zone* sits within [range_low, range_high]."""
        span = range_high - range_low
        if span <= 0:
            return ZonePosition.MID
        pct = (zone.midpoint - range_low) / span
        if pct > 0.67:
            return ZonePosition.TOP
        if pct < 0.33:
            return ZonePosition.LOWER
        return ZonePosition.MID

    def update_positions(self, range_high: float, range_low: float) -> None:
        """Recalculate position for all active zones."""
        for z in self.get_active_zones():
            z.position = self.classify_zone_position(z, range_high, range_low)

    # ------------------------------------------------------------------
    # Zone status tracking
    # ------------------------------------------------------------------

    def _update_zone_status(self, candle: Candle) -> None:
        """Check if *candle* mitigated or broke any active zone."""
        for z in self._zones:
            if z.is_broken:
                continue

            if z.zone_type == ZoneType.SUPPLY:
                if candle.close > z.high:
                    z.is_broken = True
                    z.is_mitigated = True
                elif candle.high >= z.low and not z.is_mitigated:
                    z.is_mitigated = True
                    z.is_fresh = False

            elif z.zone_type == ZoneType.DEMAND:
                if candle.close < z.low:
                    z.is_broken = True
                    z.is_mitigated = True
                elif candle.low <= z.high and not z.is_mitigated:
                    z.is_mitigated = True
                    z.is_fresh = False

    # ------------------------------------------------------------------
    # Zone creation
    # ------------------------------------------------------------------

    def _create_zone_from_bos(
        self, bos: BOS, candles_buffer: List[Candle],
    ) -> Optional[Zone]:
        """Find the origin candle of the BOS and create a zone from it."""
        if len(candles_buffer) < 2:
            return None

        # Find the origin candle: walk backwards from the BOS candle.
        lookback = min(10, len(candles_buffer) - 1)
        search_start = len(candles_buffer) - 2  # skip the BOS candle itself
        search_end = max(0, search_start - lookback)

        if search_start < 0:
            return None

        origin_idx = search_start
        if bos.direction == Direction.BULLISH:
            # Demand zone: find the candle with the lowest low.
            lowest = candles_buffer[search_start].low
            for i in range(search_start, search_end - 1, -1):
                if candles_buffer[i].low < lowest:
                    lowest = candles_buffer[i].low
                    origin_idx = i
        else:
            # Supply zone: find the candle with the highest high.
            highest = candles_buffer[search_start].high
            for i in range(search_start, search_end - 1, -1):
                if candles_buffer[i].high > highest:
                    highest = candles_buffer[i].high
                    origin_idx = i

        origin = candles_buffer[origin_idx]

        # Determine zone boundaries based on wick ratio.
        if bos.direction == Direction.BULLISH:
            zone_type = ZoneType.DEMAND
            if origin.wick_ratio > cfg.ZONE_WICK_THRESHOLD:
                z_low = origin.low
                z_high = origin.body_low
            else:
                z_low = origin.low
                z_high = origin.body_high
        else:
            zone_type = ZoneType.SUPPLY
            if origin.wick_ratio > cfg.ZONE_WICK_THRESHOLD:
                z_low = origin.body_high
                z_high = origin.high
            else:
                z_low = origin.body_low
                z_high = origin.high

        # Ensure zone has non-zero height.
        if z_high <= z_low:
            z_high = z_low + 1e-10

        # Quality score.
        quality = 0
        if bos.bos_type == BOSType.IMPULSIVE:
            quality += 1
        if bos.is_choch:
            quality += 1
        if self._has_fvg_near(candles_buffer, origin_idx):
            quality += 1

        # Pattern classification.
        pattern = self._classify_pattern(bos, candles_buffer, origin_idx)

        zone = Zone(
            timestamp=origin.timestamp,
            high=z_high,
            low=z_low,
            zone_type=zone_type,
            pattern=pattern,
            quality_score=quality,
            is_fresh=True,
            is_mitigated=False,
            is_broken=False,
            creation_bos=bos,
            candle_index=self._candle_count,
        )
        return zone

    def _has_fvg_near(
        self, candles_buffer: List[Candle], origin_idx: int,
    ) -> bool:
        """Check for a Fair Value Gap within 5 candles of *origin_idx*."""
        start = max(1, origin_idx - 5)
        end = min(len(candles_buffer) - 1, origin_idx + 5)
        for i in range(start, end):
            if i + 1 >= len(candles_buffer) or i - 1 < 0:
                continue
            prev_c = candles_buffer[i - 1]
            next_c = candles_buffer[i + 1]
            # Bullish FVG: gap between prev high and next low.
            if prev_c.high < next_c.low:
                return True
            # Bearish FVG: gap between prev low and next high.
            if prev_c.low > next_c.high:
                return True
        return False

    def _classify_pattern(
        self,
        bos: BOS,
        candles_buffer: List[Candle],
        origin_idx: int,
    ) -> Optional[ZonePattern]:
        """Determine RBR/DBD/RBD/DBR from the move before and after the base."""
        # Look at 5 candles before the origin to determine the prior move.
        look_start = max(0, origin_idx - 5)
        before_candles = candles_buffer[look_start:origin_idx]
        if not before_candles:
            return None

        bullish_before = sum(1 for c in before_candles if c.is_bullish)
        bearish_before = sum(1 for c in before_candles if c.is_bearish)
        prior_rally = bullish_before > bearish_before

        # After-move direction is given by the BOS direction.
        after_rally = bos.direction == Direction.BULLISH

        if prior_rally and after_rally:
            return ZonePattern.RBR
        if not prior_rally and not after_rally:
            return ZonePattern.DBD
        if prior_rally and not after_rally:
            return ZonePattern.RBD
        if not prior_rally and after_rally:
            return ZonePattern.DBR

        return None
