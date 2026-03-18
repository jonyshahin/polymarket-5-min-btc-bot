"""Liquidity Engine: detects equal highs/lows and liquidity sweeps.

Liquidity = clusters of stop-losses at obvious levels (EQH, EQL, trendlines).
Sweeps = price wicking through liquidity then closing back (trap + reversal signal).
Internal sweeps suggest continuation; external sweeps suggest reversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from utils.candle_types import Candle, Direction, SwingPoint
import smc_config as cfg


@dataclass
class LiquidityLevel:
    """A detected pool of resting liquidity."""

    price: float
    level_type: str  # "EQH" or "EQL"
    swing_indices: list[int] = field(default_factory=list)
    is_swept: bool = False
    sweep_timestamp: float = 0.0
    sweep_direction: Optional[Direction] = None


@dataclass
class LiquiditySweep:
    """A confirmed sweep event."""

    timestamp: float
    level: LiquidityLevel
    sweep_candle: Candle
    direction_after: Direction  # expected reversal direction
    is_external: bool = False  # True if near HTF range boundary


class LiquidityEngine:
    """Detects liquidity pools (EQH/EQL) and sweeps."""

    def __init__(self) -> None:
        self._levels: List[LiquidityLevel] = []
        self._sweeps: List[LiquiditySweep] = []
        self._range_high: float = 0.0
        self._range_low: float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_range(self, range_high: float, range_low: float) -> None:
        """Set the HTF range boundaries for internal/external classification."""
        self._range_high = range_high
        self._range_low = range_low

    def update(
        self,
        candle: Candle,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> Optional[LiquiditySweep]:
        """Refresh level detection and check for sweeps on *candle*."""
        self.detect_levels(swing_highs, swing_lows)
        return self.check_sweep(candle)

    def detect_levels(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[LiquidityLevel]:
        """Scan swings for equal highs/lows and return newly detected levels."""
        new_levels: List[LiquidityLevel] = []

        # --- Equal highs ---
        for i in range(len(swing_highs)):
            for j in range(i + 2, len(swing_highs)):
                pa, pb = swing_highs[i].price, swing_highs[j].price
                if pa == 0:
                    continue
                if abs(pa - pb) < pa * cfg.EQH_EQL_TOLERANCE:
                    avg_price = (pa + pb) / 2
                    if not self._level_exists(avg_price, "EQH"):
                        lvl = LiquidityLevel(
                            price=avg_price,
                            level_type="EQH",
                            swing_indices=[i, j],
                        )
                        self._levels.append(lvl)
                        new_levels.append(lvl)

        # --- Equal lows ---
        for i in range(len(swing_lows)):
            for j in range(i + 2, len(swing_lows)):
                pa, pb = swing_lows[i].price, swing_lows[j].price
                if pa == 0:
                    continue
                if abs(pa - pb) < pa * cfg.EQH_EQL_TOLERANCE:
                    avg_price = (pa + pb) / 2
                    if not self._level_exists(avg_price, "EQL"):
                        lvl = LiquidityLevel(
                            price=avg_price,
                            level_type="EQL",
                            swing_indices=[i, j],
                        )
                        self._levels.append(lvl)
                        new_levels.append(lvl)

        return new_levels

    def check_sweep(self, candle: Candle) -> Optional[LiquiditySweep]:
        """Check if *candle* sweeps any unswept level."""
        candidates: List[LiquiditySweep] = []

        for lvl in self._levels:
            if lvl.is_swept:
                continue

            sweep: Optional[LiquiditySweep] = None

            if lvl.level_type == "EQH":
                # Wicked above, closed below.
                if candle.high > lvl.price and candle.close < lvl.price:
                    lvl.is_swept = True
                    lvl.sweep_timestamp = candle.timestamp
                    lvl.sweep_direction = Direction.BEARISH
                    sweep = LiquiditySweep(
                        timestamp=candle.timestamp,
                        level=lvl,
                        sweep_candle=candle,
                        direction_after=Direction.BEARISH,
                        is_external=self._is_external(lvl.price),
                    )

            elif lvl.level_type == "EQL":
                # Wicked below, closed above.
                if candle.low < lvl.price and candle.close > lvl.price:
                    lvl.is_swept = True
                    lvl.sweep_timestamp = candle.timestamp
                    lvl.sweep_direction = Direction.BULLISH
                    sweep = LiquiditySweep(
                        timestamp=candle.timestamp,
                        level=lvl,
                        sweep_candle=candle,
                        direction_after=Direction.BULLISH,
                        is_external=self._is_external(lvl.price),
                    )

            if sweep is not None:
                candidates.append(sweep)

        if not candidates:
            return None

        # Pick the level closest to the range boundary (most significant).
        best = min(candidates, key=lambda s: self._distance_to_range(s.level.price))
        self._sweeps.append(best)
        return best

    def get_active_levels(self) -> List[LiquidityLevel]:
        """Return unswept levels."""
        return [lvl for lvl in self._levels if not lvl.is_swept]

    def get_recent_sweeps(self, n: int = 5) -> List[LiquiditySweep]:
        """Return last *n* sweeps."""
        if n <= 0 or n >= len(self._sweeps):
            return list(self._sweeps)
        return list(self._sweeps[-n:])

    def get_sweep_within(
        self, lookback_candles: int, current_timestamp: float,
    ) -> Optional[LiquiditySweep]:
        """Return the most recent sweep within *lookback_candles* minutes of *current_timestamp*."""
        window = lookback_candles * 60
        for sweep in reversed(self._sweeps):
            if current_timestamp - sweep.timestamp <= window:
                return sweep
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _level_exists(self, price: float, level_type: str) -> bool:
        """True if a level at approximately *price* already exists."""
        for lvl in self._levels:
            if lvl.level_type != level_type:
                continue
            if price == 0:
                continue
            if abs(lvl.price - price) < price * cfg.EQH_EQL_TOLERANCE:
                return True
        return False

    def _is_external(self, price: float) -> bool:
        """True if *price* is within 0.2% of the HTF range boundary."""
        if self._range_high == 0 and self._range_low == float("inf"):
            return False
        threshold_high = self._range_high * 0.002
        threshold_low = self._range_low * 0.002
        if abs(price - self._range_high) < threshold_high:
            return True
        if abs(price - self._range_low) < threshold_low:
            return True
        return False

    def _distance_to_range(self, price: float) -> float:
        """Distance from *price* to the nearest range boundary."""
        d_high = abs(price - self._range_high) if self._range_high > 0 else float("inf")
        d_low = abs(price - self._range_low) if self._range_low < float("inf") else float("inf")
        return min(d_high, d_low)
