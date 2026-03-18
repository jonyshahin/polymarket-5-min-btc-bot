"""Confluence Engine: detects FVG, S/D flips, QM patterns, engulfing, return type.

Each pattern detected adds to the confluence score.
These are the "bonus" confirmations that turn a marginal signal into a high-confidence bet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from utils.candle_types import (
    BOS,
    Candle,
    Direction,
    ReturnType,
    SwingPoint,
    SwingType,
    Zone,
    ZoneType,
)
from utils.math_utils import classify_return_velocity
import smc_config as cfg


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FairValueGap:
    """A detected Fair Value Gap (imbalance)."""

    timestamp: float
    top: float
    bottom: float
    direction: Direction
    fill_pct: float = 0.0
    is_filled: bool = False
    candle_index: int = 0


@dataclass
class FlipPattern:
    """A detected S/D flip (zone held then broke)."""

    timestamp: float
    direction: Direction
    broken_zone: Zone
    new_zone_price: float


@dataclass
class QMPattern:
    """A detected Quasimodo pattern."""

    timestamp: float
    direction: Direction
    mpl_price: float  # Maximum Pain Level
    over_price: float
    under_price: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ConfluenceEngine:
    """Detects FVG, S/D flip, QM, engulfing, and return-type patterns."""

    def __init__(self) -> None:
        self._fvgs: List[FairValueGap] = []
        self._flips: List[FlipPattern] = []
        self._qm_patterns: List[QMPattern] = []
        self._candle_count: int = 0
        self._last_qm_swing_ts: float = 0.0
        self._detected_flip_zones: set[int] = set()  # zone ids by id()

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        candle: Candle,
        candles: List[Candle],
        swings: List[SwingPoint],
        zones: List[Zone],
        bos: Optional[BOS],
    ) -> None:
        """Process a new candle and detect confluence patterns."""
        self._candle_count += 1
        self._detect_fvg(candles)
        self._update_fvg_fills(candle)
        self._detect_flip(zones, bos)
        self._detect_qm(swings)

    # ------------------------------------------------------------------
    # FVG Detection
    # ------------------------------------------------------------------

    def _detect_fvg(self, candles: List[Candle]) -> Optional[FairValueGap]:
        """Detect a Fair Value Gap from the last 3 candles."""
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Bullish FVG: gap between c1 high and c3 low.
        if c1.high < c3.low:
            bottom = c1.high
            top = c3.low
            ref_price = c2.close if c2.close > 0 else 1.0
            if (top - bottom) >= ref_price * cfg.FVG_MIN_SIZE_PCT:
                fvg = FairValueGap(
                    timestamp=c2.timestamp,
                    top=top,
                    bottom=bottom,
                    direction=Direction.BULLISH,
                    candle_index=self._candle_count,
                )
                self._fvgs.append(fvg)
                return fvg

        # Bearish FVG: gap between c1 low and c3 high.
        if c1.low > c3.high:
            top = c1.low
            bottom = c3.high
            ref_price = c2.close if c2.close > 0 else 1.0
            if (top - bottom) >= ref_price * cfg.FVG_MIN_SIZE_PCT:
                fvg = FairValueGap(
                    timestamp=c2.timestamp,
                    top=top,
                    bottom=bottom,
                    direction=Direction.BEARISH,
                    candle_index=self._candle_count,
                )
                self._fvgs.append(fvg)
                return fvg

        return None

    def _update_fvg_fills(self, candle: Candle) -> None:
        """Update fill percentage on existing FVGs."""
        for fvg in self._fvgs:
            if fvg.is_filled:
                continue
            gap_size = fvg.top - fvg.bottom
            if gap_size <= 0:
                continue

            if fvg.direction == Direction.BULLISH:
                # Price fills from top downward.
                if candle.low <= fvg.top:
                    filled = fvg.top - max(candle.low, fvg.bottom)
                    fvg.fill_pct = max(fvg.fill_pct, min(filled / gap_size, 1.0))
            else:
                # Price fills from bottom upward.
                if candle.high >= fvg.bottom:
                    filled = min(candle.high, fvg.top) - fvg.bottom
                    fvg.fill_pct = max(fvg.fill_pct, min(filled / gap_size, 1.0))

            if fvg.fill_pct >= 0.50:
                fvg.is_filled = True

    # ------------------------------------------------------------------
    # S/D Flip Detection
    # ------------------------------------------------------------------

    def _detect_flip(
        self, zones: List[Zone], bos: Optional[BOS],
    ) -> Optional[FlipPattern]:
        """Detect an S/D flip: zone was mitigated then broken."""
        for z in zones:
            zone_id = id(z)
            if zone_id in self._detected_flip_zones:
                continue
            if not (z.is_mitigated and z.is_broken):
                continue
            # Only recent breaks.
            if self._candle_count - z.candle_index > 30:
                continue

            self._detected_flip_zones.add(zone_id)

            if z.zone_type == ZoneType.DEMAND:
                direction = Direction.BEARISH
            else:
                direction = Direction.BULLISH

            flip = FlipPattern(
                timestamp=z.timestamp,
                direction=direction,
                broken_zone=z,
                new_zone_price=z.midpoint,
            )
            self._flips.append(flip)
            return flip

        return None

    # ------------------------------------------------------------------
    # QM Pattern Detection
    # ------------------------------------------------------------------

    def _detect_qm(self, swings: List[SwingPoint]) -> Optional[QMPattern]:
        """Detect a Quasimodo pattern from the last 4 swings."""
        if len(swings) < 4:
            return None

        s1, s2, s3, s4 = swings[-4], swings[-3], swings[-2], swings[-1]

        # Avoid re-detecting the same pattern.
        if s4.timestamp <= self._last_qm_swing_ts:
            return None

        # Bearish QM: H - L - HH - LL
        if (
            s1.is_high
            and s2.is_low
            and s3.is_high
            and s4.is_low
            and s3.price > s1.price  # HH
            and s4.price < s2.price  # LL
        ):
            self._last_qm_swing_ts = s4.timestamp
            qm = QMPattern(
                timestamp=s4.timestamp,
                direction=Direction.BEARISH,
                mpl_price=s2.price,
                over_price=s3.price,
                under_price=s4.price,
            )
            self._qm_patterns.append(qm)
            return qm

        # Bullish QM: L - H - LL - HH
        if (
            s1.is_low
            and s2.is_high
            and s3.is_low
            and s4.is_high
            and s3.price < s1.price  # LL
            and s4.price > s2.price  # HH
        ):
            self._last_qm_swing_ts = s4.timestamp
            qm = QMPattern(
                timestamp=s4.timestamp,
                direction=Direction.BULLISH,
                mpl_price=s2.price,
                over_price=s4.price,
                under_price=s3.price,
            )
            self._qm_patterns.append(qm)
            return qm

        return None

    # ------------------------------------------------------------------
    # Return Type Classification
    # ------------------------------------------------------------------

    def classify_return_type(
        self, candles: List[Candle], lookback: int = 0,
    ) -> ReturnType:
        """Classify how price is approaching a zone."""
        lb = lookback if lookback > 0 else cfg.V_SHAPE_LOOKBACK
        result = classify_return_velocity(candles, lookback=lb)
        mapping = {
            "v_shape": ReturnType.V_SHAPE,
            "rounded": ReturnType.ROUNDED,
            "corrective": ReturnType.CORRECTIVE,
        }
        return mapping.get(result, ReturnType.UNKNOWN)

    # ------------------------------------------------------------------
    # Engulfing Momentum
    # ------------------------------------------------------------------

    def check_strong_engulfing(
        self, candles: List[Candle], zone: Optional[Zone] = None,
    ) -> Optional[Direction]:
        """Detect a strong engulfing pattern with follow-through."""
        if len(candles) < 3:
            return None

        engulf_candle = candles[-2]
        prior_candle = candles[-3]
        follow = candles[-1]

        if not engulf_candle.engulfs(prior_candle):
            return None

        # If zone provided, check engulfing occurred at the zone.
        if zone is not None:
            if not zone.contains_price(engulf_candle.body_midpoint):
                return None

        # Bearish engulfing with follow-through.
        if engulf_candle.is_bearish:
            if follow.is_bearish and follow.body_size > engulf_candle.body_size * 0.50:
                return Direction.BEARISH

        # Bullish engulfing with follow-through.
        if engulf_candle.is_bullish:
            if follow.is_bullish and follow.body_size > engulf_candle.body_size * 0.50:
                return Direction.BULLISH

        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_fvgs(self) -> List[FairValueGap]:
        """Return unfilled FVGs."""
        return [f for f in self._fvgs if not f.is_filled]

    def get_filled_fvgs(self, lookback: int = 10) -> List[FairValueGap]:
        """Return recently filled FVGs."""
        cutoff = self._candle_count - lookback
        return [
            f for f in self._fvgs
            if f.is_filled and f.candle_index >= cutoff
        ]

    def get_recent_flips(self, n: int = 3) -> List[FlipPattern]:
        """Return last *n* flip patterns."""
        if n <= 0 or n >= len(self._flips):
            return list(self._flips)
        return list(self._flips[-n:])

    def get_recent_qm(self) -> Optional[QMPattern]:
        """Return the most recent QM pattern, or None."""
        return self._qm_patterns[-1] if self._qm_patterns else None

    def has_recent_fvg_fill(
        self, direction: Direction, lookback: int = 5,
    ) -> bool:
        """True if an FVG matching *direction* was filled recently."""
        cutoff = self._candle_count - lookback
        return any(
            f.is_filled and f.direction == direction and f.candle_index >= cutoff
            for f in self._fvgs
        )
