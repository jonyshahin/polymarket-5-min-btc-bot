"""Market Structure Engine: detects swing points, BOS, CHoCH, and trend state.

Reads completed 1m candles and maintains a running model of market structure.
Other engines (zone, liquidity, confluence) consume this output.
"""

from __future__ import annotations

from typing import List, Optional

from utils.candle_types import (
    BOS,
    BOSType,
    Candle,
    Direction,
    MarketPhase,
    SwingPoint,
    SwingStrength,
    SwingType,
)
import smc_config as cfg


class MarketStructure:
    """Maintains a running model of market structure from 1m candles."""

    def __init__(self) -> None:
        self._candles: List[Candle] = []
        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        self._all_swings: List[SwingPoint] = []
        self._bos_history: List[BOS] = []
        self._candle_count: int = 0
        self._prev_trend: Optional[Direction] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, candle: Candle) -> Optional[BOS]:
        """Process a new 1m candle and return a BOS if one was detected."""
        self._candles.append(candle)
        if len(self._candles) > 120:
            self._candles = self._candles[-120:]
        self._candle_count += 1

        # Detect swings from the confirmed region of the buffer.
        sh = self._detect_swing_high()
        if sh is not None:
            self._swing_highs.append(sh)
            self._all_swings.append(sh)

        sl = self._detect_swing_low()
        if sl is not None:
            self._swing_lows.append(sl)
            self._all_swings.append(sl)

        # Check for BOS using the latest candle.
        bos = self._check_bos(candle)
        if bos is not None:
            bos.is_choch = self._check_choch(bos)
            self._bos_history.append(bos)

        return bos

    def get_swings(self, n: int = 0) -> List[SwingPoint]:
        """Return last *n* swings (all if *n* <= 0)."""
        if n <= 0 or n >= len(self._all_swings):
            return list(self._all_swings)
        return list(self._all_swings[-n:])

    def get_swing_highs(self, n: int = 0) -> List[SwingPoint]:
        """Return last *n* swing highs (all if *n* <= 0)."""
        if n <= 0 or n >= len(self._swing_highs):
            return list(self._swing_highs)
        return list(self._swing_highs[-n:])

    def get_swing_lows(self, n: int = 0) -> List[SwingPoint]:
        """Return last *n* swing lows (all if *n* <= 0)."""
        if n <= 0 or n >= len(self._swing_lows):
            return list(self._swing_lows)
        return list(self._swing_lows[-n:])

    def get_latest_bos(self) -> Optional[BOS]:
        """Return the most recent BOS event, or None."""
        return self._bos_history[-1] if self._bos_history else None

    def get_recent_bos(self, n: int = 5) -> List[BOS]:
        """Return last *n* BOS events."""
        if n <= 0 or n >= len(self._bos_history):
            return list(self._bos_history)
        return list(self._bos_history[-n:])

    def get_trend(self) -> MarketPhase:
        """Determine the current market phase from recent swings and BOS."""
        # Check for recent CHoCH → TRANSITION.
        if self._bos_history:
            recent = self._bos_history[-3:]
            recent_choch = any(b.is_choch for b in recent)
            if recent_choch:
                # Only TRANSITION if the CHoCH was within the last 3 candles.
                last_choch = [b for b in recent if b.is_choch][-1]
                if self._candle_count - last_choch.candle_index <= 3:
                    return MarketPhase.TRANSITION

        # Need at least 2 swing highs and 2 swing lows.
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return MarketPhase.RANGING

        last2_highs = self._swing_highs[-2:]
        last2_lows = self._swing_lows[-2:]

        all_hh = all(s.type == SwingType.HH for s in last2_highs)
        all_hl = all(s.type == SwingType.HL for s in last2_lows)
        if all_hh and all_hl:
            return MarketPhase.TRENDING_UP

        all_lh = all(s.type == SwingType.LH for s in last2_highs)
        all_ll = all(s.type == SwingType.LL for s in last2_lows)
        if all_lh and all_ll:
            return MarketPhase.TRENDING_DOWN

        return MarketPhase.RANGING

    def is_choch_detected(self, lookback: int = 3) -> bool:
        """Return True if any BOS in the last *lookback* events is a CHoCH."""
        recent = self._bos_history[-lookback:] if self._bos_history else []
        return any(b.is_choch for b in recent)

    def get_order_flow_count(self, direction: Direction) -> int:
        """Count consecutive BOS in *direction* from the most recent backwards."""
        count = 0
        for bos in reversed(self._bos_history):
            if bos.direction == direction:
                count += 1
            else:
                break
        return count

    @property
    def has_enough_data(self) -> bool:
        """True if at least 2 swing highs and 2 swing lows have been detected."""
        return len(self._swing_highs) >= 2 and len(self._swing_lows) >= 2

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def _detect_swing_high(self) -> Optional[SwingPoint]:
        """Detect a swing high at the candidate position in the candle buffer."""
        lookback = cfg.SWING_LOOKBACK
        needed = 2 * lookback + 1
        if len(self._candles) < needed:
            return None

        candidate_idx = len(self._candles) - lookback - 1
        candidate = self._candles[candidate_idx]

        # All candles to the left must have lower highs.
        for i in range(candidate_idx - lookback, candidate_idx):
            if self._candles[i].high >= candidate.high:
                return None

        # All candles to the right must have lower highs.
        for i in range(candidate_idx + 1, candidate_idx + lookback + 1):
            if self._candles[i].high >= candidate.high:
                return None

        # Classify HH or LH.
        if self._swing_highs:
            prev = self._swing_highs[-1]
            swing_type = SwingType.HH if candidate.high > prev.price else SwingType.LH
        else:
            swing_type = SwingType.HH

        return SwingPoint(
            timestamp=candidate.timestamp,
            price=candidate.high,
            type=swing_type,
            candle_index=self._candle_count - lookback - 1,
        )

    def _detect_swing_low(self) -> Optional[SwingPoint]:
        """Detect a swing low at the candidate position in the candle buffer."""
        lookback = cfg.SWING_LOOKBACK
        needed = 2 * lookback + 1
        if len(self._candles) < needed:
            return None

        candidate_idx = len(self._candles) - lookback - 1
        candidate = self._candles[candidate_idx]

        # All candles to the left must have higher lows.
        for i in range(candidate_idx - lookback, candidate_idx):
            if self._candles[i].low <= candidate.low:
                return None

        # All candles to the right must have higher lows.
        for i in range(candidate_idx + 1, candidate_idx + lookback + 1):
            if self._candles[i].low <= candidate.low:
                return None

        # Classify HL or LL.
        if self._swing_lows:
            prev = self._swing_lows[-1]
            swing_type = SwingType.LL if candidate.low < prev.price else SwingType.HL
        else:
            swing_type = SwingType.HL

        return SwingPoint(
            timestamp=candidate.timestamp,
            price=candidate.low,
            type=swing_type,
            candle_index=self._candle_count - lookback - 1,
        )

    # ------------------------------------------------------------------
    # BOS detection and classification
    # ------------------------------------------------------------------

    def _check_bos(self, candle: Candle) -> Optional[BOS]:
        """Check if *candle* breaks beyond the most recent opposing swing."""
        bullish_bos: Optional[BOS] = None
        bearish_bos: Optional[BOS] = None

        if self._swing_highs:
            last_high = self._swing_highs[-1]
            if candle.close > last_high.price:
                bos_type = self._classify_bos(candle, Direction.BULLISH)
                bullish_bos = BOS(
                    timestamp=candle.timestamp,
                    price=last_high.price,
                    direction=Direction.BULLISH,
                    bos_type=bos_type,
                    swing_origin=last_high,
                    candle_index=self._candle_count,
                )

        if self._swing_lows:
            last_low = self._swing_lows[-1]
            if candle.close < last_low.price:
                bos_type = self._classify_bos(candle, Direction.BEARISH)
                bearish_bos = BOS(
                    timestamp=candle.timestamp,
                    price=last_low.price,
                    direction=Direction.BEARISH,
                    bos_type=bos_type,
                    swing_origin=last_low,
                    candle_index=self._candle_count,
                )

        # If both triggered, pick the one matching candle direction.
        if bullish_bos and bearish_bos:
            return bullish_bos if candle.is_bullish else bearish_bos

        return bullish_bos or bearish_bos

    def _classify_bos(self, candle: Candle, bos_direction: Direction) -> BOSType:
        """Classify a BOS as IMPULSIVE or CORRECTIVE."""
        ratio = candle.body_ratio

        if ratio >= cfg.BOS_BODY_RATIO_IMPULSIVE:
            return BOSType.IMPULSIVE

        if ratio <= cfg.BOS_BODY_RATIO_CORRECTIVE:
            return BOSType.CORRECTIVE

        # In-between: check if the break took multiple candles.
        if len(self._candles) >= 4:
            broken_level = None
            if bos_direction == Direction.BULLISH and self._swing_highs:
                broken_level = self._swing_highs[-1].price
            elif bos_direction == Direction.BEARISH and self._swing_lows:
                broken_level = self._swing_lows[-1].price

            if broken_level is not None and broken_level != 0:
                threshold = broken_level * 0.001  # 0.1% of the level
                for prev_candle in self._candles[-4:-1]:
                    if abs(prev_candle.high - broken_level) < threshold or abs(prev_candle.close - broken_level) < threshold:
                        return BOSType.CORRECTIVE

        return BOSType.IMPULSIVE

    # ------------------------------------------------------------------
    # CHoCH detection
    # ------------------------------------------------------------------

    def _check_choch(self, bos: BOS) -> bool:
        """Determine if *bos* represents a Change of Character."""
        # Need at least 2 prior BOS events to determine prevailing trend.
        prior = self._bos_history[-3:] if len(self._bos_history) >= 2 else []
        if len(prior) < 2:
            return False

        bullish_count = sum(1 for b in prior if b.direction == Direction.BULLISH)
        bearish_count = sum(1 for b in prior if b.direction == Direction.BEARISH)

        if bullish_count > bearish_count and bos.direction == Direction.BEARISH:
            self._prev_trend = Direction.BEARISH
            return True

        if bearish_count > bullish_count and bos.direction == Direction.BULLISH:
            self._prev_trend = Direction.BULLISH
            return True

        self._prev_trend = bos.direction
        return False
