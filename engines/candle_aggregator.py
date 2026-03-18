"""Candle Aggregator: converts raw 1-second price ticks into 1m and 5m OHLC candles.

This is the first module in the SMC data pipeline.
Binance WS ticks -> CandleAggregator -> [1m candles] -> MarketStructure, ZoneEngine, etc.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import smc_config as cfg
from utils.candle_types import Candle


class CandleAggregator:
    """Aggregates raw (timestamp, price) ticks into 1-minute and 5-minute OHLC candles."""

    def __init__(self) -> None:
        self._ticks: list[tuple[float, float]] = []
        self._candles_1m: deque[Candle] = deque(maxlen=cfg.CANDLE_BUFFER_1M)
        self._candles_5m: deque[Candle] = deque(maxlen=cfg.CANDLE_BUFFER_5M)
        self._current_minute: int = 0
        self._current_5m_batch: list[Candle] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_tick(self, price: float, timestamp: float) -> Optional[Candle]:
        """Accept a single price tick and return a completed 1m Candle if a minute boundary was crossed.

        Returns None while ticks are still accumulating within the same minute.
        """
        tick_minute = int(timestamp) // 60 * 60

        # Guard: ignore ticks with timestamps going backwards.
        if tick_minute < self._current_minute:
            return None

        completed: Optional[Candle] = None

        # New minute boundary — finalize the previous minute if ticks exist.
        if tick_minute != self._current_minute and self._ticks:
            completed = self._finalize_1m_candle()
            self._candles_1m.append(completed)
            self._check_5m_completion(completed)
            self._ticks = []

        # First tick ever or after a boundary reset.
        if self._current_minute == 0 or tick_minute != self._current_minute:
            self._current_minute = tick_minute

        self._ticks.append((timestamp, price))
        return completed

    def get_1m_candles(self, n: int = 0) -> List[Candle]:
        """Return the last *n* completed 1m candles (all if *n* <= 0)."""
        candles = list(self._candles_1m)
        if n <= 0 or n >= len(candles):
            return candles
        return candles[-n:]

    def get_5m_candles(self, n: int = 0) -> List[Candle]:
        """Return the last *n* completed 5m candles (all if *n* <= 0)."""
        candles = list(self._candles_5m)
        if n <= 0 or n >= len(candles):
            return candles
        return candles[-n:]

    def get_current_partial(self) -> Optional[Candle]:
        """Build a temporary candle from the current tick buffer without finalizing.

        Returns None if no ticks are buffered.
        """
        if not self._ticks:
            return None
        return self._build_candle_from_ticks(self._ticks, self._current_minute)

    @property
    def has_enough_data(self) -> bool:
        """True if at least 10 completed 1m candles are available."""
        return len(self._candles_1m) >= 10

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalize_1m_candle(self) -> Candle:
        """Create a 1m Candle from the current tick buffer."""
        return self._build_candle_from_ticks(self._ticks, self._current_minute)

    def _check_5m_completion(self, completed_1m: Candle) -> Optional[Candle]:
        """Accumulate 1m candles and finalize a 5m candle every 5 minutes.

        Uses the 5-minute boundary (timestamp % 300 == 0) to detect when
        a new 5m period starts.  If the incoming candle sits on a new
        boundary and there is a pending batch, the batch is finalized first.
        """
        # If the new 1m candle is on a fresh 5m boundary, close out the
        # previous batch before starting the new one.
        if completed_1m.timestamp % 300 == 0 and self._current_5m_batch:
            five = self._build_5m_candle(self._current_5m_batch)
            self._candles_5m.append(five)
            self._current_5m_batch = [completed_1m]
            return five

        self._current_5m_batch.append(completed_1m)

        if len(self._current_5m_batch) == 5:
            five = self._build_5m_candle(self._current_5m_batch)
            self._candles_5m.append(five)
            self._current_5m_batch = []
            return five

        return None

    # ------------------------------------------------------------------
    # Static builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candle_from_ticks(
        ticks: list[tuple[float, float]],
        candle_ts: int,
    ) -> Candle:
        """Construct a Candle from a list of (timestamp, price) ticks."""
        prices = [p for _, p in ticks]
        return Candle(
            timestamp=float(candle_ts),
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=float(len(prices)),
        )

    @staticmethod
    def _build_5m_candle(batch: list[Candle]) -> Candle:
        """Construct a 5m Candle from a list of 1m candles."""
        return Candle(
            timestamp=batch[0].timestamp,
            open=batch[0].open,
            high=max(c.high for c in batch),
            low=min(c.low for c in batch),
            close=batch[-1].close,
            volume=sum(c.volume for c in batch),
        )
