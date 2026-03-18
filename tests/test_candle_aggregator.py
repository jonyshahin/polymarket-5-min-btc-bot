"""Tests for engines.candle_aggregator."""

import pytest

from engines.candle_aggregator import CandleAggregator
from utils.candle_types import Candle


def make_ticks(
    start_ts: float,
    count: int,
    start_price: float,
    trend: float = 0.1,
) -> list[tuple[float, float]]:
    """Generate (timestamp, price) ticks with a linear trend."""
    return [(start_ts + i, start_price + i * trend) for i in range(count)]


# -----------------------------------------------------------------------
# Basic 1-minute aggregation
# -----------------------------------------------------------------------


class TestBasic1mAggregation:
    """Feed ticks within one minute, then cross the boundary."""

    def test_no_candle_during_same_minute(self) -> None:
        agg = CandleAggregator()
        # 60 ticks all within the same minute (ts 0..59 -> minute 0)
        ticks = make_ticks(start_ts=0, count=60, start_price=100.0)
        results = [agg.feed_tick(p, ts) for ts, p in ticks]
        assert all(r is None for r in results)

    def test_candle_on_boundary_cross(self) -> None:
        agg = CandleAggregator()
        ticks = make_ticks(start_ts=0, count=60, start_price=100.0, trend=0.1)
        for ts, p in ticks:
            agg.feed_tick(p, ts)

        # First tick in the next minute triggers candle completion.
        candle = agg.feed_tick(200.0, 60.0)
        assert candle is not None
        assert isinstance(candle, Candle)

    def test_ohlc_correctness(self) -> None:
        agg = CandleAggregator()
        # Prices: 100.0, 100.1, 100.2, ..., 105.9  (60 ticks)
        ticks = make_ticks(start_ts=0, count=60, start_price=100.0, trend=0.1)
        for ts, p in ticks:
            agg.feed_tick(p, ts)

        candle = agg.feed_tick(200.0, 60.0)
        assert candle is not None
        assert candle.open == pytest.approx(100.0)
        assert candle.close == pytest.approx(100.0 + 59 * 0.1)
        assert candle.high == pytest.approx(100.0 + 59 * 0.1)
        assert candle.low == pytest.approx(100.0)
        assert candle.volume == 60.0


# -----------------------------------------------------------------------
# 5-minute aggregation
# -----------------------------------------------------------------------


class TestFiveMinuteAggregation:
    """Feed ticks spanning 5 complete minutes."""

    def test_5m_candle_created(self) -> None:
        agg = CandleAggregator()
        # Use timestamps aligned to a 5m boundary: start at 300 (minute 5).
        base_ts = 300  # 5m-aligned
        completed_1m: list[Candle] = []

        for minute_offset in range(5):
            minute_start = base_ts + minute_offset * 60
            ticks = make_ticks(
                start_ts=minute_start,
                count=60,
                start_price=100.0 + minute_offset,
                trend=0.01,
            )
            for ts, p in ticks:
                result = agg.feed_tick(p, ts)
                if result is not None:
                    completed_1m.append(result)

        # Trigger the 5th minute's candle by feeding a tick in minute 6.
        trigger_ts = base_ts + 5 * 60
        result = agg.feed_tick(200.0, trigger_ts)
        if result is not None:
            completed_1m.append(result)

        # We should have 5 completed 1m candles.
        assert len(completed_1m) == 5

        # And a 5m candle should exist.
        fives = agg.get_5m_candles()
        assert len(fives) >= 1

        five = fives[-1]
        assert five.open == completed_1m[0].open
        assert five.close == completed_1m[-1].close
        assert five.high == max(c.high for c in completed_1m)
        assert five.low == min(c.low for c in completed_1m)


# -----------------------------------------------------------------------
# get_1m_candles
# -----------------------------------------------------------------------


class TestGet1mCandles:
    """Retrieval of completed 1m candles."""

    def _feed_n_minutes(self, agg: CandleAggregator, n: int) -> None:
        """Feed ticks for *n* complete minutes, then trigger the last one."""
        for m in range(n):
            ticks = make_ticks(
                start_ts=m * 60,
                count=60,
                start_price=100.0 + m,
            )
            for ts, p in ticks:
                agg.feed_tick(p, ts)
        # Trigger the final minute's candle.
        agg.feed_tick(999.0, n * 60)

    def test_get_last_n(self) -> None:
        agg = CandleAggregator()
        self._feed_n_minutes(agg, 3)
        assert len(agg.get_1m_candles(2)) == 2

    def test_get_all(self) -> None:
        agg = CandleAggregator()
        self._feed_n_minutes(agg, 3)
        assert len(agg.get_1m_candles(0)) == 3

    def test_get_more_than_available(self) -> None:
        agg = CandleAggregator()
        self._feed_n_minutes(agg, 3)
        assert len(agg.get_1m_candles(100)) == 3


# -----------------------------------------------------------------------
# get_current_partial
# -----------------------------------------------------------------------


class TestGetCurrentPartial:
    """Partial candle from uncommitted ticks."""

    def test_partial_candle(self) -> None:
        agg = CandleAggregator()
        ticks = make_ticks(start_ts=0, count=10, start_price=100.0, trend=0.5)
        for ts, p in ticks:
            agg.feed_tick(p, ts)

        partial = agg.get_current_partial()
        assert partial is not None
        assert partial.open == pytest.approx(100.0)
        assert partial.close == pytest.approx(100.0 + 9 * 0.5)
        assert partial.volume == 10.0

    def test_no_completed_candle_yet(self) -> None:
        agg = CandleAggregator()
        ticks = make_ticks(start_ts=0, count=10, start_price=100.0)
        for ts, p in ticks:
            agg.feed_tick(p, ts)
        assert len(agg.get_1m_candles()) == 0


# -----------------------------------------------------------------------
# has_enough_data
# -----------------------------------------------------------------------


class TestHasEnoughData:
    """Minimum data threshold for swing detection."""

    def test_fresh_aggregator(self) -> None:
        agg = CandleAggregator()
        assert agg.has_enough_data is False

    def test_after_10_minutes(self) -> None:
        agg = CandleAggregator()
        for m in range(10):
            ticks = make_ticks(
                start_ts=m * 60,
                count=60,
                start_price=100.0 + m,
            )
            for ts, p in ticks:
                agg.feed_tick(p, ts)
        # Trigger the 10th candle.
        agg.feed_tick(999.0, 10 * 60)
        assert agg.has_enough_data is True


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_empty_aggregator_partial_is_none(self) -> None:
        agg = CandleAggregator()
        assert agg.get_current_partial() is None

    def test_empty_aggregator_has_enough_data_false(self) -> None:
        agg = CandleAggregator()
        assert agg.has_enough_data is False

    def test_backward_timestamp_ignored(self) -> None:
        agg = CandleAggregator()
        agg.feed_tick(100.0, 120.0)  # minute 2
        result = agg.feed_tick(99.0, 50.0)  # minute 0 — backwards
        assert result is None
        # Only the first tick should be in the buffer.
        partial = agg.get_current_partial()
        assert partial is not None
        assert partial.volume == 1.0
