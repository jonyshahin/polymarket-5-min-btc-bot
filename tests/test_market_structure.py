"""Tests for engines.market_structure."""

import pytest

import smc_config as cfg
from engines.market_structure import MarketStructure
from utils.candle_types import (
    BOS,
    BOSType,
    Candle,
    Direction,
    MarketPhase,
    SwingPoint,
    SwingType,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

LB = cfg.SWING_LOOKBACK  # 5


def make_candles(
    prices: list[tuple[float, float, float, float]],
    start_ts: float = 0.0,
) -> list[Candle]:
    """Create candles from (open, high, low, close) tuples."""
    return [
        Candle(timestamp=start_ts + i * 60, open=o, high=h, low=l, close=c)
        for i, (o, h, l, c) in enumerate(prices)
    ]


def _rising_candles(
    start: float,
    end: float,
    count: int = LB,
) -> list[tuple[float, float, float, float]]:
    """Generate *count* candles with highs/lows rising from *start* to *end*.

    Each candle has: high < *end* so the peak candle remains the unique maximum.
    """
    step = (end - start) / (count + 1)  # +1 so we never reach *end*
    result: list[tuple[float, float, float, float]] = []
    for i in range(count):
        mid = start + step * (i + 1)
        result.append((mid - 0.1, mid + 0.1, mid - 0.3, mid))
    return result


def _falling_candles(
    start: float,
    end: float,
    count: int = LB,
) -> list[tuple[float, float, float, float]]:
    """Generate *count* candles with highs/lows falling from *start* to *end*.

    Each candle has: low > *end* so the trough candle remains the unique minimum.
    """
    step = (start - end) / (count + 1)
    result: list[tuple[float, float, float, float]] = []
    for i in range(count):
        mid = start - step * (i + 1)
        result.append((mid + 0.1, mid + 0.3, mid - 0.1, mid))
    return result


def _peak_candle(price: float) -> tuple[float, float, float, float]:
    """A single candle whose high is *price* + 0.5 — clearly the highest."""
    return (price - 0.2, price + 0.5, price - 0.4, price + 0.1)


def _trough_candle(price: float) -> tuple[float, float, float, float]:
    """A single candle whose low is *price* - 0.5 — clearly the lowest."""
    return (price + 0.2, price + 0.4, price - 0.5, price - 0.1)


def _make_swing_peak(
    approach_from: float,
    peak: float,
    descend_to: float,
) -> list[tuple[float, float, float, float]]:
    """LB rising candles + 1 peak candle + LB falling candles.

    Guarantees the peak candle's high is strictly higher than all neighbours.
    """
    ohlc: list[tuple[float, float, float, float]] = []
    ohlc.extend(_rising_candles(approach_from, peak))
    ohlc.append(_peak_candle(peak))
    ohlc.extend(_falling_candles(peak, descend_to))
    return ohlc


def _make_swing_trough(
    approach_from: float,
    trough: float,
    ascend_to: float,
) -> list[tuple[float, float, float, float]]:
    """LB falling candles + 1 trough candle + LB rising candles."""
    ohlc: list[tuple[float, float, float, float]] = []
    ohlc.extend(_falling_candles(approach_from, trough))
    ohlc.append(_trough_candle(trough))
    ohlc.extend(_rising_candles(trough, ascend_to))
    return ohlc


def make_uptrend_candles(
    num_swings: int = 3,
    amplitude: float = 10.0,
    base: float = 100.0,
) -> list[Candle]:
    """Generate candles with *num_swings* higher-highs and higher-lows.

    Pattern per wave: trough → peak → next_trough (each higher than last).
    """
    ohlc: list[tuple[float, float, float, float]] = []

    for wave in range(num_swings):
        trough = base + wave * amplitude * 0.4
        peak = base + (wave + 1) * amplitude
        next_trough = base + (wave + 0.5) * amplitude * 0.4

        # First wave needs an initial approach to the trough.
        if wave == 0:
            ohlc.extend(_falling_candles(base + 2, trough + 1, count=LB))
            ohlc.append(_trough_candle(trough))
            ohlc.extend(_rising_candles(trough, peak, count=LB))
        else:
            ohlc.extend(_rising_candles(ohlc[-1][3], peak, count=LB))

        ohlc.append(_peak_candle(peak))
        ohlc.extend(_falling_candles(peak, next_trough, count=LB))
        ohlc.append(_trough_candle(next_trough))

    # Trailing candles to confirm the last swing trough.
    last_val = ohlc[-1][3]
    ohlc.extend(_rising_candles(last_val, last_val + amplitude * 0.5, count=LB + 1))

    return make_candles(ohlc)


def make_downtrend_candles(
    num_swings: int = 3,
    amplitude: float = 10.0,
    base: float = 150.0,
) -> list[Candle]:
    """Generate candles with *num_swings* lower-highs and lower-lows."""
    ohlc: list[tuple[float, float, float, float]] = []

    for wave in range(num_swings):
        peak = base - wave * amplitude * 0.4
        trough = base - (wave + 1) * amplitude
        next_peak = base - (wave + 0.5) * amplitude * 0.4

        if wave == 0:
            ohlc.extend(_rising_candles(base - 2, peak - 1, count=LB))
            ohlc.append(_peak_candle(peak))
            ohlc.extend(_falling_candles(peak, trough, count=LB))
        else:
            ohlc.extend(_falling_candles(ohlc[-1][3], trough, count=LB))

        ohlc.append(_trough_candle(trough))
        ohlc.extend(_rising_candles(trough, next_peak, count=LB))
        ohlc.append(_peak_candle(next_peak))

    # Trailing candles to confirm the last swing peak.
    last_val = ohlc[-1][3]
    ohlc.extend(_falling_candles(last_val, last_val - amplitude * 0.5, count=LB + 1))

    return make_candles(ohlc)


def feed_all(ms: MarketStructure, candles: list[Candle]) -> list[BOS]:
    """Feed all candles and collect any BOS events."""
    events: list[BOS] = []
    for c in candles:
        bos = ms.update(c)
        if bos is not None:
            events.append(bos)
    return events


# -----------------------------------------------------------------------
# Test 1: Uptrend swing detection
# -----------------------------------------------------------------------


class TestUptrendSwings:
    """Swing highs should be HH and swing lows should be HL in an uptrend."""

    def test_swing_highs_are_hh(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=3)
        feed_all(ms, candles)

        highs = ms.get_swing_highs()
        assert len(highs) >= 2
        for sh in highs[1:]:
            assert sh.type == SwingType.HH

    def test_swing_lows_are_hl(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=3)
        feed_all(ms, candles)

        lows = ms.get_swing_lows()
        assert len(lows) >= 2
        for sl in lows[1:]:
            assert sl.type == SwingType.HL

    def test_trend_is_up(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=4)
        feed_all(ms, candles)

        assert ms.get_trend() == MarketPhase.TRENDING_UP


# -----------------------------------------------------------------------
# Test 2: Downtrend swing detection
# -----------------------------------------------------------------------


class TestDowntrendSwings:
    """Swing highs should be LH and swing lows should be LL in a downtrend."""

    def test_swing_highs_are_lh(self) -> None:
        ms = MarketStructure()
        candles = make_downtrend_candles(num_swings=3)
        feed_all(ms, candles)

        highs = ms.get_swing_highs()
        assert len(highs) >= 2
        for sh in highs[1:]:
            assert sh.type == SwingType.LH

    def test_swing_lows_are_ll(self) -> None:
        ms = MarketStructure()
        candles = make_downtrend_candles(num_swings=3)
        feed_all(ms, candles)

        lows = ms.get_swing_lows()
        assert len(lows) >= 2
        for sl in lows[1:]:
            assert sl.type == SwingType.LL

    def test_trend_is_down(self) -> None:
        ms = MarketStructure()
        candles = make_downtrend_candles(num_swings=4)
        feed_all(ms, candles)

        assert ms.get_trend() == MarketPhase.TRENDING_DOWN


# -----------------------------------------------------------------------
# Test 3: BOS detection
# -----------------------------------------------------------------------


class TestBOSDetection:
    """A candle closing beyond a swing level triggers a BOS."""

    def test_bullish_bos(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=3)
        bos_events = feed_all(ms, candles)

        bullish = [b for b in bos_events if b.direction == Direction.BULLISH]
        assert len(bullish) >= 1
        for b in bullish:
            assert b.swing_origin is not None

    def test_bearish_bos(self) -> None:
        ms = MarketStructure()
        candles = make_downtrend_candles(num_swings=3)
        bos_events = feed_all(ms, candles)

        bearish = [b for b in bos_events if b.direction == Direction.BEARISH]
        assert len(bearish) >= 1
        for b in bearish:
            assert b.swing_origin is not None


# -----------------------------------------------------------------------
# Test 4: BOS classification — impulsive
# -----------------------------------------------------------------------


class TestBOSImpulsive:
    """A strong-bodied candle breaking structure -> IMPULSIVE."""

    def test_impulsive_classification(self) -> None:
        ms = MarketStructure()

        # Build a swing high at 110 then break it with a strong candle.
        ohlc: list[tuple[float, float, float, float]] = []
        ohlc.extend(_make_swing_peak(100, 110, 104))
        # Rise back near 110 but don't exceed the swing high yet.
        ohlc.extend(_rising_candles(104, 109, count=LB))
        # Strong breakout: open=108, high=114, low=107.5, close=113.
        # body = 5, range = 6.5, ratio ≈ 0.77.
        ohlc.append((108, 114, 107.5, 113))

        candles = make_candles(ohlc)
        bos_events = feed_all(ms, candles)

        impulsive = [b for b in bos_events if b.bos_type == BOSType.IMPULSIVE]
        assert len(impulsive) >= 1


# -----------------------------------------------------------------------
# Test 5: BOS classification — corrective
# -----------------------------------------------------------------------


class TestBOSCorrective:
    """A weak-bodied candle breaking structure -> CORRECTIVE."""

    def test_corrective_classification(self) -> None:
        ms = MarketStructure()

        ohlc: list[tuple[float, float, float, float]] = []
        # Swing high peak at 110 -> actual swing price = 110.5 (_peak_candle adds 0.5).
        ohlc.extend(_make_swing_peak(100, 110, 104))
        ohlc.extend(_rising_candles(104, 109.5, count=LB))
        # Weak breakout: must close above 110.5.  body=0.3, range=3.0, ratio=0.10.
        ohlc.append((110.5, 112, 109, 110.8))

        candles = make_candles(ohlc)
        bos_events = feed_all(ms, candles)

        corrective = [b for b in bos_events if b.bos_type == BOSType.CORRECTIVE]
        assert len(corrective) >= 1


# -----------------------------------------------------------------------
# Test 6: CHoCH detection
# -----------------------------------------------------------------------


class TestCHoCH:
    """A BOS that reverses the prevailing trend is a CHoCH."""

    def test_choch_after_uptrend(self) -> None:
        ms = MarketStructure()

        # Build uptrend with enough swings to generate bullish BOS.
        candles = make_uptrend_candles(num_swings=4, amplitude=10.0)
        feed_all(ms, candles)

        bullish_count = sum(
            1 for b in ms.get_recent_bos(10) if b.direction == Direction.BULLISH
        )
        assert bullish_count >= 2, "Need bullish BOS history first"

        # Sharp drop below the most recent swing low -> bearish CHoCH.
        lows = ms.get_swing_lows()
        assert len(lows) >= 1
        target = lows[-1].price
        crash = Candle(
            timestamp=candles[-1].timestamp + 60,
            open=target + 1,
            high=target + 1.5,
            low=target - 5,
            close=target - 4,
        )
        bos = ms.update(crash)
        assert bos is not None
        assert bos.direction == Direction.BEARISH
        assert bos.is_choch is True
        assert ms.is_choch_detected() is True

    def test_is_choch_detected_default_false(self) -> None:
        ms = MarketStructure()
        assert ms.is_choch_detected() is False


# -----------------------------------------------------------------------
# Test 7: Order flow count
# -----------------------------------------------------------------------


class TestOrderFlowCount:
    """Consecutive BOS in one direction."""

    def test_consecutive_bullish(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=4, amplitude=10.0)
        feed_all(ms, candles)

        bullish_count = ms.get_order_flow_count(Direction.BULLISH)
        assert bullish_count >= 1

    def test_streak_broken(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=4, amplitude=10.0)
        feed_all(ms, candles)

        lows = ms.get_swing_lows()
        assert len(lows) >= 1
        target = lows[-1].price
        crash = Candle(
            timestamp=candles[-1].timestamp + 60,
            open=target + 1,
            high=target + 1.5,
            low=target - 5,
            close=target - 4,
        )
        bos = ms.update(crash)
        assert bos is not None
        assert bos.direction == Direction.BEARISH
        assert ms.get_order_flow_count(Direction.BEARISH) == 1
        assert ms.get_order_flow_count(Direction.BULLISH) == 0


# -----------------------------------------------------------------------
# Test 8: Ranging trend
# -----------------------------------------------------------------------


class TestRangingTrend:
    """Mixed swings should give RANGING."""

    def test_ranging(self) -> None:
        ms = MarketStructure()

        ohlc: list[tuple[float, float, float, float]] = []

        # Peak at 110.
        ohlc.extend(_make_swing_peak(95, 110, 100))
        # Trough at 98.
        ohlc.extend(_make_swing_trough(100, 98, 106))
        # Peak at 108 (LH vs 110).
        ohlc.extend(_make_swing_peak(106, 108, 99))
        # Trough at 97 (LL vs 98).
        ohlc.extend(_make_swing_trough(99, 97, 105))
        # Peak at 112 (HH vs 108) — contradicts LH.
        ohlc.extend(_make_swing_peak(105, 112, 100))

        # Trailing.
        ohlc.extend(_falling_candles(100, 96, count=LB + 1))

        candles = make_candles(ohlc)
        feed_all(ms, candles)

        trend = ms.get_trend()
        assert trend in (MarketPhase.RANGING, MarketPhase.TRANSITION)


# -----------------------------------------------------------------------
# Test 9: has_enough_data
# -----------------------------------------------------------------------


class TestHasEnoughData:
    """Need >= 2 swing highs and >= 2 swing lows."""

    def test_fresh_is_false(self) -> None:
        ms = MarketStructure()
        assert ms.has_enough_data is False

    def test_after_enough_candles(self) -> None:
        ms = MarketStructure()
        candles = make_uptrend_candles(num_swings=3)
        feed_all(ms, candles)
        assert ms.has_enough_data is True


# -----------------------------------------------------------------------
# Test 10: Empty / minimal data
# -----------------------------------------------------------------------


class TestMinimalData:
    """Edge cases with zero or very few candles."""

    def test_single_candle_no_crash(self) -> None:
        ms = MarketStructure()
        result = ms.update(
            Candle(timestamp=0, open=100, high=105, low=95, close=103)
        )
        assert result is None

    def test_get_latest_bos_empty(self) -> None:
        ms = MarketStructure()
        assert ms.get_latest_bos() is None

    def test_get_swings_empty(self) -> None:
        ms = MarketStructure()
        assert ms.get_swings() == []

    def test_get_trend_empty(self) -> None:
        ms = MarketStructure()
        assert ms.get_trend() == MarketPhase.RANGING

    def test_order_flow_count_empty(self) -> None:
        ms = MarketStructure()
        assert ms.get_order_flow_count(Direction.BULLISH) == 0
        assert ms.get_order_flow_count(Direction.BEARISH) == 0
