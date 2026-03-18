"""Shared mathematical utilities for the SMC engine."""

from __future__ import annotations

from typing import List, Optional, Sequence

from utils.candle_types import Candle


def calc_atr(candles: List[Candle], period: int = 14) -> float:
    """Average True Range over *period* candles.

    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close)).
    For the first candle (no previous), TR = high - low.
    Returns the simple average of the last *period* TR values.
    If fewer candles than *period*, uses what's available.
    If empty list, returns 0.0.
    """
    if not candles:
        return 0.0

    tr_values: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c.high - c.low
        else:
            prev_close = candles[i - 1].close
            tr = max(
                c.high - c.low,
                abs(c.high - prev_close),
                abs(c.low - prev_close),
            )
        tr_values.append(tr)

    recent = tr_values[-period:]
    return sum(recent) / len(recent)


def calc_velocity(values: List[float], window: int = 5) -> float:
    """Rate of change: (values[-1] - values[-window]) / window.

    If fewer values than *window*, uses what's available.
    If 0 or 1 values, returns 0.0.
    """
    if len(values) <= 1:
        return 0.0
    actual_window = min(window, len(values))
    return (values[-1] - values[-actual_window]) / actual_window


def calc_body_ratio(candle: Candle) -> float:
    """Convenience wrapper: returns *candle*.body_ratio."""
    return candle.body_ratio


def is_marubozu(candle: Candle, threshold: float = 0.80) -> bool:
    """Convenience wrapper: returns *candle*.is_marubozu(*threshold*)."""
    return candle.is_marubozu(threshold)


def calc_candle_velocity(candles: List[Candle], window: int = 5) -> float:
    """Price velocity using close prices over the last *window* candles.

    Extracts closes and delegates to :func:`calc_velocity`.
    """
    if not candles:
        return 0.0
    subset = candles[-window:] if len(candles) >= window else candles
    closes = [c.close for c in subset]
    return calc_velocity(closes, window)


def calc_acceleration(candles: List[Candle], window: int = 3) -> float:
    """Velocity change: recent_velocity - prior_velocity.

    Positive = accelerating (V-shape warning).
    Negative = decelerating (rounded / good).
    Returns 0.0 if not enough candles for both windows.
    """
    needed = window * 2
    if len(candles) < needed:
        return 0.0

    recent = candles[-window:]
    prior = candles[-needed:-window]

    recent_vel = calc_velocity([c.close for c in recent], window)
    prior_vel = calc_velocity([c.close for c in prior], window)

    return recent_vel - prior_vel


def classify_return_velocity(
    candles: List[Candle],
    lookback: int = 5,
    multiplier: float = 2.0,
) -> str:
    """Classify how price approaches a zone by its velocity profile.

    Analyses the last *lookback* candles' absolute body sizes.

    Returns:
        ``"v_shape"`` — last 3 bodies strictly increasing and largest >
        *multiplier* x average of earlier candles.
        ``"rounded"`` — last 3 bodies strictly decreasing.
        ``"corrective"`` — everything else.
    """
    if len(candles) < 3:
        return "corrective"

    subset = candles[-lookback:] if len(candles) >= lookback else list(candles)
    bodies = [c.body_size for c in subset]

    last3 = bodies[-3:]

    # Strictly increasing -> potential V-shape
    if last3[0] < last3[1] < last3[2]:
        earlier = bodies[:-3] if len(bodies) > 3 else bodies[:1]
        avg_earlier = sum(earlier) / len(earlier) if earlier else 1e-10
        if last3[-1] > multiplier * avg_earlier:
            return "v_shape"

    # Strictly decreasing -> rounded
    if last3[0] > last3[1] > last3[2]:
        return "rounded"

    return "corrective"


def find_equal_levels(
    prices: List[float],
    tolerance: float = 0.0005,
) -> List[tuple[int, int]]:
    """Find pairs of price levels that are approximately equal.

    *tolerance* is a fraction of price (0.0005 = 0.05%).
    Only compares non-adjacent pairs (indices at least 2 apart).
    """
    pairs: list[tuple[int, int]] = []
    n = len(prices)
    for i in range(n):
        for j in range(i + 2, n):
            if prices[i] == 0:
                continue
            if abs(prices[i] - prices[j]) < prices[i] * tolerance:
                pairs.append((i, j))
    return pairs
