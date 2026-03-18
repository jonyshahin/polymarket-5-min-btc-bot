"""TA engine with trend context.

Early strategy: macro trend (EMA60/180, prior window momentum, volatility)
  + micro signals (RSI7, EMA3/7, short momentum) + order book imbalance.
Late strategy: RSI(14), EMA(9/21), 10s/30s momentum (unchanged).
"""

import logging
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class TrendContext:
    """Multi-window trend indicators computed from the full tick buffer."""
    ema60: float          # EMA over ~60 ticks
    ema180: float         # EMA over ~180 ticks
    trend_signal: float   # -1 to +1: EMA60 vs EMA180 crossover
    prior_window_delta: float  # Price change in previous 5-min window (raw $)
    prior_window_signal: float # -1 to +1: mean reversion signal (flipped from prior window direction)
    volatility: float     # Stddev of 1s returns
    vol_regime: float     # 0 to 1: 0 = low vol (trends persist), 1 = high vol (noisy)


@dataclass
class SignalBreakdown:
    """Detailed breakdown of all signal components for debugging."""
    # Trend
    trend_ema_signal: float
    prior_window_signal: float
    trend_combined: float
    # Micro
    rsi_signal: float
    ema_cross_signal: float
    momentum_signal: float
    vol_spike_signal: float
    micro_combined: float
    # Other
    ob_signal: float
    vol_regime: float
    # Weights applied
    trend_weighted: float
    micro_weighted: float
    ob_weighted: float
    vol_adjustment: float


@dataclass
class Signal:
    """Output of the strategy engine."""
    score: float       # -1.0 (strong down) to +1.0 (strong up)
    rsi: float
    ema_fast: float
    ema_slow: float
    momentum: float
    confidence: float  # 0.0–1.0
    ob_imbalance: float = 0.0
    breakdown: SignalBreakdown | None = None

    @property
    def ema9(self) -> float:
        return self.ema_fast

    @property
    def ema21(self) -> float:
        return self.ema_slow

    @property
    def direction(self) -> str:
        return "UP" if self.score >= 0 else "DOWN"

    @property
    def model_prob(self) -> float:
        """Convert score to probability of Up winning (0.0–1.0)."""
        raw = (self.score + 1.0) / 2.0
        prob = 0.5 + (raw - 0.5) * self.confidence + 0.005
        return max(0.01, min(0.99, prob))


# ── Helpers ───────────────────────────────────────────────────────────────

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ── Trend context ─────────────────────────────────────────────────────────

def compute_trend_context(
    all_prices: list[float],
    prior_window_delta: float | None,
) -> TrendContext | None:
    """Compute macro trend indicators from the full rolling price buffer.

    all_prices: entire tick history (up to 600 ticks / 10 min)
    prior_window_delta: price change in the previous completed 5-min window
    """
    if len(all_prices) < 60:
        return None

    arr = np.array(all_prices, dtype=np.float64)

    # EMA(60) vs EMA(180) — macro trend
    ema60 = _ema(arr, 60)[-1]
    period_180 = min(180, len(arr))
    ema180 = _ema(arr, period_180)[-1]

    # Normalize crossover: fraction of price, scaled
    ema_diff = (ema60 - ema180) / ema180 * 500  # amplify for sensitivity
    trend_signal = float(np.clip(ema_diff, -1.0, 1.0))

    # Prior window mean reversion signal (FLIPPED from continuation).
    # Data-driven: 30-window sample showed 34% continuation / 66% reversal.
    # Prior UP window → push score DOWN, and vice versa.
    prior_signal = 0.0
    if prior_window_delta is not None:
        # Normalize by recent price level
        price_level = arr[-1]
        if price_level > 0:
            # ~$50 move in 5min is a strong signal for BTC at ~$75k
            normalized = prior_window_delta / price_level * 1000
            prior_signal = float(np.clip(-normalized, -1.0, 1.0))  # negative = mean reversion

    # Volatility regime from 1s returns
    if len(arr) >= 60:
        returns = np.diff(arr[-300:]) / arr[-300:-1] if len(arr) >= 300 else np.diff(arr[-60:]) / arr[-60:-1]
        volatility = float(np.std(returns))
    else:
        volatility = 0.0

    # Map volatility to 0–1 regime. Typical BTC 1s return stddev is ~0.0001–0.0005.
    # Above 0.0003 = high vol regime.
    vol_regime = float(np.clip(volatility / 0.0003 - 0.5, 0.0, 1.0))

    return TrendContext(
        ema60=ema60, ema180=ema180,
        trend_signal=trend_signal,
        prior_window_delta=prior_window_delta or 0.0,
        prior_window_signal=prior_signal,
        volatility=volatility,
        vol_regime=vol_regime,
    )


# ── Early-entry strategy (with trend) ────────────────────────────────────

def compute_signal_early(
    prices: list[float],
    volumes: list[float] | None = None,
    ob_imbalance: float = 0.0,
    trend: TrendContext | None = None,
) -> Signal | None:
    """Early-entry signal combining macro trend + micro indicators.

    prices: recent ticks (current window, ~8-30 data points)
    trend: macro context from full buffer (passed in from caller)
    """
    if len(prices) < 8:
        log.debug("Not enough price data (%d ticks, need 8)", len(prices))
        return None

    arr = np.array(prices, dtype=np.float64)

    # ── Micro signals ─────────────────────────────────────────────────
    rsi = _rsi(arr, 7)
    rsi_signal = float(np.clip((rsi - 50.0) / 15.0, -1.0, 1.0))

    ema3 = _ema(arr, 3)
    ema7 = _ema(arr, 7)
    ema3_now, ema7_now = ema3[-1], ema7[-1]
    ema_diff = (ema3_now - ema7_now) / ema7_now * 1000
    ema_cross_signal = float(np.clip(ema_diff, -1.0, 1.0))

    n_short = min(5, len(arr))
    n_long = min(10, len(arr))
    recent_avg = np.mean(arr[-n_short:])
    longer_avg = np.mean(arr[-n_long:])
    momentum_raw = (recent_avg - longer_avg) / longer_avg * 1000
    momentum_signal = float(np.clip(momentum_raw, -1.0, 1.0))

    vol_spike_signal = 0.0
    if volumes and len(volumes) >= 5:
        varr = np.array(volumes[-10:], dtype=np.float64)
        n_recent = min(3, len(varr))
        recent_vol = np.mean(varr[-n_recent:])
        avg_vol = np.mean(varr)
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            vol_spike_signal = float(np.clip((vol_ratio - 1.0) * 0.5, -0.5, 0.5))

    # Micro composite (internal weighting)
    micro_combined = (
        0.25 * rsi_signal
        + 0.30 * ema_cross_signal
        + 0.35 * momentum_signal
        + 0.10 * vol_spike_signal
    )

    # ── Order book signal ─────────────────────────────────────────────
    ob_signal = float(np.clip(ob_imbalance, -1.0, 1.0))

    # ── Trend signals ─────────────────────────────────────────────────
    trend_ema_signal = 0.0
    prior_window_signal = 0.0
    trend_combined = 0.0
    vol_regime = 0.5  # default: medium volatility

    if trend is not None:
        trend_ema_signal = trend.trend_signal
        prior_window_signal = trend.prior_window_signal
        vol_regime = trend.vol_regime

        # Trend composite: EMA trend is dominant, prior window adds mean reversion
        trend_combined = 0.65 * trend_ema_signal + 0.35 * prior_window_signal

    # ── Weighted final score ──────────────────────────────────────────
    # Allocate weights based on whether we have trend data
    if trend is not None:
        tw = config.TREND_WEIGHT      # 0.40
        mw = config.MICRO_WEIGHT      # 0.35
        ow = config.ORDERBOOK_WEIGHT  # 0.15
        vw = config.VOLATILITY_WEIGHT # 0.10
    else:
        # No trend data yet — fall back to micro-heavy
        tw = 0.0
        mw = 0.65
        ow = 0.25
        vw = 0.10

    trend_weighted = tw * trend_combined
    micro_weighted = mw * micro_combined
    ob_weighted = ow * ob_signal

    # Volatility adjustment: high vol reduces magnitude (less confident)
    vol_adjustment = 1.0 - (vw * vol_regime)

    score = (trend_weighted + micro_weighted + ob_weighted) * vol_adjustment
    score = float(np.clip(score, -1.0, 1.0))

    # ── Confidence ────────────────────────────────────────────────────
    # Higher when trend and micro signals agree
    all_signals = [trend_combined, micro_combined, ob_signal]
    signs = [1 if s > 0.05 else -1 if s < -0.05 else 0 for s in all_signals]
    nonzero = [s for s in signs if s != 0]
    if nonzero:
        agreement = abs(sum(nonzero)) / len(nonzero)
    else:
        agreement = 0.0
    magnitude = np.mean([abs(s) for s in all_signals])

    # Data quality factor: more ticks + trend data = higher confidence
    data_factor = min(1.0, len(prices) / 15.0)
    trend_bonus = 0.2 if trend is not None else 0.0

    # High volatility reduces confidence
    vol_penalty = vol_regime * 0.2

    confidence = float(np.clip(
        (agreement * 0.4 + magnitude * 0.3 + trend_bonus) * data_factor - vol_penalty,
        0.0, 1.0,
    ))

    breakdown = SignalBreakdown(
        trend_ema_signal=trend_ema_signal,
        prior_window_signal=prior_window_signal,
        trend_combined=trend_combined,
        rsi_signal=rsi_signal,
        ema_cross_signal=ema_cross_signal,
        momentum_signal=momentum_signal,
        vol_spike_signal=vol_spike_signal,
        micro_combined=micro_combined,
        ob_signal=ob_signal,
        vol_regime=vol_regime,
        trend_weighted=trend_weighted,
        micro_weighted=micro_weighted,
        ob_weighted=ob_weighted,
        vol_adjustment=vol_adjustment,
    )

    return Signal(
        score=score, rsi=rsi, ema_fast=ema3_now, ema_slow=ema7_now,
        momentum=float(momentum_raw), confidence=confidence,
        ob_imbalance=float(ob_signal), breakdown=breakdown,
    )


# ── LMSR price velocity signal ───────────────────────────────────────────

@dataclass
class LMSRSignal:
    """Signal derived from Polymarket price dynamics only."""
    direction: str       # "UP" or "DOWN"
    velocity: float      # $/sec of up_price movement
    acceleration: float  # change in velocity
    confidence: float    # 0.0–1.0
    up_price: float
    down_price: float
    snapshots_used: int

    @property
    def model_prob(self) -> float:
        """Convert velocity direction + confidence to probability."""
        if self.direction == "UP":
            prob = 0.5 + self.confidence * 0.3 + 0.005  # tie bias
        else:
            prob = 0.5 - self.confidence * 0.3 + 0.005
        return max(0.01, min(0.99, prob))


def compute_lmsr_velocity(snapshots: list[dict]) -> tuple[float, float]:
    """Compute velocity and acceleration from market snapshots.

    snapshots: list of dicts with 'up_price' and 'seconds_into_window' keys.
    Returns (velocity, acceleration) in $/sec.
    """
    if len(snapshots) < 2:
        return 0.0, 0.0

    velocities = []
    for i in range(1, len(snapshots)):
        dt = snapshots[i]["seconds_into_window"] - snapshots[i - 1]["seconds_into_window"]
        if dt <= 0:
            continue
        dp = snapshots[i]["up_price"] - snapshots[i - 1]["up_price"]
        velocities.append(dp / dt)

    if not velocities:
        return 0.0, 0.0

    velocity = velocities[-1]

    # Acceleration from last two velocities
    acceleration = 0.0
    if len(velocities) >= 2:
        acceleration = velocities[-1] - velocities[-2]

    return velocity, acceleration


def compute_signal_lmsr(snapshots: list[dict]) -> LMSRSignal | None:
    """LMSR strategy: use only Polymarket price dynamics.

    snapshots: list of market_snapshot dicts ordered by time.
    """
    if len(snapshots) < config.LMSR_MIN_SNAPSHOTS:
        return None

    velocity, acceleration = compute_lmsr_velocity(snapshots)

    # Determine direction from velocity
    if abs(velocity) < config.LMSR_VELOCITY_THRESHOLD * 0.5:
        # Very low velocity — no clear signal
        return None

    direction = "UP" if velocity > 0 else "DOWN"

    latest = snapshots[-1]
    up_price = latest["up_price"]
    down_price = latest["down_price"]

    # Confidence based on velocity magnitude and acceleration agreement
    vel_magnitude = min(abs(velocity) / config.LMSR_VELOCITY_THRESHOLD, 2.0) / 2.0
    accel_bonus = 0.2 if (acceleration > 0 and velocity > 0) or (acceleration < 0 and velocity < 0) else 0.0
    # More snapshots = more confident
    data_factor = min(1.0, len(snapshots) / 10.0)

    confidence = float(np.clip(vel_magnitude * 0.6 + accel_bonus + data_factor * 0.2, 0.0, 1.0))

    return LMSRSignal(
        direction=direction,
        velocity=velocity,
        acceleration=acceleration,
        confidence=confidence,
        up_price=up_price,
        down_price=down_price,
        snapshots_used=len(snapshots),
    )


# ── Selective strategy (reversal + LMSR confluence) ──────────────────────

@dataclass
class SelectiveSignal:
    """Signal from the selective strategy combining reversal + LMSR velocity."""
    direction: str        # "UP" or "DOWN"
    reversal_signal: float   # mean reversion from prior window (-1 to +1)
    velocity: float          # LMSR price velocity
    acceleration: float
    confidence: float        # 0.0–1.0
    up_price: float
    down_price: float
    skip_reason: str | None = None  # if set, signal is a skip recommendation

    @property
    def model_prob(self) -> float:
        """Convert direction + confidence to probability."""
        if self.direction == "UP":
            prob = 0.5 + self.confidence * 0.35 + 0.005  # tie bias
        else:
            prob = 0.5 - self.confidence * 0.35 + 0.005
        return max(0.01, min(0.99, prob))


def compute_signal_selective(
    prior_window_delta: float | None,
    prior_btc_price: float | None,
    lmsr_snapshots: list[dict],
) -> SelectiveSignal | None:
    """Data-driven selective strategy.

    Combines:
    1. Mean reversion from prior window (66% reversal rate from 30-window sample)
    2. LMSR velocity (follow smart money when velocity is high)
    3. Only trade when BOTH agree — skip when they disagree

    Logic:
    - If prior window was UP and LMSR velocity is negative (DOWN) -> BUY DOWN
    - If prior window was DOWN and LMSR velocity is positive (UP) -> BUY UP
    - If they disagree -> SKIP (no confluence)
    - If prior window move was small -> SKIP (no reversal signal)
    - If LMSR velocity is low -> SKIP (no smart money signal)
    """
    if not lmsr_snapshots or len(lmsr_snapshots) < config.LMSR_MIN_SNAPSHOTS:
        return None

    velocity, acceleration = compute_lmsr_velocity(lmsr_snapshots)
    latest = lmsr_snapshots[-1]
    up_price = latest["up_price"]
    down_price = latest["down_price"]

    # Check prior window reversal signal
    if prior_window_delta is None or prior_btc_price is None or prior_btc_price <= 0:
        return SelectiveSignal(
            direction="UP", reversal_signal=0.0, velocity=velocity,
            acceleration=acceleration, confidence=0.0,
            up_price=up_price, down_price=down_price,
            skip_reason="no prior window data",
        )

    pct_move = abs(prior_window_delta) / prior_btc_price
    if pct_move < config.SELECTIVE_MIN_PRIOR_MOVE:
        return SelectiveSignal(
            direction="UP", reversal_signal=0.0, velocity=velocity,
            acceleration=acceleration, confidence=0.0,
            up_price=up_price, down_price=down_price,
            skip_reason=f"prior move too small ({pct_move:.4%} < {config.SELECTIVE_MIN_PRIOR_MOVE:.4%})",
        )

    # Reversal direction: prior UP -> expect DOWN, prior DOWN -> expect UP
    reversal_dir = "DOWN" if prior_window_delta > 0 else "UP"
    normalized_rev = prior_window_delta / prior_btc_price * 1000
    reversal_signal = float(np.clip(-normalized_rev, -1.0, 1.0))

    # LMSR velocity direction
    if abs(velocity) < config.LMSR_VELOCITY_THRESHOLD:
        return SelectiveSignal(
            direction=reversal_dir, reversal_signal=reversal_signal,
            velocity=velocity, acceleration=acceleration, confidence=0.0,
            up_price=up_price, down_price=down_price,
            skip_reason=f"low LMSR velocity ({abs(velocity):.4f} < {config.LMSR_VELOCITY_THRESHOLD})",
        )

    lmsr_dir = "UP" if velocity > 0 else "DOWN"

    # Confluence check: reversal and LMSR must agree
    if reversal_dir != lmsr_dir:
        return SelectiveSignal(
            direction=reversal_dir, reversal_signal=reversal_signal,
            velocity=velocity, acceleration=acceleration, confidence=0.0,
            up_price=up_price, down_price=down_price,
            skip_reason=f"no confluence (reversal={reversal_dir}, LMSR={lmsr_dir})",
        )

    # Both agree! Compute confidence from both signal strengths
    rev_strength = min(abs(reversal_signal), 1.0)
    vel_strength = min(abs(velocity) / config.LMSR_VELOCITY_THRESHOLD, 2.0) / 2.0
    accel_bonus = 0.15 if (acceleration > 0 and velocity > 0) or (acceleration < 0 and velocity < 0) else 0.0
    data_factor = min(1.0, len(lmsr_snapshots) / 8.0)

    confidence = float(np.clip(
        rev_strength * 0.35 + vel_strength * 0.35 + accel_bonus + data_factor * 0.15,
        0.0, 1.0,
    ))

    return SelectiveSignal(
        direction=reversal_dir,
        reversal_signal=reversal_signal,
        velocity=velocity,
        acceleration=acceleration,
        confidence=confidence,
        up_price=up_price,
        down_price=down_price,
    )


# ── Late-entry strategy (unchanged) ──────────────────────────────────────

def compute_signal(prices: list[float], volumes: list[float] | None = None) -> Signal | None:
    """Late-entry signal from recent 1-second close prices. Needs >= 30 ticks."""
    if len(prices) < 30:
        log.debug("Not enough price data (%d ticks, need 30)", len(prices))
        return None

    arr = np.array(prices, dtype=np.float64)

    rsi = _rsi(arr, 14)
    rsi_signal = np.clip((rsi - 50.0) / 20.0, -1.0, 1.0)

    ema9 = _ema(arr, 9)
    ema21 = _ema(arr, 21)
    ema9_now, ema21_now = ema9[-1], ema21[-1]
    ema_diff = (ema9_now - ema21_now) / ema21_now * 1000
    ema_signal = np.clip(ema_diff, -1.0, 1.0)

    recent_avg = np.mean(arr[-10:])
    longer_avg = np.mean(arr[-30:])
    momentum = (recent_avg - longer_avg) / longer_avg * 1000
    momentum_signal = np.clip(momentum, -1.0, 1.0)

    vol_signal = 0.0
    if volumes and len(volumes) >= 10:
        varr = np.array(volumes[-30:], dtype=np.float64)
        recent_vol = np.mean(varr[-10:])
        avg_vol = np.mean(varr)
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            vol_signal = np.clip((vol_ratio - 1.0) * 0.5, -0.5, 0.5)

    score = (
        0.25 * rsi_signal
        + 0.30 * ema_signal
        + 0.35 * momentum_signal
        + 0.10 * vol_signal
    )
    score = float(np.clip(score, -1.0, 1.0))

    signals = [rsi_signal, ema_signal, momentum_signal]
    signs = [1 if s > 0 else -1 if s < 0 else 0 for s in signals]
    agreement = abs(sum(signs)) / len(signs)
    magnitude = np.mean([abs(s) for s in signals])
    confidence = float(np.clip(agreement * 0.6 + magnitude * 0.4, 0.0, 1.0))

    return Signal(
        score=score, rsi=rsi, ema_fast=ema9_now, ema_slow=ema21_now,
        momentum=float(momentum), confidence=confidence,
    )
