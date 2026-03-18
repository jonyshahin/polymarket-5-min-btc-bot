"""SMC domain dataclasses and enums.

This is the data foundation for the entire SMC engine.
Every other module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Direction(Enum):
    """Market direction."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class SwingType(Enum):
    """Swing point classification in market structure."""

    HH = "higher_high"  # Bullish continuation
    HL = "higher_low"  # Bullish continuation
    LL = "lower_low"  # Bearish continuation
    LH = "lower_high"  # Bearish continuation


class SwingStrength(Enum):
    """Whether the move from this swing broke the opposing zone."""

    STRONG = "strong"
    WEAK = "weak"
    UNCLASSIFIED = "unclassified"


class BOSType(Enum):
    """Break of Structure momentum type."""

    IMPULSIVE = "impulsive"  # Large body single candle - real momentum
    CORRECTIVE = "corrective"  # Small body multi-candle - likely liquidity grab


class ZoneType(Enum):
    """Supply or demand."""

    SUPPLY = "supply"
    DEMAND = "demand"


class ZonePattern(Enum):
    """How the zone was formed - determines conviction level."""

    RBR = "rally_base_rally"  # Continuation demand (mid-zone)
    DBD = "drop_base_drop"  # Continuation supply (mid-zone)
    RBD = "rally_base_drop"  # Reversal supply (top - highest conviction)
    DBR = "drop_base_rally"  # Reversal demand (bottom - highest conviction)


class ZonePosition(Enum):
    """Location in current range - sets directional bias."""

    TOP = "top"  # Upper third - bearish bias
    MID = "mid"  # Middle - neutral, need extra confluence
    LOWER = "lower"  # Lower third - bullish bias


class ControlStateType(Enum):
    """Which side currently controls the market."""

    SUPPLY_CONTROL = "supply_in_control"
    DEMAND_CONTROL = "demand_in_control"
    NEUTRAL = "neutral"


class MarketPhase(Enum):
    """Current phase in the market cycle."""

    RANGING = "ranging"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    TRANSITION = "transition"


class ReturnType(Enum):
    """How price approaches a zone - determines trade validity."""

    ROUNDED = "rounded"  # Decelerating - VALID, best
    CORRECTIVE = "corrective"  # Choppy - VALID
    V_SHAPE = "v_shape"  # Accelerating - REJECT
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Candle:
    """OHLCV candle with computed properties."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    # -- size properties ---------------------------------------------------

    @property
    def body_size(self) -> float:
        """Absolute size of the candle body (open-to-close)."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Full high-to-low range, floored to avoid division by zero."""
        return max(self.high - self.low, 1e-10)

    @property
    def body_ratio(self) -> float:
        """Body size as a fraction of the full range."""
        return self.body_size / self.range_size

    # -- direction ---------------------------------------------------------

    @property
    def is_bullish(self) -> bool:
        """True when close is strictly above open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True when close is strictly below open."""
        return self.close < self.open

    @property
    def direction(self) -> Direction:
        """BULLISH if close >= open, else BEARISH."""
        return Direction.BULLISH if self.close >= self.open else Direction.BEARISH

    # -- wick properties ---------------------------------------------------

    @property
    def upper_wick(self) -> float:
        """Distance from the body top to the high."""
        return self.high - self.body_high

    @property
    def lower_wick(self) -> float:
        """Distance from the low to the body bottom."""
        return self.body_low - self.low

    @property
    def wick_ratio(self) -> float:
        """Combined wick length as a fraction of range."""
        return (self.upper_wick + self.lower_wick) / self.range_size

    # -- body helpers ------------------------------------------------------

    @property
    def body_high(self) -> float:
        """The higher of open and close."""
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        """The lower of open and close."""
        return min(self.open, self.close)

    # -- midpoints ---------------------------------------------------------

    @property
    def midpoint(self) -> float:
        """Midpoint of the full range (high+low)/2."""
        return (self.high + self.low) / 2

    @property
    def body_midpoint(self) -> float:
        """Midpoint of the body (open+close)/2."""
        return (self.open + self.close) / 2

    # -- pattern detection -------------------------------------------------

    def is_marubozu(self, threshold: float = 0.80) -> bool:
        """True when body fills at least *threshold* of the range."""
        return self.body_ratio >= threshold

    def is_doji(self, threshold: float = 0.10) -> bool:
        """True when body is at most *threshold* of the range."""
        return self.body_ratio <= threshold

    def engulfs(self, other: Candle) -> bool:
        """True when this candle's body fully contains *other*'s body."""
        return self.body_high >= other.body_high and self.body_low <= other.body_low


@dataclass
class SwingPoint:
    """Confirmed swing in market structure."""

    timestamp: float
    price: float
    type: SwingType
    strength: SwingStrength = SwingStrength.UNCLASSIFIED
    candle_index: int = 0

    @property
    def is_high(self) -> bool:
        """True for HH and LH swing types."""
        return self.type in (SwingType.HH, SwingType.LH)

    @property
    def is_low(self) -> bool:
        """True for HL and LL swing types."""
        return self.type in (SwingType.HL, SwingType.LL)


@dataclass
class BOS:
    """Break of Structure event."""

    timestamp: float
    price: float
    direction: Direction
    bos_type: BOSType
    is_choch: bool = False
    swing_origin: Optional[SwingPoint] = None
    candle_index: int = 0


@dataclass
class Zone:
    """Supply or demand zone."""

    timestamp: float
    high: float
    low: float
    zone_type: ZoneType
    pattern: Optional[ZonePattern] = None
    quality_score: int = 0
    is_fresh: bool = True
    is_mitigated: bool = False
    is_broken: bool = False
    position: ZonePosition = ZonePosition.MID
    creation_bos: Optional[BOS] = None
    candle_index: int = 0

    @property
    def midpoint(self) -> float:
        """Midpoint of the zone."""
        return (self.high + self.low) / 2

    @property
    def size(self) -> float:
        """Height of the zone."""
        return self.high - self.low

    def contains_price(self, price: float) -> bool:
        """True when *price* is within [low, high]."""
        return self.low <= price <= self.high

    def is_above_price(self, price: float) -> bool:
        """True when the entire zone is above *price*."""
        return self.low > price

    def is_below_price(self, price: float) -> bool:
        """True when the entire zone is below *price*."""
        return self.high < price


@dataclass
class BetDecision:
    """Final output of the decision engine."""

    direction: Optional[Direction] = None
    confidence: float = 0.0
    bet_size_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    momentum_score: float = 0.0
    structure_score: float = 0.0
    confluence_score: float = 0.0

    @property
    def should_bet(self) -> bool:
        """True when there is a direction and non-zero confidence."""
        return self.direction is not None and self.confidence > 0

    @property
    def is_skip(self) -> bool:
        """True when no direction is set (skip this window)."""
        return self.direction is None
