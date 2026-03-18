"""SMC Engine Configuration for Polymarket BTC Bot.
All tunable parameters. Adjust through backtesting."""

# Candle Aggregation
CANDLE_BUFFER_1M: int = 60
CANDLE_BUFFER_5M: int = 24

# Swing Detection
SWING_LOOKBACK: int = 5

# BOS Classification
BOS_BODY_RATIO_IMPULSIVE: float = 0.70
BOS_BODY_RATIO_CORRECTIVE: float = 0.50

# Zone Detection
ZONE_FRESHNESS_MAX_AGE: int = 30
ZONE_WICK_THRESHOLD: float = 0.50

# Liquidity Detection
EQH_EQL_TOLERANCE: float = 0.0005
FVG_MIN_SIZE_PCT: float = 0.001

# V-Shape Filter
V_SHAPE_VELOCITY_MULTIPLIER: float = 2.0
V_SHAPE_LOOKBACK: int = 5

# Confidence Thresholds
CONFIDENCE_SKIP: float = 0.30
CONFIDENCE_SMALL: float = 0.50
CONFIDENCE_HIGH: float = 0.70

# Bet Sizing (fraction of bankroll)
BET_SIZE_SMALL: float = 0.005
BET_SIZE_STANDARD: float = 0.010
BET_SIZE_HIGH: float = 0.015

# Tilt / Risk Management
TILT_CONSECUTIVE_LOSSES: int = 3
TILT_DAILY_DRAWDOWN: float = 0.05
TILT_PAUSE_SECONDS: int = 1800

# Composite Score Weights
W_MOMENTUM: float = 0.40
W_STRUCTURE: float = 0.35
W_CONFLUENCE: float = 0.25

# LMSR Integration
LMSR_VELOCITY_THRESHOLD: float = 0.02
LMSR_DECISION_WINDOW_SECONDS: int = 60

# Volatility Filter
ATR_PERIOD: int = 14
ATR_SPIKE_MULTIPLIER: float = 3.0
