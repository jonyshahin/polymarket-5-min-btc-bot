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

# --- Composite Weights ---
MOMENTUM_WEIGHT: float = 0.40
STRUCTURE_WEIGHT: float = 0.35
CONFLUENCE_WEIGHT: float = 0.25

# --- Momentum Sub-Weights ---
LMSR_VELOCITY_SUB_WEIGHT: float = 0.40
BOS_TYPE_SUB_WEIGHT: float = 0.25
ORDER_FLOW_SUB_WEIGHT: float = 0.20
MULTI_TF_SUB_WEIGHT: float = 0.15

# --- Structure Sub-Weights ---
CONTROL_STATE_SUB_WEIGHT: float = 0.25
ZONE_POSITION_SUB_WEIGHT: float = 0.20
SWING_STRENGTH_SUB_WEIGHT: float = 0.15
RETURN_TYPE_SUB_WEIGHT: float = 0.20
ZONE_QUALITY_SUB_WEIGHT: float = 0.20

# --- Confluence Sub-Weights ---
SWEEP_SIGNAL_SUB_WEIGHT: float = 0.30
SD_FLIP_SUB_WEIGHT: float = 0.20
QM_PATTERN_SUB_WEIGHT: float = 0.15
FVG_FILL_SUB_WEIGHT: float = 0.15
ENGULFING_SUB_WEIGHT: float = 0.20

# --- Scoring Thresholds ---
MIN_TOTAL_SCORE_TO_BET: float = 0.45
HIGH_CONFIDENCE_SCORE: float = 0.65
ORDER_FLOW_HEALTHY_MIN: int = 2
ORDER_FLOW_STRONG_MIN: int = 4

# --- Decision Engine ---
BET_SIZE_BASE_PCT: float = 0.05
BET_SIZE_HIGH_CONF_PCT: float = 0.10
BET_SIZE_MAX_PCT: float = 0.15

# --- Meta-rule Vetoes ---
VETO_CONTROL_OPPOSING: bool = True
VETO_CORRECTIVE_RETURN: bool = True
VETO_NO_BOS_CONFIRMATION: bool = True
VETO_RANGING_BOTH_TF: bool = True

# --- Confidence Mapping ---
CONFIDENCE_FROM_SCORE_FLOOR: float = 0.45
CONFIDENCE_FROM_SCORE_CEILING: float = 0.80
