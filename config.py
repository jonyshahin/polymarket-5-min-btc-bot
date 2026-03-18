"""Central configuration loaded from .env with sensible defaults."""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Polymarket ---
POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER: str = os.getenv("POLYMARKET_FUNDER", "")
SIGNATURE_TYPE: int = int(os.getenv("SIGNATURE_TYPE", "1"))
CLOB_HOST: str = "https://clob.polymarket.com"
CHAIN_ID: int = 137  # Polygon

# --- Gamma API ---
GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"

# --- Strategy ---
EDGE_THRESHOLD: float = float(os.getenv("EDGE_THRESHOLD", "0.03"))
BET_AMOUNT: float = float(os.getenv("BET_AMOUNT", "10.0"))

# --- Signal weights (early strategy) ---
TREND_WEIGHT: float = float(os.getenv("TREND_WEIGHT", "0.40"))
MICRO_WEIGHT: float = float(os.getenv("MICRO_WEIGHT", "0.35"))
ORDERBOOK_WEIGHT: float = float(os.getenv("ORDERBOOK_WEIGHT", "0.15"))
VOLATILITY_WEIGHT: float = float(os.getenv("VOLATILITY_WEIGHT", "0.10"))

# --- Confidence filter ---
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.40"))

# --- Risk ---
MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "50.0"))
MAX_LOSS_PER_WINDOW: float = float(os.getenv("MAX_LOSS_PER_WINDOW", "15.0"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
MIN_ORDER_SHARES: float = 5.0  # Polymarket minimum

# --- Data feeds ---
BINANCE_WS_URL: str = os.getenv(
    "BINANCE_WS_URL",
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1s",
)
BINANCE_REST_BASE: str = "https://api.binance.com"
TREND_BUFFER_SECONDS: int = int(os.getenv("TREND_BUFFER_SECONDS", "600"))

# --- Timing: early entry (trade shortly after window opens) ---
ENTRY_WINDOW_START: int = int(os.getenv("ENTRY_WINDOW_START", "10"))   # T+10s
ENTRY_WINDOW_END: int = int(os.getenv("ENTRY_WINDOW_END", "30"))      # T+30s

# --- Timing: late entry (original, trade before window closes) ---
LEAD_TIME: int = 10   # Start analysis at T-10s
HARD_DEADLINE: int = 5 # Must decide by T-5s

# --- Shared timing ---
WINDOW_SECONDS: int = 300  # 5-minute windows
LOOP_INTERVAL: float = 2.0  # TA recalc every 2s within decision window
STALE_DATA_THRESHOLD: float = 3.0  # Max age of price data in seconds

# --- Price range gate (early strategy) ---
MIN_TOKEN_PRICE: float = float(os.getenv("MIN_TOKEN_PRICE", "0.35"))
MAX_TOKEN_PRICE: float = float(os.getenv("MAX_TOKEN_PRICE", "0.65"))

# --- LMSR strategy ---
LMSR_VELOCITY_THRESHOLD: float = float(os.getenv("LMSR_VELOCITY_THRESHOLD", "0.010"))  # $/sec (avg ~0.009, 0.015 was too aggressive)
LMSR_MIN_SNAPSHOTS: int = int(os.getenv("LMSR_MIN_SNAPSHOTS", "5"))
LMSR_SNAPSHOT_INTERVAL: float = float(os.getenv("LMSR_SNAPSHOT_INTERVAL", "2.0"))  # seconds
LMSR_ENTRY_START: int = int(os.getenv("LMSR_ENTRY_START", "15"))   # T+15s
LMSR_ENTRY_END: int = int(os.getenv("LMSR_ENTRY_END", "25"))      # T+25s
LMSR_MAX_PRICE: float = float(os.getenv("LMSR_MAX_PRICE", "0.60"))
LMSR_COLLECT_START: int = int(os.getenv("LMSR_COLLECT_START", "5"))  # Start collecting at T+5s

# --- Selective strategy ---
SELECTIVE_MIN_PRIOR_MOVE: float = float(os.getenv("SELECTIVE_MIN_PRIOR_MOVE", "0.0001"))  # 0.01% min prior window move

# --- Logging ---
TRADE_LOG_FILE: str = "trades.csv"
