# Polymarket BTC Up/Down Trading Bot

Automated trading bot for Polymarket's 5-minute BTC binary prediction markets. Buys "Up" or "Down" tokens at T-10s before each window closes using technical analysis signals and edge detection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials
```

## Usage

```bash
# Dry run (no real orders, uses live market data)
python main.py --dry-run

# Dry run, single window only
python main.py --dry-run --once

# Live trading (requires .env credentials)
python main.py
```

## How It Works

1. **Timing**: Markets open every 5 minutes at Unix timestamps divisible by 300
2. **Data**: Binance WebSocket provides real-time BTC/USDT klines (1-second)
3. **Analysis** (T-10s to T-5s): RSI(14), EMA(9/21) crossover, price momentum, volume
4. **Edge Detection**: Compares model probability vs Polymarket token prices
5. **Execution**: FOK market order if edge > threshold; GTC limit at $0.95 as fallback
6. **Risk**: Fractional Kelly sizing, per-window and daily loss limits

## Resolution Rules

- Chainlink Data Streams oracle settles each window
- Tie rule: end price >= start price → **Up wins** (built-in Up bias)
- Settlement takes ~2 minutes after window close (64-block Polygon confirmation)

## Configuration

All settings via `.env` — see `.env.example` for the full list. Key parameters:

| Variable | Default | Description |
|---|---|---|
| `EDGE_THRESHOLD` | 0.05 | Minimum edge to trade (5%) |
| `BET_AMOUNT` | 10.0 | Max bet per window ($) |
| `MAX_DAILY_LOSS` | 50.0 | Stop trading after this loss ($) |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly position sizing |

## Trade Log

All trades (including dry runs) are logged to `trades.csv` with full signal details, market prices, edge, outcome, and P&L.
