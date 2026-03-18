# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Continuous dry run (Ctrl+C to stop, prints final summary)
python main.py --dry-run --strategy early

# Single window with resolution wait
python main.py --dry-run --once --strategy early

# Original late-entry strategy
python main.py --dry-run --strategy late

# LMSR price velocity strategy
python main.py --dry-run --strategy lmsr

# Selective strategy (reversal + LMSR confluence)
python main.py --dry-run --strategy selective

# Run N windows then show analysis
python main.py --dry-run --strategy selective --max-windows 30

# Live trading (needs .env credentials)
python main.py
```

No test suite. Validate changes with `python main.py --dry-run --once`.

## Architecture

Async Python bot (asyncio + aiohttp + websockets) that trades Polymarket's 5-minute BTC Up/Down binary markets.

**Data flow**: `BinanceFeed` (WebSocket 600-tick buffer + REST) → `compute_trend_context()` + `compute_signal_early()` → `edge.compute_edge()` → `executor` → `db.py` (SQLite) + `logger.py` (thin wrapper)

### Data Storage (SQLite)

All data is stored in `bot_data.db` (SQLite). Every window observed is recorded, whether traded or not — this builds a dataset for backtesting and ML training.

**Tables:**
- **windows** — Every 5-min window observed. Resolution data (btc_open/close, winner) filled when kline fetched. Tracks momentum continuation (did winner match prior window?).
- **market_snapshots** — Polymarket price snapshots captured each TA iteration. Includes LMSR velocity/acceleration, Binance data, order book. Tracks up+down sum for arb detection.
- **trades** — All trade decisions (including SKIPs). Signal data, LMSR data at entry, execution details, outcome.
- **session_log** — Session metadata and final stats.

CSV export available via `db.export_trades_csv()` — runs automatically after resolution.

### Signal Pipeline (Early Strategy)

The signal combines four components with configurable weights:

1. **Trend (40%)**: EMA(60) vs EMA(180) crossover from full tick buffer + prior 5-min window **mean reversion** (flipped — data shows 66% reversal rate). Don't fight the EMA trend, but fade the prior window.
2. **Micro (35%)**: RSI(7), EMA(3/7) crossover, 5s/10s momentum, volume spike. Short-term within-window signals.
3. **Order book (15%)**: Binance bid/ask volume imbalance from REST depth API.
4. **Volatility (10%)**: Stddev of 1s returns. High vol → reduce signal magnitude (less confident).

The score maps to `model_prob` via confidence scaling, then `edge.py` compares against Polymarket token prices.

### LMSR Price Velocity Strategy (`--strategy lmsr`)

Uses ONLY Polymarket price dynamics — no Binance TA. Detects informed money flow by tracking how fast token prices move.

1. At T+5s: Start collecting market snapshots every 2s
2. Track `up_price` velocity ($/sec) and acceleration ($/sec²)
3. At T+15s to T+25s: If velocity > threshold AND price < $0.60, follow the smart money
4. Skip if velocity is low (nobody trading) or prices already one-sided (> $0.60)

**Key insight**: When velocity is high and acceleration is positive (move is speeding up), informed money is flowing in. This aggregates all other participants' information.

### Selective Strategy (`--strategy selective`)

Data-driven strategy combining two empirical signals that must agree (confluence):

1. **Mean reversion** (66% reversal rate): If prior window was UP, expect DOWN and vice versa. Requires prior move > 0.03% to qualify.
2. **LMSR velocity** (57% directional accuracy at high velocity): Follow the direction that smart money is pushing prices.
3. **Only trade when BOTH agree**: Prior window reversal direction must match LMSR velocity direction. Skip otherwise.

**Skip conditions** (designed to skip ~50-60% of windows):
- No prior window data (first window after launch)
- Prior window move too small (< `SELECTIVE_MIN_PRIOR_MOVE`)
- LMSR velocity too low (< `LMSR_VELOCITY_THRESHOLD`)
- No confluence (reversal and LMSR disagree)
- Price already too expensive (> `LMSR_MAX_PRICE`)

This is the highest-conviction strategy — fewer trades, higher expected win rate.

### Four Strategy Modes (`--strategy early|late|lmsr|selective`)

**Early entry (default)**: Trades at T+10s to T+30s. Composite TA signal with mean reversion from prior window. Trades ~70% of windows.

**Late entry**: Original strategy at T-10s to T-5s. RSI(14), EMA(9/21). Problem: markets are $0.99/$0.01 by then.

**LMSR**: Pure market microstructure at T+15s to T+25s. Follows Polymarket price velocity. No warmup needed.

**Selective**: Reversal + LMSR confluence at T+15s to T+25s. Highest conviction — trades only when prior window reversal AND LMSR velocity agree. Targets 55-65% win rate on ~40-50% of windows.

### Outcome Resolution

ALL observed windows are resolved (not just traded ones) to build a complete dataset. `resolve_pending_windows()` fetches Binance 5m klines for closed windows, updates both the windows and trades tables. Momentum continuation is computed by comparing each window's winner to the prior window's winner.

### Session End Analysis

At session end, four analyses are printed:
1. **Session summary** — Trades, win rate, P&L, windows observed/resolved
2. **Momentum continuation** — What % of windows continue prior direction (validates reversal signal)
3. **LMSR velocity analysis** — When velocity is high, does direction match winner?
4. **Velocity threshold analysis** — Accuracy at different velocity thresholds (0.005–0.030) to calibrate the threshold

### Modules

- **main.py** — Event loop, `_ta_loop()` (early/late), `_lmsr_loop()`, `_selective_loop()`, market snapshot collection, window recording, trade execution, session analysis printing.
- **db.py** — `BotDatabase` class with SQLite storage. Tables: windows, market_snapshots, trades, session_log. Query methods: `get_session_summary()`, `get_momentum_stats()`, `get_lmsr_velocity_stats()`, `get_velocity_threshold_analysis()`. CSV export.
- **config.py** — Signal weights, timing, risk params, LMSR thresholds, selective params (`SELECTIVE_MIN_PRIOR_MOVE`).
- **strategy.py** — `compute_trend_context()` (with mean reversion), `compute_signal_early()`, `compute_signal_lmsr()`, `compute_signal_selective()` (reversal+LMSR confluence), `compute_signal()` (late).
- **data_feed.py** — `BinanceFeed` with 600-tick deque, `get_prior_window_delta()`, `get_prices_since(ts)`, `get_order_book()`, `fetch_kline()`.
- **edge.py** — `model_prob` vs market prices → `EdgeResult`.
- **executor.py** — `get_market_prices()` (BUY price + midpoint fallback), `place_fok_order()`, `place_limit_fallback()`.
- **risk.py** — Fractional Kelly, daily/window limits, min-order rounding.
- **logger.py** — Thin wrapper over db.py. `resolve_pending_windows()` resolves all observed windows. `log_trade()` delegates to database.

## Key Domain Details

- **Window timing**: `ts = now - (now % 300)`, slug = `btc-updown-5m-{ts}`
- **Tie rule**: end >= start → Up wins (+0.005 bias in `model_prob`)
- **Gamma API `clobTokenIds`**: may be JSON string — parsed with `json.loads` fallback
- **CLOB `get_price()`**: returns `{"price": "0.52"}` dict; `{"price": "0"}` when empty → midpoint fallback
- **Order constraints**: min 5 shares, maker amount 2 decimal places
- **LMSR up+down sum**: If < 0.98, potential arb opportunity — logged in snapshots for analysis
- **Momentum reversal**: Unreliable across samples — first 30-window sample showed 66% reversal, second showed 52% (essentially random). Not statistically significant as a standalone signal.

## Key Design Decisions

- **Mean reversion is unreliable**: Two 30-window samples showed 66% and 52% reversal rates — the difference is not statistically significant and the signal is essentially noise. Do not rely on reversal as a primary signal. LMSR velocity is the strongest validated signal.
- **Trend EMA > Micro**: At 5-minute binary markets, EMA(60/180) crossover is the strongest within-window predictor. Micro TA (RSI, short EMAs) is noise without trend context. Weight allocation: 40% trend, 35% micro.
- **Cold start**: First window after launch may lack trend data (need 60+ ticks). Confidence will be lower, bot may skip. This is correct — better to skip than trade blind. LMSR/selective strategies have no warmup requirement.
- **Selective confluence underperformed**: The reversal signal's unreliability caused the confluence filter to reject good LMSR trades. Pure LMSR with a calibrated threshold is the preferred approach. Selective strategy is retained but not recommended.
- **Confidence filter**: Prevents trading when signals disagree or are weak. Reduces trade frequency by ~30% but improves win rate on remaining trades.
- **Price range gate**: Early markets start near 50/50 but move fast. By T+15s markets can reach $0.70/$0.30. Gate ensures we only enter at fair prices.
- **Resolution via Binance kline**: Uses Binance 5m candle open/close as proxy for Chainlink oracle. Accurate for backtesting; rare divergence possible.
- **Record everything**: Every window is observed and recorded, even if not traded. Market snapshots captured every TA iteration. This builds a complete dataset for backtesting, ML training, and strategy calibration.
- **LMSR velocity threshold**: Backtesting on 30 windows shows 0.012 as the sweet spot (57% win rate, best P&L/trade). Aggregate velocity analysis overstates accuracy vs actual simulated entry. Session-end threshold analysis is directional but not a substitute for entry-level backtesting.
- **Database is source of truth**: SQLite `bot_data.db` stores all data. CSV export is for convenience only.

## Common Issues

- **`round(0.005, 2)` → 0.0**: `risk.py` uses `math.ceil(amount * 100) / 100`.
- **Sync CLOB in async loop**: `get_market_prices()` wrapped in `run_in_executor()`.
- **First window skipped**: Normal on cold start — not enough ticks for trend context. Does not apply to LMSR/selective strategies.
- **Selective skips most windows**: Expected behavior. Skip rate of 50-60% means the strategy is being selective. Check skip reasons in logs — "no confluence" is the most common and correct.
- **All iterations filtered**: Either prices moved outside range (market priced in direction fast) or confidence too low (signals disagree). Both are correct skip behavior.
- **Database locked**: SQLite uses WAL mode for concurrent reads. Only one writer at a time — should not be an issue for single-process bot.
