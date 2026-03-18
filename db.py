"""SQLite database for all bot data collection — windows, market snapshots, trades, sessions."""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "bot_data.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ts INTEGER NOT NULL UNIQUE,
    slug TEXT NOT NULL,
    observed_at TEXT NOT NULL,

    -- BTC price data at observation time
    btc_price_at_entry REAL,

    -- Resolution data (filled after window closes)
    btc_open REAL,
    btc_close REAL,
    btc_delta REAL,
    btc_pct_change REAL,
    winner TEXT,
    resolved_at TEXT,

    -- Prior window context
    prior_window_ts INTEGER,
    prior_btc_delta REAL,
    prior_btc_pct_change REAL,
    prior_winner TEXT,
    momentum_continuation INTEGER
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ts INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    seconds_into_window REAL,

    -- Polymarket token prices
    up_price REAL,
    down_price REAL,
    up_plus_down REAL,
    spread REAL,

    -- LMSR dynamics
    price_velocity REAL,
    price_acceleration REAL,

    -- Binance data at this moment
    btc_price REAL,
    btc_1s_return REAL,
    ob_bid_volume REAL,
    ob_ask_volume REAL,
    ob_imbalance REAL,

    FOREIGN KEY (window_ts) REFERENCES windows(window_ts)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ts INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,

    -- Entry
    entry_time TEXT NOT NULL,
    seconds_into_window REAL,
    buy_price REAL,
    shares REAL,
    amount REAL,

    -- Signal data
    signal_score REAL,
    signal_confidence REAL,
    model_prob REAL,
    edge_pct REAL,

    -- LMSR data at entry
    up_price_at_entry REAL,
    down_price_at_entry REAL,
    price_velocity_at_entry REAL,

    -- Execution
    order_type TEXT,
    order_id TEXT,
    dry_run INTEGER DEFAULT 1,

    -- Outcome
    outcome TEXT,
    pnl REAL,

    FOREIGN KEY (window_ts) REFERENCES windows(window_ts)
);

CREATE TABLE IF NOT EXISTS session_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    strategy TEXT,
    dry_run INTEGER,
    windows_processed INTEGER DEFAULT 0,
    trades_placed INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_window ON market_snapshots(window_ts);
CREATE INDEX IF NOT EXISTS idx_trades_window ON trades(window_ts);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
CREATE INDEX IF NOT EXISTS idx_windows_resolved ON windows(resolved_at);

-- SMC tables (Step 7)

CREATE TABLE IF NOT EXISTS smc_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ts INTEGER NOT NULL,
    decision_time TEXT NOT NULL,
    candle_timestamp REAL,

    -- Decision result
    direction TEXT,
    confidence REAL,
    bet_size_pct REAL,
    is_skip INTEGER DEFAULT 0,

    -- Composite scores
    momentum_score REAL,
    structure_score REAL,
    confluence_score REAL,
    total_score REAL,

    -- Momentum sub-scores
    lmsr_velocity_score REAL,
    bos_type_score REAL,
    order_flow_score REAL,
    multi_tf_score REAL,

    -- Structure sub-scores
    control_state_score REAL,
    zone_position_score REAL,
    swing_strength_score REAL,
    return_type_score REAL,
    zone_quality_score REAL,

    -- Confluence sub-scores
    sweep_score REAL,
    sd_flip_score REAL,
    qm_score REAL,
    fvg_score REAL,
    engulfing_score REAL,

    -- Context
    trend_1m TEXT,
    trend_5m TEXT,
    control_state TEXT,
    return_type TEXT,
    nearest_zone_type TEXT,
    nearest_zone_position TEXT,
    nearest_zone_quality INTEGER,
    lmsr_velocity_raw REAL,
    order_flow_count_bull INTEGER,
    order_flow_count_bear INTEGER,
    has_sweep INTEGER DEFAULT 0,
    has_fvg_fill INTEGER DEFAULT 0,
    has_sd_flip INTEGER DEFAULT 0,
    has_qm INTEGER DEFAULT 0,
    has_engulfing INTEGER DEFAULT 0,

    -- Reasons (JSON list)
    reasons_json TEXT,

    -- Veto info
    was_vetoed INTEGER DEFAULT 0,
    veto_reason TEXT
);

CREATE TABLE IF NOT EXISTS smc_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ts INTEGER NOT NULL,
    timeframe TEXT NOT NULL,
    candle_timestamp REAL NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_time TEXT NOT NULL,
    description TEXT,
    config_json TEXT,

    -- Data range
    candle_count INTEGER,
    start_timestamp REAL,
    end_timestamp REAL,

    -- Results
    total_decisions INTEGER DEFAULT 0,
    total_bets INTEGER DEFAULT 0,
    total_skips INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate REAL,
    total_pnl REAL DEFAULT 0.0,
    avg_confidence REAL,
    avg_bet_size REAL,

    -- Veto stats
    vetoed_by_control INTEGER DEFAULT 0,
    vetoed_by_return_type INTEGER DEFAULT 0,
    vetoed_by_no_bos INTEGER DEFAULT 0,
    vetoed_by_ranging INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_smc_decisions_window ON smc_decisions(window_ts);
CREATE INDEX IF NOT EXISTS idx_smc_candles_window ON smc_candles(window_ts, timeframe);
CREATE INDEX IF NOT EXISTS idx_smc_decisions_direction ON smc_decisions(direction);
"""


@dataclass
class SessionSummary:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    pending: int = 0
    skipped: int = 0
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    windows_observed: int = 0
    windows_resolved: int = 0
    windows_traded: int = 0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0.0

    @property
    def roi(self) -> float:
        return self.total_pnl / self.total_wagered if self.total_wagered > 0 else 0.0


@dataclass
class MomentumStats:
    total_resolved: int = 0
    with_prior: int = 0
    continuations: int = 0
    reversals: int = 0

    @property
    def continuation_rate(self) -> float:
        return self.continuations / self.with_prior if self.with_prior > 0 else 0.0


@dataclass
class LMSRVelocityStats:
    total_snapshots: int = 0
    windows_with_snapshots: int = 0
    avg_velocity: float = 0.0
    high_velocity_windows: int = 0
    high_vel_correct: int = 0  # velocity direction matched winner
    low_velocity_windows: int = 0

    @property
    def high_vel_accuracy(self) -> float:
        return self.high_vel_correct / self.high_velocity_windows if self.high_velocity_windows > 0 else 0.0


class BotDatabase:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._session_id: Optional[int] = None

    def _create_tables(self):
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    # ── Session ──────────────────────────────────────────────────────────

    def start_session(self, strategy: str, dry_run: bool) -> int:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.execute(
            "INSERT INTO session_log (started_at, strategy, dry_run) VALUES (?, ?, ?)",
            (now, strategy, int(dry_run)),
        )
        self.conn.commit()
        self._session_id = cur.lastrowid
        return self._session_id

    def end_session(self):
        if self._session_id is None:
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        summary = self.get_session_summary()
        self.conn.execute(
            """UPDATE session_log SET ended_at=?, windows_processed=?,
               trades_placed=?, wins=?, losses=?, total_pnl=?
               WHERE id=?""",
            (now, summary.windows_observed, summary.total_trades,
             summary.wins, summary.losses, summary.total_pnl, self._session_id),
        )
        self.conn.commit()

    # ── Windows ──────────────────────────────────────────────────────────

    def record_window(self, window_ts: int, slug: str, btc_price: Optional[float] = None):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.conn.execute(
                """INSERT INTO windows (window_ts, slug, observed_at, btc_price_at_entry)
                   VALUES (?, ?, ?, ?)""",
                (window_ts, slug, now, btc_price),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Already recorded this window
            pass

    def resolve_window(self, window_ts: int, btc_open: float, btc_close: float):
        """Resolve a window with Binance kline data. Updates windows + trades."""
        delta = btc_close - btc_open
        pct_change = (delta / btc_open * 100) if btc_open > 0 else 0.0
        winner = "UP" if btc_close >= btc_open else "DOWN"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Check if already resolved
        row = self.conn.execute(
            "SELECT resolved_at FROM windows WHERE window_ts=?", (window_ts,)
        ).fetchone()
        if row and row["resolved_at"]:
            return []

        # Update the window
        self.conn.execute(
            """UPDATE windows SET btc_open=?, btc_close=?, btc_delta=?,
               btc_pct_change=?, winner=?, resolved_at=?
               WHERE window_ts=?""",
            (btc_open, btc_close, delta, pct_change, winner, now, window_ts),
        )

        # Compute momentum continuation from prior resolved window
        prior = self.conn.execute(
            """SELECT window_ts, btc_delta, btc_pct_change, winner
               FROM windows WHERE window_ts < ? AND resolved_at IS NOT NULL
               ORDER BY window_ts DESC LIMIT 1""",
            (window_ts,),
        ).fetchone()

        if prior:
            continuation = 1 if prior["winner"] == winner else 0
            self.conn.execute(
                """UPDATE windows SET prior_window_ts=?, prior_btc_delta=?,
                   prior_btc_pct_change=?, prior_winner=?, momentum_continuation=?
                   WHERE window_ts=?""",
                (prior["window_ts"], prior["btc_delta"],
                 prior["btc_pct_change"], prior["winner"], continuation, window_ts),
            )

        # Resolve PENDING trades for this window
        resolved_trades = []
        pending = self.conn.execute(
            "SELECT id, side, buy_price, shares FROM trades WHERE window_ts=? AND outcome='PENDING'",
            (window_ts,),
        ).fetchall()

        for trade in pending:
            bought_up = trade["side"] == "UP"
            correct = (winner == "UP") == bought_up
            shares = trade["shares"] or 0
            buy_price = trade["buy_price"] or 0

            if correct:
                pnl = (1.0 - buy_price) * shares
                outcome = "WIN"
            else:
                pnl = -buy_price * shares
                outcome = "LOSS"

            self.conn.execute(
                "UPDATE trades SET outcome=?, pnl=? WHERE id=?",
                (outcome, pnl, trade["id"]),
            )
            resolved_trades.append({
                "slug": f"btc-updown-5m-{window_ts}",
                "side": trade["side"],
                "outcome": outcome,
                "pnl": pnl,
                "btc_open": btc_open,
                "btc_close": btc_close,
            })

        self.conn.commit()
        return resolved_trades

    # ── Market snapshots ─────────────────────────────────────────────────

    def record_market_snapshot(
        self,
        window_ts: int,
        up_price: float,
        down_price: float,
        btc_price: Optional[float] = None,
        btc_1s_return: Optional[float] = None,
        ob_bid_volume: Optional[float] = None,
        ob_ask_volume: Optional[float] = None,
        ob_imbalance: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        seconds_into = time.time() - window_ts
        up_plus_down = up_price + down_price
        spread = abs(up_price - down_price)

        self.conn.execute(
            """INSERT INTO market_snapshots
               (window_ts, captured_at, seconds_into_window,
                up_price, down_price, up_plus_down, spread,
                price_velocity, price_acceleration,
                btc_price, btc_1s_return,
                ob_bid_volume, ob_ask_volume, ob_imbalance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (window_ts, now, seconds_into,
             up_price, down_price, up_plus_down, spread,
             velocity, acceleration,
             btc_price, btc_1s_return,
             ob_bid_volume, ob_ask_volume, ob_imbalance),
        )
        self.conn.commit()

    def get_recent_snapshots(self, window_ts: int, limit: int = 50) -> list[dict]:
        """Get recent market snapshots for a window, ordered by time."""
        rows = self.conn.execute(
            """SELECT * FROM market_snapshots WHERE window_ts=?
               ORDER BY id ASC LIMIT ?""",
            (window_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Trades ───────────────────────────────────────────────────────────

    def record_trade(
        self,
        window_ts: int,
        strategy: str,
        side: str,
        buy_price: float,
        shares: float,
        amount: float,
        signal_score: float = 0.0,
        signal_confidence: float = 0.0,
        model_prob: float = 0.0,
        edge_pct: float = 0.0,
        up_price_at_entry: Optional[float] = None,
        down_price_at_entry: Optional[float] = None,
        price_velocity_at_entry: Optional[float] = None,
        order_type: str = "DRY_RUN",
        order_id: Optional[str] = None,
        dry_run: bool = True,
        outcome: str = "PENDING",
    ):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        seconds_into = time.time() - window_ts

        self.conn.execute(
            """INSERT INTO trades
               (window_ts, strategy, side, entry_time, seconds_into_window,
                buy_price, shares, amount,
                signal_score, signal_confidence, model_prob, edge_pct,
                up_price_at_entry, down_price_at_entry, price_velocity_at_entry,
                order_type, order_id, dry_run, outcome, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (window_ts, strategy, side, now, seconds_into,
             buy_price, shares, amount,
             signal_score, signal_confidence, model_prob, edge_pct,
             up_price_at_entry, down_price_at_entry, price_velocity_at_entry,
             order_type, order_id, int(dry_run), outcome, 0.0),
        )
        self.conn.commit()

    # ── Queries ──────────────────────────────────────────────────────────

    def get_session_summary(self) -> SessionSummary:
        s = SessionSummary()

        row = self.conn.execute("SELECT COUNT(*) as n FROM windows").fetchone()
        s.windows_observed = row["n"]

        row = self.conn.execute(
            "SELECT COUNT(*) as n FROM windows WHERE resolved_at IS NOT NULL"
        ).fetchone()
        s.windows_resolved = row["n"]

        # Trades (exclude SKIPs)
        trades = self.conn.execute(
            "SELECT outcome, amount, pnl, order_type FROM trades"
        ).fetchall()

        for t in trades:
            if t["order_type"] == "SKIP":
                s.skipped += 1
                continue
            s.total_trades += 1
            s.total_wagered += t["amount"] or 0
            if t["outcome"] == "WIN":
                s.wins += 1
                s.total_pnl += t["pnl"] or 0
            elif t["outcome"] == "LOSS":
                s.losses += 1
                s.total_pnl += t["pnl"] or 0
            elif t["outcome"] == "PENDING":
                s.pending += 1

        s.windows_traded = s.total_trades + s.skipped
        return s

    def get_momentum_stats(self) -> MomentumStats:
        stats = MomentumStats()

        row = self.conn.execute(
            "SELECT COUNT(*) as n FROM windows WHERE resolved_at IS NOT NULL"
        ).fetchone()
        stats.total_resolved = row["n"]

        rows = self.conn.execute(
            """SELECT momentum_continuation FROM windows
               WHERE resolved_at IS NOT NULL AND momentum_continuation IS NOT NULL"""
        ).fetchall()
        stats.with_prior = len(rows)
        stats.continuations = sum(1 for r in rows if r["momentum_continuation"] == 1)
        stats.reversals = stats.with_prior - stats.continuations

        return stats

    def get_lmsr_velocity_stats(self, velocity_threshold: float = 0.005) -> LMSRVelocityStats:
        stats = LMSRVelocityStats()

        row = self.conn.execute("SELECT COUNT(*) as n FROM market_snapshots").fetchone()
        stats.total_snapshots = row["n"]

        row = self.conn.execute(
            "SELECT COUNT(DISTINCT window_ts) as n FROM market_snapshots"
        ).fetchone()
        stats.windows_with_snapshots = row["n"]

        # Average absolute velocity across all snapshots
        row = self.conn.execute(
            "SELECT AVG(ABS(price_velocity)) as avg_vel FROM market_snapshots WHERE price_velocity IS NOT NULL"
        ).fetchone()
        stats.avg_velocity = row["avg_vel"] or 0.0

        # For each window with snapshots, get the max velocity and check if direction matched winner
        window_rows = self.conn.execute(
            """SELECT ms.window_ts,
                      MAX(ms.price_velocity) as max_pos_vel,
                      MIN(ms.price_velocity) as max_neg_vel,
                      w.winner
               FROM market_snapshots ms
               JOIN windows w ON ms.window_ts = w.window_ts
               WHERE w.resolved_at IS NOT NULL AND ms.price_velocity IS NOT NULL
               GROUP BY ms.window_ts"""
        ).fetchall()

        for wr in window_rows:
            max_pos = wr["max_pos_vel"] or 0
            max_neg = wr["max_neg_vel"] or 0
            winner = wr["winner"]

            # Use whichever direction had stronger velocity
            if abs(max_pos) >= abs(max_neg):
                dominant_vel = max_pos
            else:
                dominant_vel = max_neg

            if abs(dominant_vel) >= velocity_threshold:
                stats.high_velocity_windows += 1
                vel_says_up = dominant_vel > 0
                if (vel_says_up and winner == "UP") or (not vel_says_up and winner == "DOWN"):
                    stats.high_vel_correct += 1
            else:
                stats.low_velocity_windows += 1

        return stats

    def get_velocity_threshold_analysis(self):
        """Analyze what velocity threshold produces the best accuracy.
        Prints a table of thresholds vs accuracy."""
        rows = self.conn.execute('''
            SELECT
                ms.window_ts,
                MAX(ABS(ms.price_velocity)) as max_velocity,
                w.winner,
                -- Use the velocity at max-abs snapshot to determine direction
                CASE
                    WHEN ABS(MAX(ms.price_velocity)) = MAX(ms.price_velocity) THEN 'UP'
                    ELSE 'DOWN'
                END as velocity_direction
            FROM market_snapshots ms
            JOIN windows w ON ms.window_ts = w.window_ts
            WHERE ms.price_velocity IS NOT NULL
              AND w.winner IS NOT NULL
              AND ms.seconds_into_window BETWEEN 10 AND 25
            GROUP BY ms.window_ts
        ''').fetchall()

        if not rows:
            print("    (No data for velocity threshold analysis)")
            return

        # Build per-window records with dominant velocity direction
        window_data = []
        for r in rows:
            wts = r["window_ts"]
            max_vel = r["max_velocity"]
            winner = r["winner"]
            # Re-query to get the actual dominant direction snapshot
            dom = self.conn.execute('''
                SELECT price_velocity FROM market_snapshots
                WHERE window_ts=? AND price_velocity IS NOT NULL
                  AND seconds_into_window BETWEEN 10 AND 25
                ORDER BY ABS(price_velocity) DESC LIMIT 1
            ''', (wts,)).fetchone()
            if dom:
                direction = "UP" if dom["price_velocity"] > 0 else "DOWN"
                window_data.append((max_vel, winner, direction))

        thresholds = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
        for thresh in thresholds:
            qualifying = [(w, d) for mv, w, d in window_data if mv >= thresh]
            if not qualifying:
                print(f"    Threshold {thresh:.3f}: 0 windows")
                continue
            correct = sum(1 for winner, direction in qualifying if winner == direction)
            total = len(qualifying)
            print(f"    Threshold {thresh:.3f}: {total} windows, {correct}/{total} correct ({correct/total*100:.0f}%)")

    def get_unresolved_window_timestamps(self) -> list[int]:
        """Get all window timestamps that haven't been resolved yet and whose windows have closed."""
        cutoff = time.time() - 305  # window must be closed + 5s buffer
        rows = self.conn.execute(
            "SELECT window_ts FROM windows WHERE resolved_at IS NULL AND window_ts < ?",
            (cutoff,),
        ).fetchall()
        return [r["window_ts"] for r in rows]

    def get_pending_trade_window_timestamps(self) -> list[int]:
        """Get window timestamps that have pending trades."""
        rows = self.conn.execute(
            "SELECT DISTINCT window_ts FROM trades WHERE outcome='PENDING'"
        ).fetchall()
        return [r["window_ts"] for r in rows]

    # ── CSV export ───────────────────────────────────────────────────────

    def export_trades_csv(self, path: str = "trades.csv"):
        """Export trades table as CSV for quick viewing."""
        import csv
        rows = self.conn.execute(
            """SELECT t.*, w.btc_open, w.btc_close, w.btc_delta, w.winner
               FROM trades t
               LEFT JOIN windows w ON t.window_ts = w.window_ts
               ORDER BY t.id"""
        ).fetchall()

        if not rows:
            return

        headers = rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(list(r))

        log.info("Exported %d trades to %s", len(rows), path)

    # ── SMC Tables (Step 7) ──────────────────────────────────────────────

    def record_smc_decision(
        self,
        window_ts: int,
        decision: 'object',
        score: 'object',
        *,
        trend_1m: Optional[str] = None,
        trend_5m: Optional[str] = None,
        control_state: Optional[str] = None,
        return_type: Optional[str] = None,
        nearest_zone_type: Optional[str] = None,
        nearest_zone_position: Optional[str] = None,
        nearest_zone_quality: Optional[int] = None,
        lmsr_velocity_raw: float = 0.0,
        order_flow_count_bull: int = 0,
        order_flow_count_bear: int = 0,
        has_sweep: bool = False,
        has_fvg_fill: bool = False,
        has_sd_flip: bool = False,
        has_qm: bool = False,
        has_engulfing: bool = False,
        was_vetoed: bool = False,
        veto_reason: str = "",
        candle_timestamp: float = 0.0,
    ) -> int:
        """Record a full SMC decision with all sub-scores and context.
        Returns the inserted row ID.
        """
        import json
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        reasons_json = json.dumps(decision.reasons)

        cur = self.conn.execute(
            """INSERT INTO smc_decisions (
                window_ts, decision_time, candle_timestamp,
                direction, confidence, bet_size_pct, is_skip,
                momentum_score, structure_score, confluence_score, total_score,
                lmsr_velocity_score, bos_type_score, order_flow_score, multi_tf_score,
                control_state_score, zone_position_score, swing_strength_score,
                return_type_score, zone_quality_score,
                sweep_score, sd_flip_score, qm_score, fvg_score, engulfing_score,
                trend_1m, trend_5m, control_state, return_type,
                nearest_zone_type, nearest_zone_position, nearest_zone_quality,
                lmsr_velocity_raw, order_flow_count_bull, order_flow_count_bear,
                has_sweep, has_fvg_fill, has_sd_flip, has_qm, has_engulfing,
                reasons_json, was_vetoed, veto_reason
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )""",
            (
                window_ts, now, candle_timestamp,
                decision.direction.value if decision.direction else None,
                decision.confidence, decision.bet_size_pct, int(decision.is_skip),
                score.momentum_score, score.structure_score, score.confluence_score,
                score.total_score,
                score.lmsr_velocity_score, score.bos_type_score, score.order_flow_score,
                score.multi_tf_score,
                score.control_state_score, score.zone_position_score,
                score.swing_strength_score, score.return_type_score, score.zone_quality_score,
                score.sweep_score, score.sd_flip_score, score.qm_score, score.fvg_score,
                score.engulfing_score,
                trend_1m, trend_5m, control_state, return_type,
                nearest_zone_type, nearest_zone_position, nearest_zone_quality,
                lmsr_velocity_raw, order_flow_count_bull, order_flow_count_bear,
                int(has_sweep), int(has_fvg_fill), int(has_sd_flip), int(has_qm),
                int(has_engulfing),
                reasons_json, int(was_vetoed), veto_reason,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_smc_candle(
        self, window_ts: int, timeframe: str, candle: 'object',
    ) -> None:
        """Record a candle to the smc_candles table for historical replay."""
        self.conn.execute(
            """INSERT INTO smc_candles (window_ts, timeframe, candle_timestamp,
               open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (window_ts, timeframe, candle.timestamp,
             candle.open, candle.high, candle.low, candle.close, candle.volume),
        )
        # Don't commit per candle — caller should batch commits.

    def record_backtest_run(self, run: dict) -> int:
        """Record a backtest run summary. Returns run ID."""
        import json
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.execute(
            """INSERT INTO backtest_runs (
                run_time, description, config_json,
                candle_count, start_timestamp, end_timestamp,
                total_decisions, total_bets, total_skips,
                wins, losses, win_rate, total_pnl,
                avg_confidence, avg_bet_size,
                vetoed_by_control, vetoed_by_return_type,
                vetoed_by_no_bos, vetoed_by_ranging
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, run.get("description", ""),
                json.dumps(run.get("config", {})),
                run.get("candle_count", 0),
                run.get("start_timestamp", 0),
                run.get("end_timestamp", 0),
                run.get("total_decisions", 0),
                run.get("total_bets", 0),
                run.get("total_skips", 0),
                run.get("wins", 0),
                run.get("losses", 0),
                run.get("win_rate", 0.0),
                run.get("total_pnl", 0.0),
                run.get("avg_confidence", 0.0),
                run.get("avg_bet_size", 0.0),
                run.get("vetoed_by_control", 0),
                run.get("vetoed_by_return_type", 0),
                run.get("vetoed_by_no_bos", 0),
                run.get("vetoed_by_ranging", 0),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_smc_decisions(self, window_ts: Optional[int] = None, limit: int = 100) -> list[dict]:
        """Query SMC decisions. Optionally filter by window_ts."""
        if window_ts is not None:
            rows = self.conn.execute(
                "SELECT * FROM smc_decisions WHERE window_ts=? ORDER BY id DESC LIMIT ?",
                (window_ts, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM smc_decisions ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_smc_candles(self, window_ts: int, timeframe: str = "1m") -> list[dict]:
        """Get stored candles for a window and timeframe."""
        rows = self.conn.execute(
            """SELECT * FROM smc_candles WHERE window_ts=? AND timeframe=?
               ORDER BY candle_timestamp ASC""",
            (window_ts, timeframe),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_backtest_runs(self, limit: int = 20) -> list[dict]:
        """Get recent backtest runs."""
        rows = self.conn.execute(
            "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_smc_decision_stats(self) -> dict:
        """Aggregate stats across all SMC decisions."""
        row = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_skip = 0 THEN 1 ELSE 0 END) as bets,
                SUM(is_skip) as skips,
                AVG(CASE WHEN is_skip = 0 THEN confidence END) as avg_confidence,
                AVG(CASE WHEN is_skip = 0 THEN total_score END) as avg_score,
                AVG(CASE WHEN is_skip = 0 THEN momentum_score END) as avg_momentum,
                AVG(CASE WHEN is_skip = 0 THEN structure_score END) as avg_structure,
                AVG(CASE WHEN is_skip = 0 THEN confluence_score END) as avg_confluence,
                SUM(was_vetoed) as total_vetoed
            FROM smc_decisions
        """).fetchone()
        return dict(row) if row else {}
