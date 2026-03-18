"""Backtester: replays candle data through the full SMC pipeline.

Evaluates decisions against known outcomes to measure strategy performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.candle_types import (
    BetDecision,
    Candle,
    Direction,
    ReturnType,
    ScoreBreakdown,
    ZoneType,
)
from engines.market_structure import MarketStructure
from engines.zone_engine import ZoneEngine
from engines.control_state import ControlState
from engines.liquidity_engine import LiquidityEngine
from engines.confluence_engine import ConfluenceEngine
from engines.decision_engine import DecisionEngine
from engines.smc_trade_logger import SMCTradeLogger
from db import BotDatabase

# Assumed buy price for backtesting PnL calculation.
_ASSUMED_BUY_PRICE = 0.50


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_candles: int = 0
    total_decisions: int = 0
    total_bets: int = 0
    total_skips: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    decisions: list[dict] = field(default_factory=list)

    # Veto breakdown.
    vetoed_by_control: int = 0
    vetoed_by_return_type: int = 0
    vetoed_by_no_bos: int = 0
    vetoed_by_ranging: int = 0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        bets = [d for d in self.decisions if d.get("confidence", 0) > 0]
        return sum(d["confidence"] for d in bets) / len(bets) if bets else 0.0

    @property
    def avg_bet_size(self) -> float:
        bets = [d for d in self.decisions if d.get("bet_size_pct", 0) > 0]
        return (
            sum(d["bet_size_pct"] for d in bets) / len(bets) if bets else 0.0
        )

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"Backtest: {self.total_candles} candles, "
            f"{self.total_decisions} windows\n"
            f"Bets: {self.total_bets} | Skips: {self.total_skips}\n"
            f"Wins: {self.wins} | Losses: {self.losses} | "
            f"Win Rate: {self.win_rate:.1%}\n"
            f"P&L: {self.total_pnl:+.4f} | "
            f"Avg Confidence: {self.avg_confidence:.2f} | "
            f"Avg Bet Size: {self.avg_bet_size:.3f}\n"
            f"Vetoes: control={self.vetoed_by_control}, "
            f"return_type={self.vetoed_by_return_type}, "
            f"no_bos={self.vetoed_by_no_bos}, "
            f"ranging={self.vetoed_by_ranging}"
        )


class Backtester:
    """Replays candle data through the full SMC pipeline.

    Usage:
        bt = Backtester()
        result = bt.run(candles_1m, lmsr_velocity_per_window={...})
        print(result.summary())

        # Or with database logging:
        bt = Backtester(db=BotDatabase(":memory:"))
        result = bt.run(candles_1m, ...)
    """

    def __init__(self, db: Optional[BotDatabase] = None) -> None:
        self._db = db

    def run(
        self,
        candles_1m: list[Candle],
        *,
        lmsr_velocity_per_window: Optional[dict[int, float]] = None,
        window_outcomes: Optional[dict[int, str]] = None,
        description: str = "",
    ) -> BacktestResult:
        """Run a backtest over 1-minute candles.

        Args:
            candles_1m: List of 1-minute Candle objects, sorted by timestamp.
            lmsr_velocity_per_window: Optional dict mapping window_ts to LMSR
                velocity float. If not provided, LMSR score is 0 (neutral).
            window_outcomes: Optional dict mapping window_ts to "UP" or "DOWN".
                Used to compute wins/losses/PnL.
            description: Optional description for the backtest run.
        """
        if lmsr_velocity_per_window is None:
            lmsr_velocity_per_window = {}
        if window_outcomes is None:
            window_outcomes = {}

        # Create fresh engine instances.
        ms_1m = MarketStructure()
        ms_5m = MarketStructure()
        zone_eng = ZoneEngine()
        control = ControlState()
        liq_eng = LiquidityEngine()
        conf_eng = ConfluenceEngine()
        decision_eng = DecisionEngine()

        # Logger (optional).
        logger: Optional[SMCTradeLogger] = None
        if self._db is not None:
            logger = SMCTradeLogger(self._db)

        result = BacktestResult(total_candles=len(candles_1m))

        # 5m candle accumulator.
        current_5m_batch: list[Candle] = []
        candles_buffer_1m: list[Candle] = []

        # Track which windows we've already decided on.
        decided_windows: set[int] = set()

        for candle in candles_1m:
            candles_buffer_1m.append(candle)

            # 1. Feed to MarketStructure (1m).
            bos_1m = ms_1m.update(candle)

            # 2. Feed to ZoneEngine.
            zone_eng.update(candle, bos_1m, candles_buffer_1m)

            # 3. Feed to ControlState.
            control.update(candle, zone_eng.get_active_zones())

            # 4. Feed to LiquidityEngine.
            liq_eng.update(
                candle,
                ms_1m.get_swing_highs(),
                ms_1m.get_swing_lows(),
            )

            # 5. Feed to ConfluenceEngine.
            conf_eng.update(
                candle,
                candles_buffer_1m,
                ms_1m.get_swings(),
                zone_eng.get_active_zones(),
                bos_1m,
            )

            # 6. Build 5m candles.
            current_5m_batch.append(candle)
            completed_5m: Optional[Candle] = None

            window_ts = self._get_window_ts(candle.timestamp)
            next_ts = candle.timestamp + 60
            next_window = self._get_window_ts(next_ts)

            # Complete 5m candle when we cross a 300s boundary or have 5 candles.
            if next_window != window_ts and current_5m_batch:
                completed_5m = self._build_5m_candle(current_5m_batch)
                ms_5m.update(completed_5m)
                current_5m_batch = []
            elif len(current_5m_batch) >= 5:
                completed_5m = self._build_5m_candle(current_5m_batch)
                ms_5m.update(completed_5m)
                current_5m_batch = []

            # 7. Make a decision at the end of each 5m window.
            if completed_5m is not None and window_ts not in decided_windows:
                decided_windows.add(window_ts)

                lmsr_vel = lmsr_velocity_per_window.get(window_ts, 0.0)
                current_price = candle.close

                # Update zone positions.
                all_swings = ms_1m.get_swings()
                if len(all_swings) >= 2:
                    highs = [s.price for s in ms_1m.get_swing_highs()]
                    lows = [s.price for s in ms_1m.get_swing_lows()]
                    if highs and lows:
                        liq_eng.update_range(max(highs), min(lows))
                        zone_eng.update_positions(max(highs), min(lows))

                # Gather confluence flags for both directions.
                engulfing_dir = conf_eng.check_strong_engulfing(
                    candles_buffer_1m
                )
                return_type = conf_eng.classify_return_type(candles_buffer_1m)

                nearest_demand = zone_eng.get_nearest_zone(
                    current_price, ZoneType.DEMAND,
                )
                nearest_supply = zone_eng.get_nearest_zone(
                    current_price, ZoneType.SUPPLY,
                )

                sweep = liq_eng.get_sweep_within(5, candle.timestamp)

                decision = decision_eng.decide(
                    lmsr_velocity_signal=lmsr_vel,
                    latest_bos_1m=ms_1m.get_latest_bos(),
                    trend_1m=ms_1m.get_trend(),
                    order_flow_count_bullish=ms_1m.get_order_flow_count(
                        Direction.BULLISH
                    ),
                    order_flow_count_bearish=ms_1m.get_order_flow_count(
                        Direction.BEARISH
                    ),
                    latest_bos_5m=ms_5m.get_latest_bos(),
                    trend_5m=ms_5m.get_trend(),
                    control_state=control.get_state(),
                    nearest_demand=nearest_demand,
                    nearest_supply=nearest_supply,
                    recent_sweep=sweep,
                    return_type=return_type,
                    has_fvg_fill_bullish=conf_eng.has_recent_fvg_fill(
                        Direction.BULLISH
                    ),
                    has_fvg_fill_bearish=conf_eng.has_recent_fvg_fill(
                        Direction.BEARISH
                    ),
                    has_sd_flip_bullish=any(
                        f.direction == Direction.BULLISH
                        for f in conf_eng.get_recent_flips()
                    ),
                    has_sd_flip_bearish=any(
                        f.direction == Direction.BEARISH
                        for f in conf_eng.get_recent_flips()
                    ),
                    has_qm_bullish=(
                        conf_eng.get_recent_qm() is not None
                        and conf_eng.get_recent_qm().direction == Direction.BULLISH
                    ),
                    has_qm_bearish=(
                        conf_eng.get_recent_qm() is not None
                        and conf_eng.get_recent_qm().direction == Direction.BEARISH
                    ),
                    engulfing_direction=engulfing_dir,
                    timestamp=candle.timestamp,
                )

                # Build decision record.
                scoring = decision_eng.get_scoring_engine()
                # Score for the chosen direction (or bullish as default for skips).
                score_dir = decision.direction or Direction.BULLISH
                score_breakdown = scoring.score(
                    score_dir,
                    lmsr_velocity_signal=lmsr_vel,
                    latest_bos=ms_1m.get_latest_bos(),
                    trend_1m=ms_1m.get_trend(),
                    trend_5m=ms_5m.get_trend(),
                    order_flow_count=(
                        ms_1m.get_order_flow_count(Direction.BULLISH)
                        if score_dir == Direction.BULLISH
                        else ms_1m.get_order_flow_count(Direction.BEARISH)
                    ),
                    control_state=control.get_state(),
                    nearest_zone=(
                        nearest_demand
                        if score_dir == Direction.BULLISH
                        else nearest_supply
                    ),
                    recent_sweep=sweep,
                    return_type=return_type,
                    has_fvg_fill=(
                        conf_eng.has_recent_fvg_fill(Direction.BULLISH)
                        if score_dir == Direction.BULLISH
                        else conf_eng.has_recent_fvg_fill(Direction.BEARISH)
                    ),
                    has_sd_flip=any(
                        f.direction == score_dir
                        for f in conf_eng.get_recent_flips()
                    ),
                    has_qm=(
                        conf_eng.get_recent_qm() is not None
                        and conf_eng.get_recent_qm().direction == score_dir
                    ),
                    engulfing_direction=engulfing_dir,
                    timestamp=candle.timestamp,
                )

                d_record: dict = {
                    "window_ts": window_ts,
                    "direction": (
                        decision.direction.value if decision.direction else None
                    ),
                    "confidence": decision.confidence,
                    "bet_size_pct": decision.bet_size_pct,
                    "is_skip": decision.is_skip,
                    "reasons": decision.reasons,
                    "momentum_score": score_breakdown.momentum_score,
                    "structure_score": score_breakdown.structure_score,
                    "confluence_score": score_breakdown.confluence_score,
                    "total_score": score_breakdown.total_score,
                }

                # Count vetoes.
                for reason in decision.reasons:
                    if "control state" in reason:
                        result.vetoed_by_control += 1
                    elif "corrective return" in reason:
                        result.vetoed_by_return_type += 1
                    elif "no BOS" in reason:
                        result.vetoed_by_no_bos += 1
                    elif "ranging" in reason:
                        result.vetoed_by_ranging += 1

                # Resolve outcome.
                if decision.should_bet:
                    result.total_bets += 1
                    outcome_str = window_outcomes.get(window_ts)
                    if outcome_str is not None:
                        resolved, pnl = self._resolve_outcome(
                            decision, outcome_str
                        )
                        d_record["outcome"] = resolved
                        d_record["pnl"] = pnl
                        if resolved == "WIN":
                            result.wins += 1
                        else:
                            result.losses += 1
                        result.total_pnl += pnl
                else:
                    result.total_skips += 1

                result.total_decisions += 1
                result.decisions.append(d_record)

                # Log to database if available.
                if logger is not None:
                    logger.log_decision(
                        window_ts,
                        decision,
                        score_breakdown,
                        market_structure_1m=ms_1m,
                        market_structure_5m=ms_5m,
                        control_machine=control,
                        zone_engine=zone_eng,
                        liquidity_engine=liq_eng,
                        confluence_engine=conf_eng,
                        lmsr_velocity_raw=lmsr_vel,
                        current_price=current_price,
                    )

        # Record backtest run in db.
        if self._db is not None:
            import json
            config_snapshot = {
                "MIN_TOTAL_SCORE_TO_BET": 0.45,
                "description": description,
            }
            timestamps = [c.timestamp for c in candles_1m]
            self._db.record_backtest_run(
                {
                    "description": description,
                    "config": config_snapshot,
                    "candle_count": len(candles_1m),
                    "start_timestamp": min(timestamps) if timestamps else 0,
                    "end_timestamp": max(timestamps) if timestamps else 0,
                    "total_decisions": result.total_decisions,
                    "total_bets": result.total_bets,
                    "total_skips": result.total_skips,
                    "wins": result.wins,
                    "losses": result.losses,
                    "win_rate": result.win_rate,
                    "total_pnl": result.total_pnl,
                    "avg_confidence": result.avg_confidence,
                    "avg_bet_size": result.avg_bet_size,
                    "vetoed_by_control": result.vetoed_by_control,
                    "vetoed_by_return_type": result.vetoed_by_return_type,
                    "vetoed_by_no_bos": result.vetoed_by_no_bos,
                    "vetoed_by_ranging": result.vetoed_by_ranging,
                }
            )

        return result

    def run_from_db(
        self,
        db: BotDatabase,
        window_ts_list: Optional[list[int]] = None,
        timeframe: str = "1m",
        description: str = "",
    ) -> BacktestResult:
        """Run backtest using candles stored in the smc_candles table.

        If window_ts_list is None, uses all available windows.
        Fetches candles from db.get_smc_candles() and calls self.run().
        """
        # Get window list.
        if window_ts_list is None:
            rows = db.conn.execute(
                "SELECT DISTINCT window_ts FROM smc_candles WHERE timeframe=? "
                "ORDER BY window_ts ASC",
                (timeframe,),
            ).fetchall()
            window_ts_list = [r["window_ts"] for r in rows]

        # Fetch all candles across windows.
        candles: list[Candle] = []
        for wts in window_ts_list:
            rows = db.get_smc_candles(wts, timeframe)
            for r in rows:
                candles.append(
                    Candle(
                        timestamp=r["candle_timestamp"],
                        open=r["open"],
                        high=r["high"],
                        low=r["low"],
                        close=r["close"],
                        volume=r["volume"] or 0.0,
                    )
                )

        # Sort by timestamp.
        candles.sort(key=lambda c: c.timestamp)

        # Fetch LMSR velocity from market_snapshots.
        lmsr_vel: dict[int, float] = {}
        for wts in window_ts_list:
            snapshots = db.get_recent_snapshots(wts)
            if snapshots:
                velocities = [
                    s["price_velocity"]
                    for s in snapshots
                    if s["price_velocity"] is not None
                ]
                if velocities:
                    # Use max absolute velocity.
                    max_vel = max(velocities, key=abs)
                    lmsr_vel[wts] = max_vel

        # Fetch outcomes from windows table.
        outcomes: dict[int, str] = {}
        for wts in window_ts_list:
            row = db.conn.execute(
                "SELECT winner FROM windows WHERE window_ts=? AND winner IS NOT NULL",
                (wts,),
            ).fetchone()
            if row:
                outcomes[wts] = row["winner"]

        return self.run(
            candles,
            lmsr_velocity_per_window=lmsr_vel,
            window_outcomes=outcomes,
            description=description,
        )

    def _get_window_ts(self, candle_timestamp: float) -> int:
        """Round timestamp down to nearest 300-second boundary."""
        return int(candle_timestamp) // 300 * 300

    def _resolve_outcome(
        self,
        decision: BetDecision,
        actual_winner: str,
    ) -> tuple[str, float]:
        """Resolve a bet against the known outcome.

        Returns (outcome_str, pnl_float).
        """
        if decision.direction is None:
            return ("SKIP", 0.0)

        is_win = (
            (decision.direction == Direction.BULLISH and actual_winner == "UP")
            or (
                decision.direction == Direction.BEARISH
                and actual_winner == "DOWN"
            )
        )

        if is_win:
            pnl = (1.0 - _ASSUMED_BUY_PRICE) * decision.bet_size_pct
            return ("WIN", pnl)
        else:
            pnl = -_ASSUMED_BUY_PRICE * decision.bet_size_pct
            return ("LOSS", pnl)

    @staticmethod
    def _build_5m_candle(batch: list[Candle]) -> Candle:
        """Build a 5m candle from accumulated 1m candles."""
        return Candle(
            timestamp=batch[0].timestamp,
            open=batch[0].open,
            high=max(c.high for c in batch),
            low=min(c.low for c in batch),
            close=batch[-1].close,
            volume=sum(c.volume for c in batch),
        )
