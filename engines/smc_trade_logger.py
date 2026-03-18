"""SMC Trade Logger: bridges SMC engines and the database.

Captures the full engine state snapshot when a decision is made,
and logs it to the smc_decisions table for analysis and backtesting.
"""

from __future__ import annotations

from typing import Optional

from utils.candle_types import (
    BetDecision,
    Candle,
    Direction,
    ScoreBreakdown,
    ZoneType,
)
from engines.market_structure import MarketStructure
from engines.zone_engine import ZoneEngine
from engines.control_state import ControlState
from engines.liquidity_engine import LiquidityEngine
from engines.confluence_engine import ConfluenceEngine
from db import BotDatabase


class SMCTradeLogger:
    """Logs SMC decisions and engine state to the database.

    This is the bridge between the SMC engines and the database.
    Called by the main loop after each decision cycle.
    """

    def __init__(self, db: BotDatabase) -> None:
        self._db = db

    def log_decision(
        self,
        window_ts: int,
        decision: BetDecision,
        score_breakdown: ScoreBreakdown,
        *,
        market_structure_1m: Optional[MarketStructure] = None,
        market_structure_5m: Optional[MarketStructure] = None,
        control_machine: Optional[ControlState] = None,
        zone_engine: Optional[ZoneEngine] = None,
        liquidity_engine: Optional[LiquidityEngine] = None,
        confluence_engine: Optional[ConfluenceEngine] = None,
        lmsr_velocity_raw: float = 0.0,
        current_price: float = 0.0,
    ) -> int:
        """Log a complete decision with full engine state.

        Extracts context from each engine and calls db.record_smc_decision().
        Returns the decision row ID.
        """
        # Extract trend from market structure engines.
        trend_1m: Optional[str] = None
        trend_5m: Optional[str] = None
        if market_structure_1m is not None:
            trend_1m = market_structure_1m.get_trend().value
        if market_structure_5m is not None:
            trend_5m = market_structure_5m.get_trend().value

        # Extract control state.
        control_state: Optional[str] = None
        if control_machine is not None:
            control_state = control_machine.get_state().value

        # Infer return type from score breakdown.
        return_type: Optional[str] = None
        if score_breakdown.return_type_score >= 0.9:
            return_type = "v_shape"
        elif score_breakdown.return_type_score >= 0.6:
            return_type = "rounded"
        elif score_breakdown.return_type_score >= 0.3:
            return_type = "corrective"
        else:
            return_type = "unknown"

        # Extract nearest zone info.
        nearest_zone_type: Optional[str] = None
        nearest_zone_position: Optional[str] = None
        nearest_zone_quality: Optional[int] = None
        if zone_engine is not None and decision.direction is not None:
            zt = (
                ZoneType.DEMAND
                if decision.direction == Direction.BULLISH
                else ZoneType.SUPPLY
            )
            nearest = zone_engine.get_nearest_zone(current_price, zt)
            if nearest is not None:
                nearest_zone_type = nearest.zone_type.value
                nearest_zone_position = nearest.position.value
                nearest_zone_quality = nearest.quality_score

        # Order flow counts.
        order_flow_count_bull = 0
        order_flow_count_bear = 0
        if market_structure_1m is not None:
            order_flow_count_bull = market_structure_1m.get_order_flow_count(
                Direction.BULLISH
            )
            order_flow_count_bear = market_structure_1m.get_order_flow_count(
                Direction.BEARISH
            )

        # Confluence flags.
        has_sweep = False
        if liquidity_engine is not None:
            has_sweep = len(liquidity_engine.get_recent_sweeps(1)) > 0

        has_fvg_fill = False
        has_sd_flip = False
        has_qm = False
        if confluence_engine is not None:
            if decision.direction is not None:
                has_fvg_fill = confluence_engine.has_recent_fvg_fill(
                    decision.direction
                )
            has_sd_flip = len(confluence_engine.get_recent_flips(1)) > 0
            has_qm = confluence_engine.get_recent_qm() is not None

        has_engulfing = score_breakdown.engulfing_score > 0.5

        # Infer veto info from decision.
        was_vetoed = False
        veto_reason = ""
        if decision.is_skip and decision.reasons:
            for reason in decision.reasons:
                if reason.startswith("VETO:"):
                    was_vetoed = True
                    veto_reason = reason
                    break

        return self._db.record_smc_decision(
            window_ts=window_ts,
            decision=decision,
            score=score_breakdown,
            trend_1m=trend_1m,
            trend_5m=trend_5m,
            control_state=control_state,
            return_type=return_type,
            nearest_zone_type=nearest_zone_type,
            nearest_zone_position=nearest_zone_position,
            nearest_zone_quality=nearest_zone_quality,
            lmsr_velocity_raw=lmsr_velocity_raw,
            order_flow_count_bull=order_flow_count_bull,
            order_flow_count_bear=order_flow_count_bear,
            has_sweep=has_sweep,
            has_fvg_fill=has_fvg_fill,
            has_sd_flip=has_sd_flip,
            has_qm=has_qm,
            has_engulfing=has_engulfing,
            was_vetoed=was_vetoed,
            veto_reason=veto_reason,
            candle_timestamp=score_breakdown.timestamp,
        )

    def log_candle(
        self,
        window_ts: int,
        timeframe: str,
        candle: Candle,
    ) -> None:
        """Log a candle for historical replay."""
        self._db.record_smc_candle(window_ts, timeframe, candle)

    def flush(self) -> None:
        """Commit any pending candle writes."""
        self._db.conn.commit()

    def get_decision_summary(self, last_n: int = 50) -> dict:
        """Get a summary of recent decisions for console output.

        Returns dict with:
        - total_decisions, total_bets, total_skips
        - avg_confidence, avg_total_score
        - direction_counts: {"BULLISH": n, "BEARISH": n}
        - top_veto_reasons: list of (reason, count) tuples
        - avg_momentum, avg_structure, avg_confluence
        """
        decisions = self._db.get_smc_decisions(limit=last_n)

        total = len(decisions)
        bets = [d for d in decisions if not d["is_skip"]]
        skips = [d for d in decisions if d["is_skip"]]

        # Direction counts (stored lowercase, expose uppercase).
        direction_counts: dict[str, int] = {"BULLISH": 0, "BEARISH": 0}
        for d in bets:
            dir_val = (d["direction"] or "").upper()
            if dir_val in direction_counts:
                direction_counts[dir_val] += 1

        # Veto reasons.
        veto_counts: dict[str, int] = {}
        for d in decisions:
            if d["was_vetoed"] and d["veto_reason"]:
                reason = d["veto_reason"]
                veto_counts[reason] = veto_counts.get(reason, 0) + 1
        top_veto_reasons = sorted(
            veto_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Averages for bets only.
        avg_confidence = 0.0
        avg_total_score = 0.0
        avg_momentum = 0.0
        avg_structure = 0.0
        avg_confluence = 0.0
        if bets:
            avg_confidence = sum(d["confidence"] for d in bets) / len(bets)
            avg_total_score = sum(d["total_score"] for d in bets) / len(bets)
            avg_momentum = sum(d["momentum_score"] for d in bets) / len(bets)
            avg_structure = sum(d["structure_score"] for d in bets) / len(bets)
            avg_confluence = sum(
                d["confluence_score"] for d in bets
            ) / len(bets)

        return {
            "total_decisions": total,
            "total_bets": len(bets),
            "total_skips": len(skips),
            "avg_confidence": avg_confidence,
            "avg_total_score": avg_total_score,
            "direction_counts": direction_counts,
            "top_veto_reasons": top_veto_reasons,
            "avg_momentum": avg_momentum,
            "avg_structure": avg_structure,
            "avg_confluence": avg_confluence,
        }
