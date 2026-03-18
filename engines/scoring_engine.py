"""Scoring Engine: computes 3-composite SMC score from engine outputs.

Does NOT make bet decisions. Scores the current market state for a given
direction and returns a ScoreBreakdown with momentum, structure, and
confluence composites.
"""

from __future__ import annotations

from typing import Optional

from utils.candle_types import (
    BOS,
    BOSType,
    ControlStateType,
    Direction,
    MarketPhase,
    ReturnType,
    ScoreBreakdown,
    SwingStrength,
    Zone,
    ZonePosition,
    ZoneType,
)
from engines.liquidity_engine import LiquiditySweep
import smc_config as cfg


class ScoringEngine:
    """Computes 3-composite SMC score from engine outputs.

    Stateless: takes inputs and returns a ScoreBreakdown.
    """

    def __init__(self, config: object = None) -> None:
        self._cfg = config if config is not None else cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        direction: Direction,
        *,
        lmsr_velocity_signal: float = 0.0,
        latest_bos: Optional[BOS] = None,
        trend_1m: Optional[MarketPhase] = None,
        trend_5m: Optional[MarketPhase] = None,
        order_flow_count: int = 0,
        control_state: ControlStateType = ControlStateType.NEUTRAL,
        nearest_zone: Optional[Zone] = None,
        recent_sweep: Optional[LiquiditySweep] = None,
        return_type: ReturnType = ReturnType.UNKNOWN,
        has_fvg_fill: bool = False,
        has_sd_flip: bool = False,
        has_qm: bool = False,
        engulfing_direction: Optional[Direction] = None,
        timestamp: float = 0.0,
    ) -> ScoreBreakdown:
        """Compute all three composites and return full breakdown."""
        c = self._cfg
        breakdown = ScoreBreakdown(direction=direction, timestamp=timestamp)

        # --- 1. MOMENTUM COMPOSITE ---
        breakdown.lmsr_velocity_score = self._score_lmsr(lmsr_velocity_signal, direction)
        breakdown.bos_type_score = self._score_bos_type(latest_bos, direction)
        breakdown.order_flow_score = self._score_order_flow(order_flow_count)
        breakdown.multi_tf_score = self._score_multi_tf(trend_1m, trend_5m, direction)

        breakdown.momentum_score = (
            breakdown.lmsr_velocity_score * c.LMSR_VELOCITY_SUB_WEIGHT
            + breakdown.bos_type_score * c.BOS_TYPE_SUB_WEIGHT
            + breakdown.order_flow_score * c.ORDER_FLOW_SUB_WEIGHT
            + breakdown.multi_tf_score * c.MULTI_TF_SUB_WEIGHT
        )

        # --- 2. STRUCTURE COMPOSITE ---
        breakdown.control_state_score = self._score_control_state(control_state, direction)
        breakdown.zone_position_score = self._score_zone_position(nearest_zone, direction)
        breakdown.swing_strength_score = self._score_swing_strength(nearest_zone)
        breakdown.return_type_score = self._score_return_type(return_type)
        breakdown.zone_quality_score = self._score_zone_quality(nearest_zone)

        breakdown.structure_score = (
            breakdown.control_state_score * c.CONTROL_STATE_SUB_WEIGHT
            + breakdown.zone_position_score * c.ZONE_POSITION_SUB_WEIGHT
            + breakdown.swing_strength_score * c.SWING_STRENGTH_SUB_WEIGHT
            + breakdown.return_type_score * c.RETURN_TYPE_SUB_WEIGHT
            + breakdown.zone_quality_score * c.ZONE_QUALITY_SUB_WEIGHT
        )

        # --- 3. CONFLUENCE COMPOSITE ---
        breakdown.sweep_score = self._score_sweep(recent_sweep, direction)
        breakdown.sd_flip_score = self._score_sd_flip(has_sd_flip)
        breakdown.qm_score = self._score_qm(has_qm)
        breakdown.fvg_score = self._score_fvg(has_fvg_fill)
        breakdown.engulfing_score = self._score_engulfing(engulfing_direction, direction)

        breakdown.confluence_score = (
            breakdown.sweep_score * c.SWEEP_SIGNAL_SUB_WEIGHT
            + breakdown.sd_flip_score * c.SD_FLIP_SUB_WEIGHT
            + breakdown.qm_score * c.QM_PATTERN_SUB_WEIGHT
            + breakdown.fvg_score * c.FVG_FILL_SUB_WEIGHT
            + breakdown.engulfing_score * c.ENGULFING_SUB_WEIGHT
        )

        # --- WEIGHTED TOTAL ---
        breakdown.total_score = (
            breakdown.momentum_score * c.MOMENTUM_WEIGHT
            + breakdown.structure_score * c.STRUCTURE_WEIGHT
            + breakdown.confluence_score * c.CONFLUENCE_WEIGHT
        )

        breakdown.reasons = self._build_reasons(breakdown)

        return breakdown

    # ------------------------------------------------------------------
    # Sub-scoring methods (all return 0.0 to 1.0)
    # ------------------------------------------------------------------

    def _score_lmsr(self, lmsr_signal: float, direction: Direction) -> float:
        """Score LMSR velocity alignment.

        Positive signal = bullish, negative = bearish.
        If direction matches sign: return abs(signal) clamped to 0-1.
        If direction opposes sign: return 0.0.
        If signal is ~0 (no LMSR data): return 0.5 (neutral).
        """
        if abs(lmsr_signal) < 1e-9:
            return 0.5

        signal_bullish = lmsr_signal > 0
        want_bullish = direction == Direction.BULLISH

        if signal_bullish == want_bullish:
            return min(abs(lmsr_signal), 1.0)
        return 0.0

    def _score_bos_type(self, bos: Optional[BOS], direction: Direction) -> float:
        """Score based on BOS type and direction alignment.

        IMPULSIVE BOS in our direction: 1.0
        CORRECTIVE BOS in our direction: 0.3
        IMPULSIVE BOS against us: 0.0
        CORRECTIVE BOS against us: 0.6
        No BOS: 0.4
        """
        if bos is None:
            return 0.4

        aligned = bos.direction == direction

        if aligned:
            if bos.bos_type == BOSType.IMPULSIVE:
                return 1.0
            return 0.3  # corrective
        else:
            if bos.bos_type == BOSType.IMPULSIVE:
                return 0.0
            return 0.6  # corrective against — weak counter, can fade

    def _score_order_flow(self, count: int) -> float:
        """Score consecutive BOS count.

        0: 0.0, 1: 0.3, 2: 0.5, 3: 0.7, 4+: 1.0
        Linear interpolation between breakpoints.
        """
        breakpoints = [
            (0, 0.0),
            (1, 0.3),
            (self._cfg.ORDER_FLOW_HEALTHY_MIN, 0.5),
            (3, 0.7),
            (self._cfg.ORDER_FLOW_STRONG_MIN, 1.0),
        ]

        if count <= 0:
            return 0.0
        if count >= self._cfg.ORDER_FLOW_STRONG_MIN:
            return 1.0

        # Find the two surrounding breakpoints and interpolate.
        for i in range(len(breakpoints) - 1):
            x0, y0 = breakpoints[i]
            x1, y1 = breakpoints[i + 1]
            if x0 <= count <= x1:
                if x1 == x0:
                    return y1
                t = (count - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return 1.0

    def _score_multi_tf(
        self,
        trend_1m: Optional[MarketPhase],
        trend_5m: Optional[MarketPhase],
        direction: Direction,
    ) -> float:
        """Score multi-timeframe trend alignment.

        Both agree with direction: 1.0
        One agrees, one RANGING: 0.6
        One agrees, one opposes: 0.2
        Both oppose: 0.0
        Both RANGING: 0.3
        Either is None: 0.5 (no data, neutral)
        """
        if trend_1m is None or trend_5m is None:
            return 0.5

        align_1m = self._trend_aligns(trend_1m, direction)
        align_5m = self._trend_aligns(trend_5m, direction)
        range_1m = self._is_ranging(trend_1m)
        range_5m = self._is_ranging(trend_5m)

        if align_1m and align_5m:
            return 1.0
        if (align_1m and range_5m) or (align_5m and range_1m):
            return 0.6
        if range_1m and range_5m:
            return 0.3
        if (align_1m and not range_5m) or (align_5m and not range_1m):
            # One agrees, one opposes
            return 0.2
        # Both oppose
        return 0.0

    def _score_control_state(
        self, state: ControlStateType, direction: Direction,
    ) -> float:
        """Score control state alignment.

        DEMAND_CONTROL + BULLISH: 1.0
        SUPPLY_CONTROL + BEARISH: 1.0
        NEUTRAL: 0.5
        Opposing: 0.1
        """
        if state == ControlStateType.NEUTRAL:
            return 0.5

        aligned = (
            (state == ControlStateType.DEMAND_CONTROL and direction == Direction.BULLISH)
            or (state == ControlStateType.SUPPLY_CONTROL and direction == Direction.BEARISH)
        )
        return 1.0 if aligned else 0.1

    def _score_zone_position(
        self, zone: Optional[Zone], direction: Direction,
    ) -> float:
        """Score zone position favorability (premium/discount).

        For BULLISH: LOWER demand=1.0, MID demand=0.6, TOP demand=0.3
        For BEARISH: TOP supply=1.0, MID supply=0.6, LOWER supply=0.3
        No zone: 0.4
        Zone type doesn't match direction: 0.2
        """
        if zone is None:
            return 0.4

        # Check if zone type matches direction.
        if direction == Direction.BULLISH and zone.zone_type != ZoneType.DEMAND:
            return 0.2
        if direction == Direction.BEARISH and zone.zone_type != ZoneType.SUPPLY:
            return 0.2

        position_scores = {
            ZonePosition.LOWER: 1.0,
            ZonePosition.MID: 0.6,
            ZonePosition.TOP: 0.3,
        }

        if direction == Direction.BULLISH:
            return position_scores.get(zone.position, 0.4)
        else:
            # For BEARISH: TOP is best, LOWER is worst.
            bearish_scores = {
                ZonePosition.TOP: 1.0,
                ZonePosition.MID: 0.6,
                ZonePosition.LOWER: 0.3,
            }
            return bearish_scores.get(zone.position, 0.4)

    def _score_swing_strength(self, zone: Optional[Zone]) -> float:
        """Score based on swing strength of zone's creation BOS.

        STRONG: 1.0, MODERATE: 0.6, WEAK: 0.3, UNCLASSIFIED: 0.4
        No zone or no creation_bos: 0.4
        """
        if zone is None or zone.creation_bos is None:
            return 0.4
        origin = zone.creation_bos.swing_origin
        if origin is None:
            return 0.4

        strength_scores = {
            SwingStrength.STRONG: 1.0,
            SwingStrength.MODERATE: 0.6,
            SwingStrength.WEAK: 0.3,
            SwingStrength.UNCLASSIFIED: 0.4,
        }
        return strength_scores.get(origin.strength, 0.4)

    def _score_return_type(self, rt: ReturnType) -> float:
        """Score return type (Book 12).

        V_SHAPE: 1.0, ROUNDED: 0.5, CORRECTIVE: 0.2, UNKNOWN: 0.4
        """
        scores = {
            ReturnType.V_SHAPE: 1.0,
            ReturnType.ROUNDED: 0.5,
            ReturnType.CORRECTIVE: 0.2,
            ReturnType.UNKNOWN: 0.4,
        }
        return scores.get(rt, 0.4)

    def _score_zone_quality(self, zone: Optional[Zone]) -> float:
        """Score zone quality (0-3).

        Quality 3: 1.0, 2: 0.7, 1: 0.4, 0: 0.2. No zone: 0.3.
        """
        if zone is None:
            return 0.3

        quality_scores = {3: 1.0, 2: 0.7, 1: 0.4, 0: 0.2}
        return quality_scores.get(zone.quality_score, 0.2)

    def _score_sweep(
        self, sweep: Optional[LiquiditySweep], direction: Direction,
    ) -> float:
        """Score liquidity sweep alignment.

        Aligned sweep: 0.8 base, 1.0 if external.
        Opposing sweep: 0.1.
        No sweep: 0.3.
        """
        if sweep is None:
            return 0.3

        if sweep.direction_after == direction:
            return 1.0 if sweep.is_external else 0.8
        return 0.1

    def _score_sd_flip(self, has_flip: bool) -> float:
        """Binary: 1.0 if S/D flip detected, 0.3 otherwise."""
        return 1.0 if has_flip else 0.3

    def _score_qm(self, has_qm: bool) -> float:
        """Binary: 1.0 if QM pattern detected, 0.3 otherwise."""
        return 1.0 if has_qm else 0.3

    def _score_fvg(self, has_fill: bool) -> float:
        """Binary: 1.0 if FVG fill recently occurred, 0.3 otherwise."""
        return 1.0 if has_fill else 0.3

    def _score_engulfing(
        self, engulfing_dir: Optional[Direction], direction: Direction,
    ) -> float:
        """Score engulfing pattern alignment.

        Matches direction: 1.0. No engulfing: 0.3. Opposes: 0.0.
        """
        if engulfing_dir is None:
            return 0.3
        return 1.0 if engulfing_dir == direction else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trend_aligns(self, phase: MarketPhase, direction: Direction) -> bool:
        """True if the MarketPhase aligns with the Direction."""
        if direction == Direction.BULLISH:
            return phase == MarketPhase.TRENDING_UP
        return phase == MarketPhase.TRENDING_DOWN

    def _is_ranging(self, phase: MarketPhase) -> bool:
        """True if the phase is RANGING or TRANSITION."""
        return phase in (MarketPhase.RANGING, MarketPhase.TRANSITION)

    def _build_reasons(self, b: ScoreBreakdown) -> list[str]:
        """Build human-readable list of noteworthy scoring factors."""
        reasons: list[str] = []

        # Momentum
        if b.lmsr_velocity_score > 0.7:
            reasons.append(f"LMSR velocity strongly {b.direction.value} ({b.lmsr_velocity_score:.2f})")
        elif b.lmsr_velocity_score < 0.1:
            reasons.append("LMSR velocity opposes direction")

        if b.bos_type_score >= 1.0:
            reasons.append("Impulsive BOS confirms direction")
        elif b.bos_type_score < 0.1:
            reasons.append("Impulsive BOS opposes direction")

        if b.order_flow_score >= 0.7:
            reasons.append(f"Strong order flow ({b.order_flow_score:.2f})")
        elif b.order_flow_score < 0.1:
            reasons.append("No order flow confirmation")

        if b.multi_tf_score >= 1.0:
            reasons.append("Both timeframes aligned")
        elif b.multi_tf_score < 0.1:
            reasons.append("Both timeframes oppose direction")

        # Structure
        if b.control_state_score >= 1.0:
            reasons.append("Control state confirms direction")
        elif b.control_state_score <= 0.1:
            reasons.append("Control state opposes direction")

        if b.zone_position_score >= 1.0:
            label = "discount" if b.direction == Direction.BULLISH else "premium"
            reasons.append(f"Zone at {label}")
        elif b.zone_position_score <= 0.2:
            reasons.append("Zone position unfavorable")

        if b.return_type_score >= 1.0:
            reasons.append("V-shape return — strong demand/supply")
        elif b.return_type_score <= 0.2:
            reasons.append("Return type is corrective — zone may fail")

        if b.zone_quality_score >= 1.0:
            reasons.append("High quality zone (3/3)")
        elif b.zone_quality_score <= 0.2:
            reasons.append("Low quality zone")

        # Confluence
        if b.sweep_score >= 0.8:
            reasons.append("Liquidity sweep aligned" if b.sweep_score < 1.0 else "External liquidity sweep aligned")
        elif b.sweep_score <= 0.1:
            reasons.append("Liquidity sweep opposes direction")

        if b.sd_flip_score >= 1.0:
            reasons.append("S/D flip confirmed")
        if b.qm_score >= 1.0:
            reasons.append("QM pattern detected")
        if b.fvg_score >= 1.0:
            reasons.append("FVG fill at zone")
        if b.engulfing_score >= 1.0:
            reasons.append("Strong engulfing confirms direction")
        elif b.engulfing_score < 0.1:
            reasons.append("Engulfing opposes direction")

        return reasons
