"""Decision Engine: converts ScoreBreakdown into a final BetDecision.

Applies meta-rules (vetoes, overrides, bet sizing) on top of the
ScoringEngine's composite scores. This is the top-level "brain" that
produces the final trading signal.
"""

from __future__ import annotations

from typing import Optional

from utils.candle_types import (
    BOS,
    BetDecision,
    ControlStateType,
    Direction,
    MarketPhase,
    ReturnType,
    ScoreBreakdown,
    Zone,
)
from engines.liquidity_engine import LiquiditySweep
from engines.scoring_engine import ScoringEngine
import smc_config as cfg


class DecisionEngine:
    """Converts ScoreBreakdown into a final BetDecision with meta-rules.

    Pipeline: score both directions → pick higher → apply vetoes →
    map score to confidence → map confidence to bet size → BetDecision.
    """

    def __init__(self, config: object = None) -> None:
        self._cfg = config if config is not None else cfg
        self._scoring = ScoringEngine(self._cfg)
        self._veto_reasons: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        lmsr_velocity_signal: float = 0.0,
        latest_bos_1m: Optional[BOS] = None,
        trend_1m: Optional[MarketPhase] = None,
        order_flow_count_bullish: int = 0,
        order_flow_count_bearish: int = 0,
        latest_bos_5m: Optional[BOS] = None,
        trend_5m: Optional[MarketPhase] = None,
        control_state: ControlStateType = ControlStateType.NEUTRAL,
        nearest_demand: Optional[Zone] = None,
        nearest_supply: Optional[Zone] = None,
        recent_sweep: Optional[LiquiditySweep] = None,
        return_type: ReturnType = ReturnType.UNKNOWN,
        has_fvg_fill_bullish: bool = False,
        has_fvg_fill_bearish: bool = False,
        has_sd_flip_bullish: bool = False,
        has_sd_flip_bearish: bool = False,
        has_qm_bullish: bool = False,
        has_qm_bearish: bool = False,
        engulfing_direction: Optional[Direction] = None,
        timestamp: float = 0.0,
    ) -> BetDecision:
        """Full decision pipeline."""
        self._veto_reasons = []

        # Score both directions.
        bullish_score = self._scoring.score(
            Direction.BULLISH,
            lmsr_velocity_signal=lmsr_velocity_signal,
            latest_bos=latest_bos_1m,
            trend_1m=trend_1m,
            trend_5m=trend_5m,
            order_flow_count=order_flow_count_bullish,
            control_state=control_state,
            nearest_zone=nearest_demand,
            recent_sweep=recent_sweep,
            return_type=return_type,
            has_fvg_fill=has_fvg_fill_bullish,
            has_sd_flip=has_sd_flip_bullish,
            has_qm=has_qm_bullish,
            engulfing_direction=engulfing_direction,
            timestamp=timestamp,
        )

        bearish_score = self._scoring.score(
            Direction.BEARISH,
            lmsr_velocity_signal=lmsr_velocity_signal,
            latest_bos=latest_bos_1m,
            trend_1m=trend_1m,
            trend_5m=trend_5m,
            order_flow_count=order_flow_count_bearish,
            control_state=control_state,
            nearest_zone=nearest_supply,
            recent_sweep=recent_sweep,
            return_type=return_type,
            has_fvg_fill=has_fvg_fill_bearish,
            has_sd_flip=has_sd_flip_bearish,
            has_qm=has_qm_bearish,
            engulfing_direction=engulfing_direction,
            timestamp=timestamp,
        )

        # Pick higher scoring direction first (bullish tiebreak).
        if bullish_score.total_score >= bearish_score.total_score:
            primary, secondary = bullish_score, bearish_score
        else:
            primary, secondary = bearish_score, bullish_score

        # Try primary direction.
        decision = self._try_direction(primary)
        if decision is not None:
            return decision

        # Try secondary direction.
        decision = self._try_direction(secondary)
        if decision is not None:
            return decision

        # Both vetoed or below threshold.
        return BetDecision(
            direction=None,
            confidence=0.0,
            bet_size_pct=0.0,
            reasons=self._veto_reasons.copy(),
            timestamp=timestamp,
        )

    def get_scoring_engine(self) -> ScoringEngine:
        """Expose scoring engine for direct access."""
        return self._scoring

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _try_direction(self, score: ScoreBreakdown) -> Optional[BetDecision]:
        """Apply vetoes and threshold check. Returns BetDecision or None."""
        c = self._cfg

        # Check minimum score threshold.
        if score.total_score < c.MIN_TOTAL_SCORE_TO_BET:
            self._veto_reasons.append(
                f"{score.direction.value} below threshold "
                f"({score.total_score:.2f} < {c.MIN_TOTAL_SCORE_TO_BET})"
            )
            return None

        # Apply meta-rule vetoes.
        vetoed, reason = self._check_vetoes(score)
        if vetoed:
            self._veto_reasons.append(reason)
            return None

        # Map score → confidence.
        confidence = self._score_to_confidence(score.total_score)

        # Map confidence → bet size.
        bet_size = self._confidence_to_bet_size(confidence)

        return BetDecision(
            direction=score.direction,
            confidence=confidence,
            bet_size_pct=bet_size,
            reasons=score.reasons,
            timestamp=score.timestamp,
            momentum_score=score.momentum_score,
            structure_score=score.structure_score,
            confluence_score=score.confluence_score,
        )

    def _check_vetoes(self, score: ScoreBreakdown) -> tuple[bool, str]:
        """Apply meta-rule vetoes. Returns (is_vetoed, reason)."""
        c = self._cfg

        if c.VETO_CONTROL_OPPOSING and score.control_state_score < 0.2:
            return True, f"VETO: control state opposes {score.direction.value}"

        if c.VETO_CORRECTIVE_RETURN and score.return_type_score < 0.3:
            return True, f"VETO: corrective return type for {score.direction.value}"

        if c.VETO_NO_BOS_CONFIRMATION:
            if score.bos_type_score == 0.0 and score.order_flow_score == 0.0:
                return True, f"VETO: no BOS confirmation for {score.direction.value}"

        if c.VETO_RANGING_BOTH_TF and score.multi_tf_score < 0.35:
            return True, f"VETO: both timeframes ranging/opposing for {score.direction.value}"

        return False, ""

    def _score_to_confidence(self, total_score: float) -> float:
        """Linear interpolation between floor and ceiling."""
        c = self._cfg
        if total_score <= c.CONFIDENCE_FROM_SCORE_FLOOR:
            return 0.0
        if total_score >= c.CONFIDENCE_FROM_SCORE_CEILING:
            return 1.0
        return (total_score - c.CONFIDENCE_FROM_SCORE_FLOOR) / (
            c.CONFIDENCE_FROM_SCORE_CEILING - c.CONFIDENCE_FROM_SCORE_FLOOR
        )

    def _confidence_to_bet_size(self, confidence: float) -> float:
        """Map confidence to bet size percentage.

        0.0: 0.0
        0.0-0.5: BET_SIZE_BASE_PCT
        0.5-0.8: linear base → high_conf
        0.8-1.0: linear high_conf → max
        """
        c = self._cfg
        if confidence <= 0.0:
            return 0.0
        if confidence <= 0.5:
            return c.BET_SIZE_BASE_PCT
        if confidence <= 0.8:
            t = (confidence - 0.5) / 0.3
            return c.BET_SIZE_BASE_PCT + t * (c.BET_SIZE_HIGH_CONF_PCT - c.BET_SIZE_BASE_PCT)
        # 0.8 - 1.0
        t = (confidence - 0.8) / 0.2
        return c.BET_SIZE_HIGH_CONF_PCT + t * (c.BET_SIZE_MAX_PCT - c.BET_SIZE_HIGH_CONF_PCT)
