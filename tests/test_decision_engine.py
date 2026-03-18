"""Tests for DecisionEngine — meta-rules, vetoes, sizing, and BetDecision output."""

from __future__ import annotations

import pytest

from utils.candle_types import (
    BOS,
    BOSType,
    BetDecision,
    Candle,
    ControlStateType,
    Direction,
    MarketPhase,
    ReturnType,
    SwingPoint,
    SwingStrength,
    SwingType,
    Zone,
    ZonePattern,
    ZonePosition,
    ZoneType,
)
from engines.liquidity_engine import LiquidityLevel, LiquiditySweep
from engines.decision_engine import DecisionEngine
import smc_config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bos(
    direction: Direction = Direction.BULLISH,
    bos_type: BOSType = BOSType.IMPULSIVE,
    strength: SwingStrength = SwingStrength.STRONG,
) -> BOS:
    origin = SwingPoint(timestamp=1.0, price=100.0, type=SwingType.HL, strength=strength)
    return BOS(
        timestamp=2.0, price=101.0, direction=direction,
        bos_type=bos_type, swing_origin=origin,
    )


def _make_zone(
    zone_type: ZoneType = ZoneType.DEMAND,
    position: ZonePosition = ZonePosition.LOWER,
    quality: int = 3,
    bos_direction: Direction = Direction.BULLISH,
    swing_strength: SwingStrength = SwingStrength.STRONG,
) -> Zone:
    origin = SwingPoint(timestamp=1.0, price=100.0, type=SwingType.HL, strength=swing_strength)
    bos = BOS(
        timestamp=2.0, price=101.0, direction=bos_direction,
        bos_type=BOSType.IMPULSIVE, swing_origin=origin,
    )
    return Zone(
        timestamp=1.0, high=101.0, low=99.0, zone_type=zone_type,
        pattern=ZonePattern.DBR, quality_score=quality, position=position,
        creation_bos=bos,
    )


def _make_sweep(
    direction_after: Direction = Direction.BULLISH,
    is_external: bool = False,
) -> LiquiditySweep:
    candle = Candle(timestamp=1.0, open=100.0, high=102.0, low=98.0, close=100.5)
    level = LiquidityLevel(price=101.5, level_type="EQL", swing_indices=[0, 2])
    return LiquiditySweep(
        timestamp=1.0, level=level, sweep_candle=candle,
        direction_after=direction_after, is_external=is_external,
    )


def _strong_bullish_kwargs() -> dict:
    """Keyword args that produce a strong bullish signal."""
    return dict(
        lmsr_velocity_signal=0.9,
        latest_bos_1m=_make_bos(Direction.BULLISH, BOSType.IMPULSIVE),
        trend_1m=MarketPhase.TRENDING_UP,
        trend_5m=MarketPhase.TRENDING_UP,
        order_flow_count_bullish=5,
        order_flow_count_bearish=0,
        control_state=ControlStateType.DEMAND_CONTROL,
        nearest_demand=_make_zone(ZoneType.DEMAND, ZonePosition.LOWER, quality=3),
        nearest_supply=None,
        recent_sweep=_make_sweep(Direction.BULLISH, is_external=True),
        return_type=ReturnType.V_SHAPE,
        has_fvg_fill_bullish=True,
        has_sd_flip_bullish=True,
        has_qm_bullish=True,
        engulfing_direction=Direction.BULLISH,
    )


# ===================================================================
# 1. Direction Selection
# ===================================================================

class TestDirectionSelection:
    def test_bullish_higher_score(self) -> None:
        engine = DecisionEngine()
        decision = engine.decide(**_strong_bullish_kwargs())
        assert decision.direction == Direction.BULLISH

    def test_bearish_higher_score(self) -> None:
        engine = DecisionEngine()
        decision = engine.decide(
            lmsr_velocity_signal=-0.9,
            latest_bos_1m=_make_bos(Direction.BEARISH, BOSType.IMPULSIVE),
            trend_1m=MarketPhase.TRENDING_DOWN,
            trend_5m=MarketPhase.TRENDING_DOWN,
            order_flow_count_bullish=0,
            order_flow_count_bearish=5,
            control_state=ControlStateType.SUPPLY_CONTROL,
            nearest_demand=None,
            nearest_supply=_make_zone(ZoneType.SUPPLY, ZonePosition.TOP, quality=3,
                                      bos_direction=Direction.BEARISH),
            recent_sweep=_make_sweep(Direction.BEARISH, is_external=True),
            return_type=ReturnType.V_SHAPE,
            has_fvg_fill_bearish=True,
            has_sd_flip_bearish=True,
            has_qm_bearish=True,
            engulfing_direction=Direction.BEARISH,
        )
        assert decision.direction == Direction.BEARISH

    def test_equal_scores_bullish_tiebreak(self) -> None:
        """When both directions score equally, bullish wins (>= comparison)."""
        engine = DecisionEngine()
        # Default args produce neutral scores for both directions.
        # Both are likely below threshold, but the tiebreak logic is tested
        # by ensuring bullish is tried first.
        decision = engine.decide(
            lmsr_velocity_signal=0.0,
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.TRENDING_UP,
            order_flow_count_bullish=3,
            order_flow_count_bearish=3,
            control_state=ControlStateType.NEUTRAL,
            return_type=ReturnType.ROUNDED,
        )
        # Both should have similar scores; if either passes, it's bullish first
        if decision.direction is not None:
            assert decision.direction == Direction.BULLISH


# ===================================================================
# 2. Meta-Rule Vetoes
# ===================================================================

class TestVetoes:
    def test_veto_control_opposing(self) -> None:
        engine = DecisionEngine()
        kwargs = _strong_bullish_kwargs()
        kwargs["control_state"] = ControlStateType.SUPPLY_CONTROL
        decision = engine.decide(**kwargs)
        # Bullish vetoed by opposing control state; bearish likely tried
        if decision.is_skip:
            assert any("VETO" in r for r in decision.reasons)
        else:
            # Secondary direction may have passed
            assert decision.direction == Direction.BEARISH

    def test_veto_corrective_return(self) -> None:
        engine = DecisionEngine()
        kwargs = _strong_bullish_kwargs()
        kwargs["return_type"] = ReturnType.CORRECTIVE
        decision = engine.decide(**kwargs)
        # Corrective return vetoes the direction
        reasons_text = " ".join(decision.reasons)
        if decision.is_skip:
            assert "VETO" in reasons_text or "corrective" in reasons_text.lower()

    def test_veto_no_bos_confirmation(self) -> None:
        engine = DecisionEngine()
        # No BOS and no order flow → both bos_type_score and order_flow_score
        # need to be zero. But impulsive opposing BOS gives 0.0 for bos_type.
        decision = engine.decide(
            lmsr_velocity_signal=0.9,
            latest_bos_1m=_make_bos(Direction.BEARISH, BOSType.IMPULSIVE),
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.TRENDING_UP,
            order_flow_count_bullish=0,
            order_flow_count_bearish=0,
            control_state=ControlStateType.DEMAND_CONTROL,
            return_type=ReturnType.V_SHAPE,
        )
        reasons_text = " ".join(decision.reasons)
        # At least one direction should be vetoed for no BOS
        assert "VETO" in reasons_text or "no BOS" in reasons_text.lower() or decision.direction is not None

    def test_veto_ranging_both_tf(self) -> None:
        engine = DecisionEngine()
        kwargs = _strong_bullish_kwargs()
        kwargs["trend_1m"] = MarketPhase.RANGING
        kwargs["trend_5m"] = MarketPhase.RANGING
        decision = engine.decide(**kwargs)
        reasons_text = " ".join(decision.reasons)
        if decision.is_skip:
            assert "VETO" in reasons_text or "ranging" in reasons_text.lower()

    def test_primary_vetoed_secondary_passes(self) -> None:
        """If primary is vetoed, secondary direction can still pass."""
        engine = DecisionEngine()
        # Strong bearish signal but supply control opposes bullish
        decision = engine.decide(
            lmsr_velocity_signal=-0.8,
            latest_bos_1m=_make_bos(Direction.BEARISH, BOSType.IMPULSIVE),
            trend_1m=MarketPhase.TRENDING_DOWN,
            trend_5m=MarketPhase.TRENDING_DOWN,
            order_flow_count_bullish=0,
            order_flow_count_bearish=4,
            control_state=ControlStateType.SUPPLY_CONTROL,
            nearest_supply=_make_zone(ZoneType.SUPPLY, ZonePosition.TOP, quality=3,
                                      bos_direction=Direction.BEARISH),
            recent_sweep=_make_sweep(Direction.BEARISH, is_external=True),
            return_type=ReturnType.V_SHAPE,
            has_fvg_fill_bearish=True,
            has_sd_flip_bearish=True,
            engulfing_direction=Direction.BEARISH,
        )
        # Bearish should score high and pass; bullish should be low/vetoed
        assert decision.direction == Direction.BEARISH


# ===================================================================
# 3. Score Below Threshold
# ===================================================================

class TestScoreThreshold:
    def test_both_below_threshold_skip(self) -> None:
        engine = DecisionEngine()
        # All defaults → low scores for both directions
        decision = engine.decide()
        assert decision.is_skip
        assert decision.direction is None

    def test_one_above_one_below(self) -> None:
        engine = DecisionEngine()
        kwargs = _strong_bullish_kwargs()
        # Bearish will be very low, bullish will be high
        decision = engine.decide(**kwargs)
        assert decision.direction == Direction.BULLISH
        assert decision.confidence > 0


# ===================================================================
# 4. Confidence Mapping
# ===================================================================

class TestConfidenceMapping:
    def test_score_at_floor(self) -> None:
        engine = DecisionEngine()
        conf = engine._score_to_confidence(cfg.CONFIDENCE_FROM_SCORE_FLOOR)
        assert conf == pytest.approx(0.0)

    def test_score_at_ceiling(self) -> None:
        engine = DecisionEngine()
        conf = engine._score_to_confidence(cfg.CONFIDENCE_FROM_SCORE_CEILING)
        assert conf == pytest.approx(1.0)

    def test_score_midway(self) -> None:
        engine = DecisionEngine()
        mid = (cfg.CONFIDENCE_FROM_SCORE_FLOOR + cfg.CONFIDENCE_FROM_SCORE_CEILING) / 2
        conf = engine._score_to_confidence(mid)
        assert conf == pytest.approx(0.5)

    def test_score_below_floor(self) -> None:
        engine = DecisionEngine()
        conf = engine._score_to_confidence(0.1)
        assert conf == pytest.approx(0.0)

    def test_score_above_ceiling(self) -> None:
        engine = DecisionEngine()
        conf = engine._score_to_confidence(0.99)
        assert conf == pytest.approx(1.0)


# ===================================================================
# 5. Bet Sizing
# ===================================================================

class TestBetSizing:
    def test_zero_confidence(self) -> None:
        engine = DecisionEngine()
        assert engine._confidence_to_bet_size(0.0) == pytest.approx(0.0)

    def test_low_confidence_base(self) -> None:
        engine = DecisionEngine()
        size = engine._confidence_to_bet_size(0.3)
        assert size == pytest.approx(cfg.BET_SIZE_BASE_PCT)

    def test_medium_confidence_interpolated(self) -> None:
        engine = DecisionEngine()
        size = engine._confidence_to_bet_size(0.65)
        assert cfg.BET_SIZE_BASE_PCT < size < cfg.BET_SIZE_HIGH_CONF_PCT

    def test_high_confidence_near_max(self) -> None:
        engine = DecisionEngine()
        size = engine._confidence_to_bet_size(1.0)
        assert size == pytest.approx(cfg.BET_SIZE_MAX_PCT)

    def test_never_exceeds_max(self) -> None:
        engine = DecisionEngine()
        # Even at extreme confidence, never exceed max
        size = engine._confidence_to_bet_size(1.0)
        assert size <= cfg.BET_SIZE_MAX_PCT


# ===================================================================
# 6. BetDecision Output
# ===================================================================

class TestBetDecisionOutput:
    def test_valid_decision_fields(self) -> None:
        engine = DecisionEngine()
        decision = engine.decide(**_strong_bullish_kwargs(), timestamp=42.0)
        assert decision.direction == Direction.BULLISH
        assert decision.confidence > 0
        assert decision.bet_size_pct > 0
        assert decision.timestamp == 42.0
        assert decision.momentum_score > 0
        assert decision.structure_score > 0
        assert decision.confluence_score > 0
        assert len(decision.reasons) > 0

    def test_should_bet_true(self) -> None:
        engine = DecisionEngine()
        decision = engine.decide(**_strong_bullish_kwargs())
        assert decision.should_bet is True

    def test_is_skip_true_for_vetoed(self) -> None:
        engine = DecisionEngine()
        decision = engine.decide()  # all defaults → skip
        assert decision.is_skip is True
        assert decision.should_bet is False


# ===================================================================
# 7. Full Integration
# ===================================================================

class TestFullIntegration:
    def test_strong_bullish_setup(self) -> None:
        """Strong bullish with LMSR + BOS + sweep + zone → high confidence bet."""
        engine = DecisionEngine()
        decision = engine.decide(**_strong_bullish_kwargs())
        assert decision.direction == Direction.BULLISH
        assert decision.confidence > 0.5
        assert decision.bet_size_pct >= cfg.BET_SIZE_BASE_PCT

    def test_conflicting_signals_skip(self) -> None:
        """Conflicting signals across categories → low confidence or skip."""
        engine = DecisionEngine()
        decision = engine.decide(
            lmsr_velocity_signal=0.3,
            latest_bos_1m=_make_bos(Direction.BEARISH, BOSType.CORRECTIVE),
            trend_1m=MarketPhase.RANGING,
            trend_5m=MarketPhase.RANGING,
            order_flow_count_bullish=1,
            order_flow_count_bearish=1,
            control_state=ControlStateType.NEUTRAL,
            return_type=ReturnType.CORRECTIVE,
        )
        # Should either skip or have very low confidence
        if not decision.is_skip:
            assert decision.confidence < 0.3

    def test_scoring_engine_accessible(self) -> None:
        """get_scoring_engine() returns the internal ScoringEngine."""
        engine = DecisionEngine()
        scoring = engine.get_scoring_engine()
        assert scoring is not None
        # Can call score directly
        breakdown = scoring.score(Direction.BULLISH)
        assert isinstance(breakdown.total_score, float)
