"""Tests for ScoringEngine — 3-composite SMC scoring model."""

from __future__ import annotations

import pytest

from utils.candle_types import (
    BOS,
    BOSType,
    Candle,
    ControlStateType,
    Direction,
    MarketPhase,
    ReturnType,
    ScoreBreakdown,
    SwingPoint,
    SwingStrength,
    SwingType,
    Zone,
    ZonePattern,
    ZonePosition,
    ZoneType,
)
from engines.liquidity_engine import LiquidityLevel, LiquiditySweep
from engines.scoring_engine import ScoringEngine
import smc_config as cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> ScoringEngine:
    return ScoringEngine()


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
    quality: int = 2,
    bos_direction: Direction = Direction.BULLISH,
    bos_type: BOSType = BOSType.IMPULSIVE,
    swing_strength: SwingStrength = SwingStrength.STRONG,
) -> Zone:
    origin = SwingPoint(timestamp=1.0, price=100.0, type=SwingType.HL, strength=swing_strength)
    bos = BOS(
        timestamp=2.0, price=101.0, direction=bos_direction,
        bos_type=bos_type, swing_origin=origin,
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


# ===================================================================
# 1. LMSR Scoring
# ===================================================================

class TestLMSRScoring:
    def test_positive_signal_bullish(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, lmsr_velocity_signal=0.85)
        assert result.lmsr_velocity_score == pytest.approx(0.85)

    def test_positive_signal_bearish_returns_zero(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BEARISH, lmsr_velocity_signal=0.85)
        assert result.lmsr_velocity_score == 0.0

    def test_zero_signal_neutral(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, lmsr_velocity_signal=0.0)
        assert result.lmsr_velocity_score == pytest.approx(0.5)

    def test_negative_signal_bearish(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BEARISH, lmsr_velocity_signal=-0.72)
        assert result.lmsr_velocity_score == pytest.approx(0.72)

    def test_signal_clamped_to_one(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, lmsr_velocity_signal=1.5)
        assert result.lmsr_velocity_score == pytest.approx(1.0)


# ===================================================================
# 2. BOS Type Scoring
# ===================================================================

class TestBOSTypeScoring:
    def test_impulsive_aligned(self, engine: ScoringEngine) -> None:
        bos = _make_bos(Direction.BULLISH, BOSType.IMPULSIVE)
        result = engine.score(Direction.BULLISH, latest_bos=bos)
        assert result.bos_type_score == pytest.approx(1.0)

    def test_corrective_aligned(self, engine: ScoringEngine) -> None:
        bos = _make_bos(Direction.BULLISH, BOSType.CORRECTIVE)
        result = engine.score(Direction.BULLISH, latest_bos=bos)
        assert result.bos_type_score == pytest.approx(0.3)

    def test_impulsive_opposing(self, engine: ScoringEngine) -> None:
        bos = _make_bos(Direction.BEARISH, BOSType.IMPULSIVE)
        result = engine.score(Direction.BULLISH, latest_bos=bos)
        assert result.bos_type_score == pytest.approx(0.0)

    def test_corrective_opposing(self, engine: ScoringEngine) -> None:
        bos = _make_bos(Direction.BEARISH, BOSType.CORRECTIVE)
        result = engine.score(Direction.BULLISH, latest_bos=bos)
        assert result.bos_type_score == pytest.approx(0.6)

    def test_no_bos(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, latest_bos=None)
        assert result.bos_type_score == pytest.approx(0.4)


# ===================================================================
# 3. Order Flow Scoring
# ===================================================================

class TestOrderFlowScoring:
    def test_zero_count(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, order_flow_count=0)
        assert result.order_flow_score == pytest.approx(0.0)

    def test_healthy_min(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, order_flow_count=cfg.ORDER_FLOW_HEALTHY_MIN)
        assert result.order_flow_score == pytest.approx(0.5)

    def test_strong_min(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, order_flow_count=cfg.ORDER_FLOW_STRONG_MIN)
        assert result.order_flow_score == pytest.approx(1.0)

    def test_above_strong_clamped(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, order_flow_count=10)
        assert result.order_flow_score == pytest.approx(1.0)


# ===================================================================
# 4. Multi-TF Scoring
# ===================================================================

class TestMultiTFScoring:
    def test_both_agree_bullish(self, engine: ScoringEngine) -> None:
        result = engine.score(
            Direction.BULLISH,
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.TRENDING_UP,
        )
        assert result.multi_tf_score == pytest.approx(1.0)

    def test_one_agrees_one_ranging(self, engine: ScoringEngine) -> None:
        result = engine.score(
            Direction.BULLISH,
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.RANGING,
        )
        assert result.multi_tf_score == pytest.approx(0.6)

    def test_both_oppose(self, engine: ScoringEngine) -> None:
        result = engine.score(
            Direction.BULLISH,
            trend_1m=MarketPhase.TRENDING_DOWN,
            trend_5m=MarketPhase.TRENDING_DOWN,
        )
        assert result.multi_tf_score == pytest.approx(0.0)

    def test_both_ranging(self, engine: ScoringEngine) -> None:
        result = engine.score(
            Direction.BULLISH,
            trend_1m=MarketPhase.RANGING,
            trend_5m=MarketPhase.RANGING,
        )
        assert result.multi_tf_score == pytest.approx(0.3)

    def test_none_trend_neutral(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, trend_1m=None, trend_5m=None)
        assert result.multi_tf_score == pytest.approx(0.5)


# ===================================================================
# 5. Control State Scoring
# ===================================================================

class TestControlStateScoring:
    def test_demand_control_bullish(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, control_state=ControlStateType.DEMAND_CONTROL)
        assert result.control_state_score == pytest.approx(1.0)

    def test_neutral(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, control_state=ControlStateType.NEUTRAL)
        assert result.control_state_score == pytest.approx(0.5)

    def test_supply_control_bullish_opposing(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, control_state=ControlStateType.SUPPLY_CONTROL)
        assert result.control_state_score == pytest.approx(0.1)


# ===================================================================
# 6. Zone Position Scoring
# ===================================================================

class TestZonePositionScoring:
    def test_lower_demand_bullish(self, engine: ScoringEngine) -> None:
        zone = _make_zone(ZoneType.DEMAND, ZonePosition.LOWER)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.zone_position_score == pytest.approx(1.0)

    def test_top_supply_bearish(self, engine: ScoringEngine) -> None:
        zone = _make_zone(ZoneType.SUPPLY, ZonePosition.TOP)
        result = engine.score(Direction.BEARISH, nearest_zone=zone)
        assert result.zone_position_score == pytest.approx(1.0)

    def test_no_zone(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, nearest_zone=None)
        assert result.zone_position_score == pytest.approx(0.4)

    def test_wrong_zone_type(self, engine: ScoringEngine) -> None:
        zone = _make_zone(ZoneType.SUPPLY, ZonePosition.TOP)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.zone_position_score == pytest.approx(0.2)


# ===================================================================
# 7. Return Type Scoring
# ===================================================================

class TestReturnTypeScoring:
    def test_v_shape(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, return_type=ReturnType.V_SHAPE)
        assert result.return_type_score == pytest.approx(1.0)

    def test_corrective(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, return_type=ReturnType.CORRECTIVE)
        assert result.return_type_score == pytest.approx(0.2)

    def test_unknown(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, return_type=ReturnType.UNKNOWN)
        assert result.return_type_score == pytest.approx(0.4)

    def test_rounded(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, return_type=ReturnType.ROUNDED)
        assert result.return_type_score == pytest.approx(0.5)


# ===================================================================
# 8. Zone Quality Scoring
# ===================================================================

class TestZoneQualityScoring:
    def test_quality_3(self, engine: ScoringEngine) -> None:
        zone = _make_zone(quality=3)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.zone_quality_score == pytest.approx(1.0)

    def test_quality_0(self, engine: ScoringEngine) -> None:
        zone = _make_zone(quality=0)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.zone_quality_score == pytest.approx(0.2)

    def test_no_zone(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, nearest_zone=None)
        assert result.zone_quality_score == pytest.approx(0.3)


# ===================================================================
# 9. Confluence Sub-Scores
# ===================================================================

class TestConfluenceSubScores:
    def test_sweep_aligned(self, engine: ScoringEngine) -> None:
        sweep = _make_sweep(Direction.BULLISH, is_external=False)
        result = engine.score(Direction.BULLISH, recent_sweep=sweep)
        assert result.sweep_score == pytest.approx(0.8)

    def test_sweep_external(self, engine: ScoringEngine) -> None:
        sweep = _make_sweep(Direction.BULLISH, is_external=True)
        result = engine.score(Direction.BULLISH, recent_sweep=sweep)
        assert result.sweep_score == pytest.approx(1.0)

    def test_sweep_opposing(self, engine: ScoringEngine) -> None:
        sweep = _make_sweep(Direction.BEARISH)
        result = engine.score(Direction.BULLISH, recent_sweep=sweep)
        assert result.sweep_score == pytest.approx(0.1)

    def test_no_sweep(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, recent_sweep=None)
        assert result.sweep_score == pytest.approx(0.3)

    def test_sd_flip_true(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, has_sd_flip=True)
        assert result.sd_flip_score == pytest.approx(1.0)

    def test_sd_flip_false(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, has_sd_flip=False)
        assert result.sd_flip_score == pytest.approx(0.3)

    def test_qm_true(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, has_qm=True)
        assert result.qm_score == pytest.approx(1.0)

    def test_fvg_true(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, has_fvg_fill=True)
        assert result.fvg_score == pytest.approx(1.0)

    def test_engulfing_aligned(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, engulfing_direction=Direction.BULLISH)
        assert result.engulfing_score == pytest.approx(1.0)

    def test_engulfing_opposing(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, engulfing_direction=Direction.BEARISH)
        assert result.engulfing_score == pytest.approx(0.0)

    def test_engulfing_none(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, engulfing_direction=None)
        assert result.engulfing_score == pytest.approx(0.3)


# ===================================================================
# 10. Full Composite Integration
# ===================================================================

class TestFullComposite:
    def test_all_aligned_bullish(self, engine: ScoringEngine) -> None:
        """All signals strongly bullish → high total score."""
        bos = _make_bos(Direction.BULLISH, BOSType.IMPULSIVE)
        zone = _make_zone(ZoneType.DEMAND, ZonePosition.LOWER, quality=3)
        sweep = _make_sweep(Direction.BULLISH, is_external=True)

        result = engine.score(
            Direction.BULLISH,
            lmsr_velocity_signal=0.9,
            latest_bos=bos,
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.TRENDING_UP,
            order_flow_count=5,
            control_state=ControlStateType.DEMAND_CONTROL,
            nearest_zone=zone,
            recent_sweep=sweep,
            return_type=ReturnType.V_SHAPE,
            has_fvg_fill=True,
            has_sd_flip=True,
            has_qm=True,
            engulfing_direction=Direction.BULLISH,
        )

        assert result.total_score > 0.85
        assert result.momentum_score > 0.8
        assert result.structure_score > 0.8
        assert result.confluence_score > 0.8

    def test_mixed_signals_moderate(self, engine: ScoringEngine) -> None:
        """Mixed signals → moderate total score."""
        bos = _make_bos(Direction.BULLISH, BOSType.CORRECTIVE)

        result = engine.score(
            Direction.BULLISH,
            lmsr_velocity_signal=0.5,
            latest_bos=bos,
            trend_1m=MarketPhase.TRENDING_UP,
            trend_5m=MarketPhase.RANGING,
            order_flow_count=2,
            control_state=ControlStateType.NEUTRAL,
        )

        assert 0.25 < result.total_score < 0.65

    def test_all_opposing_low(self, engine: ScoringEngine) -> None:
        """All signals oppose direction → low total score."""
        bos = _make_bos(Direction.BEARISH, BOSType.IMPULSIVE)
        zone = _make_zone(ZoneType.SUPPLY, ZonePosition.TOP)
        sweep = _make_sweep(Direction.BEARISH)

        result = engine.score(
            Direction.BULLISH,
            lmsr_velocity_signal=-0.8,
            latest_bos=bos,
            trend_1m=MarketPhase.TRENDING_DOWN,
            trend_5m=MarketPhase.TRENDING_DOWN,
            order_flow_count=0,
            control_state=ControlStateType.SUPPLY_CONTROL,
            nearest_zone=zone,
            recent_sweep=sweep,
            return_type=ReturnType.CORRECTIVE,
            engulfing_direction=Direction.BEARISH,
        )

        assert result.total_score < 0.2

    def test_weight_formula(self, engine: ScoringEngine) -> None:
        """Verify total = momentum*0.40 + structure*0.35 + confluence*0.25."""
        result = engine.score(
            Direction.BULLISH,
            lmsr_velocity_signal=0.6,
            order_flow_count=2,
            control_state=ControlStateType.NEUTRAL,
        )

        expected = (
            result.momentum_score * cfg.MOMENTUM_WEIGHT
            + result.structure_score * cfg.STRUCTURE_WEIGHT
            + result.confluence_score * cfg.CONFLUENCE_WEIGHT
        )
        assert result.total_score == pytest.approx(expected)

    def test_reasons_populated(self, engine: ScoringEngine) -> None:
        """Noteworthy scores produce reason strings."""
        bos = _make_bos(Direction.BULLISH, BOSType.IMPULSIVE)
        sweep = _make_sweep(Direction.BULLISH, is_external=True)

        result = engine.score(
            Direction.BULLISH,
            lmsr_velocity_signal=0.9,
            latest_bos=bos,
            recent_sweep=sweep,
            control_state=ControlStateType.DEMAND_CONTROL,
        )

        assert len(result.reasons) > 0
        reason_text = " ".join(result.reasons)
        assert "LMSR" in reason_text
        assert "Impulsive BOS" in reason_text


# ===================================================================
# 11. Swing Strength Scoring
# ===================================================================

class TestSwingStrengthScoring:
    def test_strong(self, engine: ScoringEngine) -> None:
        zone = _make_zone(swing_strength=SwingStrength.STRONG)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.swing_strength_score == pytest.approx(1.0)

    def test_moderate(self, engine: ScoringEngine) -> None:
        zone = _make_zone(swing_strength=SwingStrength.MODERATE)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.swing_strength_score == pytest.approx(0.6)

    def test_weak(self, engine: ScoringEngine) -> None:
        zone = _make_zone(swing_strength=SwingStrength.WEAK)
        result = engine.score(Direction.BULLISH, nearest_zone=zone)
        assert result.swing_strength_score == pytest.approx(0.3)

    def test_no_zone(self, engine: ScoringEngine) -> None:
        result = engine.score(Direction.BULLISH, nearest_zone=None)
        assert result.swing_strength_score == pytest.approx(0.4)
