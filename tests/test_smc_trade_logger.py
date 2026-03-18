"""Tests for SMCTradeLogger."""

from __future__ import annotations

import json

import pytest

from db import BotDatabase
from engines.smc_trade_logger import SMCTradeLogger
from utils.candle_types import (
    BetDecision,
    Candle,
    Direction,
    ScoreBreakdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> BotDatabase:
    return BotDatabase(":memory:")


def _make_score(
    direction: Direction = Direction.BULLISH,
    total: float = 0.55,
    momentum: float = 0.6,
    structure: float = 0.5,
    confluence: float = 0.4,
    **kwargs,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        direction=direction,
        total_score=total,
        momentum_score=momentum,
        structure_score=structure,
        confluence_score=confluence,
        lmsr_velocity_score=kwargs.get("lmsr_velocity_score", 0.7),
        bos_type_score=kwargs.get("bos_type_score", 0.5),
        order_flow_score=kwargs.get("order_flow_score", 0.4),
        multi_tf_score=kwargs.get("multi_tf_score", 0.6),
        control_state_score=kwargs.get("control_state_score", 0.8),
        zone_position_score=kwargs.get("zone_position_score", 0.5),
        swing_strength_score=kwargs.get("swing_strength_score", 0.6),
        return_type_score=kwargs.get("return_type_score", 0.7),
        zone_quality_score=kwargs.get("zone_quality_score", 0.5),
        sweep_score=kwargs.get("sweep_score", 0.3),
        sd_flip_score=kwargs.get("sd_flip_score", 0.3),
        qm_score=kwargs.get("qm_score", 0.3),
        fvg_score=kwargs.get("fvg_score", 0.3),
        engulfing_score=kwargs.get("engulfing_score", 0.3),
        reasons=kwargs.get("reasons", ["test reason"]),
        timestamp=kwargs.get("timestamp", 1000.0),
    )


def _make_decision(
    direction: Direction | None = Direction.BULLISH,
    confidence: float = 0.65,
    bet_size_pct: float = 0.07,
    reasons: list[str] | None = None,
) -> BetDecision:
    return BetDecision(
        direction=direction,
        confidence=confidence,
        bet_size_pct=bet_size_pct,
        reasons=reasons or ["test reason"],
        timestamp=1000.0,
        momentum_score=0.6,
        structure_score=0.5,
        confluence_score=0.4,
    )


# ---------------------------------------------------------------------------
# Decision Logging
# ---------------------------------------------------------------------------


class TestDecisionLogging:
    def test_log_bullish_decision(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        decision = _make_decision(Direction.BULLISH)
        score = _make_score(Direction.BULLISH)

        row_id = logger.log_decision(1000, decision, score)

        assert row_id is not None
        rows = db.get_smc_decisions(window_ts=1000)
        assert len(rows) == 1
        row = rows[0]
        assert row["direction"] == "bullish"
        assert row["confidence"] == pytest.approx(0.65)
        assert row["momentum_score"] == pytest.approx(0.6)
        assert row["structure_score"] == pytest.approx(0.5)
        assert row["confluence_score"] == pytest.approx(0.4)
        assert row["is_skip"] == 0

    def test_log_skip_decision(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        decision = _make_decision(direction=None, confidence=0.0, bet_size_pct=0.0)
        score = _make_score(Direction.BULLISH, total=0.3)

        logger.log_decision(2000, decision, score)

        rows = db.get_smc_decisions(window_ts=2000)
        assert len(rows) == 1
        row = rows[0]
        assert row["direction"] is None
        assert row["is_skip"] == 1

    def test_log_with_full_engine_context(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        decision = _make_decision(Direction.BEARISH)
        score = _make_score(Direction.BEARISH, engulfing_score=0.8)

        row_id = logger.log_decision(
            3000,
            decision,
            score,
            lmsr_velocity_raw=0.025,
        )

        rows = db.get_smc_decisions(window_ts=3000)
        assert len(rows) == 1
        row = rows[0]
        assert row["direction"] == "bearish"
        assert row["lmsr_velocity_raw"] == pytest.approx(0.025)
        assert row["has_engulfing"] == 1  # engulfing_score > 0.5

    def test_reasons_json_valid(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        reasons = ["bullish BOS", "demand zone fresh", "LMSR velocity high"]
        decision = _make_decision(reasons=reasons)
        score = _make_score()

        logger.log_decision(4000, decision, score)

        rows = db.get_smc_decisions(window_ts=4000)
        parsed = json.loads(rows[0]["reasons_json"])
        assert parsed == reasons

    def test_veto_detection(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        decision = _make_decision(
            direction=None,
            confidence=0.0,
            bet_size_pct=0.0,
            reasons=["VETO: control state opposes bullish"],
        )
        score = _make_score()

        logger.log_decision(5000, decision, score)

        rows = db.get_smc_decisions(window_ts=5000)
        assert rows[0]["was_vetoed"] == 1
        assert "control state" in rows[0]["veto_reason"]


# ---------------------------------------------------------------------------
# Candle Logging
# ---------------------------------------------------------------------------


class TestCandleLogging:
    def test_log_1m_candles(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        for i in range(5):
            c = Candle(
                timestamp=1000.0 + i * 60,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=10.0,
            )
            logger.log_candle(1000, "1m", c)
        logger.flush()

        rows = db.get_smc_candles(1000, "1m")
        assert len(rows) == 5
        assert rows[0]["candle_timestamp"] == 1000.0
        assert rows[4]["candle_timestamp"] == 1240.0

    def test_log_5m_candles(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        c = Candle(
            timestamp=1000.0,
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=50.0,
        )
        logger.log_candle(1000, "5m", c)
        logger.flush()

        rows = db.get_smc_candles(1000, "5m")
        assert len(rows) == 1
        assert rows[0]["timeframe"] == "5m"

    def test_flush_commits(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        c = Candle(timestamp=1000.0, open=100.0, high=101.0, low=99.0, close=100.5)
        logger.log_candle(1000, "1m", c)

        # Before flush, data may not be committed (but is visible via same connection).
        rows = db.get_smc_candles(1000, "1m")
        assert len(rows) == 1

        logger.flush()
        rows = db.get_smc_candles(1000, "1m")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Decision Summary
# ---------------------------------------------------------------------------


class TestDecisionSummary:
    def test_summary_counts(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        # Log 3 bets and 2 skips.
        for i in range(3):
            d = _make_decision(Direction.BULLISH)
            s = _make_score(Direction.BULLISH)
            logger.log_decision(1000 + i * 300, d, s)

        for i in range(2):
            d = _make_decision(direction=None, confidence=0.0, bet_size_pct=0.0)
            s = _make_score(Direction.BULLISH, total=0.3)
            logger.log_decision(2000 + i * 300, d, s)

        summary = logger.get_decision_summary()
        assert summary["total_decisions"] == 5
        assert summary["total_bets"] == 3
        assert summary["total_skips"] == 2

    def test_direction_counts_and_vetoes(self):
        db = _make_db()
        logger = SMCTradeLogger(db)

        # 2 bullish bets
        for i in range(2):
            d = _make_decision(Direction.BULLISH)
            s = _make_score(Direction.BULLISH)
            logger.log_decision(1000 + i * 300, d, s)

        # 1 bearish bet
        d = _make_decision(Direction.BEARISH)
        s = _make_score(Direction.BEARISH)
        logger.log_decision(2000, d, s)

        # 1 vetoed skip
        d = _make_decision(
            direction=None,
            confidence=0.0,
            bet_size_pct=0.0,
            reasons=["VETO: control state opposes bullish"],
        )
        s = _make_score(Direction.BULLISH)
        logger.log_decision(3000, d, s)

        summary = logger.get_decision_summary()
        assert summary["direction_counts"]["BULLISH"] == 2
        assert summary["direction_counts"]["BEARISH"] == 1
        assert len(summary["top_veto_reasons"]) == 1
        assert summary["top_veto_reasons"][0][1] == 1  # count
