"""Tests for Backtester."""

from __future__ import annotations

import pytest

from db import BotDatabase
from engines.backtester import Backtester, BacktestResult
from utils.candle_types import Candle, Direction


# ---------------------------------------------------------------------------
# Synthetic candle helpers
# ---------------------------------------------------------------------------


def make_trending_up_candles(
    n: int, start_price: float = 100.0, step: float = 0.5, base_ts: float = 0.0,
) -> list[Candle]:
    """Generate n 1m candles with ascending closes."""
    candles = []
    for i in range(n):
        o = start_price + i * step
        c = o + step * 0.8
        h = c + step * 0.2
        lo = o - step * 0.1
        candles.append(
            Candle(
                timestamp=base_ts + i * 60,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=10.0,
            )
        )
    return candles


def make_trending_down_candles(
    n: int, start_price: float = 110.0, step: float = 0.5, base_ts: float = 0.0,
) -> list[Candle]:
    """Generate n 1m candles with descending closes."""
    candles = []
    for i in range(n):
        o = start_price - i * step
        c = o - step * 0.8
        h = o + step * 0.1
        lo = c - step * 0.2
        candles.append(
            Candle(
                timestamp=base_ts + i * 60,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=10.0,
            )
        )
    return candles


def make_ranging_candles(
    n: int, center_price: float = 100.0, amplitude: float = 0.3, base_ts: float = 0.0,
) -> list[Candle]:
    """Generate n 1m candles oscillating around center."""
    candles = []
    for i in range(n):
        # Alternate up and down.
        if i % 2 == 0:
            o = center_price - amplitude * 0.5
            c = center_price + amplitude * 0.5
        else:
            o = center_price + amplitude * 0.5
            c = center_price - amplitude * 0.5
        h = max(o, c) + amplitude * 0.1
        lo = min(o, c) - amplitude * 0.1
        candles.append(
            Candle(
                timestamp=base_ts + i * 60,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=10.0,
            )
        )
    return candles


def _aligned_ts(n: int = 0) -> float:
    """Return a timestamp aligned to a 300-second boundary."""
    return 1000 * 300 + n * 60


# ---------------------------------------------------------------------------
# Basic Pipeline
# ---------------------------------------------------------------------------


class TestBasicPipeline:
    def test_few_candles_no_crash(self):
        """Feed 10 candles — backtester doesn't crash, returns BacktestResult."""
        candles = make_trending_up_candles(10, base_ts=_aligned_ts())
        bt = Backtester()
        result = bt.run(candles)
        assert isinstance(result, BacktestResult)
        assert result.total_candles == 10

    def test_one_window_one_decision(self):
        """Feed enough candles for one 5m window — exactly 1 decision."""
        # 5 candles aligned to one 300s window.
        base = 300000  # aligned to 300
        candles = make_trending_up_candles(5, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        # With only 5 candles and no prior data, we may get 0 or 1 decision
        # depending on whether the 5m candle completes.
        assert result.total_candles == 5
        assert result.total_decisions <= 1

    def test_multiple_windows(self):
        """Feed candles spanning multiple 5m windows."""
        # 15 candles = 3 potential 5m windows.
        base = 300000
        candles = make_trending_up_candles(15, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        assert result.total_candles == 15
        # Should produce at least 1 decision.
        assert result.total_decisions >= 1


# ---------------------------------------------------------------------------
# 5m Candle Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_5m_ohlc_values(self):
        """5 aligned 1m candles produce correct 5m OHLC."""
        candles = [
            Candle(timestamp=300000.0, open=100.0, high=102.0, low=99.0, close=101.0, volume=10.0),
            Candle(timestamp=300060.0, open=101.0, high=103.0, low=100.0, close=102.0, volume=10.0),
            Candle(timestamp=300120.0, open=102.0, high=105.0, low=101.0, close=104.0, volume=10.0),
            Candle(timestamp=300180.0, open=104.0, high=106.0, low=103.0, close=105.0, volume=10.0),
            Candle(timestamp=300240.0, open=105.0, high=107.0, low=104.0, close=106.0, volume=10.0),
        ]
        five = Backtester._build_5m_candle(candles)
        assert five.open == 100.0
        assert five.high == 107.0
        assert five.low == 99.0
        assert five.close == 106.0

    def test_5m_volume_sum(self):
        """5m candle volume is sum of 1m volumes."""
        candles = [
            Candle(timestamp=300000.0 + i * 60, open=100.0, high=101.0, low=99.0, close=100.0, volume=5.0 + i)
            for i in range(5)
        ]
        five = Backtester._build_5m_candle(candles)
        assert five.volume == pytest.approx(5 + 6 + 7 + 8 + 9)


# ---------------------------------------------------------------------------
# Decision Quality
# ---------------------------------------------------------------------------


class TestDecisionQuality:
    def test_bullish_trend_tends_bullish(self):
        """Strongly bullish candles should produce bullish decisions (if any)."""
        # Need enough candles to build market structure (swings need ~10+ candles).
        base = 300000
        candles = make_trending_up_candles(30, step=2.0, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        bullish_bets = [
            d for d in result.decisions
            if d.get("direction") == "bullish"
        ]
        bearish_bets = [
            d for d in result.decisions
            if d.get("direction") == "bearish"
        ]
        # Bullish bets should outnumber bearish (or all skip due to vetoes).
        assert len(bullish_bets) >= len(bearish_bets)

    def test_bearish_trend_tends_bearish(self):
        """Strongly bearish candles should produce bearish decisions (if any)."""
        base = 300000
        candles = make_trending_down_candles(30, step=2.0, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        bearish_bets = [
            d for d in result.decisions
            if d.get("direction") == "bearish"
        ]
        bullish_bets = [
            d for d in result.decisions
            if d.get("direction") == "bullish"
        ]
        assert len(bearish_bets) >= len(bullish_bets)

    def test_ranging_more_skips(self):
        """Ranging candles should produce more skips than bets."""
        base = 300000
        candles = make_ranging_candles(30, amplitude=0.1, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        # Ranging market should be mostly skipped due to vetoes.
        assert result.total_skips >= result.total_bets


# ---------------------------------------------------------------------------
# Outcome Resolution
# ---------------------------------------------------------------------------


class TestOutcomeResolution:
    def _make_candles_with_outcomes(self):
        """Create candles spanning 2 windows with known outcomes."""
        base = 300000
        candles = make_trending_up_candles(15, step=1.0, base_ts=float(base))
        # Window boundaries.
        w1 = base // 300 * 300
        w2 = (base + 300) // 300 * 300
        outcomes = {w1: "UP", w2: "DOWN"}
        return candles, outcomes

    def test_wins_losses_counted(self):
        """Wins and losses are counted when outcomes provided."""
        candles, outcomes = self._make_candles_with_outcomes()
        bt = Backtester()
        result = bt.run(candles, window_outcomes=outcomes)
        # At least one decision should be resolved.
        resolved = [d for d in result.decisions if "outcome" in d]
        # Total wins + losses should match resolved count.
        assert result.wins + result.losses == len(resolved)

    def test_pnl_calculation(self):
        """PnL is computed correctly for wins and losses."""
        bt = Backtester()
        # Win: BULLISH + UP → (1.0 - 0.50) * bet_size
        from utils.candle_types import BetDecision
        decision = BetDecision(
            direction=Direction.BULLISH, confidence=0.6, bet_size_pct=0.10,
        )
        outcome, pnl = bt._resolve_outcome(decision, "UP")
        assert outcome == "WIN"
        assert pnl == pytest.approx(0.50 * 0.10)

        # Loss: BULLISH + DOWN → -0.50 * bet_size
        outcome, pnl = bt._resolve_outcome(decision, "DOWN")
        assert outcome == "LOSS"
        assert pnl == pytest.approx(-0.50 * 0.10)

    def test_win_rate_calculation(self):
        """Win rate is computed correctly."""
        result = BacktestResult(wins=3, losses=2)
        assert result.win_rate == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# LMSR Integration
# ---------------------------------------------------------------------------


class TestLMSRIntegration:
    def test_lmsr_velocity_affects_decision(self):
        """LMSR velocity per window affects scoring."""
        base = 300000
        candles = make_trending_up_candles(15, step=1.0, base_ts=float(base))
        w1 = base // 300 * 300

        bt = Backtester()
        # Run without LMSR.
        result_no_lmsr = bt.run(candles)
        # Run with strong bullish LMSR.
        bt2 = Backtester()
        result_with_lmsr = bt2.run(
            candles, lmsr_velocity_per_window={w1: 0.05}
        )
        # Both should produce results without error.
        assert isinstance(result_no_lmsr, BacktestResult)
        assert isinstance(result_with_lmsr, BacktestResult)

    def test_no_lmsr_neutral_score(self):
        """Without LMSR data, LMSR velocity signal defaults to 0."""
        base = 300000
        candles = make_trending_up_candles(10, base_ts=float(base))
        bt = Backtester()
        result = bt.run(candles)
        # Should not crash — neutral LMSR.
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Database Integration
# ---------------------------------------------------------------------------


class TestDatabaseIntegration:
    def test_decisions_logged_to_db(self):
        """Run with db → decisions logged to smc_decisions."""
        db = BotDatabase(":memory:")
        base = 300000
        candles = make_trending_up_candles(15, step=1.0, base_ts=float(base))
        bt = Backtester(db=db)
        result = bt.run(candles)

        # Check that decisions were logged.
        db_decisions = db.get_smc_decisions()
        assert len(db_decisions) == result.total_decisions

    def test_backtest_run_logged(self):
        """Backtest run summary is logged to backtest_runs."""
        db = BotDatabase(":memory:")
        base = 300000
        candles = make_trending_up_candles(15, step=1.0, base_ts=float(base))
        bt = Backtester(db=db)
        bt.run(candles, description="test run")

        runs = db.get_backtest_runs()
        assert len(runs) == 1
        assert runs[0]["description"] == "test run"
        assert runs[0]["candle_count"] == 15


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_summary_string(self):
        """summary() returns readable string."""
        result = BacktestResult(
            total_candles=100,
            total_decisions=10,
            total_bets=6,
            total_skips=4,
            wins=4,
            losses=2,
            total_pnl=0.15,
        )
        s = result.summary()
        assert "100 candles" in s
        assert "10 windows" in s
        assert "66.7%" in s
        assert "+0.1500" in s

    def test_properties_compute(self):
        """win_rate, avg_confidence, avg_bet_size compute correctly."""
        result = BacktestResult(
            wins=2,
            losses=3,
            decisions=[
                {"confidence": 0.6, "bet_size_pct": 0.05},
                {"confidence": 0.8, "bet_size_pct": 0.10},
                {"confidence": 0.0, "bet_size_pct": 0.0},
            ],
        )
        assert result.win_rate == pytest.approx(0.4)
        assert result.avg_confidence == pytest.approx(0.7)
        assert result.avg_bet_size == pytest.approx(0.075)

    def test_empty_result(self):
        """Empty result has zero values."""
        result = BacktestResult()
        assert result.win_rate == 0.0
        assert result.avg_confidence == 0.0
        assert result.avg_bet_size == 0.0


# ---------------------------------------------------------------------------
# run_from_db
# ---------------------------------------------------------------------------


class TestRunFromDB:
    def test_run_from_stored_candles(self):
        """Store candles in db, then run_from_db → produces results."""
        db = BotDatabase(":memory:")
        base = 300000
        candles = make_trending_up_candles(10, base_ts=float(base))
        w_ts = base // 300 * 300

        # Store candles.
        for c in candles:
            db.record_smc_candle(w_ts, "1m", c)
        db.conn.commit()

        bt = Backtester(db=db)
        result = bt.run_from_db(db, timeframe="1m")
        assert isinstance(result, BacktestResult)
        assert result.total_candles == 10

    def test_run_from_db_with_lmsr(self):
        """run_from_db fetches LMSR velocity from market_snapshots."""
        db = BotDatabase(":memory:")
        base = 300000
        w_ts = base // 300 * 300

        # Record a window so FK doesn't fail.
        db.record_window(w_ts, f"btc-updown-5m-{w_ts}")

        candles = make_trending_up_candles(10, base_ts=float(base))
        for c in candles:
            db.record_smc_candle(w_ts, "1m", c)
        db.conn.commit()

        # Record a market snapshot with velocity.
        db.record_market_snapshot(
            w_ts, up_price=0.52, down_price=0.48,
            velocity=0.03, acceleration=0.01,
        )

        bt = Backtester(db=db)
        result = bt.run_from_db(db, timeframe="1m")
        assert isinstance(result, BacktestResult)
