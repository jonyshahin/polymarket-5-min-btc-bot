"""Entry point: async main loop with --dry-run, --once, --strategy flags."""

import argparse
import asyncio
import logging
import math
import sys
import time
from datetime import datetime, timezone

import aiohttp

import config
import market
import edge as edge_mod
import executor
import logger as trade_logger
import risk as risk_mod
import strategy
from data_feed import BinanceFeed
from db import BotDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_price(p: float) -> str:
    return f"${p:,.2f}"


def _in_price_range(up_price: float, down_price: float) -> bool:
    return (config.MIN_TOKEN_PRICE <= up_price <= config.MAX_TOKEN_PRICE
            and config.MIN_TOKEN_PRICE <= down_price <= config.MAX_TOKEN_PRICE)


def _print_session_summary(db: BotDatabase):
    s = db.get_session_summary()
    decided = s.wins + s.losses
    print()
    print(f"  ── Session Summary ({'─' * 38})")
    print(f"  Windows: {s.windows_observed} observed, {s.windows_resolved} resolved, {s.windows_traded} with trade decisions")
    print(f"  Trades: {s.total_trades} ({s.wins}W / {s.losses}L / {s.pending}P)"
          f" | Skipped: {s.skipped}")
    if decided > 0:
        print(f"  Win rate: {s.win_rate:.0%} ({s.wins}/{decided})"
              f" | P&L: ${s.total_pnl:+.2f}"
              f" | ROI: {s.roi:+.1%}")
    elif s.pending > 0:
        print(f"  P&L: pending resolution | Wagered: ${s.total_wagered:.2f}")
    print(f"  {'─' * 56}")


def _print_signal_breakdown(sig: strategy.Signal, strategy_mode: str):
    """Print detailed signal component breakdown for debugging."""
    b = sig.breakdown
    if b is None:
        return

    print(f"  ── Signal Breakdown ──")
    print(f"    Trend  ({config.TREND_WEIGHT:.0%}): EMA={b.trend_ema_signal:+.2f} PrevWin={b.prior_window_signal:+.2f} -> {b.trend_combined:+.2f} (w={b.trend_weighted:+.3f})")
    print(f"    Micro  ({config.MICRO_WEIGHT:.0%}): RSI={b.rsi_signal:+.2f} EMA={b.ema_cross_signal:+.2f} Mom={b.momentum_signal:+.2f} Vol={b.vol_spike_signal:+.2f} -> {b.micro_combined:+.2f} (w={b.micro_weighted:+.3f})")
    print(f"    OB     ({config.ORDERBOOK_WEIGHT:.0%}): {b.ob_signal:+.2f} (w={b.ob_weighted:+.3f})")
    print(f"    VolAdj ({config.VOLATILITY_WEIGHT:.0%}): regime={b.vol_regime:.2f} multiplier={b.vol_adjustment:.2f}")
    print(f"    Final: ({b.trend_weighted:+.3f} + {b.micro_weighted:+.3f} + {b.ob_weighted:+.3f}) * {b.vol_adjustment:.2f} = {sig.score:+.3f}")


def _print_lmsr_signal(sig: strategy.LMSRSignal):
    """Print LMSR signal details."""
    print(f"  ── LMSR Signal ──")
    print(f"    Velocity: {sig.velocity:+.6f} $/sec")
    print(f"    Acceleration: {sig.acceleration:+.6f} $/sec²")
    print(f"    Snapshots: {sig.snapshots_used}")
    print(f"    Market: Up=${sig.up_price:.3f}, Down=${sig.down_price:.3f}, Sum=${sig.up_price + sig.down_price:.3f}")
    print(f"    Direction: {sig.direction} (conf={sig.confidence:.2f}, prob_up={sig.model_prob:.2f})")


def _print_selective_signal(sig: strategy.SelectiveSignal):
    """Print selective signal details."""
    print(f"  ── Selective Signal ──")
    print(f"    Reversal: {sig.reversal_signal:+.2f} (prior window → expect opposite)")
    print(f"    LMSR velocity: {sig.velocity:+.6f} $/sec")
    print(f"    LMSR acceleration: {sig.acceleration:+.6f} $/sec²")
    print(f"    Market: Up=${sig.up_price:.3f}, Down=${sig.down_price:.3f}")
    print(f"    Direction: {sig.direction} (conf={sig.confidence:.2f}, prob_up={sig.model_prob:.2f})")


def _print_velocity_threshold_analysis(db: BotDatabase):
    print(f"\n  ── Velocity Threshold Analysis ──")
    db.get_velocity_threshold_analysis()


def _print_momentum_analysis(db: BotDatabase):
    stats = db.get_momentum_stats()
    print(f"\n  ── Momentum Continuation Analysis ──")
    print(f"    Resolved windows: {stats.total_resolved}")
    print(f"    With prior data: {stats.with_prior}")
    if stats.with_prior > 0:
        print(f"    Continuations: {stats.continuations} ({stats.continuation_rate:.0%})")
        print(f"    Reversals: {stats.reversals} ({1 - stats.continuation_rate:.0%})")
    else:
        print(f"    (Need more resolved windows for analysis)")


def _print_lmsr_velocity_analysis(db: BotDatabase):
    stats = db.get_lmsr_velocity_stats()
    print(f"\n  ── LMSR Velocity Analysis ──")
    print(f"    Total snapshots: {stats.total_snapshots}")
    print(f"    Windows with snapshots: {stats.windows_with_snapshots}")
    print(f"    Avg |velocity|: {stats.avg_velocity:.6f} $/sec")
    if stats.high_velocity_windows > 0:
        print(f"    High velocity windows: {stats.high_velocity_windows} ({stats.high_vel_accuracy:.0%} direction correct)")
    if stats.low_velocity_windows > 0:
        print(f"    Low velocity windows: {stats.low_velocity_windows} (no signal)")


# ── Outcome resolution ────────────────────────────────────────────────────

async def _resolve_and_update(feed: BinanceFeed, risk_mgr: risk_mod.RiskManager, db: BotDatabase) -> None:
    resolved = await trade_logger.resolve_pending_windows(feed)
    if not resolved:
        return

    for r in resolved:
        risk_mgr.record_trade(r.pnl)
        log.info("  Resolved %s: %s → %s ($%+.2f) | BTC %.2f→%.2f",
                 r.slug, r.side, r.outcome, r.pnl, r.btc_open, r.btc_close)

    print(f"\n  Resolved {len(resolved)} trade(s):")
    for r in resolved:
        delta = r.btc_close - r.btc_open
        winner = "Up" if r.btc_close >= r.btc_open else "Down"
        print(f"    {r.slug}: BTC {'+' if delta >= 0 else ''}{delta:.2f} → {winner} wins"
              f" | You: {r.side} → {r.outcome} (${r.pnl:+.2f})")

    # Export CSV after resolution
    db.export_trades_csv()


# ── Market snapshot collection ───────────────────────────────────────────

async def _collect_market_snapshot(
    window: market.Window,
    feed: BinanceFeed,
    db: BotDatabase,
    prev_snapshot: dict | None = None,
) -> dict | None:
    """Fetch market prices and record a snapshot. Returns snapshot dict or None."""
    try:
        ev_loop = asyncio.get_running_loop()
        up_price, down_price = await ev_loop.run_in_executor(
            None, executor.get_market_prices,
            window.up_token_id, window.down_token_id,
        )
    except Exception as exc:
        log.debug("Could not fetch market prices for snapshot: %s", exc)
        return None

    btc_price = feed.last_price
    btc_1s_return = None
    if btc_price and len(feed.ticks) >= 2:
        prev_price = feed.ticks[-2].close
        if prev_price > 0:
            btc_1s_return = (btc_price - prev_price) / prev_price

    ob = await feed.get_order_book()
    ob_bid = ob.bid_volume if ob else None
    ob_ask = ob.ask_volume if ob else None
    ob_imb = ob.imbalance if ob else None

    seconds_into = time.time() - window.timestamp

    # Compute velocity from previous snapshot
    velocity = None
    acceleration = None
    if prev_snapshot and prev_snapshot.get("up_price") is not None:
        dt = seconds_into - prev_snapshot["seconds_into_window"]
        if dt > 0:
            velocity = (up_price - prev_snapshot["up_price"]) / dt
            if prev_snapshot.get("velocity") is not None:
                acceleration = (velocity - prev_snapshot["velocity"]) / dt

    snapshot = {
        "window_ts": window.timestamp,
        "up_price": up_price,
        "down_price": down_price,
        "seconds_into_window": seconds_into,
        "btc_price": btc_price,
        "velocity": velocity,
        "acceleration": acceleration,
    }

    db.record_market_snapshot(
        window_ts=window.timestamp,
        up_price=up_price,
        down_price=down_price,
        btc_price=btc_price,
        btc_1s_return=btc_1s_return,
        ob_bid_volume=ob_bid,
        ob_ask_volume=ob_ask,
        ob_imbalance=ob_imb,
        velocity=velocity,
        acceleration=acceleration,
    )

    # Log arb opportunity
    total = up_price + down_price
    if total < 0.98:
        log.info("  ⚡ Up+Down=%.3f (< 0.98) — potential arb at T+%.0fs", total, seconds_into)

    return snapshot


# ── TA loop (early/late) ─────────────────────────────────────────────────

async def _ta_loop(
    window: market.Window,
    feed: BinanceFeed,
    deadline: float,
    strategy_mode: str,
    db: BotDatabase,
) -> tuple:
    """Run the TA + market-price loop until deadline.
    Returns (best_signal, best_edge_result) or (None, None).
    """
    best_signal = None
    best_edge_result = None
    best_edge_val = -999.0
    iteration = 0
    price_range_skip_count = 0
    confidence_skip_count = 0
    prev_snapshot = None

    while time.time() < deadline:
        iteration += 1

        if not feed.is_fresh:
            log.warning("  [iter %d] Stale data (%.1fs old) — skipping", iteration, feed.data_age)
            await asyncio.sleep(config.LOOP_INTERVAL)
            continue

        prices = feed.prices()
        volumes = feed.volumes()

        # Compute signal
        if strategy_mode == "early":
            all_prices = feed.prices()
            prior_delta = feed.get_prior_window_delta()
            trend = strategy.compute_trend_context(all_prices, prior_delta)

            ob = await feed.get_order_book()
            ob_imbalance = ob.imbalance if ob else 0.0

            window_start = window.timestamp
            recent_prices = feed.get_prices_since(window_start - 30)
            if len(recent_prices) < 8:
                recent_prices = prices[-30:] if len(prices) >= 8 else prices

            sig = strategy.compute_signal_early(
                recent_prices, volumes, ob_imbalance, trend
            )
        else:
            sig = strategy.compute_signal(prices, volumes)

        if sig is None:
            log.warning("  [iter %d] Insufficient data for signal", iteration)
            await asyncio.sleep(config.LOOP_INTERVAL)
            continue

        # Confidence filter
        if sig.confidence < config.MIN_CONFIDENCE:
            confidence_skip_count += 1
            elapsed = time.time() - window.timestamp
            log.info("  [iter %d | T+%.0fs] score=%+.2f conf=%.2f < %.2f — low confidence",
                     iteration, elapsed, sig.score, sig.confidence, config.MIN_CONFIDENCE)
            # Still collect market snapshot even when skipping
            prev_snapshot = await _collect_market_snapshot(window, feed, db, prev_snapshot)
            await asyncio.sleep(config.LOOP_INTERVAL)
            continue

        # Get market prices
        try:
            ev_loop = asyncio.get_running_loop()
            up_price, down_price = await ev_loop.run_in_executor(
                None, executor.get_market_prices,
                window.up_token_id, window.down_token_id,
            )
        except Exception as exc:
            log.warning("  [iter %d] Could not fetch market prices: %s", iteration, exc)
            await asyncio.sleep(config.LOOP_INTERVAL)
            continue

        # Record market snapshot
        seconds_into = time.time() - window.timestamp
        velocity = None
        if prev_snapshot and prev_snapshot.get("up_price") is not None:
            dt = seconds_into - prev_snapshot["seconds_into_window"]
            if dt > 0:
                velocity = (up_price - prev_snapshot["up_price"]) / dt

        prev_snapshot = {
            "window_ts": window.timestamp,
            "up_price": up_price,
            "down_price": down_price,
            "seconds_into_window": seconds_into,
            "velocity": velocity,
        }

        btc_now = feed.last_price
        btc_1s_return = None
        if btc_now and len(feed.ticks) >= 2:
            btc_1s_return = (btc_now - feed.ticks[-2].close) / feed.ticks[-2].close

        ob_snap = await feed.get_order_book()
        db.record_market_snapshot(
            window_ts=window.timestamp,
            up_price=up_price,
            down_price=down_price,
            btc_price=btc_now,
            btc_1s_return=btc_1s_return,
            ob_bid_volume=ob_snap.bid_volume if ob_snap else None,
            ob_ask_volume=ob_snap.ask_volume if ob_snap else None,
            ob_imbalance=ob_snap.imbalance if ob_snap else None,
            velocity=velocity,
        )

        er = edge_mod.compute_edge(
            model_prob_up=sig.model_prob,
            market_price_up=up_price,
            market_price_down=down_price,
            up_token_id=window.up_token_id,
            down_token_id=window.down_token_id,
        )

        elapsed = time.time() - window.timestamp
        log.info(
            "  [iter %d | T+%.0fs] score=%+.2f conf=%.2f prob_up=%.2f | mkt Up=$%.3f Down=$%.3f | edge=%s %.1f%%",
            iteration, elapsed, sig.score, sig.confidence, sig.model_prob,
            up_price, down_price, er.side, er.edge_pct,
        )

        # Price range gate (early only)
        if strategy_mode == "early" and not _in_price_range(up_price, down_price):
            price_range_skip_count += 1
            log.info("  [iter %d] Prices outside range ($%.2f–$%.2f)",
                     iteration, config.MIN_TOKEN_PRICE, config.MAX_TOKEN_PRICE)
            await asyncio.sleep(config.LOOP_INTERVAL)
            continue

        if er.edge > best_edge_val:
            best_edge_val = er.edge
            best_signal = sig
            best_edge_result = er

        remaining = deadline - time.time()
        if remaining > config.LOOP_INTERVAL:
            await asyncio.sleep(config.LOOP_INTERVAL)
        else:
            break

    if best_signal is None:
        if price_range_skip_count > 0:
            log.info("  %d iterations outside price range", price_range_skip_count)
        if confidence_skip_count > 0:
            log.info("  %d iterations below confidence threshold", confidence_skip_count)

    return best_signal, best_edge_result


# ── LMSR strategy loop ──────────────────────────────────────────────────

async def _lmsr_loop(
    window: market.Window,
    feed: BinanceFeed,
    db: BotDatabase,
) -> tuple:
    """Collect snapshots from T+5s, then evaluate LMSR signal at T+15s–T+25s.
    Returns (lmsr_signal, edge_result) or (None, None).
    """
    collect_start = window.timestamp + config.LMSR_COLLECT_START
    entry_start = window.timestamp + config.LMSR_ENTRY_START
    entry_end = window.timestamp + config.LMSR_ENTRY_END

    # Wait until collection start
    now = time.time()
    if now < collect_start:
        await asyncio.sleep(collect_start - now)

    prev_snapshot = None
    snapshots = []  # local list for signal computation
    best_signal = None
    best_edge = None
    best_edge_val = -999.0

    while time.time() < entry_end:
        snap = await _collect_market_snapshot(window, feed, db, prev_snapshot)
        if snap:
            prev_snapshot = snap
            snapshots.append(snap)

        now = time.time()
        if now >= entry_start and len(snapshots) >= config.LMSR_MIN_SNAPSHOTS:
            # Build snapshot dicts for signal computation
            snap_dicts = db.get_recent_snapshots(window.timestamp)
            sig = strategy.compute_signal_lmsr(snap_dicts)

            if sig is not None:
                # Price gate for LMSR
                if sig.up_price > config.LMSR_MAX_PRICE and sig.direction == "UP":
                    log.info("  LMSR: Up price $%.3f > $%.2f — too expensive", sig.up_price, config.LMSR_MAX_PRICE)
                elif sig.down_price > config.LMSR_MAX_PRICE and sig.direction == "DOWN":
                    log.info("  LMSR: Down price $%.3f > $%.2f — too expensive", sig.down_price, config.LMSR_MAX_PRICE)
                else:
                    er = edge_mod.compute_edge(
                        model_prob_up=sig.model_prob,
                        market_price_up=sig.up_price,
                        market_price_down=sig.down_price,
                        up_token_id=window.up_token_id,
                        down_token_id=window.down_token_id,
                    )
                    log.info("  LMSR: vel=%+.6f accel=%+.6f dir=%s conf=%.2f | edge=%s %.1f%%",
                             sig.velocity, sig.acceleration, sig.direction, sig.confidence,
                             er.side, er.edge_pct)

                    if er.edge > best_edge_val:
                        best_edge_val = er.edge
                        best_signal = sig
                        best_edge = er

        await asyncio.sleep(config.LMSR_SNAPSHOT_INTERVAL)

    return best_signal, best_edge


# ── Selective strategy loop ───────────────────────────────────────────────

async def _selective_loop(
    window: market.Window,
    feed: BinanceFeed,
    db: BotDatabase,
) -> tuple:
    """Selective strategy: collect LMSR snapshots, then check reversal + velocity confluence.
    Returns (selective_signal, edge_result) or (None, None).
    Collects snapshots for the full T+5s to T+25s window before deciding.
    """
    collect_start = window.timestamp + config.LMSR_COLLECT_START
    entry_start = window.timestamp + config.LMSR_ENTRY_START
    entry_end = window.timestamp + config.LMSR_ENTRY_END

    # Get prior window delta from feed
    prior_delta = feed.get_prior_window_delta()
    prior_btc_price = None
    if prior_delta is not None and feed.last_price:
        prior_btc_price = feed.last_price

    # Wait until collection start
    now = time.time()
    if now < collect_start:
        await asyncio.sleep(collect_start - now)

    prev_snapshot = None
    best_signal = None
    best_edge = None
    best_edge_val = -999.0
    last_skip_reason = None

    # Collect snapshots for the full window — don't bail early on skip
    while time.time() < entry_end:
        snap = await _collect_market_snapshot(window, feed, db, prev_snapshot)
        if snap:
            prev_snapshot = snap

        now = time.time()
        if now >= entry_start:
            snap_dicts = db.get_recent_snapshots(window.timestamp)
            sig = strategy.compute_signal_selective(prior_delta, prior_btc_price, snap_dicts)

            if sig is None:
                log.info("  Selective: not enough snapshots yet (%d)", len(snap_dicts))
            elif sig.skip_reason:
                last_skip_reason = sig.skip_reason
                log.info("  Selective: %s", sig.skip_reason)
                # DON'T return early — keep collecting, velocity may increase
            else:
                # Price gate
                buy_side_price = sig.up_price if sig.direction == "UP" else sig.down_price
                if buy_side_price > config.LMSR_MAX_PRICE:
                    log.info("  Selective: %s price $%.3f > $%.2f — too expensive",
                             sig.direction, buy_side_price, config.LMSR_MAX_PRICE)
                else:
                    er = edge_mod.compute_edge(
                        model_prob_up=sig.model_prob,
                        market_price_up=sig.up_price,
                        market_price_down=sig.down_price,
                        up_token_id=window.up_token_id,
                        down_token_id=window.down_token_id,
                    )
                    log.info("  Selective: rev=%+.2f vel=%+.6f dir=%s conf=%.2f | edge=%s %.1f%%",
                             sig.reversal_signal, sig.velocity, sig.direction,
                             sig.confidence, er.side, er.edge_pct)

                    if er.edge > best_edge_val:
                        best_edge_val = er.edge
                        best_signal = sig
                        best_edge = er

        await asyncio.sleep(config.LMSR_SNAPSHOT_INTERVAL)

    # If we found a tradeable signal at any point, return it
    if best_signal is not None:
        return best_signal, best_edge

    # Otherwise return a skip signal with the last reason
    if last_skip_reason and sig is not None:
        return sig, None

    return None, None


# ── Window handler ────────────────────────────────────────────────────────

async def run_window(
    window: market.Window,
    feed: BinanceFeed,
    risk_mgr: risk_mod.RiskManager,
    session: aiohttp.ClientSession,
    dry_run: bool,
    strategy_mode: str,
    db: BotDatabase,
) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("[%s] Window %s  [strategy=%s]", utc_now_str(), window.slug, strategy_mode)
    log.info("=" * 60)

    if not await market.fetch_token_ids(window, session):
        log.error("Skipping window — could not fetch token IDs")
        return

    btc_open = feed.last_price
    if btc_open is None:
        log.error("Skipping window — no BTC price data")
        return

    # Record window observation (every window, whether we trade or not)
    db.record_window(window.timestamp, window.slug, btc_open)

    # Print trend context at window start
    if strategy_mode in ("early", "lmsr", "selective"):
        prior_delta = feed.get_prior_window_delta()
        trend = strategy.compute_trend_context(feed.prices(), prior_delta)
        if trend:
            log.info("  Trend: EMA60/180=%+.2f | PrevWindow=%+.2f ($%+.0f) | VolRegime=%.2f",
                     trend.trend_signal, trend.prior_window_signal,
                     trend.prior_window_delta, trend.vol_regime)

    # Route to correct strategy loop
    if strategy_mode == "lmsr":
        lmsr_sig, best_edge_result = await _lmsr_loop(window, feed, db)

        if lmsr_sig is None or best_edge_result is None:
            log.info("  No LMSR signal produced — skipping window")
            return

        _print_lmsr_signal(lmsr_sig)

        er = best_edge_result
        if not er.is_tradeable:
            print(f"  NO TRADE — edge {er.edge_pct:.1f}% < threshold {config.EDGE_THRESHOLD * 100:.0f}%")
            db.record_trade(
                window_ts=window.timestamp, strategy="lmsr", side=er.side,
                buy_price=er.buy_price, shares=0, amount=0,
                signal_score=lmsr_sig.velocity, signal_confidence=lmsr_sig.confidence,
                model_prob=lmsr_sig.model_prob, edge_pct=er.edge_pct,
                up_price_at_entry=lmsr_sig.up_price, down_price_at_entry=lmsr_sig.down_price,
                price_velocity_at_entry=lmsr_sig.velocity,
                order_type="SKIP", dry_run=dry_run, outcome="SKIPPED",
            )
            _print_session_summary(db)
            return

        return await _execute_trade(
            window, feed, risk_mgr, er, dry_run, db,
            strategy_name="lmsr",
            signal_score=lmsr_sig.velocity,
            signal_confidence=lmsr_sig.confidence,
            model_prob=lmsr_sig.model_prob,
            up_price_at_entry=lmsr_sig.up_price,
            down_price_at_entry=lmsr_sig.down_price,
            price_velocity_at_entry=lmsr_sig.velocity,
        )

    # Selective strategy
    if strategy_mode == "selective":
        sel_sig, best_edge_result = await _selective_loop(window, feed, db)

        if sel_sig is None:
            log.info("  No selective signal produced — skipping window")
            return

        if sel_sig.skip_reason:
            log.info("  Selective skip: %s", sel_sig.skip_reason)
            return

        if best_edge_result is None:
            log.info("  Selective: no tradeable edge found — skipping window")
            return

        _print_selective_signal(sel_sig)

        er = best_edge_result
        if not er.is_tradeable:
            print(f"  NO TRADE — edge {er.edge_pct:.1f}% < threshold {config.EDGE_THRESHOLD * 100:.0f}%")
            db.record_trade(
                window_ts=window.timestamp, strategy="selective", side=er.side,
                buy_price=er.buy_price, shares=0, amount=0,
                signal_score=sel_sig.reversal_signal, signal_confidence=sel_sig.confidence,
                model_prob=sel_sig.model_prob, edge_pct=er.edge_pct,
                up_price_at_entry=sel_sig.up_price, down_price_at_entry=sel_sig.down_price,
                price_velocity_at_entry=sel_sig.velocity,
                order_type="SKIP", dry_run=dry_run, outcome="SKIPPED",
            )
            _print_session_summary(db)
            return

        return await _execute_trade(
            window, feed, risk_mgr, er, dry_run, db,
            strategy_name="selective",
            signal_score=sel_sig.reversal_signal,
            signal_confidence=sel_sig.confidence,
            model_prob=sel_sig.model_prob,
            up_price_at_entry=sel_sig.up_price,
            down_price_at_entry=sel_sig.down_price,
            price_velocity_at_entry=sel_sig.velocity,
        )

    # Early / late strategies
    if strategy_mode == "early":
        wait_until = window.timestamp + config.ENTRY_WINDOW_START
        deadline = window.timestamp + config.ENTRY_WINDOW_END
        label = f"T+{config.ENTRY_WINDOW_START}s"
    else:
        wait_until = window.close_time - config.LEAD_TIME
        deadline = window.close_time - config.HARD_DEADLINE
        label = f"T-{config.LEAD_TIME}s"

    now = time.time()
    if now < wait_until:
        sleep_secs = wait_until - now
        log.info("  Sleeping %.1fs until %s ...", sleep_secs, label)
        await asyncio.sleep(sleep_secs)

    best_signal, best_edge_result = await _ta_loop(window, feed, deadline, strategy_mode, db)

    if best_signal is None or best_edge_result is None:
        log.info("  No valid signal produced — skipping window")
        return

    btc_now = feed.last_price or btc_open
    btc_delta = btc_now - btc_open
    sig = best_signal
    er = best_edge_result

    # Pretty console output
    print()
    print(f"  BTC: {format_price(btc_open)} -> {format_price(btc_now)} "
          f"({'+' if btc_delta >= 0 else ''}{format_price(btc_delta).replace('$', '$')})")
    ema_label = "EMA3>EMA7" if strategy_mode == "early" else "EMA9>EMA21"
    rsi_period = 7 if strategy_mode == "early" else 14
    print(f"  RSI({rsi_period}): {sig.rsi:.1f} | {ema_label}: {sig.ema_fast > sig.ema_slow} | Momentum: {sig.momentum:+.4f}")
    if strategy_mode == "early" and sig.ob_imbalance != 0:
        print(f"  OB Imbalance: {sig.ob_imbalance:+.2f}")

    _print_signal_breakdown(sig, strategy_mode)

    print(f"  Signal: {sig.direction} (score: {sig.score:+.2f}, confidence: {sig.confidence:.2f}, prob_up: {sig.model_prob:.2f})")
    print(f"  Market: Up=${er.market_price_up:.3f}, Down=${er.market_price_down:.3f}")
    print(f"  Edge: {er.edge_pct:.1f}% (model: {sig.model_prob:.2f} vs market: {er.buy_price:.3f})")

    if not er.is_tradeable:
        print(f"  NO TRADE — edge {er.edge_pct:.1f}% < threshold {config.EDGE_THRESHOLD * 100:.0f}%")
        db.record_trade(
            window_ts=window.timestamp, strategy=strategy_mode, side=er.side,
            buy_price=er.buy_price, shares=0, amount=0,
            signal_score=sig.score, signal_confidence=sig.confidence,
            model_prob=sig.model_prob, edge_pct=er.edge_pct,
            up_price_at_entry=er.market_price_up, down_price_at_entry=er.market_price_down,
            order_type="SKIP", dry_run=dry_run, outcome="SKIPPED",
        )
        _print_session_summary(db)
        return

    return await _execute_trade(
        window, feed, risk_mgr, er, dry_run, db,
        strategy_name=strategy_mode,
        signal_score=sig.score,
        signal_confidence=sig.confidence,
        model_prob=sig.model_prob,
        up_price_at_entry=er.market_price_up,
        down_price_at_entry=er.market_price_down,
    )


async def _execute_trade(
    window: market.Window,
    feed: BinanceFeed,
    risk_mgr: risk_mod.RiskManager,
    er: edge_mod.EdgeResult,
    dry_run: bool,
    db: BotDatabase,
    strategy_name: str,
    signal_score: float,
    signal_confidence: float,
    model_prob: float,
    up_price_at_entry: float = 0.0,
    down_price_at_entry: float = 0.0,
    price_velocity_at_entry: float | None = None,
) -> None:
    """Execute trade and record to database."""
    if not risk_mgr.can_trade:
        print("  NO TRADE — risk limit reached")
        return

    amount = risk_mgr.size_position(er.edge, er.buy_price)
    if amount <= 0:
        print("  NO TRADE — Kelly sizing too small")
        return

    shares = amount / er.buy_price if er.buy_price > 0 else 0
    shares = math.floor(shares * 10) / 10

    print(f"  ACTION: BUY {er.side} @ ${er.buy_price:.3f} x {shares:.1f} shares = ${amount:.2f}")

    if dry_run:
        print("  [DRY RUN — no order placed]")
        order_id = None
        order_type = "DRY_RUN"
    else:
        order_result = executor.place_fok_order(er.token_id, amount, er.side)
        if order_result.success:
            order_id = order_result.order_id
            order_type = "FOK"
            print(f"  ORDER PLACED: FOK {order_id}")
        else:
            print(f"  FOK failed ({order_result.error}) — trying $0.95 limit fallback")
            order_result = executor.place_limit_fallback(er.token_id, amount, er.side)
            order_id = order_result.order_id if order_result.success else None
            order_type = "GTC_LIMIT" if order_result.success else "FAILED"
            if order_result.success:
                print(f"  LIMIT ORDER PLACED: {order_id}")
            else:
                print(f"  ORDER FAILED: {order_result.error}")

    print(f"  Outcome: PENDING (window closes in {window.close_time - time.time():.0f}s)")

    db.record_trade(
        window_ts=window.timestamp,
        strategy=strategy_name,
        side=er.side,
        buy_price=er.buy_price,
        shares=shares,
        amount=amount,
        signal_score=signal_score,
        signal_confidence=signal_confidence,
        model_prob=model_prob,
        edge_pct=er.edge_pct,
        up_price_at_entry=up_price_at_entry,
        down_price_at_entry=down_price_at_entry,
        price_velocity_at_entry=price_velocity_at_entry,
        order_type=order_type,
        order_id=order_id,
        dry_run=dry_run,
        outcome="PENDING",
    )

    _print_session_summary(db)


# ── Main loop ─────────────────────────────────────────────────────────────

WARMUP_TICKS = 60


async def main_loop(dry_run: bool, once: bool, strategy_mode: str, max_windows: int) -> None:
    db = BotDatabase()
    trade_logger.set_db(db)
    db.start_session(strategy_mode, dry_run)

    mode_label = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'=' * 60}")
    print(f"  Polymarket BTC Up/Down Bot — {mode_label}")
    print(f"  Strategy: {strategy_mode}")
    print(f"  Edge threshold: {config.EDGE_THRESHOLD * 100:.0f}%")
    print(f"  Min confidence: {config.MIN_CONFIDENCE:.0%}")
    print(f"  Bet amount: ${config.BET_AMOUNT:.2f}")
    print(f"  Max daily loss: ${config.MAX_DAILY_LOSS:.2f}")
    if strategy_mode == "early":
        print(f"  Entry window: T+{config.ENTRY_WINDOW_START}s to T+{config.ENTRY_WINDOW_END}s")
        print(f"  Price range gate: ${config.MIN_TOKEN_PRICE:.2f}–${config.MAX_TOKEN_PRICE:.2f}")
        print(f"  Weights: trend={config.TREND_WEIGHT:.0%} micro={config.MICRO_WEIGHT:.0%} "
              f"ob={config.ORDERBOOK_WEIGHT:.0%} vol={config.VOLATILITY_WEIGHT:.0%}")
    elif strategy_mode == "lmsr":
        print(f"  LMSR collection: T+{config.LMSR_COLLECT_START}s")
        print(f"  LMSR entry: T+{config.LMSR_ENTRY_START}s to T+{config.LMSR_ENTRY_END}s")
        print(f"  LMSR velocity threshold: {config.LMSR_VELOCITY_THRESHOLD} $/sec")
        print(f"  LMSR max price: ${config.LMSR_MAX_PRICE:.2f}")
    elif strategy_mode == "selective":
        print(f"  Selective: reversal + LMSR confluence")
        print(f"  LMSR collection: T+{config.LMSR_COLLECT_START}s, entry: T+{config.LMSR_ENTRY_START}s–T+{config.LMSR_ENTRY_END}s")
        print(f"  Velocity threshold: {config.LMSR_VELOCITY_THRESHOLD} $/sec")
        print(f"  Min prior move: {config.SELECTIVE_MIN_PRIOR_MOVE:.4%}")
        print(f"  Max price: ${config.LMSR_MAX_PRICE:.2f}")
    else:
        print(f"  Entry window: T-{config.LEAD_TIME}s to T-{config.HARD_DEADLINE}s")
    if max_windows > 0:
        print(f"  Max windows: {max_windows}")
    print(f"  Database: {db.path}")
    print(f"{'=' * 60}\n")

    feed = BinanceFeed()
    risk_mgr = risk_mod.RiskManager()

    async with aiohttp.ClientSession() as session:
        await feed.start()
        log.info("Waiting for initial price data...")
        for _ in range(30):
            if feed.last_price is not None:
                break
            await asyncio.sleep(1)
        else:
            log.error("Could not get initial BTC price — exiting")
            await feed.stop()
            db.end_session()
            db.close()
            return

        log.info("BTC price: %s — feed is live (%d ticks buffered)",
                 format_price(feed.last_price), len(feed.ticks))

        # Warmup
        if strategy_mode not in ("lmsr", "selective") and len(feed.ticks) < WARMUP_TICKS:
            log.info("Warming up — need %d ticks for trend context, have %d. Waiting ~%ds...",
                     WARMUP_TICKS, len(feed.ticks), WARMUP_TICKS - len(feed.ticks))
            while len(feed.ticks) < WARMUP_TICKS and feed._running:
                await asyncio.sleep(1)
            log.info("Warmup complete — %d ticks buffered", len(feed.ticks))

        try:
            windows_processed = 0
            last_processed_ts = 0  # prevent re-entering same window
            while True:
                await _resolve_and_update(feed, risk_mgr, db)

                now_ts = time.time()
                win_ts = market.current_window_ts()
                win = market.make_window(win_ts)

                if strategy_mode == "early":
                    entry_deadline = win.timestamp + config.ENTRY_WINDOW_END
                    if now_ts > entry_deadline or win.timestamp == last_processed_ts:
                        win_ts = market.next_window_ts()
                        win = market.make_window(win_ts)
                    sleep_until = max(win.timestamp - 5, now_ts + 0.5)
                elif strategy_mode in ("lmsr", "selective"):
                    entry_deadline = win.timestamp + config.LMSR_ENTRY_END
                    if now_ts > entry_deadline or win.timestamp == last_processed_ts:
                        win_ts = market.next_window_ts()
                        win = market.make_window(win_ts)
                    sleep_until = max(win.timestamp - 5, now_ts + 0.5)
                else:
                    if now_ts > win.close_time - config.HARD_DEADLINE or win.timestamp == last_processed_ts:
                        win_ts = market.next_window_ts()
                        win = market.make_window(win_ts)
                    sleep_until = max(win.close_time - config.LEAD_TIME - 30, now_ts + 0.5)

                wait_secs = sleep_until - time.time()
                if wait_secs > 1:
                    win_open = datetime.fromtimestamp(win.timestamp, tz=timezone.utc)
                    win_close = datetime.fromtimestamp(win.close_time, tz=timezone.utc)
                    log.info("Next window: %s (%s–%s) — waiting %.0fs (%d ticks buffered)",
                             win.slug,
                             win_open.strftime("%H:%M:%S"),
                             win_close.strftime("%H:%M:%S"),
                             wait_secs, len(feed.ticks))
                    await asyncio.sleep(wait_secs)

                risk_mgr.reset_window()
                await run_window(win, feed, risk_mgr, session, dry_run, strategy_mode, db)
                last_processed_ts = win.timestamp
                windows_processed += 1

                if once:
                    log.info("--once flag set — waiting for window to close for resolution...")
                    remaining = win.close_time + 10 - time.time()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await _resolve_and_update(feed, risk_mgr, db)
                    break

                if max_windows > 0 and windows_processed >= max_windows:
                    log.info("Reached --max-windows %d — waiting for last window to close...",
                             max_windows)
                    remaining = win.close_time + 10 - time.time()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await _resolve_and_update(feed, risk_mgr, db)
                    break

                await asyncio.sleep(2)

        except KeyboardInterrupt:
            log.info("Shutting down — resolving pending trades...")
            await _resolve_and_update(feed, risk_mgr, db)
        finally:
            await feed.stop()

    # Final output
    print(f"\n{'=' * 60}")
    print(f"  FINAL SESSION SUMMARY")
    _print_session_summary(db)
    print(f"  Risk manager P&L: ${risk_mgr.daily_pnl:+.2f}")

    _print_momentum_analysis(db)
    _print_lmsr_velocity_analysis(db)
    _print_velocity_threshold_analysis(db)

    # Print skip analysis
    summary = db.get_session_summary()
    print(f"\n  ── Window Analysis ──")
    print(f"    Observed: {summary.windows_observed}")
    print(f"    Traded: {summary.total_trades}")
    print(f"    Skipped (no edge): {summary.skipped}")
    print(f"    Skipped (no signal): {summary.windows_observed - summary.windows_traded}")

    print(f"{'=' * 60}")

    # Export CSV and close db
    db.export_trades_csv()
    db.end_session()
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC Up/Down 5-min trading bot")
    parser.add_argument("--dry-run", action="store_true", help="Run without placing real orders")
    parser.add_argument("--once", action="store_true", help="Process one window then exit")
    parser.add_argument("--strategy", choices=["early", "late", "lmsr", "selective"], default="early",
                        help="Strategy: 'early', 'late', 'lmsr' (price velocity), or 'selective' (reversal+LMSR)")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="Stop after N windows (0 = unlimited)")
    args = parser.parse_args()

    if not args.dry_run and not config.POLYMARKET_PRIVATE_KEY:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env — use --dry-run or configure credentials")
        sys.exit(1)

    try:
        asyncio.run(main_loop(
            dry_run=args.dry_run, once=args.once,
            strategy_mode=args.strategy, max_windows=args.max_windows,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
