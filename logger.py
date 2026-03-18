"""Trade logging — thin wrapper over db.py.

The database is the source of truth. CSV export is available via db.export_trades_csv().
Legacy resolve_pending_trades and get_session_summary delegate to BotDatabase.
"""

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Global database reference, set by main.py at startup
_db = None


def set_db(db):
    """Set the global database reference. Called once at startup."""
    global _db
    _db = db


def get_db():
    return _db


def _parse_window_ts(slug: str) -> int | None:
    """Extract Unix timestamp from slug like 'btc-updown-5m-1773703200'."""
    m = re.search(r"(\d{10,})$", slug)
    return int(m.group(1)) if m else None


@dataclass
class ResolvedTrade:
    """Result of resolving a single pending trade."""
    slug: str
    side: str
    outcome: str
    pnl: float
    btc_open: float
    btc_close: float


async def resolve_pending_windows(feed) -> list[ResolvedTrade]:
    """Resolve ALL unresolved windows (not just traded ones).

    Fetches Binance 5m klines for closed windows, updates db.
    Returns list of resolved trades (for risk manager updates).
    """
    if _db is None:
        return []

    # Get all unresolved windows + windows with pending trades
    unresolved_ts = set(_db.get_unresolved_window_timestamps())
    pending_ts = set(_db.get_pending_trade_window_timestamps())
    all_ts = unresolved_ts | pending_ts

    if not all_ts:
        return []

    resolved_trades = []

    for window_ts in sorted(all_ts):
        kline = await feed.fetch_kline(window_ts)
        if kline is None:
            log.debug("Could not fetch kline for ts=%d — skipping", window_ts)
            continue

        btc_open, btc_close = kline
        trade_results = _db.resolve_window(window_ts, btc_open, btc_close)

        for tr in trade_results:
            resolved_trades.append(ResolvedTrade(
                slug=tr["slug"],
                side=tr["side"],
                outcome=tr["outcome"],
                pnl=tr["pnl"],
                btc_open=tr["btc_open"],
                btc_close=tr["btc_close"],
            ))

    return resolved_trades


# Legacy aliases
resolve_pending_trades = resolve_pending_windows


def get_session_summary():
    """Delegate to database."""
    if _db is None:
        from db import SessionSummary
        return SessionSummary()
    return _db.get_session_summary()


def log_trade(
    window_slug: str,
    btc_open: float,
    btc_close: float,
    rsi: float,
    ema9: float,
    ema21: float,
    momentum: float,
    score: float,
    confidence: float,
    model_prob: float,
    market_price_up: float,
    market_price_down: float,
    edge_pct: float,
    side: str,
    amount: float,
    shares: float,
    buy_price: float,
    order_type: str,
    order_id: str | None,
    outcome: str,
    pnl: float,
    daily_pnl: float,
    dry_run: bool,
):
    """Log trade to database. Keeps the same interface for backwards compatibility."""
    if _db is None:
        log.warning("Database not initialized — trade not logged")
        return

    window_ts = _parse_window_ts(window_slug)
    if window_ts is None:
        log.warning("Could not parse window_ts from slug: %s", window_slug)
        return

    _db.record_trade(
        window_ts=window_ts,
        strategy="early",  # will be overridden by caller in main.py
        side=side,
        buy_price=buy_price,
        shares=shares,
        amount=amount,
        signal_score=score,
        signal_confidence=confidence,
        model_prob=model_prob,
        edge_pct=edge_pct,
        up_price_at_entry=market_price_up,
        down_price_at_entry=market_price_down,
        order_type=order_type,
        order_id=order_id,
        dry_run=dry_run,
        outcome=outcome,
    )
