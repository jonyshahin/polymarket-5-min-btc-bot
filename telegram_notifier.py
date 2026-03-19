"""Telegram notification module for the Polymarket bot.

Sends periodic trading reports and alerts via Telegram Bot API.
Uses aiohttp (already a project dependency) for async HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp

import config
from db import BotDatabase

log = logging.getLogger(__name__)

# Telegram Bot API base URL
API_BASE = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    """Sends trading reports and alerts to Telegram."""

    def __init__(self, bot_token: str = "", chat_id: str = "") -> None:
        self._token = bot_token or config.TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self._enabled = bool(self._token and self._chat_id)
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_report_time: float = 0.0
        self._report_interval: int = config.TELEGRAM_REPORT_INTERVAL
        # Track stats since last report
        self._period_start: float = time.time()
        self._period_windows: int = 0
        self._period_trades: int = 0
        self._period_wins: int = 0
        self._period_losses: int = 0
        self._period_skips: int = 0
        self._period_pnl: float = 0.0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        """Initialize HTTP session."""
        if not self._enabled:
            log.info("Telegram notifications disabled (no token/chat_id)")
            return
        self._session = aiohttp.ClientSession()
        self._last_report_time = time.time()
        self._period_start = time.time()
        log.info("Telegram notifier ready (reports every %ds)", self._report_interval)

    async def stop(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat. Returns True on success."""
        if not self._enabled or not self._session:
            return False

        url = f"{API_BASE.format(token=self._token)}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return True
                else:
                    body = await resp.text()
                    log.warning("Telegram API error %d: %s", resp.status, body[:200])
                    return False
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)
            return False

    # ── Event tracking ───────────────────────────────────────────────

    def record_window(self) -> None:
        """Call after each window is processed."""
        self._period_windows += 1

    def record_skip(self) -> None:
        """Call when a window is skipped."""
        self._period_skips += 1

    def record_trade(self, outcome: str, pnl: float) -> None:
        """Call when a trade outcome is resolved."""
        self._period_trades += 1
        if outcome == "WIN":
            self._period_wins += 1
        elif outcome == "LOSS":
            self._period_losses += 1
        self._period_pnl += pnl

    def _reset_period(self) -> None:
        """Reset period counters."""
        self._period_start = time.time()
        self._period_windows = 0
        self._period_trades = 0
        self._period_wins = 0
        self._period_losses = 0
        self._period_skips = 0
        self._period_pnl = 0.0

    # ── Report generation ────────────────────────────────────────────

    async def check_and_send_report(self, db: BotDatabase) -> None:
        """Check if it's time to send a periodic report. Send if due."""
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_report_time < self._report_interval:
            return

        await self.send_periodic_report(db)
        self._last_report_time = now

    async def send_periodic_report(self, db: BotDatabase) -> None:
        """Build and send the periodic report."""
        msg = self._build_periodic_report(db)
        await self.send_message(msg)
        self._reset_period()

    async def send_startup_message(self, strategy: str, dry_run: bool) -> None:
        """Send notification when bot starts."""
        mode = "DRY RUN" if dry_run else "LIVE"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = (
            f"<b>Bot Started</b>\n"
            f"Strategy: <code>{strategy}</code>\n"
            f"Mode: {mode}\n"
            f"Time: {now}\n"
            f"Reports: every {self._report_interval // 3600}h"
        )
        await self.send_message(msg)

    async def send_shutdown_message(self, db: BotDatabase, reason: str = "normal") -> None:
        """Send notification when bot stops. Includes final summary."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        summary = db.get_session_summary()
        decided = summary.wins + summary.losses

        msg = f"<b>Bot Stopped</b> ({reason})\n"
        msg += f"Time: {now}\n\n"
        msg += f"<b>Session Totals:</b>\n"
        msg += f"Windows: {summary.windows_observed}\n"
        msg += f"Trades: {summary.total_trades}"
        if decided > 0:
            msg += f" ({summary.wins}W/{summary.losses}L — {summary.win_rate:.0%})\n"
            msg += f"P&L: <code>${summary.total_pnl:+.2f}</code>\n"
        else:
            msg += f" ({summary.pending} pending)\n"
        msg += f"Skipped: {summary.skipped}"

        await self.send_message(msg)

    async def send_trade_alert(self, side: str, confidence: float,
                               edge_pct: float, amount: float,
                               momentum: float, structure: float,
                               confluence: float) -> None:
        """Send alert when a trade is placed (optional — for live mode)."""
        msg = (
            f"<b>Trade Placed</b>\n"
            f"Side: <code>{side}</code>\n"
            f"Confidence: {confidence:.0%}\n"
            f"Edge: {edge_pct:.1f}%\n"
            f"Amount: ${amount:.2f}\n"
            f"Scores: M={momentum:.2f} S={structure:.2f} C={confluence:.2f}"
        )
        await self.send_message(msg)

    async def send_error_alert(self, error: str) -> None:
        """Send alert on critical errors."""
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        msg = f"<b>Bot Error</b> ({now})\n<code>{error[:500]}</code>"
        await self.send_message(msg)

    def _build_periodic_report(self, db: BotDatabase) -> str:
        """Build the hourly summary message."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Period stats (since last report)
        period_decided = self._period_wins + self._period_losses
        period_wr = self._period_wins / period_decided if period_decided > 0 else 0.0

        # Session totals
        session = db.get_session_summary()
        session_decided = session.wins + session.losses

        msg = f"<b>Hourly Report</b> — {now}\n\n"

        # This period
        msg += f"<b>Last Hour:</b>\n"
        msg += f"Windows: {self._period_windows}"
        if self._period_skips > 0:
            msg += f" ({self._period_skips} skipped)"
        msg += "\n"

        if period_decided > 0:
            msg += f"Trades: {period_decided} ({self._period_wins}W/{self._period_losses}L — {period_wr:.0%})\n"
            msg += f"P&L: <code>${self._period_pnl:+.2f}</code>\n"
        elif self._period_trades > 0:
            msg += f"Trades: {self._period_trades} (pending resolution)\n"
        else:
            msg += f"Trades: 0 (all windows skipped)\n"

        # Session totals
        msg += f"\n<b>Session Totals:</b>\n"
        msg += f"Windows: {session.windows_observed} | Trades: {session.total_trades}\n"
        if session_decided > 0:
            msg += f"Record: {session.wins}W/{session.losses}L ({session.win_rate:.0%})\n"
            msg += f"P&L: <code>${session.total_pnl:+.2f}</code>"
            if session.total_wagered > 0:
                msg += f" (ROI: {session.roi:+.1%})"
            msg += "\n"

        # SMC-specific stats (if available)
        try:
            smc_stats = db.get_smc_decision_stats()
            if smc_stats and smc_stats.get("total", 0) > 0:
                msg += f"\n<b>SMC Engine:</b>\n"
                msg += f"Decisions: {smc_stats['total']} ({smc_stats.get('bets', 0)} bets / {smc_stats.get('skips', 0)} skips)\n"
                if smc_stats.get("avg_score"):
                    msg += f"Avg Score: {smc_stats['avg_score']:.3f}\n"
                if smc_stats.get("avg_momentum"):
                    msg += f"M={smc_stats['avg_momentum']:.2f} S={smc_stats.get('avg_structure', 0):.2f} C={smc_stats.get('avg_confluence', 0):.2f}\n"
                if smc_stats.get("total_vetoed", 0) > 0:
                    msg += f"Vetoed: {smc_stats['total_vetoed']}\n"
        except Exception:
            pass  # SMC stats not available (different strategy)

        # LMSR velocity stats
        try:
            lmsr = db.get_lmsr_velocity_stats()
            if lmsr.high_velocity_windows > 0:
                msg += f"\n<b>LMSR Signal:</b>\n"
                msg += f"High vel accuracy: {lmsr.high_vel_accuracy:.0%} ({lmsr.high_vel_correct}/{lmsr.high_velocity_windows})\n"
        except Exception:
            pass

        return msg
