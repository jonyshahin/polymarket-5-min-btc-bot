"""Position sizing (fractional Kelly) and session-level risk limits."""

import logging
import math

import config

log = logging.getLogger(__name__)


class RiskManager:
    """Tracks session P&L and enforces loss limits."""

    def __init__(self):
        self.daily_pnl: float = 0.0
        self.window_pnl: float = 0.0
        self.trades_today: int = 0

    def reset_window(self) -> None:
        self.window_pnl = 0.0

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.window_pnl = 0.0
        self.trades_today = 0

    def record_trade(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.window_pnl += pnl
        self.trades_today += 1

    @property
    def can_trade(self) -> bool:
        if self.daily_pnl <= -config.MAX_DAILY_LOSS:
            log.warning("Daily loss limit hit: $%.2f", self.daily_pnl)
            return False
        if self.window_pnl <= -config.MAX_LOSS_PER_WINDOW:
            log.warning("Window loss limit hit: $%.2f", self.window_pnl)
            return False
        return True

    def size_position(self, edge: float, buy_price: float) -> float:
        """Compute bet amount using fractional Kelly criterion.

        Kelly fraction f* = edge / (payout - 1) for binary bets where
        payout = 1/buy_price. Simplified: f* = edge / ((1 - buy_price) / buy_price)
        = edge * buy_price / (1 - buy_price).

        We use quarter-Kelly by default for safety.
        """
        if buy_price <= 0 or buy_price >= 1.0 or edge <= 0:
            return 0.0

        # Potential profit per $1 risked
        odds = (1.0 - buy_price) / buy_price  # e.g., 0.50 → odds = 1.0
        kelly_fraction = edge / odds if odds > 0 else 0.0
        kelly_fraction = max(0.0, kelly_fraction)

        # Apply safety multiplier (quarter-Kelly by default)
        bet_fraction = kelly_fraction * config.KELLY_FRACTION

        # Kelly fraction of session bankroll (MAX_DAILY_LOSS), capped at BET_AMOUNT
        amount = min(bet_fraction * config.MAX_DAILY_LOSS, config.BET_AMOUNT)

        # Enforce minimum order (5 shares * price)
        # If Kelly says to bet but below minimum, round up to minimum
        min_spend = config.MIN_ORDER_SHARES * buy_price
        if amount < min_spend:
            amount = min_spend

        # Don't exceed remaining daily budget
        remaining = config.MAX_DAILY_LOSS + self.daily_pnl
        amount = min(amount, remaining)

        # Don't exceed window limit
        window_remaining = config.MAX_LOSS_PER_WINDOW + self.window_pnl
        amount = min(amount, window_remaining)

        # Round up to nearest cent to avoid floating-point rounding to zero
        return max(0.0, math.ceil(amount * 100) / 100)
