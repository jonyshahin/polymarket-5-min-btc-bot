"""py-clob-client wrapper: FOK market orders + $0.95 GTC limit fallback."""

import logging
import math
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)

# Lazy-imported to avoid import errors in dry-run when keys aren't set
_client = None


@dataclass
class OrderResult:
    success: bool
    order_id: str | None = None
    side: str = ""
    token_id: str = ""
    amount: float = 0.0
    price: float = 0.0
    shares: float = 0.0
    order_type: str = ""
    error: str | None = None


def _get_client():
    """Lazily initialize the CLOB client."""
    global _client
    if _client is not None:
        return _client

    from py_clob_client.client import ClobClient

    if not config.POLYMARKET_PRIVATE_KEY:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY not set")

    _client = ClobClient(
        config.CLOB_HOST,
        key=config.POLYMARKET_PRIVATE_KEY,
        chain_id=config.CHAIN_ID,
        signature_type=config.SIGNATURE_TYPE,
        funder=config.POLYMARKET_FUNDER,
    )
    _client.set_api_creds(_client.create_or_derive_api_creds())
    log.info("CLOB client initialized")
    return _client


_anon_client = None


def _get_anon_client():
    """Lazily initialize an unauthenticated CLOB client for read-only ops."""
    global _anon_client
    if _anon_client is None:
        from py_clob_client.client import ClobClient
        _anon_client = ClobClient(config.CLOB_HOST, chain_id=config.CHAIN_ID)
    return _anon_client


def get_market_prices(up_token_id: str, down_token_id: str) -> tuple[float, float]:
    """Fetch current buy prices for Up and Down tokens. No auth required.

    Uses best-ask (what you'd actually pay to buy) with midpoint as fallback.
    Falls back to 0.50/0.50 if both fail.
    """
    anon = _get_anon_client()

    up_price = _fetch_buy_price(anon, up_token_id, "Up")
    down_price = _fetch_buy_price(anon, down_token_id, "Down")

    # If one side has no price, infer from the other (they should sum to ~1.0)
    if up_price is not None and down_price is None:
        down_price = max(0.01, 1.0 - up_price)
    elif down_price is not None and up_price is None:
        up_price = max(0.01, 1.0 - down_price)
    elif up_price is None and down_price is None:
        up_price, down_price = 0.50, 0.50

    return up_price, down_price


def _fetch_buy_price(client, token_id: str, label: str) -> float | None:
    """Fetch what you'd actually pay to buy a token.

    Tries BUY price first (best ask from seller's perspective).
    If 0 (no orders), falls back to midpoint.
    """
    try:
        resp = client.get_price(token_id, side="BUY")
        price = float(resp["price"]) if isinstance(resp, dict) else float(resp)
        if price > 0:
            return price
    except Exception as exc:
        log.debug("get_price failed for %s: %s", label, exc)

    # Fallback to midpoint
    try:
        resp = client.get_midpoint(token_id)
        mid = float(resp["mid"]) if isinstance(resp, dict) else float(resp)
        if mid > 0:
            return mid
    except Exception as exc:
        log.debug("get_midpoint failed for %s: %s", label, exc)

    return None


def place_fok_order(token_id: str, amount: float, side: str = "UP") -> OrderResult:
    """Place a Fill-or-Kill market order.

    amount: total USDC to spend
    """
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    try:
        client = _get_client()

        # Truncate amount to 2 decimal places (maker amount precision)
        amount = math.floor(amount * 100) / 100

        args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=BUY,
            order_type=OrderType.FOK,
        )
        signed = client.create_market_order(args)
        resp = client.post_order(signed, OrderType.FOK)

        order_id = resp.get("orderID") or resp.get("id")
        log.info("FOK order placed: %s $%.2f on %s — %s", side, amount, token_id[:12], resp)

        return OrderResult(
            success=True,
            order_id=order_id,
            side=side,
            token_id=token_id,
            amount=amount,
            order_type="FOK",
        )

    except Exception as exc:
        log.error("FOK order failed: %s", exc)
        return OrderResult(success=False, error=str(exc), side=side, token_id=token_id, amount=amount)


def place_limit_fallback(token_id: str, amount: float, side: str = "UP") -> OrderResult:
    """Place a GTC limit buy at $0.95 as a fallback when no liquidity.

    Guaranteed $0.05/share profit if filled and correct.
    """
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    try:
        client = _get_client()

        price = 0.95
        shares = math.floor(amount / price * 100) / 100  # 2 decimal places

        if shares < config.MIN_ORDER_SHARES:
            return OrderResult(
                success=False,
                error=f"Below minimum shares ({shares} < {config.MIN_ORDER_SHARES})",
                side=side,
                token_id=token_id,
            )

        args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=BUY,
            order_type=OrderType.GTC,
        )
        signed = client.create_order(args)
        resp = client.post_order(signed, OrderType.GTC)

        order_id = resp.get("orderID") or resp.get("id")
        log.info("GTC limit order placed: %s %.1f shares @ $0.95 — %s", side, shares, resp)

        return OrderResult(
            success=True,
            order_id=order_id,
            side=side,
            token_id=token_id,
            amount=round(shares * price, 2),
            price=price,
            shares=shares,
            order_type="GTC_LIMIT",
        )

    except Exception as exc:
        log.error("Limit order failed: %s", exc)
        return OrderResult(success=False, error=str(exc), side=side, token_id=token_id, amount=amount)
