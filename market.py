"""Window timing, slug calculation, and token ID fetching via Gamma API."""

import json
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

import config

log = logging.getLogger(__name__)


@dataclass
class Window:
    """Represents a single 5-minute prediction window."""
    timestamp: int  # Unix ts divisible by 300
    slug: str
    up_token_id: Optional[str] = None
    down_token_id: Optional[str] = None

    @property
    def close_time(self) -> int:
        return self.timestamp + config.WINDOW_SECONDS

    @property
    def decision_start(self) -> float:
        """When the bot should start analyzing (T - LEAD_TIME)."""
        return self.close_time - config.LEAD_TIME

    @property
    def hard_deadline(self) -> float:
        """Latest moment to place an order (T - HARD_DEADLINE)."""
        return self.close_time - config.HARD_DEADLINE


def current_window_ts() -> int:
    """Return the start timestamp of the current 5-minute window."""
    now = int(time.time())
    return now - (now % config.WINDOW_SECONDS)


def next_window_ts() -> int:
    return current_window_ts() + config.WINDOW_SECONDS


def make_slug(window_ts: int) -> str:
    return f"btc-updown-5m-{window_ts}"


def make_window(window_ts: int) -> Window:
    return Window(timestamp=window_ts, slug=make_slug(window_ts))


async def fetch_token_ids(
    window: Window,
    session: aiohttp.ClientSession,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> bool:
    """Fetch Up/Down token IDs from Gamma API. Returns True on success."""
    url = f"{config.GAMMA_API_BASE}/markets"
    params = {"slug": window.slug}

    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning("Gamma API returned %d (attempt %d/%d)", resp.status, attempt, max_retries)
                    await asyncio.sleep(retry_delay)
                    continue

                data = await resp.json()
                if not data:
                    log.debug("Market not yet available for %s (attempt %d/%d)", window.slug, attempt, max_retries)
                    await asyncio.sleep(retry_delay)
                    continue

                market = data[0] if isinstance(data, list) else data
                token_ids = market.get("clobTokenIds")
                # Gamma API may return clobTokenIds as a JSON string
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if not token_ids or len(token_ids) < 2:
                    log.warning("Incomplete token IDs for %s", window.slug)
                    await asyncio.sleep(retry_delay)
                    continue

                # outcomes[0]="Up", outcomes[1]="Down" — map token IDs accordingly
                outcomes = market.get("outcomes", [])
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)

                if outcomes and outcomes[0].lower() == "up":
                    window.up_token_id = token_ids[0]
                    window.down_token_id = token_ids[1]
                elif outcomes and outcomes[0].lower() == "down":
                    window.up_token_id = token_ids[1]
                    window.down_token_id = token_ids[0]
                else:
                    # Default assumption: index 0 = Up
                    window.up_token_id = token_ids[0]
                    window.down_token_id = token_ids[1]
                log.info("Fetched token IDs for %s: Up=%s, Down=%s",
                         window.slug, window.up_token_id[:12], window.down_token_id[:12])
                return True

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("Gamma API error (attempt %d/%d): %s", attempt, max_retries, exc)
            await asyncio.sleep(retry_delay)

    log.error("Failed to fetch token IDs for %s after %d attempts", window.slug, max_retries)
    return False
