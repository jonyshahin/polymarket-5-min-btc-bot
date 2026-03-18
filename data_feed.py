"""Binance WebSocket feed for real-time BTC/USDT kline data with reconnect logic,
plus REST order book snapshots and kline fetching."""

import asyncio
import json
import time
import logging
from collections import deque
from dataclasses import dataclass

import aiohttp
import websockets

import config

log = logging.getLogger(__name__)


@dataclass
class Tick:
    """A single 1-second kline tick from Binance."""
    timestamp: float  # local receipt time
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBookSnapshot:
    """Top-of-book bid/ask from Binance REST API."""
    timestamp: float
    bid_volume: float
    ask_volume: float
    best_bid: float
    best_ask: float

    @property
    def imbalance(self) -> float:
        """Bid/ask imbalance: +1.0 = all bids, -1.0 = all asks, 0.0 = balanced."""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total


class BinanceFeed:
    """Async Binance WebSocket consumer that maintains a rolling price buffer."""

    def __init__(self, max_history: int | None = None):
        size = max_history or config.TREND_BUFFER_SECONDS
        self.ticks: deque[Tick] = deque(maxlen=size)
        self._ws = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def last_tick(self) -> Tick | None:
        return self.ticks[-1] if self.ticks else None

    @property
    def last_price(self) -> float | None:
        t = self.last_tick
        return t.close if t else None

    @property
    def data_age(self) -> float:
        t = self.last_tick
        if t is None:
            return float("inf")
        return time.time() - t.timestamp

    @property
    def is_fresh(self) -> bool:
        return self.data_age < config.STALE_DATA_THRESHOLD

    def prices(self, n: int | None = None) -> list[float]:
        """Return recent close prices, oldest first."""
        src = list(self.ticks) if n is None else list(self.ticks)[-n:]
        return [t.close for t in src]

    def volumes(self, n: int | None = None) -> list[float]:
        src = list(self.ticks) if n is None else list(self.ticks)[-n:]
        return [t.volume for t in src]

    def get_prices_since(self, since_ts: float) -> list[float]:
        """Return close prices for all ticks after a given timestamp."""
        return [t.close for t in self.ticks if t.timestamp >= since_ts]

    def get_prior_window_delta(self) -> float | None:
        """Return price change over the last completed 5-min window.

        Looks at ticks from [now - 600s, now - 300s] to estimate the
        previous window's open-to-close delta. Returns None if not enough data.
        """
        now = time.time()
        prev_start = now - 600
        prev_end = now - 300
        prev_ticks = [t for t in self.ticks
                      if prev_start <= t.timestamp <= prev_end]
        if len(prev_ticks) < 30:
            return None
        return prev_ticks[-1].close - prev_ticks[0].close

    async def start(self) -> None:
        self._running = True
        self._http_session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def get_order_book(self, limit: int = 5) -> OrderBookSnapshot | None:
        if not self._http_session:
            return None
        url = f"{config.BINANCE_REST_BASE}/api/v3/depth"
        params = {"symbol": "BTCUSDT", "limit": limit}
        try:
            async with self._http_session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            bid_vol = sum(float(level[1]) for level in data.get("bids", []))
            ask_vol = sum(float(level[1]) for level in data.get("asks", []))
            best_bid = float(data["bids"][0][0]) if data.get("bids") else 0.0
            best_ask = float(data["asks"][0][0]) if data.get("asks") else 0.0

            return OrderBookSnapshot(
                timestamp=time.time(),
                bid_volume=bid_vol, ask_volume=ask_vol,
                best_bid=best_bid, best_ask=best_ask,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError) as exc:
            log.debug("Order book fetch failed: %s", exc)
            return None

    async def fetch_kline(self, window_ts: int) -> tuple[float, float] | None:
        """Fetch the 5-minute Binance kline covering a window.
        Returns (open_price, close_price) or None on failure."""
        if not self._http_session:
            return None
        url = f"{config.BINANCE_REST_BASE}/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "startTime": window_ts * 1000,
            "limit": 1,
        }
        try:
            async with self._http_session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            if not data:
                return None
            kline = data[0]
            return float(kline[1]), float(kline[4])
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError) as exc:
            log.debug("Kline fetch failed for ts=%d: %s", window_ts, exc)
            return None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._connect()
            except (
                websockets.ConnectionClosed,
                websockets.InvalidURI,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                if self._running:
                    log.warning("Binance WS disconnected: %s — reconnecting in 2s", exc)
                    await asyncio.sleep(2)

    async def _connect(self) -> None:
        log.info("Connecting to Binance WS: %s", config.BINANCE_WS_URL)
        async with websockets.connect(config.BINANCE_WS_URL, ping_interval=20) as ws:
            self._ws = ws
            log.info("Binance WS connected")
            async for raw in ws:
                if not self._running:
                    break
                self._handle_message(raw)

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        k = msg.get("k")
        if not k:
            return
        tick = Tick(
            timestamp=time.time(),
            open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]), close=float(k["c"]),
            volume=float(k["v"]),
        )
        self.ticks.append(tick)
