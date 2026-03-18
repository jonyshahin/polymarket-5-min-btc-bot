"""Edge calculator: compare model probability vs market-implied odds."""

import logging
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """The computed edge for a potential trade."""
    side: str  # "UP" or "DOWN"
    model_prob: float  # Our estimated probability of Up winning
    market_price_up: float  # Current price of Up token
    market_price_down: float  # Current price of Down token
    edge: float  # Absolute edge (model_prob - market_implied)
    token_id: str  # Which token to buy
    buy_price: float  # Price we'd pay per share

    @property
    def is_tradeable(self) -> bool:
        return self.edge >= config.EDGE_THRESHOLD

    @property
    def edge_pct(self) -> float:
        return self.edge * 100


def compute_edge(
    model_prob_up: float,
    market_price_up: float,
    market_price_down: float,
    up_token_id: str,
    down_token_id: str,
) -> EdgeResult:
    """Determine which side has edge and how much.

    model_prob_up: probability that Up wins (0–1)
    market_price_{up,down}: current buy price on CLOB (0–1)
    """
    model_prob_down = 1.0 - model_prob_up

    # Edge for buying Up: we think Up wins with prob P, market charges market_price_up
    edge_up = model_prob_up - market_price_up
    # Edge for buying Down
    edge_down = model_prob_down - market_price_down

    if edge_up >= edge_down:
        return EdgeResult(
            side="UP",
            model_prob=model_prob_up,
            market_price_up=market_price_up,
            market_price_down=market_price_down,
            edge=edge_up,
            token_id=up_token_id,
            buy_price=market_price_up,
        )
    else:
        return EdgeResult(
            side="DOWN",
            model_prob=model_prob_up,
            market_price_up=market_price_up,
            market_price_down=market_price_down,
            edge=edge_down,
            token_id=down_token_id,
            buy_price=market_price_down,
        )
