"""Avellaneda-Stoikov market making strategy adapted for Betfair.

The market maker quotes bid (back) and ask (lay) around an
inventory-adjusted reservation price — not the market mid, but
a price skewed by current position.

Core equation:
    r_t = s_t - q * gamma * sigma^2 * (T - t)

where q is inventory (positive = long), s_t is mid-price.

The spread widens when:
- VPIN is elevated (adverse selection risk)
- Hawkes branching ratio is high (hot market)
- Kyle's lambda is large (informed traders active)
- Inventory is near limits

Based on Avellaneda & Stoikov (2008): "High-frequency trading
in a limit order book."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from betfair_trader.config import BetfairConfig
from betfair_trader.models import (
    Confidence,
    MarketSnapshot,
    Order,
    Signal,
    Side,
    Strategy,
    TradeInstruction,
)
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.utils.odds import nearest_valid_price

logger = logging.getLogger(__name__)


@dataclass
class MarketMakerState:
    """Market maker state for a single market."""

    inventory: float = 0.0  # GBP (+ = long back, - = short/laid)
    cash: float = 0.0
    n_fills: int = 0
    pnl_history: list[float] = field(default_factory=list)


@dataclass
class QuoteResult:
    """Bid/ask quotes from the market maker."""

    bid_price: float  # price to back at (our bid)
    ask_price: float  # price to lay at (our ask)
    reservation_price: float  # inventory-adjusted fair value
    half_spread: float
    spread_multiplier: float  # from VPIN/signals


class AvellanedaStoikovStrategy(BaseStrategy):
    """Market making strategy using the Avellaneda-Stoikov model.

    Quotes bid (back) and ask (lay) around an inventory-adjusted
    reservation price, with spread widening based on adverse selection
    signals (VPIN, Kyle's lambda, Hawkes branching ratio).
    """

    def __init__(
        self,
        config: BetfairConfig,
        gamma: float = 0.1,  # risk aversion
        kappa: float = 1.5,  # execution intensity
        sigma: float = 0.05,  # contract volatility per hour
        max_inventory: float = 500,  # max position in GBP
        T_horizon: float = 4.0,  # time horizon in hours
    ):
        super().__init__(config)
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.max_inventory = max_inventory
        self.T_horizon = T_horizon

        # Per-market state
        self._states: dict[str, MarketMakerState] = {}

    def _get_state(self, market_id: str) -> MarketMakerState:
        if market_id not in self._states:
            self._states[market_id] = MarketMakerState()
        return self._states[market_id]

    def reservation_price(
        self,
        mid: float,
        inventory: float,
        time_remaining: float,
    ) -> float:
        """Inventory-adjusted fair price.

        Large long inventory -> shade price down to attract sells.
        Large short inventory -> shade price up to attract buys.
        """
        return mid - inventory * self.gamma * self.sigma ** 2 * time_remaining

    def optimal_half_spread(self, time_remaining: float) -> float:
        """Base half-spread from the A-S model."""
        base = self.gamma * self.sigma ** 2 * time_remaining
        execution_term = (1 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return base + execution_term

    def compute_quotes(
        self,
        mid: float,
        market_id: str,
        time_remaining: float,
        spread_multiplier: float = 1.0,
    ) -> QuoteResult:
        """Compute bid/ask quotes adjusted for inventory and signals.

        Args:
            mid: current mid-price (back+lay)/2
            market_id: for state lookup
            time_remaining: hours until market close/resolution
            spread_multiplier: from VPIN (1.0=normal, 2.0=double, 0=pull)
        """
        state = self._get_state(market_id)
        r = self.reservation_price(mid, state.inventory, time_remaining)
        half_spread = self.optimal_half_spread(time_remaining) * spread_multiplier

        bid = r - half_spread
        ask = r + half_spread

        # Clamp to valid range
        bid = np.clip(bid, 1.01, 1000.0)
        ask = np.clip(ask, 1.01, 1000.0)
        ask = max(ask, bid + 0.01)

        # Inventory blowup protection
        if abs(state.inventory) > self.max_inventory * 0.8:
            if state.inventory > 0:
                # Aggressively unload longs: lower the ask
                ask = min(ask, mid - 0.005)
                logger.warning("Market %s: inventory %.0f, aggressively selling",
                               market_id, state.inventory)
            else:
                # Aggressively unload shorts: raise the bid
                bid = max(bid, mid + 0.005)
                logger.warning("Market %s: inventory %.0f, aggressively buying",
                               market_id, state.inventory)

        # Snap to Betfair tick ladder
        bid = nearest_valid_price(bid)
        ask = nearest_valid_price(ask)

        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            reservation_price=r,
            half_spread=half_spread,
            spread_multiplier=spread_multiplier,
        )

    def record_fill(
        self,
        market_id: str,
        price: float,
        size: float,
        is_buy: bool,
    ) -> None:
        """Record a fill (our quote was hit)."""
        state = self._get_state(market_id)
        if is_buy:
            state.inventory += size
            state.cash -= price * size
        else:
            state.inventory -= size
            state.cash += price * size
        state.n_fills += 1

        mark = state.inventory * price + state.cash
        state.pnl_history.append(mark)

    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        """Evaluate and generate orders for market making.

        For market making, we generate BOTH a back and a lay order
        (quoting both sides of the book).
        """
        snap = None
        for s in snapshots:
            if s.selection_id == instruction.selection_id:
                snap = s
                break

        if not snap or snap.best_back <= 0 or snap.best_lay <= 0:
            return []

        mid = (snap.best_back + snap.best_lay) / 2
        quotes = self.compute_quotes(
            mid=mid,
            market_id=instruction.market_id,
            time_remaining=self.T_horizon,
            spread_multiplier=1.0,
        )

        # Check if the spread is profitable after commission
        gross_spread = quotes.ask_price - quotes.bid_price
        commission_cost = self.config.commission_rate * (quotes.ask_price + quotes.bid_price)
        if gross_spread <= commission_cost:
            return []

        size = min(self.config.max_stake, self.max_inventory * 0.1)

        orders = []
        # Back order (our bid)
        orders.append(Order(
            market_id=instruction.market_id,
            selection_id=instruction.selection_id,
            side=Side.BACK,
            price=quotes.bid_price,
            size=round(size, 2),
            reference=f"mm-bid-{instruction.selection_id}",
        ))
        # Lay order (our ask)
        orders.append(Order(
            market_id=instruction.market_id,
            selection_id=instruction.selection_id,
            side=Side.LAY,
            price=quotes.ask_price,
            size=round(size, 2),
            reference=f"mm-ask-{instruction.selection_id}",
        ))

        return orders

    def get_state_summary(self, market_id: str) -> dict:
        """Get state summary for monitoring."""
        state = self._get_state(market_id)
        return {
            "inventory": state.inventory,
            "cash": state.cash,
            "n_fills": state.n_fills,
            "inventory_pct": abs(state.inventory) / self.max_inventory * 100,
        }
