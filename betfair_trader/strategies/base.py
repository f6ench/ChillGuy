"""Base strategy interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from betfair_trader.config import BetfairConfig
from betfair_trader.models import MarketSnapshot, Order, TradeInstruction

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """All strategies inherit from this.

    A strategy receives market snapshots and trade instructions,
    and returns orders to execute.
    """

    def __init__(self, config: BetfairConfig):
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        """Evaluate whether to execute and return orders.

        Args:
            instruction: Claude's trade instruction for this selection
            snapshots: Current market data for all runners

        Returns:
            List of orders to place (empty = skip)
        """
        ...

    def _find_runner(
        self, selection_id: int, snapshots: list[MarketSnapshot]
    ) -> MarketSnapshot | None:
        for snap in snapshots:
            if snap.selection_id == selection_id:
                return snap
        return None
