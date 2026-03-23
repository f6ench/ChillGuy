"""Lay to Back (Drifter Bot) strategy.

Identify horse whose odds are drifting (rising). Lay it, then back at
higher price. Profit from the drift.

Signal sources: horse misbehaving in parade ring, trainer out of form,
market money going elsewhere.
"""

from __future__ import annotations

import logging

from betfair_trader.models import MarketSnapshot, Order, Side, TradeInstruction
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.utils.hedge import kelly_stake
from betfair_trader.utils.odds import nearest_valid_price

logger = logging.getLogger(__name__)


class LayToBackStrategy(BaseStrategy):
    """Lay a drifter, then back at higher odds to lock in profit."""

    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        runner = self._find_runner(instruction.selection_id, snapshots)
        if not runner or runner.best_lay == 0:
            return []

        lay_price = runner.best_lay
        target_back = nearest_valid_price(instruction.target_odds)

        # Only enter if current lay price is below our target back price
        if lay_price >= target_back:
            logger.debug("Lay price %.2f >= target %.2f, skipping %s",
                         lay_price, target_back, instruction.selection)
            return []

        # Edge check
        if instruction.edge < self.config.min_edge:
            return []

        stake = kelly_stake(
            probability=1 - instruction.p,  # Probability of losing (we're laying)
            odds=lay_price,
            fraction=self.config.kelly_fraction,
            bankroll=self.config.max_liability,
        )
        # Liability check
        liability = stake * (lay_price - 1)
        if liability > self.config.max_liability:
            stake = self.config.max_liability / (lay_price - 1)
        stake = min(stake, self.config.max_stake)
        if stake < 0.01:
            return []

        orders = [
            Order(
                market_id=instruction.market_id,
                selection_id=instruction.selection_id,
                side=Side.LAY,
                price=lay_price,
                size=round(stake, 2),
                reference=f"l2b-lay-{instruction.selection_id}",
            ),
        ]
        logger.info("L2B: LAY %.2f on %s (£%.2f), target back %.2f",
                     lay_price, instruction.selection, stake, target_back)
        return orders
