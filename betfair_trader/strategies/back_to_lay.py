"""Back to Lay (Springer Bot) strategy.

Identify horse whose odds are shortening. Back at higher price,
lay at lower price. Profit from the steam.

Signal sources: stable money detected, trainer/jockey in form,
positive Racing Post whispers.
"""

from __future__ import annotations

import logging

from betfair_trader.models import MarketSnapshot, Order, Side, TradeInstruction
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.utils.hedge import kelly_stake
from betfair_trader.utils.odds import nearest_valid_price

logger = logging.getLogger(__name__)


class BackToLayStrategy(BaseStrategy):
    """Back a springer, then lay at shorter odds to lock in profit."""

    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        runner = self._find_runner(instruction.selection_id, snapshots)
        if not runner or runner.best_back == 0:
            return []

        back_price = runner.best_back
        target_lay = nearest_valid_price(instruction.target_odds)

        # Only enter if current back price is above our target lay price
        if back_price <= target_lay:
            logger.debug("Back price %.2f <= target %.2f, skipping %s",
                         back_price, target_lay, instruction.selection)
            return []

        # Edge check
        if instruction.edge < self.config.min_edge:
            return []

        stake = kelly_stake(
            probability=instruction.p,
            odds=back_price,
            fraction=self.config.kelly_fraction,
            bankroll=self.config.max_liability,
        )
        stake = min(stake, self.config.max_stake)
        if stake < 0.01:
            return []

        orders = [
            Order(
                market_id=instruction.market_id,
                selection_id=instruction.selection_id,
                side=Side.BACK,
                price=back_price,
                size=round(stake, 2),
                reference=f"b2l-back-{instruction.selection_id}",
            ),
        ]
        logger.info("B2L: BACK %.2f on %s (£%.2f), target lay %.2f",
                     back_price, instruction.selection, stake, target_lay)
        return orders
