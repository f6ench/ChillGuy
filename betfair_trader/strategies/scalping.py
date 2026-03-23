"""Scalping strategy.

Back and lay very close together to scrape small profit from
price fluctuations. Works best in liquid markets with tight spreads.
"""

from __future__ import annotations

import logging

from betfair_trader.models import MarketSnapshot, Order, Side, TradeInstruction
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.utils.odds import ticks_away

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """Place back and lay orders N ticks apart to capture spread profit."""

    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        runner = self._find_runner(instruction.selection_id, snapshots)
        if not runner:
            return []

        if runner.spread == 0 or runner.best_back == 0:
            logger.debug("No spread or no back price for %s", instruction.selection)
            return []

        n_ticks = self.config.scalp_ticks
        back_price = runner.best_back
        lay_price = ticks_away(back_price, n_ticks)

        # Ensure we can make profit after commission
        gross_profit = (lay_price - back_price) * self.config.max_stake
        net_profit = gross_profit * (1 - self.config.commission_rate)
        if net_profit <= 0:
            logger.debug("No profit after commission for scalp on %s", instruction.selection)
            return []

        stake = min(self.config.max_stake, instruction.min_profit / net_profit * self.config.max_stake if instruction.min_profit > 0 else self.config.max_stake)

        orders = [
            Order(
                market_id=instruction.market_id,
                selection_id=instruction.selection_id,
                side=Side.BACK,
                price=back_price,
                size=round(stake, 2),
                reference=f"scalp-back-{instruction.selection_id}",
            ),
            Order(
                market_id=instruction.market_id,
                selection_id=instruction.selection_id,
                side=Side.LAY,
                price=lay_price,
                size=round(stake, 2),
                reference=f"scalp-lay-{instruction.selection_id}",
            ),
        ]
        logger.info("Scalp: BACK %.2f / LAY %.2f on %s (£%.2f)",
                     back_price, lay_price, instruction.selection, stake)
        return orders
