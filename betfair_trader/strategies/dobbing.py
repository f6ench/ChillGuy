"""Dobbing (Double or Bust) strategy.

Back pre-race. Auto lay at half the odds in-play.
If odds drop 50%, profit doubles. Classic in-play strategy.
"""

from __future__ import annotations

import logging

from betfair_trader.models import MarketSnapshot, Order, Side, TradeInstruction
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.utils.hedge import kelly_stake
from betfair_trader.utils.odds import nearest_valid_price

logger = logging.getLogger(__name__)


class DobbingStrategy(BaseStrategy):
    """Back pre-race, lay in-play at half the odds."""

    def evaluate(
        self,
        instruction: TradeInstruction,
        snapshots: list[MarketSnapshot],
    ) -> list[Order]:
        runner = self._find_runner(instruction.selection_id, snapshots)
        if not runner or runner.best_back == 0:
            return []

        back_price = runner.best_back

        # Target lay price is half the back odds
        target_lay = nearest_valid_price(back_price / 2)
        if target_lay < 1.01:
            return []

        # Position sizing via fractional Kelly
        stake = kelly_stake(
            probability=instruction.p,
            odds=back_price,
            fraction=self.config.kelly_fraction,
            bankroll=self.config.max_liability,
        )
        stake = min(stake, self.config.max_stake)
        if stake < 0.01:
            logger.debug("Kelly says no edge for DOB on %s", instruction.selection)
            return []

        # Entry order: back pre-race
        orders = [
            Order(
                market_id=instruction.market_id,
                selection_id=instruction.selection_id,
                side=Side.BACK,
                price=back_price,
                size=round(stake, 2),
                reference=f"dob-back-{instruction.selection_id}",
            ),
        ]
        logger.info("DOB: BACK %.2f on %s (£%.2f), target lay %.2f in-play",
                     back_price, instruction.selection, stake, target_lay)
        return orders
