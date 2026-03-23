"""Trading execution engine — the Guardian equivalent.

This is the main loop that:
1. Discovers upcoming markets
2. Streams live data
3. Asks Claude for trade instructions
4. Executes strategies
5. Monitors for hedge opportunities
6. Logs everything for the feedback loop
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from betfair_trader.ai.claude_analyst import ClaudeAnalyst
from betfair_trader.client import BetfairClient
from betfair_trader.config import BetfairConfig
from betfair_trader.healing import SelfHealingManager
from betfair_trader.models import (
    MarketSnapshot,
    Order,
    Strategy,
    TradeInstruction,
    TradeRecord,
)
from betfair_trader.storage import TradeLogger
from betfair_trader.strategies.back_to_lay import BackToLayStrategy
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.strategies.dobbing import DobbingStrategy
from betfair_trader.strategies.lay_to_back import LayToBackStrategy
from betfair_trader.strategies.scalping import ScalpingStrategy
from betfair_trader.streaming import MarketStream
from betfair_trader.utils.hedge import (
    calculate_back_to_lay_hedge,
    calculate_lay_to_back_hedge,
)

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading loop — our Guardian equivalent."""

    def __init__(self, config: BetfairConfig):
        self.config = config
        self.client = BetfairClient(config)
        self.analyst = ClaudeAnalyst(config)
        self.trade_logger = TradeLogger()
        self.healer = SelfHealingManager()
        self._strategies: dict[Strategy, BaseStrategy] = {
            Strategy.SCALPING: ScalpingStrategy(config),
            Strategy.DOBBING: DobbingStrategy(config),
            Strategy.LAY_TO_BACK: LayToBackStrategy(config),
            Strategy.BACK_TO_LAY: BackToLayStrategy(config),
        }
        self._active_trades: dict[str, TradeRecord] = {}
        self._running = False

    def start(self) -> None:
        """Main entry point — login and start the trading loop."""
        mode = "PAPER" if self.config.paper_trading else "LIVE"
        logger.info("Starting trading engine in %s mode", mode)

        if not self.config.paper_trading and not self.client.login():
            logger.error("Failed to login to Betfair. Exiting.")
            return

        if self.config.paper_trading:
            logger.info("[PAPER MODE] Skipping Betfair login")

        self._running = True
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            if not self.config.paper_trading:
                self.client.logout()
            self.trade_logger.close()

    def _run_loop(self) -> None:
        """Guardian-style loop: scan markets, analyze, execute."""
        while self._running:
            try:
                # 1. Discover markets
                markets = self._discover_markets()
                if not markets:
                    logger.info("No markets found. Waiting 60s...")
                    time.sleep(60)
                    continue

                # Check if self-healing forced paper mode
                if self.healer.strategy_healer.should_force_paper_mode():
                    if not self.config.paper_trading:
                        logger.warning("Self-healing: forcing paper mode due to poor performance")
                        self.config.paper_trading = True

                # 2. Process each market (with Level 1 recovery)
                for market in markets:
                    if not self._running:
                        break
                    self.healer.wrap_execution(self._process_market, market)

                # 3. Monitor active trades for hedge opportunities
                self._monitor_hedges()

                # 4. Heartbeat
                self.healer.code_healer.pulse()

                # 5. Keep session alive
                if not self.config.paper_trading:
                    self.client.keep_alive()

                time.sleep(5)  # Main loop interval

            except Exception as e:
                logger.error("Engine loop error: %s", e, exc_info=True)
                time.sleep(10)

    def _discover_markets(self) -> list[dict]:
        """Find upcoming horse racing markets."""
        if self.config.paper_trading:
            logger.info("[PAPER] Simulating market discovery")
            return self._paper_markets()

        return self.client.list_horse_racing_markets(hours_ahead=4)

    def _paper_markets(self) -> list[dict]:
        """Generate simulated markets for paper trading."""
        import datetime as dt

        return [{
            "market_id": "1.234567890",
            "market_name": "3:30 Cheltenham",
            "event": "Cheltenham",
            "start_time": dt.datetime.utcnow() + dt.timedelta(minutes=30),
            "runners": [
                {"id": 101, "name": "Horse A"},
                {"id": 102, "name": "Horse B"},
                {"id": 103, "name": "Horse C"},
                {"id": 104, "name": "Horse D"},
            ],
        }]

    def _process_market(self, market: dict) -> None:
        """Analyze a single market and execute any trades."""
        market_id = market["market_id"]
        logger.info("Processing market: %s (%s)", market["market_name"], market_id)

        # Get current prices
        if self.config.paper_trading:
            snapshots = self._paper_snapshots(market)
        else:
            snapshots = self.client.get_market_book(market_id)

        if not snapshots:
            return

        # Ask Claude for analysis
        instructions = self.analyst.analyze_market(market, snapshots)

        # Execute qualifying trades
        for instruction in instructions:
            if instruction.edge < self.config.min_edge:
                logger.debug("Skipping %s: edge %.3f < min %.3f",
                             instruction.selection, instruction.edge, self.config.min_edge)
                continue

            strategy = self._strategies.get(instruction.strategy)
            if not strategy:
                logger.warning("Unknown strategy: %s", instruction.strategy)
                continue

            orders = strategy.evaluate(instruction, snapshots)
            if orders:
                self._execute_orders(instruction, orders)

    def _paper_snapshots(self, market: dict) -> list[MarketSnapshot]:
        """Generate simulated market data for paper trading."""
        import random

        snapshots = []
        for runner in market["runners"]:
            price = round(random.uniform(2.0, 15.0), 2)
            snapshots.append(MarketSnapshot(
                market_id=market["market_id"],
                selection_id=runner["id"],
                runner_name=runner["name"],
                back_prices=[(price, round(random.uniform(50, 500), 2))],
                lay_prices=[(round(price + 0.02, 2), round(random.uniform(50, 500), 2))],
                last_traded_price=price,
                total_matched=round(random.uniform(1000, 50000), 2),
            ))
        return snapshots

    def _execute_orders(
        self, instruction: TradeInstruction, orders: list[Order]
    ) -> None:
        """Place orders and log the trade."""
        record = TradeRecord(
            instruction=instruction,
            orders=orders,
        )

        for order in orders:
            bet_id = self.client.place_order(order)
            if bet_id:
                record.entry_matched = True

        # Track for hedging
        key = f"{instruction.market_id}-{instruction.selection_id}"
        self._active_trades[key] = record
        self.trade_logger.log_trade(record)

        logger.info("Trade executed: %s %s on %s (edge=%.3f, confidence=%s)",
                     instruction.strategy.value, instruction.signal.value,
                     instruction.selection, instruction.edge,
                     instruction.confidence.value)

    def _monitor_hedges(self) -> None:
        """Check active trades for hedge opportunities."""
        for key, record in list(self._active_trades.items()):
            if record.hedge_matched or not record.entry_matched:
                continue
            if not record.instruction.hedge:
                continue

            instruction = record.instruction
            market_id = instruction.market_id

            if self.config.paper_trading:
                # In paper mode, simulate hedge after a few cycles
                logger.info("[PAPER] Would check hedge for %s", instruction.selection)
                continue

            snapshots = self.client.get_market_book(market_id)
            runner = None
            for snap in snapshots:
                if snap.selection_id == instruction.selection_id:
                    runner = snap
                    break

            if not runner:
                continue

            self._check_hedge(record, runner)

    def _check_hedge(self, record: TradeRecord, runner: MarketSnapshot) -> None:
        """Check if hedge conditions are met and execute."""
        instruction = record.instruction

        if instruction.strategy == Strategy.BACK_TO_LAY:
            # We backed, check if price has shortened enough to lay
            if runner.best_lay > 0 and runner.best_lay <= instruction.target_odds:
                hedge = calculate_back_to_lay_hedge(
                    back_stake=record.orders[0].size,
                    back_odds=record.orders[0].price,
                    lay_odds=runner.best_lay,
                    commission=self.config.commission_rate,
                )
                if hedge.is_profitable:
                    self._place_hedge(record, runner.best_lay, hedge.hedge_stake, Side.LAY)

        elif instruction.strategy == Strategy.LAY_TO_BACK:
            # We laid, check if price has drifted enough to back
            if runner.best_back > 0 and runner.best_back >= instruction.target_odds:
                hedge = calculate_lay_to_back_hedge(
                    lay_stake=record.orders[0].size,
                    lay_odds=record.orders[0].price,
                    back_odds=runner.best_back,
                    commission=self.config.commission_rate,
                )
                if hedge.is_profitable:
                    self._place_hedge(record, runner.best_back, hedge.hedge_stake, Side.BACK)

    def _place_hedge(
        self, record: TradeRecord, price: float, stake: float, side: str
    ) -> None:
        """Execute a hedge order."""
        from betfair_trader.models import Side as SideEnum

        instruction = record.instruction
        hedge_order = Order(
            market_id=instruction.market_id,
            selection_id=instruction.selection_id,
            side=SideEnum(side),
            price=price,
            size=round(stake, 2),
            reference=f"hedge-{instruction.selection_id}",
        )
        bet_id = self.client.place_order(hedge_order)
        if bet_id:
            record.hedge_matched = True
            record.orders.append(hedge_order)
            logger.info("Hedge placed: %s %.2f @ %.2f on %s",
                        side, stake, price, instruction.selection)

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running = False
