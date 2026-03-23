"""Trading execution engine — the Guardian equivalent.

This is the main loop that:
1. Discovers upcoming markets
2. Filters races through the race selector
3. Fetches form data from Racing API (independent of market)
4. Records price movements for steam/drift detection
5. Runs two-stage Claude analysis (form-first, then market comparison)
6. Executes strategies on qualifying opportunities
7. Monitors for hedge opportunities
8. Logs everything for outcome validation
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from betfair_trader.ai.claude_analyst import ClaudeAnalyst
from betfair_trader.client import BetfairClient
from betfair_trader.config import BetfairConfig
from betfair_trader.filters.race_selector import RaceFilterConfig, RaceSelector
from betfair_trader.healing import SelfHealingManager
from betfair_trader.models import (
    MarketSnapshot,
    Order,
    Strategy,
    TradeInstruction,
    TradeRecord,
)
from betfair_trader.outcome.outcome_logger import BetRecord, OutcomeLogger, SignalSnapshot
from betfair_trader.signals.kyle_lambda import KyleLambdaEstimator
from betfair_trader.signals.hawkes_flow import HawkesFlowEstimator
from betfair_trader.signals.steam_detector import SteamDetector
from betfair_trader.signals.vpin import VPINCalculator
from betfair_trader.storage import TradeLogger
from betfair_trader.strategies.back_to_lay import BackToLayStrategy
from betfair_trader.strategies.base import BaseStrategy
from betfair_trader.strategies.dobbing import DobbingStrategy
from betfair_trader.strategies.lay_to_back import LayToBackStrategy
from betfair_trader.strategies.market_making import AvellanedaStoikovStrategy
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
        self.outcome_logger = OutcomeLogger()
        self.healer = SelfHealingManager()

        # Steam/drift detector
        self.steam_detector = SteamDetector(
            steam_threshold=config.steam_threshold,
            drift_threshold=config.drift_threshold,
            volume_threshold=config.volume_threshold,
        )

        # Quant signal estimators (Kyle's lambda, Hawkes, VPIN)
        self.kyle_estimator = KyleLambdaEstimator(window_size=200)
        self.hawkes_estimator = HawkesFlowEstimator(max_events=1000, n_starts=5)
        self.vpin_calculator = VPINCalculator(bucket_size=50, n_buckets=10)

        # Race selection filter
        self.race_selector = RaceSelector(RaceFilterConfig(
            min_liquidity=config.filter_min_liquidity,
            min_class=config.filter_min_class,
            max_class=config.filter_max_class,
            min_field_size=config.filter_min_field,
            max_field_size=config.filter_max_field,
            flat_only=config.filter_flat_only,
            exclude_maidens=config.filter_exclude_maidens,
        ))

        # Racing API client (optional — form data source)
        self._racing_api = None
        if config.racing_api_key:
            from betfair_trader.scrapers.racing_api import RacingAPIClient
            self._racing_api = RacingAPIClient(config.racing_api_key)
            logger.info("Racing API client initialised — two-stage analysis enabled")

        # Pre-race content scraper (optional)
        self._content_scraper = None
        try:
            from betfair_trader.scrapers.pre_race_content import PreRaceContentScraper
            self._content_scraper = PreRaceContentScraper()
        except ImportError:
            logger.info("beautifulsoup4 not installed — content scraping disabled")

        self._strategies: dict[Strategy, BaseStrategy] = {
            Strategy.SCALPING: ScalpingStrategy(config),
            Strategy.DOBBING: DobbingStrategy(config),
            Strategy.LAY_TO_BACK: LayToBackStrategy(config),
            Strategy.BACK_TO_LAY: BackToLayStrategy(config),
            Strategy.MARKET_MAKING: AvellanedaStoikovStrategy(config),
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
            self.outcome_logger.close()

    def _run_loop(self) -> None:
        """Guardian-style loop: scan markets, filter, analyze, execute."""
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

                # 2. Filter and process each market
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
            "market_name": "3:30 Ascot",
            "event": "Ascot",
            "start_time": dt.datetime.utcnow() + dt.timedelta(minutes=30),
            "race_class": 3,
            "race_type": "flat",
            "is_maiden": False,
            "is_handicap": True,
            "runners": [
                {"id": 101, "name": "Frankel Junior"},
                {"id": 102, "name": "Desert Crown"},
                {"id": 103, "name": "Baaeed Star"},
                {"id": 104, "name": "Enable Legacy"},
                {"id": 105, "name": "Galileo Gold"},
                {"id": 106, "name": "Dancing Brave"},
                {"id": 107, "name": "Mill Reef"},
                {"id": 108, "name": "Nijinsky"},
            ],
        }]

    def _process_market(self, market: dict) -> None:
        """Analyze a single market through the full pipeline."""
        market_id = market["market_id"]
        market_name = market["market_name"]

        # Get current prices
        if self.config.paper_trading:
            snapshots = self._paper_snapshots(market)
        else:
            snapshots = self.client.get_market_book(market_id)

        if not snapshots:
            return

        # Calculate total matched volume for filtering
        total_matched = sum(s.total_matched for s in snapshots)

        # ── Race Selection Filter ──────────────────────────────────────
        filter_result = self.race_selector.filter_market(market, total_matched)
        if not filter_result.passed:
            logger.debug("Filtered out: %s — %s", market_name, ", ".join(filter_result.reasons))
            return

        logger.info("Processing market: %s (%s) — passed filters", market_name, market_id)

        # ── Record prices and trades for all signal estimators ─────────
        runner_names = {r["id"]: r["name"] for r in market.get("runners", [])}
        now = time.time()
        for snap in snapshots:
            if snap.last_traded_price > 0:
                self.steam_detector.record_price(
                    market_id, snap.selection_id,
                    snap.last_traded_price, snap.total_matched,
                )
                # Feed Kyle's lambda estimator
                is_buy = snap.best_back > 0 and snap.last_traded_price >= snap.best_back
                self.kyle_estimator.record_trade(
                    market_id, snap.last_traded_price, snap.total_matched, is_buy, now,
                )
                # Feed Hawkes process estimator
                self.hawkes_estimator.record_event(market_id, now)
                # Feed VPIN calculator
                self.vpin_calculator.record_trade(
                    market_id, snap.total_matched, is_buy, now,
                )
        self.steam_detector.set_estimated_daily_volume(market_id, total_matched)

        # ── Compute quant signals ──────────────────────────────────────
        kyle_estimate = self.kyle_estimator.estimate(market_id)
        hawkes_estimate = self.hawkes_estimator.estimate(market_id)
        vpin_estimate = self.vpin_calculator.compute(market_id)

        # VPIN safety check: if critical, skip this market entirely
        if vpin_estimate.is_critical:
            logger.warning(
                "SKIPPING %s — VPIN critical (%.3f), informed money dominating",
                market_name, vpin_estimate.vpin,
            )
            return

        # Log quant signal state
        if kyle_estimate.is_reliable or hawkes_estimate.is_reliable:
            logger.info(
                "Quant signals [%s]: %s | %s | %s",
                market_name,
                kyle_estimate.to_prompt_text() if kyle_estimate.is_reliable else "Lambda: N/A",
                hawkes_estimate.to_prompt_text() if hawkes_estimate.is_reliable else "Hawkes: N/A",
                vpin_estimate.to_prompt_text(),
            )

        # ── Fetch form data (Racing API) ───────────────────────────────
        form_data_text = ""
        if self._racing_api:
            form_data_text = self._fetch_form_data(market)

        # ── Get steam/drift signals ────────────────────────────────────
        steam_signals = self.steam_detector.get_signals(market_id, runner_names)
        steam_text_lines = []
        if steam_signals:
            runner_lines = [s.to_prompt_text() for s in steam_signals if s.movement.value != "STABLE"]
            if runner_lines:
                steam_text_lines.extend(runner_lines)

        # Append quant signal summaries
        if kyle_estimate.is_reliable:
            steam_text_lines.append(kyle_estimate.to_prompt_text())
        if hawkes_estimate.is_reliable:
            steam_text_lines.append(hawkes_estimate.to_prompt_text())
        steam_text_lines.append(vpin_estimate.to_prompt_text())

        steam_text = "\n".join(steam_text_lines) if steam_text_lines else ""

        # ── Get pre-race content signals ───────────────────────────────
        content_signals_text = ""
        content_sentiments_text = ""
        if self._content_scraper:
            content_signals_text, content_sentiments_text = self._fetch_content_signals(market)

        # ── Two-stage Claude analysis ──────────────────────────────────
        instructions = self.analyst.analyze_market(
            market, snapshots,
            form_data_text=form_data_text,
            steam_signals_text=steam_text,
            content_signals_text=content_signals_text,
            content_sentiments_text=content_sentiments_text,
        )

        # ── Execute qualifying trades ──────────────────────────────────
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
                self._execute_orders(instruction, orders, market, snapshots)

    def _fetch_form_data(self, market: dict) -> str:
        """Fetch form data from Racing API and format for Stage 1 prompt."""
        if not self._racing_api:
            return ""

        try:
            race_id = market.get("race_id", market.get("market_id", ""))
            race_card = self._racing_api.get_race_card(race_id)
            if race_card:
                return race_card.to_prompt_text()
        except Exception as e:
            logger.error("Failed to fetch form data: %s", e)

        return ""

    def _fetch_content_signals(self, market: dict) -> tuple[str, str]:
        """Fetch pre-race content and format for Claude prompts.

        Returns (content_for_stage1, sentiments_for_stage2).
        """
        if not self._content_scraper:
            return "", ""

        try:
            course = market.get("event", "")
            race_time = str(market.get("start_time", ""))
            runner_names = [r["name"] for r in market.get("runners", [])]

            content_by_runner = self._content_scraper.get_all_content(
                course, race_time, runner_names
            )

            # Format raw content for Stage 1
            content_lines = []
            for name, items in content_by_runner.items():
                for item in items:
                    content_lines.append(f"  [{item.source}] {name}: \"{item.text[:200]}\"")

            content_text = "\n".join(content_lines) if content_lines else ""

            # Sentiments text placeholder — would be filled by Claude classification
            # in a full implementation. For now, pass raw content.
            return content_text, content_text

        except Exception as e:
            logger.error("Failed to fetch content signals: %s", e)
            return "", ""

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
        self,
        instruction: TradeInstruction,
        orders: list[Order],
        market: dict,
        snapshots: list[MarketSnapshot],
    ) -> None:
        """Place orders, log the trade, and record for outcome tracking."""
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

        # Log for outcome validation
        bet_record = BetRecord(
            bet_id=str(uuid.uuid4()),
            race_id=instruction.market_id,
            course=market.get("event", ""),
            race_time=str(market.get("start_time", "")),
            race_class=market.get("race_class", 0),
            field_size=len(market.get("runners", [])),
            going=market.get("going", ""),
            race_type=market.get("race_type", ""),
            horse_name=instruction.selection,
            model_probability=instruction.p,
            market_implied_probability=instruction.market_p,
            edge_at_entry=instruction.edge,
            stake=orders[0].size if orders else 0,
            entry_odds=instruction.entry_odds,
            strategy=instruction.strategy.value,
            confidence=instruction.confidence.value,
            signals=SignalSnapshot(
                form_probability=instruction.p,
                signals_aligned=len(instruction.signals_used),
            ),
        )
        self.outcome_logger.log_bet(bet_record)

        logger.info("Trade executed: %s %s on %s (edge=%.3f, confidence=%s, signals=%s)",
                     instruction.strategy.value, instruction.signal.value,
                     instruction.selection, instruction.edge,
                     instruction.confidence.value,
                     ", ".join(instruction.signals_used) or "none")

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
