"""Betfair API client wrapper around betfairlightweight."""

from __future__ import annotations

import logging
from typing import Optional

import betfairlightweight
from betfairlightweight import APIClient
from betfairlightweight.filters import (
    market_filter,
    price_projection,
    ex_best_offers_overrides,
)

from betfair_trader.config import BetfairConfig
from betfair_trader.models import MarketSnapshot, Order, Side

logger = logging.getLogger(__name__)


class BetfairClient:
    """Handles authentication and API calls to Betfair Exchange."""

    def __init__(self, config: BetfairConfig):
        self.config = config
        self.api: APIClient = betfairlightweight.APIClient(
            username=config.username,
            password=config.password,
            app_key=config.app_key,
            certs=config.certs_path,
        )
        self._logged_in = False

    def login(self) -> bool:
        """Authenticate with Betfair using certificate-based login."""
        try:
            self.api.login()
            self._logged_in = True
            logger.info("Successfully logged in to Betfair")
            return True
        except Exception as e:
            logger.error("Betfair login failed: %s", e)
            return False

    def keep_alive(self) -> None:
        """Refresh the session token."""
        if self._logged_in:
            self.api.keep_alive()

    def logout(self) -> None:
        """End the Betfair session."""
        if self._logged_in:
            self.api.logout()
            self._logged_in = False

    def list_horse_racing_markets(
        self,
        hours_ahead: int = 4,
    ) -> list[dict]:
        """Find upcoming horse racing markets."""
        import datetime as dt

        now = dt.datetime.utcnow()
        time_filter = betfairlightweight.filters.time_range(
            from_=now,
            to=now + dt.timedelta(hours=hours_ahead),
        )
        mf = market_filter(
            event_type_ids=["7"],  # Horse racing
            market_type_codes=["WIN"],
            market_start_time=time_filter,
        )
        catalogues = self.api.betting.list_market_catalogue(
            filter=mf,
            market_projection=[
                "RUNNER_DESCRIPTION",
                "MARKET_START_TIME",
                "EVENT",
            ],
            max_results=100,
            sort="FIRST_TO_START",
        )
        results = []
        for cat in catalogues:
            results.append({
                "market_id": cat.market_id,
                "market_name": cat.market_name,
                "event": cat.event.name if cat.event else "",
                "start_time": cat.market_start_time,
                "runners": [
                    {"id": r.selection_id, "name": r.runner_name}
                    for r in (cat.runners or [])
                ],
            })
        return results

    def get_market_book(self, market_id: str) -> list[MarketSnapshot]:
        """Get current prices for all runners in a market."""
        pp = price_projection(
            price_data=["EX_BEST_OFFERS", "EX_TRADED"],
            ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=3),
        )
        books = self.api.betting.list_market_book(
            market_ids=[market_id],
            price_projection=pp,
        )
        snapshots = []
        if not books:
            return snapshots

        book = books[0]
        for runner in book.runners:
            back_prices = [
                (p.price, p.size) for p in (runner.ex.available_to_back or [])
            ]
            lay_prices = [
                (p.price, p.size) for p in (runner.ex.available_to_lay or [])
            ]
            snapshots.append(MarketSnapshot(
                market_id=market_id,
                selection_id=runner.selection_id,
                runner_name=str(runner.selection_id),
                back_prices=back_prices,
                lay_prices=lay_prices,
                last_traded_price=runner.last_price_traded or 0.0,
                total_matched=runner.total_matched or 0.0,
            ))
        return snapshots

    def place_order(self, order: Order) -> Optional[str]:
        """Place a single order on Betfair. Returns bet ID or None."""
        if self.config.paper_trading:
            logger.info("[PAPER] Would place %s %s @ %.2f for £%.2f on %s",
                        order.side, order.selection_id, order.price, order.size,
                        order.market_id)
            return f"PAPER-{order.selection_id}-{order.side}"

        limit_order = betfairlightweight.filters.limit_order(
            size=order.size,
            price=order.price,
            persistence_type="LAPSE",
        )
        instruction = betfairlightweight.filters.place_instruction(
            order_type="LIMIT",
            selection_id=order.selection_id,
            side=order.side,
            limit_order=limit_order,
        )
        try:
            result = self.api.betting.place_orders(
                market_id=order.market_id,
                instructions=[instruction],
                customer_strategy_ref=order.reference or None,
            )
            if result.status == "SUCCESS":
                bet_id = result.place_instruction_reports[0].bet_id
                logger.info("Order placed: bet_id=%s", bet_id)
                return bet_id
            else:
                logger.error("Order failed: %s", result.error_code)
                return None
        except Exception as e:
            logger.error("Place order error: %s", e)
            return None

    def cancel_orders(self, market_id: str) -> bool:
        """Cancel all unmatched orders in a market."""
        if self.config.paper_trading:
            logger.info("[PAPER] Would cancel all orders on %s", market_id)
            return True
        try:
            self.api.betting.cancel_orders(market_id=market_id)
            return True
        except Exception as e:
            logger.error("Cancel orders error: %s", e)
            return False
