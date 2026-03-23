"""Betfair Streaming API integration for live market data."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from betfair_trader.client import BetfairClient
from betfair_trader.models import MarketSnapshot

logger = logging.getLogger(__name__)


class MarketStream:
    """Manages a streaming connection to Betfair for live odds updates.

    Uses betfairlightweight's streaming support to receive real-time
    price/volume changes via the Betfair Streaming API.
    """

    def __init__(
        self,
        client: BetfairClient,
        market_ids: list[str],
        on_update: Optional[Callable[[str, list[MarketSnapshot]], None]] = None,
        poll_interval: float = 0.2,
    ):
        self.client = client
        self.market_ids = market_ids
        self.on_update = on_update
        self.poll_interval = poll_interval
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: dict[str, list[MarketSnapshot]] = {}

    def start(self) -> None:
        """Start streaming market data."""
        try:
            self._stream = self.client.api.streaming.create_stream(
                listener=self._create_listener(),
            )
            market_filter = {"marketIds": self.market_ids}
            market_data_filter = {
                "fields": ["EX_BEST_OFFERS", "EX_TRADED"],
                "ladderLevels": 3,
            }
            self._stream.subscribe_to_markets(
                market_filter=market_filter,
                market_data_filter=market_data_filter,
            )
            self._stream.start()
            self._running = True
            logger.info("Stream started for markets: %s", self.market_ids)
        except Exception as e:
            logger.warning("Streaming not available, falling back to polling: %s", e)
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _create_listener(self):
        """Create a betfairlightweight streaming listener."""
        from betfairlightweight.streaming import MarketListener
        return MarketListener(max_latency=0.5)

    def _poll_loop(self) -> None:
        """Fallback: poll market book at regular intervals."""
        while self._running:
            for market_id in self.market_ids:
                try:
                    snapshots = self.client.get_market_book(market_id)
                    self._latest[market_id] = snapshots
                    if self.on_update:
                        self.on_update(market_id, snapshots)
                except Exception as e:
                    logger.error("Poll error for %s: %s", market_id, e)
            time.sleep(self.poll_interval)

    def get_latest(self, market_id: str) -> list[MarketSnapshot]:
        """Get the most recent snapshots for a market."""
        return self._latest.get(market_id, [])

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Stream stopped")
