"""Steam & Drift detection from Betfair price movement.

Tracks price history per runner from market open and calculates
steam/drift ratios. A runner shortening significantly with volume
is one of the most reliable signals in UK flat racing — it reflects
informed money entering before the public.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MovementType(str, Enum):
    STEAM = "STEAM"
    DRIFT = "DRIFT"
    STABLE = "STABLE"


@dataclass
class PricePoint:
    """A single price observation."""

    price: float
    volume_matched: float
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass
class SteamSignal:
    """Output signal from the steam/drift detector."""

    selection_id: int
    runner_name: str
    movement: MovementType
    opening_price: float
    current_price: float
    steam_ratio: float  # (opening - current) / opening, positive = shortening
    volume_ratio: float  # volume matched / estimated daily average
    confidence: float  # 0.0 - 1.0
    price_history: list[PricePoint] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format for Claude Stage 2 prompt."""
        direction = "SHORTENING" if self.steam_ratio > 0 else "DRIFTING"
        return (
            f"  {self.runner_name}: {self.movement.value} ({direction}) "
            f"| Open: {self.opening_price:.2f} -> Current: {self.current_price:.2f} "
            f"| Steam Ratio: {self.steam_ratio:+.1%} "
            f"| Volume Ratio: {self.volume_ratio:.1%} "
            f"| Confidence: {self.confidence:.2f}"
        )


# Thresholds
STEAM_THRESHOLD = 0.15  # 15% price shortening
DRIFT_THRESHOLD = -0.20  # 20% price lengthening
VOLUME_THRESHOLD = 0.10  # 10% of estimated daily volume


class SteamDetector:
    """Tracks price movement from market open to detect steam and drift.

    Steam: a runner's price shortening significantly with heavy volume.
    This indicates informed money (syndicates, connections, professionals)
    entering the market before the general public.

    Drift: a runner's price lengthening significantly — negative signal,
    suggests insiders are NOT backing or are actively laying.
    """

    def __init__(
        self,
        steam_threshold: float = STEAM_THRESHOLD,
        drift_threshold: float = DRIFT_THRESHOLD,
        volume_threshold: float = VOLUME_THRESHOLD,
    ):
        self.steam_threshold = steam_threshold
        self.drift_threshold = drift_threshold
        self.volume_threshold = volume_threshold

        # market_id -> selection_id -> list of PricePoints
        self._price_history: dict[str, dict[int, list[PricePoint]]] = {}
        # market_id -> selection_id -> opening price (first observed)
        self._opening_prices: dict[str, dict[int, float]] = {}
        # market_id -> estimated total daily volume
        self._estimated_daily_volume: dict[str, float] = {}

    def record_price(
        self,
        market_id: str,
        selection_id: int,
        price: float,
        volume_matched: float,
    ) -> None:
        """Record a price observation for a runner.

        Should be called on every streaming update / poll cycle.
        """
        if price <= 0:
            return

        if market_id not in self._price_history:
            self._price_history[market_id] = {}
            self._opening_prices[market_id] = {}

        if selection_id not in self._price_history[market_id]:
            self._price_history[market_id][selection_id] = []
            self._opening_prices[market_id][selection_id] = price

        self._price_history[market_id][selection_id].append(
            PricePoint(price=price, volume_matched=volume_matched)
        )

    def set_estimated_daily_volume(self, market_id: str, volume: float) -> None:
        """Set the estimated total daily volume for a market.

        Use the total matched on the market to estimate this.
        """
        self._estimated_daily_volume[market_id] = volume

    def get_signals(
        self,
        market_id: str,
        runner_names: dict[int, str] | None = None,
    ) -> list[SteamSignal]:
        """Calculate steam/drift signals for all runners in a market."""
        if market_id not in self._price_history:
            return []

        runner_names = runner_names or {}
        daily_vol = self._estimated_daily_volume.get(market_id, 0)
        signals = []

        for selection_id, history in self._price_history[market_id].items():
            if len(history) < 2:
                continue

            opening = self._opening_prices[market_id][selection_id]
            current = history[-1].price
            current_volume = history[-1].volume_matched

            if opening <= 0:
                continue

            # Steam ratio: positive means shortening (good signal)
            steam_ratio = (opening - current) / opening

            # Volume ratio: what fraction of daily volume has been matched
            volume_ratio = current_volume / daily_vol if daily_vol > 0 else 0

            # Determine movement type
            if steam_ratio >= self.steam_threshold and volume_ratio >= self.volume_threshold:
                movement = MovementType.STEAM
                confidence = min(1.0, (steam_ratio / self.steam_threshold) * 0.5
                                 + (volume_ratio / self.volume_threshold) * 0.5)
            elif steam_ratio <= self.drift_threshold:
                movement = MovementType.DRIFT
                confidence = min(1.0, abs(steam_ratio) / abs(self.drift_threshold))
            else:
                movement = MovementType.STABLE
                confidence = 0.0

            signals.append(SteamSignal(
                selection_id=selection_id,
                runner_name=runner_names.get(selection_id, str(selection_id)),
                movement=movement,
                opening_price=opening,
                current_price=current,
                steam_ratio=steam_ratio,
                volume_ratio=volume_ratio,
                confidence=confidence,
                price_history=history[-10:],  # last 10 observations
            ))

        return signals

    def get_steam_runners(self, market_id: str) -> list[SteamSignal]:
        """Get only runners flagged as STEAM (shortening with volume)."""
        return [s for s in self.get_signals(market_id) if s.movement == MovementType.STEAM]

    def get_drift_runners(self, market_id: str) -> list[SteamSignal]:
        """Get only runners flagged as DRIFT (lengthening)."""
        return [s for s in self.get_signals(market_id) if s.movement == MovementType.DRIFT]

    def clear_market(self, market_id: str) -> None:
        """Clear all data for a finished market."""
        self._price_history.pop(market_id, None)
        self._opening_prices.pop(market_id, None)
        self._estimated_daily_volume.pop(market_id, None)
