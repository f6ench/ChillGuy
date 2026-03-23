"""Kyle's Lambda — price impact estimation from order flow.

Estimates how much a unit of signed order flow moves the price.
High lambda = informed traders are active (large orders move price).
Low lambda = liquid market where big orders are noise.

Based on Kyle (1985): p = mu + lambda * y
where y is aggregate signed order flow.

On Betfair: if lambda > 0.002, informed money is present.
Widen spread or stay out.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# Thresholds
HIGH_LAMBDA = 0.002  # Above this = high informed trading
NORMAL_LAMBDA = 0.001
MIN_SAMPLES = 30  # Minimum trades before estimation is reliable


@dataclass
class LambdaEstimate:
    """Result of Kyle's lambda estimation."""

    lambda_value: float
    r_squared: float
    std_error: float
    p_value: float
    sample_size: int
    is_reliable: bool  # True if enough data and statistically significant

    @property
    def informed_trading(self) -> bool:
        """True if lambda suggests informed traders are active."""
        return self.is_reliable and self.lambda_value > HIGH_LAMBDA

    @property
    def interpretation(self) -> str:
        if not self.is_reliable:
            return "Insufficient data"
        if self.lambda_value > HIGH_LAMBDA:
            return "High informed trading — widen spread"
        if self.lambda_value > NORMAL_LAMBDA:
            return "Moderate information flow"
        return "Normal liquidity"

    def to_prompt_text(self) -> str:
        return (
            f"Kyle's Lambda: {self.lambda_value:.6f} "
            f"(R²={self.r_squared:.3f}, n={self.sample_size}) "
            f"— {self.interpretation}"
        )


@dataclass
class TradeObservation:
    """A single trade observation for lambda estimation."""

    price: float
    volume: float
    sign: int  # +1 = buy (back), -1 = sell (lay)
    timestamp: float = 0.0


class KyleLambdaEstimator:
    """Estimates Kyle's lambda from a rolling window of trades.

    Lambda is estimated via regression:
        delta_p_t = lambda * Q_t + epsilon_t

    where Q_t is the signed order flow (volume * sign) and
    delta_p_t is the price change.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        # market_id -> deque of TradeObservations
        self._trades: dict[str, deque[TradeObservation]] = {}

    def record_trade(
        self,
        market_id: str,
        price: float,
        volume: float,
        is_buy: bool,
        timestamp: float = 0.0,
    ) -> None:
        """Record a trade observation."""
        if market_id not in self._trades:
            self._trades[market_id] = deque(maxlen=self.window_size)

        sign = 1 if is_buy else -1
        self._trades[market_id].append(
            TradeObservation(price=price, volume=volume, sign=sign, timestamp=timestamp)
        )

    def estimate(self, market_id: str) -> LambdaEstimate:
        """Estimate Kyle's lambda for a market.

        Uses OLS regression: delta_p = lambda * signed_volume + epsilon
        """
        trades = self._trades.get(market_id)
        if not trades or len(trades) < MIN_SAMPLES:
            return LambdaEstimate(
                lambda_value=0.0, r_squared=0.0, std_error=0.0,
                p_value=1.0, sample_size=len(trades) if trades else 0,
                is_reliable=False,
            )

        prices = np.array([t.price for t in trades])
        volumes = np.array([t.volume for t in trades])
        signs = np.array([t.sign for t in trades])

        # Signed order flow
        signed_volume = volumes * signs

        # Price changes
        price_changes = np.diff(prices)

        # Align arrays (signed_volume[1:] corresponds to price_changes)
        x = signed_volume[1:]
        y = price_changes

        # Drop zero-change ticks
        mask = y != 0
        if mask.sum() < MIN_SAMPLES // 2:
            return LambdaEstimate(
                lambda_value=0.0, r_squared=0.0, std_error=0.0,
                p_value=1.0, sample_size=len(trades),
                is_reliable=False,
            )

        x_filtered = x[mask]
        y_filtered = y[mask]

        slope, _intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

        estimate = LambdaEstimate(
            lambda_value=slope,
            r_squared=r_value ** 2,
            std_error=std_err,
            p_value=p_value,
            sample_size=len(trades),
            is_reliable=(p_value < 0.05 and len(trades) >= MIN_SAMPLES),
        )

        if estimate.informed_trading:
            logger.warning(
                "Market %s: HIGH informed trading detected (lambda=%.6f, R²=%.3f)",
                market_id, slope, r_value ** 2,
            )

        return estimate

    def clear_market(self, market_id: str) -> None:
        self._trades.pop(market_id, None)
