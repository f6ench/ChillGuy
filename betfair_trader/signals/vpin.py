"""VPIN — Volume-synchronized Probability of Informed Trading.

Detects adverse selection: when informed traders are hitting your quotes,
you systematically sell cheap and buy expensive.

VPIN = |V_buy - V_sell| / (V_buy + V_sell)

Thresholds:
- < 0.40: Normal market, safe to quote
- 0.40 - 0.65: Elevated, consider widening spread
- 0.65 - 0.80: High informed trading, double your spread
- > 0.80: Pull quotes entirely

Based on Easley et al. (2012): "Flow Toxicity and Liquidity in a
High-frequency World."
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Thresholds
VPIN_ELEVATED = 0.40
VPIN_HIGH = 0.65
VPIN_CRITICAL = 0.80
DEFAULT_BUCKET_SIZE = 50
DEFAULT_N_BUCKETS = 10


@dataclass
class VPINEstimate:
    """Result of VPIN computation."""

    vpin: float
    n_buckets: int
    total_buy_volume: float
    total_sell_volume: float
    sample_size: int  # number of trades in the window

    @property
    def is_elevated(self) -> bool:
        return self.vpin > VPIN_ELEVATED

    @property
    def is_high(self) -> bool:
        return self.vpin > VPIN_HIGH

    @property
    def is_critical(self) -> bool:
        return self.vpin > VPIN_CRITICAL

    @property
    def spread_multiplier(self) -> float:
        """How much to multiply the base spread by."""
        if self.vpin > VPIN_CRITICAL:
            return 0.0  # pull quotes
        if self.vpin > VPIN_HIGH:
            return 2.0  # double spread
        if self.vpin > VPIN_ELEVATED:
            return 1.5  # widen 50%
        return 1.0  # normal

    @property
    def interpretation(self) -> str:
        if self.vpin > VPIN_CRITICAL:
            return "CRITICAL — pull quotes, informed money dominating"
        if self.vpin > VPIN_HIGH:
            return "HIGH — double spread, likely adverse selection"
        if self.vpin > VPIN_ELEVATED:
            return "Elevated — widen spread 50%"
        return "Normal — safe to quote"

    def to_prompt_text(self) -> str:
        return (
            f"VPIN: {self.vpin:.3f} (spread x{self.spread_multiplier:.1f}) "
            f"— {self.interpretation}"
        )


@dataclass
class TradeVolume:
    """A single trade's volume with buy/sell classification."""

    volume: float
    is_buy: bool
    timestamp: float = 0.0


class VPINCalculator:
    """Computes VPIN from a rolling window of trades.

    Trades are accumulated into volume buckets, then VPIN is
    computed across the last N buckets.
    """

    def __init__(
        self,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
        n_buckets: int = DEFAULT_N_BUCKETS,
        max_trades: int = 2000,
    ):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        # market_id -> deque of TradeVolumes
        self._trades: dict[str, deque[TradeVolume]] = {}
        self._max_trades = max_trades

    def record_trade(
        self,
        market_id: str,
        volume: float,
        is_buy: bool,
        timestamp: float = 0.0,
    ) -> None:
        """Record a trade observation."""
        if market_id not in self._trades:
            self._trades[market_id] = deque(maxlen=self._max_trades)
        self._trades[market_id].append(
            TradeVolume(volume=volume, is_buy=is_buy, timestamp=timestamp)
        )

    def compute(self, market_id: str) -> VPINEstimate:
        """Compute VPIN for a market from the trade window."""
        trades = self._trades.get(market_id)
        if not trades:
            return VPINEstimate(vpin=0.0, n_buckets=0, total_buy_volume=0,
                                total_sell_volume=0, sample_size=0)

        trade_list = list(trades)
        n = len(trade_list)
        min_trades = self.bucket_size * self.n_buckets

        if n < min_trades:
            # Not enough data for full VPIN, compute simple imbalance
            buy_vol = sum(t.volume for t in trade_list if t.is_buy)
            sell_vol = sum(t.volume for t in trade_list if not t.is_buy)
            total = buy_vol + sell_vol
            simple_vpin = abs(buy_vol - sell_vol) / total if total > 0 else 0
            return VPINEstimate(
                vpin=simple_vpin, n_buckets=0,
                total_buy_volume=buy_vol, total_sell_volume=sell_vol,
                sample_size=n,
            )

        # Use the most recent trades
        recent = trade_list[-min_trades:]

        # Split into volume buckets and compute VPIN
        vpin_values = []
        total_buy = 0.0
        total_sell = 0.0

        for bucket_idx in range(self.n_buckets):
            start = bucket_idx * self.bucket_size
            end = start + self.bucket_size
            bucket = recent[start:end]

            v_buy = sum(t.volume for t in bucket if t.is_buy)
            v_sell = sum(t.volume for t in bucket if not t.is_buy)
            v_total = v_buy + v_sell

            total_buy += v_buy
            total_sell += v_sell

            if v_total > 0:
                vpin_values.append(abs(v_buy - v_sell) / v_total)

        if not vpin_values:
            return VPINEstimate(vpin=0.0, n_buckets=0,
                                total_buy_volume=total_buy,
                                total_sell_volume=total_sell,
                                sample_size=n)

        avg_vpin = float(np.mean(vpin_values))

        estimate = VPINEstimate(
            vpin=avg_vpin,
            n_buckets=len(vpin_values),
            total_buy_volume=total_buy,
            total_sell_volume=total_sell,
            sample_size=n,
        )

        if estimate.is_critical:
            logger.warning(
                "Market %s: VPIN CRITICAL %.3f — pull quotes",
                market_id, avg_vpin,
            )
        elif estimate.is_high:
            logger.warning(
                "Market %s: VPIN HIGH %.3f — double spread",
                market_id, avg_vpin,
            )

        return estimate

    def clear_market(self, market_id: str) -> None:
        self._trades.pop(market_id, None)
