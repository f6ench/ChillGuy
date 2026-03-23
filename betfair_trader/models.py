"""Core data models for the trading system."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Signal(str, Enum):
    SPRINGER = "springer"
    DRIFTER = "drifter"
    SCALP = "scalp"
    DOB = "dob"


class Strategy(str, Enum):
    BACK_TO_LAY = "back_to_lay"
    LAY_TO_BACK = "lay_to_back"
    SCALPING = "scalping"
    DOBBING = "dobbing"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Side(str, Enum):
    BACK = "BACK"
    LAY = "LAY"


class TradeInstruction(BaseModel):
    """Claude's output per race — the core trading signal."""

    race: str
    selection: str
    selection_id: int = 0
    market_id: str = ""
    signal: Signal
    p: float = Field(ge=0, le=1, description="Our probability estimate")
    market_p: float = Field(ge=0, le=1, description="Implied probability from odds")
    edge: float = Field(description="p - market_p")
    strategy: Strategy
    entry_odds: float
    target_odds: float
    hedge: bool = True
    min_profit: float = 0.0
    confidence: Confidence = Confidence.MEDIUM
    reasoning: str = ""
    signals_used: list[str] = []  # which signals support this trade

    @property
    def ev(self) -> float:
        """Expected value of the trade."""
        payout = self.entry_odds - 1
        return self.p * payout - (1 - self.p)


class Order(BaseModel):
    """A single order to place on Betfair."""

    market_id: str
    selection_id: int
    side: Side
    price: float
    size: float
    reference: str = ""


class TradeRecord(BaseModel):
    """Logged trade for the feedback loop."""

    id: str = ""
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    instruction: TradeInstruction
    orders: list[Order] = []
    entry_matched: bool = False
    hedge_matched: bool = False
    pnl: float = 0.0
    actual_outcome: Optional[str] = None
    notes: str = ""


class MarketSnapshot(BaseModel):
    """Point-in-time snapshot of a runner's market data."""

    market_id: str
    selection_id: int
    runner_name: str
    back_prices: list[tuple[float, float]] = []  # (price, size)
    lay_prices: list[tuple[float, float]] = []
    last_traded_price: float = 0.0
    total_matched: float = 0.0
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    @property
    def best_back(self) -> float:
        return self.back_prices[0][0] if self.back_prices else 0.0

    @property
    def best_lay(self) -> float:
        return self.lay_prices[0][0] if self.lay_prices else 0.0

    @property
    def spread(self) -> float:
        if self.best_back and self.best_lay:
            return self.best_lay - self.best_back
        return 0.0

    @property
    def implied_probability(self) -> float:
        if self.best_back:
            return 1.0 / self.best_back
        return 0.0
