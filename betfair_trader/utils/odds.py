"""Betfair odds ladder utilities.

Betfair uses a fixed odds ladder with specific tick increments.
This module provides helpers for working with valid Betfair prices.
"""

from __future__ import annotations

# Betfair odds ladder: price -> tick increment
TICK_RANGES = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2),
    (10.0, 20.0, 0.5),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1001.0, 10.0),
]


def nearest_valid_price(price: float) -> float:
    """Snap a price to the nearest valid Betfair tick."""
    for low, high, tick in TICK_RANGES:
        if low <= price < high:
            return round(round((price - low) / tick) * tick + low, 2)
    return round(price, 2)


def ticks_away(price: float, n_ticks: int) -> float:
    """Move n ticks from a price (positive = higher odds, negative = lower)."""
    current = nearest_valid_price(price)
    direction = 1 if n_ticks > 0 else -1
    remaining = abs(n_ticks)

    while remaining > 0:
        for low, high, tick in TICK_RANGES:
            if low <= current < high:
                current = round(current + direction * tick, 2)
                break
        current = max(1.01, min(current, 1000.0))
        remaining -= 1

    return nearest_valid_price(current)


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def odds_from_probability(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0:
        return 1000.0
    if prob >= 1:
        return 1.01
    return nearest_valid_price(1.0 / prob)
