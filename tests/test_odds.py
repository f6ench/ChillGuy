"""Tests for odds utilities."""

from betfair_trader.utils.odds import (
    implied_probability,
    nearest_valid_price,
    odds_from_probability,
    ticks_away,
)


def test_nearest_valid_price_low():
    assert nearest_valid_price(1.55) == 1.55
    assert nearest_valid_price(1.556) == 1.56


def test_nearest_valid_price_mid():
    assert nearest_valid_price(4.15) == 4.2
    assert nearest_valid_price(6.3) == 6.2  # Rounds to nearest tick (0.2 increments)


def test_ticks_up():
    assert ticks_away(2.0, 1) == 2.02
    assert ticks_away(2.0, 2) == 2.04


def test_ticks_down():
    assert ticks_away(2.04, -1) == 2.02
    assert ticks_away(2.02, -1) == 2.0


def test_implied_probability():
    assert implied_probability(2.0) == 0.5
    assert implied_probability(4.0) == 0.25


def test_odds_from_probability():
    result = odds_from_probability(0.5)
    assert result == 2.0
