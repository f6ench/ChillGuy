"""Tests for the hedge calculator."""

from betfair_trader.utils.hedge import (
    calculate_back_to_lay_hedge,
    calculate_lay_to_back_hedge,
    kelly_stake,
)


def test_back_to_lay_profitable():
    """Back at 5.0, lay at 3.0 should be profitable."""
    result = calculate_back_to_lay_hedge(
        back_stake=10.0,
        back_odds=5.0,
        lay_odds=3.0,
        commission=0.05,
    )
    assert result.is_profitable
    assert result.guaranteed_profit > 0
    assert result.hedge_stake > 0


def test_back_to_lay_unprofitable():
    """Back at 3.0, lay at 5.0 (price drifted) should lose."""
    result = calculate_back_to_lay_hedge(
        back_stake=10.0,
        back_odds=3.0,
        lay_odds=5.0,
        commission=0.05,
    )
    assert not result.is_profitable


def test_lay_to_back_profitable():
    """Lay at 3.0, back at 5.0 should be profitable."""
    result = calculate_lay_to_back_hedge(
        lay_stake=10.0,
        lay_odds=3.0,
        back_odds=5.0,
        commission=0.05,
    )
    assert result.is_profitable
    assert result.guaranteed_profit > 0


def test_kelly_positive_edge():
    """With positive edge, Kelly should return non-zero stake."""
    stake = kelly_stake(probability=0.6, odds=3.0, fraction=0.25, bankroll=100.0)
    assert stake > 0


def test_kelly_no_edge():
    """With no edge, Kelly should return zero."""
    stake = kelly_stake(probability=0.3, odds=3.0, fraction=0.25, bankroll=100.0)
    assert stake == 0.0


def test_kelly_negative_edge():
    """With negative edge, Kelly should return zero."""
    stake = kelly_stake(probability=0.1, odds=3.0, fraction=0.25, bankroll=100.0)
    assert stake == 0.0
