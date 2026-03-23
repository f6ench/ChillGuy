"""Greening up / hedge calculator.

Calculates the optimal hedge bet to lock in profit (or minimize loss)
regardless of outcome. This is the core math behind "greening up" on
the Betfair Exchange.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HedgeResult:
    hedge_stake: float
    profit_if_wins: float
    profit_if_loses: float
    guaranteed_profit: float

    @property
    def is_profitable(self) -> bool:
        return self.guaranteed_profit > 0


def calculate_back_to_lay_hedge(
    back_stake: float,
    back_odds: float,
    lay_odds: float,
    commission: float = 0.05,
) -> HedgeResult:
    """Calculate lay hedge after backing a selection.

    You backed at `back_odds`. Price has shortened to `lay_odds`.
    Calculate the lay stake to green up across all outcomes.

    Args:
        back_stake: Original back stake
        back_odds: Odds at which you backed
        lay_odds: Current lay odds to hedge at
        commission: Betfair commission rate (default 5%)
    """
    back_profit = back_stake * (back_odds - 1)

    # Lay stake to equalize profit across outcomes
    lay_stake = (back_stake * back_odds) / lay_odds

    # If selection wins: back profit - lay liability
    lay_liability = lay_stake * (lay_odds - 1)
    profit_if_wins = back_profit - lay_liability

    # If selection loses: lay stake winnings - back stake lost
    profit_if_loses = lay_stake - back_stake

    # Apply commission to the winning side
    profit_if_wins_after_comm = profit_if_wins * (1 - commission)
    profit_if_loses_after_comm = profit_if_loses * (1 - commission)

    # Guaranteed profit is the minimum of both outcomes
    guaranteed = min(profit_if_wins_after_comm, profit_if_loses_after_comm)

    return HedgeResult(
        hedge_stake=round(lay_stake, 2),
        profit_if_wins=round(profit_if_wins_after_comm, 2),
        profit_if_loses=round(profit_if_loses_after_comm, 2),
        guaranteed_profit=round(guaranteed, 2),
    )


def calculate_lay_to_back_hedge(
    lay_stake: float,
    lay_odds: float,
    back_odds: float,
    commission: float = 0.05,
) -> HedgeResult:
    """Calculate back hedge after laying a selection.

    You laid at `lay_odds`. Price has drifted to `back_odds`.
    Calculate the back stake to green up.

    Args:
        lay_stake: Original lay stake
        lay_odds: Odds at which you laid
        back_odds: Current back odds to hedge at
        commission: Betfair commission rate (default 5%)
    """
    lay_liability = lay_stake * (lay_odds - 1)

    # Back stake to equalize
    back_stake = (lay_stake * lay_odds) / back_odds

    # If selection wins: back profit - lay liability
    back_profit = back_stake * (back_odds - 1)
    profit_if_wins = back_profit - lay_liability

    # If selection loses: lay winnings - back stake lost
    profit_if_loses = lay_stake - back_stake

    profit_if_wins_after_comm = profit_if_wins * (1 - commission)
    profit_if_loses_after_comm = profit_if_loses * (1 - commission)

    guaranteed = min(profit_if_wins_after_comm, profit_if_loses_after_comm)

    return HedgeResult(
        hedge_stake=round(back_stake, 2),
        profit_if_wins=round(profit_if_wins_after_comm, 2),
        profit_if_loses=round(profit_if_loses_after_comm, 2),
        guaranteed_profit=round(guaranteed, 2),
    )


def kelly_stake(
    probability: float,
    odds: float,
    fraction: float = 0.25,
    bankroll: float = 100.0,
) -> float:
    """Fractional Kelly criterion for position sizing.

    f = (b*p - (1-p)) / b
    where b = odds - 1, p = probability

    Args:
        probability: Our estimated probability of winning
        odds: Decimal odds offered
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
        bankroll: Current bankroll
    """
    b = odds - 1
    if b <= 0:
        return 0.0

    f = (b * probability - (1 - probability)) / b
    if f <= 0:
        return 0.0  # Negative Kelly = no edge, don't bet

    return round(f * fraction * bankroll, 2)
