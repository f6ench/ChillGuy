"""Almgren-Chriss optimal execution for large orders.

When you want to buy/sell a large position, dumping it all at once
causes massive price impact. This module computes the optimal schedule
to split the order across time intervals, minimizing total transaction
costs while managing market risk.

The key tradeoff: execute fast (high market risk but low timing risk)
vs execute slowly (low impact but the price may move against you).

Based on Almgren & Chriss (2001): "Optimal execution of portfolio
transactions."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionParams:
    """Parameters for optimal execution."""

    total_size: float  # Total position size to execute (in GBP)
    T: float  # Execution horizon (hours)
    N: int  # Number of intervals
    sigma: float  # Price volatility (per hour)
    eta: float  # Permanent impact coefficient (GBP per GBP of volume)
    gamma_temp: float  # Temporary impact coefficient
    risk_aversion: float  # 0 = ignore risk (TWAP), high = trade fast (front-loaded)


@dataclass
class ExecutionSlice:
    """A single slice of the execution schedule."""

    time: float  # hours from start
    trade_size: float  # GBP to trade in this interval
    remaining: float  # GBP remaining after this slice
    cumulative_pct: float  # % of total executed so far


@dataclass
class ExecutionSchedule:
    """Complete optimal execution schedule."""

    slices: list[ExecutionSlice]
    expected_cost: float  # total expected transaction cost (GBP)
    implementation_shortfall: float  # cost as fraction of total size
    kappa: float  # urgency parameter
    urgency: str  # human-readable urgency level
    permanent_impact: float
    temporary_impact: float
    variance_cost: float

    def to_prompt_text(self) -> str:
        lines = [
            f"Execution Schedule: {len(self.slices)} slices over {self.slices[-1].time:.1f}h",
            f"  Urgency: {self.urgency} (kappa={self.kappa:.4f})",
            f"  Expected cost: £{self.expected_cost:.2f} "
            f"({self.implementation_shortfall:.3%} of total)",
            f"  Permanent impact: £{self.permanent_impact:.2f}",
            f"  Temporary impact: £{self.temporary_impact:.2f}",
        ]
        return "\n".join(lines)


def compute_execution_schedule(params: ExecutionParams) -> ExecutionSchedule:
    """Compute the Almgren-Chriss optimal execution trajectory.

    The optimal remaining position at step k:
        X_k = X_0 * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    where kappa = sqrt(risk_aversion * sigma^2 / eta)
    """
    tau = params.T / params.N
    if params.eta <= 0:
        params.eta = 1e-6  # avoid division by zero

    kappa_sq = params.risk_aversion * params.sigma ** 2 / params.eta
    kappa = np.sqrt(max(kappa_sq, 1e-10))

    times = np.linspace(0, params.T, params.N + 1)

    # Optimal remaining position
    sinh_kT = np.sinh(kappa * params.T)
    if abs(sinh_kT) < 1e-10:
        # Very low urgency: uniform (TWAP)
        remaining = params.total_size * (1 - times / params.T)
    else:
        remaining = params.total_size * np.sinh(kappa * (params.T - times)) / sinh_kT

    # Trade sizes per interval
    trade_sizes = -np.diff(remaining)

    # Cost components
    temporary_impact = params.gamma_temp * np.sum(trade_sizes ** 2) / tau
    permanent_impact = 0.5 * params.eta * params.total_size ** 2
    variance_cost = params.sigma ** 2 * tau * np.sum(remaining[:-1] ** 2)
    total_cost = temporary_impact + permanent_impact

    # Build schedule
    cumulative = 0.0
    slices = []
    for i in range(params.N):
        cumulative += trade_sizes[i]
        slices.append(ExecutionSlice(
            time=times[i],
            trade_size=trade_sizes[i],
            remaining=remaining[i + 1],
            cumulative_pct=cumulative / params.total_size,
        ))

    # Urgency classification
    if kappa > 2.0:
        urgency = "High — front-load execution"
    elif kappa > 0.5:
        urgency = "Moderate"
    else:
        urgency = "Low — spread evenly (near TWAP)"

    schedule = ExecutionSchedule(
        slices=slices,
        expected_cost=total_cost,
        implementation_shortfall=total_cost / params.total_size if params.total_size > 0 else 0,
        kappa=kappa,
        urgency=urgency,
        permanent_impact=permanent_impact,
        temporary_impact=temporary_impact,
        variance_cost=variance_cost,
    )

    logger.info(
        "Execution schedule: £%.0f over %.1fh, %d slices, cost £%.2f (%.3f%%)",
        params.total_size, params.T, params.N,
        total_cost, schedule.implementation_shortfall * 100,
    )

    return schedule


def quick_execution_plan(
    total_size: float,
    hours: float = 2.0,
    kyle_lambda: float = 0.001,
    volatility: float = 0.02,
    risk_aversion: float = 1e-6,
) -> ExecutionSchedule:
    """Quick helper to generate an execution plan from minimal inputs.

    Uses Kyle's lambda to derive impact parameters.
    """
    # Derive Almgren-Chriss params from Kyle's lambda
    # eta (permanent) ~ lambda, gamma (temporary) ~ 5 * lambda
    eta = kyle_lambda
    gamma = kyle_lambda * 5

    n_intervals = max(4, int(hours * 2))  # 2 intervals per hour minimum

    params = ExecutionParams(
        total_size=total_size,
        T=hours,
        N=n_intervals,
        sigma=volatility,
        eta=eta,
        gamma_temp=gamma,
        risk_aversion=risk_aversion,
    )

    return compute_execution_schedule(params)
