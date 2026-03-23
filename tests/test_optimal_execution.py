"""Tests for Almgren-Chriss optimal execution."""

from betfair_trader.execution.optimal_execution import (
    ExecutionParams,
    compute_execution_schedule,
    quick_execution_plan,
)


def test_schedule_sums_to_total():
    """All trade sizes should sum to the total position."""
    params = ExecutionParams(
        total_size=10_000,
        T=4.0,
        N=8,
        sigma=0.02,
        eta=0.001,
        gamma_temp=0.005,
        risk_aversion=1e-6,
    )
    schedule = compute_execution_schedule(params)

    total_traded = sum(s.trade_size for s in schedule.slices)
    assert abs(total_traded - 10_000) < 1.0  # within £1 of total


def test_high_urgency_front_loads():
    """High risk aversion should front-load execution."""
    params = ExecutionParams(
        total_size=10_000,
        T=4.0,
        N=8,
        sigma=0.02,
        eta=0.001,
        gamma_temp=0.005,
        risk_aversion=1.0,  # very high risk aversion
    )
    schedule = compute_execution_schedule(params)

    # First slice should be much larger than last slice
    first = schedule.slices[0].trade_size
    last = schedule.slices[-1].trade_size
    assert first > last * 1.5  # front-loaded


def test_low_urgency_near_twap():
    """Low risk aversion should produce near-uniform (TWAP) schedule."""
    params = ExecutionParams(
        total_size=10_000,
        T=4.0,
        N=8,
        sigma=0.02,
        eta=0.001,
        gamma_temp=0.005,
        risk_aversion=1e-10,  # essentially zero
    )
    schedule = compute_execution_schedule(params)

    # All slices should be roughly equal
    sizes = [s.trade_size for s in schedule.slices]
    avg = sum(sizes) / len(sizes)
    for s in sizes:
        assert abs(s - avg) / avg < 0.3  # within 30% of mean


def test_implementation_shortfall_positive():
    """Expected cost should be positive."""
    params = ExecutionParams(
        total_size=10_000,
        T=4.0,
        N=8,
        sigma=0.02,
        eta=0.001,
        gamma_temp=0.005,
        risk_aversion=1e-6,
    )
    schedule = compute_execution_schedule(params)
    assert schedule.expected_cost > 0
    assert schedule.implementation_shortfall > 0


def test_quick_plan():
    """Quick execution plan should produce valid schedule."""
    schedule = quick_execution_plan(
        total_size=5000,
        hours=2.0,
        kyle_lambda=0.001,
    )
    assert len(schedule.slices) > 0
    total = sum(s.trade_size for s in schedule.slices)
    assert abs(total - 5000) < 1.0


def test_cumulative_pct():
    """Cumulative percentage should end at ~100%."""
    schedule = quick_execution_plan(total_size=1000, hours=1.0)
    last_slice = schedule.slices[-1]
    assert abs(last_slice.cumulative_pct - 1.0) < 0.01
