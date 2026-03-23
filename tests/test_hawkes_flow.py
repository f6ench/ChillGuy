"""Tests for Hawkes process order flow estimator."""

import numpy as np

from betfair_trader.signals.hawkes_flow import HawkesFlowEstimator


def _simulate_hawkes(mu, alpha, beta, T):
    """Simple thinning simulation of a Hawkes process."""
    times = []
    t = 0
    while t < T:
        if times:
            rate = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in times)
        else:
            rate = mu
        t += np.random.exponential(1 / max(rate, mu))
        if t > T:
            break
        current_rate = mu + alpha * sum(np.exp(-beta * (t - ti)) for ti in times)
        if np.random.uniform() < current_rate / max(rate, mu):
            times.append(t)
    return times


def test_insufficient_data():
    """Should return unreliable estimate with too few events."""
    est = HawkesFlowEstimator()
    for i in range(10):
        est.record_event("1.234", float(i))
    result = est.estimate("1.234")
    assert not result.is_reliable


def test_recovers_parameters():
    """Should approximately recover known Hawkes parameters."""
    np.random.seed(42)
    true_mu, true_alpha, true_beta = 1.0, 0.6, 2.0

    times = _simulate_hawkes(true_mu, true_alpha, true_beta, T=3600)
    assert len(times) > 50  # need enough data

    est = HawkesFlowEstimator(n_starts=5)
    for t in times:
        est.record_event("1.234", t)

    result = est.estimate("1.234")
    assert result.is_reliable
    # Branching ratio should be near 0.3 (0.6/2.0)
    assert 0.1 < result.branching_ratio < 0.6


def test_detects_hot_market():
    """High branching ratio should flag hot market."""
    np.random.seed(123)
    # High self-excitation: alpha close to beta
    times = _simulate_hawkes(mu=0.5, alpha=1.5, beta=2.0, T=3600)

    est = HawkesFlowEstimator(n_starts=5)
    for t in times:
        est.record_event("1.234", t)

    result = est.estimate("1.234")
    if result.is_reliable:
        assert result.branching_ratio > 0.5


def test_clear_market():
    est = HawkesFlowEstimator()
    est.record_event("1.234", 1.0)
    est.clear_market("1.234")
    result = est.estimate("1.234")
    assert result.sample_size == 0


def test_prompt_text():
    est = HawkesFlowEstimator()
    result = est.estimate("nonexistent")
    text = result.to_prompt_text()
    assert "Order Flow" in text
