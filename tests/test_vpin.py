"""Tests for VPIN calculator."""

import numpy as np

from betfair_trader.signals.vpin import VPINCalculator


def test_balanced_flow():
    """Balanced buy/sell should give low VPIN."""
    calc = VPINCalculator(bucket_size=50, n_buckets=5)
    np.random.seed(42)

    for _ in range(250):
        is_buy = np.random.random() > 0.5
        vol = np.random.exponential(100)
        calc.record_trade("1.234", vol, is_buy)

    result = calc.compute("1.234")
    # Balanced flow: VPIN should be relatively low
    assert result.vpin < 0.5
    assert result.spread_multiplier <= 1.5


def test_one_sided_flow():
    """One-sided flow should give high VPIN."""
    calc = VPINCalculator(bucket_size=50, n_buckets=5)
    np.random.seed(42)

    for _ in range(250):
        # 90% buys, 10% sells → strong imbalance
        is_buy = np.random.random() > 0.1
        vol = np.random.exponential(100)
        calc.record_trade("1.234", vol, is_buy)

    result = calc.compute("1.234")
    assert result.vpin > 0.5
    assert result.spread_multiplier >= 1.5


def test_critical_vpin():
    """Extreme imbalance should trigger critical VPIN."""
    calc = VPINCalculator(bucket_size=50, n_buckets=5)

    # All buys, no sells → VPIN = 1.0
    for _ in range(250):
        calc.record_trade("1.234", 100, is_buy=True)

    result = calc.compute("1.234")
    assert result.is_critical
    assert result.spread_multiplier == 0.0  # pull quotes


def test_insufficient_data():
    """Should still compute simple imbalance with few trades."""
    calc = VPINCalculator(bucket_size=50, n_buckets=5)

    for _ in range(10):
        calc.record_trade("1.234", 100, is_buy=True)

    result = calc.compute("1.234")
    assert result.n_buckets == 0  # not enough for full VPIN
    assert result.vpin > 0  # but should still give a simple estimate


def test_clear_market():
    calc = VPINCalculator()
    calc.record_trade("1.234", 100, True)
    calc.clear_market("1.234")
    result = calc.compute("1.234")
    assert result.sample_size == 0


def test_spread_multiplier_levels():
    """Test the discrete spread multiplier levels."""
    calc = VPINCalculator(bucket_size=10, n_buckets=2)

    # Create data with known imbalance
    # 20 trades: 15 buys, 5 sells → VPIN ~ 0.5
    for _ in range(15):
        calc.record_trade("1.234", 100, is_buy=True)
    for _ in range(5):
        calc.record_trade("1.234", 100, is_buy=False)

    result = calc.compute("1.234")
    assert result.spread_multiplier >= 1.0  # at minimum, normal
