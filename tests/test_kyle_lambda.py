"""Tests for Kyle's Lambda estimator."""

import numpy as np

from betfair_trader.signals.kyle_lambda import KyleLambdaEstimator


def test_insufficient_data():
    """Should return unreliable estimate with too few trades."""
    est = KyleLambdaEstimator()
    est.record_trade("1.234", 5.0, 100, True)
    est.record_trade("1.234", 5.1, 100, True)
    result = est.estimate("1.234")
    assert not result.is_reliable
    assert result.interpretation == "Insufficient data"


def test_detects_informed_trading():
    """Synthetic data with strong price impact should show high lambda."""
    est = KyleLambdaEstimator(window_size=500)
    np.random.seed(42)

    price = 5.0
    lambda_true = 0.003  # high informed trading

    for _ in range(200):
        is_buy = np.random.random() > 0.4
        vol = np.random.exponential(300)
        sign = 1 if is_buy else -1
        price += lambda_true * sign * vol / 100 + np.random.normal(0, 0.001)
        price = max(1.01, price)
        est.record_trade("1.234", price, vol, is_buy)

    result = est.estimate("1.234")
    assert result.is_reliable
    assert result.lambda_value > 0  # positive lambda = price follows flow


def test_normal_liquidity():
    """Random noise with no information should show low lambda."""
    est = KyleLambdaEstimator(window_size=500)
    np.random.seed(99)

    price = 5.0
    for _ in range(200):
        is_buy = np.random.random() > 0.5
        vol = np.random.exponential(100)
        price += np.random.normal(0, 0.01)  # price moves independently of flow
        price = max(1.01, price)
        est.record_trade("1.234", price, vol, is_buy)

    result = est.estimate("1.234")
    # Lambda should be near zero (no information in flow)
    assert abs(result.lambda_value) < 0.01


def test_clear_market():
    est = KyleLambdaEstimator()
    est.record_trade("1.234", 5.0, 100, True)
    est.clear_market("1.234")
    result = est.estimate("1.234")
    assert result.sample_size == 0


def test_prompt_text():
    est = KyleLambdaEstimator()
    result = est.estimate("nonexistent")
    text = result.to_prompt_text()
    assert "Kyle's Lambda" in text
