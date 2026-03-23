"""Tests for the self-healing system."""

import datetime as dt

from betfair_trader.healing import (
    CodeHealer,
    HealLog,
    HealingThresholds,
    ModelHealer,
    SelfHealingManager,
    StrategyHealer,
    TradeResult,
)


def test_code_healer_retries_and_succeeds():
    """Level 1: retries a failing function and succeeds."""
    thresholds = HealingThresholds(max_self_repair_attempts=5)
    log = HealLog(path=None)
    healer = CodeHealer(thresholds, log)

    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("timeout")
        return "success"

    result = healer.execute_with_recovery(flaky_func, max_retries=3, backoff_base=0.01)
    assert result == "success"
    assert call_count == 3


def test_code_healer_gives_up():
    """Level 1: gives up after max retries."""
    thresholds = HealingThresholds(max_self_repair_attempts=2)
    log = HealLog(path=None)

    healer = CodeHealer(thresholds, log)

    def always_fails():
        raise RuntimeError("broken")

    result = healer.execute_with_recovery(always_fails, max_retries=2, backoff_base=0.01)
    assert result is None


def test_strategy_healer_detects_degradation():
    """Level 2: switches strategy when win rate drops below threshold."""
    thresholds = HealingThresholds(
        min_win_rate=0.55, review_window=10, emergency_win_rate=0.10,
    )
    log = HealLog(path=None)

    healer = StrategyHealer(thresholds, log)
    healer._active_strategy = "scalping"

    # Record 10 trades with 20% win rate (below 55% but above 10% emergency)
    for i in range(10):
        healer.record_result(TradeResult(
            strategy="scalping", signal="scalp", edge=0.02,
            pnl=1.0 if i < 2 else -1.0,
            won=(i < 2),
        ))

    # Strategy should have switched away from scalping
    assert healer.active_strategy != "scalping"


def test_strategy_healer_emergency_paper_mode():
    """Level 2: forces paper mode when win rate is critically low."""
    thresholds = HealingThresholds(emergency_win_rate=0.45, review_window=10)
    log = HealLog(path=None)

    healer = StrategyHealer(thresholds, log)

    for _ in range(10):
        healer.record_result(TradeResult(
            strategy="scalping", signal="scalp", edge=0.02, pnl=-2.0, won=False,
        ))

    assert healer.should_force_paper_mode()


def test_strategy_healer_over_trading():
    """Level 2: blocks trades when too many in one race."""
    thresholds = HealingThresholds(max_trades_per_race=3)
    log = HealLog(path=None)

    healer = StrategyHealer(thresholds, log)

    assert healer.check_trade_frequency("race1") is True
    assert healer.check_trade_frequency("race1") is True
    assert healer.check_trade_frequency("race1") is True
    assert healer.check_trade_frequency("race1") is False  # 4th blocked


def test_model_healer_weight_adjustment():
    """Level 3: adjusts signal weights based on accuracy."""
    log = HealLog(path=None)

    healer = ModelHealer(log)

    # Record 60 predictions — good accuracy for racing_post_form
    for i in range(60):
        healer.record_prediction(
            predicted_p=0.6,
            actual_won=(i % 3 != 0),  # ~67% accuracy
            signals_used=["racing_post_form"],
        )

    updates = healer.update_signal_weights()
    assert "racing_post_form" in updates
    assert updates["racing_post_form"] > 1.0  # Weight increased


def test_model_healer_bias_detection():
    """Level 3: detects overconfidence bias."""
    log = HealLog(path=None)

    healer = ModelHealer(log)

    # Record 120 predictions with overconfident estimates
    for _ in range(120):
        healer.record_prediction(
            predicted_p=0.8,  # We think 80%
            actual_won=False,  # But always loses
            signals_used=["racing_post_form"],
        )

    biases = healer.detect_biases()
    assert len(biases) > 0
    assert "overconfidence" in biases[0].lower()


def test_self_healing_manager_health_report():
    """Combined manager returns a health report."""
    manager = SelfHealingManager()
    manager.heal_log._save = lambda: None
    report = manager.get_health_report()

    assert "active_strategy" in report
    assert "force_paper_mode" in report
    assert "heartbeat_ok" in report
    assert "signal_weights" in report
