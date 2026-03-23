"""Tests for the steam/drift detection module."""

from betfair_trader.signals.steam_detector import (
    MovementType,
    SteamDetector,
)


def test_steam_detection_shortening_with_volume():
    """A runner shortening 20% with 15% volume should flag STEAM."""
    detector = SteamDetector(steam_threshold=0.15, drift_threshold=-0.20, volume_threshold=0.10)

    market_id = "1.234"
    selection_id = 101

    # Opening price: 10.0
    detector.record_price(market_id, selection_id, 10.0, 5000)
    # Price moves to 8.0 (20% shortening) with volume
    detector.record_price(market_id, selection_id, 8.0, 20000)

    detector.set_estimated_daily_volume(market_id, 100000)

    signals = detector.get_signals(market_id, {101: "Horse A"})
    assert len(signals) == 1
    assert signals[0].movement == MovementType.STEAM
    assert signals[0].steam_ratio == 0.2  # (10-8)/10
    assert signals[0].runner_name == "Horse A"


def test_drift_detection():
    """A runner drifting 25% should flag DRIFT."""
    detector = SteamDetector()

    market_id = "1.234"
    detector.record_price(market_id, 101, 4.0, 1000)
    detector.record_price(market_id, 101, 5.2, 2000)  # 30% drift

    detector.set_estimated_daily_volume(market_id, 100000)

    signals = detector.get_signals(market_id)
    assert len(signals) == 1
    assert signals[0].movement == MovementType.DRIFT
    assert signals[0].steam_ratio < -0.20


def test_stable_price():
    """Small price movements should be STABLE."""
    detector = SteamDetector()

    market_id = "1.234"
    detector.record_price(market_id, 101, 5.0, 1000)
    detector.record_price(market_id, 101, 4.8, 2000)  # 4% shortening

    detector.set_estimated_daily_volume(market_id, 100000)

    signals = detector.get_signals(market_id)
    assert len(signals) == 1
    assert signals[0].movement == MovementType.STABLE


def test_steam_requires_volume():
    """Price shortening without sufficient volume should NOT flag STEAM."""
    detector = SteamDetector()

    market_id = "1.234"
    detector.record_price(market_id, 101, 10.0, 100)
    detector.record_price(market_id, 101, 8.0, 500)  # 20% shortening but tiny volume

    detector.set_estimated_daily_volume(market_id, 100000)

    signals = detector.get_signals(market_id)
    assert len(signals) == 1
    # Volume ratio is 500/100000 = 0.5% < 10% threshold
    assert signals[0].movement == MovementType.STABLE


def test_multiple_runners():
    """Should track multiple runners independently."""
    detector = SteamDetector()

    market_id = "1.234"
    detector.record_price(market_id, 101, 10.0, 5000)
    detector.record_price(market_id, 102, 4.0, 5000)

    detector.record_price(market_id, 101, 8.0, 15000)  # steaming
    detector.record_price(market_id, 102, 5.5, 15000)  # drifting

    detector.set_estimated_daily_volume(market_id, 100000)

    steam = detector.get_steam_runners(market_id)
    drift = detector.get_drift_runners(market_id)

    assert len(steam) == 1
    assert steam[0].selection_id == 101
    assert len(drift) == 1
    assert drift[0].selection_id == 102


def test_clear_market():
    """Clearing a market should remove all data."""
    detector = SteamDetector()

    detector.record_price("1.234", 101, 10.0, 5000)
    detector.clear_market("1.234")

    assert detector.get_signals("1.234") == []
