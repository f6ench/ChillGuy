"""Tests for core data models."""

from betfair_trader.models import (
    Confidence,
    MarketSnapshot,
    Signal,
    Strategy,
    TradeInstruction,
)


def test_trade_instruction_ev():
    instruction = TradeInstruction(
        race="3:30 Cheltenham",
        selection="Horse A",
        signal=Signal.SPRINGER,
        p=0.5,
        market_p=0.4,
        edge=0.1,
        strategy=Strategy.BACK_TO_LAY,
        entry_odds=2.5,
        target_odds=2.0,
    )
    # EV = p * (odds-1) - (1-p) = 0.5 * 1.5 - 0.5 = 0.25
    assert abs(instruction.ev - 0.25) < 0.001


def test_market_snapshot_properties():
    snap = MarketSnapshot(
        market_id="1.234",
        selection_id=101,
        runner_name="Horse A",
        back_prices=[(3.0, 100.0)],
        lay_prices=[(3.1, 100.0)],
    )
    assert snap.best_back == 3.0
    assert snap.best_lay == 3.1
    assert abs(snap.spread - 0.1) < 0.001
    assert abs(snap.implied_probability - 1 / 3.0) < 0.001


def test_snapshot_empty_prices():
    snap = MarketSnapshot(
        market_id="1.234",
        selection_id=101,
        runner_name="Horse A",
    )
    assert snap.best_back == 0.0
    assert snap.best_lay == 0.0
    assert snap.spread == 0.0
    assert snap.implied_probability == 0.0
