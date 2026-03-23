"""Tests for the outcome logger."""

import tempfile
from pathlib import Path

from betfair_trader.outcome.outcome_logger import BetRecord, OutcomeLogger, SignalSnapshot


def _make_logger():
    """Create a logger with a temp database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    return OutcomeLogger(db_path=Path(tmp.name))


def test_log_and_retrieve():
    """Should log a bet and retrieve summary."""
    ol = _make_logger()
    record = BetRecord(
        bet_id="test-001",
        horse_name="Frankel",
        model_probability=0.35,
        market_implied_probability=0.25,
        edge_at_entry=0.10,
        stake=2.0,
        entry_odds=4.0,
        strategy="back_to_lay",
        confidence="high",
    )
    row_id = ol.log_bet(record)
    assert row_id > 0

    # Record outcome
    ol.record_outcome("test-001", finishing_position=1, won=True, pnl=6.0)

    summary = ol.get_summary()
    assert summary["total_bets"] == 1
    assert summary["winners"] == 1
    assert summary["total_pnl"] == 6.0
    ol.close()


def test_edge_vs_outcome():
    """Should group bets by edge bucket."""
    ol = _make_logger()

    for i in range(5):
        record = BetRecord(
            bet_id=f"edge-{i}",
            horse_name=f"Horse {i}",
            model_probability=0.30,
            market_implied_probability=0.20,
            edge_at_entry=0.10,
            stake=2.0,
            entry_odds=5.0,
            strategy="back_to_lay",
        )
        ol.log_bet(record)
        ol.record_outcome(f"edge-{i}", finishing_position=i + 1, won=(i == 0), pnl=8.0 if i == 0 else -2.0)

    buckets = ol.get_edge_vs_outcome()
    assert len(buckets) > 0
    ol.close()


def test_signal_accuracy():
    """Should track per-signal accuracy."""
    ol = _make_logger()

    record = BetRecord(bet_id="sig-001", horse_name="Test", model_probability=0.3,
                       market_implied_probability=0.2, edge_at_entry=0.1)
    ol.log_bet(record)

    ol.record_signal_accuracy("sig-001", "steam", "positive", True)
    ol.record_signal_accuracy("sig-001", "form", "positive", True)
    ol.record_signal_accuracy("sig-001", "trainer_quote", "positive", False)

    contributions = ol.get_signal_contribution()
    assert len(contributions) == 3

    steam = [c for c in contributions if c["signal_name"] == "steam"][0]
    assert steam["accuracy"] == 1.0
    ol.close()


def test_export_csv():
    """Should export to CSV without error."""
    ol = _make_logger()

    record = BetRecord(bet_id="csv-001", horse_name="Export Test",
                       model_probability=0.3, market_implied_probability=0.2,
                       edge_at_entry=0.1)
    ol.log_bet(record)
    ol.record_outcome("csv-001", 1, True, 5.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    ol.export_csv(tmp.name)

    with open(tmp.name) as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 row
    ol.close()
