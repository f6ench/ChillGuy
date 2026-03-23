"""Tests for Avellaneda-Stoikov market making strategy."""

import numpy as np

from betfair_trader.config import BetfairConfig
from betfair_trader.strategies.market_making import AvellanedaStoikovStrategy


def _make_config():
    """Create a minimal config for testing."""
    return BetfairConfig(
        BETFAIR_USERNAME="test",
        BETFAIR_PASSWORD="test",
        BETFAIR_APP_KEY="test",
        PAPER_TRADING=True,
        MAX_STAKE=10.0,
    )


def test_reservation_price_adjusts_for_inventory():
    """Long inventory should lower the reservation price."""
    mm = AvellanedaStoikovStrategy(_make_config())

    mid = 5.0
    time_remaining = 2.0

    r_neutral = mm.reservation_price(mid, inventory=0, time_remaining=time_remaining)
    r_long = mm.reservation_price(mid, inventory=100, time_remaining=time_remaining)
    r_short = mm.reservation_price(mid, inventory=-100, time_remaining=time_remaining)

    assert r_neutral == mid  # no inventory, no adjustment
    assert r_long < mid  # long -> shade price down
    assert r_short > mid  # short -> shade price up


def test_quotes_straddle_mid():
    """Bid should be below mid, ask above mid."""
    mm = AvellanedaStoikovStrategy(_make_config())

    mid = 5.0
    quotes = mm.compute_quotes(mid, "1.234", time_remaining=2.0)

    assert quotes.bid_price < mid
    assert quotes.ask_price > mid
    assert quotes.ask_price > quotes.bid_price


def test_spread_widens_with_multiplier():
    """Higher spread multiplier should widen the spread."""
    mm = AvellanedaStoikovStrategy(_make_config())

    mid = 5.0
    normal = mm.compute_quotes(mid, "1.234", time_remaining=2.0, spread_multiplier=1.0)
    wide = mm.compute_quotes(mid, "1.234", time_remaining=2.0, spread_multiplier=2.0)

    normal_spread = normal.ask_price - normal.bid_price
    wide_spread = wide.ask_price - wide.bid_price

    assert wide_spread > normal_spread


def test_fill_updates_inventory():
    """Fills should update inventory and cash."""
    mm = AvellanedaStoikovStrategy(_make_config())

    mm.record_fill("1.234", price=5.0, size=100, is_buy=True)
    state = mm.get_state_summary("1.234")

    assert state["inventory"] == 100
    assert state["cash"] == -500  # bought 100 @ 5.0
    assert state["n_fills"] == 1


def test_inventory_blowup_protection():
    """Near max inventory should cause aggressive unloading quotes."""
    mm = AvellanedaStoikovStrategy(_make_config(), max_inventory=500)

    # Accumulate large long position
    mm.record_fill("1.234", price=5.0, size=450, is_buy=True)

    mid = 5.0
    quotes = mm.compute_quotes(mid, "1.234", time_remaining=2.0)

    # With 90% inventory utilization, ask should be aggressively low
    assert quotes.ask_price <= mid
