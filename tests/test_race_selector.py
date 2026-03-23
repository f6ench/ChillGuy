"""Tests for the race selection filter."""

from betfair_trader.filters.race_selector import RaceFilterConfig, RaceSelector


def test_passes_valid_race():
    """A race meeting all criteria should pass."""
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=3,
        field_size=10,
        race_type="flat",
        is_maiden=False,
        total_matched=75000,
    )
    assert result.passed
    assert result.reasons == []


def test_filters_low_liquidity():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=3,
        field_size=10,
        race_type="flat",
        is_maiden=False,
        total_matched=10000,
    )
    assert not result.passed
    assert any("liquidity" in r.lower() for r in result.reasons)


def test_filters_wrong_class():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=6,
        field_size=10,
        race_type="flat",
        is_maiden=False,
        total_matched=75000,
    )
    assert not result.passed
    assert any("class" in r.lower() for r in result.reasons)


def test_filters_too_few_runners():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=3,
        field_size=3,
        race_type="flat",
        is_maiden=False,
        total_matched=75000,
    )
    assert not result.passed
    assert any("few" in r.lower() for r in result.reasons)


def test_filters_too_many_runners():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=3,
        field_size=20,
        race_type="flat",
        is_maiden=False,
        total_matched=75000,
    )
    assert not result.passed
    assert any("many" in r.lower() for r in result.reasons)


def test_filters_jumps_when_flat_only():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Cheltenham",
        race_class=2,
        field_size=10,
        race_type="jumps",
        is_maiden=False,
        total_matched=75000,
    )
    assert not result.passed
    assert any("flat" in r.lower() for r in result.reasons)


def test_filters_maidens():
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Ascot",
        race_class=3,
        field_size=10,
        race_type="flat",
        is_maiden=True,
        total_matched=75000,
    )
    assert not result.passed
    assert any("maiden" in r.lower() for r in result.reasons)


def test_multiple_failures():
    """Multiple filter violations should all be reported."""
    selector = RaceSelector()
    result = selector.filter_race(
        race_id="1.234",
        race_name="Bad Race",
        race_class=6,
        field_size=3,
        race_type="jumps",
        is_maiden=True,
        total_matched=1000,
    )
    assert not result.passed
    assert len(result.reasons) >= 4


def test_custom_config():
    """Custom thresholds should be respected."""
    config = RaceFilterConfig(
        min_liquidity=10000,
        max_class=6,
        min_field_size=4,
        flat_only=False,
        exclude_maidens=False,
    )
    selector = RaceSelector(config)
    result = selector.filter_race(
        race_id="1.234",
        race_name="3:30 Cheltenham",
        race_class=5,
        field_size=5,
        race_type="jumps",
        is_maiden=True,
        total_matched=15000,
    )
    assert result.passed


def test_filter_market_dict():
    """Should extract fields from market dict correctly."""
    selector = RaceSelector()
    market = {
        "market_id": "1.234",
        "market_name": "3:30 Ascot",
        "race_class": 3,
        "race_type": "flat",
        "is_maiden": False,
        "runners": [{"id": i, "name": f"Horse {i}"} for i in range(10)],
    }
    result = selector.filter_market(market, total_matched=75000)
    assert result.passed


def test_score_race():
    """Secondary scoring should reward handicaps and steam."""
    selector = RaceSelector()

    base = selector.score_race(race_class=1, is_handicap=False,
                               has_steam=False, has_strong_trainer_signal=False)
    handicap_steam = selector.score_race(race_class=3, is_handicap=True,
                                         has_steam=True, has_strong_trainer_signal=True)

    assert handicap_steam > base
