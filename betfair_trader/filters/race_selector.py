"""Race selection filter — pre-processing gate before the analysis pipeline.

Filters races to those where the signal stack has information value.
Running the full pipeline on every race wastes compute and creates noise.
Edge is not evenly distributed across race types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RaceFilterConfig:
    """Configurable thresholds for race selection."""

    min_liquidity: float = 50_000  # minimum matched volume in GBP
    min_class: int = 1
    max_class: int = 4
    min_field_size: int = 6
    max_field_size: int = 16
    flat_only: bool = True  # exclude jumps during testing
    exclude_maidens: bool = True
    handicaps_preferred: bool = True  # for secondary filter


@dataclass
class FilterResult:
    """Result of filtering a race."""

    passed: bool
    race_id: str
    race_name: str
    reasons: list[str]  # why it was filtered out (empty if passed)

    def __str__(self) -> str:
        if self.passed:
            return f"PASS: {self.race_name}"
        return f"SKIP: {self.race_name} — {', '.join(self.reasons)}"


class RaceSelector:
    """Pre-processing filter that gates which races enter the analysis pipeline.

    Primary filter: hard requirements (liquidity, class, field size, race type)
    Secondary filter: preference scoring (handicaps, class range, steam presence)
    """

    def __init__(self, config: RaceFilterConfig | None = None):
        self.config = config or RaceFilterConfig()

    def filter_race(
        self,
        race_id: str,
        race_name: str,
        race_class: int,
        field_size: int,
        race_type: str,
        is_maiden: bool,
        total_matched: float,
        is_handicap: bool = False,
    ) -> FilterResult:
        """Apply primary filter criteria to a single race."""
        reasons = []

        if total_matched < self.config.min_liquidity:
            reasons.append(
                f"Low liquidity: £{total_matched:,.0f} < £{self.config.min_liquidity:,.0f}"
            )

        if race_class < self.config.min_class or race_class > self.config.max_class:
            reasons.append(
                f"Class {race_class} outside range [{self.config.min_class}-{self.config.max_class}]"
            )

        if field_size < self.config.min_field_size:
            reasons.append(
                f"Too few runners: {field_size} < {self.config.min_field_size}"
            )

        if field_size > self.config.max_field_size:
            reasons.append(
                f"Too many runners: {field_size} > {self.config.max_field_size}"
            )

        if self.config.flat_only and race_type.lower() != "flat":
            reasons.append(f"Not flat racing: {race_type}")

        if self.config.exclude_maidens and is_maiden:
            reasons.append("Maiden race excluded")

        result = FilterResult(
            passed=len(reasons) == 0,
            race_id=race_id,
            race_name=race_name,
            reasons=reasons,
        )

        if not result.passed:
            logger.debug("Race filtered: %s", result)
        else:
            logger.info("Race selected: %s", race_name)

        return result

    def filter_market(self, market: dict, total_matched: float = 0) -> FilterResult:
        """Filter a market dict (as returned by the engine's market discovery).

        Extracts fields from the market dict and applies the filter.
        For fields not present in the market dict, uses permissive defaults
        to avoid false negatives.
        """
        race_id = market.get("market_id", "")
        race_name = market.get("market_name", "")
        runners = market.get("runners", [])
        field_size = len(runners)

        # These fields may not be in the basic Betfair market data.
        # The Racing API race card provides them. Use permissive defaults
        # so we don't filter races before we have form data.
        race_class = market.get("race_class", 3)  # assume mid-class if unknown
        race_type = market.get("race_type", "flat")
        is_maiden = market.get("is_maiden", False)
        is_handicap = market.get("is_handicap", False)

        return self.filter_race(
            race_id=race_id,
            race_name=race_name,
            race_class=race_class,
            field_size=field_size,
            race_type=race_type,
            is_maiden=is_maiden,
            total_matched=total_matched,
            is_handicap=is_handicap,
        )

    def score_race(
        self,
        race_class: int,
        is_handicap: bool,
        has_steam: bool,
        has_strong_trainer_signal: bool,
    ) -> float:
        """Secondary scoring for race prioritisation.

        Returns a score 0.0 - 1.0 indicating how attractive the race is
        for our signal stack. Higher = more likely to contain mispricing.
        """
        score = 0.5  # baseline

        # Handicap races: more runners, wider market, more mispricings
        if is_handicap:
            score += 0.15

        # Class 2-4: less professional attention than Class 1
        if 2 <= race_class <= 4:
            score += 0.1

        # Steam detected: signal stack is more likely to have genuine info
        if has_steam:
            score += 0.15

        # Strong trainer signal: correlates with race day intent
        if has_strong_trainer_signal:
            score += 0.1

        return min(1.0, score)
