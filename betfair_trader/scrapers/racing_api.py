"""Racing API client for independent form data.

Connects to racingapi.pro to pull per-runner form data that is independent
of market prices. This is the primary source for Stage 1 probability
estimation — giving Claude form-based data to reason from BEFORE seeing odds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Cache TTL: form data doesn't change intra-day, so cache aggressively
_CACHE_TTL = 3600  # 1 hour


@dataclass
class RunResult:
    """A single past race result for a runner."""

    date: str
    course: str
    distance_furlongs: float
    class_level: int
    going: str
    position: int
    runners_in_race: int
    beaten_lengths: float
    speed_figure: float
    rpr: float  # Racing Post Rating
    weight_carried_lbs: float


@dataclass
class RunnerForm:
    """Complete form profile for a runner in today's race."""

    horse_name: str
    horse_id: str
    age: int
    sex: str
    trainer: str
    jockey: str
    draw: int  # stall number
    weight_carried_lbs: float
    official_rating: int
    last_runs: list[RunResult] = field(default_factory=list)

    # Aggregated stats
    course_wins: int = 0
    course_runs: int = 0
    distance_wins: int = 0
    distance_runs: int = 0
    going_wins: int = 0
    going_runs: int = 0
    days_since_last_run: int = 0

    # Trainer/jockey stats at this course
    trainer_course_win_pct: float = 0.0
    trainer_season_win_pct: float = 0.0
    jockey_course_win_pct: float = 0.0
    jockey_season_win_pct: float = 0.0

    # Class movement
    class_change: int = 0  # negative = dropping in class (positive signal)

    def to_prompt_text(self) -> str:
        """Format this runner's form for the Claude prompt."""
        lines = [
            f"  {self.horse_name} (Age: {self.age}, Sex: {self.sex})",
            f"    Trainer: {self.trainer} | Jockey: {self.jockey} | Draw: {self.draw}",
            f"    Official Rating: {self.official_rating} | Weight: {self.weight_carried_lbs}lbs",
            f"    Days Since Last Run: {self.days_since_last_run}",
            f"    Course Record: {self.course_wins}/{self.course_runs} "
            f"| Distance Record: {self.distance_wins}/{self.distance_runs} "
            f"| Going Record: {self.going_wins}/{self.going_runs}",
            f"    Class Change: {self.class_change:+d} (negative = dropping, positive = rising)",
            f"    Trainer Course Win%: {self.trainer_course_win_pct:.1f}% "
            f"| Trainer Season Win%: {self.trainer_season_win_pct:.1f}%",
            f"    Jockey Course Win%: {self.jockey_course_win_pct:.1f}% "
            f"| Jockey Season Win%: {self.jockey_season_win_pct:.1f}%",
        ]

        if self.last_runs:
            lines.append("    Last Runs:")
            for run in self.last_runs[:6]:
                lines.append(
                    f"      {run.date} | {run.course} {run.distance_furlongs}f "
                    f"Class {run.class_level} | {run.going} | "
                    f"Pos {run.position}/{run.runners_in_race} "
                    f"(beaten {run.beaten_lengths}L) | "
                    f"RPR {run.rpr} | Speed {run.speed_figure}"
                )

        return "\n".join(lines)


@dataclass
class RaceCard:
    """Full race card with form data for all runners."""

    race_id: str
    course: str
    race_time: str
    race_name: str
    race_class: int
    distance_furlongs: float
    going: str
    field_size: int
    prize_money: float
    race_type: str  # "flat" or "jumps"
    is_handicap: bool
    is_maiden: bool
    runners: list[RunnerForm] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format the full race card for the Claude Stage 1 prompt."""
        lines = [
            f"Race: {self.race_name}",
            f"Course: {self.course} | Time: {self.race_time}",
            f"Class: {self.race_class} | Distance: {self.distance_furlongs}f | Going: {self.going}",
            f"Type: {self.race_type} | Handicap: {self.is_handicap} | Maiden: {self.is_maiden}",
            f"Field Size: {self.field_size} | Prize: £{self.prize_money:,.0f}",
            "",
            "Runners:",
        ]
        for runner in self.runners:
            lines.append(runner.to_prompt_text())
            lines.append("")

        return "\n".join(lines)


class RacingAPIClient:
    """Client for racingapi.pro REST API.

    Fetches per-runner form data, trainer/jockey stats, course records,
    and historical results. All data is cached per race to avoid redundant calls.
    """

    BASE_URL = "https://api.racingapi.pro/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        self._cache: dict[str, tuple[float, object]] = {}

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a cached GET request to the Racing API."""
        cache_key = f"{endpoint}:{params}"
        now = time.time()

        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < _CACHE_TTL:
                return cached_data

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._cache[cache_key] = (now, data)
            return data
        except requests.RequestException as e:
            logger.error("Racing API request failed: %s %s — %s", endpoint, params, e)
            return {}

    def get_todays_races(self, race_type: str = "flat") -> list[dict]:
        """Get today's race cards for UK racing."""
        data = self._get("racecards/today", params={"region": "gb", "type": race_type})
        return data.get("races", [])

    def get_race_card(self, race_id: str) -> Optional[RaceCard]:
        """Fetch a full race card with per-runner form data."""
        data = self._get(f"racecards/{race_id}")
        if not data:
            return None

        try:
            runners = []
            for r in data.get("runners", []):
                form = r.get("form", {})
                last_runs = []
                for run in form.get("last_runs", [])[:6]:
                    last_runs.append(RunResult(
                        date=run.get("date", ""),
                        course=run.get("course", ""),
                        distance_furlongs=run.get("distance_furlongs", 0),
                        class_level=run.get("class", 0),
                        going=run.get("going", ""),
                        position=run.get("position", 0),
                        runners_in_race=run.get("runners", 0),
                        beaten_lengths=run.get("beaten_lengths", 0),
                        speed_figure=run.get("speed_figure", 0),
                        rpr=run.get("rpr", 0),
                        weight_carried_lbs=run.get("weight_lbs", 0),
                    ))

                stats = r.get("stats", {})
                runner = RunnerForm(
                    horse_name=r.get("name", ""),
                    horse_id=str(r.get("id", "")),
                    age=r.get("age", 0),
                    sex=r.get("sex", ""),
                    trainer=r.get("trainer", ""),
                    jockey=r.get("jockey", ""),
                    draw=r.get("draw", 0),
                    weight_carried_lbs=r.get("weight_lbs", 0),
                    official_rating=r.get("official_rating", 0),
                    last_runs=last_runs,
                    course_wins=stats.get("course_wins", 0),
                    course_runs=stats.get("course_runs", 0),
                    distance_wins=stats.get("distance_wins", 0),
                    distance_runs=stats.get("distance_runs", 0),
                    going_wins=stats.get("going_wins", 0),
                    going_runs=stats.get("going_runs", 0),
                    days_since_last_run=r.get("days_since_run", 0),
                    trainer_course_win_pct=stats.get("trainer_course_pct", 0),
                    trainer_season_win_pct=stats.get("trainer_season_pct", 0),
                    jockey_course_win_pct=stats.get("jockey_course_pct", 0),
                    jockey_season_win_pct=stats.get("jockey_season_pct", 0),
                    class_change=r.get("class_change", 0),
                )
                runners.append(runner)

            race = data.get("race", data)
            return RaceCard(
                race_id=race_id,
                course=race.get("course", ""),
                race_time=race.get("time", ""),
                race_name=race.get("name", ""),
                race_class=race.get("class", 0),
                distance_furlongs=race.get("distance_furlongs", 0),
                going=race.get("going", ""),
                field_size=len(runners),
                prize_money=race.get("prize", 0),
                race_type=race.get("type", "flat"),
                is_handicap=race.get("is_handicap", False),
                is_maiden=race.get("is_maiden", False),
                runners=runners,
            )
        except (KeyError, TypeError) as e:
            logger.error("Failed to parse race card %s: %s", race_id, e)
            return None

    def get_runner_form(self, horse_id: str, limit: int = 6) -> list[RunResult]:
        """Fetch historical form for a specific horse."""
        data = self._get(f"horses/{horse_id}/form", params={"limit": limit})
        results = []
        for run in data.get("runs", []):
            results.append(RunResult(
                date=run.get("date", ""),
                course=run.get("course", ""),
                distance_furlongs=run.get("distance_furlongs", 0),
                class_level=run.get("class", 0),
                going=run.get("going", ""),
                position=run.get("position", 0),
                runners_in_race=run.get("runners", 0),
                beaten_lengths=run.get("beaten_lengths", 0),
                speed_figure=run.get("speed_figure", 0),
                rpr=run.get("rpr", 0),
                weight_carried_lbs=run.get("weight_lbs", 0),
            ))
        return results

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
