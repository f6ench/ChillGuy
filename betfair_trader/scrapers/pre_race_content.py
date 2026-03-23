"""Pre-race content scraper for trainer quotes and parade ring reports.

Scrapes same-day intelligence that represents physical, on-the-ground
information that takes time to propagate into market prices:
- Trainer quotes from Racing Post
- Live text commentary from At The Races
- Pre-race interviews from ITV Racing

Each piece of content is classified by Claude as positive/negative/neutral
per runner, creating a sentiment signal independent of market prices.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    STRONGLY_POSITIVE = "strongly_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONGLY_NEGATIVE = "strongly_negative"


@dataclass
class ContentItem:
    """A single piece of pre-race content about a runner."""

    source: str  # "racing_post", "atr", "itv"
    runner_name: str
    text: str
    author: str = ""
    timestamp: str = ""


@dataclass
class RunnerSentiment:
    """Aggregated sentiment classification for a runner from all content sources."""

    runner_name: str
    sentiment: Sentiment = Sentiment.NEUTRAL
    confidence: float = 0.0  # 0.0 - 1.0
    content_items: list[ContentItem] = field(default_factory=list)
    classification_reasoning: str = ""

    def to_prompt_text(self) -> str:
        """Format for Claude Stage 2 prompt."""
        return (
            f"  {self.runner_name}: {self.sentiment.value} "
            f"(confidence: {self.confidence:.2f}) "
            f"[{len(self.content_items)} sources] "
            f"— {self.classification_reasoning}"
        )


class PreRaceContentScraper:
    """Scrapes and classifies pre-race trainer quotes and parade ring reports.

    Sources:
    - Racing Post trainer quotes (morning of race)
    - At The Races live text commentary (90 mins before)

    The raw content is collected per runner, then fed to Claude for
    sentiment classification.
    """

    # Racing Post race card page
    RP_BASE_URL = "https://www.racingpost.com"
    # At The Races live commentary
    ATR_BASE_URL = "https://www.attheraces.com"

    def __init__(self, request_delay: float = 2.0):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self._request_delay = request_delay
        self._cache: dict[str, list[ContentItem]] = {}

    def scrape_trainer_quotes(
        self,
        course: str,
        race_time: str,
        runner_names: list[str],
    ) -> list[ContentItem]:
        """Scrape Racing Post for trainer quotes about declared runners.

        Should be called morning of race day (after 9am typically).
        """
        cache_key = f"rp:{course}:{race_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        items = []
        try:
            # Build search URL for the race card
            course_slug = course.lower().replace(" ", "-")
            url = f"{self.RP_BASE_URL}/racecards/{course_slug}"
            time.sleep(self._request_delay)
            resp = self._session.get(url, timeout=15)

            if resp.status_code != 200:
                logger.warning("Racing Post returned %d for %s", resp.status_code, url)
                return items

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract trainer quotes from race card comments sections
            comment_blocks = soup.find_all(
                "div", class_=re.compile(r"(trainer|comment|quote)", re.I)
            )

            for block in comment_blocks:
                text = block.get_text(strip=True)
                if not text:
                    continue

                # Match quotes to runners
                for name in runner_names:
                    # Check if this quote mentions this runner
                    if name.lower() in text.lower():
                        items.append(ContentItem(
                            source="racing_post",
                            runner_name=name,
                            text=text[:500],  # cap length
                            author="Trainer quote",
                        ))

        except requests.RequestException as e:
            logger.error("Failed to scrape Racing Post: %s", e)

        self._cache[cache_key] = items
        return items

    def scrape_atr_commentary(
        self,
        course: str,
        race_time: str,
        runner_names: list[str],
    ) -> list[ContentItem]:
        """Scrape At The Races for live text commentary.

        Should be called from ~90 minutes before each race, repeatedly.
        Parade ring reports, going updates, horse demeanour observations.
        """
        cache_key = f"atr:{course}:{race_time}"
        items = []

        try:
            course_slug = course.lower().replace(" ", "-")
            url = f"{self.ATR_BASE_URL}/racecard/{course_slug}/{race_time}"
            time.sleep(self._request_delay)
            resp = self._session.get(url, timeout=15)

            if resp.status_code != 200:
                logger.warning("ATR returned %d for %s", resp.status_code, url)
                return items

            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for live commentary / paddock report sections
            commentary_blocks = soup.find_all(
                "div", class_=re.compile(r"(commentary|paddock|preview|live)", re.I)
            )

            for block in commentary_blocks:
                text = block.get_text(strip=True)
                if not text or len(text) < 20:
                    continue

                for name in runner_names:
                    if name.lower() in text.lower():
                        items.append(ContentItem(
                            source="atr",
                            runner_name=name,
                            text=text[:500],
                            author="ATR Commentary",
                        ))

        except requests.RequestException as e:
            logger.error("Failed to scrape ATR: %s", e)

        # Merge with any previously cached items
        if cache_key in self._cache:
            existing_texts = {i.text for i in self._cache[cache_key]}
            for item in items:
                if item.text not in existing_texts:
                    self._cache[cache_key].append(item)
        else:
            self._cache[cache_key] = items

        return self._cache.get(cache_key, items)

    def get_all_content(
        self,
        course: str,
        race_time: str,
        runner_names: list[str],
    ) -> dict[str, list[ContentItem]]:
        """Get all pre-race content grouped by runner name."""
        rp_items = self.scrape_trainer_quotes(course, race_time, runner_names)
        atr_items = self.scrape_atr_commentary(course, race_time, runner_names)

        all_items = rp_items + atr_items
        by_runner: dict[str, list[ContentItem]] = {}
        for item in all_items:
            by_runner.setdefault(item.runner_name, []).append(item)

        return by_runner

    def clear_cache(self) -> None:
        self._cache.clear()


def build_classification_prompt(
    runner_name: str, content_items: list[ContentItem]
) -> str:
    """Build the Claude classification prompt for a runner's pre-race content.

    This is used by the analyst to classify sentiment per runner.
    """
    quotes = "\n".join(
        f"- [{item.source}] {item.author}: \"{item.text}\""
        for item in content_items
    )

    return f"""Based on the following pre-race quotes and commentary about {runner_name},
classify the stable's intent and confidence for this runner today.

Quotes and commentary:
{quotes}

Respond with JSON:
{{
  "sentiment": "strongly_positive|positive|neutral|negative|strongly_negative",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of classification"
}}

Focus on:
- Trainer language indicating confidence vs uncertainty
- Physical condition reports (looks well, sweating, on toes)
- Trip/distance/going suitability comments
- Any indication of whether connections expect a big run"""
