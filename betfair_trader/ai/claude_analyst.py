"""Claude AI nomination layer.

This is Layer 2 of the architecture. Claude processes market data
and generates TradeInstructions with probability estimates, edge
calculations, and strategy recommendations.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from betfair_trader.config import BetfairConfig
from betfair_trader.models import (
    Confidence,
    MarketSnapshot,
    Signal,
    Strategy,
    TradeInstruction,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional Betfair Exchange horse racing trader.
You analyze market data and identify trading opportunities using these strategies:

1. SCALPING: Back and lay close together for small spread profit. Best in liquid markets.
2. DOBBING (Double or Bust): Back pre-race, lay at half odds in-play.
3. LAY TO BACK (Drifter Bot): Lay a horse whose odds are drifting, back at higher price.
4. BACK TO LAY (Springer Bot): Back a horse whose odds are shortening, lay at lower price.

Your job:
- Estimate probability (p) for each runner based on the data provided
- Compare p vs market implied probability (market_p = 1/odds)
- Calculate edge (p - market_p)
- Only recommend trades with positive edge
- Assign confidence level based on strength of signals
- Choose the best strategy for each opportunity

You must respond with a JSON array of trade instructions. Each instruction:
{
  "selection": "Horse Name",
  "selection_id": 12345,
  "signal": "springer|drifter|scalp|dob",
  "p": 0.0-1.0,
  "market_p": 0.0-1.0,
  "edge": float,
  "strategy": "back_to_lay|lay_to_back|scalping|dobbing",
  "entry_odds": float,
  "target_odds": float,
  "hedge": true|false,
  "min_profit": float,
  "confidence": "low|medium|high",
  "reasoning": "Brief explanation"
}

If no opportunities exist, return an empty array: []

Be disciplined. Skip marginal opportunities. Only trade when the edge is clear.
Factor in Betfair commission (typically 5%) when assessing profitability.
"""


class ClaudeAnalyst:
    """Uses Claude to analyze markets and generate trade instructions."""

    def __init__(self, config: BetfairConfig):
        self.config = config
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def analyze_market(
        self,
        market: dict,
        snapshots: list[MarketSnapshot],
    ) -> list[TradeInstruction]:
        """Ask Claude to analyze a market and return trade instructions."""
        if not self.config.anthropic_api_key:
            logger.warning("No Anthropic API key configured, using rule-based fallback")
            return self._fallback_analysis(market, snapshots)

        market_data = self._format_market_data(market, snapshots)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Analyze this horse racing market and identify any trading opportunities:\n\n{market_data}",
                }],
            )

            text = response.content[0].text
            # Extract JSON from response
            instructions = self._parse_response(text, market)
            logger.info("Claude identified %d trade opportunities in %s",
                        len(instructions), market["market_name"])
            return instructions

        except Exception as e:
            logger.error("Claude analysis failed: %s", e)
            return self._fallback_analysis(market, snapshots)

    def _format_market_data(
        self, market: dict, snapshots: list[MarketSnapshot]
    ) -> str:
        """Format market data for Claude's analysis."""
        lines = [
            f"Race: {market['market_name']}",
            f"Event: {market.get('event', 'Unknown')}",
            f"Market ID: {market['market_id']}",
            f"Start Time: {market.get('start_time', 'Unknown')}",
            "",
            "Runners:",
        ]
        runner_names = {r["id"]: r["name"] for r in market.get("runners", [])}

        for snap in snapshots:
            name = runner_names.get(snap.selection_id, snap.runner_name)
            lines.append(f"\n  {name} (ID: {snap.selection_id})")
            lines.append(f"    Best Back: {snap.best_back:.2f}" if snap.best_back else "    Best Back: N/A")
            lines.append(f"    Best Lay: {snap.best_lay:.2f}" if snap.best_lay else "    Best Lay: N/A")
            lines.append(f"    Spread: {snap.spread:.2f}")
            lines.append(f"    Last Traded: {snap.last_traded_price:.2f}")
            lines.append(f"    Total Matched: £{snap.total_matched:,.2f}")
            lines.append(f"    Implied Prob: {snap.implied_probability:.1%}")

        return "\n".join(lines)

    def _parse_response(
        self, text: str, market: dict
    ) -> list[TradeInstruction]:
        """Parse Claude's JSON response into TradeInstructions."""
        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.error("Failed to parse Claude response as JSON")
            return []

        instructions = []
        for item in data:
            try:
                instruction = TradeInstruction(
                    race=market["market_name"],
                    market_id=market["market_id"],
                    selection=item["selection"],
                    selection_id=item.get("selection_id", 0),
                    signal=Signal(item["signal"]),
                    p=item["p"],
                    market_p=item["market_p"],
                    edge=item["edge"],
                    strategy=Strategy(item["strategy"]),
                    entry_odds=item["entry_odds"],
                    target_odds=item["target_odds"],
                    hedge=item.get("hedge", True),
                    min_profit=item.get("min_profit", 0),
                    confidence=Confidence(item.get("confidence", "medium")),
                    reasoning=item.get("reasoning", ""),
                )
                instructions.append(instruction)
            except Exception as e:
                logger.warning("Failed to parse instruction: %s", e)

        return instructions

    def _fallback_analysis(
        self, market: dict, snapshots: list[MarketSnapshot]
    ) -> list[TradeInstruction]:
        """Simple rule-based analysis when Claude is unavailable.

        Looks for basic scalping opportunities based on spread.
        """
        instructions = []
        runner_names = {r["id"]: r["name"] for r in market.get("runners", [])}

        for snap in snapshots:
            if snap.spread > 0 and snap.best_back > 0 and snap.total_matched > 5000:
                market_p = snap.implied_probability
                # Simple heuristic: if spread is tight, it's a scalp opportunity
                if snap.spread <= 0.05 * snap.best_back:
                    instructions.append(TradeInstruction(
                        race=market["market_name"],
                        market_id=market["market_id"],
                        selection=runner_names.get(snap.selection_id, str(snap.selection_id)),
                        selection_id=snap.selection_id,
                        signal=Signal.SCALP,
                        p=market_p + 0.02,  # Slight edge assumed
                        market_p=market_p,
                        edge=0.02,
                        strategy=Strategy.SCALPING,
                        entry_odds=snap.best_back,
                        target_odds=snap.best_lay,
                        hedge=False,
                        confidence=Confidence.LOW,
                        reasoning="Tight spread, liquid market. Rule-based scalp opportunity.",
                    ))

        return instructions
