"""Claude AI nomination layer — Two-Stage Independent Probability Estimation.

This is Layer 2 of the architecture. Claude processes signals in two stages:

Stage 1: Form + condition signals ONLY (no market odds). Generates an
independent probability estimate per runner from Racing API form data,
trainer quotes, and physical condition reports.

Stage 2: Stage 1 output + market prices + steam/drift signals. Identifies
where our model probability materially differs from market implied
probability — these are the actionable information imbalances.

CRITICAL: Stage 1 must NEVER see market odds. If Claude sees the odds
first, it anchors to the market and produces no independent estimate.
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

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Independent probability estimation from form data ONLY
# ──────────────────────────────────────────────────────────────────────────────

STAGE_1_SYSTEM_PROMPT = """You are a professional horse racing form analyst.
You estimate the probability of each runner winning based ONLY on the form data,
trainer/jockey statistics, course records, and condition signals provided.

CRITICAL RULES:
- You must NOT use market odds or implied probabilities in your analysis
- Your estimates must be based purely on the racing data provided
- Probabilities across all runners must sum to approximately 1.0
- Be honest about uncertainty — if form data is limited, reflect that

For each runner, assess:
1. Recent form quality (last 3-6 runs, finishing positions, beaten lengths)
2. Speed figures and RPR trends (improving, declining, consistent)
3. Course and distance suitability (past record at this track/distance)
4. Going preference (does today's ground suit this horse?)
5. Class level (is this horse running above/below its usual class?)
6. Trainer and jockey form (current win percentages, course record)
7. Draw bias (is the stall number advantageous at this course/distance?)
8. Freshness vs fitness (days since last run)
9. Pre-race intelligence (trainer quotes, paddock reports if available)

You must respond with a JSON object:
{
  "runners": [
    {
      "horse_name": "Name",
      "probability": 0.0-1.0,
      "form_rating": "strong|good|moderate|weak|unknown",
      "key_positives": ["list of positive factors"],
      "key_negatives": ["list of negative factors"],
      "reasoning": "Brief analysis"
    }
  ],
  "race_assessment": "Brief overall race assessment",
  "confidence_in_estimates": "high|medium|low"
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Edge detection — compare independent estimates to market
# ──────────────────────────────────────────────────────────────────────────────

STAGE_2_SYSTEM_PROMPT = """You are a professional Betfair Exchange horse racing trader.
You have received independent probability estimates from a form analyst (Stage 1).
Now compare those estimates against the current market prices to find information imbalances.

Your job:
- Compare our Stage 1 probability (p) vs market implied probability (market_p = 1/odds)
- Calculate edge (p - market_p) for each runner
- Only recommend trades where edge exceeds the threshold AND at least two signals align
- Factor in steam/drift signals — if informed money is moving, this strengthens the signal
- Factor in trainer sentiment — if stable connections are positive, this strengthens the signal
- Consider Betfair commission (5%) when assessing profitability
- Choose the best strategy for each opportunity

Available strategies:
1. BACK TO LAY (Springer Bot): Back a shortening horse, lay at lower price
2. LAY TO BACK (Drifter Bot): Lay a drifting horse, back at higher price
3. SCALPING: Back and lay close together for spread profit
4. DOBBING (Double or Bust): Back pre-race, lay at half odds in-play

ENTRY RULES — all three must be true:
1. Edge >= 15% (model probability exceeds market implied probability by 15%+)
2. At least two independent signals aligned (form AND steam, or form AND trainer quote, etc.)
3. Race passes quality filters

You must respond with a JSON array of trade instructions:
[{
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
  "reasoning": "Brief explanation including which signals aligned",
  "signals_used": ["list of signal names that support this trade"]
}]

If no opportunities meet all three entry rules, return an empty array: []

Be disciplined. Skip marginal opportunities. Only trade when the edge is clear."""


class ClaudeAnalyst:
    """Uses Claude to analyze markets via two-stage independent probability estimation."""

    def __init__(self, config: BetfairConfig):
        self.config = config
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1: Independent probability estimation (NO market data)
    # ──────────────────────────────────────────────────────────────────────

    def stage_1_form_analysis(
        self,
        form_data_text: str,
        content_signals_text: str = "",
    ) -> dict:
        """Stage 1: Estimate probabilities from form data only.

        Args:
            form_data_text: Formatted race card from RaceCard.to_prompt_text()
            content_signals_text: Pre-race trainer quotes and sentiment

        Returns:
            Parsed JSON with per-runner probability estimates
        """
        if not self.config.anthropic_api_key:
            logger.warning("No API key — Stage 1 skipped, returning empty estimates")
            return {}

        user_message = "Analyze this race and estimate each runner's win probability:\n\n"
        user_message += form_data_text

        if content_signals_text:
            user_message += "\n\nPre-race intelligence (trainer quotes, paddock reports):\n"
            user_message += content_signals_text

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=STAGE_1_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text
            # Extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                logger.error("Stage 1: No JSON found in response")
                return {}

            result = json.loads(text[start:end])
            logger.info(
                "Stage 1 complete: %d runners estimated, confidence=%s",
                len(result.get("runners", [])),
                result.get("confidence_in_estimates", "unknown"),
            )
            return result

        except Exception as e:
            logger.error("Stage 1 analysis failed: %s", e)
            return {}

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2: Edge detection (Stage 1 output + market data)
    # ──────────────────────────────────────────────────────────────────────

    def stage_2_edge_detection(
        self,
        stage_1_output: dict,
        market: dict,
        snapshots: list[MarketSnapshot],
        steam_signals_text: str = "",
        content_sentiments_text: str = "",
    ) -> list[TradeInstruction]:
        """Stage 2: Compare independent estimates to market for edge.

        Args:
            stage_1_output: JSON dict from stage_1_form_analysis
            market: Market dict from engine
            snapshots: Current market price snapshots
            steam_signals_text: Formatted steam/drift signals
            content_sentiments_text: Formatted trainer/paddock sentiment

        Returns:
            List of TradeInstructions for qualifying opportunities
        """
        if not self.config.anthropic_api_key:
            logger.warning("No API key — using rule-based fallback")
            return self._fallback_analysis(market, snapshots)

        user_message = self._format_stage_2_input(
            stage_1_output, market, snapshots, steam_signals_text, content_sentiments_text
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=STAGE_2_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text
            instructions = self._parse_response(text, market)
            logger.info(
                "Stage 2 complete: %d trade opportunities in %s",
                len(instructions), market["market_name"],
            )
            return instructions

        except Exception as e:
            logger.error("Stage 2 analysis failed: %s", e)
            return self._fallback_analysis(market, snapshots)

    # ──────────────────────────────────────────────────────────────────────
    # Combined pipeline (convenience method for engine)
    # ──────────────────────────────────────────────────────────────────────

    def analyze_market(
        self,
        market: dict,
        snapshots: list[MarketSnapshot],
        form_data_text: str = "",
        steam_signals_text: str = "",
        content_signals_text: str = "",
        content_sentiments_text: str = "",
    ) -> list[TradeInstruction]:
        """Full two-stage analysis pipeline.

        If form_data_text is provided, runs Stage 1 -> Stage 2.
        If not, falls back to single-stage analysis (legacy behavior).
        """
        if form_data_text:
            # Two-stage pipeline
            stage_1 = self.stage_1_form_analysis(form_data_text, content_signals_text)
            if stage_1:
                return self.stage_2_edge_detection(
                    stage_1, market, snapshots, steam_signals_text, content_sentiments_text
                )

        # Fallback: single-stage (legacy) — market data only
        return self._legacy_single_stage(market, snapshots)

    # ──────────────────────────────────────────────────────────────────────
    # Formatting helpers
    # ──────────────────────────────────────────────────────────────────────

    def _format_stage_2_input(
        self,
        stage_1_output: dict,
        market: dict,
        snapshots: list[MarketSnapshot],
        steam_signals_text: str,
        content_sentiments_text: str,
    ) -> str:
        """Format all inputs for Stage 2 prompt."""
        lines = [
            "=== STAGE 1 OUTPUT (Independent Probability Estimates) ===",
            json.dumps(stage_1_output, indent=2),
            "",
            "=== CURRENT MARKET DATA ===",
            f"Race: {market['market_name']}",
            f"Market ID: {market['market_id']}",
            "",
            "Runners (market prices):",
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
            lines.append(f"    Market Implied Prob: {snap.implied_probability:.1%}")

        if steam_signals_text:
            lines.append("")
            lines.append("=== STEAM/DRIFT SIGNALS ===")
            lines.append(steam_signals_text)

        if content_sentiments_text:
            lines.append("")
            lines.append("=== TRAINER/PADDOCK SENTIMENT ===")
            lines.append(content_sentiments_text)

        return "\n".join(lines)

    def _legacy_single_stage(
        self, market: dict, snapshots: list[MarketSnapshot]
    ) -> list[TradeInstruction]:
        """Legacy single-stage analysis when no form data is available.

        This is the circular mode — estimating from market data only.
        Kept for backwards compatibility but should not be the primary path.
        """
        if not self.config.anthropic_api_key:
            return self._fallback_analysis(market, snapshots)

        legacy_prompt = """You are a professional Betfair Exchange horse racing trader.
Analyze market data and identify trading opportunities. Note: without independent
form data, your edge is limited. Focus on scalping and spread opportunities rather
than directional bets.

Respond with a JSON array of trade instructions (or empty array []):
[{"selection": "Name", "selection_id": 123, "signal": "springer|drifter|scalp|dob",
  "p": 0.0-1.0, "market_p": 0.0-1.0, "edge": float,
  "strategy": "back_to_lay|lay_to_back|scalping|dobbing",
  "entry_odds": float, "target_odds": float, "hedge": true|false,
  "min_profit": float, "confidence": "low|medium|high",
  "reasoning": "explanation"}]"""

        market_data = self._format_market_data(market, snapshots)
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=legacy_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Analyze this market:\n\n{market_data}",
                }],
            )
            text = response.content[0].text
            return self._parse_response(text, market)
        except Exception as e:
            logger.error("Legacy analysis failed: %s", e)
            return self._fallback_analysis(market, snapshots)

    def _format_market_data(
        self, market: dict, snapshots: list[MarketSnapshot]
    ) -> str:
        """Format market data for display."""
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
                    signals_used=item.get("signals_used", []),
                )
                instructions.append(instruction)
            except Exception as e:
                logger.warning("Failed to parse instruction: %s", e)

        return instructions

    def _fallback_analysis(
        self, market: dict, snapshots: list[MarketSnapshot]
    ) -> list[TradeInstruction]:
        """Simple rule-based analysis when Claude is unavailable."""
        instructions = []
        runner_names = {r["id"]: r["name"] for r in market.get("runners", [])}

        for snap in snapshots:
            if snap.spread > 0 and snap.best_back > 0 and snap.total_matched > 5000:
                market_p = snap.implied_probability
                if snap.spread <= 0.05 * snap.best_back:
                    instructions.append(TradeInstruction(
                        race=market["market_name"],
                        market_id=market["market_id"],
                        selection=runner_names.get(snap.selection_id, str(snap.selection_id)),
                        selection_id=snap.selection_id,
                        signal=Signal.SCALP,
                        p=market_p + 0.02,
                        market_p=market_p,
                        edge=0.02,
                        strategy=Strategy.SCALPING,
                        entry_odds=snap.best_back,
                        target_odds=snap.best_lay,
                        hedge=False,
                        confidence=Confidence.LOW,
                        reasoning="Tight spread, liquid market. Rule-based scalp (no form data).",
                    ))

        return instructions
