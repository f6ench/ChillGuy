"""Self-healing system — Levels 1, 2, and 3.

Level 1: Code self-healing (uptime) — error recovery, retry, restart
Level 2: Strategy self-healing (performance) — detect degradation, switch strategies
Level 3: Probability model self-healing (intelligence) — bias detection, weight adjustment
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

HEAL_LOG_PATH = Path("heal_log.json")


# ─── Thresholds ────────────────────────────────────────────────────────────────

@dataclass
class HealingThresholds:
    min_win_rate: float = 0.55
    min_edge: float = 0.08
    max_trades_per_race: int = 12
    min_market_volume: float = 50000
    max_self_repair_attempts: int = 3
    heartbeat_interval: float = 60.0
    review_window: int = 50  # review after N trades
    emergency_win_rate: float = 0.45  # force paper mode below this


# ─── Heal Log ──────────────────────────────────────────────────────────────────

class HealLog:
    """Persistent log of all self-healing actions."""

    def __init__(self, path: Path | None = HEAL_LOG_PATH):
        self.path = path
        self._entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._entries = []

    def log(self, level: str, action: str, details: dict | None = None) -> None:
        entry = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "level": level,
            "action": action,
            "details": details or {},
        }
        self._entries.append(entry)
        self._save()
        logger.info("[HEAL-%s] %s: %s", level, action, details or "")

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.write_text(json.dumps(self._entries, indent=2, default=str))
        except OSError as e:
            logger.error("Failed to save heal log: %s", e)

    @property
    def entries(self) -> list[dict]:
        return self._entries


# ─── Level 1: Code Self-Healing ───────────────────────────────────────────────

class CodeHealer:
    """Handles API errors, timeouts, rate limits, and process recovery."""

    def __init__(self, thresholds: HealingThresholds, heal_log: HealLog):
        self.thresholds = thresholds
        self.heal_log = heal_log
        self._error_counts: dict[str, int] = {}
        self._last_heartbeat: float = time.time()

    def execute_with_recovery(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        **kwargs,
    ):
        """Execute a function with automatic retry and error recovery.

        Retries with exponential backoff. After max_retries, logs
        the structural failure and returns None.
        """
        error_key = func.__name__
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                # Reset error count on success
                self._error_counts[error_key] = 0
                return result
            except Exception as e:
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
                wait = backoff_base ** attempt

                self.heal_log.log("L1", "retry", {
                    "function": error_key,
                    "attempt": attempt + 1,
                    "error": str(e),
                    "wait_seconds": wait,
                })

                if self._error_counts[error_key] >= self.thresholds.max_self_repair_attempts:
                    self.heal_log.log("L1", "structural_failure", {
                        "function": error_key,
                        "total_errors": self._error_counts[error_key],
                        "action": "pausing",
                    })
                    return None

                time.sleep(wait)

        return None

    def check_heartbeat(self) -> bool:
        """Check if the system is responsive. Returns False if stale."""
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed > self.thresholds.heartbeat_interval * 2:
            self.heal_log.log("L1", "heartbeat_stale", {
                "elapsed_seconds": elapsed,
            })
            return False
        return True

    def pulse(self) -> None:
        """Record a heartbeat."""
        self._last_heartbeat = time.time()


# ─── Level 2: Strategy Self-Healing ───────────────────────────────────────────

@dataclass
class TradeResult:
    strategy: str
    signal: str
    edge: float
    pnl: float
    won: bool
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)


class StrategyHealer:
    """Monitors strategy performance and switches when degraded."""

    def __init__(self, thresholds: HealingThresholds, heal_log: HealLog):
        self.thresholds = thresholds
        self.heal_log = heal_log
        self._results: deque[TradeResult] = deque(maxlen=200)
        self._trades_per_race: dict[str, int] = {}
        self._active_strategy: str = "scalping"
        self._force_paper: bool = False

    def record_result(self, result: TradeResult) -> None:
        self._results.append(result)
        if len(self._results) % self.thresholds.review_window == 0:
            self._review_performance()

    def should_force_paper_mode(self) -> bool:
        return self._force_paper

    @property
    def active_strategy(self) -> str:
        return self._active_strategy

    def check_trade_frequency(self, race_id: str) -> bool:
        """Returns True if it's OK to trade, False if over-trading."""
        count = self._trades_per_race.get(race_id, 0)
        if count >= self.thresholds.max_trades_per_race:
            self.heal_log.log("L2", "over_trading", {
                "race_id": race_id,
                "count": count,
                "max": self.thresholds.max_trades_per_race,
            })
            return False
        self._trades_per_race[race_id] = count + 1
        return True

    def check_market_volume(self, volume: float) -> bool:
        """Returns True if market has enough liquidity."""
        if volume < self.thresholds.min_market_volume:
            self.heal_log.log("L2", "low_liquidity", {
                "volume": volume,
                "min_required": self.thresholds.min_market_volume,
            })
            return False
        return True

    def _review_performance(self) -> None:
        """Analyze recent trades and adjust strategy if needed."""
        window = list(self._results)[-self.thresholds.review_window:]
        if not window:
            return

        win_rate = sum(1 for r in window if r.won) / len(window)
        total_pnl = sum(r.pnl for r in window)
        avg_edge = sum(r.edge for r in window) / len(window)

        # Emergency: force paper mode
        if win_rate < self.thresholds.emergency_win_rate:
            self._force_paper = True
            self.heal_log.log("L2", "emergency_paper_mode", {
                "win_rate": win_rate,
                "threshold": self.thresholds.emergency_win_rate,
            })
            return

        # Strategy-specific performance
        strategy_stats: dict[str, dict] = {}
        for r in window:
            if r.strategy not in strategy_stats:
                strategy_stats[r.strategy] = {"wins": 0, "total": 0, "pnl": 0.0}
            strategy_stats[r.strategy]["total"] += 1
            strategy_stats[r.strategy]["pnl"] += r.pnl
            if r.won:
                strategy_stats[r.strategy]["wins"] += 1

        # Find underperforming strategies
        for strat, stats in strategy_stats.items():
            strat_win_rate = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            if strat_win_rate < self.thresholds.min_win_rate and strat == self._active_strategy:
                new_strategy = self._pick_replacement(strat, strategy_stats)
                self.heal_log.log("L2", "strategy_switch", {
                    "from": strat,
                    "to": new_strategy,
                    "old_win_rate": strat_win_rate,
                    "pnl": stats["pnl"],
                })
                self._active_strategy = new_strategy

    def _pick_replacement(
        self, current: str, stats: dict[str, dict]
    ) -> str:
        """Pick the best alternative strategy based on recent performance."""
        STRATEGY_FALLBACKS = {
            "scalping": "lay_to_back",
            "lay_to_back": "back_to_lay",
            "back_to_lay": "lay_to_back",
            "dobbing": "back_to_lay",
        }
        # Try the best performing strategy first
        best = None
        best_win_rate = 0
        for strat, s in stats.items():
            if strat == current:
                continue
            wr = s["wins"] / s["total"] if s["total"] > 0 else 0
            if wr > best_win_rate:
                best = strat
                best_win_rate = wr

        if best and best_win_rate >= self.thresholds.min_win_rate:
            return best

        return STRATEGY_FALLBACKS.get(current, "scalping")


# ─── Level 3: Probability Model Self-Healing ──────────────────────────────────

@dataclass
class SignalWeight:
    source: str
    weight: float = 1.0
    accuracy: float = 0.0
    sample_size: int = 0


class ModelHealer:
    """Adjusts signal weights and detects systematic biases."""

    def __init__(self, heal_log: HealLog):
        self.heal_log = heal_log
        self.signal_weights: dict[str, SignalWeight] = {
            "racing_post_form": SignalWeight("racing_post_form", weight=1.0),
            "betfair_money_flow": SignalWeight("betfair_money_flow", weight=1.2),
            "twitter_stable_accounts": SignalWeight("twitter_stable_accounts", weight=0.8),
            "parade_ring_audio": SignalWeight("parade_ring_audio", weight=1.0),
            "telegram_tipsters": SignalWeight("telegram_tipsters", weight=0.6),
            "trainer_interview": SignalWeight("trainer_interview", weight=0.9),
        }
        self._predictions: deque[dict] = deque(maxlen=500)

    def record_prediction(
        self,
        predicted_p: float,
        actual_won: bool,
        signals_used: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Record a prediction outcome for tracking."""
        self._predictions.append({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "predicted_p": predicted_p,
            "actual_won": actual_won,
            "signals_used": signals_used,
            "metadata": metadata or {},
        })

    def update_signal_weights(self) -> dict[str, float]:
        """Recalculate signal weights based on rolling accuracy.

        Should be called periodically (e.g. weekly or every 200 trades).
        """
        if len(self._predictions) < 50:
            return {k: v.weight for k, v in self.signal_weights.items()}

        # Calculate accuracy per signal source
        signal_correct: dict[str, list[bool]] = {}
        for pred in self._predictions:
            for signal in pred["signals_used"]:
                if signal not in signal_correct:
                    signal_correct[signal] = []
                signal_correct[signal].append(pred["actual_won"])

        updates = {}
        for source, outcomes in signal_correct.items():
            if source not in self.signal_weights:
                self.signal_weights[source] = SignalWeight(source)

            accuracy = sum(outcomes) / len(outcomes)
            old_weight = self.signal_weights[source].weight

            # Adjust weight: above 60% accuracy → increase, below 40% → decrease
            if accuracy >= 0.6:
                new_weight = min(old_weight * 1.1, 2.0)
            elif accuracy <= 0.4:
                new_weight = max(old_weight * 0.8, 0.1)
            else:
                new_weight = old_weight

            self.signal_weights[source].weight = round(new_weight, 2)
            self.signal_weights[source].accuracy = round(accuracy, 3)
            self.signal_weights[source].sample_size = len(outcomes)
            updates[source] = new_weight

            if abs(new_weight - old_weight) > 0.05:
                self.heal_log.log("L3", "weight_updated", {
                    "source": source,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "accuracy": accuracy,
                    "sample_size": len(outcomes),
                })

        return updates

    def detect_biases(self) -> list[str]:
        """Analyze predictions to find systematic biases.

        Returns a list of human-readable bias descriptions.
        """
        if len(self._predictions) < 100:
            return []

        biases = []
        preds = list(self._predictions)

        # Check overall calibration
        predicted_avg = sum(p["predicted_p"] for p in preds) / len(preds)
        actual_avg = sum(1 for p in preds if p["actual_won"]) / len(preds)
        diff = predicted_avg - actual_avg

        if diff > 0.1:
            biases.append(
                f"Overall overconfidence: predicted {predicted_avg:.1%} "
                f"vs actual {actual_avg:.1%} win rate"
            )
        elif diff < -0.1:
            biases.append(
                f"Overall underconfidence: predicted {predicted_avg:.1%} "
                f"vs actual {actual_avg:.1%} win rate"
            )

        # Check by confidence buckets
        high_conf = [p for p in preds if p["predicted_p"] >= 0.7]
        if high_conf:
            high_actual = sum(1 for p in high_conf if p["actual_won"]) / len(high_conf)
            if high_actual < 0.5:
                biases.append(
                    f"High-confidence bets underperforming: "
                    f"{high_actual:.1%} win rate on {len(high_conf)} trades"
                )

        return biases


# ─── Combined Self-Healing Manager ────────────────────────────────────────────

class SelfHealingManager:
    """Combines all three healing levels into a single interface."""

    def __init__(self, thresholds: HealingThresholds | None = None):
        self.thresholds = thresholds or HealingThresholds()
        self.heal_log = HealLog()
        self.code_healer = CodeHealer(self.thresholds, self.heal_log)
        self.strategy_healer = StrategyHealer(self.thresholds, self.heal_log)
        self.model_healer = ModelHealer(self.heal_log)

    def wrap_execution(self, func: Callable, *args, **kwargs):
        """Execute a function with Level 1 self-healing."""
        return self.code_healer.execute_with_recovery(func, *args, **kwargs)

    def record_trade_outcome(
        self,
        strategy: str,
        signal: str,
        edge: float,
        pnl: float,
        won: bool,
        predicted_p: float,
        signals_used: list[str],
    ) -> None:
        """Record a trade outcome for Levels 2 and 3."""
        self.strategy_healer.record_result(TradeResult(
            strategy=strategy,
            signal=signal,
            edge=edge,
            pnl=pnl,
            won=won,
        ))
        self.model_healer.record_prediction(
            predicted_p=predicted_p,
            actual_won=won,
            signals_used=signals_used,
        )

    def get_health_report(self) -> dict:
        """Generate a full system health report."""
        biases = self.model_healer.detect_biases()
        weights = {k: v.weight for k, v in self.model_healer.signal_weights.items()}

        return {
            "active_strategy": self.strategy_healer.active_strategy,
            "force_paper_mode": self.strategy_healer.should_force_paper_mode(),
            "heartbeat_ok": self.code_healer.check_heartbeat(),
            "signal_weights": weights,
            "biases_detected": biases,
            "heal_log_entries": len(self.heal_log.entries),
        }
