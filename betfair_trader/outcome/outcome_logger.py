"""Outcome logging for model validation and signal calibration.

Logs all signal values, model probabilities, edge calculations, and bet
details at entry time. Records outcomes after each race settles.
Generates summary reports for model improvement.

This is the foundation for all future model improvement — without outcome
logging, signal weights and edge thresholds cannot be validated.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("outcomes.db")


@dataclass
class SignalSnapshot:
    """Snapshot of all signal values at the moment a bet was placed."""

    # Form-based signals (from Racing API)
    form_probability: float = 0.0  # Stage 1 independent estimate
    speed_figure_avg: float = 0.0
    course_win_pct: float = 0.0
    distance_win_pct: float = 0.0
    class_change: int = 0
    days_since_run: int = 0

    # Market signals
    steam_ratio: float = 0.0
    volume_ratio: float = 0.0
    movement_type: str = "STABLE"  # STEAM/DRIFT/STABLE

    # Content signals
    trainer_sentiment: str = "neutral"
    trainer_sentiment_confidence: float = 0.0
    content_sources_count: int = 0

    # Composite
    signals_aligned: int = 0  # how many signals point same direction


@dataclass
class BetRecord:
    """Complete record of a bet for outcome logging."""

    # Identifiers
    bet_id: str = ""
    timestamp: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())

    # Race info
    race_id: str = ""
    course: str = ""
    race_time: str = ""
    race_class: int = 0
    field_size: int = 0
    going: str = ""
    race_type: str = ""

    # Selection info
    horse_name: str = ""
    horse_id: str = ""
    trainer: str = ""
    jockey: str = ""
    draw: int = 0

    # Signal values at entry
    signals: SignalSnapshot = field(default_factory=SignalSnapshot)

    # Probability estimates
    model_probability: float = 0.0  # Stage 1 independent estimate
    market_implied_probability: float = 0.0  # 1/odds at entry
    edge_at_entry: float = 0.0  # model_p - market_p

    # Execution
    stake: float = 0.0
    entry_odds: float = 0.0
    strategy: str = ""
    confidence: str = ""

    # Outcome (filled post-race)
    finishing_position: int = 0
    won: bool = False
    pnl: float = 0.0
    bsp: float = 0.0  # Betfair Starting Price

    # Post-race review
    signals_correct: Optional[bool] = None
    review_notes: str = ""


class OutcomeLogger:
    """SQLite-backed outcome logger for model validation.

    Logs every bet with full signal snapshots, tracks outcomes,
    and generates validation reports.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                race_id TEXT,
                course TEXT,
                race_time TEXT,
                race_class INTEGER,
                field_size INTEGER,
                going TEXT,
                race_type TEXT,
                horse_name TEXT,
                horse_id TEXT,
                trainer TEXT,
                jockey TEXT,
                draw INTEGER,
                signals_json TEXT,
                model_probability REAL,
                market_implied_probability REAL,
                edge_at_entry REAL,
                stake REAL,
                entry_odds REAL,
                strategy TEXT,
                confidence TEXT,
                finishing_position INTEGER DEFAULT 0,
                won INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0.0,
                bsp REAL DEFAULT 0.0,
                signals_correct INTEGER,
                review_notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                direction TEXT,
                was_correct INTEGER,
                bet_id TEXT REFERENCES bets(bet_id),
                timestamp TEXT
            )
        """)
        conn.commit()

    def log_bet(self, record: BetRecord) -> int:
        """Log a bet at entry time with all signal values."""
        conn = self._get_conn()
        signals_json = json.dumps(asdict(record.signals))

        cursor = conn.execute(
            """
            INSERT INTO bets (
                bet_id, timestamp, race_id, course, race_time, race_class,
                field_size, going, race_type, horse_name, horse_id, trainer,
                jockey, draw, signals_json, model_probability,
                market_implied_probability, edge_at_entry, stake, entry_odds,
                strategy, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.bet_id, record.timestamp, record.race_id, record.course,
                record.race_time, record.race_class, record.field_size,
                record.going, record.race_type, record.horse_name,
                record.horse_id, record.trainer, record.jockey, record.draw,
                signals_json, record.model_probability,
                record.market_implied_probability, record.edge_at_entry,
                record.stake, record.entry_odds, record.strategy, record.confidence,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info("Bet logged: %s on %s (edge=%.3f)", record.horse_name,
                     record.course, record.edge_at_entry)
        return row_id

    def record_outcome(
        self,
        bet_id: str,
        finishing_position: int,
        won: bool,
        pnl: float,
        bsp: float = 0.0,
    ) -> None:
        """Record the outcome after a race settles."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE bets SET finishing_position = ?, won = ?, pnl = ?, bsp = ?
            WHERE bet_id = ?
            """,
            (finishing_position, int(won), pnl, bsp, bet_id),
        )
        conn.commit()
        logger.info("Outcome recorded: %s — pos=%d, won=%s, pnl=%.2f",
                     bet_id, finishing_position, won, pnl)

    def record_signal_accuracy(
        self,
        bet_id: str,
        signal_name: str,
        direction: str,
        was_correct: bool,
    ) -> None:
        """Record whether an individual signal was correct for a bet."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO signal_accuracy (signal_name, direction, was_correct, bet_id, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (signal_name, direction, int(was_correct), bet_id, dt.datetime.utcnow().isoformat()),
        )
        conn.commit()

    def get_summary(self, last_n: int = 0) -> dict:
        """Get overall performance summary."""
        conn = self._get_conn()
        query = """
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as winners,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(edge_at_entry) as avg_edge,
                AVG(model_probability) as avg_model_p,
                AVG(market_implied_probability) as avg_market_p
            FROM bets
            WHERE finishing_position > 0
        """
        if last_n > 0:
            query += f" ORDER BY id DESC LIMIT {last_n}"

        row = conn.execute(query).fetchone()
        if not row or row["total_bets"] == 0:
            return {}

        total = row["total_bets"]
        winners = row["winners"]
        return {
            "total_bets": total,
            "winners": winners,
            "win_rate": winners / total if total > 0 else 0,
            "total_pnl": row["total_pnl"],
            "avg_pnl": row["avg_pnl"],
            "avg_edge": row["avg_edge"],
            "avg_model_p": row["avg_model_p"],
            "avg_market_p": row["avg_market_p"],
            "roi": row["total_pnl"] / (total * row["avg_pnl"]) if row["avg_pnl"] else 0,
        }

    def get_edge_vs_outcome(self) -> list[dict]:
        """Analyze whether edge at entry correlates with win rate.

        Groups bets by edge bucket and shows win rate per bucket.
        This is the most important validation check.
        """
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT
                CASE
                    WHEN edge_at_entry < 0.05 THEN '0-5%'
                    WHEN edge_at_entry < 0.10 THEN '5-10%'
                    WHEN edge_at_entry < 0.15 THEN '10-15%'
                    WHEN edge_at_entry < 0.20 THEN '15-20%'
                    ELSE '20%+'
                END as edge_bucket,
                COUNT(*) as count,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as winners,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM bets
            WHERE finishing_position > 0
            GROUP BY edge_bucket
            ORDER BY edge_bucket
        """).fetchall()
        return [dict(r) for r in rows]

    def get_signal_contribution(self) -> list[dict]:
        """Analyze which signals are contributing predictive value."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT
                signal_name,
                COUNT(*) as sample_size,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                ROUND(
                    CAST(SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) AS REAL)
                    / COUNT(*), 3
                ) as accuracy
            FROM signal_accuracy
            GROUP BY signal_name
            ORDER BY accuracy DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_weekly_report(self) -> dict:
        """Generate a weekly summary report."""
        conn = self._get_conn()
        week_ago = (dt.datetime.utcnow() - dt.timedelta(days=7)).isoformat()

        row = conn.execute("""
            SELECT
                COUNT(*) as bets_placed,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as winners,
                SUM(pnl) as week_pnl,
                AVG(edge_at_entry) as avg_edge
            FROM bets
            WHERE timestamp >= ? AND finishing_position > 0
        """, (week_ago,)).fetchone()

        # Strategy breakdown
        strategies = conn.execute("""
            SELECT
                strategy,
                COUNT(*) as count,
                SUM(pnl) as pnl,
                AVG(edge_at_entry) as avg_edge
            FROM bets
            WHERE timestamp >= ? AND finishing_position > 0
            GROUP BY strategy
        """, (week_ago,)).fetchall()

        return {
            "period": f"Week ending {dt.datetime.utcnow().strftime('%Y-%m-%d')}",
            "bets_placed": row["bets_placed"] if row else 0,
            "winners": row["winners"] if row else 0,
            "week_pnl": row["week_pnl"] if row else 0,
            "avg_edge": row["avg_edge"] if row else 0,
            "by_strategy": [dict(s) for s in strategies],
        }

    def export_csv(self, filepath: str) -> None:
        """Export all bet records to CSV for external analysis."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM bets ORDER BY timestamp").fetchall()

        if not rows:
            logger.warning("No bets to export")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(tuple(row))

        logger.info("Exported %d bets to %s", len(rows), filepath)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
