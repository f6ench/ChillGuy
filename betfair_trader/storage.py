"""Trade logging and feedback loop storage.

Layer 4: Logs every trade with Claude's reasoning, tracks outcomes,
and provides data for the feedback loop to improve probability estimation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from betfair_trader.models import TradeRecord

logger = logging.getLogger(__name__)

DB_PATH = Path("trades.db")


class TradeLogger:
    """SQLite-backed trade logger for the feedback loop."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                selection TEXT NOT NULL,
                selection_id INTEGER NOT NULL,
                signal TEXT NOT NULL,
                strategy TEXT NOT NULL,
                p REAL NOT NULL,
                market_p REAL NOT NULL,
                edge REAL NOT NULL,
                confidence TEXT NOT NULL,
                reasoning TEXT,
                entry_odds REAL,
                target_odds REAL,
                entry_matched INTEGER DEFAULT 0,
                hedge_matched INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0.0,
                actual_outcome TEXT,
                orders_json TEXT,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER REFERENCES trades(id),
                timestamp TEXT NOT NULL,
                predicted_p REAL NOT NULL,
                actual_won INTEGER,
                bias_type TEXT,
                lesson TEXT
            )
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def log_trade(self, record: TradeRecord) -> int:
        """Log a trade and return the trade ID."""
        conn = self._get_conn()
        orders_json = json.dumps([o.model_dump() for o in record.orders], default=str)
        cursor = conn.execute(
            """
            INSERT INTO trades (
                timestamp, market_id, selection, selection_id,
                signal, strategy, p, market_p, edge, confidence,
                reasoning, entry_odds, target_odds,
                entry_matched, hedge_matched, pnl, orders_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp.isoformat(),
                record.instruction.market_id,
                record.instruction.selection,
                record.instruction.selection_id,
                record.instruction.signal.value,
                record.instruction.strategy.value,
                record.instruction.p,
                record.instruction.market_p,
                record.instruction.edge,
                record.instruction.confidence.value,
                record.instruction.reasoning,
                record.instruction.entry_odds,
                record.instruction.target_odds,
                int(record.entry_matched),
                int(record.hedge_matched),
                record.pnl,
                orders_json,
                record.notes,
            ),
        )
        conn.commit()
        trade_id = cursor.lastrowid
        logger.info("Trade logged: id=%d, %s on %s", trade_id,
                     record.instruction.strategy.value, record.instruction.selection)
        return trade_id

    def update_outcome(self, trade_id: int, pnl: float, outcome: str) -> None:
        """Update a trade with the actual outcome for the feedback loop."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE trades SET pnl = ?, actual_outcome = ? WHERE id = ?",
            (pnl, outcome, trade_id),
        )
        conn.commit()

    def log_feedback(
        self,
        trade_id: int,
        predicted_p: float,
        actual_won: bool,
        bias_type: str = "",
        lesson: str = "",
    ) -> None:
        """Log feedback for model improvement."""
        import datetime as dt

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO feedback (trade_id, timestamp, predicted_p, actual_won, bias_type, lesson)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trade_id, dt.datetime.utcnow().isoformat(), predicted_p,
             int(actual_won), bias_type, lesson),
        )
        conn.commit()

    def get_performance_summary(self) -> dict:
        """Get overall trading performance metrics."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(edge) as avg_edge,
                AVG(p) as avg_predicted_p
            FROM trades
        """).fetchone()
        return dict(row) if row else {}

    def get_bias_report(self) -> list[dict]:
        """Analyze prediction accuracy to identify systematic biases."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT
                strategy,
                signal,
                COUNT(*) as count,
                AVG(p) as avg_predicted_p,
                AVG(CASE WHEN actual_outcome = 'won' THEN 1.0 ELSE 0.0 END) as actual_win_rate,
                AVG(edge) as avg_edge,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE actual_outcome IS NOT NULL
            GROUP BY strategy, signal
        """).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
