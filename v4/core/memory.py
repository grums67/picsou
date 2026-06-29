"""Picsou v4 — SQLite memory for persistent agent state.

All trades, strategies, observations, and lessons live here.
The LLM reads from and writes to this database every cycle.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Memory:
    """Persistent memory backed by SQLite.

    Tables:
      - trades: every executed trade with full context
      - strategies: AI-generated strategies, their status and weights
      - observations: things the agent noticed (market patterns, anomalies)
      - lessons: learned rules that persist across sessions
      - snapshots: portfolio state at each cycle
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self._conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL DEFAULT 0,
                strategy TEXT,
                confidence REAL DEFAULT 0,
                reasoning TEXT,
                pnl REAL,
                close_price REAL,
                close_timestamp TEXT,
                status TEXT DEFAULT 'open'
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                name TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT DEFAULT 'probation',
                weight REAL DEFAULT 0.1,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                created_at TEXT,
                dormant_since TEXT,
                last_evaluated TEXT,
                metadata TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                relevance TEXT DEFAULT 'medium'
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                lesson TEXT NOT NULL,
                context TEXT,
                active INTEGER DEFAULT 1
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL,
                positions_count INTEGER,
                total_pnl REAL,
                return_pct REAL,
                active_strategies TEXT,
                cycle_number INTEGER
            )
        """)

        self._conn.commit()

    # ── Trades ──────────────────────────────────────────────────────────

    def log_trade(self, exchange: str, symbol: str, side: str,
                  amount: float, price: float, fee: float = 0,
                  strategy: str = "", confidence: float = 0,
                  reasoning: str = "") -> int:
        """Log an executed trade."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO trades (timestamp, exchange, symbol, side, amount, price, fee,
                                strategy, confidence, reasoning, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (datetime.now(timezone.utc).isoformat(), exchange, symbol, side,
              amount, price, fee, strategy, confidence, reasoning))
        self._conn.commit()
        return cur.lastrowid

    def close_trade(self, trade_id: int, close_price: float, pnl: float):
        """Mark a trade as closed with PnL."""
        self._conn.execute("""
            UPDATE trades SET status='closed', close_price=?, pnl=?,
                              close_timestamp=?
            WHERE id=?
        """, (close_price, pnl, datetime.now(timezone.utc).isoformat(), trade_id))
        self._conn.commit()

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open (unclosed) trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get all closed trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status='closed' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_by_strategy(self, strategy: str, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE strategy=? ORDER BY id DESC LIMIT ?",
            (strategy, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_stats(self, since_hours: int = 24) -> Dict[str, Any]:
        """Aggregate trade statistics for the last N hours."""
        rows = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl
            FROM trades
            WHERE status='closed'
              AND close_timestamp > datetime('now', ?)
        """, (f"-{since_hours} hours",)).fetchone()
        total = rows["total"] or 0
        wins = rows["wins"] or 0
        return {
            "total_closed_trades": total,
            "winning_trades": wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": rows["total_pnl"] or 0,
            "avg_pnl": rows["avg_pnl"] or 0,
        }

    # ── Strategies ──────────────────────────────────────────────────────

    def register_strategy(self, name: str, filename: str = "",
                          metadata: Dict = None) -> bool:
        """Register a new strategy. Returns True if created, False if exists."""
        try:
            self._conn.execute("""
                INSERT INTO strategies (name, filename, created_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (name, filename, datetime.now(timezone.utc).isoformat(),
                  json.dumps(metadata or {})))
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Already exists

    def update_strategy(self, name: str, **kwargs):
        """Update strategy fields."""
        if not kwargs:
            return
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [name]
        self._conn.execute(f"UPDATE strategies SET {sets} WHERE name=?", vals)
        self._conn.commit()

    def get_strategy(self, name: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE name=?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_active_strategies(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM strategies WHERE status IN ('active', 'probation') ORDER BY weight DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_strategies(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM strategies ORDER BY weight DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_strategy_status(self, name: str, status: str):
        """Set strategy status: active, probation, dormant."""
        dormant_since = None
        if status == "dormant":
            dormant_since = datetime.now(timezone.utc).isoformat()
        self._conn.execute("""
            UPDATE strategies SET status=?, dormant_since=? WHERE name=?
        """, (status, dormant_since, name))
        self._conn.commit()

    # ── Observations ────────────────────────────────────────────────────

    def add_observation(self, category: str, content: str, relevance: str = "medium"):
        """Add an observation, but skip if a very similar one already exists."""
        # Deduplication: check if a similar observation exists in the last 50
        recent = self._conn.execute(
            "SELECT content FROM observations ORDER BY id DESC LIMIT 50"
        ).fetchall()
        normalized = content.strip().lower()
        for row in recent:
            if normalized in row[0].strip().lower() or row[0].strip().lower() in normalized:
                # Similar observation already exists, skip
                return

        self._conn.execute("""
            INSERT INTO observations (timestamp, category, content, relevance)
            VALUES (?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), category, content, relevance))
        self._conn.commit()

    def get_recent_observations(self, limit: int = 20) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM observations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Lessons ─────────────────────────────────────────────────────────

    def add_lesson(self, lesson: str, context: str = ""):
        """Add a lesson, but skip if a very similar one already exists."""
        # Deduplication: check if a similar lesson exists among active lessons
        active = self._conn.execute(
            "SELECT lesson FROM lessons WHERE active=1"
        ).fetchall()
        normalized = lesson.strip().lower()
        for row in active:
            existing = row[0].strip().lower()
            # Skip if one is a substring of the other (same core lesson)
            if normalized in existing or existing in normalized:
                return

        # Also deactivate old lessons that are substrings of the new one
        # (new one is more specific/refined)
        for row in active:
            existing = row[0].strip().lower()
            if existing in normalized and existing != normalized:
                # Find the id for this old lesson and deactivate it
                self._conn.execute(
                    "UPDATE lessons SET active=0 WHERE lesson=? AND active=1",
                    (row[0],)
                )

        self._conn.execute("""
            INSERT INTO lessons (timestamp, lesson, context) VALUES (?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), lesson, context))
        self._conn.commit()

    def get_active_lessons(self, limit: int = 20) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM lessons WHERE active=1 ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def deactivate_lesson(self, lesson_id: int):
        self._conn.execute("UPDATE lessons SET active=0 WHERE id=?", (lesson_id,))
        self._conn.commit()

    # ── Snapshots ───────────────────────────────────────────────────────

    def save_snapshot(self, balance: float, positions_count: int,
                      total_pnl: float, return_pct: float,
                      active_strategies: List[str], cycle_number: int):
        self._conn.execute("""
            INSERT INTO snapshots (timestamp, balance, positions_count, total_pnl,
                                   return_pct, active_strategies, cycle_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), balance, positions_count,
              total_pnl, return_pct, json.dumps(active_strategies), cycle_number))
        self._conn.commit()

    def get_latest_snapshot(self) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ── Context for LLM ─────────────────────────────────────────────────

    def get_context_for_llm(self) -> Dict[str, Any]:
        """Build the full context dict that gets sent to the LLM each brain cycle."""
        return {
            "trade_stats_24h": self.get_trade_stats(since_hours=24),
            "trade_stats_7d": self.get_trade_stats(since_hours=168),
            "recent_trades": self.get_recent_trades(limit=10),
            "strategies": self.get_all_strategies(),
            "active_strategies": self.get_active_strategies(),
            "recent_observations": self.get_recent_observations(limit=10),
            "lessons": self.get_active_lessons(limit=10),
            "latest_snapshot": self.get_latest_snapshot(),
        }