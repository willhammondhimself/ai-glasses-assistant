"""
HistoryTracker: SQLite-based query history and statistics tracking.

Stores all queries for analysis, debugging, and cost tracking.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class HistoryTracker:
    """
    Query history tracker using SQLite.

    Features:
    - Stores query details: engine, query, result, cost, duration
    - Provides usage statistics by engine
    - Tracks cache hit rates
    - Calculates total API costs
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the history tracker.

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "query_history.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    engine TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    result_summary TEXT,
                    cost REAL DEFAULT 0.0,
                    duration_ms INTEGER DEFAULT 0,
                    cached BOOLEAN DEFAULT FALSE,
                    tokens_used INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON queries(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_engine ON queries(engine)")

            # LeetCode Problems Cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leetcode_problems (
                    id INTEGER PRIMARY KEY,
                    slug TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    difficulty TEXT CHECK(difficulty IN ('Easy', 'Medium', 'Hard')),
                    content TEXT,
                    code_templates TEXT,
                    topics TEXT,
                    hints TEXT,
                    test_cases TEXT,
                    fetched_at INTEGER,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_leetcode_slug ON leetcode_problems(slug)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_leetcode_difficulty ON leetcode_problems(difficulty)")

            # User Solutions for LeetCode
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leetcode_solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_id INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    language TEXT NOT NULL,
                    passed BOOLEAN DEFAULT FALSE,
                    time_taken_ms INTEGER,
                    hints_used INTEGER DEFAULT 0,
                    submitted_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (problem_id) REFERENCES leetcode_problems(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_solutions_problem ON leetcode_solutions(problem_id)")

            # Flashcards with SM-2 Spaced Repetition
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    tags TEXT,
                    easiness_factor REAL DEFAULT 2.5,
                    interval_days INTEGER DEFAULT 1,
                    repetitions INTEGER DEFAULT 0,
                    due_date INTEGER NOT NULL,
                    last_reviewed INTEGER,
                    review_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_due ON flashcards(due_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_source ON flashcards(source_type, source_id)")

            # Flashcard Review History (for heatmap)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flashcard_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flashcard_id INTEGER NOT NULL,
                    quality INTEGER NOT NULL,
                    time_ms INTEGER,
                    reviewed_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (flashcard_id) REFERENCES flashcards(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_date ON flashcard_reviews(reviewed_at)")

            # Daily Activity (pre-computed for heatmap)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_activity (
                    date TEXT PRIMARY KEY,
                    flashcard_reviews INTEGER DEFAULT 0,
                    leetcode_solved INTEGER DEFAULT 0,
                    quant_problems INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    def log_query(
        self,
        engine: str,
        query: str,
        result: Optional[Dict[str, Any]] = None,
        cost: float = 0.0,
        duration_ms: int = 0,
        cached: bool = False,
        tokens: int = 0
    ) -> int:
        """
        Log a query to the history.

        Args:
            engine: Name of the engine used
            query: The query text
            result: Result dict (will be summarized)
            cost: API cost in dollars
            duration_ms: Query duration in milliseconds
            cached: Whether result was from cache
            tokens: Number of tokens used

        Returns:
            ID of the inserted record
        """
        # Create a summary of the result (truncate if needed)
        result_summary = None
        if result:
            try:
                result_str = json.dumps(result)
                result_summary = result_str[:500] if len(result_str) > 500 else result_str
            except (TypeError, ValueError):
                result_summary = str(result)[:500]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO queries (engine, query_text, result_summary, cost, duration_ms, cached, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (engine, query[:1000], result_summary, cost, duration_ms, cached, tokens)
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(
        self,
        limit: int = 50,
        engine: Optional[str] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get query history.

        Args:
            limit: Maximum number of records to return
            engine: Filter by engine name (optional)
            offset: Number of records to skip

        Returns:
            List of query records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if engine:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, engine, query_text, result_summary, cost, duration_ms, cached, tokens_used
                    FROM queries
                    WHERE engine = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (engine, limit, offset)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, engine, query_text, result_summary, cost, duration_ms, cached, tokens_used
                    FROM queries
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            dict with total_queries, total_cost, queries_by_engine,
            avg_duration_ms, cache_hit_rate, total_tokens
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total queries and cost
            cursor = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost), 0), COALESCE(SUM(tokens_used), 0) FROM queries"
            )
            total_queries, total_cost, total_tokens = cursor.fetchone()

            # Average duration
            cursor = conn.execute(
                "SELECT COALESCE(AVG(duration_ms), 0) FROM queries WHERE duration_ms > 0"
            )
            avg_duration = cursor.fetchone()[0]

            # Cache hit rate
            cursor = conn.execute(
                "SELECT COUNT(*) FROM queries WHERE cached = TRUE"
            )
            cached_count = cursor.fetchone()[0]
            cache_hit_rate = cached_count / total_queries if total_queries > 0 else 0

            # Queries by engine
            cursor = conn.execute(
                """
                SELECT engine, COUNT(*), COALESCE(SUM(cost), 0)
                FROM queries
                GROUP BY engine
                ORDER BY COUNT(*) DESC
                """
            )
            queries_by_engine = {
                row[0]: {"count": row[1], "cost": round(row[2], 4)}
                for row in cursor.fetchall()
            }

            # Recent queries (last 24 hours)
            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM queries
                WHERE timestamp > datetime('now', '-1 day')
                """
            )
            queries_last_24h = cursor.fetchone()[0]

            return {
                "total_queries": total_queries,
                "total_cost": round(total_cost, 4),
                "total_tokens": total_tokens,
                "avg_duration_ms": round(avg_duration, 1),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "queries_by_engine": queries_by_engine,
                "queries_last_24h": queries_last_24h
            }

    def clear_history(self) -> int:
        """
        Clear all history records.

        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM queries")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM queries")
            conn.commit()
            return count

    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cost breakdown for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            dict with daily costs and engine breakdown
        """
        with sqlite3.connect(self.db_path) as conn:
            # Daily costs
            cursor = conn.execute(
                """
                SELECT DATE(timestamp) as day, COALESCE(SUM(cost), 0) as daily_cost
                FROM queries
                WHERE timestamp > datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY day DESC
                """,
                (f'-{days} days',)
            )
            daily_costs = {row[0]: round(row[1], 4) for row in cursor.fetchall()}

            # Cost by engine for period
            cursor = conn.execute(
                """
                SELECT engine, COALESCE(SUM(cost), 0)
                FROM queries
                WHERE timestamp > datetime('now', ?)
                GROUP BY engine
                """,
                (f'-{days} days',)
            )
            cost_by_engine = {row[0]: round(row[1], 4) for row in cursor.fetchall()}

            return {
                "daily_costs": daily_costs,
                "cost_by_engine": cost_by_engine,
                "period_days": days
            }


# Global tracker instance
_tracker_instance: Optional[HistoryTracker] = None


def get_tracker() -> HistoryTracker:
    """Get the global history tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = HistoryTracker()
    return _tracker_instance
