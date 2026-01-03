"""
Offline Cache

SQLite-based cache for offline mode support.
Stores pre-fetched problems and queues actions for sync.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import aiosqlite

logger = logging.getLogger(__name__)


class OfflineCache:
    """
    SQLite-based offline cache for glasses client.

    Features:
    - Pre-cached problems for offline use
    - Pending action queue for sync
    - Session data persistence
    - Automatic cleanup of old data
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the cache.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.halo_cache.db
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.halo_cache.db")

        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Connect to the database and initialize tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._init_tables()
        logger.info(f"Connected to offline cache: {self.db_path}")

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_tables(self):
        """Initialize database tables."""
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                problem TEXT NOT NULL,
                solution TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_count INTEGER DEFAULT 0,
                last_used TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS pending_sync (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS session_data (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_problems_type_diff
            ON problems(type, difficulty)
        """)

        await self._db.commit()

    # Problem cache methods

    async def store_problem(self, problem: Dict[str, Any]):
        """
        Store a problem in the cache.

        Args:
            problem: Problem data with id, type, category, difficulty, problem, solution
        """
        await self._db.execute("""
            INSERT OR REPLACE INTO problems
            (id, type, category, difficulty, problem, solution, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            problem.get("id", ""),
            problem.get("type", "math"),
            problem.get("category", "arithmetic"),
            problem.get("difficulty", 2),
            problem.get("problem", ""),
            str(problem.get("solution", "")),
            json.dumps(problem.get("metadata", {}))
        ))
        await self._db.commit()

    async def get_random_problem(
        self,
        problem_type: str,
        difficulty: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get a random cached problem.

        Args:
            problem_type: Type of problem (e.g., "mental_math")
            difficulty: Difficulty level (1-5)

        Returns:
            Problem dict or None if no cached problems
        """
        async with self._db.execute("""
            SELECT id, type, category, difficulty, problem, solution, metadata
            FROM problems
            WHERE type = ? AND difficulty = ?
            ORDER BY RANDOM()
            LIMIT 1
        """, (problem_type, difficulty)) as cursor:
            row = await cursor.fetchone()

            if row:
                # Update usage stats
                await self._db.execute("""
                    UPDATE problems
                    SET used_count = used_count + 1, last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (row[0],))
                await self._db.commit()

                return {
                    "id": row[0],
                    "type": row[1],
                    "category": row[2],
                    "difficulty": row[3],
                    "problem": row[4],
                    "solution": row[5],
                    "metadata": json.loads(row[6]) if row[6] else {}
                }

        return None

    async def get_problem_count(
        self,
        problem_type: Optional[str] = None,
        difficulty: Optional[int] = None
    ) -> int:
        """Get count of cached problems."""
        query = "SELECT COUNT(*) FROM problems WHERE 1=1"
        params = []

        if problem_type:
            query += " AND type = ?"
            params.append(problem_type)

        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear_old_problems(self, days: int = 30):
        """Clear problems older than N days that haven't been used."""
        await self._db.execute("""
            DELETE FROM problems
            WHERE created_at < datetime('now', '-' || ? || ' days')
            AND (last_used IS NULL OR last_used < datetime('now', '-' || ? || ' days'))
        """, (days, days))
        await self._db.commit()

    # Pending sync methods

    async def queue_for_sync(self, action: str, data: Dict[str, Any]) -> int:
        """
        Queue an action for sync when back online.

        Args:
            action: Action type (e.g., "record_attempt")
            data: Action data

        Returns:
            ID of the queued item
        """
        cursor = await self._db.execute("""
            INSERT INTO pending_sync (action, data)
            VALUES (?, ?)
        """, (action, json.dumps(data)))
        await self._db.commit()
        return cursor.lastrowid

    async def get_pending_sync(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pending sync items."""
        items = []

        async with self._db.execute("""
            SELECT id, action, data, created_at, attempts
            FROM pending_sync
            ORDER BY created_at
            LIMIT ?
        """, (limit,)) as cursor:
            async for row in cursor:
                items.append({
                    "id": row[0],
                    "action": row[1],
                    "data": json.loads(row[2]),
                    "created_at": row[3],
                    "attempts": row[4]
                })

        return items

    async def mark_synced(self, item_id: int):
        """Mark an item as synced (removes from queue)."""
        await self._db.execute(
            "DELETE FROM pending_sync WHERE id = ?",
            (item_id,)
        )
        await self._db.commit()

    async def mark_sync_attempt(self, item_id: int):
        """Record a sync attempt."""
        await self._db.execute("""
            UPDATE pending_sync
            SET attempts = attempts + 1, last_attempt = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (item_id,))
        await self._db.commit()

    async def get_pending_count(self) -> int:
        """Get count of pending sync items."""
        async with self._db.execute(
            "SELECT COUNT(*) FROM pending_sync"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    # Session data methods

    async def save_session_data(self, key: str, value: Any):
        """Save session data."""
        await self._db.execute("""
            INSERT OR REPLACE INTO session_data (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, json.dumps(value)))
        await self._db.commit()

    async def get_session_data(self, key: str) -> Optional[Any]:
        """Get session data."""
        async with self._db.execute(
            "SELECT value FROM session_data WHERE key = ?",
            (key,)
        ) as cursor:
            row = await cursor.fetchone()
            return json.loads(row[0]) if row else None

    async def clear_session_data(self):
        """Clear all session data."""
        await self._db.execute("DELETE FROM session_data")
        await self._db.commit()

    # Statistics

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}

        # Problem counts by type
        async with self._db.execute("""
            SELECT type, difficulty, COUNT(*)
            FROM problems
            GROUP BY type, difficulty
        """) as cursor:
            problems = {}
            async for row in cursor:
                key = f"{row[0]}_d{row[1]}"
                problems[key] = row[2]
            stats["problems"] = problems

        # Pending sync count
        stats["pending_sync"] = await self.get_pending_count()

        # Database size
        try:
            stats["db_size_bytes"] = os.path.getsize(self.db_path)
        except Exception:
            stats["db_size_bytes"] = 0

        return stats
