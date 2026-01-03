"""
Sync Manager

Manages synchronization between offline cache and backend.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from glasses_client.api.client import APIClient
from .cache import OfflineCache

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Manages offline/online synchronization.

    Features:
    - Pre-cache problems for offline use
    - Sync pending actions when online
    - Background sync with retry
    """

    def __init__(
        self,
        cache: OfflineCache,
        api: APIClient,
        auto_sync_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the sync manager.

        Args:
            cache: Offline cache instance
            api: API client instance
            auto_sync_interval: Auto-sync interval in seconds
        """
        self.cache = cache
        self.api = api
        self.auto_sync_interval = auto_sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._is_online = True

    async def start_background_sync(self):
        """Start background sync task."""
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Started background sync")

    async def stop_background_sync(self):
        """Stop background sync task."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        logger.info("Stopped background sync")

    async def _sync_loop(self):
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.auto_sync_interval)

                if await self._check_online():
                    await self.sync_pending()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    async def _check_online(self) -> bool:
        """Check if we're online."""
        try:
            self._is_online = await self.api.health_check()
        except Exception:
            self._is_online = False

        return self._is_online

    # Pre-caching

    async def warm_cache(
        self,
        problem_types: List[str],
        count_per_difficulty: int = 10
    ):
        """
        Pre-download problems for offline use.

        Args:
            problem_types: List of problem types to cache
            count_per_difficulty: Number of problems per difficulty level
        """
        logger.info(f"Warming cache for types: {problem_types}")

        for ptype in problem_types:
            for difficulty in range(1, 6):  # D1-D5
                try:
                    # Fetch problems from backend
                    problems = await self.api.get(
                        f"/prefetch/{ptype}",
                        params={
                            "difficulty": difficulty,
                            "count": count_per_difficulty
                        }
                    )

                    # Store in cache
                    for problem in problems.get("problems", []):
                        problem["type"] = ptype
                        await self.cache.store_problem(problem)

                    logger.debug(
                        f"Cached {len(problems.get('problems', []))} "
                        f"{ptype} D{difficulty} problems"
                    )

                except Exception as e:
                    logger.warning(f"Failed to cache {ptype} D{difficulty}: {e}")

        stats = await self.cache.get_cache_stats()
        logger.info(f"Cache warmed. Total problems: {sum(stats.get('problems', {}).values())}")

    async def ensure_minimum_cache(
        self,
        problem_type: str,
        difficulty: int,
        minimum: int = 5
    ) -> bool:
        """
        Ensure minimum number of problems are cached.

        Args:
            problem_type: Type of problems
            difficulty: Difficulty level
            minimum: Minimum number to cache

        Returns:
            True if minimum is met
        """
        current = await self.cache.get_problem_count(problem_type, difficulty)

        if current >= minimum:
            return True

        if not self._is_online:
            return current > 0

        try:
            needed = minimum - current
            problems = await self.api.get(
                f"/prefetch/{problem_type}",
                params={
                    "difficulty": difficulty,
                    "count": needed
                }
            )

            for problem in problems.get("problems", []):
                problem["type"] = problem_type
                await self.cache.store_problem(problem)

            return True

        except Exception as e:
            logger.warning(f"Failed to ensure cache minimum: {e}")
            return current > 0

    # Pending sync

    async def sync_pending(self, batch_size: int = 10) -> Dict[str, int]:
        """
        Sync pending actions to backend.

        Args:
            batch_size: Number of items to sync per batch

        Returns:
            Stats dict with synced/failed counts
        """
        stats = {"synced": 0, "failed": 0}

        pending = await self.cache.get_pending_sync(limit=batch_size)

        for item in pending:
            try:
                await self.api.post("/sync", {
                    "action": item["action"],
                    "data": item["data"],
                    "queued_at": item["created_at"]
                })

                await self.cache.mark_synced(item["id"])
                stats["synced"] += 1

            except Exception as e:
                logger.warning(f"Failed to sync item {item['id']}: {e}")
                await self.cache.mark_sync_attempt(item["id"])
                stats["failed"] += 1

        if stats["synced"] > 0:
            logger.info(f"Synced {stats['synced']} pending items")

        return stats

    async def sync_all_pending(self) -> Dict[str, int]:
        """Sync all pending items."""
        total_stats = {"synced": 0, "failed": 0}

        while True:
            pending_count = await self.cache.get_pending_count()
            if pending_count == 0:
                break

            stats = await self.sync_pending(batch_size=50)
            total_stats["synced"] += stats["synced"]
            total_stats["failed"] += stats["failed"]

            # If all failed, stop trying
            if stats["synced"] == 0 and stats["failed"] > 0:
                break

        return total_stats

    # Status

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status."""
        return {
            "is_online": self._is_online,
            "pending_count": await self.cache.get_pending_count(),
            "cache_stats": await self.cache.get_cache_stats()
        }
