"""
CacheManager: Redis-based caching with in-memory fallback.

Provides transparent caching for API responses to reduce costs and latency.
Falls back to in-memory dict if Redis is unavailable.
"""

import os
import json
import hashlib
import time
from typing import Optional, Any, Dict
from dotenv import load_dotenv

load_dotenv()


class CacheManager:
    """
    Cache manager with Redis primary and in-memory fallback.

    Features:
    - Automatic Redis detection with silent fallback
    - TTL-based expiration (default 24 hours)
    - Cache key generation from query hash
    - Hit/miss statistics tracking
    """

    def __init__(self):
        self._redis = None
        self._memory: Dict[str, Dict[str, Any]] = {}  # key -> {value, expires_at}
        self._stats = {"hits": 0, "misses": 0}
        self._try_connect_redis()

    def _try_connect_redis(self):
        """Attempt to connect to Redis, fail silently if unavailable."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self._redis.ping()
        except Exception:
            # Redis not available, use in-memory fallback
            self._redis = None

    def generate_key(self, engine: str, query: Any) -> str:
        """
        Generate a cache key from engine name and query.

        Args:
            engine: Name of the engine (math, cs, poker, etc.)
            query: Query data (string or dict)

        Returns:
            SHA256 hash string as cache key
        """
        if isinstance(query, dict):
            query_str = json.dumps(query, sort_keys=True)
        else:
            query_str = str(query)

        combined = f"{engine}:{query_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value dict or None if not found/expired
        """
        # Try Redis first
        if self._redis:
            try:
                cached = self._redis.get(key)
                if cached:
                    self._stats["hits"] += 1
                    return json.loads(cached)
            except Exception:
                pass  # Fall through to memory cache

        # Try memory cache
        if key in self._memory:
            entry = self._memory[key]
            if entry["expires_at"] > time.time():
                self._stats["hits"] += 1
                return entry["value"]
            else:
                # Expired, remove it
                del self._memory[key]

        self._stats["misses"] += 1
        return None

    def set(self, key: str, value: dict, ttl: int = 86400) -> bool:
        """
        Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default 24 hours)

        Returns:
            True if cached successfully
        """
        # Try Redis first
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value))
                return True
            except Exception:
                pass  # Fall through to memory cache

        # Use memory cache
        self._memory[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
        return True

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        deleted = False

        if self._redis:
            try:
                deleted = self._redis.delete(key) > 0
            except Exception:
                pass

        if key in self._memory:
            del self._memory[key]
            deleted = True

        return deleted

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of keys cleared
        """
        count = 0

        if self._redis:
            try:
                # Only clear keys we might have set (be careful in production!)
                # In a real app, you'd use a key prefix
                count = self._redis.flushdb()
            except Exception:
                pass

        memory_count = len(self._memory)
        self._memory.clear()

        return count + memory_count

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict with hits, misses, hit_rate, size, backend
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        size = len(self._memory)
        if self._redis:
            try:
                size = self._redis.dbsize()
            except Exception:
                pass

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 3),
            "size": size,
            "backend": "redis" if self._redis else "memory"
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from memory cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired = [k for k, v in self._memory.items() if v["expires_at"] <= now]
        for key in expired:
            del self._memory[key]
        return len(expired)

    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is the active backend."""
        return self._redis is not None


# Global cache instance
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance
