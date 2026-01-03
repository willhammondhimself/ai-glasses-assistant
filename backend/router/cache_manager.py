"""
Semantic Cache Manager

Enhanced caching with semantic similarity matching for better hit rates.
Wraps the existing CacheManager and adds:
- Query normalization
- N-gram based semantic similarity
- Type-indexed lookups
"""

import json
import time
import re
from typing import Optional, Any, Dict, Set
from dataclasses import dataclass

# Import existing cache using importlib to avoid path issues
import importlib.util
import os

def _import_base_cache():
    """Import base cache avoiding circular imports."""
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'cache', 'redis_cache.py'
    )
    spec = importlib.util.spec_from_file_location("redis_cache", cache_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_cache, module.CacheManager

# Lazy import to avoid circular dependency at module load time
_base_cache_imported = False
_get_cache = None
_BaseCacheManager = None

def _ensure_base_cache():
    global _base_cache_imported, _get_cache, _BaseCacheManager
    if not _base_cache_imported:
        _get_cache, _BaseCacheManager = _import_base_cache()
        _base_cache_imported = True
    return _get_cache, _BaseCacheManager


@dataclass
class CacheResult:
    """Result from a cache lookup."""
    hit: bool
    data: Optional[Any] = None
    cache_type: Optional[str] = None  # "exact" | "semantic" | None
    similarity: float = 0.0
    key: Optional[str] = None


class SemanticCacheManager:
    """
    Enhanced cache manager with semantic similarity matching.

    Features:
    - Query normalization for better exact match rates
    - 3-gram Jaccard similarity for semantic matching
    - Type-based indexing for efficient lookups
    - Configurable similarity threshold
    """

    # Operator normalization map
    OPERATOR_MAP = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "—": "-",
        "·": "*",
        "∗": "*",
    }

    def __init__(
        self,
        base_cache = None,
        similarity_threshold: float = 0.85,
        max_candidates: int = 100,
        ngram_size: int = 3
    ):
        """
        Initialize semantic cache manager.

        Args:
            base_cache: Existing cache manager (uses global if not provided)
            similarity_threshold: Minimum similarity for semantic match (0.0-1.0)
            max_candidates: Max candidates to check for semantic similarity
            ngram_size: Size of n-grams for similarity calculation
        """
        if base_cache is None:
            get_cache, _ = _ensure_base_cache()
            self._base = get_cache()
        else:
            self._base = base_cache
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.ngram_size = ngram_size

        # In-memory index of queries by type for semantic search
        # Format: {problem_type: [(normalized_query, cache_key), ...]}
        self._type_index: Dict[str, list] = {}

        # Stats for semantic matching
        self._semantic_stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "similarity_checks": 0
        }

    def normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache matching.

        Transformations:
        - Lowercase
        - Collapse whitespace
        - Normalize operators
        - Remove punctuation variations
        - Standardize variable names (optional)
        """
        if not query:
            return ""

        # Lowercase and strip
        normalized = query.lower().strip()

        # Collapse whitespace
        normalized = " ".join(normalized.split())

        # Normalize operators
        for old, new in self.OPERATOR_MAP.items():
            normalized = normalized.replace(old, new)

        # Remove extra spaces around operators
        normalized = re.sub(r'\s*([+\-*/=^])\s*', r'\1', normalized)

        # Normalize common mathematical notation
        normalized = normalized.replace("**", "^")  # power notation

        return normalized

    def get_ngrams(self, text: str) -> Set[str]:
        """
        Extract character n-grams from text.

        Args:
            text: Input text

        Returns:
            Set of n-gram strings
        """
        if len(text) < self.ngram_size:
            return {text} if text else set()

        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def jaccard_similarity(self, s1: Set[str], s2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            s1: First set of n-grams
            s2: Second set of n-grams

        Returns:
            Similarity score 0.0-1.0
        """
        if not s1 or not s2:
            return 0.0

        intersection = len(s1 & s2)
        union = len(s1 | s2)

        return intersection / union if union > 0 else 0.0

    def get(self, query: str, problem_type: str) -> CacheResult:
        """
        Get cached result with semantic similarity fallback.

        Args:
            query: The problem query
            problem_type: Type of problem (math, cs, etc.)

        Returns:
            CacheResult with hit status and data
        """
        normalized = self.normalize_query(query)

        # Step 1: Try exact match (fast path)
        exact_key = self._make_key(normalized, problem_type)
        cached = self._base.get(exact_key)

        if cached:
            self._semantic_stats["exact_hits"] += 1
            return CacheResult(
                hit=True,
                data=cached.get("result"),
                cache_type="exact",
                similarity=1.0,
                key=exact_key
            )

        # Step 2: Try semantic similarity search
        if problem_type in self._type_index:
            candidates = self._type_index[problem_type][-self.max_candidates:]
            query_ngrams = self.get_ngrams(normalized)

            best_match = None
            best_similarity = 0.0

            for candidate_query, candidate_key in candidates:
                self._semantic_stats["similarity_checks"] += 1
                candidate_ngrams = self.get_ngrams(candidate_query)
                similarity = self.jaccard_similarity(query_ngrams, candidate_ngrams)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (candidate_query, candidate_key)

            if best_match and best_similarity >= self.similarity_threshold:
                cached = self._base.get(best_match[1])
                if cached:
                    self._semantic_stats["semantic_hits"] += 1
                    return CacheResult(
                        hit=True,
                        data=cached.get("result"),
                        cache_type="semantic",
                        similarity=best_similarity,
                        key=best_match[1]
                    )

        self._semantic_stats["misses"] += 1
        return CacheResult(hit=False)

    def set(
        self,
        query: str,
        problem_type: str,
        result: Any,
        ttl: int = 86400,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Store result in cache with indexing for semantic search.

        Args:
            query: The problem query
            problem_type: Type of problem
            result: Result data to cache
            ttl: Time to live in seconds
            metadata: Optional metadata to store

        Returns:
            True if cached successfully
        """
        normalized = self.normalize_query(query)
        key = self._make_key(normalized, problem_type)

        # Prepare cache entry
        cache_entry = {
            "query": normalized,
            "result": result,
            "problem_type": problem_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # Store in base cache
        success = self._base.set(key, cache_entry, ttl)

        if success:
            # Add to type index for semantic search
            if problem_type not in self._type_index:
                self._type_index[problem_type] = []

            # Avoid duplicate entries
            entry = (normalized, key)
            if entry not in self._type_index[problem_type]:
                self._type_index[problem_type].append(entry)

                # Keep index bounded
                if len(self._type_index[problem_type]) > self.max_candidates * 2:
                    self._type_index[problem_type] = self._type_index[problem_type][-self.max_candidates:]

        return success

    def _make_key(self, normalized_query: str, problem_type: str) -> str:
        """Generate cache key from normalized query and type."""
        return self._base.generate_key(f"router:{problem_type}", normalized_query)

    def stats(self) -> dict:
        """Get combined cache statistics."""
        base_stats = self._base.stats()

        total_hits = self._semantic_stats["exact_hits"] + self._semantic_stats["semantic_hits"]
        total_requests = total_hits + self._semantic_stats["misses"]

        return {
            **base_stats,
            "semantic": {
                "exact_hits": self._semantic_stats["exact_hits"],
                "semantic_hits": self._semantic_stats["semantic_hits"],
                "misses": self._semantic_stats["misses"],
                "similarity_checks": self._semantic_stats["similarity_checks"],
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
                "semantic_hit_rate": self._semantic_stats["semantic_hits"] / total_hits if total_hits > 0 else 0,
                "indexed_types": list(self._type_index.keys()),
                "index_size": sum(len(v) for v in self._type_index.values())
            }
        }

    def clear_index(self):
        """Clear the semantic search index."""
        self._type_index.clear()

    def warm_cache(self, entries: list[tuple[str, str, Any]]):
        """
        Warm the cache with pre-computed entries.

        Args:
            entries: List of (query, problem_type, result) tuples
        """
        for query, problem_type, result in entries:
            self.set(query, problem_type, result)


# Global instance
_semantic_cache: Optional[SemanticCacheManager] = None


def get_semantic_cache() -> SemanticCacheManager:
    """Get the global semantic cache manager instance."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCacheManager()
    return _semantic_cache


# Alias for cleaner imports
CacheManager = SemanticCacheManager
