"""
Poker Cache - Smart caching to avoid repeated API calls.
Caches analysis results for similar spots.
"""
import time
import hashlib
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached analysis result."""
    key: str
    result: Dict[str, Any]
    timestamp: float
    hits: int = 0
    hero_cards: str = ""
    board: str = ""
    villain_type: str = ""


class PokerCache:
    """
    Cache for poker analysis results.

    Saves API costs by returning cached results for:
    - Identical spots (same cards, board, villain type)
    - Similar spots (same hand category, similar board texture)

    LRU eviction with time-based expiry.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 3600,  # 1 hour default
        similarity_matching: bool = True
    ):
        """
        Initialize poker cache.

        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for entries
            similarity_matching: Enable fuzzy matching for similar spots
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_matching = similarity_matching

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cost_saved": 0.0
        }

    def _make_key(
        self,
        hero_cards: List[str],
        board: List[str],
        villain_type: str,
        action_facing: str = ""
    ) -> str:
        """
        Generate cache key from spot parameters.

        Args:
            hero_cards: ["Ah", "Kc"]
            board: ["9s", "7h", "3d"]
            villain_type: "calling_station"
            action_facing: "bet", "check", etc.

        Returns:
            Hash key string
        """
        # Normalize cards (sort for consistent ordering)
        hero_sorted = sorted(hero_cards)
        board_sorted = sorted(board)

        key_parts = [
            "_".join(hero_sorted),
            "_".join(board_sorted),
            villain_type,
            action_facing
        ]

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _make_fuzzy_key(
        self,
        hero_cards: List[str],
        board: List[str],
        villain_type: str
    ) -> str:
        """
        Generate fuzzy key for similarity matching.

        Uses hand categories and board textures instead of exact cards.
        """
        # Categorize hero's hand
        hand_category = self._categorize_hand(hero_cards)

        # Categorize board texture
        board_texture = self._categorize_board(board)

        return f"{hand_category}|{board_texture}|{villain_type}"

    def _categorize_hand(self, cards: List[str]) -> str:
        """Categorize hand into type (for fuzzy matching)."""
        if len(cards) != 2:
            return "unknown"

        ranks = [c[0] for c in cards]
        suits = [c[1] for c in cards]

        # Check for pairs
        if ranks[0] == ranks[1]:
            rank = ranks[0]
            if rank in "AKQJ":
                return "premium_pair"
            elif rank in "T987":
                return "medium_pair"
            else:
                return "small_pair"

        # Check suited
        suited = suits[0] == suits[1]

        # Categorize by high card
        high_cards = sum(1 for r in ranks if r in "AKQJT")

        if high_cards == 2:
            return "suited_broadway" if suited else "broadway"
        elif high_cards == 1:
            return "suited_ax" if suited and "A" in ranks else "one_high"
        else:
            return "suited_connector" if suited else "low_cards"

    def _categorize_board(self, board: List[str]) -> str:
        """Categorize board texture."""
        if not board:
            return "preflop"

        ranks = [c[0] for c in board]
        suits = [c[1] for c in board]

        # Check for flush possibilities
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit = max(suit_counts.values()) if suit_counts else 0

        # Check for pairs on board
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        paired = max(rank_counts.values()) > 1 if rank_counts else False

        # Texture categories
        textures = []

        if paired:
            textures.append("paired")

        if max_suit >= 3:
            textures.append("flush_draw" if max_suit == 3 else "four_flush")

        # High/low board
        high_count = sum(1 for r in ranks if r in "AKQJT")
        if high_count >= 2:
            textures.append("high")
        else:
            textures.append("low")

        return "_".join(textures) if textures else "dry"

    def get(
        self,
        hero_cards: List[str],
        board: List[str],
        villain_type: str,
        action_facing: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available.

        Args:
            hero_cards: Hero's hole cards
            board: Community cards
            villain_type: Villain archetype
            action_facing: Action hero is facing

        Returns:
            Cached result dict or None
        """
        # Try exact match first
        key = self._make_key(hero_cards, board, villain_type, action_facing)

        if key in self._cache:
            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.timestamp < self.ttl_seconds:
                entry.hits += 1
                self._stats["hits"] += 1
                self._stats["cost_saved"] += 0.009  # Saved one API call

                # Move to end (LRU)
                self._cache.move_to_end(key)

                logger.debug(f"Cache hit (exact): {key}")
                return entry.result.copy()

            # Expired - remove
            del self._cache[key]

        # Try fuzzy match if enabled
        if self.similarity_matching:
            fuzzy_key = self._make_fuzzy_key(hero_cards, board, villain_type)

            for cached_key, entry in self._cache.items():
                cached_fuzzy = self._make_fuzzy_key(
                    entry.hero_cards.split(),
                    entry.board.split(),
                    entry.villain_type
                )

                if cached_fuzzy == fuzzy_key:
                    if time.time() - entry.timestamp < self.ttl_seconds:
                        entry.hits += 1
                        self._stats["hits"] += 1
                        self._stats["cost_saved"] += 0.009

                        logger.debug(f"Cache hit (fuzzy): {cached_key}")

                        # Return with note that it's a similar (not exact) match
                        result = entry.result.copy()
                        result["_cache_fuzzy"] = True
                        return result

        self._stats["misses"] += 1
        return None

    def set(
        self,
        hero_cards: List[str],
        board: List[str],
        villain_type: str,
        result: Dict[str, Any],
        action_facing: str = ""
    ):
        """
        Cache an analysis result.

        Args:
            hero_cards: Hero's hole cards
            board: Community cards
            villain_type: Villain archetype
            result: Analysis result to cache
            action_facing: Action hero was facing
        """
        key = self._make_key(hero_cards, board, villain_type, action_facing)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats["evictions"] += 1

        entry = CacheEntry(
            key=key,
            result=result,
            timestamp=time.time(),
            hero_cards=" ".join(hero_cards),
            board=" ".join(board),
            villain_type=villain_type
        )

        self._cache[key] = entry
        logger.debug(f"Cached: {key}")

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "cost_saved": self._stats["cost_saved"]
        }

    def format_stats(self) -> str:
        """Format stats for display."""
        stats = self.get_stats()
        return (
            f"Cache: {stats['size']}/{stats['max_size']} entries\n"
            f"Hit rate: {stats['hit_rate']:.1%}\n"
            f"Cost saved: ${stats['cost_saved']:.2f}"
        )


# Test
def test_cache():
    """Test poker cache."""
    print("=== Poker Cache Test ===\n")

    cache = PokerCache(max_size=100, ttl_seconds=60)

    # Cache a result
    result = {
        "action": "RAISE",
        "sizing": "3/4 pot",
        "equity": 0.65,
        "reasoning": "Value bet vs calling station"
    }

    cache.set(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        villain_type="calling_station",
        result=result
    )

    # Exact match
    hit1 = cache.get(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        villain_type="calling_station"
    )
    print(f"Exact match: {hit1 is not None}")

    # Different order (should still hit)
    hit2 = cache.get(
        hero_cards=["Kc", "Ah"],
        board=["3d", "7h", "9s"],
        villain_type="calling_station"
    )
    print(f"Reordered match: {hit2 is not None}")

    # Fuzzy match (similar hand, same texture)
    hit3 = cache.get(
        hero_cards=["As", "Kd"],  # Different suits
        board=["Ts", "6h", "2d"],  # Different low cards
        villain_type="calling_station"
    )
    print(f"Fuzzy match: {hit3 is not None}")

    # Miss (different villain)
    miss = cache.get(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        villain_type="maniac"
    )
    print(f"Different villain (miss): {miss is None}")

    print()
    print(cache.format_stats())


if __name__ == "__main__":
    test_cache()
