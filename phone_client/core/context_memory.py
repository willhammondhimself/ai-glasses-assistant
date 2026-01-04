"""
Context Memory - Long-term memory and context recall system.
Remember facts, preferences, and context for intelligent assistance.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories."""
    FACT = "fact"                    # General facts (e.g., "Sarah's birthday is March 15")
    PREFERENCE = "preference"        # Preferences (e.g., "Sarah prefers email over calls")
    RELATIONSHIP = "relationship"    # People/relationships
    LOCATION = "location"            # Places and locations
    EVENT = "event"                  # Past events
    SKILL = "skill"                  # Learned skills/patterns
    CONTEXT = "context"              # Contextual information
    HABIT = "habit"                  # User habits and patterns


class MemoryImportance(Enum):
    """Importance levels for memories."""
    TRIVIAL = 1       # May forget
    LOW = 2           # Keep but deprioritize
    NORMAL = 3        # Standard retention
    HIGH = 4          # Important, surface often
    CRITICAL = 5      # Never forget, always consider


@dataclass
class Memory:
    """Single memory item."""
    id: str
    type: MemoryType
    content: str                               # The actual memory content
    keywords: List[str]                        # Searchable keywords
    importance: MemoryImportance = MemoryImportance.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    related_memories: List[str] = field(default_factory=list)  # IDs of related memories
    source: str = "user"                       # Where the memory came from
    expires_at: Optional[datetime] = None      # Optional expiration
    entity: Optional[str] = None               # Associated entity (person, place, etc.)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "keywords": self.keywords,
            "importance": self.importance.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "context": self.context,
            "related_memories": self.related_memories,
            "source": self.source,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "entity": self.entity
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            keywords=data.get("keywords", []),
            importance=MemoryImportance(data.get("importance", 3)),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", data["created_at"])),
            access_count=data.get("access_count", 0),
            context=data.get("context", {}),
            related_memories=data.get("related_memories", []),
            source=data.get("source", "user"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            entity=data.get("entity")
        )

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def relevance_score(self, query_keywords: List[str]) -> float:
        """Calculate relevance score for a query."""
        if not query_keywords:
            return 0.0

        # Keyword matching
        keyword_matches = sum(
            1 for k in query_keywords
            if any(k.lower() in m.lower() for m in self.keywords)
        )
        keyword_score = keyword_matches / len(query_keywords)

        # Content matching
        content_lower = self.content.lower()
        content_matches = sum(1 for k in query_keywords if k.lower() in content_lower)
        content_score = content_matches / len(query_keywords)

        # Recency factor (more recent = higher score)
        days_old = (datetime.now() - self.last_accessed).days
        recency_score = max(0, 1 - (days_old / 365))  # Linear decay over 1 year

        # Access frequency factor
        frequency_score = min(1.0, self.access_count / 10)

        # Importance factor
        importance_score = self.importance.value / 5

        # Combined score
        score = (
            keyword_score * 0.35 +
            content_score * 0.25 +
            recency_score * 0.15 +
            frequency_score * 0.10 +
            importance_score * 0.15
        )

        return score


@dataclass
class Entity:
    """Represents a person, place, or thing in the memory system."""
    id: str
    name: str
    type: str  # person, place, organization, thing
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    memory_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "memory_ids": self.memory_ids,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            memory_ids=data.get("memory_ids", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )


class ContextMemory:
    """
    Long-term context memory system for WHAM.

    Stores and retrieves facts, preferences, relationships, and contextual
    information for intelligent assistance.
    """

    # Common memory trigger phrases
    REMEMBER_TRIGGERS = [
        "remember that",
        "keep in mind",
        "note that",
        "don't forget",
        "remember",
        "for future reference",
    ]

    RECALL_TRIGGERS = [
        "what do you know about",
        "what did i say about",
        "remind me about",
        "what's the deal with",
        "tell me about",
    ]

    def __init__(self, config: dict, storage_dir: str = "./memory"):
        """
        Initialize Context Memory.

        Args:
            config: Configuration dictionary
            storage_dir: Directory to store memory data
        """
        self.config = config
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory stores
        self._memories: Dict[str, Memory] = {}
        self._entities: Dict[str, Entity] = {}
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword -> memory IDs

        # Load settings
        memory_config = config.get("context_memory", {})
        self.max_memories = memory_config.get("max_memories", 10000)
        self.auto_extract_entities = memory_config.get("auto_extract_entities", True)
        self.relevance_threshold = memory_config.get("relevance_threshold", 0.3)
        self.cleanup_interval_days = memory_config.get("cleanup_interval_days", 30)

        # Load data
        self._load_data()

        logger.info(f"ContextMemory initialized ({len(self._memories)} memories, {len(self._entities)} entities)")

    def _load_data(self):
        """Load memories and entities from storage."""
        memories_file = self.storage_dir / "memories.json"
        entities_file = self.storage_dir / "entities.json"

        if memories_file.exists():
            try:
                with open(memories_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        memory = Memory.from_dict(item)
                        if not memory.is_expired():
                            self._memories[memory.id] = memory
                            self._index_memory(memory)
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")

        if entities_file.exists():
            try:
                with open(entities_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        entity = Entity.from_dict(item)
                        self._entities[entity.id] = entity
            except Exception as e:
                logger.error(f"Failed to load entities: {e}")

    def _save_data(self):
        """Save memories and entities to storage."""
        memories_file = self.storage_dir / "memories.json"
        entities_file = self.storage_dir / "entities.json"

        try:
            with open(memories_file, "w") as f:
                json.dump([m.to_dict() for m in self._memories.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

        try:
            with open(entities_file, "w") as f:
                json.dump([e.to_dict() for e in self._entities.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save entities: {e}")

    def _index_memory(self, memory: Memory):
        """Index memory by keywords."""
        for keyword in memory.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._keyword_index:
                self._keyword_index[keyword_lower] = set()
            self._keyword_index[keyword_lower].add(memory.id)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction
        # Remove common words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "about",
            "like", "through", "after", "over", "between", "out", "against",
            "during", "without", "before", "under", "around", "among", "that",
            "this", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "what", "which", "who", "whom", "whose", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "also", "now", "here", "there",
            "then", "once", "always", "never", "really", "remember", "that"
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Also extract proper nouns (capitalized words in original)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', content)
        keywords.extend([n.lower() for n in proper_nouns])

        return list(set(keywords))

    def _detect_memory_type(self, content: str) -> MemoryType:
        """Detect memory type from content."""
        content_lower = content.lower()

        # Preference indicators
        if any(w in content_lower for w in ["prefers", "likes", "dislikes", "wants", "favorite"]):
            return MemoryType.PREFERENCE

        # Relationship indicators
        if any(w in content_lower for w in ["friend", "colleague", "boss", "partner", "family"]):
            return MemoryType.RELATIONSHIP

        # Location indicators
        if any(w in content_lower for w in ["lives", "located", "address", "place", "restaurant", "store"]):
            return MemoryType.LOCATION

        # Event indicators
        if any(w in content_lower for w in ["happened", "occurred", "event", "meeting", "party"]):
            return MemoryType.EVENT

        # Habit indicators
        if any(w in content_lower for w in ["usually", "always", "habit", "routine", "every"]):
            return MemoryType.HABIT

        return MemoryType.FACT

    def _extract_entity(self, content: str) -> Optional[str]:
        """Extract primary entity from content."""
        # Look for proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        if proper_nouns:
            return proper_nouns[0]
        return None

    def remember(
        self,
        content: str,
        memory_type: MemoryType = None,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        keywords: List[str] = None,
        entity: str = None,
        context: Dict[str, Any] = None,
        expires_in_days: int = None
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The memory content
            memory_type: Type of memory (auto-detected if not provided)
            importance: Importance level
            keywords: Keywords for search (auto-extracted if not provided)
            entity: Associated entity
            context: Additional context
            expires_in_days: Days until memory expires

        Returns:
            Created Memory
        """
        import uuid

        # Auto-detect type if not provided
        if memory_type is None:
            memory_type = self._detect_memory_type(content)

        # Auto-extract keywords
        if keywords is None:
            keywords = self._extract_keywords(content)

        # Auto-extract entity
        if entity is None and self.auto_extract_entities:
            entity = self._extract_entity(content)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        memory = Memory(
            id=str(uuid.uuid4())[:8],
            type=memory_type,
            content=content,
            keywords=keywords,
            importance=importance,
            context=context or {},
            entity=entity,
            expires_at=expires_at
        )

        # Store memory
        self._memories[memory.id] = memory
        self._index_memory(memory)

        # Update entity if exists
        if entity:
            self._update_entity_memory(entity, memory.id)

        self._save_data()
        logger.info(f"Stored memory: [{memory.type.value}] {content[:50]}...")

        return memory

    def _update_entity_memory(self, entity_name: str, memory_id: str):
        """Update or create entity with new memory reference."""
        entity_id = entity_name.lower().replace(" ", "_")

        if entity_id in self._entities:
            entity = self._entities[entity_id]
            if memory_id not in entity.memory_ids:
                entity.memory_ids.append(memory_id)
        else:
            entity = Entity(
                id=entity_id,
                name=entity_name,
                type="unknown",
                memory_ids=[memory_id]
            )
            self._entities[entity_id] = entity

    def recall(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        entity: str = None,
        limit: int = 5,
        min_relevance: float = None
    ) -> List[Memory]:
        """
        Recall relevant memories.

        Args:
            query: Search query
            memory_types: Filter by memory types
            entity: Filter by entity
            limit: Maximum results
            min_relevance: Minimum relevance score

        Returns:
            List of relevant memories
        """
        if min_relevance is None:
            min_relevance = self.relevance_threshold

        query_keywords = self._extract_keywords(query)
        candidates = []

        # Get candidate memories
        for memory in self._memories.values():
            # Skip expired
            if memory.is_expired():
                continue

            # Filter by type
            if memory_types and memory.type not in memory_types:
                continue

            # Filter by entity
            if entity and memory.entity and entity.lower() not in memory.entity.lower():
                continue

            # Calculate relevance
            score = memory.relevance_score(query_keywords)
            if score >= min_relevance:
                candidates.append((memory, score))

        # Sort by relevance
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Update access stats and return
        results = []
        for memory, score in candidates[:limit]:
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            results.append(memory)

        self._save_data()
        return results

    def recall_about_entity(self, entity_name: str, limit: int = 10) -> List[Memory]:
        """Recall all memories about an entity."""
        entity_id = entity_name.lower().replace(" ", "_")

        if entity_id not in self._entities:
            return []

        entity = self._entities[entity_id]
        memories = []

        for memory_id in entity.memory_ids:
            if memory_id in self._memories:
                memory = self._memories[memory_id]
                if not memory.is_expired():
                    memories.append(memory)

        return memories[:limit]

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        entity_id = name.lower().replace(" ", "_")
        return self._entities.get(entity_id)

    def update_entity_attribute(self, name: str, attribute: str, value: Any):
        """Update an entity's attribute."""
        entity_id = name.lower().replace(" ", "_")

        if entity_id in self._entities:
            self._entities[entity_id].attributes[attribute] = value
            self._save_data()
            logger.debug(f"Updated {name}.{attribute} = {value}")

    def forget(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._memories:
            memory = self._memories[memory_id]

            # Remove from entity
            if memory.entity:
                entity_id = memory.entity.lower().replace(" ", "_")
                if entity_id in self._entities:
                    entity = self._entities[entity_id]
                    if memory_id in entity.memory_ids:
                        entity.memory_ids.remove(memory_id)

            # Remove from index
            for keyword in memory.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in self._keyword_index:
                    self._keyword_index[keyword_lower].discard(memory_id)

            # Delete memory
            del self._memories[memory_id]
            self._save_data()
            logger.info(f"Forgot memory: {memory_id}")
            return True

        return False

    def cleanup_expired(self) -> int:
        """Remove expired memories."""
        expired_ids = [
            mid for mid, m in self._memories.items()
            if m.is_expired()
        ]

        for memory_id in expired_ids:
            self.forget(memory_id)

        logger.info(f"Cleaned up {len(expired_ids)} expired memories")
        return len(expired_ids)

    def parse_remember_command(self, text: str) -> Optional[Memory]:
        """
        Parse a 'remember' command from user input.

        Args:
            text: User input text

        Returns:
            Created Memory if command detected, None otherwise
        """
        text_lower = text.lower()

        for trigger in self.REMEMBER_TRIGGERS:
            if trigger in text_lower:
                # Extract the content after the trigger
                idx = text_lower.find(trigger)
                content = text[idx + len(trigger):].strip()
                if content:
                    return self.remember(content)

        return None

    def parse_recall_command(self, text: str) -> List[Memory]:
        """
        Parse a 'recall' command from user input.

        Args:
            text: User input text

        Returns:
            List of recalled memories
        """
        text_lower = text.lower()

        for trigger in self.RECALL_TRIGGERS:
            if trigger in text_lower:
                # Extract the query after the trigger
                idx = text_lower.find(trigger)
                query = text[idx + len(trigger):].strip()
                if query:
                    return self.recall(query)

        return []

    def get_context_for_topic(self, topic: str, max_items: int = 3) -> List[str]:
        """
        Get relevant context for a topic (for LLM prompting).

        Args:
            topic: Topic to find context for
            max_items: Maximum context items

        Returns:
            List of context strings
        """
        memories = self.recall(topic, limit=max_items)
        return [m.content for m in memories]

    def get_stats(self) -> dict:
        """Get memory statistics."""
        by_type = {}
        by_importance = {}

        for memory in self._memories.values():
            # Count by type
            type_name = memory.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by importance
            imp_name = memory.importance.name
            by_importance[imp_name] = by_importance.get(imp_name, 0) + 1

        return {
            "total_memories": len(self._memories),
            "total_entities": len(self._entities),
            "by_type": by_type,
            "by_importance": by_importance,
            "indexed_keywords": len(self._keyword_index)
        }

    def format_memories_for_display(self, memories: List[Memory]) -> List[str]:
        """Format memories for HUD display."""
        lines = []

        for i, memory in enumerate(memories, 1):
            icon = "💭"
            if memory.type == MemoryType.PREFERENCE:
                icon = "❤️"
            elif memory.type == MemoryType.RELATIONSHIP:
                icon = "👤"
            elif memory.type == MemoryType.LOCATION:
                icon = "📍"

            lines.append(f"{icon} {memory.content[:60]}{'...' if len(memory.content) > 60 else ''}")

        return lines


# Test
def test_context_memory():
    """Test context memory functionality."""
    import tempfile

    print("=== Context Memory Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "context_memory": {
                "max_memories": 1000,
                "auto_extract_entities": True,
                "relevance_threshold": 0.2
            }
        }

        cm = ContextMemory(config, storage_dir=tmpdir)

        # Store memories
        print("1. Storing memories:")
        m1 = cm.remember("Sarah prefers email over phone calls")
        print(f"   [{m1.type.value}] {m1.content}")
        print(f"   Keywords: {m1.keywords}")
        print(f"   Entity: {m1.entity}")
        print()

        m2 = cm.remember("John's birthday is on March 15th")
        print(f"   [{m2.type.value}] {m2.content}")
        print()

        m3 = cm.remember("The team usually meets on Tuesdays at 10am")
        print(f"   [{m3.type.value}] {m3.content}")
        print()

        # Recall
        print("2. Recalling 'Sarah':")
        results = cm.recall("Sarah")
        for r in results:
            print(f"   {r.content}")
        print()

        print("3. Recalling 'birthday':")
        results = cm.recall("birthday")
        for r in results:
            print(f"   {r.content}")
        print()

        # Entity lookup
        print("4. Entity lookup 'Sarah':")
        entity = cm.get_entity("Sarah")
        if entity:
            print(f"   Name: {entity.name}")
            print(f"   Memories: {len(entity.memory_ids)}")
        print()

        # Parse commands
        print("5. Parse remember command:")
        m4 = cm.parse_remember_command("Remember that Mike is allergic to peanuts")
        if m4:
            print(f"   Created: {m4.content}")
        print()

        print("6. Parse recall command:")
        results = cm.parse_recall_command("What do you know about Mike?")
        for r in results:
            print(f"   {r.content}")
        print()

        # Stats
        print("7. Stats:")
        stats = cm.get_stats()
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Total entities: {stats['total_entities']}")
        print(f"   By type: {stats['by_type']}")


if __name__ == "__main__":
    test_context_memory()
