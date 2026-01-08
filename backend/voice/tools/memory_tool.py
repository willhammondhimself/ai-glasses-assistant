"""Memory voice tool - wraps ContextMemory + RAGStore for voice interaction."""
import logging
from typing import Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class MemoryVoiceTool(VoiceTool):
    """Voice tool for memory operations.

    Wraps ContextMemory (facts, preferences, entities) and RAGStore (document search)
    for natural voice interaction.
    """

    name = "memory"
    description = "Remember facts, recall information, and search documents"

    keywords = [
        # Remember triggers
        r"\bremember\s+that\b",
        r"\bkeep\s+in\s+mind\b",
        r"\bnote\s+that\b",
        r"\bdon'?t\s+forget\b",
        r"\bfor\s+future\s+reference\b",
        # Recall triggers
        r"\bwhat\s+do\s+you\s+know\s+about\b",
        r"\bwhat\s+did\s+i\s+(say|tell\s+you)\s+about\b",
        r"\bremind\s+me\s+about\b",
        r"\bwhat('s|\s+is)\s+the\s+deal\s+with\b",
        r"\btell\s+me\s+about\b",
        r"\bdo\s+you\s+remember\b",
        # Document search triggers
        r"\bsearch\s+(my\s+)?(documents?|files?|notes?)\s+(for|about)\b",
        r"\bfind\s+(in\s+)?(my\s+)?(documents?|files?|notes?)\b",
        r"\blook\s+up\s+(in\s+)?my\b",
        r"\bwhat('s|\s+is)\s+in\s+my\s+(documents?|files?|notes?)\b",
        # Memory stats
        r"\b(how\s+many|what)\s+memories?\s+(do\s+you\s+have|are\s+stored)\b",
        r"\bmemory\s+stats?\b",
        # Forget triggers
        r"\bforget\s+(about\s+)?that\b",
        r"\bdelete\s+(that\s+)?memory\b",
        r"\bclear\s+(my\s+)?memories?\b",
    ]

    priority = 8  # Medium-high priority for memory queries

    def __init__(self):
        self._context_memory = None
        self._rag_store = None

    def _get_context_memory(self):
        """Lazy load context memory."""
        if self._context_memory is None:
            from phone_client.core.context_memory import ContextMemory
            config = {
                "context_memory": {
                    "max_memories": 10000,
                    "auto_extract_entities": True,
                    "relevance_threshold": 0.2
                }
            }
            self._context_memory = ContextMemory(config, storage_dir="./memory")
        return self._context_memory

    def _get_rag_store(self):
        """Lazy load RAG store."""
        if self._rag_store is None:
            from backend.rag.store import get_rag_store
            self._rag_store = get_rag_store()
        return self._rag_store

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute memory operation based on voice query.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with voice-friendly response
        """
        try:
            action, params = self._parse_query(query)
            logger.info(f"Memory action: {action}, params: {params}")

            if action == "remember":
                return await self._handle_remember(params.get("content", ""))
            elif action == "recall":
                return await self._handle_recall(params.get("query", ""))
            elif action == "search_docs":
                return await self._handle_doc_search(params.get("query", ""))
            elif action == "stats":
                return await self._handle_stats()
            elif action == "forget":
                return await self._handle_forget(params.get("query", ""))
            else:
                return VoiceToolResult(
                    success=False,
                    message="I'm not sure what you want me to remember or recall."
                )

        except Exception as e:
            logger.error(f"Memory voice tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with that memory operation."
            )

    def _parse_query(self, query: str) -> Tuple[str, dict]:
        """Parse voice query to determine action and parameters.

        Args:
            query: The user's voice query

        Returns:
            Tuple of (action, params dict)
        """
        query_lower = query.lower()

        # Check for remember operations
        remember_triggers = [
            "remember that", "keep in mind", "note that",
            "don't forget", "dont forget", "for future reference"
        ]
        for trigger in remember_triggers:
            if trigger in query_lower:
                idx = query_lower.find(trigger)
                content = query[idx + len(trigger):].strip()
                # Clean up leading words
                content = content.lstrip(", ")
                return "remember", {"content": content}

        # Check for document search (before generic recall)
        doc_patterns = [
            ("search my documents for", "query"),
            ("search my files for", "query"),
            ("search my notes for", "query"),
            ("find in my documents", "query"),
            ("find in my files", "query"),
            ("find in my notes", "query"),
            ("look up in my", "query"),
            ("what's in my documents about", "query"),
            ("what is in my documents about", "query"),
        ]
        for pattern, param_name in doc_patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern)
                content = query[idx + len(pattern):].strip()
                return "search_docs", {param_name: content}

        # Check for stats
        if any(p in query_lower for p in ["memory stats", "how many memories", "what memories"]):
            return "stats", {}

        # Check for forget
        forget_triggers = ["forget about", "forget that", "delete memory", "clear memories"]
        for trigger in forget_triggers:
            if trigger in query_lower:
                idx = query_lower.find(trigger)
                content = query[idx + len(trigger):].strip()
                return "forget", {"query": content}

        # Check for recall operations (catch-all for "what do you know", "tell me about", etc.)
        recall_triggers = [
            "what do you know about", "what did i say about", "what did i tell you about",
            "remind me about", "what's the deal with", "what is the deal with",
            "tell me about", "do you remember"
        ]
        for trigger in recall_triggers:
            if trigger in query_lower:
                idx = query_lower.find(trigger)
                content = query[idx + len(trigger):].strip()
                # Clean up
                content = content.rstrip("?")
                return "recall", {"query": content}

        # Default: treat as recall query
        return "recall", {"query": query}

    async def _handle_remember(self, content: str) -> VoiceToolResult:
        """Store a new memory."""
        if not content:
            return VoiceToolResult(
                success=False,
                message="What would you like me to remember?"
            )

        cm = self._get_context_memory()
        memory = cm.remember(content)

        response = f"Got it. I'll remember that {content[:50]}{'...' if len(content) > 50 else ''}."
        if memory.entity:
            response = f"Got it, noted about {memory.entity}."

        return VoiceToolResult(
            success=True,
            message=response,
            data={
                "memory_id": memory.id,
                "type": memory.type.value,
                "entity": memory.entity,
                "keywords": memory.keywords
            }
        )

    async def _handle_recall(self, query: str) -> VoiceToolResult:
        """Recall relevant memories."""
        if not query:
            return VoiceToolResult(
                success=False,
                message="What would you like me to recall?"
            )

        cm = self._get_context_memory()

        # Try entity lookup first
        entity = cm.get_entity(query)
        if entity:
            memories = cm.recall_about_entity(query, limit=3)
        else:
            memories = cm.recall(query, limit=3)

        if not memories:
            return VoiceToolResult(
                success=True,
                message=f"I don't have any memories about {query}.",
                data={"count": 0}
            )

        # Format for voice
        if len(memories) == 1:
            return VoiceToolResult(
                success=True,
                message=f"I remember: {memories[0].content}",
                data={"count": 1, "memories": [m.content for m in memories]}
            )

        # Multiple memories - summarize
        summaries = [m.content[:60] for m in memories[:3]]
        response = f"I have {len(memories)} memories about {query}. "
        response += "; ".join(summaries) + "."

        return VoiceToolResult(
            success=True,
            message=response,
            data={"count": len(memories), "memories": [m.content for m in memories]}
        )

    async def _handle_doc_search(self, query: str) -> VoiceToolResult:
        """Search documents in RAG store."""
        if not query:
            return VoiceToolResult(
                success=False,
                message="What would you like me to search for?"
            )

        try:
            rag = self._get_rag_store()
            results = await rag.query(query, top_k=3)

            if not results:
                return VoiceToolResult(
                    success=True,
                    message=f"I didn't find any documents about {query}.",
                    data={"count": 0}
                )

            # Format for voice
            top_result = results[0]
            filename = top_result.get("metadata", {}).get("filename", "a document")
            snippet = top_result.get("content", "")[:100]

            response = f"Found {len(results)} document{'s' if len(results) > 1 else ''}. "
            response += f"Best match is {filename}: {snippet}..."

            return VoiceToolResult(
                success=True,
                message=response,
                data={
                    "count": len(results),
                    "results": [
                        {
                            "filename": r.get("metadata", {}).get("filename"),
                            "score": r.get("score"),
                            "snippet": r.get("content", "")[:200]
                        }
                        for r in results
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Document search error: {e}")
            return VoiceToolResult(
                success=False,
                message="I couldn't search your documents right now."
            )

    async def _handle_stats(self) -> VoiceToolResult:
        """Get memory statistics."""
        cm = self._get_context_memory()
        stats = cm.get_stats()

        total = stats.get("total_memories", 0)
        entities = stats.get("total_entities", 0)

        response = f"I have {total} memories about {entities} entities."

        # Add RAG stats if available
        try:
            rag = self._get_rag_store()
            rag_stats = rag.get_stats()
            docs = rag_stats.get("total_documents", 0)
            if docs > 0:
                response += f" Plus {docs} documents in my search index."
        except Exception:
            pass

        return VoiceToolResult(
            success=True,
            message=response,
            data=stats
        )

    async def _handle_forget(self, query: str) -> VoiceToolResult:
        """Forget memories (requires confirmation for safety)."""
        if not query:
            return VoiceToolResult(
                success=False,
                message="What would you like me to forget?"
            )

        cm = self._get_context_memory()
        memories = cm.recall(query, limit=1)

        if not memories:
            return VoiceToolResult(
                success=True,
                message=f"I don't have any memories about {query} to forget."
            )

        # For safety, just identify what would be forgotten
        # Actual deletion could require confirmation
        memory = memories[0]
        return VoiceToolResult(
            success=True,
            message=f"Found memory: '{memory.content[:50]}...' Say 'confirm forget' to delete it.",
            data={"memory_id": memory.id, "content": memory.content}
        )
