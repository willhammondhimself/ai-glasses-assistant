"""Voice Notes tool - capture and search notes by voice."""
import logging
import re
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class NotesVoiceTool(VoiceTool):
    """Voice-controlled note taking and search."""

    name = "notes"
    description = "Take voice notes, search notes, recall information"

    keywords = [
        r"\bnote\b",
        r"\bremember\s+(that|this)?\b",
        r"\bwrite\s+down\b",
        r"\bsave\s+(this|that|idea|thought)\b",
        r"\bjot\s+down\b",
        r"\bwhat\s+did\s+i\s+(note|say|write)\b",
        r"\bsearch\s+notes?\b",
        r"\bmy\s+notes?\b",
        r"\brecent\s+notes?\b",
        r"\btoday'?s?\s+notes?\b",
        r"\bfind\s+note\b",
        r"\brecall\b",
        r"\bdelete\s+note\b",
    ]

    priority = 8

    def __init__(self):
        self._notes_service = None

    def _get_service(self):
        """Get notes service (lazy load)."""
        if self._notes_service is None:
            from backend.services.notes_service import get_notes_service
            self._notes_service = get_notes_service()
        return self._notes_service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute notes command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with notes info
        """
        query_lower = query.lower()
        service = self._get_service()

        try:
            # Create a note
            if self._is_create_request(query_lower):
                content = self._extract_note_content(query)
                return await self._handle_create(service, content, kwargs.get("context"))

            # Search notes
            if self._is_search_request(query_lower):
                search_query = self._extract_search_query(query_lower)
                return await self._handle_search(service, search_query)

            # Get recent notes
            if self._is_recent_request(query_lower):
                return await self._handle_recent(service)

            # Get today's notes
            if self._is_today_request(query_lower):
                return await self._handle_today(service)

            # Delete note
            if self._is_delete_request(query_lower):
                note_id = self._extract_note_id(query_lower)
                return await self._handle_delete(service, note_id)

            # Default: treat as creating a note
            content = self._extract_note_content(query)
            if content:
                return await self._handle_create(service, content, kwargs.get("context"))

            return VoiceToolResult(
                success=False,
                message="I'm not sure what you want to do with notes. Try 'note: [content]' or 'search notes for [topic]'.",
                data={"error": "unclear_intent"}
            )

        except Exception as e:
            logger.error(f"Notes tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble with that notes command.",
                data={"error": str(e)}
            )

    def _is_create_request(self, query: str) -> bool:
        patterns = [
            "note:", "note that", "remember that", "remember this",
            "write down", "save this", "save that", "jot down",
            "make a note", "take a note", "add note"
        ]
        return any(p in query for p in patterns)

    def _is_search_request(self, query: str) -> bool:
        patterns = [
            "search notes", "find note", "what did i note",
            "what did i say about", "what did i write",
            "notes about", "notes on", "recall"
        ]
        return any(p in query for p in patterns)

    def _is_recent_request(self, query: str) -> bool:
        patterns = [
            "recent notes", "my notes", "latest notes",
            "last notes", "show notes"
        ]
        return any(p in query for p in patterns) and "today" not in query

    def _is_today_request(self, query: str) -> bool:
        patterns = ["today's notes", "todays notes", "notes from today", "notes today"]
        return any(p in query for p in patterns)

    def _is_delete_request(self, query: str) -> bool:
        patterns = ["delete note", "remove note", "delete the note"]
        return any(p in query for p in patterns)

    def _extract_note_content(self, query: str) -> str:
        """Extract note content from query."""
        # Remove common prefixes
        prefixes = [
            r"^note[:\s]+",
            r"^note that\s+",
            r"^remember that\s+",
            r"^remember this[:\s]*",
            r"^write down[:\s]*",
            r"^save this[:\s]*",
            r"^save that[:\s]*",
            r"^jot down[:\s]*",
            r"^make a note[:\s]*",
            r"^take a note[:\s]*",
            r"^add note[:\s]*",
        ]

        content = query
        for prefix in prefixes:
            content = re.sub(prefix, "", content, flags=re.IGNORECASE)

        return content.strip()

    def _extract_search_query(self, query: str) -> str:
        """Extract search query from request."""
        # Remove search prefixes
        patterns = [
            r"search\s+notes?\s+(for\s+)?",
            r"find\s+notes?\s+(about\s+)?",
            r"what\s+did\s+i\s+note\s+about\s+",
            r"what\s+did\s+i\s+say\s+about\s+",
            r"what\s+did\s+i\s+write\s+about\s+",
            r"notes?\s+about\s+",
            r"notes?\s+on\s+",
            r"recall\s+",
        ]

        search_query = query
        for pattern in patterns:
            search_query = re.sub(pattern, "", search_query, flags=re.IGNORECASE)

        return search_query.strip() or query

    def _extract_note_id(self, query: str) -> str:
        """Extract note ID from query."""
        # Look for note ID pattern
        match = re.search(r"note\s+([a-f0-9]{8})", query)
        if match:
            return match.group(1)
        return ""

    async def _handle_create(
        self,
        service,
        content: str,
        context: str = None
    ) -> VoiceToolResult:
        """Handle note creation."""
        if not content:
            return VoiceToolResult(
                success=False,
                message="What would you like me to note?",
                data={"needs_content": True}
            )

        note = await service.create_note(content, context=context)

        # Build response
        tag_str = f" Tagged as {', '.join(note.tags)}." if note.tags else ""
        message = f"Got it. Noted.{tag_str}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "created",
                "note": note.to_dict()
            }
        )

    async def _handle_search(self, service, search_query: str) -> VoiceToolResult:
        """Handle note search."""
        if not search_query:
            return VoiceToolResult(
                success=False,
                message="What would you like me to search for?",
                data={"needs_query": True}
            )

        # Try FTS first, fall back to simple search
        try:
            notes = await service.search(search_query, limit=5)
        except Exception:
            notes = await service.search_simple(search_query, limit=5)

        if not notes:
            return VoiceToolResult(
                success=True,
                message=f"I didn't find any notes about '{search_query}'.",
                data={"action": "search", "query": search_query, "results": []}
            )

        # Build response
        if len(notes) == 1:
            message = f"Found one note: {notes[0].to_voice_summary()}"
        else:
            summaries = [n.to_voice_summary() for n in notes[:3]]
            message = f"Found {len(notes)} notes. Most relevant: {summaries[0]}"
            if len(summaries) > 1:
                message += f" Also: {summaries[1]}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "search",
                "query": search_query,
                "results": [n.to_dict() for n in notes]
            }
        )

    async def _handle_recent(self, service) -> VoiceToolResult:
        """Handle recent notes request."""
        notes = await service.get_recent(limit=5)

        if not notes:
            return VoiceToolResult(
                success=True,
                message="You don't have any notes yet. Say 'note' followed by what you want to remember.",
                data={"action": "recent", "results": []}
            )

        # Build response
        if len(notes) == 1:
            message = f"You have one recent note: {notes[0].to_voice_summary()}"
        else:
            message = f"Your {len(notes)} most recent notes. Latest: {notes[0].to_voice_summary()}"
            if len(notes) > 1:
                message += f" Before that: {notes[1].to_voice_summary()}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "recent",
                "results": [n.to_dict() for n in notes]
            }
        )

    async def _handle_today(self, service) -> VoiceToolResult:
        """Handle today's notes request."""
        notes = await service.get_today()

        if not notes:
            return VoiceToolResult(
                success=True,
                message="You haven't made any notes today.",
                data={"action": "today", "results": []}
            )

        # Build response
        if len(notes) == 1:
            message = f"You made one note today: {notes[0].to_voice_summary()}"
        else:
            message = f"You made {len(notes)} notes today. Most recent: {notes[0].to_voice_summary()}"

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "action": "today",
                "results": [n.to_dict() for n in notes]
            }
        )

    async def _handle_delete(self, service, note_id: str) -> VoiceToolResult:
        """Handle note deletion."""
        if not note_id:
            return VoiceToolResult(
                success=False,
                message="Which note do you want to delete? Please specify the note ID.",
                data={"needs_id": True}
            )

        deleted = await service.delete_note(note_id)

        if deleted:
            return VoiceToolResult(
                success=True,
                message=f"Deleted note {note_id}.",
                data={"action": "deleted", "note_id": note_id}
            )
        else:
            return VoiceToolResult(
                success=False,
                message=f"Couldn't find note {note_id}.",
                data={"error": "not_found", "note_id": note_id}
            )
