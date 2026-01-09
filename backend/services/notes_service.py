"""Voice Notes Service - SQLite-backed notes with full-text search."""
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """A voice note with metadata."""
    id: str
    content: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    context: Optional[str] = None

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        # Truncate long notes for voice
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content

        if self.tags:
            return f"{content_preview} Tagged: {', '.join(self.tags[:3])}"
        return content_preview

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "category": self.category,
            "context": self.context
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Note":
        """Create Note from database row."""
        return cls(
            id=row[0],
            content=row[1],
            created_at=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
            updated_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
            tags=json.loads(row[4]) if row[4] else [],
            category=row[5],
            context=row[6]
        )


class NotesService:
    """SQLite-backed voice notes with FTS5 full-text search."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize notes service.

        Args:
            db_path: Path to SQLite database. Defaults to ./data/notes.db
        """
        self.db_path = db_path or os.getenv("NOTES_DB_PATH", "./data/notes.db")
        self._ensure_data_dir()
        self._init_db()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize database schema with FTS5."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                category TEXT,
                context TEXT
            )
        """)

        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                id,
                content,
                tags,
                category,
                content='notes',
                content_rowid='rowid'
            )
        """)

        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, id, content, tags, category)
                VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags, NEW.category);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, content, tags, category)
                VALUES('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags, OLD.category);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, id, content, tags, category)
                VALUES('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags, OLD.category);
                INSERT INTO notes_fts(rowid, id, content, tags, category)
                VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags, NEW.category);
            END
        """)

        conn.commit()
        conn.close()
        logger.info(f"Notes database initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    async def create_note(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        context: Optional[str] = None
    ) -> Note:
        """Create a new note.

        Args:
            content: Note content
            tags: Optional list of tags
            category: Optional category (idea, todo, reminder, etc.)
            context: Optional context (where/when created)

        Returns:
            Created Note
        """
        note_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        # Auto-generate tags if not provided
        if tags is None:
            tags = self._auto_tag(content)

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO notes (id, content, created_at, updated_at, tags, category, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            note_id,
            content,
            now.isoformat(),
            now.isoformat(),
            json.dumps(tags),
            category,
            context
        ))

        conn.commit()
        conn.close()

        logger.info(f"Created note {note_id}: {content[:50]}...")

        return Note(
            id=note_id,
            content=content,
            created_at=now,
            updated_at=now,
            tags=tags,
            category=category,
            context=context
        )

    async def get_note(self, note_id: str) -> Optional[Note]:
        """Get a note by ID.

        Args:
            note_id: Note ID

        Returns:
            Note or None if not found
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes WHERE id = ?
        """, (note_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return Note.from_row(row)
        return None

    async def update_note(
        self,
        note_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> Optional[Note]:
        """Update an existing note.

        Args:
            note_id: Note ID
            content: New content (optional)
            tags: New tags (optional)
            category: New category (optional)

        Returns:
            Updated Note or None if not found
        """
        existing = await self.get_note(note_id)
        if not existing:
            return None

        conn = self._get_conn()
        cursor = conn.cursor()

        # Update only provided fields
        new_content = content if content is not None else existing.content
        new_tags = tags if tags is not None else existing.tags
        new_category = category if category is not None else existing.category
        now = datetime.now()

        cursor.execute("""
            UPDATE notes
            SET content = ?, tags = ?, category = ?, updated_at = ?
            WHERE id = ?
        """, (
            new_content,
            json.dumps(new_tags),
            new_category,
            now.isoformat(),
            note_id
        ))

        conn.commit()
        conn.close()

        return Note(
            id=note_id,
            content=new_content,
            created_at=existing.created_at,
            updated_at=now,
            tags=new_tags,
            category=new_category,
            context=existing.context
        )

    async def delete_note(self, note_id: str) -> bool:
        """Delete a note.

        Args:
            note_id: Note ID

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    async def search(self, query: str, limit: int = 10) -> List[Note]:
        """Search notes using full-text search.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching Notes
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Use FTS5 match for full-text search
        cursor.execute("""
            SELECT n.id, n.content, n.created_at, n.updated_at, n.tags, n.category, n.context
            FROM notes n
            JOIN notes_fts fts ON n.id = fts.id
            WHERE notes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def search_simple(self, query: str, limit: int = 10) -> List[Note]:
        """Simple LIKE-based search (fallback if FTS fails).

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching Notes
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes
            WHERE content LIKE ? OR tags LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def get_recent(self, limit: int = 5) -> List[Note]:
        """Get most recent notes.

        Args:
            limit: Maximum results to return

        Returns:
            List of recent Notes
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def get_by_tag(self, tag: str, limit: int = 20) -> List[Note]:
        """Get notes by tag.

        Args:
            tag: Tag to filter by
            limit: Maximum results to return

        Returns:
            List of Notes with the tag
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Search in JSON tags array
        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes
            WHERE tags LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{tag}%", limit))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def get_today(self) -> List[Note]:
        """Get notes created today.

        Returns:
            List of today's Notes
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes
            WHERE created_at LIKE ?
            ORDER BY created_at DESC
        """, (f"{today}%",))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def get_by_category(self, category: str, limit: int = 20) -> List[Note]:
        """Get notes by category.

        Args:
            category: Category to filter by
            limit: Maximum results to return

        Returns:
            List of Notes in the category
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, created_at, updated_at, tags, category, context
            FROM notes
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (category, limit))

        rows = cursor.fetchall()
        conn.close()

        return [Note.from_row(row) for row in rows]

    async def count_notes(self) -> int:
        """Get total note count.

        Returns:
            Total number of notes
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def _auto_tag(self, content: str) -> List[str]:
        """Auto-generate tags based on content.

        Args:
            content: Note content

        Returns:
            List of auto-generated tags
        """
        tags = []
        content_lower = content.lower()

        # Category keywords
        tag_patterns = {
            "work": ["meeting", "project", "deadline", "client", "email", "call", "office"],
            "personal": ["family", "friend", "birthday", "vacation", "home"],
            "idea": ["idea", "thought", "concept", "maybe", "could"],
            "todo": ["todo", "need to", "must", "should", "remember to", "don't forget"],
            "reminder": ["remind", "reminder", "don't forget", "remember"],
            "poker": ["poker", "hand", "bet", "fold", "raise", "pot", "equity"],
            "code": ["code", "bug", "feature", "api", "function", "class", "debug"],
            "health": ["workout", "exercise", "diet", "sleep", "health"],
            "finance": ["money", "budget", "expense", "pay", "cost", "$"],
        }

        for tag, keywords in tag_patterns.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        return tags[:5]  # Max 5 auto-tags


# Global instance
_notes_service: Optional[NotesService] = None


def get_notes_service() -> NotesService:
    """Get or create global notes service."""
    global _notes_service
    if _notes_service is None:
        _notes_service = NotesService()
    return _notes_service
