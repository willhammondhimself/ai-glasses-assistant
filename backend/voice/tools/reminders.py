"""Reminders and timers tool using SQLite."""
import asyncio
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "reminders.db"


class RemindersTool(VoiceTool):
    """Set reminders and timers with local SQLite storage."""

    name = "reminders"
    description = "Set reminders and timers"

    keywords = [
        r"\bremind\s+me\b",
        r"\bset\s+(a\s+)?reminder\b",
        r"\bset\s+(a\s+)?timer\b",
        r"\btimer\s+for\b",
        r"\bin\s+\d+\s+(minute|hour|second)s?\b",
        r"\bwhat\s+are\s+my\s+reminders\b",
        r"\blist\s+(my\s+)?reminders\b",
        r"\bcancel\s+(the\s+)?reminder\b",
        r"\bdelete\s+(the\s+)?reminder\b",
    ]

    priority = 15  # High priority for reminder-specific queries

    def __init__(self):
        self._init_db()
        self._active_timers: Dict[int, asyncio.Task] = {}

    def _init_db(self) -> None:
        """Initialize the reminders database."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                due_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_timer BOOLEAN DEFAULT FALSE,
                completed BOOLEAN DEFAULT FALSE
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Reminders database initialized at {DB_PATH}")

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Handle reminder/timer commands.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with reminder info
        """
        query_lower = query.lower()

        # Determine the action
        if any(word in query_lower for word in ["list", "what are", "show"]):
            return await self._list_reminders()
        elif any(word in query_lower for word in ["cancel", "delete", "remove"]):
            return await self._cancel_reminder(query)
        elif "timer" in query_lower:
            return await self._set_timer(query)
        else:
            return await self._set_reminder(query)

    async def _set_reminder(self, query: str) -> VoiceToolResult:
        """Set a reminder for a specific time."""
        # Extract time and message
        time_delta, time_str = self._parse_time(query)
        message = self._extract_reminder_message(query)

        if not time_delta:
            return VoiceToolResult(
                success=False,
                message="I couldn't understand when you want to be reminded. Try saying something like 'remind me in 30 minutes'."
            )

        due_at = datetime.now() + time_delta

        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (message, due_at, is_timer) VALUES (?, ?, ?)",
            (message, due_at.isoformat(), False)
        )
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return VoiceToolResult(
            success=True,
            message=f"Got it. I'll remind you {time_str}: {message}",
            data={"reminder_id": reminder_id, "due_at": due_at.isoformat(), "message": message}
        )

    async def _set_timer(self, query: str) -> VoiceToolResult:
        """Set a countdown timer."""
        time_delta, time_str = self._parse_time(query)

        if not time_delta:
            return VoiceToolResult(
                success=False,
                message="I couldn't understand the timer duration. Try saying something like 'set a timer for 5 minutes'."
            )

        due_at = datetime.now() + time_delta
        message = f"Timer for {time_str}"

        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (message, due_at, is_timer) VALUES (?, ?, ?)",
            (message, due_at.isoformat(), True)
        )
        timer_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Format the duration nicely
        total_seconds = int(time_delta.total_seconds())
        if total_seconds >= 3600:
            duration_str = f"{total_seconds // 3600} hour{'s' if total_seconds >= 7200 else ''}"
            if total_seconds % 3600:
                duration_str += f" and {(total_seconds % 3600) // 60} minutes"
        elif total_seconds >= 60:
            duration_str = f"{total_seconds // 60} minute{'s' if total_seconds >= 120 else ''}"
        else:
            duration_str = f"{total_seconds} second{'s' if total_seconds != 1 else ''}"

        return VoiceToolResult(
            success=True,
            message=f"Timer set for {duration_str}.",
            data={"timer_id": timer_id, "due_at": due_at.isoformat(), "duration_seconds": total_seconds}
        )

    async def _list_reminders(self) -> VoiceToolResult:
        """List all active reminders."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, message, due_at, is_timer FROM reminders WHERE completed = FALSE AND due_at > ? ORDER BY due_at",
            (datetime.now().isoformat(),)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return VoiceToolResult(
                success=True,
                message="You don't have any active reminders or timers.",
                data={"reminders": []}
            )

        reminders = []
        for row in rows:
            id_, message, due_at_str, is_timer = row
            due_at = datetime.fromisoformat(due_at_str)
            time_until = due_at - datetime.now()

            if time_until.total_seconds() > 0:
                reminders.append({
                    "id": id_,
                    "message": message,
                    "due_at": due_at_str,
                    "is_timer": bool(is_timer),
                    "time_until_seconds": time_until.total_seconds()
                })

        # Build voice response
        if len(reminders) == 1:
            r = reminders[0]
            time_str = self._format_time_until(r["time_until_seconds"])
            return VoiceToolResult(
                success=True,
                message=f"You have one reminder: {r['message']}, due {time_str}.",
                data={"reminders": reminders}
            )
        else:
            response = f"You have {len(reminders)} reminders. "
            for r in reminders[:3]:  # Limit to 3 for voice
                time_str = self._format_time_until(r["time_until_seconds"])
                response += f"{r['message']} due {time_str}. "

            return VoiceToolResult(
                success=True,
                message=response.strip(),
                data={"reminders": reminders}
            )

    async def _cancel_reminder(self, query: str) -> VoiceToolResult:
        """Cancel a reminder or timer."""
        # For simplicity, cancel the most recent or all
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if "all" in query.lower():
            cursor.execute("UPDATE reminders SET completed = TRUE WHERE completed = FALSE")
            count = cursor.rowcount
            conn.commit()
            conn.close()
            return VoiceToolResult(
                success=True,
                message=f"Cancelled {count} reminder{'s' if count != 1 else ''}.",
                data={"cancelled_count": count}
            )
        else:
            # Cancel the most recent
            cursor.execute(
                "SELECT id FROM reminders WHERE completed = FALSE ORDER BY created_at DESC LIMIT 1"
            )
            row = cursor.fetchone()

            if row:
                cursor.execute("UPDATE reminders SET completed = TRUE WHERE id = ?", (row[0],))
                conn.commit()
                conn.close()
                return VoiceToolResult(
                    success=True,
                    message="Cancelled your most recent reminder.",
                    data={"cancelled_id": row[0]}
                )
            else:
                conn.close()
                return VoiceToolResult(
                    success=True,
                    message="You don't have any active reminders to cancel."
                )

    def _parse_time(self, query: str) -> tuple[Optional[timedelta], str]:
        """Parse time duration from query.

        Returns:
            Tuple of (timedelta, human-readable string) or (None, "")
        """
        query_lower = query.lower()

        # Patterns: "in X minutes", "for X hours", "X seconds"
        patterns = [
            (r"(\d+)\s*(?:second|sec)s?", "seconds"),
            (r"(\d+)\s*(?:minute|min)s?", "minutes"),
            (r"(\d+)\s*(?:hour|hr)s?", "hours"),
            (r"half\s+(?:an?\s+)?hour", "half_hour"),
            (r"quarter\s+(?:of\s+)?(?:an?\s+)?hour", "quarter_hour"),
        ]

        for pattern, unit in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if unit == "half_hour":
                    return timedelta(minutes=30), "in 30 minutes"
                elif unit == "quarter_hour":
                    return timedelta(minutes=15), "in 15 minutes"
                else:
                    value = int(match.group(1))
                    if unit == "seconds":
                        return timedelta(seconds=value), f"in {value} seconds"
                    elif unit == "minutes":
                        return timedelta(minutes=value), f"in {value} minutes"
                    elif unit == "hours":
                        return timedelta(hours=value), f"in {value} hours"

        return None, ""

    def _extract_reminder_message(self, query: str) -> str:
        """Extract the reminder message from query."""
        # Remove common prefixes
        patterns_to_remove = [
            r"remind\s+me\s+(to\s+)?",
            r"set\s+(a\s+)?reminder\s+(to\s+)?",
            r"in\s+\d+\s+\w+\s+(to\s+)?",
        ]

        cleaned = query
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.strip()
        return cleaned if cleaned else "Reminder"

    def _format_time_until(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"in {int(seconds)} seconds"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"in {mins} minute{'s' if mins != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            if mins:
                return f"in {hours} hour{'s' if hours != 1 else ''} and {mins} minutes"
            return f"in {hours} hour{'s' if hours != 1 else ''}"
