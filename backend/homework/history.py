"""
JSON-based homework history logging for WHAM Homework Copilot.

Simple file-based storage with file locking for concurrent access.
Stores homework problems, solutions, and user annotations.
"""

import json
import uuid
import time
import fcntl
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default history file location
HISTORY_FILE = Path("logs/homework_history.json")


def _ensure_file_exists() -> None:
    """Ensure the history file and directory exist."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(json.dumps({"entries": [], "metadata": {
            "created_at": time.time(),
            "version": "1.0"
        }}, indent=2))


def _load_history() -> Dict[str, Any]:
    """
    Load history with file locking for safe concurrent access.

    Returns:
        dict with 'entries' list and 'metadata'
    """
    _ensure_file_exists()

    with open(HISTORY_FILE, 'r') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error("Corrupted history file, resetting")
            data = {"entries": [], "metadata": {"created_at": time.time(), "version": "1.0"}}
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return data


def _save_history(data: Dict[str, Any]) -> None:
    """
    Save history with file locking for safe concurrent access.

    Args:
        data: Full history dict to save
    """
    _ensure_file_exists()

    with open(HISTORY_FILE, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_entry(entry: Dict[str, Any]) -> str:
    """
    Append a homework entry to history.

    Args:
        entry: Homework entry dict containing:
            - problem_latex: LaTeX equation string
            - problem_type: Type (algebra, calculus, etc.)
            - solution_summary: Brief solution summary
            - full_solution: Optional complete solution details
            - tags: Optional list of tags
            - starred: Optional boolean

    Returns:
        str: Generated entry ID
    """
    data = _load_history()

    # Generate ID and timestamp if not provided
    entry_id = entry.get("id", str(uuid.uuid4()))
    entry["id"] = entry_id
    entry["timestamp"] = entry.get("timestamp", time.time())
    entry.setdefault("tags", [])
    entry.setdefault("starred", False)

    data["entries"].append(entry)
    data["metadata"]["last_updated"] = time.time()
    data["metadata"]["total_count"] = len(data["entries"])

    _save_history(data)
    logger.info(f"Appended homework entry {entry_id}: {entry.get('problem_type', 'unknown')}")

    return entry_id


def query_history(
    limit: int = 20,
    offset: int = 0,
    problem_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    starred_only: bool = False,
    start_date: Optional[float] = None,
    end_date: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query history with filters.

    Args:
        limit: Maximum entries to return (1-100)
        offset: Pagination offset
        problem_type: Filter by problem type
        tags: Filter by tags (any match)
        starred_only: Only return starred entries
        start_date: Filter by start timestamp
        end_date: Filter by end timestamp

    Returns:
        dict with 'entries' list and 'total' count
    """
    data = _load_history()
    entries = data.get("entries", [])

    # Apply filters
    filtered = []
    for entry in entries:
        # Problem type filter
        if problem_type and entry.get("problem_type") != problem_type:
            continue

        # Tags filter (any match)
        if tags:
            entry_tags = set(entry.get("tags", []))
            if not entry_tags.intersection(tags):
                continue

        # Starred filter
        if starred_only and not entry.get("starred", False):
            continue

        # Date range filters
        entry_time = entry.get("timestamp", 0)
        if start_date and entry_time < start_date:
            continue
        if end_date and entry_time > end_date:
            continue

        filtered.append(entry)

    # Sort by timestamp descending (newest first)
    filtered.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

    # Apply pagination
    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return {
        "entries": paginated,
        "total": total,
    }


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single entry by ID.

    Args:
        entry_id: Entry UUID

    Returns:
        Entry dict or None if not found
    """
    data = _load_history()

    for entry in data.get("entries", []):
        if entry.get("id") == entry_id:
            return entry

    return None


def update_entry(entry_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update an entry by ID.

    Args:
        entry_id: Entry UUID
        updates: Fields to update (tags, starred, etc.)

    Returns:
        bool: True if updated, False if not found
    """
    data = _load_history()

    for i, entry in enumerate(data.get("entries", [])):
        if entry.get("id") == entry_id:
            # Only allow updating certain fields
            allowed_fields = {"tags", "starred"}
            for key, value in updates.items():
                if key in allowed_fields:
                    data["entries"][i][key] = value

            data["metadata"]["last_updated"] = time.time()
            _save_history(data)
            logger.info(f"Updated homework entry {entry_id}")
            return True

    return False


def delete_entry(entry_id: str) -> bool:
    """
    Delete an entry by ID.

    Args:
        entry_id: Entry UUID

    Returns:
        bool: True if deleted, False if not found
    """
    data = _load_history()

    original_count = len(data.get("entries", []))
    data["entries"] = [e for e in data.get("entries", []) if e.get("id") != entry_id]

    if len(data["entries"]) < original_count:
        data["metadata"]["last_updated"] = time.time()
        data["metadata"]["total_count"] = len(data["entries"])
        _save_history(data)
        logger.info(f"Deleted homework entry {entry_id}")
        return True

    return False


def get_stats() -> Dict[str, Any]:
    """
    Get homework history statistics.

    Returns:
        dict with counts by type, total, starred count, etc.
    """
    data = _load_history()
    entries = data.get("entries", [])

    # Count by problem type
    by_type: Dict[str, int] = {}
    starred_count = 0

    for entry in entries:
        ptype = entry.get("problem_type", "unknown")
        by_type[ptype] = by_type.get(ptype, 0) + 1

        if entry.get("starred"):
            starred_count += 1

    # Recent activity (last 7 days)
    week_ago = time.time() - (7 * 24 * 3600)
    recent_count = sum(1 for e in entries if e.get("timestamp", 0) > week_ago)

    return {
        "total_count": len(entries),
        "by_problem_type": by_type,
        "starred_count": starred_count,
        "recent_count": recent_count,
        "metadata": data.get("metadata", {}),
    }
