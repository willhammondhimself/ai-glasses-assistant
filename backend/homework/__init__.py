"""Homework Vision Copilot module for WHAM."""

from .history import (
    append_entry,
    query_history,
    get_entry,
    update_entry,
    delete_entry,
    get_stats,
)

__all__ = [
    "append_entry",
    "query_history",
    "get_entry",
    "update_entry",
    "delete_entry",
    "get_stats",
]
