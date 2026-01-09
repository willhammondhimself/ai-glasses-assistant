"""Multi-table manager for handling concurrent poker tables.

Supports multiple simultaneous poker tables with independent state tracking.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TableStatus(Enum):
    """Table status states."""
    ACTIVE = "active"
    WAITING = "waiting"  # Waiting for action
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class TableState:
    """State for a single poker table."""
    table_id: str
    site: str = "generic"  # ignition, ggpoker, etc.
    status: TableStatus = TableStatus.ACTIVE
    window_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height

    # Last OCR state
    last_hole_cards: Optional[str] = None
    last_community: Optional[str] = None
    last_pot: float = 0.0
    last_stack: float = 0.0
    last_position: Optional[str] = None
    last_street: str = "unknown"

    # Table-specific tracking
    hands_played: int = 0
    villain_count: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "site": self.site,
            "status": self.status.value,
            "window_bounds": self.window_bounds,
            "last_hole_cards": self.last_hole_cards,
            "last_community": self.last_community,
            "last_pot": self.last_pot,
            "last_stack": self.last_stack,
            "last_position": self.last_position,
            "last_street": self.last_street,
            "hands_played": self.hands_played,
            "villain_count": self.villain_count,
            "last_update": self.last_update.isoformat(),
            "created_at": self.created_at.isoformat()
        }

    def to_summary(self) -> Dict[str, Any]:
        """Get short summary for table selector."""
        return {
            "table_id": self.table_id,
            "site": self.site,
            "status": self.status.value,
            "pot": self.last_pot,
            "position": self.last_position,
            "street": self.last_street,
            "hands": self.hands_played
        }

    def update_from_ocr(self, state_dict: Dict[str, Any]):
        """Update table state from OCR result.

        Args:
            state_dict: Dictionary from PokerTableState.to_dict()
        """
        self.last_hole_cards = " ".join(state_dict.get("hole_cards", [])) or None
        self.last_community = " ".join(state_dict.get("community_cards", [])) or None
        self.last_pot = state_dict.get("pot_size", 0.0)
        self.last_stack = state_dict.get("player_stack") or 0.0
        self.last_position = state_dict.get("position")
        self.last_street = state_dict.get("street", "unknown")
        self.last_update = datetime.utcnow()

        # Detect new hand
        if state_dict.get("street") == "preflop" and self.last_street in ("turn", "river"):
            self.hands_played += 1


class TableManager:
    """Manage multiple concurrent poker tables."""

    def __init__(self, max_tables: int = 8):
        """Initialize table manager.

        Args:
            max_tables: Maximum concurrent tables allowed
        """
        self.tables: Dict[str, TableState] = {}
        self.active_table: Optional[str] = None
        self.max_tables = max_tables
        self._table_counter = 0

    def register_table(
        self,
        table_id: Optional[str] = None,
        site: str = "generic",
        window_bounds: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """Register a new table.

        Args:
            table_id: Optional custom ID, auto-generates if not provided
            site: Poker site (ignition, ggpoker, etc.)
            window_bounds: Screen coordinates for the table window

        Returns:
            The table ID
        """
        if len(self.tables) >= self.max_tables:
            # Remove oldest inactive table
            self._cleanup_oldest()

        if table_id is None:
            self._table_counter += 1
            table_id = f"table_{self._table_counter}"

        self.tables[table_id] = TableState(
            table_id=table_id,
            site=site,
            window_bounds=window_bounds
        )

        # Set as active if first table
        if self.active_table is None:
            self.active_table = table_id

        logger.info(f"Registered table: {table_id} ({site})")
        return table_id

    def _cleanup_oldest(self):
        """Remove the oldest inactive table."""
        oldest_id = None
        oldest_time = datetime.utcnow()

        for tid, table in self.tables.items():
            if table.status != TableStatus.ACTIVE and table.last_update < oldest_time:
                oldest_time = table.last_update
                oldest_id = tid

        if oldest_id:
            self.remove_table(oldest_id)
        elif self.tables:
            # Remove first non-active table
            for tid, table in self.tables.items():
                if tid != self.active_table:
                    self.remove_table(tid)
                    break

    def remove_table(self, table_id: str) -> bool:
        """Remove a table.

        Args:
            table_id: ID of table to remove

        Returns:
            True if removed
        """
        if table_id in self.tables:
            del self.tables[table_id]

            # Switch active table if needed
            if self.active_table == table_id:
                self.active_table = next(iter(self.tables), None)

            logger.info(f"Removed table: {table_id}")
            return True
        return False

    def set_active_table(self, table_id: str) -> bool:
        """Set the active table for HUD display.

        Args:
            table_id: ID of table to activate

        Returns:
            True if successful
        """
        if table_id in self.tables:
            self.active_table = table_id
            logger.info(f"Active table set to: {table_id}")
            return True
        return False

    def next_table(self) -> Optional[str]:
        """Switch to next table in rotation.

        Returns:
            New active table ID or None
        """
        if not self.tables:
            return None

        table_ids = list(self.tables.keys())
        if self.active_table in table_ids:
            current_idx = table_ids.index(self.active_table)
            next_idx = (current_idx + 1) % len(table_ids)
            self.active_table = table_ids[next_idx]
        else:
            self.active_table = table_ids[0]

        return self.active_table

    def prev_table(self) -> Optional[str]:
        """Switch to previous table in rotation.

        Returns:
            New active table ID or None
        """
        if not self.tables:
            return None

        table_ids = list(self.tables.keys())
        if self.active_table in table_ids:
            current_idx = table_ids.index(self.active_table)
            prev_idx = (current_idx - 1) % len(table_ids)
            self.active_table = table_ids[prev_idx]
        else:
            self.active_table = table_ids[0]

        return self.active_table

    def update_table(self, table_id: str, state_dict: Dict[str, Any]) -> Optional[TableState]:
        """Update state for a specific table.

        Args:
            table_id: Table to update
            state_dict: OCR state dictionary

        Returns:
            Updated TableState or None
        """
        if table_id not in self.tables:
            # Auto-register if not exists
            self.register_table(table_id)

        table = self.tables[table_id]
        table.update_from_ocr(state_dict)
        return table

    def get_table(self, table_id: str) -> Optional[TableState]:
        """Get state for a specific table.

        Args:
            table_id: Table ID

        Returns:
            TableState or None
        """
        return self.tables.get(table_id)

    def get_active_table(self) -> Optional[TableState]:
        """Get the currently active table.

        Returns:
            Active TableState or None
        """
        if self.active_table:
            return self.tables.get(self.active_table)
        return None

    def get_all_tables(self) -> List[Dict[str, Any]]:
        """Get summary of all tables.

        Returns:
            List of table summaries
        """
        return [table.to_summary() for table in self.tables.values()]

    def get_table_count(self) -> int:
        """Get number of active tables."""
        return len(self.tables)

    def clear_all(self):
        """Clear all tables."""
        self.tables.clear()
        self.active_table = None
        logger.info("All tables cleared")

    def get_tables_needing_action(self) -> List[str]:
        """Get tables where action is needed (waiting status).

        Returns:
            List of table IDs needing action
        """
        return [
            tid for tid, table in self.tables.items()
            if table.status == TableStatus.WAITING
        ]


# Global instance
table_manager = TableManager()
