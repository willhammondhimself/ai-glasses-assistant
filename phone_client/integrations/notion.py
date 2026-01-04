"""
Notion API Integration for WHAM

Export captures, memories, and session summaries to Notion databases.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - Notion API calls will be mocked")


class NotionBlockType(Enum):
    """Supported Notion block types."""
    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    BULLETED_LIST = "bulleted_list_item"
    NUMBERED_LIST = "numbered_list_item"
    TODO = "to_do"
    CODE = "code"
    QUOTE = "quote"
    DIVIDER = "divider"
    CALLOUT = "callout"


@dataclass
class NotionPage:
    """Represents a Notion page."""
    id: str
    title: str
    url: str
    created_time: datetime
    last_edited: datetime
    parent_id: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: dict) -> "NotionPage":
        """Create from Notion API response."""
        # Extract title from properties
        title = ""
        if "properties" in data:
            title_prop = data["properties"].get("title") or data["properties"].get("Name")
            if title_prop and "title" in title_prop:
                title = "".join(t.get("plain_text", "") for t in title_prop["title"])

        return cls(
            id=data["id"],
            title=title,
            url=data.get("url", ""),
            created_time=datetime.fromisoformat(data["created_time"].replace("Z", "+00:00")),
            last_edited=datetime.fromisoformat(data["last_edited_time"].replace("Z", "+00:00")),
            parent_id=data.get("parent", {}).get("database_id")
        )


@dataclass
class NotionDatabase:
    """Represents a Notion database."""
    id: str
    title: str
    url: str
    properties: Dict[str, str]  # property_name -> property_type


class NotionIntegration:
    """
    Notion API integration for exporting WHAM data.

    Get your integration token at: https://www.notion.so/my-integrations
    """

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, token: Optional[str] = None):
        """
        Initialize Notion integration.

        Args:
            token: Notion integration token. If not provided, uses NOTION_TOKEN env var.
        """
        self.token = token or os.environ.get("NOTION_TOKEN")
        self._headers = {
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION
        }

    @property
    def is_configured(self) -> bool:
        """Check if API token is configured."""
        return bool(self.token) and HAS_REQUESTS

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None
    ) -> Optional[dict]:
        """Make authenticated request to Notion API."""
        if not self.is_configured:
            logger.warning("Notion not configured")
            return None

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Notion API error: {e}")
            return None

    def create_page(
        self,
        title: str,
        content: str,
        database_id: str,
        properties: Optional[Dict[str, Any]] = None,
        icon: Optional[str] = None
    ) -> Optional[NotionPage]:
        """
        Create a new page in a Notion database.

        Args:
            title: Page title
            content: Page content (markdown-like text)
            database_id: Parent database ID
            properties: Additional database properties
            icon: Emoji icon for the page

        Returns:
            Created NotionPage or None on failure
        """
        if not self.is_configured:
            logger.info("Notion not configured, mock page creation")
            return self._mock_page(title)

        # Build page properties
        page_properties = {
            "Name": {
                "title": [{"text": {"content": title}}]
            }
        }

        # Add custom properties
        if properties:
            for key, value in properties.items():
                if isinstance(value, str):
                    page_properties[key] = {
                        "rich_text": [{"text": {"content": value}}]
                    }
                elif isinstance(value, bool):
                    page_properties[key] = {"checkbox": value}
                elif isinstance(value, (int, float)):
                    page_properties[key] = {"number": value}
                elif isinstance(value, datetime):
                    page_properties[key] = {
                        "date": {"start": value.isoformat()}
                    }
                elif isinstance(value, list):  # Multi-select
                    page_properties[key] = {
                        "multi_select": [{"name": v} for v in value]
                    }

        # Build request body
        body: Dict[str, Any] = {
            "parent": {"database_id": database_id},
            "properties": page_properties,
            "children": self._parse_content_to_blocks(content)
        }

        if icon:
            body["icon"] = {"type": "emoji", "emoji": icon}

        result = self._make_request("POST", "pages", body)
        if result:
            return NotionPage.from_api_response(result)
        return None

    def update_page(
        self,
        page_id: str,
        properties: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None
    ) -> bool:
        """
        Update an existing Notion page.

        Args:
            page_id: Page ID to update
            properties: Properties to update
            content: New content (replaces existing)

        Returns:
            True on success
        """
        if not self.is_configured:
            logger.info("Notion not configured, mock page update")
            return True

        if properties:
            page_props = {}
            for key, value in properties.items():
                if isinstance(value, str):
                    page_props[key] = {
                        "rich_text": [{"text": {"content": value}}]
                    }
                elif isinstance(value, bool):
                    page_props[key] = {"checkbox": value}

            result = self._make_request("PATCH", f"pages/{page_id}", {
                "properties": page_props
            })
            if not result:
                return False

        # Update content by appending blocks
        if content:
            blocks = self._parse_content_to_blocks(content)
            result = self._make_request(
                "PATCH",
                f"blocks/{page_id}/children",
                {"children": blocks}
            )
            if not result:
                return False

        return True

    def search_pages(
        self,
        query: str,
        database_id: Optional[str] = None
    ) -> List[NotionPage]:
        """
        Search for pages in Notion.

        Args:
            query: Search query
            database_id: Optional database to search within

        Returns:
            List of matching pages
        """
        if not self.is_configured:
            logger.info("Notion not configured, returning empty search")
            return []

        body: Dict[str, Any] = {
            "query": query,
            "filter": {"property": "object", "value": "page"}
        }

        if database_id:
            # Use database query endpoint instead
            result = self._make_request("POST", f"databases/{database_id}/query", {
                "filter": {
                    "property": "Name",
                    "title": {"contains": query}
                }
            })
        else:
            result = self._make_request("POST", "search", body)

        if not result:
            return []

        return [
            NotionPage.from_api_response(page)
            for page in result.get("results", [])
        ]

    def get_databases(self) -> List[NotionDatabase]:
        """
        Get all accessible databases.

        Returns:
            List of databases the integration can access
        """
        if not self.is_configured:
            logger.info("Notion not configured, returning mock databases")
            return self._mock_databases()

        result = self._make_request("POST", "search", {
            "filter": {"property": "object", "value": "database"}
        })

        if not result:
            return []

        databases = []
        for db in result.get("results", []):
            title = ""
            if db.get("title"):
                title = "".join(t.get("plain_text", "") for t in db["title"])

            properties = {
                name: prop["type"]
                for name, prop in db.get("properties", {}).items()
            }

            databases.append(NotionDatabase(
                id=db["id"],
                title=title,
                url=db.get("url", ""),
                properties=properties
            ))

        return databases

    def _parse_content_to_blocks(self, content: str) -> List[dict]:
        """
        Parse text content into Notion blocks.

        Supports basic markdown-like formatting:
        - # Heading 1
        - ## Heading 2
        - ### Heading 3
        - - Bullet item
        - 1. Numbered item
        - [ ] Todo item
        - > Quote
        - ``` Code block
        """
        blocks = []
        lines = content.split("\n")
        in_code_block = False
        code_content = []

        for line in lines:
            stripped = line.strip()

            # Handle code blocks
            if stripped.startswith("```"):
                if in_code_block:
                    blocks.append({
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": [{"text": {"content": "\n".join(code_content)}}],
                            "language": "plain text"
                        }
                    })
                    code_content = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_content.append(line)
                continue

            # Skip empty lines
            if not stripped:
                continue

            # Headings
            if stripped.startswith("### "):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"text": {"content": stripped[4:]}}]
                    }
                })
            elif stripped.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": stripped[3:]}}]
                    }
                })
            elif stripped.startswith("# "):
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": stripped[2:]}}]
                    }
                })
            # Bullets
            elif stripped.startswith("- ") or stripped.startswith("* "):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": stripped[2:]}}]
                    }
                })
            # Numbered lists
            elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in [". ", ") "]:
                blocks.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": stripped[3:]}}]
                    }
                })
            # Todos
            elif stripped.startswith("[ ] ") or stripped.startswith("[x] "):
                checked = stripped.startswith("[x]")
                blocks.append({
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": stripped[4:]}}],
                        "checked": checked
                    }
                })
            # Quotes
            elif stripped.startswith("> "):
                blocks.append({
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{"text": {"content": stripped[2:]}}]
                    }
                })
            # Dividers
            elif stripped in ["---", "***", "___"]:
                blocks.append({
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                })
            # Regular paragraph
            else:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": stripped}}]
                    }
                })

        return blocks

    def _mock_page(self, title: str) -> NotionPage:
        """Generate mock page for testing."""
        return NotionPage(
            id="mock-page-id",
            title=title,
            url="https://notion.so/mock-page",
            created_time=datetime.now(),
            last_edited=datetime.now()
        )

    def _mock_databases(self) -> List[NotionDatabase]:
        """Generate mock databases for testing."""
        return [
            NotionDatabase(
                id="mock-captures-db",
                title="WHAM Captures",
                url="https://notion.so/captures",
                properties={"Name": "title", "Type": "select", "Priority": "select"}
            ),
            NotionDatabase(
                id="mock-memories-db",
                title="WHAM Memories",
                url="https://notion.so/memories",
                properties={"Name": "title", "Category": "select", "Context": "rich_text"}
            )
        ]


# Export helpers for WHAM features
class WHAMNotionExporter:
    """
    High-level exporter for WHAM data to Notion.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        captures_database_id: Optional[str] = None,
        memories_database_id: Optional[str] = None,
        sessions_database_id: Optional[str] = None
    ):
        """
        Initialize WHAM Notion exporter.

        Args:
            token: Notion API token
            captures_database_id: Database ID for captures
            memories_database_id: Database ID for memories
            sessions_database_id: Database ID for session summaries
        """
        self.notion = NotionIntegration(token)
        self.captures_db = captures_database_id or os.environ.get("NOTION_CAPTURES_DB")
        self.memories_db = memories_database_id or os.environ.get("NOTION_MEMORIES_DB")
        self.sessions_db = sessions_database_id or os.environ.get("NOTION_SESSIONS_DB")

    def export_capture(
        self,
        title: str,
        content: str,
        capture_type: str,
        priority: str = "medium",
        tags: Optional[List[str]] = None
    ) -> Optional[NotionPage]:
        """
        Export a capture to Notion.

        Args:
            title: Capture title
            content: Capture content
            capture_type: Type (thought, task, reminder, etc.)
            priority: Priority level
            tags: Optional tags

        Returns:
            Created page or None
        """
        if not self.captures_db:
            logger.warning("Captures database ID not configured")
            return None

        properties = {
            "Type": capture_type.capitalize(),
            "Priority": priority.capitalize(),
            "Created": datetime.now()
        }

        if tags:
            properties["Tags"] = tags

        icon_map = {
            "thought": "💭",
            "task": "✅",
            "reminder": "⏰",
            "idea": "💡",
            "code": "💻",
            "meeting": "📅"
        }

        return self.notion.create_page(
            title=title,
            content=content,
            database_id=self.captures_db,
            properties=properties,
            icon=icon_map.get(capture_type, "📝")
        )

    def export_memory(
        self,
        key: str,
        value: str,
        category: str,
        context: Optional[str] = None
    ) -> Optional[NotionPage]:
        """
        Export a memory to Notion.

        Args:
            key: Memory key/title
            value: Memory value
            category: Memory category
            context: Optional context

        Returns:
            Created page or None
        """
        if not self.memories_db:
            logger.warning("Memories database ID not configured")
            return None

        content = f"## Value\n{value}"
        if context:
            content += f"\n\n## Context\n{context}"

        properties = {
            "Category": category.capitalize(),
            "Created": datetime.now()
        }

        icon_map = {
            "personal": "👤",
            "work": "💼",
            "health": "🏥",
            "finance": "💰",
            "preferences": "⚙️"
        }

        return self.notion.create_page(
            title=key,
            content=content,
            database_id=self.memories_db,
            properties=properties,
            icon=icon_map.get(category.lower(), "🧠")
        )

    def export_session_summary(
        self,
        date: datetime,
        summary: str,
        stats: Dict[str, Any]
    ) -> Optional[NotionPage]:
        """
        Export a session summary to Notion.

        Args:
            date: Session date
            summary: Session summary text
            stats: Session statistics

        Returns:
            Created page or None
        """
        if not self.sessions_db:
            logger.warning("Sessions database ID not configured")
            return None

        title = f"WHAM Session - {date.strftime('%Y-%m-%d')}"

        content = f"## Summary\n{summary}\n\n## Statistics\n"
        for key, value in stats.items():
            content += f"- **{key}**: {value}\n"

        properties = {
            "Date": date,
            "Captures": stats.get("captures_count", 0),
            "XP Earned": stats.get("xp_earned", 0)
        }

        return self.notion.create_page(
            title=title,
            content=content,
            database_id=self.sessions_db,
            properties=properties,
            icon="📊"
        )


# Convenience function
def create_page(
    title: str,
    content: str,
    database_id: str,
    token: Optional[str] = None
) -> Optional[NotionPage]:
    """
    Quick helper to create a Notion page.

    Args:
        title: Page title
        content: Page content
        database_id: Parent database ID
        token: Optional API token (uses env var if not provided)

    Returns:
        Created NotionPage or None
    """
    integration = NotionIntegration(token)
    return integration.create_page(title, content, database_id)
