"""Tool router for voice queries - keyword detection and routing."""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

# Patterns to detect compound queries (tool chaining)
CHAIN_DELIMITERS = [
    r'\band\s+(?:also|then)\b',  # "and also", "and then"
    r'\bthen\b',                  # "then"
    r'\balso\b',                  # "also"
    r'\bafter\s+that\b',          # "after that"
    r'\bplus\b',                  # "plus"
    r',\s+and\b',                 # ", and"
]
CHAIN_PATTERN = re.compile('|'.join(CHAIN_DELIMITERS), re.IGNORECASE)


@dataclass
class ToolMatch:
    """Result of matching a tool to a query."""
    tool: VoiceTool
    confidence: float


class ToolRouter:
    """Routes voice queries to appropriate tools based on keywords."""

    def __init__(self):
        self.tools: List[VoiceTool] = []
        self._sorted = False

    def register(self, tool: VoiceTool) -> None:
        """Register a tool with the router.

        Args:
            tool: VoiceTool instance to register
        """
        self.tools.append(tool)
        self._sorted = False
        logger.info(f"Registered voice tool: {tool.name}")

    def register_all(self, tools: List[VoiceTool]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of VoiceTool instances
        """
        for tool in tools:
            self.register(tool)

    def _ensure_sorted(self) -> None:
        """Sort tools by priority (highest first)."""
        if not self._sorted:
            self.tools.sort(key=lambda t: t.priority, reverse=True)
            self._sorted = True

    def find_tool(self, query: str, min_confidence: float = 0.1) -> Optional[VoiceTool]:
        """Find the best matching tool for a query using confidence scoring.

        Args:
            query: The user's voice query
            min_confidence: Minimum confidence threshold (default 0.1)

        Returns:
            Matching VoiceTool or None
        """
        match = self.find_best_match(query, min_confidence)
        if match:
            return match.tool
        return None

    def find_tools_with_confidence(self, query: str, min_confidence: float = 0.1) -> List[ToolMatch]:
        """Find all matching tools with their confidence scores.

        Args:
            query: The user's voice query
            min_confidence: Minimum confidence threshold

        Returns:
            List of ToolMatch objects sorted by confidence (highest first)
        """
        matches = []
        for tool in self.tools:
            if tool.matches(query):
                confidence = tool.calculate_confidence(query)
                if confidence >= min_confidence:
                    matches.append(ToolMatch(tool=tool, confidence=confidence))

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def find_best_match(self, query: str, min_confidence: float = 0.1) -> Optional[ToolMatch]:
        """Find the best matching tool with confidence scoring.

        Args:
            query: The user's voice query
            min_confidence: Minimum confidence threshold

        Returns:
            Best ToolMatch or None
        """
        matches = self.find_tools_with_confidence(query, min_confidence)

        if not matches:
            logger.debug(f"No tool matched query: {query[:50]}...")
            return None

        best = matches[0]
        logger.debug(
            f"Query '{query[:50]}...' matched tool: {best.tool.name} "
            f"(confidence: {best.confidence:.2f})"
        )

        # Log alternatives if any
        if len(matches) > 1:
            alternatives = ", ".join(
                f"{m.tool.name}:{m.confidence:.2f}" for m in matches[1:3]
            )
            logger.debug(f"Alternative matches: {alternatives}")

        return best

    async def route(self, query: str, **kwargs) -> Optional[VoiceToolResult]:
        """Route a query to the appropriate tool and execute.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult from the matched tool, or None if no match
        """
        match = self.find_best_match(query)

        if match is None:
            return None

        tool = match.tool
        confidence = match.confidence

        try:
            logger.info(
                f"Executing tool '{tool.name}' (confidence: {confidence:.2f}) "
                f"for query: {query[:50]}..."
            )
            result = await tool.execute(query, **kwargs)

            # Add confidence to result data
            if result.data is None:
                result.data = {}
            if isinstance(result.data, dict):
                result.data["_routing"] = {
                    "tool": tool.name,
                    "confidence": confidence
                }

            logger.info(f"Tool '{tool.name}' completed: {result.success}")
            return result
        except Exception as e:
            logger.error(f"Tool '{tool.name}' failed: {e}")
            return VoiceToolResult(
                success=False,
                message=f"Sorry, the {tool.name} tool encountered an error.",
                data={"error": str(e), "_routing": {"tool": tool.name, "confidence": confidence}}
            )

    def is_compound_query(self, query: str) -> bool:
        """Check if query contains multiple tool requests (chaining).

        Args:
            query: The user's voice query

        Returns:
            True if query appears to have multiple parts
        """
        # Must have a chain delimiter
        if not CHAIN_PATTERN.search(query):
            return False

        # Split and check if multiple parts have tool matches
        parts = self.split_compound_query(query)
        if len(parts) < 2:
            return False

        # At least 2 parts must match different tools
        matched_tools = set()
        for part in parts:
            match = self.find_best_match(part)
            if match:
                matched_tools.add(match.tool.name)

        return len(matched_tools) >= 2

    def split_compound_query(self, query: str) -> List[str]:
        """Split a compound query into individual sub-queries.

        Args:
            query: The compound voice query

        Returns:
            List of individual query strings
        """
        parts = CHAIN_PATTERN.split(query)
        # Clean up and filter empty parts
        cleaned = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 3:  # Ignore very short fragments
                cleaned.append(part)
        return cleaned

    async def route_chain(self, query: str, **kwargs) -> VoiceToolResult:
        """Route a compound query to multiple tools and combine results.

        Args:
            query: The compound voice query
            **kwargs: Additional context

        Returns:
            Combined VoiceToolResult from all matched tools
        """
        parts = self.split_compound_query(query)
        logger.info(f"Chaining {len(parts)} tool requests")

        results = []
        tool_names = []

        for i, part in enumerate(parts):
            match = self.find_best_match(part)
            if match is None:
                logger.debug(f"Chain part {i+1} has no match: {part[:30]}...")
                continue

            tool = match.tool
            try:
                logger.info(f"Chain step {i+1}: {tool.name} (confidence: {match.confidence:.2f})")
                result = await tool.execute(part, **kwargs)
                results.append({
                    "tool": tool.name,
                    "success": result.success,
                    "message": result.message,
                    "data": result.data
                })
                tool_names.append(tool.name)
            except Exception as e:
                logger.error(f"Chain step {i+1} ({tool.name}) failed: {e}")
                results.append({
                    "tool": tool.name,
                    "success": False,
                    "message": f"{tool.name} failed",
                    "error": str(e)
                })

        if not results:
            return VoiceToolResult(
                success=False,
                message="I couldn't understand any of those requests.",
                data={"chain": []}
            )

        # Combine messages for voice output
        successful = [r for r in results if r["success"]]
        combined_message = ". ".join(r["message"] for r in successful if r.get("message"))

        if not combined_message:
            combined_message = f"Completed {len(results)} tasks."

        return VoiceToolResult(
            success=len(successful) > 0,
            message=combined_message,
            data={
                "chain": results,
                "tools_used": tool_names,
                "_routing": {"chained": True, "count": len(results)}
            }
        )

    async def route_smart(self, query: str, **kwargs) -> Optional[VoiceToolResult]:
        """Smart routing that handles both single and compound queries.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult from appropriate tool(s)
        """
        # Check for compound query first
        if self.is_compound_query(query):
            logger.info("Detected compound query, using chain routing")
            return await self.route_chain(query, **kwargs)

        # Fall back to single tool routing
        return await self.route(query, **kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool info dicts
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "keywords": tool.keywords,
                "priority": tool.priority
            }
            for tool in self.tools
        ]


# Global router instance
_router: Optional[ToolRouter] = None


def get_router() -> ToolRouter:
    """Get or create the global tool router."""
    global _router
    if _router is None:
        _router = ToolRouter()
    return _router


def register_default_tools(router: ToolRouter) -> None:
    """Register all default voice tools.

    This imports and registers all available tools.
    """
    from .perplexity import PerplexityTool
    from .weather import WeatherTool
    from .stocks import StocksTool
    from .reminders import RemindersTool
    from .maps import MapsTool
    from .math_tool import MathTool
    from .calendar_tool import CalendarVoiceTool
    from .memory_tool import MemoryVoiceTool
    from .code_tool import CodeVoiceTool
    from .health_tool import HealthVoiceTool
    from .poker_tool import PokerVoiceTool
    from .physics_tool import PhysicsVoiceTool
    from .news_tool import PerplexityNewsTool

    # JARVIS features - Sprint 1
    from .email_tool import EmailVoiceTool
    from .music_tool import MusicVoiceTool
    from .package_tool import PackageVoiceTool
    from .briefing_tool import BriefingVoiceTool

    # JARVIS features - Sprint 2
    from .vision_assistant_tool import VisionAssistantTool
    from .screenshot_tool import ScreenshotVoiceTool
    from .context_tool import ContextVoiceTool

    # JARVIS features - Sprint 3
    from .notes_tool import NotesVoiceTool
    from .smart_home_tool import SmartHomeVoiceTool
    from .fitness_tool import FitnessVoiceTool
    from .travel_tool import TravelVoiceTool

    # JARVIS features - Sprint 4
    from .learning_tool import LearningVoiceTool
    from .alert_tool import AlertVoiceTool

    tools = [
        # Core tools
        PerplexityTool(),
        WeatherTool(),
        StocksTool(),
        RemindersTool(),
        MapsTool(),
        MathTool(),
        CalendarVoiceTool(),
        MemoryVoiceTool(),
        CodeVoiceTool(),
        HealthVoiceTool(),
        PokerVoiceTool(),
        PhysicsVoiceTool(),
        PerplexityNewsTool(),  # Higher priority than generic Perplexity

        # JARVIS features - Sprint 1
        EmailVoiceTool(),
        MusicVoiceTool(),
        PackageVoiceTool(),
        BriefingVoiceTool(),

        # JARVIS features - Sprint 2
        VisionAssistantTool(),
        ScreenshotVoiceTool(),
        ContextVoiceTool(),

        # JARVIS features - Sprint 3
        NotesVoiceTool(),
        SmartHomeVoiceTool(),
        FitnessVoiceTool(),
        TravelVoiceTool(),

        # JARVIS features - Sprint 4
        LearningVoiceTool(),
        AlertVoiceTool(),
    ]

    router.register_all(tools)
    logger.info(f"Registered {len(tools)} default voice tools")
