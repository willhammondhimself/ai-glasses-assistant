"""Base class for agent tools."""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    message: str


class BaseTool(ABC):
    """Abstract base class for agent tools."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for Gemini function calling

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
