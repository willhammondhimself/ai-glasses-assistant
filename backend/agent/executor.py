"""Agent executor with Gemini function calling.

Uses the new google-genai SDK (unified SDK).
"""
import os
import asyncio
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from .tools import BaseTool, CalendarTool, PokerTool, CodeTool, ToolResult


class AgentExecutor:
    """Execute agent tasks using Gemini function calling."""

    def __init__(self, tools: List[BaseTool] = None):
        """Initialize agent with tools.

        Args:
            tools: List of tools available to the agent. Defaults to CalendarTool.
        """
        self.tools = {t.name: t for t in (tools or [CalendarTool(), PokerTool(), CodeTool()])}
        self._model = "gemini-2.0-flash"
        self._system_instruction = """You are WHAM, an AI assistant with tool execution capabilities.
When the user asks you to perform calendar operations like scheduling, listing, or clearing events,
use the calendar tool. When the user asks about poker odds, equity calculations, or EV,
use the poker tool. When the user wants to run Python code, execute shell commands, or needs
help debugging/explaining errors, use the code tool. Be concise and helpful in your responses."""

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self._tool_declarations = self._build_tool_declarations()
        self._chat_history: List[types.Content] = []

    def _build_tool_declarations(self) -> List[types.Tool]:
        """Build tool declarations for the new SDK."""
        function_declarations = []

        for tool in self.tools.values():
            # Build properties schema
            properties = {}
            for param_name, param_spec in tool.parameters.get("properties", {}).items():
                prop_schema = {
                    "type": param_spec.get("type", "STRING").upper(),
                    "description": param_spec.get("description", ""),
                }
                if "enum" in param_spec:
                    prop_schema["enum"] = param_spec["enum"]
                properties[param_name] = prop_schema

            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters={
                    "type": "OBJECT",
                    "properties": properties,
                    "required": tool.parameters.get("required", [])
                }
            )
            function_declarations.append(func_decl)

        return [types.Tool(function_declarations=function_declarations)]

    async def run(self, user_message: str) -> Dict[str, Any]:
        """Process user message, call tools if needed, return response.

        Args:
            user_message: Natural language input from user

        Returns:
            Dict with response, tool_used, tool_result, and success fields
        """
        try:
            # Add user message to history
            self._chat_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)]
                )
            )

            # Generate with function calling
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self._model,
                contents=self._chat_history,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    tools=self._tool_declarations,
                )
            )

            # Check if model wants to call a function
            if (response.candidates and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].function_call):

                fc = response.candidates[0].content.parts[0].function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                # Add model's function call to history
                self._chat_history.append(response.candidates[0].content)

                # Execute the tool
                if tool_name in self.tools:
                    result = await self.tools[tool_name].execute(**tool_args)

                    # Add function result to history
                    function_response = types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result.message, "data": str(result.data)}
                        )]
                    )
                    self._chat_history.append(function_response)

                    # Get final response from model
                    final_response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self._model,
                        contents=self._chat_history,
                        config=types.GenerateContentConfig(
                            system_instruction=self._system_instruction,
                        )
                    )

                    # Add model response to history
                    if final_response.candidates:
                        self._chat_history.append(final_response.candidates[0].content)

                    return {
                        "response": final_response.text,
                        "tool_used": tool_name,
                        "tool_result": result.message,
                        "success": result.success
                    }
                else:
                    return {
                        "response": f"Unknown tool: {tool_name}",
                        "tool_used": tool_name,
                        "success": False
                    }

            # No function call - direct response
            if response.candidates:
                self._chat_history.append(response.candidates[0].content)

            return {
                "response": response.text,
                "tool_used": None,
                "success": True
            }

        except Exception as e:
            return {
                "response": f"Agent error: {str(e)}",
                "tool_used": None,
                "success": False
            }

    def reset_chat(self):
        """Reset conversation history."""
        self._chat_history = []


_agent: Optional[AgentExecutor] = None


def get_agent() -> AgentExecutor:
    """Get singleton AgentExecutor instance."""
    global _agent
    if _agent is None:
        _agent = AgentExecutor()
    return _agent
