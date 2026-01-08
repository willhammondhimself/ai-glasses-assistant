"""Code debug voice tool - wraps CodeTool for voice interaction."""
import logging
import re
from typing import Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class CodeVoiceTool(VoiceTool):
    """Voice tool for code execution and debugging.

    Wraps the existing CodeTool for natural voice interaction.
    Supports running Python, shell commands, and error explanation.
    """

    name = "code"
    description = "Run code, execute commands, and debug errors"

    keywords = [
        # Run Python
        r"\brun\s+(this\s+)?python\b",
        r"\bexecute\s+(this\s+)?python\b",
        r"\brun\s+(this\s+)?code\b",
        r"\bexecute\s+(this\s+)?code\b",
        r"\btry\s+(this\s+)?code\b",
        r"\btest\s+(this\s+)?code\b",
        # Run shell/command
        r"\brun\s+(this\s+)?command\b",
        r"\brun\s+(this\s+)?shell\b",
        r"\bexecute\s+(this\s+)?command\b",
        r"\brun\s+in\s+(the\s+)?terminal\b",
        # Error explanation
        r"\bexplain\s+(this\s+)?error\b",
        r"\bwhat\s+does\s+(this\s+)?error\s+mean\b",
        r"\bdebug\s+(this|my)\b",
        r"\bwhat('s|\s+is)\s+wrong\s+with\b",
        r"\bhelp\s+me\s+(with\s+)?(this\s+)?error\b",
        r"\bwhy\s+am\s+i\s+getting\b",
        r"\bfix\s+(this\s+)?error\b",
        # Quick eval
        r"\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b",
        r"\bcalculate\s+\d+\b",
    ]

    priority = 7  # Medium priority

    def __init__(self):
        self._code_tool = None

    def _get_code_tool(self):
        """Lazy load the code tool."""
        if self._code_tool is None:
            from backend.agent.tools.code import CodeTool
            self._code_tool = CodeTool()
        return self._code_tool

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute code operation based on voice query.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with voice-friendly response
        """
        try:
            action, params = self._parse_query(query)
            logger.info(f"Code action: {action}, code length: {len(params.get('code', ''))}")

            if action == "none":
                return VoiceToolResult(
                    success=False,
                    message="I couldn't understand what code you want me to run. Please say 'run python' followed by your code, or 'explain error' followed by the error message."
                )

            tool = self._get_code_tool()
            result = await tool.execute(action=action, **params)

            if not result.success:
                return VoiceToolResult(
                    success=False,
                    message=self._format_error_for_voice(result.message or "Code execution failed.")
                )

            message = self._format_for_voice(result, action, params)
            return VoiceToolResult(
                success=True,
                message=message,
                data=result.data
            )

        except Exception as e:
            logger.error(f"Code voice tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble running that code."
            )

    def _parse_query(self, query: str) -> Tuple[str, dict]:
        """Parse voice query to determine action and parameters.

        Args:
            query: The user's voice query

        Returns:
            Tuple of (action, params dict)
        """
        query_lower = query.lower()

        # Check for error explanation
        error_patterns = [
            "explain this error", "explain error", "what does this error mean",
            "what does error mean", "debug this", "debug my", "what's wrong with",
            "what is wrong with", "help me with this error", "help with error",
            "why am i getting", "fix this error", "fix error"
        ]
        for pattern in error_patterns:
            if pattern in query_lower:
                # Extract the error message after the trigger
                idx = query_lower.find(pattern)
                error_msg = query[idx + len(pattern):].strip()
                # Clean up
                error_msg = error_msg.lstrip(": ")
                if error_msg:
                    return "explain_error", {"code": error_msg}
                # Maybe the error is before the trigger
                error_msg = query[:idx].strip()
                if error_msg:
                    return "explain_error", {"code": error_msg}

        # Check for shell command
        shell_patterns = [
            "run command", "run this command", "run shell", "run this shell",
            "execute command", "run in terminal", "run in the terminal"
        ]
        for pattern in shell_patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern)
                cmd = query[idx + len(pattern):].strip()
                cmd = cmd.lstrip(": ")
                if cmd:
                    return "run_shell", {"code": cmd}

        # Check for Python code
        python_patterns = [
            "run python", "run this python", "execute python", "execute this python",
            "run code", "run this code", "execute code", "execute this code",
            "try code", "try this code", "test code", "test this code"
        ]
        for pattern in python_patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern)
                code = query[idx + len(pattern):].strip()
                code = code.lstrip(": ")
                if code:
                    return "run_python", {"code": code}

        # Quick math evaluation
        math_match = re.search(r'(\d+\s*[\+\-\*\/\%\*\*]+\s*\d+(?:\s*[\+\-\*\/\%\*\*]+\s*\d+)*)', query)
        if math_match:
            expr = math_match.group(1)
            return "run_python", {"code": f"print({expr})"}

        # "Calculate X" pattern
        if "calculate" in query_lower:
            idx = query_lower.find("calculate")
            expr = query[idx + 9:].strip()
            # Clean up spoken math
            expr = expr.replace(" plus ", "+").replace(" minus ", "-")
            expr = expr.replace(" times ", "*").replace(" divided by ", "/")
            expr = expr.replace(" to the power of ", "**")
            return "run_python", {"code": f"print({expr})"}

        # Default: try to find any code-like content
        # Look for code after a colon
        if ":" in query:
            code = query.split(":", 1)[1].strip()
            if code:
                return "run_python", {"code": code}

        return "none", {}

    def _format_for_voice(self, result, action: str, params: dict) -> str:
        """Format code result for voice output.

        Args:
            result: The ToolResult from CodeTool
            action: The action performed
            params: The parameters used

        Returns:
            Voice-friendly message
        """
        if action == "run_python" or action == "run_shell":
            output = result.data.get("output", "") if result.data else ""
            output = output.strip()

            if not output or output == "(no output)":
                return "Code ran successfully with no output."

            # Keep it concise for voice
            if len(output) > 200:
                output = output[:200] + "..."

            # Clean up for speech
            output = output.replace("\n", ". ")
            return f"Result: {output}"

        elif action == "explain_error":
            data = result.data or {}
            error_type = data.get("error_type", "")
            explanation = data.get("explanation", result.message)

            if error_type:
                return f"{error_type}. {explanation}"
            return explanation or "I couldn't determine the specific error type. Check line numbers in the error message."

        return result.message or "Done."

    def _format_error_for_voice(self, error_msg: str) -> str:
        """Format error message for voice output.

        Args:
            error_msg: Raw error message

        Returns:
            Voice-friendly error message
        """
        # Extract key error info
        if "SyntaxError" in error_msg:
            return "Syntax error in your code. Check for missing parentheses, quotes, or colons."
        elif "IndentationError" in error_msg:
            return "Indentation error. Check your code's spacing and tabs."
        elif "NameError" in error_msg:
            # Try to extract the undefined name
            match = re.search(r"name '(\w+)' is not defined", error_msg)
            if match:
                return f"Variable '{match.group(1)}' is not defined. Check the spelling."
            return "A variable is not defined. Check spelling and that it's defined before use."
        elif "TypeError" in error_msg:
            return "Type error. Check that you're using the right types for your operations."
        elif "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
            match = re.search(r"No module named '(\w+)'", error_msg)
            if match:
                return f"Module '{match.group(1)}' not found. You may need to install it with pip."
            return "Module not found. Check the import statement."
        elif "timed out" in error_msg.lower():
            return "The code took too long to run. It may have an infinite loop."
        elif "blocked" in error_msg.lower():
            return "That command was blocked for safety reasons."

        # Generic cleanup
        if len(error_msg) > 150:
            error_msg = error_msg[:150] + "..."
        return f"Error: {error_msg}"
