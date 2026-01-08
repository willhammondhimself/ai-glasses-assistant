"""Code debug/execute tool for agent."""
import subprocess
import tempfile
import os
from .base import BaseTool, ToolResult


class CodeTool(BaseTool):
    """Execute and debug code snippets."""

    name = "code"
    description = "Execute code snippets, run shell commands, or explain/debug errors."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run_python", "run_shell", "explain_error"],
                "description": "Action: run_python (execute Python), run_shell (run command), or explain_error (debug help)"
            },
            "code": {"type": "string", "description": "Code to execute or error message to explain"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 5)"}
        },
        "required": ["action", "code"]
    }

    async def execute(self, action: str, code: str = "", **kwargs) -> ToolResult:
        """Execute code action."""
        timeout = int(kwargs.get("timeout", 5))
        code = str(code) if code else ""

        try:
            if action == "run_python":
                return await self._run_python(code, timeout)
            elif action == "run_shell":
                return await self._run_shell(code, timeout)
            elif action == "explain_error":
                return await self._explain_error(code)
            return ToolResult(False, None, f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(False, None, f"Code tool error: {str(e)}")

    async def _run_python(self, code: str, timeout: int) -> ToolResult:
        """Execute Python code in sandbox."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                temp_path = f.name

            try:
                result = subprocess.run(
                    ['python3', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            finally:
                os.unlink(temp_path)

            if result.returncode == 0:
                output = result.stdout.strip() or "(no output)"
                return ToolResult(
                    True,
                    {"output": result.stdout, "returncode": 0},
                    f"Output: {output[:500]}"
                )
            else:
                error = result.stderr.strip() or result.stdout.strip()
                return ToolResult(
                    False,
                    {"error": result.stderr, "returncode": result.returncode},
                    f"Error: {error[:500]}"
                )
        except subprocess.TimeoutExpired:
            return ToolResult(False, None, f"Execution timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, None, f"Execution error: {str(e)}")

    async def _run_shell(self, cmd: str, timeout: int) -> ToolResult:
        """Execute shell command (with safety restrictions)."""
        # Block dangerous commands
        blocked_patterns = [
            'rm -rf', 'sudo', 'chmod 777', '> /dev', 'mkfs', 'dd if=',
            'curl | sh', 'wget | sh', ':(){', 'fork bomb'
        ]
        cmd_lower = cmd.lower()
        if any(blocked in cmd_lower for blocked in blocked_patterns):
            return ToolResult(False, None, "Command blocked for safety")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = (result.stdout or result.stderr).strip() or "(no output)"
            return ToolResult(
                result.returncode == 0,
                {"output": output, "returncode": result.returncode},
                f"Exit {result.returncode}: {output[:300]}"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, None, f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, None, f"Shell error: {str(e)}")

    async def _explain_error(self, error: str) -> ToolResult:
        """Explain common Python/programming errors."""
        explanations = {
            "ModuleNotFoundError": "Package not installed. Fix: pip install <package_name>",
            "ImportError": "Import failed. Check package name, spelling, or circular imports.",
            "SyntaxError": "Invalid Python syntax. Check quotes, parentheses, colons, and indentation.",
            "TypeError": "Wrong type passed to function. Check argument types and counts.",
            "KeyError": "Dictionary key doesn't exist. Use dict.get('key') for safe access.",
            "IndexError": "List index out of range. Check array length with len() first.",
            "AttributeError": "Object doesn't have that attribute. Check object type and spelling.",
            "NameError": "Variable not defined. Check spelling or if it's defined before use.",
            "ValueError": "Invalid value for the operation. Check input data format.",
            "FileNotFoundError": "File doesn't exist. Check path and working directory.",
            "PermissionError": "No permission to access file. Check file permissions.",
            "ZeroDivisionError": "Can't divide by zero. Add a check before dividing.",
            "RecursionError": "Too many recursive calls. Add a base case or increase limit.",
            "ConnectionError": "Network connection failed. Check internet and server status.",
            "TimeoutError": "Operation timed out. Increase timeout or check network.",
        }

        # Find matching error type
        for error_type, explanation in explanations.items():
            if error_type.lower() in error.lower():
                return ToolResult(
                    True,
                    {"error_type": error_type, "explanation": explanation},
                    f"{error_type}: {explanation}"
                )

        # Generic help
        return ToolResult(
            True,
            {"raw_error": error},
            "Check the error message for line numbers. Common fixes: check imports, spelling, and syntax."
        )
