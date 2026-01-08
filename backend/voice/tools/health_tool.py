"""Health check voice tool for system status queries."""
import httpx
import logging
import os
import psutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

# Backend API URL (local by default)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class HealthVoiceTool(VoiceTool):
    """Check system health and status via voice.

    Handles queries like:
    - "How's the system doing?"
    - "Check system health"
    - "Is everything working?"
    - "System status"
    - "Are all services running?"
    """

    name = "health"
    description = "Check system health and service status"

    keywords = [
        r"\bsystem\s+(?:health|status|check)\b",
        r"\bhealth\s+(?:check|status)\b",
        r"\bstatus\s+(?:check|report)\b",
        r"\bhow(?:'s| is)\s+(?:the\s+)?system\b",
        r"\bis\s+everything\s+(?:working|ok|okay|running)\b",
        r"\bcheck\s+(?:the\s+)?(?:system|server|services?)\b",
        r"\bare\s+(?:all\s+)?services?\s+(?:working|running|ok)\b",
        r"\bdiagnostic[s]?\b",
        r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:server\s+)?status\b",
        r"\brun\s+(?:a\s+)?health\s+check\b",
    ]

    priority = 8  # Medium-high priority for system queries

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute health check and return voice-friendly status.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with health status summary
        """
        try:
            # Determine what kind of health check to run
            check_type = self._determine_check_type(query)

            if check_type == "full":
                result = await self._full_health_check()
            elif check_type == "voice":
                result = await self._voice_health_check()
            elif check_type == "quick":
                result = await self._quick_health_check()
            else:
                result = await self._full_health_check()

            return VoiceToolResult(
                success=True,
                message=result["message"],
                data=result["data"]
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return VoiceToolResult(
                success=False,
                message=f"Couldn't complete the health check. Error: {str(e)}"
            )

    def _determine_check_type(self, query: str) -> str:
        """Determine what type of health check to run based on query.

        Args:
            query: User's voice query

        Returns:
            Check type: "full", "voice", "quick"
        """
        query_lower = query.lower()

        # Voice-specific checks
        if any(word in query_lower for word in ["voice", "livekit", "audio", "speak"]):
            return "voice"

        # Quick status check
        if any(phrase in query_lower for phrase in ["quick", "brief", "fast"]):
            return "quick"

        # Default to full check
        return "full"

    async def _full_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check.

        Returns:
            Dict with message and data
        """
        health_data = {}
        issues = []
        all_ok = True

        # 1. Check backend API health
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BACKEND_URL}/health")
                if response.status_code == 200:
                    health_data["api"] = response.json()
                else:
                    issues.append("API server returned error")
                    all_ok = False
        except Exception as e:
            issues.append(f"API server unreachable")
            all_ok = False
            health_data["api"] = {"error": str(e)}

        # 2. Check system resources
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            health_data["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 1),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 1),
            }

            # Check for resource warnings
            if cpu_percent > 80:
                issues.append(f"CPU usage high at {cpu_percent}%")
            if memory.percent > 85:
                issues.append(f"Memory usage high at {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"Disk usage critical at {disk.percent}%")

        except Exception as e:
            logger.warning(f"System metrics unavailable: {e}")
            health_data["system"] = {"error": str(e)}

        # 3. Check voice agent status
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BACKEND_URL}/voice/agent-status")
                if response.status_code == 200:
                    voice_status = response.json()
                    health_data["voice_agent"] = voice_status
                else:
                    health_data["voice_agent"] = {"state": "unknown"}
        except Exception as e:
            health_data["voice_agent"] = {"error": str(e)}

        # Build voice-friendly message
        message = self._build_health_message(health_data, issues, all_ok)

        return {
            "message": message,
            "data": {
                "healthy": all_ok and len(issues) == 0,
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat(),
                **health_data
            }
        }

    async def _voice_health_check(self) -> Dict[str, Any]:
        """Check voice-specific components.

        Returns:
            Dict with message and data
        """
        voice_data = {}
        issues = []

        # Check voice status endpoint
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BACKEND_URL}/voice/status")
                if response.status_code == 200:
                    voice_data = response.json()
                else:
                    issues.append("Voice status endpoint error")
        except Exception as e:
            issues.append(f"Voice API unreachable: {str(e)}")
            voice_data["error"] = str(e)

        # Build message
        if voice_data.get("livekit_sdk") == "installed":
            sdk_status = "installed"
        else:
            sdk_status = "not installed"
            issues.append("LiveKit SDK not installed")

        api_key_set = voice_data.get("api_key_set", False)
        api_secret_set = voice_data.get("api_secret_set", False)

        if not api_key_set or not api_secret_set:
            issues.append("LiveKit credentials not fully configured")

        # Get agent status
        agent_status = voice_data.get("agent_status", {})
        agent_state = agent_status.get("state", "unknown")

        if issues:
            message = f"Voice system has issues. LiveKit SDK is {sdk_status}. {'. '.join(issues)}."
        else:
            message = f"Voice system is healthy. LiveKit SDK is {sdk_status} and credentials are configured. Agent is currently {agent_state}."

        return {
            "message": message,
            "data": {
                "healthy": len(issues) == 0,
                "issues": issues,
                "voice_status": voice_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _quick_health_check(self) -> Dict[str, Any]:
        """Run a quick health check.

        Returns:
            Dict with message and data
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{BACKEND_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    version = data.get("version", "unknown")

                    return {
                        "message": f"System is {status}, running version {version}.",
                        "data": {
                            "healthy": status == "healthy",
                            "status": status,
                            "version": version,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                else:
                    return {
                        "message": f"System returned status code {response.status_code}. May need attention.",
                        "data": {
                            "healthy": False,
                            "status_code": response.status_code,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
        except Exception as e:
            return {
                "message": f"Couldn't reach the server. It may be down or unreachable.",
                "data": {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    def _build_health_message(
        self,
        health_data: Dict[str, Any],
        issues: List[str],
        all_ok: bool
    ) -> str:
        """Build a voice-friendly health status message.

        Args:
            health_data: Collected health data
            issues: List of issues found
            all_ok: Whether all checks passed

        Returns:
            Voice-friendly message string
        """
        parts = []

        # Overall status
        if all_ok and len(issues) == 0:
            parts.append("All systems are healthy and running normally.")
        elif len(issues) > 0:
            parts.append(f"Found {len(issues)} issue{'s' if len(issues) > 1 else ''}.")

        # API status
        api_data = health_data.get("api", {})
        if "error" not in api_data:
            version = api_data.get("version", "unknown")
            parts.append(f"API server is healthy, version {version}.")

            # Check engines
            engines = api_data.get("engines", {})
            available_engines = sum(1 for v in engines.values() if v == "available" or isinstance(v, dict))
            if available_engines > 0:
                parts.append(f"{available_engines} processing engines available.")

        # System resources
        system_data = health_data.get("system", {})
        if "error" not in system_data:
            cpu = system_data.get("cpu_percent", 0)
            memory = system_data.get("memory_percent", 0)
            disk_free = system_data.get("disk_free_gb", 0)

            # Only mention if there are concerns
            if cpu > 50 or memory > 70:
                parts.append(f"CPU at {cpu}%, memory at {memory}%.")
            if disk_free < 10:
                parts.append(f"Only {disk_free} GB disk space remaining.")

        # Voice agent status
        voice_data = health_data.get("voice_agent", {})
        if "error" not in voice_data:
            state = voice_data.get("state", "unknown")
            if state == "disconnected":
                parts.append("Voice agent is ready but not connected.")
            elif state != "unknown":
                parts.append(f"Voice agent is {state}.")

        # Summarize issues
        if issues:
            parts.append("Issues: " + ". ".join(issues) + ".")

        return " ".join(parts)
