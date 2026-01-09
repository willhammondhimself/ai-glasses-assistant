"""Smart Home voice tool - control lights, thermostat, locks via Home Assistant."""
import logging
import re
from typing import Optional, Tuple
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class SmartHomeVoiceTool(VoiceTool):
    """Voice-controlled smart home via Home Assistant."""

    name = "smart_home"
    description = "Control smart home devices - lights, thermostat, locks, scenes"

    keywords = [
        r"\b(turn|switch)\s+(on|off)\b",
        r"\blights?\s+(on|off)\b",
        r"\b(dim|brighten)\b",
        r"\btemperature\s+(to\s+)?\d+\b",
        r"\bthermostat\b",
        r"\b(set|change)\s+(the\s+)?(temp|temperature)\b",
        r"\block\b.*\b(door|front|back)\b",
        r"\bunlock\b",
        r"\b(garage|door)\s+(open|close|status)\b",
        r"\bhome\s+(status|devices)\b",
        r"\bscene\b",
        r"\b(all\s+)?lights\s+(on|off)\b",
    ]

    priority = 9

    def __init__(self):
        self._home_service = None

    def _get_service(self):
        """Get Home Assistant service (lazy load)."""
        if self._home_service is None:
            from backend.services.home_assistant_service import get_home_assistant
            self._home_service = get_home_assistant()
        return self._home_service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute smart home command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with action result
        """
        query_lower = query.lower()
        service = self._get_service()

        if not service.is_configured():
            return VoiceToolResult(
                success=False,
                message="Smart home is not configured. Please set up Home Assistant.",
                data={"error": "not_configured"}
            )

        try:
            # Light controls
            if self._is_light_command(query_lower):
                return await self._handle_light(service, query_lower)

            # Climate/thermostat
            if self._is_climate_command(query_lower):
                return await self._handle_climate(service, query_lower)

            # Lock controls
            if self._is_lock_command(query_lower):
                return await self._handle_lock(service, query_lower)

            # Cover/garage controls
            if self._is_cover_command(query_lower):
                return await self._handle_cover(service, query_lower)

            # Scene activation
            if self._is_scene_command(query_lower):
                return await self._handle_scene(service, query_lower)

            # Status query
            if self._is_status_query(query_lower):
                return await self._handle_status(service, query_lower)

            return VoiceToolResult(
                success=False,
                message="I'm not sure what you want to control. Try 'turn off the lights' or 'set temperature to 72'.",
                data={"error": "unclear_command"}
            )

        except Exception as e:
            logger.error(f"Smart home error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble controlling your smart home.",
                data={"error": str(e)}
            )

    def _is_light_command(self, query: str) -> bool:
        patterns = ["light", "lamp", "dim", "brighten", "brightness"]
        return any(p in query for p in patterns)

    def _is_climate_command(self, query: str) -> bool:
        patterns = ["thermostat", "temperature", "temp", "heat", "cool", "ac", "air"]
        return any(p in query for p in patterns)

    def _is_lock_command(self, query: str) -> bool:
        patterns = ["lock", "unlock"]
        return any(p in query for p in patterns)

    def _is_cover_command(self, query: str) -> bool:
        patterns = ["garage", "blinds", "shades", "curtain"]
        return any(p in query for p in patterns)

    def _is_scene_command(self, query: str) -> bool:
        return "scene" in query or "activate" in query

    def _is_status_query(self, query: str) -> bool:
        patterns = ["status", "what's", "is the", "are the", "check"]
        return any(p in query for p in patterns)

    def _extract_room(self, query: str) -> Optional[str]:
        """Extract room name from query."""
        rooms = [
            "living room", "bedroom", "kitchen", "bathroom", "office",
            "dining room", "hallway", "garage", "basement", "attic",
            "master", "guest", "front", "back", "porch", "patio"
        ]
        for room in rooms:
            if room in query:
                return room
        return None

    def _extract_brightness(self, query: str) -> Optional[int]:
        """Extract brightness percentage from query."""
        # Look for "to X%" or "X percent" or just "X%"
        match = re.search(r"(\d+)\s*%|\bto\s+(\d+)\b", query)
        if match:
            value = int(match.group(1) or match.group(2))
            return min(100, max(0, value))

        # Look for descriptive brightness
        if "dim" in query:
            return 30
        if "low" in query:
            return 20
        if "half" in query:
            return 50
        if "bright" in query or "full" in query:
            return 100

        return None

    def _extract_temperature(self, query: str) -> Optional[int]:
        """Extract temperature from query."""
        match = re.search(r"(\d+)\s*(?:degrees?|°)?", query)
        if match:
            temp = int(match.group(1))
            # Sanity check
            if 50 <= temp <= 90:
                return temp
        return None

    async def _handle_light(self, service, query: str) -> VoiceToolResult:
        """Handle light control commands."""
        room = self._extract_room(query)
        brightness = self._extract_brightness(query)

        # Determine action
        if "off" in query:
            action = "off"
        elif "on" in query or "turn on" in query:
            action = "on"
        elif brightness is not None:
            action = "dim"
        else:
            action = "on"  # Default

        # Find the light
        if room:
            device = await service.find_device_by_name(room, domain="light")
        else:
            # Default to first light or "all lights"
            lights = await service.get_all_lights()
            device = lights[0] if lights else None

        if not device:
            return VoiceToolResult(
                success=False,
                message=f"I couldn't find a light{' in ' + room if room else ''}.",
                data={"error": "device_not_found"}
            )

        # Execute action
        if action == "off":
            success = await service.turn_off(device.entity_id)
            message = f"Turned off {device.name}."
        elif action == "dim" and brightness is not None:
            success = await service.set_brightness(device.entity_id, brightness)
            message = f"Set {device.name} to {brightness}%."
        else:
            if brightness is not None:
                success = await service.set_brightness(device.entity_id, brightness)
                message = f"Turned on {device.name} at {brightness}%."
            else:
                success = await service.turn_on(device.entity_id)
                message = f"Turned on {device.name}."

        return VoiceToolResult(
            success=success,
            message=message if success else f"Failed to control {device.name}.",
            data={
                "action": action,
                "device": device.to_dict(),
                "brightness": brightness
            }
        )

    async def _handle_climate(self, service, query: str) -> VoiceToolResult:
        """Handle thermostat commands."""
        temperature = self._extract_temperature(query)

        # Get climate devices
        devices = await service.get_climate_status()
        if not devices:
            return VoiceToolResult(
                success=False,
                message="I couldn't find a thermostat.",
                data={"error": "device_not_found"}
            )

        device = devices[0]

        # Status query
        if "what" in query or "status" in query or temperature is None:
            return VoiceToolResult(
                success=True,
                message=device.to_voice_summary(),
                data={"device": device.to_dict()}
            )

        # Set temperature
        success = await service.set_climate(device.entity_id, temperature)

        if success:
            message = f"Set {device.name} to {temperature} degrees."
        else:
            message = f"Failed to set temperature."

        return VoiceToolResult(
            success=success,
            message=message,
            data={
                "action": "set_temperature",
                "device": device.to_dict(),
                "temperature": temperature
            }
        )

    async def _handle_lock(self, service, query: str) -> VoiceToolResult:
        """Handle lock commands."""
        room = self._extract_room(query)

        # Find lock
        if room:
            device = await service.find_device_by_name(room, domain="lock")
        else:
            locks = await service.get_devices(domain="lock")
            device = locks[0] if locks else None

        if not device:
            return VoiceToolResult(
                success=False,
                message="I couldn't find a lock.",
                data={"error": "device_not_found"}
            )

        # Determine action
        if "unlock" in query:
            success = await service.unlock(device.entity_id)
            action = "unlock"
            message = f"Unlocked {device.name}."
        else:
            success = await service.lock(device.entity_id)
            action = "lock"
            message = f"Locked {device.name}."

        return VoiceToolResult(
            success=success,
            message=message if success else f"Failed to {action} {device.name}.",
            data={
                "action": action,
                "device": device.to_dict()
            }
        )

    async def _handle_cover(self, service, query: str) -> VoiceToolResult:
        """Handle cover/garage commands."""
        # Find cover device
        covers = await service.get_devices(domain="cover")
        if not covers:
            return VoiceToolResult(
                success=False,
                message="I couldn't find a garage door or cover.",
                data={"error": "device_not_found"}
            )

        device = covers[0]

        # Status query
        if "status" in query or "is" in query:
            return VoiceToolResult(
                success=True,
                message=device.to_voice_summary(),
                data={"device": device.to_dict()}
            )

        # Determine action
        if "open" in query:
            success = await service.open_cover(device.entity_id)
            action = "open"
            message = f"Opening {device.name}."
        else:
            success = await service.close_cover(device.entity_id)
            action = "close"
            message = f"Closing {device.name}."

        return VoiceToolResult(
            success=success,
            message=message if success else f"Failed to {action} {device.name}.",
            data={
                "action": action,
                "device": device.to_dict()
            }
        )

    async def _handle_scene(self, service, query: str) -> VoiceToolResult:
        """Handle scene activation."""
        # Extract scene name
        match = re.search(r"scene\s+(\w+)", query)
        if not match:
            # Try common scene names
            scene_keywords = ["movie", "dinner", "party", "relax", "night", "morning", "away", "home"]
            for kw in scene_keywords:
                if kw in query:
                    match = kw
                    break

        scene_name = match.group(1) if hasattr(match, 'group') else match if match else None

        if not scene_name:
            return VoiceToolResult(
                success=False,
                message="Which scene would you like to activate?",
                data={"error": "no_scene_specified"}
            )

        success = await service.execute_scene(scene_name)

        return VoiceToolResult(
            success=success,
            message=f"Activated {scene_name} scene." if success else f"Couldn't find scene '{scene_name}'.",
            data={
                "action": "scene",
                "scene": scene_name
            }
        )

    async def _handle_status(self, service, query: str) -> VoiceToolResult:
        """Handle status queries."""
        # Determine what to check
        if "light" in query:
            devices = await service.get_all_lights()
            on_count = sum(1 for d in devices if d.state == "on")
            message = f"You have {len(devices)} lights. {on_count} are on."
        elif "thermostat" in query or "temperature" in query:
            devices = await service.get_climate_status()
            if devices:
                message = devices[0].to_voice_summary()
            else:
                message = "No thermostat found."
        elif "lock" in query:
            devices = await service.get_devices(domain="lock")
            if devices:
                locked = sum(1 for d in devices if d.state == "locked")
                message = f"{locked} of {len(devices)} locks are locked."
            else:
                message = "No locks found."
        else:
            # General status
            all_devices = await service.get_devices()
            message = f"You have {len(all_devices)} smart home devices."

        return VoiceToolResult(
            success=True,
            message=message,
            data={"action": "status"}
        )
