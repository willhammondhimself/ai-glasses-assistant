"""Home Assistant Service - Smart home control via Home Assistant API."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HomeDevice:
    """A smart home device."""
    entity_id: str          # "light.living_room"
    name: str               # "Living Room Light"
    domain: str             # "light", "switch", "climate", "lock"
    state: str              # "on", "off", "72", "locked"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        if self.domain == "light":
            brightness = self.attributes.get("brightness", 0)
            if self.state == "on" and brightness:
                pct = int((brightness / 255) * 100)
                return f"{self.name} is on at {pct}%"
            return f"{self.name} is {self.state}"

        if self.domain == "climate":
            current = self.attributes.get("current_temperature")
            target = self.attributes.get("temperature")
            if current and target:
                return f"{self.name} is set to {target}°, currently {current}°"
            return f"{self.name} is {self.state}"

        if self.domain == "lock":
            return f"{self.name} is {self.state}"

        if self.domain == "cover":
            return f"{self.name} is {self.state}"

        return f"{self.name} is {self.state}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "domain": self.domain,
            "state": self.state,
            "attributes": self.attributes
        }


class HomeAssistantService:
    """Home Assistant REST API integration."""

    def __init__(self):
        self.base_url = os.getenv("HOME_ASSISTANT_URL", "http://localhost:8123")
        self.token = os.getenv("HOME_ASSISTANT_TOKEN")

        # Device cache
        self._device_cache: Dict[str, HomeDevice] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    def is_configured(self) -> bool:
        """Check if Home Assistant is configured."""
        return bool(self.token and self.base_url)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
        return self._client

    async def _refresh_cache(self, force: bool = False):
        """Refresh device cache if stale."""
        if not force and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_ttl:
                return

        try:
            client = await self._get_client()
            response = await client.get("/api/states")
            response.raise_for_status()

            states = response.json()
            self._device_cache.clear()

            for state in states:
                entity_id = state.get("entity_id", "")
                domain = entity_id.split(".")[0] if "." in entity_id else ""

                # Only cache controllable devices
                if domain in ["light", "switch", "climate", "lock", "cover", "fan", "scene"]:
                    device = HomeDevice(
                        entity_id=entity_id,
                        name=state.get("attributes", {}).get("friendly_name", entity_id),
                        domain=domain,
                        state=state.get("state", "unknown"),
                        attributes=state.get("attributes", {})
                    )
                    self._device_cache[entity_id] = device

            self._cache_time = datetime.now()
            logger.info(f"Cached {len(self._device_cache)} Home Assistant devices")

        except Exception as e:
            logger.error(f"Failed to refresh Home Assistant cache: {e}")

    async def get_devices(self, domain: Optional[str] = None) -> List[HomeDevice]:
        """Get all devices, optionally filtered by domain.

        Args:
            domain: Optional domain filter (light, switch, climate, etc.)

        Returns:
            List of HomeDevice objects
        """
        await self._refresh_cache()

        if domain:
            return [d for d in self._device_cache.values() if d.domain == domain]
        return list(self._device_cache.values())

    async def get_device(self, entity_id: str) -> Optional[HomeDevice]:
        """Get a specific device by entity ID.

        Args:
            entity_id: Home Assistant entity ID

        Returns:
            HomeDevice or None
        """
        await self._refresh_cache()
        return self._device_cache.get(entity_id)

    async def find_device_by_name(self, name: str, domain: Optional[str] = None) -> Optional[HomeDevice]:
        """Find a device by fuzzy name matching.

        Args:
            name: Device name to search for
            domain: Optional domain filter

        Returns:
            Best matching HomeDevice or None
        """
        await self._refresh_cache()

        best_match = None
        best_score = 0.0

        name_lower = name.lower()
        devices = self._device_cache.values()
        if domain:
            devices = [d for d in devices if d.domain == domain]

        for device in devices:
            # Check exact match first
            device_name_lower = device.name.lower()
            if name_lower in device_name_lower or device_name_lower in name_lower:
                return device

            # Fuzzy match
            score = SequenceMatcher(None, name_lower, device_name_lower).ratio()
            if score > best_score and score > 0.5:
                best_score = score
                best_match = device

        return best_match

    async def turn_on(self, entity_id: str, **kwargs) -> bool:
        """Turn on a device.

        Args:
            entity_id: Device entity ID
            **kwargs: Additional service data (brightness, color_temp, etc.)

        Returns:
            True if successful
        """
        domain = entity_id.split(".")[0]
        return await self._call_service(domain, "turn_on", entity_id, **kwargs)

    async def turn_off(self, entity_id: str) -> bool:
        """Turn off a device.

        Args:
            entity_id: Device entity ID

        Returns:
            True if successful
        """
        domain = entity_id.split(".")[0]
        return await self._call_service(domain, "turn_off", entity_id)

    async def toggle(self, entity_id: str) -> bool:
        """Toggle a device.

        Args:
            entity_id: Device entity ID

        Returns:
            True if successful
        """
        domain = entity_id.split(".")[0]
        return await self._call_service(domain, "toggle", entity_id)

    async def set_brightness(self, entity_id: str, brightness_pct: int) -> bool:
        """Set light brightness.

        Args:
            entity_id: Light entity ID
            brightness_pct: Brightness percentage (0-100)

        Returns:
            True if successful
        """
        # Convert percentage to 0-255 range
        brightness = int((brightness_pct / 100) * 255)
        return await self.turn_on(entity_id, brightness=brightness)

    async def set_climate(self, entity_id: str, temperature: float, hvac_mode: Optional[str] = None) -> bool:
        """Set thermostat temperature.

        Args:
            entity_id: Climate entity ID
            temperature: Target temperature
            hvac_mode: Optional mode (heat, cool, auto)

        Returns:
            True if successful
        """
        data = {"temperature": temperature}
        if hvac_mode:
            data["hvac_mode"] = hvac_mode

        return await self._call_service("climate", "set_temperature", entity_id, **data)

    async def lock(self, entity_id: str) -> bool:
        """Lock a lock.

        Args:
            entity_id: Lock entity ID

        Returns:
            True if successful
        """
        return await self._call_service("lock", "lock", entity_id)

    async def unlock(self, entity_id: str) -> bool:
        """Unlock a lock.

        Args:
            entity_id: Lock entity ID

        Returns:
            True if successful
        """
        return await self._call_service("lock", "unlock", entity_id)

    async def open_cover(self, entity_id: str) -> bool:
        """Open a cover (garage door, blinds, etc.).

        Args:
            entity_id: Cover entity ID

        Returns:
            True if successful
        """
        return await self._call_service("cover", "open_cover", entity_id)

    async def close_cover(self, entity_id: str) -> bool:
        """Close a cover.

        Args:
            entity_id: Cover entity ID

        Returns:
            True if successful
        """
        return await self._call_service("cover", "close_cover", entity_id)

    async def execute_scene(self, scene_name: str) -> bool:
        """Execute a scene by name.

        Args:
            scene_name: Scene name to activate

        Returns:
            True if successful
        """
        # Find scene entity
        await self._refresh_cache()

        scene_name_lower = scene_name.lower()
        for entity_id, device in self._device_cache.items():
            if device.domain == "scene":
                if scene_name_lower in device.name.lower():
                    return await self._call_service("scene", "turn_on", entity_id)

        logger.warning(f"Scene not found: {scene_name}")
        return False

    async def get_all_lights(self) -> List[HomeDevice]:
        """Get all lights.

        Returns:
            List of light devices
        """
        return await self.get_devices(domain="light")

    async def get_climate_status(self) -> List[HomeDevice]:
        """Get all thermostats/climate devices.

        Returns:
            List of climate devices
        """
        return await self.get_devices(domain="climate")

    async def _call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        **data
    ) -> bool:
        """Call a Home Assistant service.

        Args:
            domain: Service domain
            service: Service name
            entity_id: Target entity
            **data: Additional service data

        Returns:
            True if successful
        """
        if not self.is_configured():
            logger.warning("Home Assistant not configured")
            return False

        try:
            client = await self._get_client()

            payload = {"entity_id": entity_id, **data}

            response = await client.post(
                f"/api/services/{domain}/{service}",
                json=payload
            )
            response.raise_for_status()

            # Refresh cache after state change
            self._cache_time = None

            logger.info(f"Called {domain}.{service} on {entity_id}")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"Home Assistant API error: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Home Assistant service call failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_home_assistant: Optional[HomeAssistantService] = None


def get_home_assistant() -> HomeAssistantService:
    """Get or create global Home Assistant service."""
    global _home_assistant
    if _home_assistant is None:
        _home_assistant = HomeAssistantService()
    return _home_assistant
