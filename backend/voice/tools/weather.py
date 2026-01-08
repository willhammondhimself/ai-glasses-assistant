"""Weather tool using wttr.in API (free, no key required)."""
import httpx
import logging
import re
from typing import Optional
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)

WTTR_BASE_URL = "https://wttr.in"


class WeatherTool(VoiceTool):
    """Get current weather and forecasts using wttr.in."""

    name = "weather"
    description = "Get current weather conditions and forecasts"

    keywords = [
        r"\bweather\b",
        r"\btemperature\b",
        r"\bforecast\b",
        r"\bhow\s+(hot|cold|warm)\b",
        r"\bis\s+it\s+(raining|snowing|sunny|cloudy)\b",
        r"\bwill\s+it\s+rain\b",
        r"\bwhat('s| is)\s+it\s+like\s+outside\b",
    ]

    priority = 10  # Higher priority for specific weather queries

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Get weather information.

        Args:
            query: The user's voice query

        Returns:
            VoiceToolResult with weather info
        """
        # Extract location if mentioned, otherwise use auto-detect
        location = self._extract_location(query)
        is_forecast = any(word in query.lower() for word in ["forecast", "tomorrow", "week", "will it"])

        try:
            if is_forecast:
                result = await self._get_forecast(location)
            else:
                result = await self._get_current_weather(location)

            return VoiceToolResult(
                success=True,
                message=result,
                data={"location": location, "is_forecast": is_forecast}
            )
        except Exception as e:
            import traceback
            logger.error(f"Weather lookup failed: {e}\n{traceback.format_exc()}")
            return VoiceToolResult(
                success=False,
                message=f"Sorry, I couldn't get the weather right now. Error: {str(e)}"
            )

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query."""
        # Common patterns: "weather in X", "weather for X", "X weather"
        patterns = [
            r"weather\s+(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$|today|tomorrow|this)",
            r"temperature\s+(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+weather",
            r"how\s+(?:hot|cold|warm)\s+(?:is\s+it\s+)?(?:in|at)\s+([a-zA-Z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Filter out common non-location words
                if location.lower() not in ["the", "today", "tomorrow", "now", "outside"]:
                    return location

        return None  # Use IP-based location

    async def _get_current_weather(self, location: Optional[str]) -> str:
        """Get current weather conditions."""
        # Use JSON format directly for all data
        url = f"{WTTR_BASE_URL}/{location or ''}?format=j1"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers={"Accept-Language": "en"})
            response.raise_for_status()

            data = response.json()
            current = data.get("current_condition", [{}])[0]

            feels_like = current.get("FeelsLikeF", "")
            desc = current.get("weatherDesc", [{}])[0].get("value", "")
            temp_f = current.get("temp_F", "")
            humidity = current.get("humidity", "")

            area = data.get("nearest_area", [{}])[0]
            city = area.get("areaName", [{}])[0].get("value", "your location")

            return f"Currently in {city}: {desc}, {temp_f}°F (feels like {feels_like}°F), {humidity}% humidity."

    async def _get_forecast(self, location: Optional[str]) -> str:
        """Get weather forecast."""
        url = f"{WTTR_BASE_URL}/{location or ''}?format=j1"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            weather = data.get("weather", [])

            if not weather:
                return "Couldn't get forecast data."

            area = data.get("nearest_area", [{}])[0]
            city = area.get("areaName", [{}])[0].get("value", "your location")

            # Get today and tomorrow
            forecasts = []
            for i, day in enumerate(weather[:3]):
                date_str = day.get("date", "")
                max_temp = day.get("maxtempF", "")
                min_temp = day.get("mintempF", "")
                hourly = day.get("hourly", [{}])

                # Get most common weather description
                desc = hourly[len(hourly)//2].get("weatherDesc", [{}])[0].get("value", "") if hourly else ""

                if i == 0:
                    day_name = "Today"
                elif i == 1:
                    day_name = "Tomorrow"
                else:
                    day_name = date_str

                forecasts.append(f"{day_name}: {desc}, high {max_temp}°F, low {min_temp}°F")

            return f"Forecast for {city}: " + ". ".join(forecasts[:2]) + "."
