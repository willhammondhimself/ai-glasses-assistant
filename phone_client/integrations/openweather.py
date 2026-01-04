"""
OpenWeather API Integration for WHAM

Provides weather data for morning briefings and context-aware features.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - weather API calls will use mock data")


class WeatherCondition(Enum):
    """Weather condition categories."""
    CLEAR = "clear"
    CLOUDS = "clouds"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    MIST = "mist"
    FOG = "fog"
    HAZE = "haze"
    UNKNOWN = "unknown"


@dataclass
class Weather:
    """Current weather data."""
    temperature: float  # Celsius
    feels_like: float
    condition: WeatherCondition
    description: str
    humidity: int  # Percentage
    wind_speed: float  # m/s
    visibility: int  # meters
    icon: str  # OpenWeather icon code
    location: str
    timestamp: datetime

    @property
    def temperature_f(self) -> float:
        """Temperature in Fahrenheit."""
        return (self.temperature * 9/5) + 32

    @property
    def feels_like_f(self) -> float:
        """Feels like temperature in Fahrenheit."""
        return (self.feels_like * 9/5) + 32

    @property
    def wind_speed_mph(self) -> float:
        """Wind speed in mph."""
        return self.wind_speed * 2.237

    def to_speech(self, units: str = "metric") -> str:
        """Generate speech-friendly weather summary."""
        if units == "imperial":
            temp = f"{self.temperature_f:.0f} degrees Fahrenheit"
            wind = f"{self.wind_speed_mph:.0f} miles per hour"
        else:
            temp = f"{self.temperature:.0f} degrees Celsius"
            wind = f"{self.wind_speed:.0f} meters per second"

        return (
            f"Currently {self.description} in {self.location}. "
            f"Temperature is {temp}, feels like {self.feels_like_f if units == 'imperial' else self.feels_like:.0f}. "
            f"Humidity at {self.humidity} percent with winds at {wind}."
        )


@dataclass
class Forecast:
    """Weather forecast entry."""
    datetime: datetime
    temperature: float
    condition: WeatherCondition
    description: str
    precipitation_chance: float  # 0-1

    @property
    def temperature_f(self) -> float:
        """Temperature in Fahrenheit."""
        return (self.temperature * 9/5) + 32


class OpenWeatherIntegration:
    """
    OpenWeather API integration for weather data.

    Get your free API key at: https://openweathermap.org/api
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenWeather integration.

        Args:
            api_key: OpenWeather API key. If not provided, uses OPENWEATHER_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 minutes

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key) and HAS_REQUESTS

    def _parse_condition(self, weather_main: str) -> WeatherCondition:
        """Parse OpenWeather condition to enum."""
        mapping = {
            "clear": WeatherCondition.CLEAR,
            "clouds": WeatherCondition.CLOUDS,
            "rain": WeatherCondition.RAIN,
            "drizzle": WeatherCondition.DRIZZLE,
            "thunderstorm": WeatherCondition.THUNDERSTORM,
            "snow": WeatherCondition.SNOW,
            "mist": WeatherCondition.MIST,
            "fog": WeatherCondition.FOG,
            "haze": WeatherCondition.HAZE,
        }
        return mapping.get(weather_main.lower(), WeatherCondition.UNKNOWN)

    def get_weather(self, location: str) -> Weather:
        """
        Get current weather for a location.

        Args:
            location: City name (e.g., "San Francisco" or "London,UK")

        Returns:
            Weather object with current conditions
        """
        if not self.is_configured:
            logger.info("OpenWeather not configured, returning mock data")
            return self._mock_weather(location)

        # Check cache
        cache_key = f"weather:{location}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data

        try:
            response = requests.get(
                f"{self.BASE_URL}/weather",
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            weather = Weather(
                temperature=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                condition=self._parse_condition(data["weather"][0]["main"]),
                description=data["weather"][0]["description"],
                humidity=data["main"]["humidity"],
                wind_speed=data.get("wind", {}).get("speed", 0),
                visibility=data.get("visibility", 10000),
                icon=data["weather"][0]["icon"],
                location=data["name"],
                timestamp=datetime.now()
            )

            # Cache result
            self._cache[cache_key] = (datetime.now(), weather)
            return weather

        except Exception as e:
            logger.error(f"Failed to fetch weather: {e}")
            return self._mock_weather(location)

    def get_forecast(self, location: str, days: int = 3) -> List[Forecast]:
        """
        Get weather forecast for a location.

        Args:
            location: City name
            days: Number of days to forecast (max 5 for free tier)

        Returns:
            List of Forecast objects
        """
        if not self.is_configured:
            logger.info("OpenWeather not configured, returning mock forecast")
            return self._mock_forecast(location, days)

        try:
            response = requests.get(
                f"{self.BASE_URL}/forecast",
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": days * 8  # 8 entries per day (3-hour intervals)
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            forecasts = []
            for item in data["list"]:
                forecasts.append(Forecast(
                    datetime=datetime.fromtimestamp(item["dt"]),
                    temperature=item["main"]["temp"],
                    condition=self._parse_condition(item["weather"][0]["main"]),
                    description=item["weather"][0]["description"],
                    precipitation_chance=item.get("pop", 0)
                ))

            return forecasts

        except Exception as e:
            logger.error(f"Failed to fetch forecast: {e}")
            return self._mock_forecast(location, days)

    def get_weather_alert(self, location: str) -> Optional[str]:
        """
        Check for weather alerts/warnings.

        Args:
            location: City name

        Returns:
            Alert message if any, None otherwise
        """
        weather = self.get_weather(location)

        alerts = []

        # Temperature alerts
        if weather.temperature > 35:
            alerts.append("Extreme heat warning - stay hydrated")
        elif weather.temperature < 0:
            alerts.append("Freezing conditions - dress warmly")

        # Condition alerts
        if weather.condition == WeatherCondition.THUNDERSTORM:
            alerts.append("Thunderstorm warning - seek shelter")
        elif weather.condition == WeatherCondition.SNOW:
            alerts.append("Snow conditions - drive carefully")
        elif weather.condition in (WeatherCondition.FOG, WeatherCondition.MIST):
            alerts.append("Low visibility conditions")

        # Humidity alerts
        if weather.humidity > 85:
            alerts.append("High humidity - may feel uncomfortable")

        return "; ".join(alerts) if alerts else None

    def _mock_weather(self, location: str) -> Weather:
        """Generate mock weather data for testing."""
        return Weather(
            temperature=22.5,
            feels_like=23.0,
            condition=WeatherCondition.CLEAR,
            description="clear sky",
            humidity=45,
            wind_speed=3.5,
            visibility=10000,
            icon="01d",
            location=location,
            timestamp=datetime.now()
        )

    def _mock_forecast(self, location: str, days: int) -> List[Forecast]:
        """Generate mock forecast data for testing."""
        forecasts = []
        base_time = datetime.now()

        conditions = [
            (WeatherCondition.CLEAR, "clear sky", 0.0),
            (WeatherCondition.CLOUDS, "scattered clouds", 0.1),
            (WeatherCondition.CLOUDS, "overcast clouds", 0.2),
            (WeatherCondition.RAIN, "light rain", 0.6),
        ]

        for i in range(days * 8):
            hour_offset = i * 3
            cond = conditions[i % len(conditions)]
            forecasts.append(Forecast(
                datetime=base_time.replace(hour=(base_time.hour + hour_offset) % 24),
                temperature=20 + (i % 8) - 4,  # Vary between 16-24
                condition=cond[0],
                description=cond[1],
                precipitation_chance=cond[2]
            ))

        return forecasts


# Convenience function
def get_weather(location: str, api_key: Optional[str] = None) -> Weather:
    """
    Quick helper to get current weather.

    Args:
        location: City name (e.g., "San Francisco")
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Weather object
    """
    integration = OpenWeatherIntegration(api_key)
    return integration.get_weather(location)
