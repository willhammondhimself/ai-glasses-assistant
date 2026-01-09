"""HealthKit Service - Bridge to Apple HealthKit via phone client."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Daily health metrics from HealthKit."""
    date: datetime
    steps: int = 0
    calories_active: int = 0
    calories_total: int = 0
    distance_km: float = 0.0
    heart_rate_current: Optional[int] = None
    heart_rate_resting: Optional[int] = None
    sleep_hours: Optional[float] = None
    workouts_today: List[Dict] = field(default_factory=list)

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        parts = []

        if self.steps:
            parts.append(f"{self.steps:,} steps")

        if self.calories_active:
            parts.append(f"{self.calories_active} active calories")

        if self.distance_km > 0:
            miles = self.distance_km * 0.621371
            parts.append(f"{miles:.1f} miles")

        if self.workouts_today:
            parts.append(f"{len(self.workouts_today)} workout{'s' if len(self.workouts_today) > 1 else ''}")

        return ", ".join(parts) if parts else "No activity data yet today"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "steps": self.steps,
            "calories_active": self.calories_active,
            "calories_total": self.calories_total,
            "distance_km": self.distance_km,
            "heart_rate_current": self.heart_rate_current,
            "heart_rate_resting": self.heart_rate_resting,
            "sleep_hours": self.sleep_hours,
            "workouts": self.workouts_today
        }


@dataclass
class WorkoutSummary:
    """Summary of a single workout."""
    workout_type: str           # "Running", "Cycling", "Strength"
    start_time: datetime
    duration_minutes: int
    calories: int
    distance_km: Optional[float] = None
    heart_rate_avg: Optional[int] = None

    def to_voice_summary(self) -> str:
        """Create a voice-friendly summary."""
        parts = [f"{self.duration_minutes} minute {self.workout_type.lower()}"]

        if self.calories:
            parts.append(f"{self.calories} calories")

        if self.distance_km:
            miles = self.distance_km * 0.621371
            parts.append(f"{miles:.1f} miles")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.workout_type,
            "start_time": self.start_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "calories": self.calories,
            "distance_km": self.distance_km,
            "heart_rate_avg": self.heart_rate_avg
        }


class HealthKitService:
    """Bridge to HealthKit via phone_client REST API."""

    def __init__(self):
        self._api_url = os.getenv("HEALTHKIT_BRIDGE_URL", "http://localhost:5001/health")
        self._client: Optional[httpx.AsyncClient] = None

        # Fitness goals (customizable)
        self._goals = {
            "steps": int(os.getenv("HEALTH_GOAL_STEPS", "10000")),
            "active_calories": int(os.getenv("HEALTH_GOAL_CALORIES", "500")),
            "exercise_minutes": int(os.getenv("HEALTH_GOAL_EXERCISE", "30"))
        }

        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

    def is_configured(self) -> bool:
        """Check if HealthKit bridge is configured."""
        return bool(self._api_url)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                timeout=10.0
            )
        return self._client

    async def _fetch_data(self, endpoint: str) -> Optional[Dict]:
        """Fetch data from HealthKit bridge.

        Args:
            endpoint: API endpoint

        Returns:
            Response data or None
        """
        try:
            client = await self._get_client()
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"HealthKit bridge error: {e}")
            return None
        except Exception as e:
            logger.error(f"HealthKit fetch failed: {e}")
            return None

    async def get_today_metrics(self) -> HealthMetrics:
        """Get today's health metrics.

        Returns:
            HealthMetrics for today
        """
        data = await self._fetch_data("/today")

        if data:
            return HealthMetrics(
                date=datetime.now(),
                steps=data.get("steps", 0),
                calories_active=data.get("active_calories", 0),
                calories_total=data.get("total_calories", 0),
                distance_km=data.get("distance_km", 0.0),
                heart_rate_current=data.get("heart_rate"),
                heart_rate_resting=data.get("resting_heart_rate"),
                sleep_hours=data.get("sleep_hours"),
                workouts_today=data.get("workouts", [])
            )

        # Return mock data if bridge unavailable
        logger.info("HealthKit bridge unavailable, returning mock data")
        return self._get_mock_metrics()

    async def get_weekly_summary(self) -> Dict[str, Any]:
        """Get weekly health summary.

        Returns:
            Weekly aggregated metrics
        """
        data = await self._fetch_data("/weekly")

        if data:
            return data

        # Mock weekly data
        return {
            "total_steps": 45000,
            "avg_steps": 6400,
            "total_calories": 2800,
            "total_workouts": 4,
            "total_exercise_minutes": 180,
            "avg_sleep_hours": 7.2
        }

    async def get_workouts(self, days: int = 7) -> List[WorkoutSummary]:
        """Get recent workouts.

        Args:
            days: Number of days to look back

        Returns:
            List of WorkoutSummary
        """
        data = await self._fetch_data(f"/workouts?days={days}")

        if data and "workouts" in data:
            return [
                WorkoutSummary(
                    workout_type=w.get("type", "Unknown"),
                    start_time=datetime.fromisoformat(w["start_time"]) if w.get("start_time") else datetime.now(),
                    duration_minutes=w.get("duration", 0),
                    calories=w.get("calories", 0),
                    distance_km=w.get("distance_km"),
                    heart_rate_avg=w.get("avg_heart_rate")
                )
                for w in data["workouts"]
            ]

        # Mock workouts
        return [
            WorkoutSummary(
                workout_type="Running",
                start_time=datetime.now() - timedelta(hours=5),
                duration_minutes=30,
                calories=320,
                distance_km=5.0,
                heart_rate_avg=145
            )
        ]

    async def get_goal_progress(self) -> Dict[str, Any]:
        """Get progress toward daily goals.

        Returns:
            Dict with goal progress percentages
        """
        metrics = await self.get_today_metrics()

        steps_pct = min(100, int((metrics.steps / self._goals["steps"]) * 100))
        calories_pct = min(100, int((metrics.calories_active / self._goals["active_calories"]) * 100))

        # Calculate exercise minutes from workouts
        exercise_min = sum(w.get("duration", 0) for w in metrics.workouts_today) if isinstance(metrics.workouts_today, list) else 0
        exercise_pct = min(100, int((exercise_min / self._goals["exercise_minutes"]) * 100))

        return {
            "steps": {
                "current": metrics.steps,
                "goal": self._goals["steps"],
                "percent": steps_pct,
                "remaining": max(0, self._goals["steps"] - metrics.steps)
            },
            "calories": {
                "current": metrics.calories_active,
                "goal": self._goals["active_calories"],
                "percent": calories_pct,
                "remaining": max(0, self._goals["active_calories"] - metrics.calories_active)
            },
            "exercise": {
                "current": exercise_min,
                "goal": self._goals["exercise_minutes"],
                "percent": exercise_pct,
                "remaining": max(0, self._goals["exercise_minutes"] - exercise_min)
            }
        }

    async def get_heart_rate(self) -> Optional[int]:
        """Get current heart rate.

        Returns:
            Current heart rate or None
        """
        data = await self._fetch_data("/heart_rate")

        if data:
            return data.get("current")

        return None

    async def get_sleep_analysis(self) -> Dict[str, Any]:
        """Get last night's sleep analysis.

        Returns:
            Sleep data
        """
        data = await self._fetch_data("/sleep")

        if data:
            return data

        # Mock sleep data
        return {
            "total_hours": 7.2,
            "deep_sleep_hours": 1.5,
            "rem_hours": 1.8,
            "awake_count": 2,
            "sleep_quality": "good"
        }

    def _get_mock_metrics(self) -> HealthMetrics:
        """Get mock metrics when bridge unavailable."""
        return HealthMetrics(
            date=datetime.now(),
            steps=7500,
            calories_active=320,
            calories_total=1850,
            distance_km=5.8,
            heart_rate_current=72,
            heart_rate_resting=58,
            sleep_hours=7.2,
            workouts_today=[{
                "type": "Running",
                "duration": 30,
                "calories": 280
            }]
        )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_healthkit_service: Optional[HealthKitService] = None


def get_healthkit_service() -> HealthKitService:
    """Get or create global HealthKit service."""
    global _healthkit_service
    if _healthkit_service is None:
        _healthkit_service = HealthKitService()
    return _healthkit_service
