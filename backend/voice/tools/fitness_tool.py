"""Fitness Coach voice tool - health metrics and workout tracking via HealthKit."""
import logging
from .base import VoiceTool, VoiceToolResult

logger = logging.getLogger(__name__)


class FitnessVoiceTool(VoiceTool):
    """Voice-controlled fitness tracking and health metrics."""

    name = "fitness"
    description = "Track fitness - steps, calories, workouts, sleep, heart rate"

    keywords = [
        r"\bsteps?\b",
        r"\bcalories\b",
        r"\bworkout\b",
        r"\bexercise\b",
        r"\bheart\s+rate\b",
        r"\bsleep\b",
        r"\bfitness\s+(stats?|summary|status)\b",
        r"\bhow\s+(many|much)\s+(did\s+)?i\s+(walk|run|burn)\b",
        r"\bgoals?\b.*\b(fitness|health|step)\b",
        r"\bactivity\b",
        r"\bdistance\b",
        r"\bweekly\s+(summary|stats)\b",
    ]

    priority = 7

    def __init__(self):
        self._health_service = None

    def _get_service(self):
        """Get HealthKit service (lazy load)."""
        if self._health_service is None:
            from backend.services.healthkit_service import get_healthkit_service
            self._health_service = get_healthkit_service()
        return self._health_service

    async def execute(self, query: str, **kwargs) -> VoiceToolResult:
        """Execute fitness command.

        Args:
            query: The user's voice query
            **kwargs: Additional context

        Returns:
            VoiceToolResult with fitness info
        """
        query_lower = query.lower()
        service = self._get_service()

        try:
            # Steps query
            if self._is_steps_query(query_lower):
                return await self._handle_steps(service)

            # Calories query
            if self._is_calories_query(query_lower):
                return await self._handle_calories(service)

            # Workout query
            if self._is_workout_query(query_lower):
                return await self._handle_workouts(service)

            # Heart rate query
            if self._is_heart_rate_query(query_lower):
                return await self._handle_heart_rate(service)

            # Sleep query
            if self._is_sleep_query(query_lower):
                return await self._handle_sleep(service)

            # Goals query
            if self._is_goals_query(query_lower):
                return await self._handle_goals(service)

            # Weekly summary
            if self._is_weekly_query(query_lower):
                return await self._handle_weekly(service)

            # Default: daily summary
            return await self._handle_summary(service)

        except Exception as e:
            logger.error(f"Fitness tool error: {e}")
            return VoiceToolResult(
                success=False,
                message="Sorry, I had trouble getting your fitness data.",
                data={"error": str(e)}
            )

    def _is_steps_query(self, query: str) -> bool:
        patterns = ["step", "walked", "walking"]
        return any(p in query for p in patterns)

    def _is_calories_query(self, query: str) -> bool:
        patterns = ["calorie", "burned", "burn"]
        return any(p in query for p in patterns)

    def _is_workout_query(self, query: str) -> bool:
        patterns = ["workout", "exercise", "training", "run", "running"]
        return any(p in query for p in patterns)

    def _is_heart_rate_query(self, query: str) -> bool:
        patterns = ["heart rate", "heart-rate", "bpm", "pulse"]
        return any(p in query for p in patterns)

    def _is_sleep_query(self, query: str) -> bool:
        patterns = ["sleep", "slept", "rest"]
        return any(p in query for p in patterns)

    def _is_goals_query(self, query: str) -> bool:
        patterns = ["goal", "target", "progress"]
        return any(p in query for p in patterns)

    def _is_weekly_query(self, query: str) -> bool:
        patterns = ["week", "weekly", "this week", "past week"]
        return any(p in query for p in patterns)

    async def _handle_steps(self, service) -> VoiceToolResult:
        """Handle steps query."""
        metrics = await service.get_today_metrics()
        goals = await service.get_goal_progress()

        steps = metrics.steps
        goal = goals["steps"]["goal"]
        remaining = goals["steps"]["remaining"]
        percent = goals["steps"]["percent"]

        if percent >= 100:
            message = f"Great job! You've hit your step goal with {steps:,} steps today."
        elif percent >= 75:
            message = f"You've taken {steps:,} steps. Just {remaining:,} more to hit your goal!"
        else:
            message = f"You've taken {steps:,} steps today. That's {percent}% of your {goal:,} step goal."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "steps",
                "value": steps,
                "goal": goal,
                "percent": percent
            }
        )

    async def _handle_calories(self, service) -> VoiceToolResult:
        """Handle calories query."""
        metrics = await service.get_today_metrics()
        goals = await service.get_goal_progress()

        active = metrics.calories_active
        total = metrics.calories_total
        goal = goals["calories"]["goal"]
        percent = goals["calories"]["percent"]

        message = f"You've burned {active} active calories today, {total} total."
        if percent >= 100:
            message += " You've hit your calorie goal!"
        elif percent >= 50:
            message += f" That's {percent}% of your goal."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "calories",
                "active": active,
                "total": total,
                "goal": goal,
                "percent": percent
            }
        )

    async def _handle_workouts(self, service) -> VoiceToolResult:
        """Handle workout query."""
        workouts = await service.get_workouts(days=7)

        if not workouts:
            return VoiceToolResult(
                success=True,
                message="No workouts recorded in the past week. Time to get moving!",
                data={"metric": "workouts", "count": 0}
            )

        # Today's workouts
        today_workouts = [w for w in workouts if w.start_time.date() == workouts[0].start_time.date()]

        if today_workouts:
            latest = today_workouts[0]
            message = f"You did a {latest.to_voice_summary()} today."
            if len(today_workouts) > 1:
                message += f" That's {len(today_workouts)} workouts today."
        else:
            message = f"No workouts today. You had {len(workouts)} workouts this week."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "workouts",
                "today": [w.to_dict() for w in today_workouts],
                "week": [w.to_dict() for w in workouts]
            }
        )

    async def _handle_heart_rate(self, service) -> VoiceToolResult:
        """Handle heart rate query."""
        hr = await service.get_heart_rate()
        metrics = await service.get_today_metrics()

        if hr:
            resting = metrics.heart_rate_resting
            message = f"Your current heart rate is {hr} BPM."
            if resting:
                message += f" Your resting heart rate is {resting} BPM."
        else:
            message = "I couldn't get your current heart rate. Make sure your Apple Watch is connected."

        return VoiceToolResult(
            success=hr is not None,
            message=message,
            data={
                "metric": "heart_rate",
                "current": hr,
                "resting": metrics.heart_rate_resting
            }
        )

    async def _handle_sleep(self, service) -> VoiceToolResult:
        """Handle sleep query."""
        sleep = await service.get_sleep_analysis()

        total = sleep.get("total_hours", 0)
        deep = sleep.get("deep_sleep_hours", 0)
        quality = sleep.get("sleep_quality", "unknown")

        if total > 0:
            message = f"You slept {total:.1f} hours last night"
            if deep:
                message += f" with {deep:.1f} hours of deep sleep"
            message += f". Sleep quality: {quality}."
        else:
            message = "No sleep data available for last night."

        return VoiceToolResult(
            success=total > 0,
            message=message,
            data={
                "metric": "sleep",
                "total_hours": total,
                "deep_hours": deep,
                "quality": quality
            }
        )

    async def _handle_goals(self, service) -> VoiceToolResult:
        """Handle goals/progress query."""
        goals = await service.get_goal_progress()

        steps_pct = goals["steps"]["percent"]
        cal_pct = goals["calories"]["percent"]
        exercise_pct = goals["exercise"]["percent"]

        # Build message based on progress
        parts = []

        if steps_pct >= 100:
            parts.append("Steps goal: done!")
        else:
            parts.append(f"Steps: {steps_pct}%")

        if cal_pct >= 100:
            parts.append("Calories: done!")
        else:
            parts.append(f"Calories: {cal_pct}%")

        if exercise_pct >= 100:
            parts.append("Exercise: done!")
        else:
            parts.append(f"Exercise: {exercise_pct}%")

        avg_pct = (steps_pct + cal_pct + exercise_pct) // 3

        if avg_pct >= 100:
            message = "You've crushed all your goals today! " + ", ".join(parts)
        elif avg_pct >= 75:
            message = "Great progress! " + ", ".join(parts)
        else:
            message = "Goal progress: " + ", ".join(parts)

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "goals",
                "steps": goals["steps"],
                "calories": goals["calories"],
                "exercise": goals["exercise"],
                "average_percent": avg_pct
            }
        )

    async def _handle_weekly(self, service) -> VoiceToolResult:
        """Handle weekly summary query."""
        weekly = await service.get_weekly_summary()

        total_steps = weekly.get("total_steps", 0)
        avg_steps = weekly.get("avg_steps", 0)
        workouts = weekly.get("total_workouts", 0)
        exercise_min = weekly.get("total_exercise_minutes", 0)

        message = (
            f"This week: {total_steps:,} total steps, averaging {avg_steps:,} per day. "
            f"You did {workouts} workouts totaling {exercise_min} minutes."
        )

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "weekly",
                "summary": weekly
            }
        )

    async def _handle_summary(self, service) -> VoiceToolResult:
        """Handle general fitness summary."""
        metrics = await service.get_today_metrics()
        goals = await service.get_goal_progress()

        # Build concise summary
        parts = []
        parts.append(f"{metrics.steps:,} steps")
        parts.append(f"{metrics.calories_active} active calories")

        if metrics.workouts_today:
            count = len(metrics.workouts_today)
            parts.append(f"{count} workout{'s' if count > 1 else ''}")

        # Add goal status
        avg_pct = (goals["steps"]["percent"] + goals["calories"]["percent"]) // 2
        if avg_pct >= 100:
            parts.append("Goals complete!")
        elif avg_pct >= 75:
            parts.append(f"{avg_pct}% to goals")

        message = "Today's fitness: " + ", ".join(parts) + "."

        return VoiceToolResult(
            success=True,
            message=message,
            data={
                "metric": "summary",
                "metrics": metrics.to_dict(),
                "goals": {
                    "steps": goals["steps"]["percent"],
                    "calories": goals["calories"]["percent"],
                    "exercise": goals["exercise"]["percent"]
                }
            }
        )
