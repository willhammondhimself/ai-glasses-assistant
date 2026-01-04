"""
Daily Challenges - Gamification system with XP, streaks, and achievements.
Keeps users engaged with daily goals and progress tracking.
"""
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChallengeCategory(Enum):
    """Challenge categories."""
    MENTAL_MATH = "mental_math"
    FOCUS = "focus"
    LEARNING = "learning"
    PRODUCTIVITY = "productivity"
    WELLNESS = "wellness"
    SOCIAL = "social"
    CUSTOM = "custom"


class ChallengeDifficulty(Enum):
    """Challenge difficulty levels with XP multipliers."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 5


@dataclass
class Challenge:
    """A single challenge."""
    id: str
    title: str
    description: str
    category: ChallengeCategory
    difficulty: ChallengeDifficulty
    xp_reward: int
    target: int                              # Target to complete (e.g., 10 problems)
    progress: int = 0                        # Current progress
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    bonus_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.progress >= self.target

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def progress_percent(self) -> float:
        return min(100, (self.progress / self.target) * 100)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "xp_reward": self.xp_reward,
            "target": self.target,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "bonus_conditions": self.bonus_conditions,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Challenge":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            category=ChallengeCategory(data["category"]),
            difficulty=ChallengeDifficulty(data["difficulty"]),
            xp_reward=data["xp_reward"],
            target=data["target"],
            progress=data.get("progress", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            bonus_conditions=data.get("bonus_conditions", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class Achievement:
    """Unlockable achievement."""
    id: str
    title: str
    description: str
    icon: str
    xp_reward: int
    requirement: str                         # Human-readable requirement
    unlocked_at: Optional[datetime] = None

    @property
    def is_unlocked(self) -> bool:
        return self.unlocked_at is not None


@dataclass
class UserProgress:
    """User's overall progress and stats."""
    total_xp: int = 0
    level: int = 1
    current_streak: int = 0
    longest_streak: int = 0
    last_active_date: Optional[datetime] = None
    challenges_completed: int = 0
    achievements_unlocked: List[str] = field(default_factory=list)
    category_stats: Dict[str, int] = field(default_factory=dict)  # category -> completed count

    def to_dict(self) -> dict:
        return {
            "total_xp": self.total_xp,
            "level": self.level,
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "last_active_date": self.last_active_date.isoformat() if self.last_active_date else None,
            "challenges_completed": self.challenges_completed,
            "achievements_unlocked": self.achievements_unlocked,
            "category_stats": self.category_stats
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProgress":
        return cls(
            total_xp=data.get("total_xp", 0),
            level=data.get("level", 1),
            current_streak=data.get("current_streak", 0),
            longest_streak=data.get("longest_streak", 0),
            last_active_date=datetime.fromisoformat(data["last_active_date"]) if data.get("last_active_date") else None,
            challenges_completed=data.get("challenges_completed", 0),
            achievements_unlocked=data.get("achievements_unlocked", []),
            category_stats=data.get("category_stats", {})
        )


class DailyChallenges:
    """
    Daily Challenges gamification system.

    Provides daily challenges, XP tracking, streaks, and achievements
    to keep users engaged with WHAM.
    """

    # XP thresholds for levels
    LEVEL_THRESHOLDS = [
        0,      # Level 1
        100,    # Level 2
        250,    # Level 3
        500,    # Level 4
        1000,   # Level 5
        2000,   # Level 6
        3500,   # Level 7
        5500,   # Level 8
        8000,   # Level 9
        12000,  # Level 10
        # Additional levels follow geometric progression
    ]

    # Challenge templates
    CHALLENGE_TEMPLATES = {
        ChallengeCategory.MENTAL_MATH: [
            {"title": "Quick Math", "desc": "Solve {target} mental math problems", "base_xp": 25},
            {"title": "Speed Demon", "desc": "Solve {target} problems under time target", "base_xp": 40},
            {"title": "Perfect Streak", "desc": "Get {target} correct answers in a row", "base_xp": 50},
            {"title": "Difficulty Climber", "desc": "Complete {target} D3+ problems", "base_xp": 60},
        ],
        ChallengeCategory.FOCUS: [
            {"title": "Deep Focus", "desc": "Complete {target} Pomodoro sessions", "base_xp": 30},
            {"title": "Distraction Free", "desc": "Block {target} distractions", "base_xp": 20},
            {"title": "Focus Marathon", "desc": "Focus for {target} total minutes", "base_xp": 40},
        ],
        ChallengeCategory.LEARNING: [
            {"title": "Knowledge Seeker", "desc": "Learn {target} new facts", "base_xp": 25},
            {"title": "Context Builder", "desc": "Save {target} memories", "base_xp": 20},
            {"title": "Study Session", "desc": "Use homework help {target} times", "base_xp": 30},
        ],
        ChallengeCategory.PRODUCTIVITY: [
            {"title": "Task Master", "desc": "Complete {target} captures", "base_xp": 20},
            {"title": "Reminder Keeper", "desc": "Clear {target} reminders", "base_xp": 15},
            {"title": "Meeting Pro", "desc": "Use meeting mode {target} times", "base_xp": 35},
        ],
        ChallengeCategory.WELLNESS: [
            {"title": "Early Bird", "desc": "Start before 8am {target} days", "base_xp": 30},
            {"title": "Break Taker", "desc": "Take {target} focus breaks", "base_xp": 15},
            {"title": "Balanced Day", "desc": "Complete both focus and breaks {target} times", "base_xp": 25},
        ],
    }

    # Built-in achievements
    ACHIEVEMENTS = [
        Achievement("first_challenge", "First Steps", "Complete your first challenge", "🎯", 50, "Complete 1 challenge"),
        Achievement("streak_3", "On a Roll", "Maintain a 3-day streak", "🔥", 75, "3-day streak"),
        Achievement("streak_7", "Week Warrior", "Maintain a 7-day streak", "⚡", 150, "7-day streak"),
        Achievement("streak_30", "Monthly Master", "Maintain a 30-day streak", "🏆", 500, "30-day streak"),
        Achievement("level_5", "Rising Star", "Reach level 5", "⭐", 100, "Reach level 5"),
        Achievement("level_10", "Expert", "Reach level 10", "🌟", 250, "Reach level 10"),
        Achievement("math_master", "Math Master", "Complete 100 math challenges", "🧮", 200, "100 math challenges"),
        Achievement("focus_king", "Focus King", "Complete 50 focus sessions", "👑", 200, "50 focus sessions"),
        Achievement("all_categories", "Well Rounded", "Complete challenges in all categories", "🎨", 150, "All categories"),
        Achievement("perfect_day", "Perfect Day", "Complete all daily challenges", "💯", 100, "All dailies in one day"),
    ]

    def __init__(self, config: dict, storage_dir: str = "./challenges"):
        """
        Initialize Daily Challenges.

        Args:
            config: Configuration dictionary
            storage_dir: Directory to store challenge data
        """
        self.config = config
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._daily_challenges: List[Challenge] = []
        self._completed_today: List[Challenge] = []
        self._progress = UserProgress()
        self._achievements = {a.id: a for a in self.ACHIEVEMENTS}
        self._last_daily_generation: Optional[datetime] = None

        # Callbacks
        self._on_challenge_complete: List[Callable[[Challenge, int], None]] = []
        self._on_achievement_unlock: List[Callable[[Achievement], None]] = []
        self._on_level_up: List[Callable[[int, int], None]] = []

        # Load settings
        challenge_config = config.get("daily_challenges", {})
        self.daily_challenge_count = challenge_config.get("daily_count", 3)
        self.streak_bonus_multiplier = challenge_config.get("streak_bonus", 0.1)

        # Load data
        self._load_data()

        # Check if we need new daily challenges
        self._check_daily_refresh()

        logger.info(f"DailyChallenges initialized (Level {self._progress.level}, "
                   f"{self._progress.total_xp} XP, {self._progress.current_streak} day streak)")

    def _load_data(self):
        """Load progress and challenges from storage."""
        progress_file = self.storage_dir / "progress.json"
        challenges_file = self.storage_dir / "daily_challenges.json"

        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    self._progress = UserProgress.from_dict(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")

        if challenges_file.exists():
            try:
                with open(challenges_file, "r") as f:
                    data = json.load(f)
                    self._daily_challenges = [Challenge.from_dict(c) for c in data.get("daily", [])]
                    self._last_daily_generation = datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else None
            except Exception as e:
                logger.error(f"Failed to load challenges: {e}")

    def _save_data(self):
        """Save progress and challenges to storage."""
        progress_file = self.storage_dir / "progress.json"
        challenges_file = self.storage_dir / "daily_challenges.json"

        try:
            with open(progress_file, "w") as f:
                json.dump(self._progress.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

        try:
            with open(challenges_file, "w") as f:
                json.dump({
                    "daily": [c.to_dict() for c in self._daily_challenges],
                    "generated_at": self._last_daily_generation.isoformat() if self._last_daily_generation else None
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save challenges: {e}")

    def _check_daily_refresh(self):
        """Check if daily challenges need to be refreshed."""
        now = datetime.now()

        # Check streak
        if self._progress.last_active_date:
            days_since = (now.date() - self._progress.last_active_date.date()).days
            if days_since > 1:
                # Streak broken
                logger.info(f"Streak broken after {self._progress.current_streak} days")
                self._progress.current_streak = 0
            elif days_since == 1:
                # New day, streak continues
                self._progress.current_streak += 1
                if self._progress.current_streak > self._progress.longest_streak:
                    self._progress.longest_streak = self._progress.current_streak
                logger.info(f"Streak continues: {self._progress.current_streak} days")

        # Check if we need new daily challenges
        if self._last_daily_generation is None or self._last_daily_generation.date() < now.date():
            self._generate_daily_challenges()

    def _generate_daily_challenges(self):
        """Generate new daily challenges."""
        import uuid

        self._daily_challenges = []
        self._last_daily_generation = datetime.now()

        # Select categories (try to vary)
        categories = list(self.CHALLENGE_TEMPLATES.keys())
        selected_categories = random.sample(categories, min(self.daily_challenge_count, len(categories)))

        for i, category in enumerate(selected_categories):
            templates = self.CHALLENGE_TEMPLATES[category]
            template = random.choice(templates)

            # Determine difficulty based on user level
            if self._progress.level <= 2:
                difficulty = ChallengeDifficulty.EASY
                target = random.randint(3, 5)
            elif self._progress.level <= 5:
                difficulty = random.choice([ChallengeDifficulty.EASY, ChallengeDifficulty.MEDIUM])
                target = random.randint(5, 10)
            elif self._progress.level <= 8:
                difficulty = random.choice([ChallengeDifficulty.MEDIUM, ChallengeDifficulty.HARD])
                target = random.randint(8, 15)
            else:
                difficulty = random.choice([ChallengeDifficulty.HARD, ChallengeDifficulty.EXPERT])
                target = random.randint(10, 20)

            # Calculate XP
            xp = template["base_xp"] * difficulty.value

            challenge = Challenge(
                id=str(uuid.uuid4())[:8],
                title=template["title"],
                description=template["desc"].format(target=target),
                category=category,
                difficulty=difficulty,
                xp_reward=xp,
                target=target,
                expires_at=datetime.now().replace(hour=23, minute=59, second=59)
            )

            self._daily_challenges.append(challenge)

        self._save_data()
        logger.info(f"Generated {len(self._daily_challenges)} daily challenges")

    def get_daily_challenges(self) -> List[Challenge]:
        """Get today's challenges."""
        self._check_daily_refresh()
        return self._daily_challenges

    def record_progress(self, category: ChallengeCategory, amount: int = 1, metadata: Dict[str, Any] = None):
        """
        Record progress towards challenges.

        Args:
            category: Challenge category
            amount: Progress amount
            metadata: Additional metadata for bonus conditions
        """
        for challenge in self._daily_challenges:
            if challenge.category == category and not challenge.is_completed:
                challenge.progress += amount

                if challenge.is_completed:
                    self._complete_challenge(challenge)

        # Update last active date
        self._progress.last_active_date = datetime.now()
        self._save_data()

    def _complete_challenge(self, challenge: Challenge):
        """Handle challenge completion."""
        challenge.completed_at = datetime.now()

        # Calculate XP with streak bonus
        bonus_multiplier = 1 + (self._progress.current_streak * self.streak_bonus_multiplier)
        xp_earned = int(challenge.xp_reward * bonus_multiplier)

        # Update progress
        old_level = self._progress.level
        self._progress.total_xp += xp_earned
        self._progress.challenges_completed += 1

        # Update category stats
        cat_name = challenge.category.value
        self._progress.category_stats[cat_name] = self._progress.category_stats.get(cat_name, 0) + 1

        # Check for level up
        new_level = self._calculate_level(self._progress.total_xp)
        if new_level > old_level:
            self._progress.level = new_level
            logger.info(f"Level up! {old_level} → {new_level}")
            for callback in self._on_level_up:
                try:
                    callback(old_level, new_level)
                except Exception as e:
                    logger.error(f"Level up callback error: {e}")

        # Notify callbacks
        for callback in self._on_challenge_complete:
            try:
                callback(challenge, xp_earned)
            except Exception as e:
                logger.error(f"Challenge complete callback error: {e}")

        # Check achievements
        self._check_achievements()

        self._completed_today.append(challenge)
        logger.info(f"Challenge completed: {challenge.title} (+{xp_earned} XP)")

    def _calculate_level(self, xp: int) -> int:
        """Calculate level from XP."""
        level = 1
        for i, threshold in enumerate(self.LEVEL_THRESHOLDS):
            if xp >= threshold:
                level = i + 1
            else:
                break

        # Handle levels beyond predefined thresholds
        if level >= len(self.LEVEL_THRESHOLDS):
            extra_xp = xp - self.LEVEL_THRESHOLDS[-1]
            extra_levels = extra_xp // 5000  # 5000 XP per level after threshold
            level += extra_levels

        return level

    def _check_achievements(self):
        """Check and unlock achievements."""
        unlocked = []

        # First challenge
        if self._progress.challenges_completed >= 1:
            unlocked.append("first_challenge")

        # Streaks
        if self._progress.current_streak >= 3:
            unlocked.append("streak_3")
        if self._progress.current_streak >= 7:
            unlocked.append("streak_7")
        if self._progress.current_streak >= 30:
            unlocked.append("streak_30")

        # Levels
        if self._progress.level >= 5:
            unlocked.append("level_5")
        if self._progress.level >= 10:
            unlocked.append("level_10")

        # Category specific
        if self._progress.category_stats.get("mental_math", 0) >= 100:
            unlocked.append("math_master")
        if self._progress.category_stats.get("focus", 0) >= 50:
            unlocked.append("focus_king")

        # All categories
        if len(self._progress.category_stats) >= len(ChallengeCategory) - 1:  # Exclude CUSTOM
            unlocked.append("all_categories")

        # Perfect day
        if len(self._completed_today) >= len(self._daily_challenges) and len(self._daily_challenges) > 0:
            unlocked.append("perfect_day")

        # Unlock new achievements
        for achievement_id in unlocked:
            if achievement_id not in self._progress.achievements_unlocked:
                if achievement_id in self._achievements:
                    achievement = self._achievements[achievement_id]
                    achievement.unlocked_at = datetime.now()
                    self._progress.achievements_unlocked.append(achievement_id)
                    self._progress.total_xp += achievement.xp_reward

                    logger.info(f"Achievement unlocked: {achievement.title}!")

                    for callback in self._on_achievement_unlock:
                        try:
                            callback(achievement)
                        except Exception as e:
                            logger.error(f"Achievement callback error: {e}")

    def get_progress(self) -> UserProgress:
        """Get user progress."""
        return self._progress

    def get_xp_to_next_level(self) -> int:
        """Get XP needed for next level."""
        current_level = self._progress.level
        if current_level < len(self.LEVEL_THRESHOLDS):
            return self.LEVEL_THRESHOLDS[current_level] - self._progress.total_xp
        else:
            # Post-threshold progression
            return 5000 - (self._progress.total_xp % 5000)

    def get_achievements(self, unlocked_only: bool = False) -> List[Achievement]:
        """Get all achievements."""
        if unlocked_only:
            return [a for a in self._achievements.values() if a.is_unlocked]
        return list(self._achievements.values())

    def on_challenge_complete(self, callback: Callable[[Challenge, int], None]):
        """Register callback for challenge completion."""
        self._on_challenge_complete.append(callback)

    def on_achievement_unlock(self, callback: Callable[[Achievement], None]):
        """Register callback for achievement unlock."""
        self._on_achievement_unlock.append(callback)

    def on_level_up(self, callback: Callable[[int, int], None]):
        """Register callback for level up (old_level, new_level)."""
        self._on_level_up.append(callback)

    def format_for_display(self) -> List[str]:
        """Format challenges for HUD display."""
        lines = []

        # Header
        lines.append(f"Level {self._progress.level} | {self._progress.total_xp} XP")
        if self._progress.current_streak > 0:
            lines.append(f"🔥 {self._progress.current_streak} day streak")
        lines.append("")

        # Daily challenges
        lines.append("─── Daily Challenges ───")
        for challenge in self._daily_challenges:
            status = "✅" if challenge.is_completed else "⬜"
            progress = f"{challenge.progress}/{challenge.target}"
            lines.append(f"{status} {challenge.title}")
            lines.append(f"   {progress} | +{challenge.xp_reward} XP")

        # XP to next level
        lines.append("")
        lines.append(f"Next level: {self.get_xp_to_next_level()} XP")

        return lines

    def format_for_tts(self) -> str:
        """Format status for TTS."""
        incomplete = [c for c in self._daily_challenges if not c.is_completed]
        completed = len(self._daily_challenges) - len(incomplete)

        parts = [f"Level {self._progress.level}"]

        if self._progress.current_streak > 0:
            parts.append(f"{self._progress.current_streak} day streak")

        parts.append(f"{completed} of {len(self._daily_challenges)} challenges completed")

        if incomplete:
            parts.append(f"Next: {incomplete[0].title}")

        return ". ".join(parts)


# Test
def test_daily_challenges():
    """Test daily challenges functionality."""
    import tempfile

    print("=== Daily Challenges Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "daily_challenges": {
                "daily_count": 3,
                "streak_bonus": 0.1
            }
        }

        dc = DailyChallenges(config, storage_dir=tmpdir)

        # Track events
        events = []
        dc.on_challenge_complete(lambda c, xp: events.append(f"Completed: {c.title} +{xp}XP"))
        dc.on_level_up(lambda old, new: events.append(f"Level up: {old} → {new}"))

        # Get daily challenges
        print("1. Daily Challenges:")
        for challenge in dc.get_daily_challenges():
            print(f"   [{challenge.difficulty.name}] {challenge.title}")
            print(f"      {challenge.description}")
            print(f"      Reward: {challenge.xp_reward} XP")
            print()

        # Simulate progress
        print("2. Recording progress...")
        for challenge in dc.get_daily_challenges():
            dc.record_progress(challenge.category, challenge.target)

        print()

        # Check events
        print("3. Events:")
        for event in events:
            print(f"   {event}")
        print()

        # Progress
        print("4. Progress:")
        progress = dc.get_progress()
        print(f"   Level: {progress.level}")
        print(f"   XP: {progress.total_xp}")
        print(f"   Completed: {progress.challenges_completed}")
        print()

        # Achievements
        print("5. Achievements unlocked:")
        for achievement in dc.get_achievements(unlocked_only=True):
            print(f"   {achievement.icon} {achievement.title}")
        print()

        # Display
        print("6. Display format:")
        for line in dc.format_for_display():
            print(f"   {line}")
        print()

        # TTS
        print("7. TTS format:")
        print(f"   {dc.format_for_tts()}")


if __name__ == "__main__":
    test_daily_challenges()
