"""
JARVIS Response Templates.
Tony Stark's butler-style AI responses.

All templates use {placeholders} for dynamic content:
    {name} - User's name (Will)
    {address} - How JARVIS addresses user (sir)
    {time} - Time value
    {answer} - Math answer
    {streak} - Streak count
    {accuracy} - Accuracy percentage
"""
import random
from typing import Dict, List
from datetime import datetime


class Templates:
    """JARVIS response template manager."""

    # ============================================================
    # GREETINGS - Time-of-day aware
    # ============================================================

    GREETINGS = {
        "morning": [
            "Good morning, {address}. Systems online.",
            "Morning, {address}. Ready when you are.",
            "Good morning. Shall we begin?",
            "Rise and shine, {address}. JARVIS at your service.",
        ],
        "afternoon": [
            "Good afternoon, {address}.",
            "Afternoon, {address}. How may I assist?",
            "Good afternoon. Systems standing by.",
        ],
        "evening": [
            "Good evening, {address}.",
            "Evening, {address}. Ready for a session?",
            "Good evening. JARVIS online.",
        ],
        "night": [
            "Burning the midnight oil, {address}?",
            "Late night session, {address}. I admire the dedication.",
            "The night shift. Very well, {address}.",
        ],
    }

    # ============================================================
    # SPEED FEEDBACK - Jane Street calibrated
    # ============================================================

    SPEED_FEEDBACK = {
        "exceptional": [  # < 50% of target time
            "{time}s. Exceptional, {address}.",
            "{time}s. That's trading floor caliber.",
            "Remarkable. {time}s.",
            "{time}s. Jane Street would approve.",
            "Lightning fast. {time}s.",
        ],
        "excellent": [  # 50-75% of target
            "Well executed, {address}. {time}s.",
            "{time}s. Interview ready.",
            "Solid. {time}s.",
            "{time}s. Clean work.",
            "Sharp. {time}s.",
        ],
        "good": [  # 75-100% of target
            "{time}s. Good.",
            "Correct. {time}s.",
            "{time}s. On target.",
            "Done. {time}s.",
        ],
        "slow": [  # > 100% of target
            "{time}s. We can do better.",
            "Correct, but {time}s. Push harder.",
            "{time}s. Speed it up.",
            "Right answer, slow execution. {time}s.",
        ],
    }

    # ============================================================
    # WRONG ANSWER FEEDBACK
    # ============================================================

    WRONG_ANSWER = [
        "The correct answer was {answer}. Onward.",
        "Incorrect. {answer}. Next.",
        "{answer} was the answer. Keep moving.",
        "That's {answer}, {address}. Let's continue.",
        "Not quite. {answer}. Focus.",
    ]

    # ============================================================
    # STREAK MILESTONES
    # ============================================================

    STREAK_MILESTONES = {
        3: [
            "Three consecutive. Building momentum.",
            "Three in a row. Good start.",
        ],
        5: [
            "Five consecutive, {address}. You're in the zone.",
            "Five straight. Warming up nicely.",
        ],
        7: [
            "Seven. Keep this pace.",
            "Lucky seven. But we don't believe in luck.",
        ],
        10: [
            "A perfect ten, {address}. Exemplary.",
            "Ten consecutive. Now we're talking.",
        ],
        15: [
            "Fifteen, {address}. Outstanding.",
            "Fifteen in a row. Impressive focus.",
        ],
        20: [
            "Twenty consecutive. Shall I notify the trading firms?",
            "Twenty. Interview ready, {address}.",
        ],
        25: [
            "Twenty-five. Legendary performance, {address}.",
            "A quarter century of correct answers. Remarkable.",
        ],
        50: [
            "Fifty consecutive, {address}. I'm genuinely impressed.",
            "Half a hundred. You've earned a break. Perhaps.",
        ],
    }

    # ============================================================
    # STREAK BREAK ENCOURAGEMENT
    # ============================================================

    STREAK_BREAK = [
        "Streak ended. Reset and refocus.",
        "The streak ends. Shall we build another?",
        "And we start again. One at a time.",
        "Back to zero. But you know the way now.",
    ]

    # ============================================================
    # ENCOURAGEMENT - After errors
    # ============================================================

    ENCOURAGEMENT = [
        "One miss doesn't define the session.",
        "Recalibrate and continue.",
        "The next one is yours.",
        "Focus, {address}. You've got this.",
        "Shake it off. Next problem.",
    ]

    # ============================================================
    # SESSION ENDINGS
    # ============================================================

    SESSION_END = {
        "excellent": [  # 95%+ accuracy
            "Exceptional session, {address}. {accuracy}% accuracy.",
            "Outstanding performance. {accuracy}%. Well done.",
        ],
        "good": [  # 80-95%
            "Solid session. {accuracy}% accuracy.",
            "Good work, {address}. {accuracy}%.",
        ],
        "average": [  # 65-80%
            "Session complete. {accuracy}%. Room for improvement.",
            "{accuracy}%. Let's do better next time.",
        ],
        "poor": [  # < 65%
            "{accuracy}%. We need to practice more.",
            "Challenging session. {accuracy}%. Don't give up.",
        ],
    }

    # ============================================================
    # MODE ACTIVATION
    # ============================================================

    MODE_ACTIVATION = {
        "mental_math": [
            "Mental math mode engaged. Difficulty {difficulty}.",
            "Arithmetic drill active. Level {difficulty}. Ready when you are.",
            "Speed math. Difficulty {difficulty}. Let's see what you've got.",
        ],
        "poker": [
            "Poker analysis online. Show me the cards.",
            "GTO mode active. What's the situation?",
        ],
        "code": [
            "Code analysis ready. What needs debugging?",
            "Debug mode online. Show me the problem.",
        ],
    }

    # ============================================================
    # LOCATION-AWARE COMMENTS
    # ============================================================

    LOCATION_COMMENTS = {
        "Claremont, CA": [
            "Claremont weather looking clear for focus.",
            "Good conditions in Claremont today.",
        ],
        "Santa Barbara, CA": [
            "Surf's probably good in Santa Barbara. After practice.",
            "Santa Barbara. Nice. But first, math.",
        ],
    }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @classmethod
    def get_greeting(cls, name: str = "Will", address: str = "sir") -> str:
        """Get time-appropriate greeting."""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"

        template = random.choice(cls.GREETINGS[period])
        return template.format(name=name, address=address)

    @classmethod
    def get_speed_feedback(
        cls,
        time_ms: float,
        target_ms: float,
        address: str = "sir"
    ) -> str:
        """Get feedback based on speed performance."""
        ratio = time_ms / target_ms if target_ms > 0 else 1.0
        time_str = f"{time_ms / 1000:.2f}"

        if ratio < 0.5:
            tier = "exceptional"
        elif ratio < 0.75:
            tier = "excellent"
        elif ratio <= 1.0:
            tier = "good"
        else:
            tier = "slow"

        template = random.choice(cls.SPEED_FEEDBACK[tier])
        return template.format(time=time_str, address=address)

    @classmethod
    def get_wrong_answer_feedback(cls, answer: float, address: str = "sir") -> str:
        """Get feedback for wrong answer."""
        # Format answer nicely
        if answer == int(answer):
            ans_str = str(int(answer))
        else:
            ans_str = f"{answer:.2f}"

        template = random.choice(cls.WRONG_ANSWER)
        return template.format(answer=ans_str, address=address)

    @classmethod
    def get_streak_message(cls, streak: int, address: str = "sir") -> str:
        """Get streak milestone message if applicable."""
        if streak in cls.STREAK_MILESTONES:
            template = random.choice(cls.STREAK_MILESTONES[streak])
            return template.format(address=address)
        return ""

    @classmethod
    def get_streak_break_message(cls) -> str:
        """Get message when streak breaks."""
        return random.choice(cls.STREAK_BREAK)

    @classmethod
    def get_encouragement(cls, address: str = "sir") -> str:
        """Get encouragement after errors."""
        template = random.choice(cls.ENCOURAGEMENT)
        return template.format(address=address)

    @classmethod
    def get_session_end(cls, accuracy: float, address: str = "sir") -> str:
        """Get session ending message."""
        if accuracy >= 95:
            tier = "excellent"
        elif accuracy >= 80:
            tier = "good"
        elif accuracy >= 65:
            tier = "average"
        else:
            tier = "poor"

        template = random.choice(cls.SESSION_END[tier])
        return template.format(accuracy=f"{accuracy:.1f}", address=address)

    @classmethod
    def get_mode_activation(cls, mode: str, difficulty: int = 2) -> str:
        """Get mode activation message."""
        templates = cls.MODE_ACTIVATION.get(mode, ["Mode activated."])
        template = random.choice(templates)
        return template.format(difficulty=difficulty)

    @classmethod
    def get_location_comment(cls, location: str) -> str:
        """Get location-specific comment if available."""
        comments = cls.LOCATION_COMMENTS.get(location, [])
        return random.choice(comments) if comments else ""


# Quick access functions
def greeting(name: str = "Will", address: str = "sir") -> str:
    return Templates.get_greeting(name, address)


def speed_feedback(time_ms: float, target_ms: float, address: str = "sir") -> str:
    return Templates.get_speed_feedback(time_ms, target_ms, address)


def wrong_answer(answer: float, address: str = "sir") -> str:
    return Templates.get_wrong_answer_feedback(answer, address)


def streak_message(streak: int, address: str = "sir") -> str:
    return Templates.get_streak_message(streak, address)


# Test
if __name__ == "__main__":
    print("=== JARVIS Templates Test ===\n")

    print(f"Greeting: {greeting()}")
    print()

    print("Speed feedback:")
    print(f"  Fast (1.5s/4s): {speed_feedback(1500, 4000)}")
    print(f"  Good (3s/4s): {speed_feedback(3000, 4000)}")
    print(f"  Slow (5s/4s): {speed_feedback(5000, 4000)}")
    print()

    print(f"Wrong answer: {wrong_answer(3901)}")
    print()

    print("Streak messages:")
    for s in [3, 5, 10, 20]:
        msg = streak_message(s)
        if msg:
            print(f"  {s}: {msg}")
