"""
WHAM Response Templates.
Will Hammond's Augmented Mind - Direct, confident, analytical.

All templates use {placeholders} for dynamic content:
    {name} - User's name (Will)
    {time} - Time value
    {answer} - Math answer
    {streak} - Streak count
    {accuracy} - Accuracy percentage
"""
import random
from typing import Dict, List
from datetime import datetime


class Templates:
    """WHAM response template manager."""

    # ============================================================
    # GREETINGS - Direct and ready
    # ============================================================

    GREETINGS = {
        "morning": [
            "WHAM online. Morning, {name}.",
            "Morning, {name}. Ready to work.",
            "WHAM active. Let's go.",
            "Systems up. Morning grind.",
        ],
        "afternoon": [
            "WHAM online. What's the task?",
            "Afternoon, {name}. Ready.",
            "WHAM active. Let's get it.",
        ],
        "evening": [
            "WHAM here. Evening grind.",
            "Evening session. Let's work.",
            "WHAM online. Sharp as ever.",
        ],
        "night": [
            "WHAM online. Late session. Stay sharp.",
            "Night mode. Focus up, {name}.",
            "Late grind. WHAM ready.",
        ],
    }

    # ============================================================
    # SPEED FEEDBACK - Quant-calibrated
    # ============================================================

    SPEED_FEEDBACK = {
        "exceptional": [  # < 50% of target time
            "{time}s. Blazing, {name}. Top 5% speed.",
            "{time}s. Quant speed. Keep it.",
            "{time}s. That's elite.",
            "{time}s. Trading floor ready.",
        ],
        "excellent": [  # 50-75% of target
            "{time}s. Solid execution.",
            "{time}s. Clean.",
            "{time}s. That's the pace.",
            "{time}s. Good.",
        ],
        "good": [  # 75-100% of target
            "{time}s. On target.",
            "{time}s. Acceptable.",
            "{time}s. Done.",
        ],
        "slow": [  # > 100% of target
            "{time}s. Too slow. Focus.",
            "{time}s. Push harder.",
            "{time}s. Speed it up.",
            "{time}s. We need faster.",
        ],
    }

    # ============================================================
    # WRONG ANSWER FEEDBACK - Direct, move on
    # ============================================================

    WRONG_ANSWER = [
        "{answer}. Next.",
        "Wrong. {answer}. Move on.",
        "Incorrect. {answer}.",
        "{answer} was it. Keep going.",
        "Miss. {answer}. Next.",
    ]

    # ============================================================
    # STREAK MILESTONES - Achievement focused
    # ============================================================

    STREAK_MILESTONES = {
        3: [
            "3 straight. Building.",
            "3. Keep it going.",
        ],
        5: [
            "5 in a row. Warming up.",
            "5. You're locked in.",
        ],
        7: [
            "7 straight. Flow state.",
            "7. Don't break now.",
        ],
        10: [
            "10 consecutive. Elite.",
            "10. That's the standard.",
        ],
        15: [
            "15. Outstanding.",
            "15 straight. Peak focus.",
        ],
        20: [
            "20. Interview ready.",
            "20 consecutive. Exceptional.",
        ],
        25: [
            "25. Legendary run.",
            "25. Trading desk speed.",
        ],
        50: [
            "50. Genuinely impressed, {name}.",
            "50 straight. That's rare.",
        ],
    }

    # ============================================================
    # STREAK BREAK
    # ============================================================

    STREAK_BREAK = [
        "Streak done. Rebuild.",
        "Reset. Start again.",
        "Back to zero. Go.",
        "Streak ends. Next one starts now.",
    ]

    # ============================================================
    # ENCOURAGEMENT - Brief, forward-focused
    # ============================================================

    ENCOURAGEMENT = [
        "Next one.",
        "Shake it off.",
        "Focus up.",
        "Recalibrate. Go.",
        "Keep moving.",
    ]

    # ============================================================
    # SESSION ENDINGS
    # ============================================================

    SESSION_END = {
        "excellent": [  # 95%+ accuracy
            "{accuracy}%. Excellent session.",
            "Outstanding. {accuracy}%.",
        ],
        "good": [  # 80-95%
            "{accuracy}%. Solid work.",
            "Good session. {accuracy}%.",
        ],
        "average": [  # 65-80%
            "{accuracy}%. Room to improve.",
            "Session done. {accuracy}%. Push harder next time.",
        ],
        "poor": [  # < 65%
            "{accuracy}%. Need more practice.",
            "Tough session. {accuracy}%. Come back stronger.",
        ],
    }

    # ============================================================
    # MODE ACTIVATION
    # ============================================================

    MODE_ACTIVATION = {
        "mental_math": [
            "Math mode. Difficulty {difficulty}. Go.",
            "Speed math active. Level {difficulty}.",
            "Arithmetic drill. {difficulty}. Ready.",
        ],
        "poker": [
            "Poker coach online. Show cards.",
            "Live poker mode. Analyzing.",
            "Poker active. What's the spot?",
        ],
        "code": [
            "Code mode. What's the problem?",
            "Debug active. Show me.",
        ],
    }

    # ============================================================
    # POKER-SPECIFIC TEMPLATES
    # ============================================================

    POKER_ANALYSIS = {
        "thinking": [
            "Analyzing...",
            "Reading the spot...",
            "Processing...",
        ],
        "recommendation": [
            "{action}. {reasoning}",
            "{action} here. {reasoning}",
        ],
        "villain_read": [
            "Villain: {type}. {exploit}",
            "{type} detected. {exploit}",
        ],
    }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @classmethod
    def get_greeting(cls, name: str = "Will", address: str = "") -> str:
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
        return template.format(name=name)

    @classmethod
    def get_speed_feedback(
        cls,
        time_ms: float,
        target_ms: float,
        address: str = "",
        name: str = "Will"
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
        return template.format(time=time_str, name=name)

    @classmethod
    def get_wrong_answer_feedback(cls, answer: float, address: str = "") -> str:
        """Get feedback for wrong answer."""
        if answer == int(answer):
            ans_str = str(int(answer))
        else:
            ans_str = f"{answer:.2f}"

        template = random.choice(cls.WRONG_ANSWER)
        return template.format(answer=ans_str)

    @classmethod
    def get_streak_message(cls, streak: int, address: str = "", name: str = "Will") -> str:
        """Get streak milestone message if applicable."""
        if streak in cls.STREAK_MILESTONES:
            template = random.choice(cls.STREAK_MILESTONES[streak])
            return template.format(name=name)
        return ""

    @classmethod
    def get_streak_break_message(cls) -> str:
        """Get message when streak breaks."""
        return random.choice(cls.STREAK_BREAK)

    @classmethod
    def get_encouragement(cls, address: str = "") -> str:
        """Get encouragement after errors."""
        return random.choice(cls.ENCOURAGEMENT)

    @classmethod
    def get_session_end(cls, accuracy: float, address: str = "") -> str:
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
        return template.format(accuracy=f"{accuracy:.1f}")

    @classmethod
    def get_mode_activation(cls, mode: str, difficulty: int = 2) -> str:
        """Get mode activation message."""
        templates = cls.MODE_ACTIVATION.get(mode, ["Mode active."])
        template = random.choice(templates)
        return template.format(difficulty=difficulty)

    @classmethod
    def get_location_comment(cls, location: str) -> str:
        """Location comments - not used in WHAM (too chatty)."""
        return ""

    @classmethod
    def get_poker_thinking(cls) -> str:
        """Get poker thinking message."""
        return random.choice(cls.POKER_ANALYSIS["thinking"])

    @classmethod
    def get_poker_recommendation(cls, action: str, reasoning: str) -> str:
        """Get poker recommendation message."""
        template = random.choice(cls.POKER_ANALYSIS["recommendation"])
        return template.format(action=action, reasoning=reasoning)


# Quick access functions
def greeting(name: str = "Will") -> str:
    return Templates.get_greeting(name)


def speed_feedback(time_ms: float, target_ms: float, name: str = "Will") -> str:
    return Templates.get_speed_feedback(time_ms, target_ms, name=name)


def wrong_answer(answer: float) -> str:
    return Templates.get_wrong_answer_feedback(answer)


def streak_message(streak: int, name: str = "Will") -> str:
    return Templates.get_streak_message(streak, name=name)


# Test
if __name__ == "__main__":
    print("=== WHAM Templates Test ===\n")

    print(f"Greeting: {greeting()}")
    print()

    print("Speed feedback:")
    print(f"  Blazing (1.5s/4s): {speed_feedback(1500, 4000)}")
    print(f"  Solid (3s/4s): {speed_feedback(3000, 4000)}")
    print(f"  Too slow (5s/4s): {speed_feedback(5000, 4000)}")
    print()

    print(f"Wrong answer: {wrong_answer(3901)}")
    print()

    print("Streak messages:")
    for s in [3, 5, 10, 20]:
        msg = streak_message(s)
        if msg:
            print(f"  {s}: {msg}")
