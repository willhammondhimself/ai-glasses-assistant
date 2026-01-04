"""
WHAM Response Templates

All phrases and templates for WHAM-style responses.
Personal AI assistant personality for Will.
"""

import random
from typing import Optional

# Time-of-day greetings
GREETINGS = {
    "morning": [
        "Good morning, {address}. Ready to sharpen those mental faculties?",
        "Good morning, {address}. Your morning drill awaits.",
        "Morning, {address}. Shall we begin?",
    ],
    "afternoon": [
        "Good afternoon, {address}. Back for more?",
        "Afternoon, {address}. Your timing is impeccable.",
        "{address}. Picking up where we left off?",
    ],
    "evening": [
        "Good evening, {address}. A productive session ahead, I trust.",
        "Evening, {address}. Let's make it count.",
        "{address}. Evening practice—that's dedication.",
    ],
    "late_night": [
        "{address}, burning the midnight oil. Admirable commitment.",
        "Late night session, {address}? Jane Street won't know what hit them.",
        "The nocturnal approach. I approve, {address}.",
    ],
}

# Speed feedback (Jane Street calibrated)
SPEED_FEEDBACK = {
    "exceptional": [  # < 50% of target time
        "{time}s—that's Jane Street caliber, {address}.",
        "Remarkable. {time}s. The quant desks would notice.",
        "{time}s. You're operating at trading floor speed.",
        "Exceptional. {time}s. Citadel would be intrigued.",
    ],
    "excellent": [  # 50-75% of target
        "Well executed, {address}. {time}s.",
        "{time}s—interview ready.",
        "Solid. {time}s is competition pace.",
    ],
    "good": [  # 75-100% of target
        "{time}s. Within parameters.",
        "On pace, {address}.",
        "{time}s. Adequate.",
    ],
    "needs_work": [  # > 100% of target
        "{time}s. We can do better, {address}.",
        "Room for improvement. {time}s is above target.",
        "{time}s. Focus on speed, {address}.",
    ],
}

# Streak milestone messages
STREAK_MILESTONES = {
    3: "Three consecutive. Building momentum.",
    5: "Five in a row. You're in the zone, {address}.",
    7: "Seven straight. The focus is admirable.",
    10: "A perfect ten, {address}. Exemplary.",
    15: "Fifteen consecutive. I'm running out of superlatives.",
    20: "Twenty. Shall I notify the trading firms?",
    25: "Twenty-five. This is approaching legendary status, {address}.",
    50: "Fifty consecutive correct. Unprecedented territory.",
}

# Milestone celebrations
MILESTONES = {
    "first_correct": "First blood, {address}. The session has begun.",
    "ten_problems": "Ten problems in. Warmed up nicely.",
    "perfect_set": "A flawless set of ten. Most impressive.",
    "new_high_score": "New personal best, {address}. I've updated your records.",
    "speed_record": "That's a new speed record for this difficulty level.",
    "comeback": "Recovered nicely, {address}. Back on track.",
}

# Encouragement after incorrect answers
ENCOURAGEMENT = [
    "The correct answer was {answer}. Onward.",
    "{answer}. A minor setback. Resume.",
    "Not quite. {answer}. Shake it off, {address}.",
    "{answer}. Even the best miss occasionally.",
]

# Proactive suggestions
SUGGESTIONS = {
    "break_recommended": "You've been at this for {duration} minutes. A brief respite might sharpen focus.",
    "difficulty_increase": "Consistent excellence. Shall I increase the difficulty?",
    "difficulty_decrease": "I'm detecting some struggle. Would you prefer an easier tier?",
    "change_category": "Perhaps a change of pace? Try {category} problems.",
    "session_end": "Solid session, {address}. {correct}/{total} at {accuracy}% accuracy.",
}

# Problem type introductions
PROBLEM_INTROS = {
    "mental_math": "Mental arithmetic. Standard rules apply.",
    "algebra": "Algebraic manipulation. Show your work mentally.",
    "calculus": "Calculus. Integration or differentiation—let's see.",
    "probability": "Probability. Think in terms of outcomes.",
    "poker": "Poker analysis. GTO framework engaged.",
}

# Mode activation confirmations
MODE_ACTIVATIONS = {
    "mental_math": "Mental math mode activated. How many problems, {address}?",
    "poker": "Poker analysis mode. Show me your cards.",
    "homework": "Homework assistance mode. Point at the problem.",
    "debug": "Code debugging mode. Let's find that bug.",
}

# Session summary templates
SESSION_SUMMARY = [
    "Session complete. {correct}/{total} correct. Average: {avg_time}s. {trend_comment}",
    "That's a wrap, {address}. {accuracy}% accuracy across {total} problems.",
]

TREND_COMMENTS = {
    "improving": "You've improved {percent}% this week. Jane Street, here we come.",
    "steady": "Maintaining consistent performance.",
    "declining": "A slight dip today. Rest and return stronger.",
}


class ResponseTemplates:
    """Template manager for WHAM responses."""

    def get_greeting(self, time_period: str, address: str) -> str:
        """Get appropriate greeting for time of day."""
        templates = GREETINGS.get(time_period, GREETINGS["afternoon"])
        return random.choice(templates).format(address=address)

    def get_speed_feedback(
        self,
        time_ms: int,
        target_ms: int,
        address: str
    ) -> str:
        """Get speed-appropriate feedback."""
        ratio = time_ms / target_ms
        time_s = time_ms / 1000

        if ratio < 0.5:
            tier = "exceptional"
        elif ratio < 0.75:
            tier = "excellent"
        elif ratio <= 1.0:
            tier = "good"
        else:
            tier = "needs_work"

        templates = SPEED_FEEDBACK[tier]
        return random.choice(templates).format(time=f"{time_s:.2f}", address=address)

    def get_streak_message(self, streak: int, address: str) -> Optional[str]:
        """Get streak acknowledgment if at milestone."""
        if streak in STREAK_MILESTONES:
            return STREAK_MILESTONES[streak].format(address=address)

        # For very high streaks, generate dynamic message
        if streak > 25 and streak % 5 == 0:
            return f"{streak} consecutive. Extraordinary, {address}."

        return None

    def get_encouragement(self, answer: str, address: str) -> str:
        """Get encouraging message after incorrect answer."""
        return random.choice(ENCOURAGEMENT).format(answer=answer, address=address)

    def get_milestone_message(self, milestone: str, address: str) -> Optional[str]:
        """Get milestone celebration message."""
        if milestone in MILESTONES:
            return MILESTONES[milestone].format(address=address)
        return None

    def get_suggestion(
        self,
        suggestion_type: str,
        address: str,
        **kwargs
    ) -> Optional[str]:
        """Get proactive suggestion."""
        if suggestion_type in SUGGESTIONS:
            return SUGGESTIONS[suggestion_type].format(address=address, **kwargs)
        return None

    def get_mode_activation(self, mode: str, address: str) -> str:
        """Get mode activation confirmation."""
        if mode in MODE_ACTIVATIONS:
            return MODE_ACTIVATIONS[mode].format(address=address)
        return f"{mode.replace('_', ' ').title()} mode activated, {address}."

    def get_session_summary(
        self,
        address: str,
        correct: int,
        total: int,
        avg_time_ms: int,
        trend: str = "steady"
    ) -> str:
        """Get session summary message."""
        accuracy = int((correct / total) * 100) if total > 0 else 0
        avg_time_s = avg_time_ms / 1000

        trend_comment = TREND_COMMENTS.get(trend, "")
        if "{percent}" in trend_comment:
            trend_comment = trend_comment.format(percent=random.randint(2, 8))

        template = random.choice(SESSION_SUMMARY)
        return template.format(
            address=address,
            correct=correct,
            total=total,
            accuracy=accuracy,
            avg_time=f"{avg_time_s:.2f}",
            trend_comment=trend_comment
        )
