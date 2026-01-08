"""Dynamic personality for WHAM voice agent."""
from datetime import datetime
from typing import Optional
import random


def get_time_of_day() -> str:
    """Get the current time of day period."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_day_type() -> str:
    """Get whether it's a weekday or weekend."""
    day = datetime.now().weekday()
    return "weekend" if day >= 5 else "weekday"


def get_time_aware_greeting() -> str:
    """Generate a time-appropriate greeting instruction."""
    time_of_day = get_time_of_day()
    day_type = get_day_type()

    greetings = {
        "morning": [
            "Good morning, Will. WHAM online. What's the plan today?",
            "Morning, Will. Systems ready. What do you need?",
            "WHAM online. Good morning. Ready when you are.",
        ],
        "afternoon": [
            "Afternoon, Will. WHAM ready. What can I help with?",
            "Good afternoon. Systems active. What do you need?",
            "WHAM online. How's the afternoon going?",
        ],
        "evening": [
            "Evening, Will. WHAM standing by. What's up?",
            "Good evening. What can I help with?",
            "WHAM online. Winding down or ramping up?",
        ],
        "night": [
            "Late night session? WHAM's ready. What do you need?",
            "WHAM online. Burning the midnight oil?",
            "Night mode active. What are we working on?",
        ],
    }

    # Weekend-specific additions
    weekend_additions = [
        "Hope the weekend's going well.",
        "Weekend mode - what's on the agenda?",
    ]

    greeting = random.choice(greetings.get(time_of_day, greetings["afternoon"]))

    # Occasionally add weekend context
    if day_type == "weekend" and random.random() < 0.3:
        greeting = greeting.replace(".", f". {random.choice(weekend_additions)}")

    return greeting


def get_time_context() -> str:
    """Get time context for the system prompt."""
    now = datetime.now()
    time_of_day = get_time_of_day()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    return f"""
CURRENT CONTEXT:
- It's {time_of_day} ({time_str}) on {day_name}, {date_str}
- Adjust your energy and tone to match the time of day
- Morning: brisk and focused
- Afternoon: steady and productive
- Evening: more relaxed
- Late night: efficient and understanding of the late hour
"""


def get_dynamic_system_prompt(base_prompt: str) -> str:
    """Enhance the base system prompt with dynamic context.

    Args:
        base_prompt: The static base system prompt

    Returns:
        Enhanced prompt with time context
    """
    time_context = get_time_context()
    return f"{base_prompt}\n{time_context}"


def get_greeting_instruction() -> str:
    """Get a dynamic greeting instruction for session start."""
    greeting = get_time_aware_greeting()
    return f"Greet Will naturally. Say something like: '{greeting}' Keep it brief and confident - no more than 2 sentences."
