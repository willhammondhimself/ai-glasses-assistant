"""
Poker Prompt Builder - Optimized prompts for DeepSeek.
Builds structured prompts for live hand analysis and post-session review.
"""
from typing import List, Dict, Optional


def build_poker_prompt(
    hero_cards: List[str],
    board: List[str],
    pot_size: float,
    bet_size: float,
    villain_stats: Dict,
    position: str,
    stack_size: float,
    action_sequence: str = "",
    stakes: str = ""
) -> str:
    """
    Build optimal prompt for DeepSeek V3.1 live analysis.

    Args:
        hero_cards: ["Ah", "Kc"]
        board: ["9s", "7h", "3d"] or []
        pot_size: Current pot in big blinds
        bet_size: Bet hero is facing (0 if no bet)
        villain_stats: From OpponentTracker.get_stats()
        position: BTN, CO, MP, UTG, SB, BB
        stack_size: Hero's stack in big blinds
        action_sequence: Optional action history
        stakes: Optional stakes description

    Returns:
        Optimized prompt string
    """
    # Extract villain info
    villain_type = villain_stats.get("type", "unknown")
    vpip = villain_stats.get("vpip", 0.25)
    aggression = villain_stats.get("aggression", 0.40)
    fold_to_cbet = villain_stats.get("fold_to_cbet", 0.50)
    hands_observed = villain_stats.get("hands_observed", 0)

    # Calculate pot odds if facing bet
    if bet_size > 0:
        pot_odds = bet_size / (pot_size + bet_size)
        pot_odds_str = f"{pot_odds:.1%}"
    else:
        pot_odds_str = "N/A"

    # Format board
    board_str = ' '.join(board) if board else "Preflop"

    # Build prompt
    prompt = f"""You are a poker coach analyzing a live hand. Be concise and exploitative.

SITUATION:
Hero: {' '.join(hero_cards)} | Position: {position} | Stack: {stack_size:.0f}bb
Board: {board_str}
Pot: {pot_size:.1f}bb | Facing: {bet_size:.1f}bb | Pot Odds: {pot_odds_str}"""

    if action_sequence:
        prompt += f"\nAction: {action_sequence}"

    if stakes:
        prompt += f"\nStakes: {stakes}"

    # Villain section
    prompt += f"""

VILLAIN ({hands_observed} hands observed):
Type: {villain_type}
VPIP: {vpip:.0%} | Aggression: {aggression:.0%} | Fold to C-bet: {fold_to_cbet:.0%}"""

    # Add notes if any
    notes = villain_stats.get("notes", [])
    if notes:
        prompt += "\nNotes: " + "; ".join(notes[-2:])  # Last 2 notes

    # Task section
    prompt += """

TASK:
1. Estimate equity vs villain's likely range
2. Recommend: FOLD / CALL / RAISE (with sizing if raising)
3. Explain exploitative adjustment vs this villain type (2 sentences max)

FORMAT:
ACTION: [FOLD/CALL/RAISE]
SIZING: [if raising: 1/2 pot, 2/3 pot, 3/4 pot, pot, all-in]
EQUITY: [estimated % vs villain range]
REASONING: [brief exploitative explanation]"""

    return prompt


def build_preflop_prompt(
    hero_cards: List[str],
    position: str,
    villain_stats: Dict,
    action_to_hero: str,
    stack_size: float,
    stakes: str = ""
) -> str:
    """
    Build prompt for preflop decision.

    Args:
        hero_cards: ["Ah", "Kc"]
        position: BTN, CO, MP, UTG, SB, BB
        villain_stats: From OpponentTracker
        action_to_hero: "fold to hero", "1 limper", "raise to 3bb", etc.
        stack_size: Hero's stack in big blinds
        stakes: Optional stakes description

    Returns:
        Preflop analysis prompt
    """
    villain_type = villain_stats.get("type", "unknown")
    vpip = villain_stats.get("vpip", 0.25)
    hands = villain_stats.get("hands_observed", 0)

    return f"""Preflop decision. Be concise.

HERO: {' '.join(hero_cards)} | Position: {position} | Stack: {stack_size:.0f}bb
Action to hero: {action_to_hero}

VILLAIN ({hands} hands): {villain_type} (VPIP {vpip:.0%})

What should hero do? Consider position, stack depth, and villain tendencies.

FORMAT:
ACTION: [FOLD/CALL/RAISE]
SIZING: [if raising]
REASONING: [one sentence]"""


def build_review_prompt(
    hands: List[Dict],
    session_context: str = "",
    focus: str = ""
) -> str:
    """
    Build prompt for post-session hand review.

    Args:
        hands: List of hand dictionaries
        session_context: Stakes, duration, etc.
        focus: Specific area to review (optional)

    Returns:
        Review prompt for DeepSeek V3.2 thinking mode
    """
    prompt = f"""Review this poker session and identify mistakes/leaks.

SESSION: {session_context if session_context else "Unknown stakes"}
HANDS REVIEWED: {len(hands)}
"""

    if focus:
        prompt += f"FOCUS AREA: {focus}\n"

    prompt += "\n--- HANDS ---\n"

    for i, hand in enumerate(hands, 1):
        prompt += f"\nHand {i}:\n"
        prompt += f"  Hero: {hand.get('hero_cards', '?')}\n"
        prompt += f"  Board: {hand.get('board', '?')}\n"
        prompt += f"  Villain: {hand.get('villain_type', 'unknown')}\n"
        prompt += f"  Action: {hand.get('action_taken', '?')}\n"
        prompt += f"  Result: {hand.get('result', '?')}\n"

        if hand.get("notes"):
            prompt += f"  Notes: {hand['notes']}\n"

    prompt += """

ANALYZE:
1. Identify the 3 biggest mistakes (be specific about the error)
2. Spot any patterns or leaks in hero's play
3. Grade each mistake (minor/moderate/severe)
4. Suggest specific, actionable improvements
5. Rate overall session play (1-10)

Be direct and specific. Reference hand numbers."""

    return prompt


def build_range_query(
    villain_stats: Dict,
    action: str,
    position: str,
    board: List[str] = None
) -> str:
    """
    Build prompt to estimate villain's range.

    Args:
        villain_stats: Villain profile
        action: What villain did (raised, called, check-raised, etc.)
        position: Villain's position
        board: Community cards if postflop

    Returns:
        Range estimation prompt
    """
    villain_type = villain_stats.get("type", "unknown")
    vpip = villain_stats.get("vpip", 0.25)
    aggression = villain_stats.get("aggression", 0.40)
    hands = villain_stats.get("hands_observed", 0)

    board_str = ' '.join(board) if board else "Preflop"

    return f"""Estimate villain's range given their action.

VILLAIN ({hands} hands): {villain_type}
Stats: VPIP {vpip:.0%}, Aggression {aggression:.0%}
Position: {position}
Board: {board_str}
Action: {action}

What hands are in villain's range? Be specific.

FORMAT:
STRONG: [top of range, e.g., "AA-JJ, AK"]
MEDIUM: [middle of range]
WEAK/BLUFFS: [bottom of range]
RANGE WIDTH: [narrow/medium/wide]"""


# Exploitative strategy templates
EXPLOIT_TEMPLATES = {
    "calling_station": """vs Calling Station:
- Value bet thin (top pair good kicker+)
- Never bluff (they don't fold)
- Size up value bets (3/4 pot to pot)
- Don't slow play (they call anyway)""",

    "tight_passive": """vs Tight-Passive:
- Steal relentlessly (they fold too much)
- Respect their aggression (they have it)
- Bluff when they check
- Fold to their raises""",

    "maniac": """vs Maniac:
- Call down lighter (second pair+)
- Trap with monsters
- Let them bluff (don't bluff them)
- Check-raise for value""",

    "tag": """vs TAG:
- Play straightforward
- Mix in 3-bet bluffs
- Respect postflop raises
- Don't over-adjust""",

    "lag": """vs LAG:
- Widen calling range slightly
- 3-bet for value more
- Don't get into leveling wars
- Stay disciplined"""
}


def get_exploit_template(villain_type: str) -> str:
    """Get exploitative strategy template for villain type."""
    return EXPLOIT_TEMPLATES.get(villain_type, "Play standard poker.")


# Test
def test_prompts():
    """Test prompt builders."""
    print("=== Prompt Builder Test ===\n")

    villain = {
        "type": "calling_station",
        "vpip": 0.48,
        "aggression": 0.25,
        "fold_to_cbet": 0.30,
        "hands_observed": 45,
        "notes": ["Never folds draws", "Calls river with any pair"]
    }

    prompt = build_poker_prompt(
        hero_cards=["Ah", "Kc"],
        board=["9s", "7h", "3d"],
        pot_size=12,
        bet_size=8,
        villain_stats=villain,
        position="BTN",
        stack_size=100
    )

    print("Live Analysis Prompt:")
    print("-" * 40)
    print(prompt)
    print()

    print("Exploit Template:")
    print("-" * 40)
    print(get_exploit_template("calling_station"))


if __name__ == "__main__":
    test_prompts()
