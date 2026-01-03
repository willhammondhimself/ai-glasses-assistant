"""
PokerEngine: GTO-based poker hand analysis using Claude Sonnet 3.5.

Provides strategic advice based on Game Theory Optimal (GTO) principles
without requiring external solver integration.
"""

import os
from typing import Optional, List, Dict
import anthropic
from dotenv import load_dotenv

load_dotenv()


class PokerEngine:
    """Poker strategy assistant powered by Claude Sonnet 3.5."""

    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_hand(self, hand_info: dict) -> dict:
        """
        Analyze a poker hand and provide GTO-based recommendations.

        Args:
            hand_info: Dictionary containing:
                - hero_cards: str (e.g., "AhKs" or "Ace of hearts, King of spades")
                - board: str (e.g., "Qh Jd 2c" or empty for preflop)
                - position: str (e.g., "BTN", "CO", "MP", "BB", "SB")
                - villain_position: str (optional)
                - pot_size: float (current pot in BBs)
                - stack_size: float (hero's stack in BBs)
                - villain_stack: float (optional, villain's stack in BBs)
                - action: str (what action hero faces, e.g., "facing 3BB raise")
                - game_type: str (optional, e.g., "cash 6max", "MTT", "HU")
                - villain_tendencies: str (optional, e.g., "tight", "aggressive")

        Returns:
            dict with keys: recommendation, reasoning, ev_estimate, range_analysis, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            # Build the hand description
            hand_description = self._build_hand_description(hand_info)

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system="""You are an expert poker strategist with deep knowledge of GTO (Game Theory Optimal) play.
Analyze hands from a GTO perspective while considering exploitative adjustments when villain tendencies are provided.
Be concise but thorough. Focus on actionable advice.""",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this poker hand:

{hand_description}

Provide:
1. RECOMMENDATION: What action to take (fold/call/raise) with sizing if applicable
2. REASONING: Brief GTO-based explanation (2-3 sentences)
3. EV_ESTIMATE: Rough expected value assessment (positive/negative/neutral)
4. RANGE_ANALYSIS: How this hand fits in hero's range for this spot

Format your response exactly as:
RECOMMENDATION: [action and sizing]
REASONING: [GTO-based explanation]
EV_ESTIMATE: [+EV/-EV/neutral with brief explanation]
RANGE_ANALYSIS: [where this hand falls in the range]"""
                    }
                ]
            )

            return self._parse_analysis_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def get_preflop_range(self, position: str, action: str, game_type: str = "cash 6max") -> dict:
        """
        Get GTO preflop ranges for a given position and action.

        Args:
            position: Position at the table (UTG, MP, CO, BTN, SB, BB)
            action: The action context (e.g., "RFI", "vs 3bet", "facing raise")
            game_type: Type of game (cash 6max, cash 9max, MTT, etc.)

        Returns:
            dict with keys: range, description, adjustments, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Provide GTO preflop ranges for:
Position: {position}
Action: {action}
Game Type: {game_type}

Format your response as:
RANGE: [list hands in standard notation, grouped by strength]
DESCRIPTION: [brief description of the range]
ADJUSTMENTS: [common adjustments based on table dynamics]"""
                    }
                ]
            )

            return self._parse_range_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def calculate_pot_odds(self, pot_size: float, bet_size: float, stack_size: float = None) -> dict:
        """
        Calculate pot odds and implied odds.

        Args:
            pot_size: Current pot size (in BBs or chips)
            bet_size: Size of bet hero faces
            stack_size: Remaining stack (for implied odds calculation)

        Returns:
            dict with keys: pot_odds, break_even_equity, implied_odds_factor, error
        """
        try:
            # Pot odds calculation
            total_pot = pot_size + bet_size
            pot_odds = bet_size / total_pot
            break_even_equity = pot_odds * 100

            # Implied odds factor (if stack provided)
            implied_odds_factor = None
            if stack_size and stack_size > bet_size:
                implied_odds_factor = (stack_size - bet_size) / bet_size

            return {
                "pot_odds": f"{pot_odds:.2%}",
                "break_even_equity": f"{break_even_equity:.1f}%",
                "implied_odds_factor": f"{implied_odds_factor:.1f}x" if implied_odds_factor else "N/A",
                "explanation": f"Need {break_even_equity:.1f}% equity to call. Pot is offering {1/pot_odds:.1f}:1",
                "error": None
            }
        except Exception as e:
            return {
                "error": f"Calculation error: {str(e)}"
            }

    def analyze_bet_sizing(self, situation: dict) -> dict:
        """
        Analyze optimal bet sizing for a given situation.

        Args:
            situation: Dictionary containing:
                - hero_cards: str
                - board: str
                - pot_size: float
                - position: str
                - action_type: str (e.g., "cbet", "value bet", "bluff")

        Returns:
            dict with keys: recommended_size, sizing_rationale, alternatives, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze optimal bet sizing for this situation:

Hero: {situation.get('hero_cards', 'Unknown')}
Board: {situation.get('board', 'Preflop')}
Pot: {situation.get('pot_size', 'Unknown')} BBs
Position: {situation.get('position', 'Unknown')}
Action: {situation.get('action_type', 'bet')}

Provide:
RECOMMENDED_SIZE: [size as % of pot or BBs]
SIZING_RATIONALE: [why this size is optimal]
ALTERNATIVES: [other viable sizes and when to use them]"""
                    }
                ]
            )

            return self._parse_sizing_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def _build_hand_description(self, hand_info: dict) -> str:
        """Build a formatted hand description for the prompt."""
        lines = []

        # Required fields
        lines.append(f"Hero's Hand: {hand_info.get('hero_cards', 'Unknown')}")

        if hand_info.get('board'):
            lines.append(f"Board: {hand_info['board']}")
        else:
            lines.append("Street: Preflop")

        if hand_info.get('position'):
            lines.append(f"Hero's Position: {hand_info['position']}")

        if hand_info.get('villain_position'):
            lines.append(f"Villain's Position: {hand_info['villain_position']}")

        if hand_info.get('pot_size'):
            lines.append(f"Pot Size: {hand_info['pot_size']} BBs")

        if hand_info.get('stack_size'):
            lines.append(f"Hero's Stack: {hand_info['stack_size']} BBs")

        if hand_info.get('villain_stack'):
            lines.append(f"Villain's Stack: {hand_info['villain_stack']} BBs")

        if hand_info.get('action'):
            lines.append(f"Action: {hand_info['action']}")

        if hand_info.get('game_type'):
            lines.append(f"Game Type: {hand_info['game_type']}")

        if hand_info.get('villain_tendencies'):
            lines.append(f"Villain Tendencies: {hand_info['villain_tendencies']}")

        return "\n".join(lines)

    def _parse_analysis_response(self, response_text: str) -> dict:
        """Parse hand analysis response."""
        import re

        recommendation = None
        reasoning = None
        ev_estimate = None
        range_analysis = None

        # Parse RECOMMENDATION
        rec_match = re.search(r'RECOMMENDATION:\s*(.+?)(?=REASONING:|$)', response_text, re.DOTALL)
        if rec_match:
            recommendation = rec_match.group(1).strip()

        # Parse REASONING
        reas_match = re.search(r'REASONING:\s*(.+?)(?=EV_ESTIMATE:|$)', response_text, re.DOTALL)
        if reas_match:
            reasoning = reas_match.group(1).strip()

        # Parse EV_ESTIMATE
        ev_match = re.search(r'EV_ESTIMATE:\s*(.+?)(?=RANGE_ANALYSIS:|$)', response_text, re.DOTALL)
        if ev_match:
            ev_estimate = ev_match.group(1).strip()

        # Parse RANGE_ANALYSIS
        range_match = re.search(r'RANGE_ANALYSIS:\s*(.+?)$', response_text, re.DOTALL)
        if range_match:
            range_analysis = range_match.group(1).strip()

        return {
            "recommendation": recommendation,
            "reasoning": reasoning,
            "ev_estimate": ev_estimate,
            "range_analysis": range_analysis,
            "error": None
        }

    def _parse_range_response(self, response_text: str) -> dict:
        """Parse preflop range response."""
        import re

        range_str = None
        description = None
        adjustments = None

        # Parse RANGE
        range_match = re.search(r'RANGE:\s*(.+?)(?=DESCRIPTION:|$)', response_text, re.DOTALL)
        if range_match:
            range_str = range_match.group(1).strip()

        # Parse DESCRIPTION
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=ADJUSTMENTS:|$)', response_text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        # Parse ADJUSTMENTS
        adj_match = re.search(r'ADJUSTMENTS:\s*(.+?)$', response_text, re.DOTALL)
        if adj_match:
            adjustments = adj_match.group(1).strip()

        return {
            "range": range_str,
            "description": description,
            "adjustments": adjustments,
            "error": None
        }

    def _parse_sizing_response(self, response_text: str) -> dict:
        """Parse bet sizing response."""
        import re

        recommended_size = None
        sizing_rationale = None
        alternatives = None

        # Parse RECOMMENDED_SIZE
        size_match = re.search(r'RECOMMENDED_SIZE:\s*(.+?)(?=SIZING_RATIONALE:|$)', response_text, re.DOTALL)
        if size_match:
            recommended_size = size_match.group(1).strip()

        # Parse SIZING_RATIONALE
        rat_match = re.search(r'SIZING_RATIONALE:\s*(.+?)(?=ALTERNATIVES:|$)', response_text, re.DOTALL)
        if rat_match:
            sizing_rationale = rat_match.group(1).strip()

        # Parse ALTERNATIVES
        alt_match = re.search(r'ALTERNATIVES:\s*(.+?)$', response_text, re.DOTALL)
        if alt_match:
            alternatives = alt_match.group(1).strip()

        return {
            "recommended_size": recommended_size,
            "sizing_rationale": sizing_rationale,
            "alternatives": alternatives,
            "error": None
        }

    def _no_api_key_error(self) -> dict:
        """Return error for missing API key."""
        return {
            "recommendation": None,
            "reasoning": None,
            "ev_estimate": None,
            "range_analysis": None,
            "error": "ANTHROPIC_API_KEY not configured"
        }

    def _api_error(self, e: anthropic.APIError) -> dict:
        """Return error for API errors."""
        return {
            "recommendation": None,
            "reasoning": None,
            "ev_estimate": None,
            "range_analysis": None,
            "error": f"Claude API error: {str(e)}"
        }

    def _unexpected_error(self, e: Exception) -> dict:
        """Return error for unexpected errors."""
        return {
            "recommendation": None,
            "reasoning": None,
            "ev_estimate": None,
            "range_analysis": None,
            "error": f"Unexpected error: {str(e)}"
        }
