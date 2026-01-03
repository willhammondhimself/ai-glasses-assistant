"""
ProbabilityEngine: Probability problems for quant finance interviews.

Card problems are the most common at Jane Street, Citadel, etc.
This engine covers:
- Card probability (drawing without replacement)
- Dice probability
- Expected value calculations
- Classic problems (Monty Hall, Birthday Paradox, Two Child)
"""

import random
import math
import uuid
from typing import Optional
from math import comb, factorial
from fractions import Fraction


class ProbabilityEngine:
    """Probability problem generator for quant interview prep."""

    def __init__(self):
        self._problems = {}

    # ==================== Card Problems ====================

    def generate_card_problem(self, difficulty: int = 2) -> dict:
        """
        Generate a card probability problem.

        Args:
            difficulty: 1-4 (higher = harder)

        Returns:
            dict with problem details and solution
        """
        difficulty = max(1, min(4, difficulty))
        problem_id = str(uuid.uuid4())[:8]

        generators = [
            self._card_single_draw,
            self._card_multiple_same,
            self._card_sequence,
            self._card_conditional,
        ]

        # Weight towards appropriate difficulty
        if difficulty == 1:
            problem_text, answer, explanation = generators[0]()
        elif difficulty == 2:
            problem_text, answer, explanation = random.choice(generators[:2])()
        elif difficulty == 3:
            problem_text, answer, explanation = random.choice(generators[1:3])()
        else:
            problem_text, answer, explanation = random.choice(generators[2:])()

        self._problems[problem_id] = {'answer': answer, 'explanation': explanation}

        return {
            'problem_id': problem_id,
            'problem': problem_text,
            'problem_type': 'card',
            'difficulty': difficulty,
            'hint': self._get_card_hint(difficulty),
            'error': None
        }

    def _card_single_draw(self) -> tuple:
        """P(drawing specific card type)."""
        scenarios = [
            ("What is the probability of drawing an Ace from a standard deck?",
             Fraction(4, 52), "4 Aces / 52 cards = 1/13"),
            ("What is the probability of drawing a heart from a standard deck?",
             Fraction(13, 52), "13 hearts / 52 cards = 1/4"),
            ("What is the probability of drawing a face card (J, Q, K)?",
             Fraction(12, 52), "12 face cards / 52 cards = 3/13"),
            ("What is the probability of drawing a red card?",
             Fraction(26, 52), "26 red cards / 52 cards = 1/2"),
        ]
        problem, answer, explanation = random.choice(scenarios)
        return problem, float(answer), explanation

    def _card_multiple_same(self) -> tuple:
        """P(drawing multiple of same type without replacement)."""
        # P(2 aces in 2 draws)
        n_aces = random.randint(2, 3)
        n_draws = n_aces

        prob = 1
        for i in range(n_aces):
            prob *= (4 - i) / (52 - i)

        problem = f"Drawing {n_draws} cards without replacement, what is P(all {n_aces} are Aces)?"
        explanation = " × ".join([f"{4-i}/{52-i}" for i in range(n_aces)])

        return problem, round(prob, 6), explanation

    def _card_sequence(self) -> tuple:
        """P(specific sequence of cards)."""
        scenarios = [
            ("Drawing 2 cards, what is P(first is Ace AND second is King)?",
             (4/52) * (4/51), "P(Ace first) × P(King second) = (4/52)(4/51)"),
            ("Drawing 2 cards, what is P(both are hearts)?",
             (13/52) * (12/51), "P(heart first) × P(heart second) = (13/52)(12/51)"),
            ("Drawing 3 cards, what is P(all different suits)?",
             (52/52) * (39/51) * (26/50), "(52/52)(39/51)(26/50)"),
        ]
        problem, answer, explanation = random.choice(scenarios)
        return problem, round(answer, 6), explanation

    def _card_conditional(self) -> tuple:
        """Conditional probability with cards."""
        scenarios = [
            ("Given that a card drawn is red, what is P(it's a heart)?",
             0.5, "P(heart|red) = 13 hearts / 26 red = 1/2"),
            ("Given that a card is a face card, what is P(it's a King)?",
             4/12, "P(King|face) = 4 Kings / 12 face cards = 1/3"),
            ("You draw 2 cards. Given the first is an Ace, what is P(second is also Ace)?",
             3/51, "P(Ace₂|Ace₁) = 3 remaining Aces / 51 remaining cards"),
        ]
        problem, answer, explanation = random.choice(scenarios)
        return problem, round(answer, 6), explanation

    def _get_card_hint(self, difficulty: int) -> str:
        hints = {
            1: "Count favorable outcomes / total outcomes",
            2: "For sequential draws, multiply conditional probabilities",
            3: "Remember: without replacement changes the denominator",
            4: "Use Bayes' Theorem: P(A|B) = P(A∩B) / P(B)"
        }
        return hints.get(difficulty, "")

    # ==================== Dice Problems ====================

    def generate_dice_problem(self, difficulty: int = 2) -> dict:
        """Generate a dice probability problem."""
        difficulty = max(1, min(4, difficulty))
        problem_id = str(uuid.uuid4())[:8]

        if difficulty == 1:
            problem_text, answer, explanation = self._dice_single()
        elif difficulty == 2:
            problem_text, answer, explanation = self._dice_sum()
        elif difficulty == 3:
            problem_text, answer, explanation = self._dice_multiple()
        else:
            problem_text, answer, explanation = self._dice_complex()

        self._problems[problem_id] = {'answer': answer, 'explanation': explanation}

        return {
            'problem_id': problem_id,
            'problem': problem_text,
            'problem_type': 'dice',
            'difficulty': difficulty,
            'hint': self._get_dice_hint(difficulty),
            'error': None
        }

    def _dice_single(self) -> tuple:
        """Single die probability."""
        target = random.randint(1, 6)
        scenarios = [
            (f"Roll a fair die. What is P(getting {target})?", 1/6, "1 outcome / 6 possible = 1/6"),
            (f"Roll a fair die. What is P(getting ≥ {random.randint(2,5)})?",
             (7 - target) / 6, f"{7-target} favorable outcomes / 6"),
            ("Roll a fair die. What is P(getting an even number)?", 3/6, "3 evens {2,4,6} / 6 = 1/2"),
        ]
        return random.choice(scenarios)

    def _dice_sum(self) -> tuple:
        """Two dice sum probability."""
        target_sum = random.randint(5, 9)
        # Count ways to make target_sum with 2 dice
        ways = sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 == target_sum)
        prob = ways / 36

        return (
            f"Roll 2 fair dice. What is P(sum = {target_sum})?",
            round(prob, 4),
            f"{ways} ways to get {target_sum} / 36 total outcomes"
        )

    def _dice_multiple(self) -> tuple:
        """Multiple dice problems."""
        scenarios = [
            ("Roll 2 dice. What is P(both show same number)?",
             6/36, "6 doubles {(1,1), (2,2), ..., (6,6)} / 36 = 1/6"),
            ("Roll 3 dice. What is P(all show 6)?",
             1/216, "(1/6)³ = 1/216"),
            ("Roll 2 dice. What is P(at least one 6)?",
             11/36, "1 - P(no 6s) = 1 - (5/6)² = 11/36"),
        ]
        return random.choice(scenarios)

    def _dice_complex(self) -> tuple:
        """Complex dice problems."""
        scenarios = [
            ("Roll 3 dice. What is P(sum = 10)?",
             27/216, "Enumerate: 27 ways to sum to 10 / 216 total"),
            ("Roll 2 dice. What is P(product is even)?",
             27/36, "1 - P(both odd) = 1 - (3/6)² = 3/4"),
            ("Roll n dice until you get a 6. What is E[n]?",
             6, "Geometric distribution: E[n] = 1/p = 6"),
        ]
        return random.choice(scenarios)

    def _get_dice_hint(self, difficulty: int) -> str:
        hints = {
            1: "Each die has 6 equally likely outcomes",
            2: "Two dice have 36 total outcomes (6×6)",
            3: "P(at least one) = 1 - P(none)",
            4: "Consider complementary counting or expected value formulas"
        }
        return hints.get(difficulty, "")

    # ==================== Expected Value Problems ====================

    def generate_ev_problem(self, difficulty: int = 2) -> dict:
        """Generate an expected value problem (Jane Street favorite)."""
        difficulty = max(1, min(4, difficulty))
        problem_id = str(uuid.uuid4())[:8]

        if difficulty == 1:
            problem_text, answer, explanation = self._ev_simple()
        elif difficulty == 2:
            problem_text, answer, explanation = self._ev_betting()
        elif difficulty == 3:
            problem_text, answer, explanation = self._ev_conditional()
        else:
            problem_text, answer, explanation = self._ev_complex()

        self._problems[problem_id] = {'answer': answer, 'explanation': explanation}

        return {
            'problem_id': problem_id,
            'problem': problem_text,
            'problem_type': 'expected_value',
            'difficulty': difficulty,
            'hint': "E[X] = Σ x·P(x) for all outcomes x",
            'error': None
        }

    def _ev_simple(self) -> tuple:
        """Simple expected value."""
        return (
            "Roll a fair die. What is E[outcome]?",
            3.5,
            "E[X] = (1+2+3+4+5+6)/6 = 21/6 = 3.5"
        )

    def _ev_betting(self) -> tuple:
        """Betting/game expected value."""
        win_prob = random.choice([0.25, 0.3, 0.4, 0.5])
        win_amount = random.randint(10, 50)
        lose_amount = random.randint(5, 20)

        ev = win_prob * win_amount - (1 - win_prob) * lose_amount

        return (
            f"A game: you win ${win_amount} with probability {win_prob}, "
            f"lose ${lose_amount} otherwise. What is your expected value?",
            round(ev, 2),
            f"E[X] = {win_prob}×{win_amount} - {1-win_prob}×{lose_amount} = ${round(ev, 2)}"
        )

    def _ev_conditional(self) -> tuple:
        """Conditional expected value."""
        scenarios = [
            ("You flip a coin until you get heads. What is E[number of flips]?",
             2, "Geometric distribution: E[X] = 1/p = 1/0.5 = 2"),
            ("Roll a die. If it's ≤3, roll again and add. What is E[total]?",
             5.25, "E = 0.5×E[X|≤3] + 0.5×E[X|>3] where E[X|≤3] includes reroll"),
        ]
        return random.choice(scenarios)

    def _ev_complex(self) -> tuple:
        """Complex expected value (interview-style)."""
        scenarios = [
            ("You can bet on a coin flip: $100 on heads. If heads, you win $100. "
             "You can also choose to flip again (losing your first bet). "
             "What's optimal strategy and expected value?",
             50, "Always take first flip: E = 0.5×100 - 0.5×100 = 0, but max E[winnings] = $50"),
            ("Draw cards until you get an Ace. What is E[cards drawn]?",
             52/5, "E = 52/4 = 13... actually E = (52+1)/(4+1) ≈ 10.6 by symmetry"),
        ]
        return random.choice(scenarios)

    # ==================== Classic Problems ====================

    def monty_hall_simulation(self, iterations: int = 10000) -> dict:
        """
        Simulate the Monty Hall problem.

        Returns:
            dict with simulation results and explanation
        """
        stay_wins = 0
        switch_wins = 0

        for _ in range(iterations):
            # Car behind one door (0, 1, or 2)
            car = random.randint(0, 2)
            # Player's initial choice
            choice = random.randint(0, 2)

            # Host opens a door (not car, not player's choice)
            doors = [0, 1, 2]
            doors.remove(choice)
            if car in doors:
                doors.remove(car)
            host_opens = random.choice(doors)

            # Stay strategy
            if choice == car:
                stay_wins += 1

            # Switch strategy
            switch_to = [d for d in [0, 1, 2] if d != choice and d != host_opens][0]
            if switch_to == car:
                switch_wins += 1

        return {
            'iterations': iterations,
            'stay_win_rate': round(stay_wins / iterations, 4),
            'switch_win_rate': round(switch_wins / iterations, 4),
            'theoretical_stay': 1/3,
            'theoretical_switch': 2/3,
            'explanation': (
                "Initially: P(car behind chosen door) = 1/3, P(car behind other doors) = 2/3. "
                "When host opens a goat door, the 2/3 probability concentrates on remaining door. "
                "ALWAYS SWITCH!"
            ),
            'error': None
        }

    def birthday_paradox(self, n_people: int = 23) -> dict:
        """
        Calculate birthday paradox probability.

        Args:
            n_people: Number of people in room

        Returns:
            dict with probability and explanation
        """
        if n_people > 365:
            prob = 1.0
        else:
            # P(no match) = 365/365 × 364/365 × ... × (366-n)/365
            prob_no_match = 1.0
            for i in range(n_people):
                prob_no_match *= (365 - i) / 365

            prob = 1 - prob_no_match

        return {
            'n_people': n_people,
            'probability_match': round(prob, 6),
            'probability_no_match': round(1 - prob, 6),
            'formula': f"P(match) = 1 - ∏(365-i)/365 for i=0 to {n_people-1}",
            'key_insight': "At n=23, P(match) ≈ 50.7%. At n=50, P(match) ≈ 97%",
            'error': None
        }

    def two_child_problem(self, variant: str = 'at_least_one_boy') -> dict:
        """
        Solve variants of the two-child problem.

        Args:
            variant: 'at_least_one_boy', 'older_is_boy', or 'boy_named_tuesday'

        Returns:
            dict with solution and explanation
        """
        if variant == 'older_is_boy':
            # Given older child is boy, P(both boys)?
            return {
                'problem': "A family has two children. The older child is a boy. What is P(both children are boys)?",
                'answer': 0.5,
                'explanation': "Sample space: {BB, BG}. Both equally likely. P(BB) = 1/2",
                'error': None
            }
        elif variant == 'boy_named_tuesday':
            # Given one is "a boy born on Tuesday"
            return {
                'problem': "A family has two children. One is a boy born on Tuesday. What is P(both are boys)?",
                'answer': 13/27,
                'explanation': (
                    "Sample space: 27 cases where at least one is boy-Tuesday. "
                    "13 cases are two boys. P = 13/27 ≈ 0.481"
                ),
                'error': None
            }
        else:  # at_least_one_boy
            return {
                'problem': "A family has two children. At least one is a boy. What is P(both are boys)?",
                'answer': 1/3,
                'explanation': "Sample space: {BB, BG, GB}. Only BB has both boys. P(BB|at least one B) = 1/3",
                'error': None
            }

    # ==================== Answer Checking ====================

    def check_answer(self, problem_id: str, user_answer: str) -> dict:
        """Check user's answer to a probability problem."""
        if problem_id not in self._problems:
            return {"error": "Problem not found. Generate a new problem."}

        problem_data = self._problems[problem_id]
        correct_answer = problem_data['answer']
        explanation = problem_data['explanation']

        # Parse user answer
        try:
            if '/' in str(user_answer):
                parts = user_answer.split('/')
                user_val = float(parts[0]) / float(parts[1])
            elif '%' in str(user_answer):
                user_val = float(user_answer.replace('%', '')) / 100
            else:
                user_val = float(user_answer)

            # Allow tolerance for floating point
            is_correct = abs(user_val - correct_answer) < 0.001

        except (ValueError, TypeError):
            is_correct = False
            user_val = user_answer

        del self._problems[problem_id]

        return {
            'correct': is_correct,
            'correct_answer': correct_answer,
            'user_answer': user_val,
            'explanation': explanation,
            'error': None
        }

    # ==================== General Problem Generator ====================

    def generate_problem(self, problem_type: str = None, difficulty: int = 2) -> dict:
        """Generate any type of probability problem."""
        types = ['card', 'dice', 'expected_value']

        if problem_type is None:
            problem_type = random.choice(types)

        if problem_type == 'card':
            return self.generate_card_problem(difficulty)
        elif problem_type == 'dice':
            return self.generate_dice_problem(difficulty)
        elif problem_type == 'expected_value':
            return self.generate_ev_problem(difficulty)
        else:
            return {"error": f"Invalid problem type. Choose from: {types}"}
