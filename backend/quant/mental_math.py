"""
MentalMathEngine: Speed arithmetic for quant finance interviews.

Difficulty Levels (Jane Street calibrated):
- Level 1: Single digit operations (warmup)
- Level 2: Two-digit operations (3-5 seconds target)
- Level 3: Three-digit or complex operations (5-10 seconds)
- Level 4: Multi-step or large number operations (10-15 seconds)

Problem Types:
- multiplication: 2-digit × 2-digit, etc.
- division: Mental division with remainders
- percentage: Percentage calculations
- fraction_decimal: Fraction to decimal conversions
- square_root: Perfect and approximate square roots
"""

import random
import math
import uuid
from typing import Optional
from fractions import Fraction


class MentalMathEngine:
    """Mental math problem generator for quant interview prep."""

    # Time targets in milliseconds by difficulty
    TIME_TARGETS = {
        1: 3000,   # 3 seconds
        2: 5000,   # 5 seconds (Jane Street standard for 2-digit × 2-digit)
        3: 10000,  # 10 seconds
        4: 15000,  # 15 seconds
    }

    # Problem types
    PROBLEM_TYPES = ['multiplication', 'division', 'percentage', 'fraction_decimal', 'square_root']

    def __init__(self):
        # Store generated problems for answer verification
        self._problems = {}

    def generate_problem(self, problem_type: str = None, difficulty: int = 2) -> dict:
        """
        Generate a mental math problem.

        Args:
            problem_type: One of PROBLEM_TYPES, or None for random
            difficulty: 1-4 (higher = harder)

        Returns:
            dict with problem_id, problem, answer, time_target_ms, problem_type, difficulty
        """
        difficulty = max(1, min(4, difficulty))

        if problem_type is None:
            problem_type = random.choice(self.PROBLEM_TYPES)

        if problem_type not in self.PROBLEM_TYPES:
            return {"error": f"Invalid problem type. Choose from: {self.PROBLEM_TYPES}"}

        # Generate problem based on type
        generators = {
            'multiplication': self._gen_multiplication,
            'division': self._gen_division,
            'percentage': self._gen_percentage,
            'fraction_decimal': self._gen_fraction_decimal,
            'square_root': self._gen_square_root,
        }

        problem_text, answer = generators[problem_type](difficulty)
        problem_id = str(uuid.uuid4())[:8]

        # Store for verification
        self._problems[problem_id] = {
            'answer': answer,
            'type': problem_type,
            'difficulty': difficulty,
            'time_target_ms': self.TIME_TARGETS[difficulty]
        }

        return {
            'problem_id': problem_id,
            'problem': problem_text,
            'problem_type': problem_type,
            'difficulty': difficulty,
            'time_target_ms': self.TIME_TARGETS[difficulty],
            'hint': self._get_hint(problem_type, difficulty),
            'error': None
        }

    def check_answer(self, problem_id: str, user_answer: str, time_ms: int) -> dict:
        """
        Check user's answer and calculate score with time bonus.

        Args:
            problem_id: ID from generate_problem
            user_answer: User's answer (string)
            time_ms: Time taken in milliseconds

        Returns:
            dict with correct, correct_answer, score, time_bonus, feedback
        """
        if problem_id not in self._problems:
            return {"error": "Problem not found. Generate a new problem."}

        problem_data = self._problems[problem_id]
        correct_answer = problem_data['answer']
        time_target = problem_data['time_target_ms']

        # Parse user answer
        try:
            user_val = self._parse_answer(user_answer)
            correct_val = self._parse_answer(str(correct_answer))

            # Check if correct (allow small floating point tolerance)
            if isinstance(correct_val, float):
                is_correct = abs(user_val - correct_val) < 0.01
            else:
                is_correct = user_val == correct_val
        except (ValueError, TypeError):
            is_correct = str(user_answer).strip().lower() == str(correct_answer).strip().lower()

        # Calculate time bonus
        if is_correct:
            if time_ms <= time_target * 0.5:
                time_bonus = 50  # Double speed bonus
                time_feedback = "Lightning fast!"
            elif time_ms <= time_target:
                time_bonus = 25  # On target bonus
                time_feedback = "Great speed!"
            elif time_ms <= time_target * 1.5:
                time_bonus = 10  # Slightly over
                time_feedback = "Good, but try to be faster"
            else:
                time_bonus = 0
                time_feedback = "Work on speed"

            base_score = 100
            score = base_score + time_bonus
        else:
            time_bonus = 0
            score = 0
            time_feedback = "Incorrect"

        # Clean up
        del self._problems[problem_id]

        return {
            'correct': is_correct,
            'correct_answer': correct_answer,
            'user_answer': user_answer,
            'score': score,
            'time_bonus': time_bonus,
            'time_ms': time_ms,
            'time_target_ms': time_target,
            'feedback': time_feedback,
            'error': None
        }

    def _parse_answer(self, answer: str) -> float:
        """Parse answer string to numeric value."""
        answer = str(answer).strip().replace(',', '')

        # Handle fractions
        if '/' in answer:
            parts = answer.split('/')
            return float(parts[0]) / float(parts[1])

        # Handle percentages
        if '%' in answer:
            return float(answer.replace('%', '')) / 100

        return float(answer)

    def _gen_multiplication(self, difficulty: int) -> tuple:
        """Generate multiplication problem."""
        if difficulty == 1:
            a = random.randint(2, 12)
            b = random.randint(2, 12)
        elif difficulty == 2:
            # Jane Street standard: 2-digit × 2-digit
            a = random.randint(11, 99)
            b = random.randint(11, 99)
        elif difficulty == 3:
            a = random.randint(11, 99)
            b = random.randint(100, 999)
        else:
            a = random.randint(100, 999)
            b = random.randint(100, 999)

        # Sometimes use "nice" numbers that have shortcuts
        if random.random() < 0.3:
            nice_numbers = [11, 25, 50, 99, 101, 125, 250, 500]
            a = random.choice([n for n in nice_numbers if n < 10 ** difficulty])

        return f"{a} × {b}", a * b

    def _gen_division(self, difficulty: int) -> tuple:
        """Generate division problem (clean integer results)."""
        if difficulty == 1:
            b = random.randint(2, 12)
            answer = random.randint(2, 12)
        elif difficulty == 2:
            b = random.randint(2, 25)
            answer = random.randint(10, 99)
        elif difficulty == 3:
            b = random.randint(2, 50)
            answer = random.randint(10, 200)
        else:
            b = random.randint(11, 99)
            answer = random.randint(50, 500)

        a = b * answer

        # Sometimes include remainder problems
        if difficulty >= 3 and random.random() < 0.3:
            remainder = random.randint(1, b - 1)
            a += remainder
            return f"{a} ÷ {b} (include remainder)", f"{answer} r{remainder}"

        return f"{a} ÷ {b}", answer

    def _gen_percentage(self, difficulty: int) -> tuple:
        """Generate percentage problem."""
        if difficulty == 1:
            # Simple percentages: 10%, 25%, 50%
            pct = random.choice([10, 20, 25, 50])
            base = random.randint(2, 20) * 10
        elif difficulty == 2:
            pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
            base = random.randint(10, 200)
        elif difficulty == 3:
            pct = random.randint(1, 99)
            base = random.randint(50, 500)
        else:
            # Complex: "What percent is X of Y?"
            answer = random.randint(5, 95)
            base = random.randint(100, 1000)
            part = round(base * answer / 100)
            return f"What percent is {part} of {base}?", round(part / base * 100, 2)

        answer = round(base * pct / 100, 2)
        if answer == int(answer):
            answer = int(answer)

        return f"{pct}% of {base}", answer

    def _gen_fraction_decimal(self, difficulty: int) -> tuple:
        """Generate fraction to decimal conversion."""
        if difficulty == 1:
            # Common fractions
            fractions = [(1, 2), (1, 4), (3, 4), (1, 5), (2, 5)]
            num, denom = random.choice(fractions)
        elif difficulty == 2:
            # Thirds, sixths, eighths
            fractions = [(1, 3), (2, 3), (1, 6), (5, 6), (1, 8), (3, 8), (5, 8), (7, 8)]
            num, denom = random.choice(fractions)
        elif difficulty == 3:
            # Random proper fractions
            denom = random.choice([7, 9, 11, 12, 15, 16])
            num = random.randint(1, denom - 1)
        else:
            # Improper fractions
            denom = random.randint(3, 16)
            num = random.randint(denom + 1, denom * 3)

        answer = round(num / denom, 4)

        return f"Convert {num}/{denom} to decimal", answer

    def _gen_square_root(self, difficulty: int) -> tuple:
        """Generate square root problem."""
        if difficulty == 1:
            # Perfect squares up to 144
            root = random.randint(2, 12)
            return f"√{root ** 2}", root
        elif difficulty == 2:
            # Perfect squares up to 625
            root = random.randint(10, 25)
            return f"√{root ** 2}", root
        elif difficulty == 3:
            # Approximate roots (to 1 decimal)
            n = random.randint(50, 200)
            while int(math.sqrt(n)) ** 2 == n:  # Avoid perfect squares
                n = random.randint(50, 200)
            answer = round(math.sqrt(n), 1)
            return f"√{n} (to 1 decimal)", answer
        else:
            # Larger perfect squares or cube roots
            if random.random() < 0.5:
                root = random.randint(20, 50)
                return f"√{root ** 2}", root
            else:
                root = random.randint(3, 10)
                return f"∛{root ** 3}", root

    def _get_hint(self, problem_type: str, difficulty: int) -> str:
        """Return a strategy hint for the problem type."""
        hints = {
            'multiplication': {
                1: "Use multiplication tables",
                2: "Break into (a×100) + (a×b). Example: 23×47 = 23×50 - 23×3",
                3: "Use distributive property: a×b = a×(b₁+b₂)",
                4: "Look for patterns: 25×n = n/4 × 100"
            },
            'division': {
                1: "Use division facts",
                2: "Factor when possible: 72÷12 = 72÷4÷3",
                3: "Use estimation and adjust",
                4: "Break into simpler divisions"
            },
            'percentage': {
                1: "10% = divide by 10, 50% = divide by 2",
                2: "25% = divide by 4, 75% = 50% + 25%",
                3: "Find 10% first, then combine",
                4: "Part ÷ Whole × 100 = Percentage"
            },
            'fraction_decimal': {
                1: "1/2=0.5, 1/4=0.25, 1/5=0.2",
                2: "1/3≈0.333, 1/6≈0.167, 1/8=0.125",
                3: "Divide numerator by denominator",
                4: "Convert to equivalent fraction with base 10 denominator"
            },
            'square_root': {
                1: "Memorize: √4=2, √9=3, √16=4, √25=5...",
                2: "√100=10, √400=20, √625=25",
                3: "Estimate between perfect squares",
                4: "For √n: start at √100=10, adjust"
            }
        }
        return hints.get(problem_type, {}).get(difficulty, "Work carefully")

    def get_problem_types(self) -> list:
        """Return available problem types."""
        return self.PROBLEM_TYPES

    def get_difficulty_info(self) -> dict:
        """Return difficulty level information."""
        return {
            1: {
                'name': 'Warmup',
                'time_target_ms': self.TIME_TARGETS[1],
                'description': 'Single digit operations, basic facts'
            },
            2: {
                'name': 'Interview Standard',
                'time_target_ms': self.TIME_TARGETS[2],
                'description': 'Two-digit operations (Jane Street level)'
            },
            3: {
                'name': 'Advanced',
                'time_target_ms': self.TIME_TARGETS[3],
                'description': 'Three-digit or complex operations'
            },
            4: {
                'name': 'Expert',
                'time_target_ms': self.TIME_TARGETS[4],
                'description': 'Multi-step or large number operations'
            }
        }
