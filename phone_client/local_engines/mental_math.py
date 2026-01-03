"""
Pure Python Mental Math Engine.
No API calls - instant response for <100ms feedback.
Parses spoken math like "47 times 83" and computes answers.
"""
import re
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class ProblemType(Enum):
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    PERCENTAGE = "percentage"
    SQUARE = "square"
    SQUARE_ROOT = "square_root"


@dataclass
class MathProblem:
    """A mental math problem."""
    problem_text: str       # Display text: "47 x 83"
    spoken_text: str        # For TTS: "47 times 83"
    answer: float           # Correct answer
    problem_type: ProblemType
    difficulty: int         # 1-5
    time_target_ms: int     # Target solve time


@dataclass
class AnswerResult:
    """Result of checking an answer."""
    correct: bool
    expected: float
    given: float
    time_ms: float
    within_target: bool


# Time targets in milliseconds (Jane Street calibrated)
TIME_TARGETS = {
    1: 2000,   # D1: 2 seconds - warmup
    2: 4000,   # D2: 4 seconds - interview standard
    3: 8000,   # D3: 8 seconds - advanced
    4: 12000,  # D4: 12 seconds - expert
    5: 20000,  # D5: 20 seconds - extreme
}

# Word to number mapping for parsing spoken input
WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
}

# Operation word mappings
OP_WORDS = {
    'times': '*', 'multiplied': '*', 'x': '*', 'multiply': '*',
    'divided': '/', 'over': '/', 'divide': '/',
    'plus': '+', 'add': '+', 'added': '+',
    'minus': '-', 'subtract': '-', 'less': '-',
    'percent': '%', 'percentage': '%',
    'squared': '^2', 'square': '^2',
    'root': 'sqrt', 'sqrt': 'sqrt',
}


class MentalMathEngine:
    """
    Pure Python mental math engine.
    Generates problems and evaluates answers with sub-millisecond latency.
    """

    def __init__(self, difficulty: int = 2):
        self.difficulty = max(1, min(5, difficulty))
        self._problem_generators = {
            ProblemType.MULTIPLICATION: self._gen_multiplication,
            ProblemType.DIVISION: self._gen_division,
            ProblemType.ADDITION: self._gen_addition,
            ProblemType.SUBTRACTION: self._gen_subtraction,
            ProblemType.PERCENTAGE: self._gen_percentage,
            ProblemType.SQUARE: self._gen_square,
        }

    def generate_problem(
        self,
        problem_type: Optional[ProblemType] = None,
        difficulty: Optional[int] = None
    ) -> MathProblem:
        """
        Generate a random math problem.

        Args:
            problem_type: Specific type or None for random
            difficulty: Override difficulty or use instance default
        """
        diff = difficulty or self.difficulty
        diff = max(1, min(5, diff))

        if problem_type is None:
            # Weight towards multiplication (most common in interviews)
            weights = [0.35, 0.2, 0.15, 0.15, 0.1, 0.05]
            problem_type = random.choices(list(ProblemType)[:6], weights=weights)[0]

        generator = self._problem_generators.get(problem_type, self._gen_multiplication)
        return generator(diff)

    def _gen_multiplication(self, difficulty: int) -> MathProblem:
        """Generate multiplication problem."""
        ranges = {
            1: (2, 12, 2, 12),       # Single digit
            2: (11, 99, 2, 12),      # 2-digit x 1-digit
            3: (11, 99, 11, 99),     # 2-digit x 2-digit
            4: (100, 999, 11, 99),   # 3-digit x 2-digit
            5: (100, 999, 100, 999), # 3-digit x 3-digit
        }
        a_min, a_max, b_min, b_max = ranges[difficulty]

        a = random.randint(a_min, a_max)
        b = random.randint(b_min, b_max)

        # Occasionally use "nice" numbers that have shortcuts
        if random.random() < 0.2:
            b = random.choice([5, 10, 11, 15, 25, 50])

        return MathProblem(
            problem_text=f"{a} x {b}",
            spoken_text=f"{a} times {b}",
            answer=a * b,
            problem_type=ProblemType.MULTIPLICATION,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def _gen_division(self, difficulty: int) -> MathProblem:
        """Generate division problem (clean divisors)."""
        ranges = {
            1: (2, 12, 2, 9),
            2: (2, 12, 2, 12),
            3: (11, 99, 2, 12),
            4: (11, 99, 11, 25),
            5: (100, 500, 11, 50),
        }
        divisor_min, divisor_max, mult_min, mult_max = ranges[difficulty]

        divisor = random.randint(divisor_min, divisor_max)
        multiplier = random.randint(mult_min, mult_max)
        dividend = divisor * multiplier  # Ensures clean division

        return MathProblem(
            problem_text=f"{dividend} / {divisor}",
            spoken_text=f"{dividend} divided by {divisor}",
            answer=multiplier,
            problem_type=ProblemType.DIVISION,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def _gen_addition(self, difficulty: int) -> MathProblem:
        """Generate addition problem."""
        ranges = {
            1: (10, 99, 10, 99),
            2: (100, 999, 100, 999),
            3: (1000, 9999, 1000, 9999),
            4: (10000, 99999, 10000, 99999),
            5: (100000, 999999, 100000, 999999),
        }
        a_min, a_max, b_min, b_max = ranges[difficulty]

        a = random.randint(a_min, a_max)
        b = random.randint(b_min, b_max)

        return MathProblem(
            problem_text=f"{a} + {b}",
            spoken_text=f"{a} plus {b}",
            answer=a + b,
            problem_type=ProblemType.ADDITION,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def _gen_subtraction(self, difficulty: int) -> MathProblem:
        """Generate subtraction problem (positive result)."""
        ranges = {
            1: (50, 99, 10, 49),
            2: (100, 999, 50, 500),
            3: (1000, 9999, 100, 5000),
            4: (10000, 99999, 1000, 50000),
            5: (100000, 999999, 10000, 500000),
        }
        a_min, a_max, b_min, b_max = ranges[difficulty]

        a = random.randint(a_min, a_max)
        b = random.randint(b_min, min(b_max, a - 1))  # Ensure positive result

        return MathProblem(
            problem_text=f"{a} - {b}",
            spoken_text=f"{a} minus {b}",
            answer=a - b,
            problem_type=ProblemType.SUBTRACTION,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def _gen_percentage(self, difficulty: int) -> MathProblem:
        """Generate percentage problem."""
        percents = {
            1: [10, 20, 25, 50],
            2: [5, 10, 15, 20, 25, 50],
            3: [5, 10, 12, 15, 20, 25, 30, 50, 75],
            4: [7, 8, 12, 15, 17, 22, 33, 66],
            5: [3, 7, 11, 13, 17, 23, 37, 43],
        }
        bases = {
            1: [20, 40, 50, 80, 100, 200],
            2: [50, 80, 100, 120, 150, 200, 250],
            3: [60, 80, 120, 150, 200, 250, 300, 400],
            4: [75, 125, 175, 225, 275, 350, 450],
            5: [123, 234, 345, 456, 567, 678, 789],
        }

        percent = random.choice(percents[difficulty])
        base = random.choice(bases[difficulty])
        answer = (percent / 100) * base

        return MathProblem(
            problem_text=f"{percent}% of {base}",
            spoken_text=f"{percent} percent of {base}",
            answer=answer,
            problem_type=ProblemType.PERCENTAGE,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def _gen_square(self, difficulty: int) -> MathProblem:
        """Generate square problem."""
        ranges = {
            1: (2, 12),
            2: (11, 20),
            3: (21, 30),
            4: (31, 50),
            5: (51, 99),
        }
        n_min, n_max = ranges[difficulty]
        n = random.randint(n_min, n_max)

        return MathProblem(
            problem_text=f"{n}^2",
            spoken_text=f"{n} squared",
            answer=n * n,
            problem_type=ProblemType.SQUARE,
            difficulty=difficulty,
            time_target_ms=TIME_TARGETS[difficulty]
        )

    def parse_spoken_answer(self, spoken: str) -> Optional[float]:
        """
        Parse a spoken answer into a number.

        Examples:
            "forty seven" -> 47
            "one thousand two hundred thirty four" -> 1234
            "47" -> 47
            "negative five" -> -5
        """
        spoken = spoken.lower().strip()

        # Try direct numeric parse first
        try:
            return float(spoken.replace(',', ''))
        except ValueError:
            pass

        # Handle negative
        negative = False
        if spoken.startswith('negative ') or spoken.startswith('minus '):
            negative = True
            spoken = spoken.split(' ', 1)[1]

        # Parse word numbers
        result = self._parse_word_number(spoken)
        if result is not None and negative:
            result = -result

        return result

    def _parse_word_number(self, text: str) -> Optional[float]:
        """Parse word-based number representation."""
        text = text.strip()

        # Handle decimal point
        if ' point ' in text:
            parts = text.split(' point ')
            whole = self._parse_word_number(parts[0]) or 0
            decimal_str = parts[1]
            # Parse each digit after point
            decimal = 0.0
            for i, word in enumerate(decimal_str.split()):
                digit = WORD_TO_NUM.get(word)
                if digit is not None and digit < 10:
                    decimal += digit / (10 ** (i + 1))
            return whole + decimal

        words = text.split()
        if not words:
            return None

        total = 0
        current = 0

        for word in words:
            word = word.replace(',', '').replace('-', ' ')

            # Check for direct number
            if word.isdigit():
                current += int(word)
                continue

            # Check word mapping
            value = WORD_TO_NUM.get(word)
            if value is None:
                # Try compound words like "twenty-three"
                if '-' in word:
                    parts = word.split('-')
                    compound_val = sum(WORD_TO_NUM.get(p, 0) for p in parts)
                    if compound_val > 0:
                        current += compound_val
                        continue
                return None

            if value == 100:
                current = current * 100 if current else 100
            elif value == 1000:
                current = current * 1000 if current else 1000
                total += current
                current = 0
            else:
                current += value

        return total + current

    def parse_spoken_problem(self, spoken: str) -> Optional[Tuple[float, str, float]]:
        """
        Parse a spoken math problem.

        Args:
            spoken: e.g., "47 times 83" or "what is 25 percent of 200"

        Returns:
            Tuple of (operand1, operator, operand2) or None
        """
        spoken = spoken.lower().strip()

        # Remove common prefixes
        for prefix in ['what is ', 'calculate ', 'compute ', "what's "]:
            if spoken.startswith(prefix):
                spoken = spoken[len(prefix):]

        # Find operator
        op = None
        op_pos = -1
        for word, symbol in OP_WORDS.items():
            if f' {word} ' in f' {spoken} ':
                op = symbol
                op_pos = spoken.find(word)
                break

        if op is None:
            return None

        # Split around operator
        parts = spoken.split()
        op_idx = None
        for i, w in enumerate(parts):
            if w in OP_WORDS:
                op_idx = i
                break

        if op_idx is None:
            return None

        left = ' '.join(parts[:op_idx])
        right = ' '.join(parts[op_idx + 1:])

        # Handle "of" for percentages
        if op == '%' and 'of' in right:
            right = right.replace(' of ', ' ').strip()

        num1 = self.parse_spoken_answer(left)
        num2 = self.parse_spoken_answer(right)

        if num1 is None or num2 is None:
            return None

        return (num1, op, num2)

    def compute(self, num1: float, op: str, num2: float) -> float:
        """Compute result of operation."""
        if op == '*':
            return num1 * num2
        elif op == '/':
            return num1 / num2 if num2 != 0 else float('inf')
        elif op == '+':
            return num1 + num2
        elif op == '-':
            return num1 - num2
        elif op == '%':
            return (num1 / 100) * num2
        elif op == '^2':
            return num1 * num1
        elif op == 'sqrt':
            return num1 ** 0.5
        return 0

    def check_answer(
        self,
        problem: MathProblem,
        answer: float,
        time_ms: float,
        tolerance: float = 0.01
    ) -> AnswerResult:
        """
        Check if answer is correct.

        Args:
            problem: The problem that was posed
            answer: User's answer
            time_ms: Time taken to answer in milliseconds
            tolerance: Relative tolerance for floating point comparison
        """
        expected = problem.answer

        # Use relative tolerance for large numbers, absolute for small
        if abs(expected) > 1:
            correct = abs(answer - expected) / abs(expected) <= tolerance
        else:
            correct = abs(answer - expected) <= tolerance

        return AnswerResult(
            correct=correct,
            expected=expected,
            given=answer,
            time_ms=time_ms,
            within_target=time_ms <= problem.time_target_ms
        )


# Convenience function for quick evaluation
def quick_eval(expression: str) -> float:
    """
    Quickly evaluate a math expression string.
    For <1ms response time on simple operations.
    """
    engine = MentalMathEngine()
    parsed = engine.parse_spoken_problem(expression)
    if parsed:
        return engine.compute(*parsed)

    # Try direct eval for numeric expressions (safe subset)
    try:
        # Only allow numbers, operators, parentheses, spaces
        cleaned = re.sub(r'[^0-9+\-*/().x\s]', '', expression)
        cleaned = cleaned.replace('x', '*')
        if cleaned:
            return eval(cleaned)
    except:
        pass

    return 0


# Test
if __name__ == "__main__":
    engine = MentalMathEngine(difficulty=2)

    print("=== Mental Math Engine Test ===\n")

    # Generate problems
    for _ in range(5):
        problem = engine.generate_problem()
        print(f"Problem: {problem.problem_text}")
        print(f"Answer: {problem.answer}")
        print(f"Time target: {problem.time_target_ms}ms")
        print()

    # Test parsing
    test_inputs = [
        "forty seven",
        "one thousand two hundred thirty four",
        "negative five",
        "three point one four",
        "1234",
    ]

    print("=== Parsing Test ===\n")
    for inp in test_inputs:
        result = engine.parse_spoken_answer(inp)
        print(f"'{inp}' -> {result}")

    # Test spoken problems
    print("\n=== Spoken Problem Test ===\n")
    problems = [
        "47 times 83",
        "twenty five percent of two hundred",
        "144 divided by 12",
    ]
    for p in problems:
        parsed = engine.parse_spoken_problem(p)
        if parsed:
            result = engine.compute(*parsed)
            print(f"'{p}' -> {parsed} = {result}")
