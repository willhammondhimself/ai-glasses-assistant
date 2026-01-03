"""
FermiEngine: Fermi estimation problems for quant interviews.

Classic interview question type: "How many piano tuners are in Chicago?"
Tests ability to break down complex problems and make reasonable estimates.

Categories:
- market_size: Market sizing problems
- counting: Counting/enumeration problems
- rates: Rate-based calculations
- finance: Financial estimation
"""

import random
import math
import uuid
from typing import Optional, List


class FermiEngine:
    """Fermi estimation problem generator for quant interview prep."""

    def __init__(self):
        self._problems = {}

        # Pre-loaded problem bank with reasonable ranges
        self._problem_bank = self._load_problems()

    def _load_problems(self) -> dict:
        """Load problem bank with hints and reasonable answers."""
        return {
            'market_size': [
                {
                    'problem': "How many golf balls fit in a school bus?",
                    'hints': [
                        "Estimate school bus dimensions (~8ft × 6ft × 20ft interior)",
                        "Golf ball diameter is ~1.68 inches",
                        "Account for packing efficiency (~64% for random packing)",
                    ],
                    'answer_range': (300000, 600000),
                    'solution': "Bus ~500 ft³ = 864,000 in³. Golf ball ~2.5 in³. 864,000/2.5 × 0.64 ≈ 220,000-500,000"
                },
                {
                    'problem': "How many piano tuners are there in Chicago?",
                    'hints': [
                        "Chicago population ~2.7 million",
                        "How many households? (~1 million)",
                        "What % have pianos? (~5%?)",
                        "How often tuned? (~1/year)",
                        "How many can one tuner service per year?",
                    ],
                    'answer_range': (100, 300),
                    'solution': "50,000 pianos ÷ 200 tunings/tuner/year ≈ 100-250 tuners"
                },
                {
                    'problem': "How many gas stations are in the United States?",
                    'hints': [
                        "US population ~330 million",
                        "Cars per capita ~0.8",
                        "Average fill-up frequency",
                        "Average gas station capacity per day",
                    ],
                    'answer_range': (100000, 200000),
                    'solution': "~260M cars, each fills ~50 times/year. 13B fill-ups ÷ 300 days ÷ 200 fills/station/day ≈ 150,000"
                },
                {
                    'problem': "What is the annual revenue of all McDonald's in the US?",
                    'hints': [
                        "How many McDonald's? (~14,000)",
                        "Average customers per day per location?",
                        "Average transaction size?",
                    ],
                    'answer_range': (35000000000, 50000000000),
                    'solution': "14,000 locations × 500 customers/day × $8 avg × 365 days ≈ $20-40B"
                },
            ],
            'counting': [
                {
                    'problem': "How many windows are in Manhattan?",
                    'hints': [
                        "Manhattan area ~23 sq miles",
                        "Average building height and density",
                        "Windows per floor per building",
                    ],
                    'answer_range': (5000000, 15000000),
                    'solution': "~50,000 buildings × average 20 floors × average 20 windows ≈ 20 million"
                },
                {
                    'problem': "How many leaves are on a mature oak tree?",
                    'hints': [
                        "Estimate branch structure",
                        "Leaves per branch",
                        "Number of branches",
                    ],
                    'answer_range': (100000, 500000),
                    'solution': "~2000 branch tips × 100-200 leaves per cluster ≈ 200,000-400,000"
                },
                {
                    'problem': "How many tennis balls can fit in this room?",
                    'hints': [
                        "Estimate room dimensions (10ft × 12ft × 8ft typical)",
                        "Tennis ball diameter ~2.7 inches",
                        "Packing efficiency ~64%",
                    ],
                    'answer_range': (30000, 80000),
                    'solution': "960 ft³ × 1728 in³/ft³ ÷ 10.3 in³ × 0.64 ≈ 100,000"
                },
            ],
            'rates': [
                {
                    'problem': "How much water flows over Niagara Falls in one hour?",
                    'hints': [
                        "Estimate width of falls (~3,600 ft total)",
                        "Water depth flowing over (~2 ft average)",
                        "Water velocity (~20 mph)",
                    ],
                    'answer_range': (500000000, 1000000000),
                    'solution': "3600 ft × 2 ft × 30 ft/s × 3600 sec = ~750 million cubic ft ≈ 5.5B gallons"
                },
                {
                    'problem': "How many Google searches happen per second globally?",
                    'hints': [
                        "Internet users ~5 billion",
                        "What % use Google? (~90%)",
                        "Searches per user per day?",
                    ],
                    'answer_range': (80000, 120000),
                    'solution': "4.5B users × 3 searches/day ÷ 86,400 seconds ≈ 100,000+ searches/sec"
                },
                {
                    'problem': "How many babies are born per day worldwide?",
                    'hints': [
                        "World population ~8 billion",
                        "Birth rate ~18 per 1000 per year",
                    ],
                    'answer_range': (350000, 450000),
                    'solution': "8B × 0.018 / 365 ≈ 400,000 births per day"
                },
            ],
            'finance': [
                {
                    'problem': "Estimate Apple's daily revenue.",
                    'hints': [
                        "Annual revenue ~$400B",
                        "Or: iPhones sold per year × ASP",
                    ],
                    'answer_range': (900000000, 1200000000),
                    'solution': "$400B / 365 ≈ $1.1B per day"
                },
                {
                    'problem': "How much money is in circulation in the US?",
                    'hints': [
                        "US GDP ~$25 trillion",
                        "Money velocity ~5x per year",
                        "Or: Population × average cash holdings",
                    ],
                    'answer_range': (2000000000000, 3000000000000),
                    'solution': "M1 money supply ≈ $2-3 trillion (actual: ~$2.3T)"
                },
                {
                    'problem': "What is the total value of all real estate in NYC?",
                    'hints': [
                        "NYC population ~8.3 million",
                        "Housing units ~3.5 million",
                        "Average property value?",
                        "Commercial real estate?",
                    ],
                    'answer_range': (1000000000000, 1500000000000),
                    'solution': "3.5M units × $500K avg + commercial ≈ $1+ trillion"
                },
            ]
        }

    def generate_problem(self, category: str = None) -> dict:
        """
        Generate a Fermi estimation problem.

        Args:
            category: 'market_size', 'counting', 'rates', 'finance', or None for random

        Returns:
            dict with problem and metadata
        """
        categories = list(self._problem_bank.keys())

        if category is None:
            category = random.choice(categories)
        elif category not in categories:
            return {"error": f"Invalid category. Choose from: {categories}"}

        problem_data = random.choice(self._problem_bank[category])
        problem_id = str(uuid.uuid4())[:8]

        self._problems[problem_id] = {
            'answer_range': problem_data['answer_range'],
            'solution': problem_data['solution'],
            'hints': problem_data['hints']
        }

        return {
            'problem_id': problem_id,
            'problem': problem_data['problem'],
            'category': category,
            'hint_count': len(problem_data['hints']),
            'error': None
        }

    def get_hints(self, problem_id: str, hint_level: int = 1) -> dict:
        """
        Get progressive hints for a problem.

        Args:
            problem_id: Problem ID
            hint_level: 1, 2, 3... (reveals hints incrementally)

        Returns:
            dict with hints up to requested level
        """
        if problem_id not in self._problems:
            return {"error": "Problem not found"}

        hints = self._problems[problem_id]['hints']
        hint_level = max(1, min(hint_level, len(hints)))

        return {
            'problem_id': problem_id,
            'hints': hints[:hint_level],
            'hints_remaining': len(hints) - hint_level,
            'error': None
        }

    def evaluate_estimate(self, problem_id: str, estimate: float) -> dict:
        """
        Evaluate user's estimate using order of magnitude scoring.

        Args:
            problem_id: Problem ID
            estimate: User's numerical estimate

        Returns:
            dict with score and feedback
        """
        if problem_id not in self._problems:
            return {"error": "Problem not found"}

        problem_data = self._problems[problem_id]
        low, high = problem_data['answer_range']
        midpoint = (low + high) / 2
        solution = problem_data['solution']

        # Calculate order of magnitude difference
        if estimate <= 0:
            return {"error": "Estimate must be positive"}

        log_diff = abs(math.log10(estimate) - math.log10(midpoint))

        # Scoring based on order of magnitude
        if low <= estimate <= high:
            score = 100
            feedback = "Excellent! Within expected range."
        elif log_diff < 0.5:  # Within ~3x
            score = 80
            feedback = "Very good! Close to expected range."
        elif log_diff < 1:  # Within one order of magnitude
            score = 60
            feedback = "Good! Within one order of magnitude."
        elif log_diff < 1.5:
            score = 40
            feedback = "Fair. About 30x off, but reasonable approach."
        elif log_diff < 2:
            score = 20
            feedback = "Off by ~2 orders of magnitude. Review assumptions."
        else:
            score = 0
            feedback = "Way off. Check your approach."

        # Clean up
        del self._problems[problem_id]

        return {
            'estimate': estimate,
            'expected_range': {'low': low, 'high': high, 'midpoint': midpoint},
            'order_of_magnitude_diff': round(log_diff, 2),
            'score': score,
            'feedback': feedback,
            'solution_approach': solution,
            'error': None
        }

    def get_approach_template(self) -> dict:
        """Return a template for solving Fermi problems."""
        return {
            'steps': [
                "1. CLARIFY: Make sure you understand what's being asked",
                "2. DECOMPOSE: Break into smaller, estimatable pieces",
                "3. ESTIMATE: Make reasonable assumptions for each piece",
                "4. CALCULATE: Combine estimates (watch for unit conversions)",
                "5. SANITY CHECK: Does the answer make sense?",
                "6. BOUND: Provide a range, not just a point estimate"
            ],
            'useful_numbers': {
                'US population': '330 million',
                'World population': '8 billion',
                'US households': '130 million',
                'US GDP': '$25 trillion',
                'Seconds per year': '31.5 million',
                'Hours per year': '8,760',
                'Days per year': '365',
                'Weeks per year': '52',
                'Square miles in US': '3.8 million',
                'Square miles in Manhattan': '23',
                'Average US household size': '2.5',
                'US life expectancy': '78 years',
                'Speed of light': '300,000 km/s',
                'Speed of sound': '340 m/s'
            },
            'estimation_techniques': [
                "Dimensional analysis: Check that units work out",
                "Anchoring: Start from a known number",
                "Top-down: Total → subdivisions",
                "Bottom-up: Individual units → aggregate",
                "Sanity bounds: Find upper and lower limits first"
            ],
            'error': None
        }

    def get_categories(self) -> List[str]:
        """Return available problem categories."""
        return list(self._problem_bank.keys())

    def custom_problem(
        self,
        problem_text: str,
        answer_low: float,
        answer_high: float,
        hints: List[str] = None,
        solution: str = None
    ) -> dict:
        """
        Create a custom Fermi problem.

        Args:
            problem_text: The estimation problem
            answer_low: Lower bound of reasonable answer
            answer_high: Upper bound of reasonable answer
            hints: List of progressive hints
            solution: Solution explanation

        Returns:
            dict with problem_id
        """
        problem_id = str(uuid.uuid4())[:8]

        self._problems[problem_id] = {
            'answer_range': (answer_low, answer_high),
            'solution': solution or f"Expected range: {answer_low:,.0f} - {answer_high:,.0f}",
            'hints': hints or []
        }

        return {
            'problem_id': problem_id,
            'problem': problem_text,
            'category': 'custom',
            'hint_count': len(hints or []),
            'error': None
        }
