"""
InterviewModeEngine: Mixed problem sessions simulating real quant interviews.

Features:
- Firm-specific problem distributions (Jane Street, Citadel, Two Sigma)
- Adaptive difficulty based on performance
- Progress tracking and readiness scoring
- Timed sessions with performance analytics
"""

import time
import random
import uuid
from typing import Optional, Dict, List
from datetime import datetime

from .mental_math import MentalMathEngine
from .probability import ProbabilityEngine
from .fermi import FermiEngine
from .market_making import MarketMakingEngine


class InterviewModeEngine:
    """Interview simulation engine for quant interview prep."""

    # Firm-specific problem distributions (weights for each category)
    FIRM_STYLES = {
        'jane_street': {
            'mental_math': 0.30,
            'probability': 0.35,
            'expected_value': 0.20,
            'market_making': 0.10,
            'fermi': 0.05,
            'time_pressure': 'high',
            'description': 'Heavy on probability and mental math. Fast-paced.'
        },
        'citadel': {
            'mental_math': 0.20,
            'probability': 0.25,
            'expected_value': 0.20,
            'market_making': 0.20,
            'fermi': 0.15,
            'time_pressure': 'medium',
            'description': 'Balanced mix with more market making questions.'
        },
        'two_sigma': {
            'mental_math': 0.15,
            'probability': 0.30,
            'expected_value': 0.25,
            'market_making': 0.15,
            'fermi': 0.15,
            'time_pressure': 'medium',
            'description': 'Emphasis on expected value and probability theory.'
        },
        'de_shaw': {
            'mental_math': 0.25,
            'probability': 0.25,
            'expected_value': 0.20,
            'market_making': 0.15,
            'fermi': 0.15,
            'time_pressure': 'medium',
            'description': 'Technical and quantitative focus.'
        },
        'general': {
            'mental_math': 0.25,
            'probability': 0.25,
            'expected_value': 0.20,
            'market_making': 0.15,
            'fermi': 0.15,
            'time_pressure': 'medium',
            'description': 'Balanced practice for any quant interview.'
        }
    }

    # Readiness score category weights
    READINESS_WEIGHTS = {
        'mental_math': 0.25,
        'probability': 0.25,
        'expected_value': 0.20,
        'market_making': 0.15,
        'fermi': 0.10,
        'options': 0.05
    }

    def __init__(self):
        # Initialize sub-engines
        self.mental_math = MentalMathEngine()
        self.probability = ProbabilityEngine()
        self.fermi = FermiEngine()
        self.market_making = MarketMakingEngine()

        # Active sessions
        self._sessions: Dict[str, dict] = {}

        # Historical performance (in-memory for now)
        self._performance_history: Dict[str, List[dict]] = {}

    def start_session(
        self,
        duration_min: int = 30,
        firm_style: str = 'general',
        difficulty: int = 2
    ) -> dict:
        """
        Start an interview practice session.

        Args:
            duration_min: Session duration in minutes
            firm_style: Firm style for problem distribution
            difficulty: Starting difficulty (1-4)

        Returns:
            dict with session_id and first problem
        """
        if firm_style not in self.FIRM_STYLES:
            return {"error": f"Invalid firm style. Choose from: {list(self.FIRM_STYLES.keys())}"}

        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self._sessions[session_id] = {
            'start_time': start_time,
            'end_time': start_time + duration_min * 60,
            'duration_min': duration_min,
            'firm_style': firm_style,
            'difficulty': difficulty,
            'problems_asked': 0,
            'correct_count': 0,
            'category_stats': {
                'mental_math': {'asked': 0, 'correct': 0, 'total_time_ms': 0},
                'probability': {'asked': 0, 'correct': 0, 'total_time_ms': 0},
                'expected_value': {'asked': 0, 'correct': 0, 'total_time_ms': 0},
                'market_making': {'asked': 0, 'correct': 0, 'total_time_ms': 0},
                'fermi': {'asked': 0, 'correct': 0, 'total_time_ms': 0},
            },
            'current_problem': None,
            'problem_start_time': None,
            'consecutive_correct': 0,
            'consecutive_wrong': 0,
        }

        # Generate first problem
        first_problem = self._generate_problem_for_session(session_id)

        return {
            'session_id': session_id,
            'firm_style': firm_style,
            'firm_description': self.FIRM_STYLES[firm_style]['description'],
            'duration_min': duration_min,
            'difficulty': difficulty,
            'time_remaining_sec': duration_min * 60,
            'first_problem': first_problem,
            'error': None
        }

    def get_next_problem(
        self,
        session_id: str,
        prev_answer: str = None,
        prev_time_ms: int = None
    ) -> dict:
        """
        Submit answer to current problem and get next problem.

        Args:
            session_id: Active session ID
            prev_answer: Answer to previous problem (None if skipping)
            prev_time_ms: Time taken on previous problem

        Returns:
            dict with result of previous problem and next problem
        """
        if session_id not in self._sessions:
            return {"error": "Session not found or expired"}

        session = self._sessions[session_id]

        # Check if session is still active
        current_time = time.time()
        if current_time >= session['end_time']:
            return self.end_session(session_id)

        # Evaluate previous answer if provided
        prev_result = None
        if prev_answer is not None and session['current_problem']:
            prev_result = self._evaluate_answer(session_id, prev_answer, prev_time_ms or 0)

        # Generate next problem
        next_problem = self._generate_problem_for_session(session_id)

        time_remaining = int(session['end_time'] - current_time)

        return {
            'session_id': session_id,
            'previous_result': prev_result,
            'next_problem': next_problem,
            'problems_completed': session['problems_asked'],
            'correct_count': session['correct_count'],
            'current_difficulty': session['difficulty'],
            'time_remaining_sec': time_remaining,
            'error': None
        }

    def _generate_problem_for_session(self, session_id: str) -> dict:
        """Generate a problem based on firm style and current difficulty."""
        session = self._sessions[session_id]
        firm_style = session['firm_style']
        difficulty = session['difficulty']

        # Select category based on firm weights
        weights = self.FIRM_STYLES[firm_style]
        categories = ['mental_math', 'probability', 'expected_value', 'market_making', 'fermi']
        probs = [weights.get(c, 0.2) for c in categories]

        category = random.choices(categories, probs)[0]

        # Generate problem from appropriate engine
        if category == 'mental_math':
            problem = self.mental_math.generate_problem(difficulty=difficulty)
        elif category == 'probability':
            problem = self.probability.generate_card_problem(difficulty=difficulty)
        elif category == 'expected_value':
            problem = self.probability.generate_ev_problem(difficulty=difficulty)
        elif category == 'market_making':
            problem = self.market_making.generate_scenario(difficulty=difficulty)
        elif category == 'fermi':
            problem = self.fermi.generate_problem()
        else:
            problem = self.mental_math.generate_problem(difficulty=difficulty)

        # Store current problem info
        session['current_problem'] = {
            'category': category,
            'problem_data': problem
        }
        session['problem_start_time'] = time.time()

        return {
            'category': category,
            'difficulty': difficulty,
            **problem
        }

    def _evaluate_answer(
        self,
        session_id: str,
        answer: str,
        time_ms: int
    ) -> dict:
        """Evaluate an answer and update session stats."""
        session = self._sessions[session_id]
        current = session['current_problem']

        if not current:
            return {'error': 'No active problem'}

        category = current['category']
        problem_data = current['problem_data']
        problem_id = problem_data.get('problem_id')

        # Check answer using appropriate engine
        if category == 'mental_math':
            result = self.mental_math.check_answer(problem_id, answer, time_ms)
        elif category in ['probability', 'expected_value']:
            result = self.probability.check_answer(problem_id, answer)
        elif category == 'market_making':
            result = self.market_making.check_answer(problem_id, answer)
        elif category == 'fermi':
            try:
                estimate = float(answer.replace(',', ''))
                result = self.fermi.evaluate_estimate(problem_id, estimate)
                result['correct'] = result.get('score', 0) >= 60
            except ValueError:
                result = {'correct': False, 'error': 'Invalid estimate'}
        else:
            result = {'correct': False}

        is_correct = result.get('correct', False)

        # Update session stats
        session['problems_asked'] += 1
        session['category_stats'][category]['asked'] += 1
        session['category_stats'][category]['total_time_ms'] += time_ms

        if is_correct:
            session['correct_count'] += 1
            session['category_stats'][category]['correct'] += 1
            session['consecutive_correct'] += 1
            session['consecutive_wrong'] = 0

            # Adaptive difficulty: increase if doing well
            if session['consecutive_correct'] >= 3 and session['difficulty'] < 4:
                session['difficulty'] += 1
                session['consecutive_correct'] = 0
        else:
            session['consecutive_wrong'] += 1
            session['consecutive_correct'] = 0

            # Adaptive difficulty: decrease if struggling
            if session['consecutive_wrong'] >= 2 and session['difficulty'] > 1:
                session['difficulty'] -= 1
                session['consecutive_wrong'] = 0

        return {
            'category': category,
            'correct': is_correct,
            'time_ms': time_ms,
            'details': result,
            'difficulty_adjusted': session['difficulty']
        }

    def end_session(self, session_id: str) -> dict:
        """
        End a session and generate performance report.

        Args:
            session_id: Session to end

        Returns:
            dict with full performance report
        """
        if session_id not in self._sessions:
            return {"error": "Session not found"}

        session = self._sessions[session_id]
        total_problems = session['problems_asked']
        correct = session['correct_count']

        # Calculate category-level stats
        category_performance = {}
        for cat, stats in session['category_stats'].items():
            if stats['asked'] > 0:
                accuracy = stats['correct'] / stats['asked']
                avg_time = stats['total_time_ms'] / stats['asked']
            else:
                accuracy = 0
                avg_time = 0

            category_performance[cat] = {
                'questions': stats['asked'],
                'correct': stats['correct'],
                'accuracy': round(accuracy * 100, 1),
                'avg_time_ms': round(avg_time, 0)
            }

        # Calculate overall accuracy
        overall_accuracy = (correct / total_problems * 100) if total_problems > 0 else 0

        # Calculate readiness score
        readiness_score = self._calculate_readiness_score(category_performance)

        # Determine strengths and weaknesses
        strengths = []
        weaknesses = []
        for cat, perf in category_performance.items():
            if perf['questions'] >= 2:
                if perf['accuracy'] >= 80:
                    strengths.append(cat)
                elif perf['accuracy'] < 50:
                    weaknesses.append(cat)

        # Generate recommendations
        recommendations = self._generate_recommendations(category_performance, weaknesses)

        # Store in history
        if session_id not in self._performance_history:
            self._performance_history[session_id] = []

        report = {
            'session_id': session_id,
            'firm_style': session['firm_style'],
            'duration_min': session['duration_min'],
            'total_problems': total_problems,
            'correct': correct,
            'overall_accuracy': round(overall_accuracy, 1),
            'category_performance': category_performance,
            'readiness_score': readiness_score,
            'interview_ready': readiness_score >= 75,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'error': None
        }

        self._performance_history[session_id].append(report)

        # Clean up session
        del self._sessions[session_id]

        return report

    def _calculate_readiness_score(self, category_performance: dict) -> float:
        """
        Calculate Jane Street readiness score.

        Score = 0.6 × accuracy + 0.4 × speed_score
        """
        total_score = 0
        total_weight = 0

        for cat, weight in self.READINESS_WEIGHTS.items():
            if cat in category_performance and category_performance[cat]['questions'] > 0:
                perf = category_performance[cat]
                accuracy_score = perf['accuracy']

                # Speed score (based on time targets)
                # Assume good speed if avg_time < 10 seconds
                if perf['avg_time_ms'] > 0:
                    speed_score = max(0, 100 - (perf['avg_time_ms'] - 5000) / 100)
                    speed_score = min(100, speed_score)
                else:
                    speed_score = 50

                category_score = 0.6 * accuracy_score + 0.4 * speed_score
                total_score += category_score * weight
                total_weight += weight

        if total_weight > 0:
            return round(total_score / total_weight, 1)
        return 0

    def _generate_recommendations(
        self,
        category_performance: dict,
        weaknesses: List[str]
    ) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []

        if 'mental_math' in weaknesses:
            recommendations.append(
                "Practice mental math daily. Focus on 2-digit × 2-digit multiplication. "
                "Use the 'times tables + adjustment' technique."
            )

        if 'probability' in weaknesses:
            recommendations.append(
                "Review card probability problems - they're the most common in interviews. "
                "Practice conditional probability and Bayes' theorem."
            )

        if 'expected_value' in weaknesses:
            recommendations.append(
                "Work on expected value calculations. Remember: E[X] = Σ x·P(x). "
                "Practice identifying when to use linearity of expectation."
            )

        if 'market_making' in weaknesses:
            recommendations.append(
                "Study the Kelly criterion formula: f* = (bp-q)/b. "
                "Understand bid/ask spreads and edge calculation."
            )

        if 'fermi' in weaknesses:
            recommendations.append(
                "Practice breaking down estimation problems. "
                "Memorize key numbers (US population, etc.) and practice dimensional analysis."
            )

        if not recommendations:
            recommendations.append(
                "Great progress! Focus on increasing speed while maintaining accuracy. "
                "Try higher difficulty levels."
            )

        return recommendations

    def get_progress(self, include_history: bool = False) -> dict:
        """
        Get overall progress across all sessions.

        Returns:
            dict with progress metrics
        """
        if not self._performance_history:
            return {
                'sessions_completed': 0,
                'total_problems': 0,
                'overall_accuracy': 0,
                'current_readiness': 0,
                'trend': 'no data',
                'error': None
            }

        all_sessions = []
        for sessions in self._performance_history.values():
            all_sessions.extend(sessions)

        total_problems = sum(s['total_problems'] for s in all_sessions)
        total_correct = sum(s['correct'] for s in all_sessions)
        avg_accuracy = (total_correct / total_problems * 100) if total_problems > 0 else 0

        # Get most recent readiness score
        most_recent = sorted(all_sessions, key=lambda x: x['timestamp'])[-1]
        current_readiness = most_recent['readiness_score']

        # Calculate trend (compare last 3 sessions)
        if len(all_sessions) >= 3:
            recent_scores = [s['readiness_score'] for s in sorted(all_sessions, key=lambda x: x['timestamp'])[-3:]]
            if recent_scores[-1] > recent_scores[0]:
                trend = 'improving'
            elif recent_scores[-1] < recent_scores[0]:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'need more data'

        result = {
            'sessions_completed': len(all_sessions),
            'total_problems': total_problems,
            'overall_accuracy': round(avg_accuracy, 1),
            'current_readiness': current_readiness,
            'interview_ready': current_readiness >= 75,
            'trend': trend,
            'error': None
        }

        if include_history:
            result['history'] = all_sessions

        return result

    def get_firm_styles(self) -> dict:
        """Return available firm styles and their descriptions."""
        return {
            firm: {
                'description': data['description'],
                'time_pressure': data['time_pressure'],
                'problem_weights': {
                    k: v for k, v in data.items()
                    if k not in ['time_pressure', 'description']
                }
            }
            for firm, data in self.FIRM_STYLES.items()
        }
