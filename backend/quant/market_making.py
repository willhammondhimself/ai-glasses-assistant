"""
MarketMakingEngine: Market making concepts for quant interviews.

Covers:
- Bid/ask spread scenarios
- Edge calculation
- Kelly criterion for optimal bet sizing
- Sharpe ratio and Sortino ratio
- Risk management concepts
"""

import random
import math
import uuid
from typing import List, Optional
import statistics


class MarketMakingEngine:
    """Market making problem generator for quant interview prep."""

    def __init__(self):
        self._problems = {}

    # ==================== Scenario Generation ====================

    def generate_scenario(self, scenario_type: str = None, difficulty: int = 2) -> dict:
        """
        Generate a market making scenario problem.

        Args:
            scenario_type: 'spread', 'edge', 'inventory', or None for random
            difficulty: 1-4

        Returns:
            dict with problem details
        """
        difficulty = max(1, min(4, difficulty))
        problem_id = str(uuid.uuid4())[:8]

        types = ['spread', 'edge', 'inventory']
        if scenario_type is None:
            scenario_type = random.choice(types)

        if scenario_type == 'spread':
            problem, answer, explanation = self._gen_spread_problem(difficulty)
        elif scenario_type == 'edge':
            problem, answer, explanation = self._gen_edge_problem(difficulty)
        elif scenario_type == 'inventory':
            problem, answer, explanation = self._gen_inventory_problem(difficulty)
        else:
            return {"error": f"Invalid scenario type. Choose from: {types}"}

        self._problems[problem_id] = {'answer': answer, 'explanation': explanation}

        return {
            'problem_id': problem_id,
            'problem': problem,
            'scenario_type': scenario_type,
            'difficulty': difficulty,
            'error': None
        }

    def _gen_spread_problem(self, difficulty: int) -> tuple:
        """Generate bid/ask spread problem."""
        if difficulty == 1:
            bid = random.randint(95, 99)
            ask = bid + 1
            return (
                f"A stock has bid ${bid} and ask ${ask}. What is the spread?",
                1,
                f"Spread = Ask - Bid = ${ask} - ${bid} = $1"
            )
        elif difficulty == 2:
            bid = random.randint(90, 99) + random.random()
            spread = random.choice([0.05, 0.10, 0.25])
            ask = bid + spread
            midpoint = (bid + ask) / 2
            return (
                f"Bid: ${bid:.2f}, Ask: ${ask:.2f}. What is the midpoint?",
                round(midpoint, 2),
                f"Midpoint = (Bid + Ask) / 2 = (${bid:.2f} + ${ask:.2f}) / 2 = ${midpoint:.2f}"
            )
        elif difficulty == 3:
            bid = 100
            ask = 100.50
            volume_bid = random.randint(100, 500)
            volume_ask = random.randint(100, 500)
            # Weighted midpoint
            weighted_mid = (bid * volume_ask + ask * volume_bid) / (volume_bid + volume_ask)
            return (
                f"Bid: ${bid} (size {volume_bid}), Ask: ${ask} (size {volume_ask}). "
                f"Calculate the volume-weighted midpoint.",
                round(weighted_mid, 4),
                f"VWAP mid = (bid×ask_vol + ask×bid_vol)/(bid_vol + ask_vol) = ${weighted_mid:.4f}"
            )
        else:
            # Market impact
            shares = random.randint(10000, 50000)
            avg_daily_vol = random.randint(100000, 500000)
            spread_bps = random.randint(5, 20)
            impact = (shares / avg_daily_vol) * spread_bps
            return (
                f"You need to buy {shares:,} shares. Average daily volume is {avg_daily_vol:,}. "
                f"Spread is {spread_bps} bps. Estimate market impact in bps.",
                round(impact, 2),
                f"Impact ≈ (trade_size / ADV) × spread = ({shares}/{avg_daily_vol}) × {spread_bps} = {impact:.2f} bps"
            )

    def _gen_edge_problem(self, difficulty: int) -> tuple:
        """Generate edge/expected value problem."""
        if difficulty <= 2:
            prob = random.choice([0.4, 0.45, 0.5, 0.55, 0.6])
            win = random.randint(10, 50)
            lose = random.randint(5, 30)
            edge = prob * win - (1 - prob) * lose
            return (
                f"You win ${win} with probability {prob}, lose ${lose} otherwise. "
                f"What is your edge (expected value)?",
                round(edge, 2),
                f"Edge = p×win - (1-p)×loss = {prob}×{win} - {1-prob}×{lose} = ${edge:.2f}"
            )
        else:
            # Multiple outcome scenario
            scenarios = [
                (0.3, 50),   # 30% chance win $50
                (0.4, -10),  # 40% chance lose $10
                (0.3, -20),  # 30% chance lose $20
            ]
            ev = sum(p * v for p, v in scenarios)
            return (
                f"Outcomes: {[f'{int(p*100)}% → ${v}' for p, v in scenarios]}. "
                f"Calculate expected value.",
                round(ev, 2),
                f"EV = Σ(p×v) = {' + '.join([f'{p}×{v}' for p, v in scenarios])} = ${ev:.2f}"
            )

    def _gen_inventory_problem(self, difficulty: int) -> tuple:
        """Generate inventory management problem."""
        if difficulty <= 2:
            position = random.choice([-1000, -500, 500, 1000])
            price = 100
            risk_per_share = random.randint(1, 5)
            total_risk = abs(position) * risk_per_share
            return (
                f"You are {'short' if position < 0 else 'long'} {abs(position)} shares at ${price}. "
                f"Stock can move ${risk_per_share} against you. What's your max loss?",
                total_risk,
                f"Max loss = |position| × adverse_move = {abs(position)} × ${risk_per_share} = ${total_risk}"
            )
        else:
            # Position limit scenario
            current_pos = random.randint(-500, 500)
            max_pos = 1000
            incoming_order = random.randint(200, 800)
            fillable = min(incoming_order, max_pos - current_pos)
            return (
                f"Current position: {current_pos:+d} shares. Position limit: ±{max_pos}. "
                f"Incoming buy order: {incoming_order} shares. How many can you fill?",
                max(0, fillable),
                f"Can fill min(order, limit - current_pos) = min({incoming_order}, {max_pos} - {current_pos}) = {max(0, fillable)}"
            )

    # ==================== Edge Calculation ====================

    def calculate_edge(
        self,
        prob_win: float,
        payout_win: float,
        payout_lose: float
    ) -> dict:
        """
        Calculate edge (expected value) of a bet.

        Args:
            prob_win: Probability of winning (0-1)
            payout_win: Amount won if successful
            payout_lose: Amount lost if unsuccessful (positive number)

        Returns:
            dict with edge calculation and recommendation
        """
        if not 0 <= prob_win <= 1:
            return {"error": "Probability must be between 0 and 1"}

        prob_lose = 1 - prob_win
        expected_value = prob_win * payout_win - prob_lose * payout_lose

        # Calculate breakeven probability
        if payout_win + payout_lose > 0:
            breakeven_prob = payout_lose / (payout_win + payout_lose)
        else:
            breakeven_prob = None

        # Calculate implied odds
        if prob_win > 0:
            fair_odds = (1 - prob_win) / prob_win
        else:
            fair_odds = float('inf')

        return {
            'expected_value': round(expected_value, 4),
            'prob_win': prob_win,
            'prob_lose': prob_lose,
            'payout_win': payout_win,
            'payout_lose': payout_lose,
            'breakeven_prob': round(breakeven_prob, 4) if breakeven_prob else None,
            'fair_odds': round(fair_odds, 4),
            'recommendation': 'TAKE BET' if expected_value > 0 else 'PASS',
            'formula': f'EV = {prob_win}×{payout_win} - {prob_lose}×{payout_lose} = {expected_value:.4f}',
            'error': None
        }

    # ==================== Kelly Criterion ====================

    def kelly_criterion(
        self,
        prob_win: float,
        odds: float,
        bankroll: float = 10000
    ) -> dict:
        """
        Calculate optimal bet size using Kelly Criterion.

        f* = (bp - q) / b

        where:
        - b = odds received on the bet (net payout per $1 wagered)
        - p = probability of winning
        - q = probability of losing = 1 - p
        - f* = fraction of bankroll to bet

        Args:
            prob_win: Probability of winning (0-1)
            odds: Net odds (e.g., 2.0 means you win $2 for every $1 bet)
            bankroll: Total bankroll

        Returns:
            dict with Kelly fraction and bet size
        """
        if not 0 < prob_win < 1:
            return {"error": "Probability must be between 0 and 1 (exclusive)"}
        if odds <= 0:
            return {"error": "Odds must be positive"}

        b = odds
        p = prob_win
        q = 1 - p

        # Kelly formula
        kelly_fraction = (b * p - q) / b

        # Practical adjustments
        if kelly_fraction <= 0:
            return {
                'kelly_fraction': 0,
                'kelly_percent': "0%",
                'optimal_bet': 0,
                'expected_value': round(p * odds - q, 6),
                'recommendation': 'DO NOT BET - Negative edge',
                'formula': f'f* = (bp - q) / b = ({b}×{p} - {q}) / {b} = {kelly_fraction:.4f}',
                'error': None
            }

        optimal_bet = bankroll * kelly_fraction

        # Half-Kelly (common in practice for safety)
        half_kelly = kelly_fraction / 2
        half_kelly_bet = bankroll * half_kelly

        return {
            'kelly_fraction': round(kelly_fraction, 6),
            'kelly_percent': f"{kelly_fraction * 100:.2f}%",
            'optimal_bet': round(optimal_bet, 2),
            'half_kelly_fraction': round(half_kelly, 6),
            'half_kelly_bet': round(half_kelly_bet, 2),
            'expected_value': round(p * odds - q, 6),
            'expected_growth': round(p * math.log(1 + b * kelly_fraction) + q * math.log(1 - kelly_fraction), 6),
            'bankroll': bankroll,
            'formula': f'f* = (bp - q) / b = ({b}×{p} - {q}) / {b} = {kelly_fraction:.4f}',
            'recommendation': f'Bet {kelly_fraction*100:.1f}% of bankroll (${optimal_bet:.2f})',
            'note': 'Many professionals use half-Kelly for reduced volatility',
            'error': None
        }

    # ==================== Sharpe Ratio ====================

    def sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> dict:
        """
        Calculate Sharpe Ratio.

        Sharpe = (E[R] - Rf) / σ(R)

        Args:
            returns: List of periodic returns (as decimals, e.g., 0.01 for 1%)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Number of periods in a year (252 for daily)

        Returns:
            dict with Sharpe ratio and components
        """
        if len(returns) < 2:
            return {"error": "Need at least 2 returns to calculate"}

        # Calculate statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return {"error": "Standard deviation is zero (no volatility)"}

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * math.sqrt(periods_per_year)

        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_std

        return {
            'sharpe_ratio': round(sharpe, 4),
            'mean_return': round(mean_return, 6),
            'std_return': round(std_return, 6),
            'annualized_return': round(annualized_return, 4),
            'annualized_return_pct': f"{annualized_return * 100:.2f}%",
            'annualized_volatility': round(annualized_std, 4),
            'annualized_volatility_pct': f"{annualized_std * 100:.2f}%",
            'risk_free_rate': risk_free_rate,
            'periods_per_year': periods_per_year,
            'n_observations': len(returns),
            'interpretation': self._interpret_sharpe(sharpe),
            'formula': 'Sharpe = (E[R] - Rf) / σ(R)',
            'error': None
        }

    def sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> dict:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total std).

        Sortino = (E[R] - Rf) / σ_downside

        Args:
            returns: List of periodic returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            dict with Sortino ratio
        """
        if len(returns) < 2:
            return {"error": "Need at least 2 returns to calculate"}

        mean_return = statistics.mean(returns)

        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]

        if len(negative_returns) < 2:
            downside_dev = 0.0001  # Avoid division by zero
        else:
            downside_dev = statistics.stdev(negative_returns)

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_downside = downside_dev * math.sqrt(periods_per_year)

        # Sortino ratio
        sortino = (annualized_return - risk_free_rate) / annualized_downside

        return {
            'sortino_ratio': round(sortino, 4),
            'downside_deviation': round(downside_dev, 6),
            'annualized_downside': round(annualized_downside, 4),
            'negative_return_count': len(negative_returns),
            'interpretation': self._interpret_sharpe(sortino),  # Same scale
            'formula': 'Sortino = (E[R] - Rf) / σ_downside',
            'note': 'Sortino only penalizes downside volatility',
            'error': None
        }

    def _interpret_sharpe(self, ratio: float) -> str:
        """Interpret Sharpe/Sortino ratio value."""
        if ratio < 0:
            return "Negative risk-adjusted return"
        elif ratio < 0.5:
            return "Poor"
        elif ratio < 1.0:
            return "Below average"
        elif ratio < 1.5:
            return "Good"
        elif ratio < 2.0:
            return "Very good"
        elif ratio < 3.0:
            return "Excellent"
        else:
            return "Exceptional (verify data)"

    # ==================== Answer Checking ====================

    def check_answer(self, problem_id: str, user_answer: str) -> dict:
        """Check user's answer to a market making problem."""
        if problem_id not in self._problems:
            return {"error": "Problem not found. Generate a new problem."}

        problem_data = self._problems[problem_id]
        correct_answer = problem_data['answer']
        explanation = problem_data['explanation']

        try:
            user_val = float(str(user_answer).replace('$', '').replace(',', ''))
            is_correct = abs(user_val - correct_answer) < 0.01
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

    # ==================== Formulas Reference ====================

    def get_formulas(self) -> dict:
        """Return key market making formulas."""
        return {
            'edge': 'Edge = E[V] = p × win - (1-p) × loss',
            'kelly': 'f* = (bp - q) / b where b=odds, p=P(win), q=P(lose)',
            'sharpe': 'Sharpe = (E[R] - Rf) / σ(R)',
            'sortino': 'Sortino = (E[R] - Rf) / σ_downside',
            'spread': 'Spread = Ask - Bid',
            'midpoint': 'Midpoint = (Bid + Ask) / 2',
            'vwap': 'VWAP = Σ(Price × Volume) / Σ(Volume)',
            'market_impact': 'Impact ≈ k × √(Trade Size / ADV)',
            'position_pnl': 'PnL = Position × (Current Price - Entry Price)'
        }
