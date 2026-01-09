"""
ICM (Independent Chip Model) Calculator for Tournament Poker.

Implements Malmuth-Harville probability model for calculating
payout-adjusted equity in tournament situations.

Mathematical Foundation:
- P(i wins) = chips[i] / total_chips
- P(i gets 2nd | j wins) = chips[i] / (total - chips[j])
- ICM Equity = Σ P(finish_k) × payout[k]

Key Features:
- Malmuth-Harville finish probability calculation
- Jam/fold EV analysis with ICM adjustment
- Bubble factor calculation
- Support for 2-9 player tables
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ICMResult:
    """Result of ICM equity calculation."""
    equity: float              # Payout equity in $
    equity_pct: float          # Share of prize pool
    chip_ev: float             # Chip EV (raw)
    icm_pressure: float        # How much ICM differs from chip EV
    finish_probs: List[float]  # Probability of each finish position
    bubble_factor: float       # ICM multiplier at bubble


@dataclass
class JamFoldResult:
    """Result of jam/fold ICM analysis."""
    jam_ev: float              # EV of jamming (shoving all-in)
    fold_ev: float             # EV of folding
    call_ev: float             # EV of calling (if facing shove)
    recommendation: str        # "jam", "fold", or "call"
    ev_difference: float       # Difference between best and worst
    risk_premium: float        # ICM risk adjustment


class ICMCalculator:
    """
    ICM Calculator using Malmuth-Harville probability model.

    The Malmuth-Harville model recursively calculates finish probabilities:
    - First place prob = chips / total_chips (proportional)
    - Subsequent places calculated by removing winner and recursing

    Example:
        icm = ICMCalculator()
        result = icm.calculate_icm_equity(
            stacks=[5000, 3000, 2000],
            payouts=[50, 30, 20],
            hero_position=0
        )
        print(f"Your ICM equity: ${result.equity:.2f}")
    """

    def __init__(self, max_iterations: int = 100000):
        """
        Initialize ICM Calculator.

        Args:
            max_iterations: Max recursion for probability calculations
        """
        self.max_iterations = max_iterations

    def calculate_icm_equity(
        self,
        stacks: List[int],
        payouts: List[float],
        hero_position: int = 0,
        _skip_bubble: bool = False
    ) -> ICMResult:
        """
        Calculate ICM equity for hero position.

        Args:
            stacks: List of chip stacks [5000, 3000, 2000]
            payouts: Prize payouts [50, 30, 20] (can be $ or %)
            hero_position: Index of hero in stacks list

        Returns:
            ICMResult with equity, finish probabilities, and analysis
        """
        n_players = len(stacks)
        total_chips = sum(stacks)
        total_prizes = sum(payouts)

        # Validate inputs
        if hero_position < 0 or hero_position >= n_players:
            logger.error(f"Invalid hero_position: {hero_position}")
            return self._empty_result(n_players)

        if n_players < 2:
            logger.error("Need at least 2 players for ICM")
            return self._empty_result(n_players)

        # Extend payouts if needed (positions past payout get 0)
        payouts = list(payouts) + [0.0] * (n_players - len(payouts))
        payouts = payouts[:n_players]

        # Calculate finish probabilities using Malmuth-Harville
        finish_probs = self._calculate_finish_probs(
            stacks, hero_position, n_players
        )

        # Calculate ICM equity
        equity = sum(prob * payout for prob, payout in zip(finish_probs, payouts))
        equity_pct = equity / total_prizes if total_prizes > 0 else 0

        # Calculate chip EV (proportional)
        chip_ev = (stacks[hero_position] / total_chips) * total_prizes

        # ICM pressure: difference between chip EV and ICM equity
        # Positive = ICM hurts us, Negative = ICM helps us
        icm_pressure = chip_ev - equity

        # Bubble factor (how much 1 chip is worth in ICM terms)
        # Skip if called recursively to prevent infinite recursion
        if _skip_bubble:
            bubble_factor = 1.0
        else:
            bubble_factor = self._calculate_bubble_factor(
                stacks, payouts, hero_position
            )

        return ICMResult(
            equity=equity,
            equity_pct=equity_pct,
            chip_ev=chip_ev,
            icm_pressure=icm_pressure,
            finish_probs=finish_probs,
            bubble_factor=bubble_factor
        )

    def _calculate_finish_probs(
        self,
        stacks: List[int],
        hero_idx: int,
        n_positions: int
    ) -> List[float]:
        """
        Calculate probability of finishing in each position using Malmuth-Harville.

        This uses recursive probability calculation:
        P(hero finishes kth) = Σ P(others finish 1st..k-1) × P(hero wins remaining)

        Args:
            stacks: Current chip stacks
            hero_idx: Index of hero
            n_positions: Number of finishing positions to calculate

        Returns:
            List of probabilities [P(1st), P(2nd), P(3rd), ...]
        """
        n_players = len(stacks)
        finish_probs = [0.0] * n_positions

        # Use memoization for efficiency
        memo = {}

        def malmuth_harville(
            remaining_stacks: Tuple[int, ...],
            remaining_indices: Tuple[int, ...],
            target_idx: int,
            position: int
        ) -> float:
            """
            Recursive Malmuth-Harville calculation.

            Args:
                remaining_stacks: Stacks of remaining players
                remaining_indices: Original indices of remaining players
                target_idx: Which player we're calculating for
                position: Which position we're calculating (0=1st, 1=2nd, etc.)

            Returns:
                Probability of target_idx finishing in position
            """
            key = (remaining_stacks, remaining_indices, target_idx, position)
            if key in memo:
                return memo[key]

            total = sum(remaining_stacks)
            if total == 0:
                return 0.0

            n_remaining = len(remaining_stacks)

            # Base case: target player is only one left
            if n_remaining == 1:
                result = 1.0 if remaining_indices[0] == target_idx else 0.0
                memo[key] = result
                return result

            # If this is the last paying position, just use chip proportion
            if position >= n_positions - 1:
                for i, idx in enumerate(remaining_indices):
                    if idx == target_idx:
                        result = remaining_stacks[i] / total
                        memo[key] = result
                        return result
                return 0.0

            prob = 0.0

            # For first place, probability is proportional to chips
            if position == 0:
                for i, idx in enumerate(remaining_indices):
                    if idx == target_idx:
                        prob = remaining_stacks[i] / total
                        break
            else:
                # For subsequent places, sum over all possible winners of previous positions
                # P(target gets position) = Σ P(other wins) × P(target gets position | other won)
                for i in range(n_remaining):
                    if remaining_indices[i] == target_idx:
                        continue

                    # Probability this player wins
                    p_win = remaining_stacks[i] / total

                    # Remove winner and recurse
                    new_stacks = tuple(
                        s for j, s in enumerate(remaining_stacks) if j != i
                    )
                    new_indices = tuple(
                        idx for j, idx in enumerate(remaining_indices) if j != i
                    )

                    # Probability target finishes in (position) among remaining
                    p_finish = malmuth_harville(
                        new_stacks, new_indices, target_idx, position - 1
                    )

                    prob += p_win * p_finish

            memo[key] = prob
            return prob

        # Calculate probability for each finishing position
        initial_stacks = tuple(stacks)
        initial_indices = tuple(range(n_players))

        for pos in range(n_positions):
            finish_probs[pos] = malmuth_harville(
                initial_stacks, initial_indices, hero_idx, pos
            )

        # Normalize probabilities (should sum to ~1.0)
        total_prob = sum(finish_probs)
        if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
            finish_probs = [p / total_prob for p in finish_probs]

        return finish_probs

    def _calculate_bubble_factor(
        self,
        stacks: List[int],
        payouts: List[float],
        hero_idx: int
    ) -> float:
        """
        Calculate bubble factor (risk premium for tournament play).

        Bubble factor = ($ lost if eliminated) / ($ gained if double up)

        A bubble factor > 1 means losing chips hurts more than gaining helps.
        This is why tight play is correct on the bubble.

        Args:
            stacks: Current chip stacks
            payouts: Prize payouts
            hero_idx: Hero position

        Returns:
            Bubble factor (typically 1.0 - 3.0)
        """
        n_players = len(stacks)
        if n_players < 2:
            return 1.0

        hero_stack = stacks[hero_idx]
        total_chips = sum(stacks)

        # Current ICM equity (skip bubble to prevent recursion)
        current = self.calculate_icm_equity(stacks, payouts, hero_idx, _skip_bubble=True)
        current_equity = current.equity

        # Find smallest stack to bust hero against
        min_villain_idx = -1
        min_villain_stack = float('inf')
        for i, stack in enumerate(stacks):
            if i != hero_idx and stack < min_villain_stack and stack > 0:
                min_villain_idx = i
                min_villain_stack = stack

        if min_villain_idx == -1:
            return 1.0

        # Equity if hero doubles through min villain
        double_stacks = stacks.copy()
        chips_won = min(hero_stack, min_villain_stack)
        double_stacks[hero_idx] += chips_won
        double_stacks[min_villain_idx] -= chips_won

        # Remove busted player if needed
        if double_stacks[min_villain_idx] <= 0:
            double_stacks.pop(min_villain_idx)
            new_hero_idx = hero_idx if hero_idx < min_villain_idx else hero_idx - 1
            payouts_for_calc = payouts[:len(double_stacks)]
        else:
            new_hero_idx = hero_idx
            payouts_for_calc = payouts

        if len(double_stacks) >= 2:
            double_result = self.calculate_icm_equity(
                double_stacks, payouts_for_calc, new_hero_idx, _skip_bubble=True
            )
            gain = double_result.equity - current_equity
        else:
            gain = sum(payouts) - current_equity

        # Equity if hero busts (loses to min villain)
        # Hero gets current payout position (last place among remaining)
        bust_equity = payouts[n_players - 1] if n_players <= len(payouts) else 0
        loss = current_equity - bust_equity

        # Bubble factor
        if gain > 0:
            bubble_factor = loss / gain
        else:
            bubble_factor = float('inf') if loss > 0 else 1.0

        # Clamp to reasonable range
        return max(0.5, min(10.0, bubble_factor))

    def jam_fold_ev(
        self,
        stacks: List[int],
        payouts: List[float],
        hero_idx: int,
        win_prob: float,
        pot: int = 0,
        ante: int = 0,
        bb: int = 0
    ) -> JamFoldResult:
        """
        Calculate jam/fold EV with ICM adjustment.

        This calculates:
        - EV of jamming all-in
        - EV of folding
        - Recommendation based on ICM-adjusted EV

        Args:
            stacks: Current chip stacks
            payouts: Prize payouts
            hero_idx: Hero position in stacks
            win_prob: Probability hero wins if called (0.0 - 1.0)
            pot: Current pot size (blinds + antes)
            ante: Ante amount
            bb: Big blind amount

        Returns:
            JamFoldResult with EVs and recommendation
        """
        n_players = len(stacks)
        hero_stack = stacks[hero_idx]

        # Current ICM equity
        current = self.calculate_icm_equity(stacks, payouts, hero_idx)
        fold_ev = current.equity

        # If pushing, we either:
        # 1. Everyone folds - we win blinds/antes
        # 2. Someone calls - we flip for stacks

        # Simplification: assume one caller with avg stack
        total_other_chips = sum(s for i, s in enumerate(stacks) if i != hero_idx)
        avg_villain_stack = total_other_chips / (n_players - 1) if n_players > 1 else 0

        # Effective stack (the smaller of hero or villain)
        effective_stack = min(hero_stack, avg_villain_stack)

        # Fold equity: probability villain folds
        # Rough estimate based on push size relative to pot
        push_to_pot = hero_stack / max(1, pot + bb + ante * n_players)
        fold_equity = max(0.1, min(0.8, 0.6 - push_to_pot * 0.1))

        # EV if called and we win
        win_stacks = stacks.copy()
        win_stacks[hero_idx] += effective_stack + pot

        # Find a caller index (use largest stack as proxy)
        caller_idx = max(
            (i for i in range(n_players) if i != hero_idx),
            key=lambda i: stacks[i],
            default=0
        )
        win_stacks[caller_idx] -= effective_stack

        # Handle if caller busts
        if win_stacks[caller_idx] <= 0:
            win_stacks.pop(caller_idx)
            new_hero_idx = hero_idx if hero_idx < caller_idx else hero_idx - 1
        else:
            new_hero_idx = hero_idx

        if len(win_stacks) >= 2:
            win_result = self.calculate_icm_equity(
                win_stacks, payouts[:len(win_stacks)], new_hero_idx
            )
            win_ev = win_result.equity
        else:
            win_ev = sum(payouts)

        # EV if called and we lose (we bust)
        lose_ev = payouts[n_players - 1] if n_players <= len(payouts) else 0

        # EV if everyone folds
        steal_stacks = stacks.copy()
        steal_stacks[hero_idx] += pot + ante * n_players
        steal_result = self.calculate_icm_equity(steal_stacks, payouts, hero_idx)
        steal_ev = steal_result.equity

        # Combined jam EV
        # jam_ev = fold_equity * steal_ev + (1 - fold_equity) * (win_prob * win_ev + (1 - win_prob) * lose_ev)
        call_ev_component = win_prob * win_ev + (1 - win_prob) * lose_ev
        jam_ev = fold_equity * steal_ev + (1 - fold_equity) * call_ev_component

        # Calculate call EV (if we're facing a shove)
        call_ev = win_prob * win_ev + (1 - win_prob) * lose_ev

        # Recommendation
        ev_diff = jam_ev - fold_ev
        if ev_diff > 0:
            recommendation = "jam"
        else:
            recommendation = "fold"

        # Risk premium (how much ICM costs us)
        chip_ev_jam = hero_stack * 2 * win_prob  # Raw chip EV
        icm_ev_jam = jam_ev - fold_ev
        risk_premium = chip_ev_jam - icm_ev_jam if icm_ev_jam < chip_ev_jam else 0

        return JamFoldResult(
            jam_ev=jam_ev,
            fold_ev=fold_ev,
            call_ev=call_ev,
            recommendation=recommendation,
            ev_difference=ev_diff,
            risk_premium=risk_premium
        )

    def _empty_result(self, n_positions: int) -> ICMResult:
        """Return empty result for error cases."""
        return ICMResult(
            equity=0.0,
            equity_pct=0.0,
            chip_ev=0.0,
            icm_pressure=0.0,
            finish_probs=[0.0] * n_positions,
            bubble_factor=1.0
        )

    def format_icm_summary(self, result: ICMResult) -> str:
        """
        Format ICM result for voice output.

        Args:
            result: ICMResult to format

        Returns:
            Human-readable summary string
        """
        finish_str = ", ".join(
            f"{i+1}st: {p:.1%}" if i == 0 else f"{i+1}{'nd' if i==1 else 'rd' if i==2 else 'th'}: {p:.1%}"
            for i, p in enumerate(result.finish_probs[:3])
        )

        pressure_desc = "neutral"
        if result.icm_pressure > result.equity * 0.05:
            pressure_desc = "ICM hurting you"
        elif result.icm_pressure < -result.equity * 0.05:
            pressure_desc = "ICM helping you"

        return (
            f"ICM equity ${result.equity:.2f} ({result.equity_pct:.1%} of pool). "
            f"Finish odds: {finish_str}. "
            f"Bubble factor {result.bubble_factor:.1f}x ({pressure_desc})."
        )


# Test
if __name__ == "__main__":
    print("=== ICM Calculator Test ===\n")

    icm = ICMCalculator()

    # Test 1: 3-player SNG final table
    print("Test 1: 3-player SNG (50/30/20 payout)")
    print("Stacks: [5000, 3000, 2000]")
    result = icm.calculate_icm_equity(
        stacks=[5000, 3000, 2000],
        payouts=[50, 30, 20],
        hero_position=0
    )
    print(f"  Hero ICM Equity: ${result.equity:.2f}")
    print(f"  Chip EV: ${result.chip_ev:.2f}")
    print(f"  ICM Pressure: ${result.icm_pressure:.2f}")
    print(f"  Finish Probs: 1st={result.finish_probs[0]:.1%}, 2nd={result.finish_probs[1]:.1%}, 3rd={result.finish_probs[2]:.1%}")
    print(f"  Bubble Factor: {result.bubble_factor:.2f}x")
    print()

    # Test 2: Bubble situation (4 players, 3 paid)
    print("Test 2: Bubble (4 players, 3 paid)")
    print("Stacks: [8000, 4000, 2500, 500]")
    print("Payouts: [50, 30, 20]")
    result = icm.calculate_icm_equity(
        stacks=[8000, 4000, 2500, 500],
        payouts=[50, 30, 20],
        hero_position=2  # Medium stack
    )
    print(f"  Medium Stack ICM: ${result.equity:.2f}")
    print(f"  Bubble Factor: {result.bubble_factor:.2f}x (higher = more ICM pressure)")
    print()

    # Test 3: Jam/fold analysis
    print("Test 3: Jam/fold with AA (85% win rate)")
    print("Stacks: [3000, 5000, 2000], Hero has 3000")
    jf_result = icm.jam_fold_ev(
        stacks=[3000, 5000, 2000],
        payouts=[50, 30, 20],
        hero_idx=0,
        win_prob=0.85,
        pot=150,
        bb=100
    )
    print(f"  Jam EV: ${jf_result.jam_ev:.2f}")
    print(f"  Fold EV: ${jf_result.fold_ev:.2f}")
    print(f"  Recommendation: {jf_result.recommendation.upper()}")
    print(f"  EV Difference: ${jf_result.ev_difference:.2f}")
    print()

    # Test 4: Short stack on bubble
    print("Test 4: Short stack jam with AK (55% vs calling range)")
    jf_result = icm.jam_fold_ev(
        stacks=[500, 8000, 4000, 2500],
        payouts=[50, 30, 20],
        hero_idx=0,  # Short stack
        win_prob=0.55,
        pot=150,
        bb=100
    )
    print(f"  Jam EV: ${jf_result.jam_ev:.2f}")
    print(f"  Fold EV: ${jf_result.fold_ev:.2f}")
    print(f"  Recommendation: {jf_result.recommendation.upper()}")
    print(f"  Risk Premium: ${jf_result.risk_premium:.2f}")
    print()

    # Test 5: Voice output format
    print("Test 5: Voice output format")
    result = icm.calculate_icm_equity([5000, 3000, 2000], [50, 30, 20], 0)
    print(f"  {icm.format_icm_summary(result)}")
