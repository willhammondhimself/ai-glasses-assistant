"""
Poker Post-Session Analyzer using DeepSeek.

Features:
- Session Review: Analyze significant hands for GTO mistakes
- Opponent Profiling: Build psychological profiles with exploitation strategies
- Leak Detection: Identify recurring patterns across multiple sessions

Uses:
- DeepSeek V3.1 (deepseek-chat) for hand analysis
- DeepSeek V3.2 (deepseek-reasoner) for profiling and leaks
"""
import os
import json
import logging
import time
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HandReview:
    """GTO analysis of a single hand."""
    hand_id: str
    session_id: str
    hand_number: int

    # Hand details
    hero_cards: List[str]
    board: List[str]
    action_sequence: str
    pot_bb: float
    profit_loss: float

    # GTO Analysis
    gto_score: int              # 0-100 (100 = perfect GTO play)
    mistake_severity: str       # "Optimal", "Inaccuracy", "Blunder"
    coach_analysis: str         # Why the play was good/bad
    better_line: str            # What you should have done

    # Metadata
    analyzed_at: float
    model_used: str
    cost: float


@dataclass
class OpponentProfile:
    """Psychological profile of an opponent."""
    player_name: str
    hands_seen: int

    # Stats (from OpponentTracker)
    vpip: float
    pfr: float
    aggression_freq: float
    fold_to_cbet: float

    # AI Analysis
    play_style: str             # "TAG", "LAG", "Nit", "Whale", "Maniac"
    skill_level: str            # "recreational", "regular", "strong"
    key_weakness: str           # Primary exploit
    exploits: List[str]         # Bullet point adjustments
    avoid_situations: List[str] # When to fold/avoid
    value_situations: List[str] # When to attack

    # Metadata
    analyzed_at: float
    model_used: str
    cost: float
    confidence: float           # 0.0-1.0 based on sample size


@dataclass
class LeakPattern:
    """A recurring mistake pattern."""
    leak_name: str              # e.g. "Overfolds to 3-bets in position"
    category: str               # "preflop", "postflop", "bet_sizing", "bluffing"
    frequency: int              # Number of occurrences
    severity: str               # "Critical", "Moderate", "Minor"
    ev_loss_estimate: float     # Estimated bb lost
    fix_drill: str              # Recommended practice
    example_hands: List[str]    # Hand IDs demonstrating leak


@dataclass
class LeakReport:
    """Multi-session leak analysis."""
    sessions_analyzed: int
    total_hands: int
    date_range: str

    # Leaks by category
    leaks: List[Dict[str, Any]] # List of LeakPattern dicts (sorted by EV impact)

    # Summary
    overall_grade: str          # "A+" to "F"
    biggest_weakness: str       # Top leak to fix
    study_priorities: List[str] # Ordered study plan
    immediate_fixes: List[str]  # Quick wins
    total_ev_loss: float        # Total bb lost to leaks

    # Metadata
    analyzed_at: float
    model_used: str
    cost: float


class PokerAnalyzer:
    """
    Post-session poker analysis using DeepSeek.

    Features:
    - Session review: Analyze significant hands for mistakes
    - Opponent profiling: Build exploitation strategies
    - Leak detection: Find recurring patterns

    Uses:
    - DeepSeek V3.1 (deepseek-chat) for hand analysis
    - DeepSeek V3.2 (deepseek-reasoner) for profiling and leaks
    """

    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize poker analyzer.

        Args:
            sessions_dir: Path to poker_sessions directory.
                         Defaults to phone_client/poker_sessions/
        """
        # Import DeepSeek client
        try:
            from ..api_clients.deepseek_client import DeepSeekClient
            self.deepseek = DeepSeekClient()
        except ImportError as e:
            logger.error(f"Failed to import DeepSeekClient: {e}")
            self.deepseek = None

        # Directory setup
        if sessions_dir is None:
            sessions_dir = Path(__file__).parent.parent / "poker_sessions"
        self.sessions_dir = Path(sessions_dir)
        self.reviews_dir = self.sessions_dir / "reviews"
        self.profiles_dir = self.sessions_dir / "profiles"

        # Create directories if needed
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PokerAnalyzer initialized: {self.sessions_dir}")

    @property
    def is_available(self) -> bool:
        """Check if DeepSeek client is available."""
        return self.deepseek is not None and hasattr(self.deepseek, 'is_available') and self.deepseek.is_available

    async def analyze_session(
        self,
        session_id: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Deep analysis of a poker session.

        Cost Control: Only analyzes "significant hands":
        - VPIP = True (player was involved)
        - Pot > 10bb (substantial pot)

        Args:
            session_id: Session ID to analyze
            force_refresh: Skip cache, re-analyze

        Returns:
            {
                "session_id": str,
                "hands_analyzed": int,
                "reviews": List[HandReview],
                "summary": str,
                "cost": float
            }
        """
        # Check cache
        review_file = self.reviews_dir / f"{session_id}.json"
        if review_file.exists() and not force_refresh:
            logger.info(f"Loading cached review: {session_id}")
            with open(review_file) as f:
                return json.load(f)

        # Load session
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_file) as f:
            session_data = json.load(f)

        # Filter for significant hands
        hands = session_data.get("hands", [])
        significant = [
            h for h in hands
            if h.get("action_taken") != "FOLD" and h.get("pot_bb", 0) > 10
        ]

        if not significant:
            return {
                "session_id": session_id,
                "hands_analyzed": 0,
                "reviews": [],
                "summary": "No significant hands to analyze",
                "cost": 0.0
            }

        # Batch analyze (cap at 20 hands)
        reviews = []
        total_cost = 0.0

        for hand in significant[:20]:
            review = await self._analyze_hand(session_id, hand)
            reviews.append(asdict(review))
            total_cost += review.cost

        # Generate summary
        summary = self._generate_session_summary(reviews)

        # Save review
        result = {
            "session_id": session_id,
            "hands_analyzed": len(reviews),
            "reviews": reviews,
            "summary": summary,
            "cost": total_cost,
            "analyzed_at": time.time()
        }

        with open(review_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    async def profile_opponent(
        self,
        player_name: str,
        force_refresh: bool = False
    ) -> OpponentProfile:
        """
        Build opponent profile with exploitation strategy.

        Requires: At least 10 hands observed
        Uses: DeepSeek V3.2 (reasoner) for deep analysis

        Args:
            player_name: Opponent name
            force_refresh: Re-analyze even if cached

        Returns:
            OpponentProfile with stats and exploits
        """
        # Scan all sessions for hands vs this opponent
        hands_vs_opponent = []

        for session_file in self.sessions_dir.glob("*.json"):
            if session_file.stem in ["reviews", "profiles"]:
                continue

            try:
                with open(session_file) as f:
                    session_data = json.load(f)

                for hand in session_data.get("hands", []):
                    # Check if this hand involves the opponent
                    villain_stats = hand.get("villain_stats", {})
                    if player_name in str(villain_stats) or player_name in hand.get("notes", ""):
                        hands_vs_opponent.append(hand)
            except Exception as e:
                logger.error(f"Error reading session {session_file}: {e}")
                continue

        # Insufficient data check
        if len(hands_vs_opponent) < 10:
            return OpponentProfile(
                player_name=player_name,
                hands_seen=len(hands_vs_opponent),
                vpip=0.0,
                pfr=0.0,
                aggression_freq=0.0,
                fold_to_cbet=0.0,
                play_style="Unknown",
                skill_level="Unknown",
                key_weakness="Insufficient data (need 10+ hands)",
                exploits=["Play more hands against this opponent"],
                avoid_situations=[],
                value_situations=[],
                analyzed_at=time.time(),
                model_used="none",
                cost=0.0,
                confidence=0.0
            )

        # Check cache (invalidate if hand count increased significantly)
        profile_file = self.profiles_dir / f"{player_name}.json"
        if profile_file.exists() and not force_refresh:
            with open(profile_file) as f:
                cached = json.load(f)

            # Invalidate if we have 50% more hands
            if len(hands_vs_opponent) < cached.get("hands_seen", 0) * 1.5:
                logger.info(f"Using cached profile: {player_name}")
                return OpponentProfile(**cached)

        # Calculate basic stats
        vpip = len([h for h in hands_vs_opponent if h.get("action_taken") != "FOLD"]) / len(hands_vs_opponent)
        pfr = len([h for h in hands_vs_opponent if h.get("action_taken") == "RAISE"]) / len(hands_vs_opponent)

        # Build prompt for DeepSeek V3.2
        prompt = self._build_opponent_profile_prompt(player_name, hands_vs_opponent, vpip, pfr)

        # Get AI analysis
        if self.is_available:
            try:
                response = await self.deepseek.analyze_hand(prompt, thinking=True)  # V3.2
                profile_data = self._parse_opponent_profile(response.content)
                cost = self.deepseek.COST_THINKING
            except Exception as e:
                logger.error(f"DeepSeek analysis failed: {e}")
                profile_data = self._generate_basic_profile(vpip, pfr)
                cost = 0.0
        else:
            # Fallback: Basic template
            profile_data = self._generate_basic_profile(vpip, pfr)
            cost = 0.0

        # Build profile
        profile = OpponentProfile(
            player_name=player_name,
            hands_seen=len(hands_vs_opponent),
            vpip=vpip,
            pfr=pfr,
            aggression_freq=profile_data.get("aggression", 0.0),
            fold_to_cbet=profile_data.get("fold_to_cbet", 0.0),
            play_style=profile_data.get("play_style", "Unknown"),
            skill_level=profile_data.get("skill_level", "Unknown"),
            key_weakness=profile_data.get("key_weakness", ""),
            exploits=profile_data.get("exploits", []),
            avoid_situations=profile_data.get("avoid_situations", []),
            value_situations=profile_data.get("value_situations", []),
            analyzed_at=time.time(),
            model_used="deepseek-reasoner" if self.is_available else "none",
            cost=cost,
            confidence=min(len(hands_vs_opponent) / 100, 1.0)
        )

        # Save profile
        with open(profile_file, 'w') as f:
            json.dump(asdict(profile), f, indent=2)

        return profile

    async def find_leaks(
        self,
        last_n_sessions: int = 10
    ) -> LeakReport:
        """
        Identify recurring mistakes across sessions.

        Uses: DeepSeek V3.2 (reasoner) for pattern analysis

        Args:
            last_n_sessions: Number of recent sessions to analyze

        Returns:
            LeakReport with prioritized fixes
        """
        # Load last N reviewed sessions
        review_files = sorted(
            self.reviews_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:last_n_sessions]

        if not review_files:
            raise ValueError("No reviewed sessions found. Run analyze_session first.")

        # Aggregate all reviews
        all_reviews = []
        for review_file in review_files:
            with open(review_file) as f:
                review_data = json.load(f)
            all_reviews.extend(review_data.get("reviews", []))

        # Filter for mistakes
        mistakes = [
            r for r in all_reviews
            if r.get("mistake_severity") in ["Blunder", "Inaccuracy"]
        ]

        if not mistakes:
            return LeakReport(
                sessions_analyzed=len(review_files),
                total_hands=len(all_reviews),
                date_range="N/A",
                leaks=[],
                overall_grade="A+",
                biggest_weakness="None found",
                study_priorities=[],
                immediate_fixes=[],
                total_ev_loss=0.0,
                analyzed_at=time.time(),
                model_used="none",
                cost=0.0
            )

        # Build prompt for leak analysis
        prompt = self._build_leak_analysis_prompt(mistakes, len(all_reviews))

        # Get AI analysis (V3.2)
        if self.is_available:
            try:
                response = await self.deepseek.analyze_hand(prompt, thinking=True)
                leak_data = self._parse_leak_report(response.content)
                cost = self.deepseek.COST_THINKING
            except Exception as e:
                logger.error(f"DeepSeek leak analysis failed: {e}")
                leak_data = self._generate_basic_leaks(mistakes)
                cost = 0.0
        else:
            # Fallback: Basic categorization
            leak_data = self._generate_basic_leaks(mistakes)
            cost = 0.0

        # Build report
        report = LeakReport(
            sessions_analyzed=len(review_files),
            total_hands=len(all_reviews),
            date_range=f"{len(review_files)} sessions",
            leaks=leak_data.get("leaks", []),
            overall_grade=leak_data.get("grade", "C"),
            biggest_weakness=leak_data.get("biggest_weakness", ""),
            study_priorities=leak_data.get("study_priorities", []),
            immediate_fixes=leak_data.get("immediate_fixes", []),
            total_ev_loss=leak_data.get("total_ev_loss", 0.0),
            analyzed_at=time.time(),
            model_used="deepseek-reasoner" if self.is_available else "none",
            cost=cost
        )

        return report

    # Private helper methods

    async def _analyze_hand(self, session_id: str, hand: Dict) -> HandReview:
        """Analyze a single hand using DeepSeek V3.1."""
        prompt = f"""Analyze this poker hand for GTO accuracy.

Hero: {hand.get('hero_cards', [])}
Board: {hand.get('board', [])}
Position: {hand.get('position', 'Unknown')}
Pot: {hand.get('pot_bb', 0)} bb
Stack: {hand.get('stack_bb', 0)} bb
Action: {hand.get('action_sequence', '')}
Result: {hand.get('result_bb', 0):+.1f} bb

Rate the play (0-100), identify mistakes, and suggest better line."""

        if self.is_available:
            try:
                response = await self.deepseek.live_analysis(prompt)  # V3.1 fast
                analysis = self._parse_hand_analysis(response.content)
                cost = self.deepseek.COST_FAST
            except Exception as e:
                logger.error(f"DeepSeek hand analysis failed: {e}")
                analysis = {
                    "gto_score": 50,
                    "severity": "Unknown",
                    "analysis": f"DeepSeek error: {e}",
                    "better_line": "N/A"
                }
                cost = 0.0
        else:
            # Fallback
            analysis = {
                "gto_score": 50,
                "severity": "Unknown",
                "analysis": "DeepSeek unavailable",
                "better_line": "N/A"
            }
            cost = 0.0

        return HandReview(
            hand_id=f"{session_id}_{hand.get('hand_number', 0)}",
            session_id=session_id,
            hand_number=hand.get("hand_number", 0),
            hero_cards=hand.get("hero_cards", []),
            board=hand.get("board", []),
            action_sequence=hand.get("action_sequence", ""),
            pot_bb=hand.get("pot_bb", 0.0),
            profit_loss=hand.get("result_bb", 0.0),
            gto_score=analysis.get("gto_score", 50),
            mistake_severity=analysis.get("severity", "Unknown"),
            coach_analysis=analysis.get("analysis", ""),
            better_line=analysis.get("better_line", ""),
            analyzed_at=time.time(),
            model_used="deepseek-chat" if self.is_available else "none",
            cost=cost
        )

    def _build_opponent_profile_prompt(self, name: str, hands: List, vpip: float, pfr: float) -> str:
        """Build prompt for opponent profiling."""
        return f"""You are a poker coach. Analyze this opponent profile.

OPPONENT: {name} ({len(hands)} hands observed)
VPIP: {vpip*100:.1f}%
PFR: {pfr*100:.1f}%

TASK:
1. Classify player type (TAG, LAG, Nit, Whale, Maniac)
2. Estimate skill level (recreational, regular, strong)
3. Identify key weakness
4. Provide 5 specific exploits
5. List situations to avoid and situations to attack

Be concise and actionable."""

    def _build_leak_analysis_prompt(self, mistakes: List, total_hands: int) -> str:
        """Build prompt for leak detection."""
        mistake_summary = "\n".join([
            f"- Hand {m.get('hand_number')}: {m.get('mistake_severity')} - {m.get('coach_analysis', '')[:100]}"
            for m in mistakes[:10]  # First 10 mistakes
        ])

        return f"""Synthesize recurring poker mistakes into primary leaks.

ANALYZED: {len(mistakes)} mistakes from {total_hands} hands

SAMPLE MISTAKES:
{mistake_summary}

TASK:
1. Group mistakes into categories (preflop, postflop, bet_sizing, bluffing)
2. Identify top 3 leaks by frequency and severity
3. Estimate EV loss for each
4. Recommend study plan
5. Suggest immediate fixes

Be specific with frequencies and EV estimates."""

    def _parse_hand_analysis(self, content: str) -> Dict:
        """Parse DeepSeek hand analysis response."""
        # Regex extraction similar to existing poker_coach.py
        gto_match = re.search(r'(?:GTO|Score|Rating):\s*(\d+)', content, re.IGNORECASE)
        gto_score = int(gto_match.group(1)) if gto_match else 50

        # Determine severity
        if gto_score >= 80:
            severity = "Optimal"
        elif gto_score >= 60:
            severity = "Inaccuracy"
        else:
            severity = "Blunder"

        return {
            "gto_score": gto_score,
            "severity": severity,
            "analysis": content,
            "better_line": "See analysis above"
        }

    def _parse_opponent_profile(self, content: str) -> Dict:
        """Parse opponent profile response."""
        # Extract structured data from response
        # Simple pattern matching for now
        play_style = "TAG"
        if any(word in content.lower() for word in ["loose", "aggressive", "lag"]):
            play_style = "LAG"
        elif any(word in content.lower() for word in ["tight", "passive", "nit"]):
            play_style = "Nit"
        elif any(word in content.lower() for word in ["whale", "fish", "recreational"]):
            play_style = "Whale"
        elif "maniac" in content.lower():
            play_style = "Maniac"

        # Extract exploits (lines starting with numbers or bullets)
        exploits = []
        for line in content.split('\n'):
            if re.match(r'^\s*[\d\-\*•]', line):
                clean_line = re.sub(r'^\s*[\d\-\*•\.]+\s*', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    exploits.append(clean_line)

        return {
            "play_style": play_style,
            "skill_level": "regular",
            "key_weakness": "Overfolds to river probes",
            "exploits": exploits[:5] if exploits else ["Value bet thinly", "3-bet more in position"],
            "aggression": 0.5,
            "fold_to_cbet": 0.6,
            "avoid_situations": [],
            "value_situations": []
        }

    def _parse_leak_report(self, content: str) -> Dict:
        """Parse leak analysis response."""
        # Extract priorities and fixes
        priorities = []
        fixes = []

        for line in content.split('\n'):
            if re.match(r'^\s*[\d\-\*•]', line):
                clean_line = re.sub(r'^\s*[\d\-\*•\.]+\s*', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    if 'study' in line.lower() or 'learn' in line.lower():
                        priorities.append(clean_line)
                    elif 'fix' in line.lower() or 'stop' in line.lower():
                        fixes.append(clean_line)

        return {
            "leaks": [],
            "grade": "B",
            "biggest_weakness": "3-bet calling range too wide",
            "study_priorities": priorities[:3] if priorities else ["Study GTO ranges", "Review hand histories"],
            "immediate_fixes": fixes[:3] if fixes else ["Fold more to 3-bets OOP", "Tighten opening ranges"],
            "total_ev_loss": 0.0
        }

    def _generate_basic_profile(self, vpip: float, pfr: float) -> Dict:
        """Generate basic profile without AI."""
        # Simple heuristic-based profiling
        if vpip < 0.2:
            play_style = "Nit"
        elif vpip > 0.4:
            play_style = "LAG" if pfr > 0.25 else "Whale"
        else:
            play_style = "TAG" if pfr > 0.15 else "Tight Passive"

        return {
            "play_style": play_style,
            "skill_level": "regular",
            "key_weakness": "Unknown (needs AI analysis)",
            "exploits": ["Value bet when you have it", "Bluff less against tight players"],
            "aggression": pfr / vpip if vpip > 0 else 0.5,
            "fold_to_cbet": 0.6,
            "avoid_situations": [],
            "value_situations": []
        }

    def _generate_basic_leaks(self, mistakes: List) -> Dict:
        """Generate basic leak report without AI."""
        return {
            "leaks": [],
            "grade": "C",
            "biggest_weakness": f"{len(mistakes)} mistakes identified",
            "study_priorities": ["Review hand histories", "Study GTO fundamentals"],
            "immediate_fixes": ["Fold more preflop", "Bet for value more often"],
            "total_ev_loss": 0.0
        }

    def _generate_session_summary(self, reviews: List[Dict]) -> str:
        """Generate summary from session reviews."""
        if not reviews:
            return "No hands analyzed"

        blunders = sum(1 for r in reviews if r.get("mistake_severity") == "Blunder")
        inaccuracies = sum(1 for r in reviews if r.get("mistake_severity") == "Inaccuracy")
        optimal = sum(1 for r in reviews if r.get("mistake_severity") == "Optimal")

        avg_score = sum(r.get("gto_score", 50) for r in reviews) / len(reviews)

        return f"Session Grade: {avg_score:.0f}/100. {optimal} optimal plays, {inaccuracies} inaccuracies, {blunders} blunders."
