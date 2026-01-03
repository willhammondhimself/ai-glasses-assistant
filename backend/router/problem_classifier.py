"""
Problem Classifier

Classifies incoming problems by type and complexity
to enable intelligent routing decisions.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Classification:
    """Result of problem classification."""
    problem_type: str  # math, cs, chemistry, biology, statistics, poker, quant, vision
    complexity: str  # trivial, simple, moderate, complex, expert
    confidence: float  # 0.0-1.0
    patterns_matched: list[str] = field(default_factory=list)
    suggested_engine: Optional[str] = None
    sub_type: Optional[str] = None  # e.g., "calculus", "probability"


class ProblemClassifier:
    """
    Classifies problems by type and complexity using pattern matching.

    Type detection identifies the domain (math, cs, etc.)
    Complexity detection estimates computational difficulty.
    """

    # Complexity patterns (ordered from trivial to expert)
    COMPLEXITY_PATTERNS = {
        "trivial": [
            r"^\s*\d+\s*[\+\-\*\/]\s*\d+\s*$",  # Basic arithmetic: 2 + 2
            r"^\s*\d+\s*[\+\-\*\/]\s*\d+\s*[\+\-\*\/]\s*\d+\s*$",  # 2 + 2 + 2
            r"^(what is|calculate|compute)\s+\d+\s*[\+\-\*\/]",  # "what is 5 + 3"
            r"^\d+\s*[\%]\s*\d+$",  # modulo
            r"^\d+\s*\*\*\s*\d+$",  # simple power
        ],
        "simple": [
            r"(derivative|integral)\s+of\s+\w+",  # Basic calculus
            r"solve\s+(for\s+)?\w+",  # solve for x
            r"factor(ize)?\s+",  # factoring
            r"simplify\s+",  # simplification
            r"(evaluate|find)\s+",  # evaluation
            r"balance\s+.*(equation|reaction)",  # balance equation
            r"probability\s+of\s+",  # basic probability
            r"(mean|median|mode)\s+of",  # basic stats
        ],
        "moderate": [
            r"system\s+of\s+(equations|inequalities)",  # systems
            r"(optimize|maximize|minimize)",  # optimization
            r"(matrix|matrices)",  # linear algebra
            r"(regression|correlation)",  # statistics
            r"(expected\s+value|variance|std)",  # probability theory
            r"black.?scholes",  # options pricing
            r"(binary\s+search|sort|hash)",  # algorithms
            r"(recursive|recursion)",  # recursive problems
            r"time\s+complexity",  # complexity analysis
        ],
        "complex": [
            r"(differential\s+equation|PDE|ODE)",  # differential equations
            r"(proof|prove|theorem|lemma)",  # proofs
            r"(eigenvalue|eigenvector)",  # advanced linear algebra
            r"(fourier|laplace)\s+transform",  # transforms
            r"(monte\s+carlo|simulation)",  # simulations
            r"(dynamic\s+programming|DP)",  # DP problems
            r"(graph|tree)\s+(algorithm|traversal)",  # graph algorithms
            r"(NP|polynomial\s+time|complexity\s+class)",  # complexity theory
        ],
        "expert": [
            r"(research|novel|original|cutting.?edge)",  # research-level
            r"(unsolved|open\s+problem)",  # unsolved problems
            r"(stochastic\s+calculus|ito)",  # advanced quant
            r"(tensor|manifold)",  # advanced math
        ]
    }

    # Type patterns with keywords
    TYPE_PATTERNS = {
        "math": {
            "patterns": [
                r"(equation|integral|derivative|differentiate)",
                r"(solve|calculate|compute|evaluate)",
                r"(factor|simplify|expand)",
                r"(limit|series|sequence)",
                r"(matrix|vector|eigenvalue)",
                r"(algebra|calculus|geometry)",
                r"[\+\-\*\/\=\^]",  # math operators
                r"\d+\s*[\+\-\*\/]\s*\d+",  # numeric expressions
            ],
            "keywords": ["x", "y", "sin", "cos", "log", "ln", "sqrt", "pi", "∫", "∑"],
            "priority": 1
        },
        "cs": {
            "patterns": [
                r"(algorithm|code|function|method|class)",
                r"(complexity|big.?o|runtime)",
                r"(debug|error|bug|fix)",
                r"(array|list|dict|hash|tree|graph)",
                r"(recursive|iterate|loop)",
                r"(sort|search|traverse)",
                r"```\w+",  # code blocks
            ],
            "keywords": ["python", "javascript", "java", "c++", "def", "function", "return"],
            "priority": 2
        },
        "chemistry": {
            "patterns": [
                r"(molecule|compound|element|atom)",
                r"(reaction|equation|balance)",
                r"(molarity|concentration|dilution)",
                r"(pH|acid|base|buffer)",
                r"(oxidation|reduction|redox)",
                r"(bond|orbital|hybridization)",
                r"[A-Z][a-z]?\d*",  # chemical formulas
            ],
            "keywords": ["H2O", "NaCl", "CO2", "mol", "molar", "stoichiometry"],
            "priority": 3
        },
        "biology": {
            "patterns": [
                r"(gene|DNA|RNA|protein)",
                r"(punnett|genetics|allele|genotype)",
                r"(cell|membrane|mitochondria)",
                r"(evolution|natural\s+selection)",
                r"(photosynthesis|respiration)",
                r"(organism|species|taxonomy)",
            ],
            "keywords": ["dominant", "recessive", "chromosome", "mutation"],
            "priority": 4
        },
        "statistics": {
            "patterns": [
                r"(probability|likelihood|chance)",
                r"(mean|median|mode|average)",
                r"(variance|std|deviation)",
                r"(regression|correlation|r.?squared)",
                r"(hypothesis|p.?value|significance)",
                r"(distribution|normal|binomial|poisson)",
                r"(sample|population|confidence)",
            ],
            "keywords": ["μ", "σ", "α", "β", "null hypothesis", "z-score", "t-test"],
            "priority": 5
        },
        "poker": {
            "patterns": [
                r"(poker|hand|cards?)",
                r"(pot\s+odds|equity|EV)",
                r"(fold|call|raise|bet|all.?in)",
                r"(flop|turn|river|pre.?flop)",
                r"(suited|offsuit|pair|flush|straight)",
                r"([2-9TJQKA][shdc])",  # card notation
            ],
            "keywords": ["texas", "holdem", "blinds", "position"],
            "priority": 6
        },
        "quant": {
            "patterns": [
                r"(option|derivative|futures)",
                r"(black.?scholes|greeks?|delta|gamma|theta|vega)",
                r"(volatility|implied\s+vol)",
                r"(sharpe|sortino|risk)",
                r"(kelly|bankroll|bet\s+size)",
                r"(fermi|estimation|approximate)",
                r"(market.?making|spread|bid.?ask)",
            ],
            "keywords": ["strike", "expiry", "call", "put", "hedge", "arbitrage"],
            "priority": 7
        },
        "vision": {
            "patterns": [
                r"(image|picture|photo|screenshot)",
                r"(describe|analyze|identify|recognize)",
                r"(OCR|text\s+in|read\s+from)",
                r"(chart|graph|diagram|figure)",
            ],
            "keywords": ["attached", "uploaded", "see", "look at"],
            "priority": 8
        }
    }

    # Sub-type detection for more granular classification
    SUB_TYPES = {
        "math": {
            "calculus": [r"(derivative|integral|differentiate|limit)"],
            "algebra": [r"(solve|equation|factor|simplify)"],
            "linear_algebra": [r"(matrix|vector|eigenvalue|determinant)"],
            "geometry": [r"(triangle|circle|angle|area|volume)"],
        },
        "cs": {
            "algorithms": [r"(algorithm|sort|search|complexity)"],
            "debugging": [r"(debug|error|bug|fix|issue)"],
            "code_review": [r"(review|improve|refactor|optimize)"],
        },
        "quant": {
            "options": [r"(option|black.?scholes|greek|strike)"],
            "probability": [r"(probability|dice|cards|combinatorics)"],
            "mental_math": [r"(mental|quick|fast|calculate)"],
            "fermi": [r"(fermi|estimate|approximate|how\s+many)"],
            "market_making": [r"(market.?mak|spread|bid.?ask)"],
        }
    }

    def __init__(self):
        # Compile all patterns for efficiency
        self._compiled_complexity = {
            level: [re.compile(p, re.IGNORECASE) for p in patterns]
            for level, patterns in self.COMPLEXITY_PATTERNS.items()
        }
        self._compiled_types = {
            ptype: [re.compile(p, re.IGNORECASE) for p in config["patterns"]]
            for ptype, config in self.TYPE_PATTERNS.items()
        }
        self._compiled_subtypes = {
            ptype: {
                subtype: [re.compile(p, re.IGNORECASE) for p in patterns]
                for subtype, patterns in subtypes.items()
            }
            for ptype, subtypes in self.SUB_TYPES.items()
        }

    def classify(self, problem: str, **kwargs) -> Classification:
        """
        Classify a problem by type and complexity.

        Args:
            problem: The problem text
            **kwargs: Additional context (e.g., endpoint hint)

        Returns:
            Classification with type, complexity, and confidence
        """
        problem_lower = problem.lower()
        matched_patterns = []

        # Detect problem type
        problem_type, type_confidence = self._detect_type(problem, problem_lower, matched_patterns)

        # Detect complexity
        complexity, complexity_confidence = self._detect_complexity(problem_lower, matched_patterns)

        # Detect sub-type
        sub_type = self._detect_subtype(problem_type, problem_lower)

        # Overall confidence is combination of type and complexity detection
        confidence = (type_confidence * 0.7 + complexity_confidence * 0.3)

        # Suggest engine based on type and complexity
        suggested_engine = self._suggest_engine(problem_type, complexity)

        return Classification(
            problem_type=problem_type,
            complexity=complexity,
            confidence=confidence,
            patterns_matched=matched_patterns,
            suggested_engine=suggested_engine,
            sub_type=sub_type
        )

    def _detect_type(self, problem: str, problem_lower: str, matched: list) -> tuple[str, float]:
        """Detect problem type with confidence score."""
        scores = {}

        for ptype, patterns in self._compiled_types.items():
            config = self.TYPE_PATTERNS[ptype]
            score = 0

            # Check regex patterns
            for pattern in patterns:
                if pattern.search(problem):
                    score += 1
                    matched.append(f"{ptype}:{pattern.pattern[:30]}")

            # Check keywords
            for keyword in config.get("keywords", []):
                if keyword.lower() in problem_lower:
                    score += 0.5

            if score > 0:
                scores[ptype] = score

        if not scores:
            return "math", 0.3  # Default to math with low confidence

        # Get highest scoring type
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]

        # Normalize confidence (cap at 1.0)
        confidence = min(1.0, max_score / 3)

        return best_type, confidence

    def _detect_complexity(self, problem_lower: str, matched: list) -> tuple[str, float]:
        """Detect problem complexity level."""
        # Check patterns from expert down to trivial
        complexity_order = ["expert", "complex", "moderate", "simple", "trivial"]

        for level in complexity_order:
            patterns = self._compiled_complexity[level]
            for pattern in patterns:
                if pattern.search(problem_lower):
                    matched.append(f"complexity:{level}")
                    # Higher confidence for more specific matches
                    confidence = 0.9 if level in ["trivial", "expert"] else 0.7
                    return level, confidence

        # Default to moderate if no patterns match
        return "moderate", 0.5

    def _detect_subtype(self, problem_type: str, problem_lower: str) -> Optional[str]:
        """Detect sub-type within a problem type."""
        if problem_type not in self._compiled_subtypes:
            return None

        subtypes = self._compiled_subtypes[problem_type]
        for subtype, patterns in subtypes.items():
            for pattern in patterns:
                if pattern.search(problem_lower):
                    return subtype

        return None

    def _suggest_engine(self, problem_type: str, complexity: str) -> str:
        """Suggest an engine based on type and complexity."""
        # Engine suggestions based on type
        engine_map = {
            "math": "sympy" if complexity in ["trivial", "simple"] else "claude",
            "cs": "local_analysis" if complexity == "trivial" else "claude",
            "chemistry": "rdkit" if complexity in ["trivial", "simple"] else "claude",
            "biology": "claude",  # Usually needs Claude
            "statistics": "scipy" if complexity in ["trivial", "simple"] else "claude",
            "poker": "poker_calc",
            "quant": "numpy" if complexity in ["trivial", "simple"] else "claude",
            "vision": "claude",  # Always needs Claude
        }

        return engine_map.get(problem_type, "claude")
