"""
CSEngine: Computer Science assistant for code explanation, debugging, and algorithm analysis.

Uses Claude Sonnet 3.5 for intelligent code analysis and explanation.
"""

import os
from typing import Optional, List
import anthropic
from dotenv import load_dotenv

load_dotenv()


class CSEngine:
    """Computer Science assistant powered by Claude Sonnet 3.5."""

    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def explain_code(self, code: str, language: str = "auto") -> dict:
        """
        Explain what a piece of code does.

        Args:
            code: The source code to explain
            language: Programming language (or "auto" to detect)

        Returns:
            dict with keys: explanation, key_concepts, complexity, suggestions, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            lang_hint = f" (Language: {language})" if language != "auto" else ""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Explain this code clearly and concisely.{lang_hint}

```
{code}
```

Provide:
1. EXPLANATION: What does this code do? (2-4 sentences)
2. KEY_CONCEPTS: List the main programming concepts used (comma-separated)
3. COMPLEXITY: Time/space complexity if applicable (or "N/A")
4. SUGGESTIONS: Any improvements or best practices (1-2 bullet points, or "None")

Format your response exactly as:
EXPLANATION: [explanation]
KEY_CONCEPTS: [concept1, concept2, ...]
COMPLEXITY: [complexity analysis]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]"""
                    }
                ]
            )

            return self._parse_explanation_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def debug_code(self, code: str, error: str, language: str = "auto") -> dict:
        """
        Help debug code given an error message.

        Args:
            code: The source code with the bug
            error: The error message or description of the problem
            language: Programming language

        Returns:
            dict with keys: diagnosis, fix, fixed_code, explanation, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            lang_hint = f" (Language: {language})" if language != "auto" else ""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Debug this code.{lang_hint}

CODE:
```
{code}
```

ERROR/PROBLEM:
{error}

Provide:
1. DIAGNOSIS: What's causing the issue? (1-2 sentences)
2. FIX: How to fix it (brief description)
3. FIXED_CODE: The corrected code
4. EXPLANATION: Why this fix works (1 sentence)

Format your response exactly as:
DIAGNOSIS: [diagnosis]
FIX: [fix description]
FIXED_CODE:
```
[corrected code]
```
EXPLANATION: [explanation]"""
                    }
                ]
            )

            return self._parse_debug_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def explain_algorithm(self, algorithm: str) -> dict:
        """
        Explain an algorithm in detail.

        Args:
            algorithm: Name or description of the algorithm

        Returns:
            dict with keys: explanation, how_it_works, complexity, use_cases, pseudocode, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Explain this algorithm: {algorithm}

Provide:
1. EXPLANATION: Brief overview (2-3 sentences)
2. HOW_IT_WORKS: Step-by-step explanation (numbered list)
3. COMPLEXITY: Time and space complexity
4. USE_CASES: When to use this algorithm (2-3 examples)
5. PSEUDOCODE: Simple pseudocode implementation

Format your response exactly as:
EXPLANATION: [overview]
HOW_IT_WORKS:
1. [step 1]
2. [step 2]
...
COMPLEXITY: Time: O(?), Space: O(?)
USE_CASES:
- [use case 1]
- [use case 2]
PSEUDOCODE:
```
[pseudocode]
```"""
                    }
                ]
            )

            return self._parse_algorithm_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def compare_approaches(self, problem: str, approaches: List[str]) -> dict:
        """
        Compare different approaches to solving a problem.

        Args:
            problem: Description of the problem
            approaches: List of approach names or descriptions

        Returns:
            dict with keys: comparison, recommendation, tradeoffs, error
        """
        if not self.client:
            return self._no_api_key_error()

        try:
            approaches_str = "\n".join(f"- {a}" for a in approaches)

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Compare these approaches for solving a problem.

PROBLEM: {problem}

APPROACHES:
{approaches_str}

Provide:
1. COMPARISON: Brief comparison table or description
2. TRADEOFFS: Key tradeoffs between approaches
3. RECOMMENDATION: Which to use and when

Format your response exactly as:
COMPARISON:
[comparison]
TRADEOFFS:
- [tradeoff 1]
- [tradeoff 2]
RECOMMENDATION: [recommendation]"""
                    }
                ]
            )

            return self._parse_comparison_response(message.content[0].text)

        except anthropic.APIError as e:
            return self._api_error(e)
        except Exception as e:
            return self._unexpected_error(e)

    def _parse_explanation_response(self, response_text: str) -> dict:
        """Parse code explanation response."""
        import re

        explanation = None
        key_concepts = []
        complexity = None
        suggestions = []

        # Parse EXPLANATION
        exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=KEY_CONCEPTS:|$)', response_text, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()

        # Parse KEY_CONCEPTS
        kc_match = re.search(r'KEY_CONCEPTS:\s*(.+?)(?=COMPLEXITY:|$)', response_text, re.DOTALL)
        if kc_match:
            concepts_str = kc_match.group(1).strip()
            key_concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]

        # Parse COMPLEXITY
        comp_match = re.search(r'COMPLEXITY:\s*(.+?)(?=SUGGESTIONS:|$)', response_text, re.DOTALL)
        if comp_match:
            complexity = comp_match.group(1).strip()

        # Parse SUGGESTIONS
        sug_match = re.search(r'SUGGESTIONS:\s*(.+?)$', response_text, re.DOTALL)
        if sug_match:
            sug_text = sug_match.group(1).strip()
            if sug_text.lower() != "none":
                suggestions = [line.strip().lstrip('- ') for line in sug_text.split('\n') if line.strip().startswith('-')]

        return {
            "explanation": explanation,
            "key_concepts": key_concepts,
            "complexity": complexity,
            "suggestions": suggestions,
            "error": None
        }

    def _parse_debug_response(self, response_text: str) -> dict:
        """Parse debug response."""
        import re

        diagnosis = None
        fix = None
        fixed_code = None
        explanation = None

        # Parse DIAGNOSIS
        diag_match = re.search(r'DIAGNOSIS:\s*(.+?)(?=FIX:|$)', response_text, re.DOTALL)
        if diag_match:
            diagnosis = diag_match.group(1).strip()

        # Parse FIX
        fix_match = re.search(r'FIX:\s*(.+?)(?=FIXED_CODE:|$)', response_text, re.DOTALL)
        if fix_match:
            fix = fix_match.group(1).strip()

        # Parse FIXED_CODE
        code_match = re.search(r'FIXED_CODE:\s*```[\w]*\n(.+?)```', response_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()

        # Parse EXPLANATION
        exp_match = re.search(r'```\s*\nEXPLANATION:\s*(.+?)$', response_text, re.DOTALL)
        if not exp_match:
            exp_match = re.search(r'EXPLANATION:\s*([^`]+?)$', response_text, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()

        return {
            "diagnosis": diagnosis,
            "fix": fix,
            "fixed_code": fixed_code,
            "explanation": explanation,
            "error": None
        }

    def _parse_algorithm_response(self, response_text: str) -> dict:
        """Parse algorithm explanation response."""
        import re

        explanation = None
        how_it_works = []
        complexity = None
        use_cases = []
        pseudocode = None

        # Parse EXPLANATION
        exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=HOW_IT_WORKS:|$)', response_text, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()

        # Parse HOW_IT_WORKS
        how_match = re.search(r'HOW_IT_WORKS:\s*(.+?)(?=COMPLEXITY:|$)', response_text, re.DOTALL)
        if how_match:
            how_text = how_match.group(1).strip()
            how_it_works = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in how_text.split('\n') if re.match(r'^\d+\.', line.strip())]

        # Parse COMPLEXITY
        comp_match = re.search(r'COMPLEXITY:\s*(.+?)(?=USE_CASES:|$)', response_text, re.DOTALL)
        if comp_match:
            complexity = comp_match.group(1).strip()

        # Parse USE_CASES
        use_match = re.search(r'USE_CASES:\s*(.+?)(?=PSEUDOCODE:|$)', response_text, re.DOTALL)
        if use_match:
            use_text = use_match.group(1).strip()
            use_cases = [line.strip().lstrip('- ') for line in use_text.split('\n') if line.strip().startswith('-')]

        # Parse PSEUDOCODE
        pseudo_match = re.search(r'PSEUDOCODE:\s*```[\w]*\n(.+?)```', response_text, re.DOTALL)
        if pseudo_match:
            pseudocode = pseudo_match.group(1).strip()

        return {
            "explanation": explanation,
            "how_it_works": how_it_works,
            "complexity": complexity,
            "use_cases": use_cases,
            "pseudocode": pseudocode,
            "error": None
        }

    def _parse_comparison_response(self, response_text: str) -> dict:
        """Parse comparison response."""
        import re

        comparison = None
        tradeoffs = []
        recommendation = None

        # Parse COMPARISON
        comp_match = re.search(r'COMPARISON:\s*(.+?)(?=TRADEOFFS:|$)', response_text, re.DOTALL)
        if comp_match:
            comparison = comp_match.group(1).strip()

        # Parse TRADEOFFS
        trade_match = re.search(r'TRADEOFFS:\s*(.+?)(?=RECOMMENDATION:|$)', response_text, re.DOTALL)
        if trade_match:
            trade_text = trade_match.group(1).strip()
            tradeoffs = [line.strip().lstrip('- ') for line in trade_text.split('\n') if line.strip().startswith('-')]

        # Parse RECOMMENDATION
        rec_match = re.search(r'RECOMMENDATION:\s*(.+?)$', response_text, re.DOTALL)
        if rec_match:
            recommendation = rec_match.group(1).strip()

        return {
            "comparison": comparison,
            "tradeoffs": tradeoffs,
            "recommendation": recommendation,
            "error": None
        }

    def _no_api_key_error(self) -> dict:
        """Return error for missing API key."""
        return {
            "error": "ANTHROPIC_API_KEY not configured"
        }

    def _api_error(self, e: anthropic.APIError) -> dict:
        """Return error for API errors."""
        return {
            "error": f"Claude API error: {str(e)}"
        }

    def _unexpected_error(self, e: Exception) -> dict:
        """Return error for unexpected errors."""
        return {
            "error": f"Unexpected error: {str(e)}"
        }
