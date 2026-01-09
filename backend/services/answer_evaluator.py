"""Answer Evaluator - Semantic answer comparison using Gemini."""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """Evaluate user answers against correct answers using Gemini.

    Handles:
    - Numeric variations (42 vs "forty-two")
    - Synonyms and paraphrasing
    - Partial answers
    - Mathematical expressions
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel("gemini-2.0-flash-exp")
        return self._client

    async def evaluate(
        self,
        user_answer: str,
        correct_answer: str,
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate if user's answer matches the correct answer.

        Args:
            user_answer: What the user said
            correct_answer: The expected correct answer
            question: Optional question for context

        Returns:
            {
                "correct": bool,
                "confidence": float (0-1),
                "quality": int (0-5 for SM-2),
                "feedback": str (voice response)
            }
        """
        try:
            model = self._get_client()

            # Build prompt for evaluation
            prompt = self._build_prompt(user_answer, correct_answer, question)

            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200
                }
            )

            return self._parse_response(response.text, user_answer, correct_answer)

        except Exception as e:
            logger.error(f"Answer evaluation error: {e}")
            # Fallback to simple comparison
            return self._simple_compare(user_answer, correct_answer)

    def _build_prompt(
        self,
        user_answer: str,
        correct_answer: str,
        question: Optional[str]
    ) -> str:
        """Build the evaluation prompt."""
        context = f"Question: {question}\n" if question else ""

        return f"""You are evaluating a flashcard quiz answer. Determine if the user's answer is correct.

{context}Correct Answer: {correct_answer}
User's Answer: {user_answer}

Evaluate whether the user's answer is semantically equivalent to the correct answer.
Consider:
- Numeric equivalence (42 = forty-two = 42.0)
- Synonyms and paraphrasing
- Partial correctness (got the main concept)
- Minor errors vs wrong answer

Respond with ONLY a JSON object (no markdown, no explanation):
{{"correct": true/false, "confidence": 0.0-1.0, "quality": 0-5, "reason": "brief reason"}}

Quality scale (SM-2 algorithm):
5 = Perfect response with no hesitation
4 = Correct response after brief hesitation
3 = Correct but with difficulty
2 = Wrong but related/close
1 = Wrong but remembered when shown
0 = Complete blackout"""

    def _parse_response(
        self,
        response_text: str,
        user_answer: str,
        correct_answer: str
    ) -> Dict[str, Any]:
        """Parse Gemini's response into evaluation result."""
        import json

        try:
            # Clean response text (remove markdown if present)
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)

            correct = result.get("correct", False)
            confidence = float(result.get("confidence", 0.5))
            quality = int(result.get("quality", 1 if not correct else 4))
            reason = result.get("reason", "")

            # Build feedback message
            if correct:
                if confidence >= 0.9:
                    feedback = "Correct!"
                elif confidence >= 0.7:
                    feedback = f"That's right! {reason}" if reason else "Correct!"
                else:
                    feedback = f"I'll count that as correct. {reason}" if reason else "Close enough!"
            else:
                feedback = f"Not quite. The answer is: {correct_answer}"

            return {
                "correct": correct,
                "confidence": confidence,
                "quality": quality,
                "feedback": feedback
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return self._simple_compare(user_answer, correct_answer)

    def _simple_compare(
        self,
        user_answer: str,
        correct_answer: str
    ) -> Dict[str, Any]:
        """Fallback: simple string comparison."""
        user_clean = user_answer.lower().strip()
        correct_clean = correct_answer.lower().strip()

        # Exact match
        if user_clean == correct_clean:
            return {
                "correct": True,
                "confidence": 1.0,
                "quality": 5,
                "feedback": "Correct!"
            }

        # Check if user answer contains the correct answer
        if correct_clean in user_clean or user_clean in correct_clean:
            return {
                "correct": True,
                "confidence": 0.8,
                "quality": 4,
                "feedback": "Correct!"
            }

        # Try numeric comparison
        try:
            user_num = float(user_answer)
            correct_num = float(correct_answer)
            if abs(user_num - correct_num) < 0.001:
                return {
                    "correct": True,
                    "confidence": 1.0,
                    "quality": 5,
                    "feedback": "Correct!"
                }
        except (ValueError, TypeError):
            pass

        return {
            "correct": False,
            "confidence": 0.9,
            "quality": 1,
            "feedback": f"Not quite. The answer is: {correct_answer}"
        }


# Global instance
_evaluator: Optional[AnswerEvaluator] = None


def get_answer_evaluator() -> AnswerEvaluator:
    """Get or create global AnswerEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = AnswerEvaluator()
    return _evaluator
