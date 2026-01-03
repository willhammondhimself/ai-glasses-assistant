"""
FlashcardEngine - SM-2 Spaced Repetition Implementation

The SM-2 algorithm adjusts review intervals based on recall quality.
Quality ratings:
  0 - Complete blackout
  1 - Incorrect, remembered upon seeing answer
  2 - Incorrect, but easy to recall once seen
  3 - Correct with serious difficulty
  4 - Correct with some hesitation
  5 - Perfect recall
"""

import os
import json
import sqlite3
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def sm2_algorithm(quality: int, easiness: float, interval: int, repetitions: int) -> tuple:
    """
    SM-2 Spaced Repetition Algorithm.

    Args:
        quality: Rating 0-5 (0-2 = incorrect, 3-5 = correct)
        easiness: Current easiness factor (min 1.3)
        interval: Current interval in days
        repetitions: Number of successful repetitions

    Returns:
        Tuple of (new_easiness, new_interval, new_repetitions)
    """
    if quality >= 3:
        # Correct response
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = round(interval * easiness)
        new_repetitions = repetitions + 1
    else:
        # Incorrect response - reset
        new_repetitions = 0
        new_interval = 1

    # Update easiness factor (minimum 1.3)
    new_easiness = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

    return (new_easiness, new_interval, new_repetitions)


class FlashcardEngine:
    """
    Flashcard management with SM-2 spaced repetition.

    Features:
    - Create cards manually or from solved problems
    - SM-2 scheduling for optimal retention
    - Review history for heatmap visualization
    - Tag-based filtering and organization
    """

    def __init__(self, db_path: str = None):
        """Initialize the flashcard engine."""
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "query_history.db")
        self.db_path = db_path

    def _execute(self, query: str, params: list = None) -> Any:
        """Execute a query and return results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params or [])
            conn.commit()
            return cursor

    def _fetch_all(self, query: str, params: list = None) -> List[Dict]:
        """Execute query and fetch all results as dicts."""
        cursor = self._execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _fetch_one(self, query: str, params: list = None) -> Optional[Dict]:
        """Execute query and fetch one result as dict."""
        cursor = self._execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_card(
        self,
        front: str,
        back: str,
        source_type: str = "manual",
        source_id: str = None,
        tags: List[str] = None
    ) -> Dict:
        """
        Create a new flashcard.

        Args:
            front: Question/prompt text (supports LaTeX with $...$)
            back: Answer text (supports LaTeX)
            source_type: 'manual', 'quant', 'leetcode', etc.
            source_id: ID of source problem if auto-generated
            tags: List of tags for filtering

        Returns:
            Created card info with id
        """
        now = int(time.time())
        tags_json = json.dumps(tags) if tags else None

        cursor = self._execute(
            """
            INSERT INTO flashcards (front, back, source_type, source_id, tags, due_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [front, back, source_type, source_id, tags_json, now, now]
        )

        return {
            "id": cursor.lastrowid,
            "created": True,
            "due_date": now
        }

    def get_card(self, card_id: int) -> Optional[Dict]:
        """Get a single card by ID."""
        return self._fetch_one(
            "SELECT * FROM flashcards WHERE id = ?",
            [card_id]
        )

    def get_due_cards(
        self,
        limit: int = 20,
        source_type: str = None,
        tags: List[str] = None
    ) -> List[Dict]:
        """
        Get cards due for review.

        Uses indexed due_date column for efficient queries.

        Args:
            limit: Maximum cards to return
            source_type: Filter by source type
            tags: Filter by tags (OR matching)

        Returns:
            List of due cards sorted by urgency
        """
        now = int(time.time())
        query = "SELECT * FROM flashcards WHERE due_date <= ?"
        params = [now]

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        if tags:
            tag_conditions = ["tags LIKE ?" for _ in tags]
            query += f" AND ({' OR '.join(tag_conditions)})"
            params.extend([f'%"{t}"%' for t in tags])

        query += " ORDER BY due_date ASC LIMIT ?"
        params.append(limit)

        cards = self._fetch_all(query, params)

        # Parse tags JSON
        for card in cards:
            if card.get('tags'):
                try:
                    card['tags'] = json.loads(card['tags'])
                except (json.JSONDecodeError, TypeError):
                    card['tags'] = []

        return cards

    def get_all_cards(
        self,
        limit: int = 100,
        offset: int = 0,
        source_type: str = None
    ) -> List[Dict]:
        """Get all cards with pagination."""
        query = "SELECT * FROM flashcards WHERE 1=1"
        params = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        return self._fetch_all(query, params)

    def review_card(self, card_id: int, quality: int, time_ms: int = None) -> Dict:
        """
        Process a card review and update SM-2 parameters.

        Args:
            card_id: ID of the card being reviewed
            quality: Rating 0-5 (0=blackout, 5=perfect)
            time_ms: Response time in milliseconds

        Returns:
            Updated card info with next review date
        """
        if quality < 0 or quality > 5:
            return {"error": "Quality must be 0-5"}

        card = self.get_card(card_id)
        if not card:
            return {"error": "Card not found"}

        # Apply SM-2 algorithm
        new_ef, new_interval, new_reps = sm2_algorithm(
            quality,
            card['easiness_factor'],
            card['interval_days'],
            card['repetitions']
        )

        now = int(time.time())
        due_date = now + (new_interval * 86400)  # Convert days to seconds

        # Update card
        self._execute(
            """
            UPDATE flashcards
            SET easiness_factor = ?,
                interval_days = ?,
                repetitions = ?,
                due_date = ?,
                last_reviewed = ?,
                review_count = review_count + 1,
                correct_count = correct_count + ?
            WHERE id = ?
            """,
            [new_ef, new_interval, new_reps, due_date, now,
             1 if quality >= 3 else 0, card_id]
        )

        # Log review for heatmap
        self._log_review(card_id, quality, time_ms)

        # Update daily activity
        self._update_daily_activity()

        return {
            "card_id": card_id,
            "quality": quality,
            "correct": quality >= 3,
            "new_easiness": round(new_ef, 2),
            "new_interval_days": new_interval,
            "next_review": due_date,
            "next_review_human": self._format_interval(new_interval)
        }

    def _format_interval(self, days: int) -> str:
        """Format interval as human-readable string."""
        if days == 0:
            return "now"
        elif days == 1:
            return "1 day"
        elif days < 7:
            return f"{days} days"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''}"
        else:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''}"

    def _log_review(self, card_id: int, quality: int, time_ms: int = None):
        """Log a review to the review history."""
        self._execute(
            """
            INSERT INTO flashcard_reviews (flashcard_id, quality, time_ms)
            VALUES (?, ?, ?)
            """,
            [card_id, quality, time_ms]
        )

    def _update_daily_activity(self):
        """Update daily activity counter for heatmap."""
        today = datetime.now().strftime('%Y-%m-%d')
        self._execute(
            """
            INSERT INTO daily_activity (date, flashcard_reviews)
            VALUES (?, 1)
            ON CONFLICT(date) DO UPDATE SET flashcard_reviews = flashcard_reviews + 1
            """,
            [today]
        )

    def generate_from_problem(self, source_type: str, problem_data: Dict) -> Dict:
        """
        Auto-generate a flashcard from a solved problem.

        Args:
            source_type: Type of source ('quant', 'leetcode', 'math', etc.)
            problem_data: Dict with problem, answer, explanation, problem_id

        Returns:
            Created card info
        """
        front = problem_data.get("problem", "")
        back = f"Answer: {problem_data.get('answer', '')}"

        if problem_data.get('explanation'):
            back += f"\n\n{problem_data['explanation']}"

        if problem_data.get('steps'):
            back += f"\n\nSteps:\n{problem_data['steps']}"

        tags = [source_type]
        if problem_data.get('problem_type'):
            tags.append(problem_data['problem_type'])
        if problem_data.get('difficulty'):
            tags.append(problem_data['difficulty'])

        return self.create_card(
            front=front,
            back=back,
            source_type=source_type,
            source_id=problem_data.get('problem_id'),
            tags=tags
        )

    def sync_reviews(self, reviews: List[Dict]) -> Dict:
        """
        Sync multiple reviews (for offline mode).

        Args:
            reviews: List of {card_id, quality, reviewed_at}

        Returns:
            Sync results
        """
        results = []
        for review in reviews:
            result = self.review_card(
                review.get('card_id'),
                review.get('quality'),
                review.get('time_ms')
            )
            results.append(result)

        return {
            "synced": len(results),
            "results": results
        }

    def delete_card(self, card_id: int) -> Dict:
        """Delete a flashcard."""
        card = self.get_card(card_id)
        if not card:
            return {"error": "Card not found"}

        self._execute("DELETE FROM flashcard_reviews WHERE flashcard_id = ?", [card_id])
        self._execute("DELETE FROM flashcards WHERE id = ?", [card_id])

        return {"deleted": True, "card_id": card_id}

    def get_stats(self) -> Dict:
        """Get flashcard statistics."""
        total = self._fetch_one("SELECT COUNT(*) as count FROM flashcards")
        due = self._fetch_one(
            "SELECT COUNT(*) as count FROM flashcards WHERE due_date <= ?",
            [int(time.time())]
        )
        mastered = self._fetch_one(
            "SELECT COUNT(*) as count FROM flashcards WHERE interval_days >= 21"
        )
        reviews_today = self._fetch_one(
            """
            SELECT COUNT(*) as count FROM flashcard_reviews
            WHERE DATE(reviewed_at, 'unixepoch') = DATE('now')
            """
        )

        # Calculate retention rate (correct / total reviews)
        retention = self._fetch_one(
            """
            SELECT
                COALESCE(SUM(CASE WHEN quality >= 3 THEN 1 ELSE 0 END), 0) as correct,
                COUNT(*) as total
            FROM flashcard_reviews
            """
        )
        retention_rate = 0
        if retention and retention['total'] > 0:
            retention_rate = round(retention['correct'] / retention['total'] * 100, 1)

        return {
            "total_cards": total['count'] if total else 0,
            "due_today": due['count'] if due else 0,
            "mastered": mastered['count'] if mastered else 0,
            "reviews_today": reviews_today['count'] if reviews_today else 0,
            "retention_rate": retention_rate
        }

    def get_heatmap_data(self, year: int = None) -> List[Dict]:
        """
        Get daily activity data for heatmap visualization.

        Args:
            year: Year to get data for (default: current year)

        Returns:
            List of {date, flashcard_reviews, leetcode_solved, quant_problems}
        """
        if year is None:
            year = datetime.now().year

        return self._fetch_all(
            """
            SELECT date, flashcard_reviews, leetcode_solved, quant_problems
            FROM daily_activity
            WHERE date LIKE ?
            ORDER BY date
            """,
            [f"{year}-%"]
        )

    def get_review_forecast(self, days: int = 7) -> List[Dict]:
        """
        Get forecast of cards due in the next N days.

        Args:
            days: Number of days to forecast

        Returns:
            List of {date, count}
        """
        now = int(time.time())
        end = now + (days * 86400)

        return self._fetch_all(
            """
            SELECT DATE(due_date, 'unixepoch') as date, COUNT(*) as count
            FROM flashcards
            WHERE due_date BETWEEN ? AND ?
            GROUP BY DATE(due_date, 'unixepoch')
            ORDER BY date
            """,
            [now, end]
        )
