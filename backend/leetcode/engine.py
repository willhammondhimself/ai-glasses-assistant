"""
LeetCodeEngine - LeetCode problem fetching and caching.

Uses LeetCode's GraphQL API to fetch problems on-demand.
Caches results in SQLite for offline access and performance.
"""

import os
import json
import sqlite3
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Try to import httpx, fall back to requests
try:
    import httpx
    ASYNC_AVAILABLE = True
except ImportError:
    import requests
    ASYNC_AVAILABLE = False


class LeetCodeEngine:
    """
    LeetCode problem manager with caching.

    Features:
    - Fetch problems via GraphQL API
    - Cache problems in SQLite
    - Search and filter cached problems
    - Track user solutions
    """

    GRAPHQL_URL = "https://leetcode.com/graphql"
    CACHE_TTL_DAYS = 30  # Refetch problems older than this

    # Topics relevant for quant interviews
    JANE_STREET_TOPICS = [
        'dynamic-programming', 'math', 'recursion', 'binary-search',
        'divide-and-conquer', 'game-theory', 'probability-and-statistics',
        'number-theory', 'combinatorics', 'graph', 'tree'
    ]

    def __init__(self, db_path: str = None):
        """Initialize the LeetCode engine."""
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "query_history.db")
        self.db_path = db_path

    def _execute(self, query: str, params: list = None) -> Any:
        """Execute a query and return cursor."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params or [])
            conn.commit()
            return cursor

    def _fetch_all(self, query: str, params: list = None) -> List[Dict]:
        """Execute query and fetch all results."""
        cursor = self._execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _fetch_one(self, query: str, params: list = None) -> Optional[Dict]:
        """Execute query and fetch one result."""
        cursor = self._execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    async def fetch_problem(self, slug: str) -> Dict:
        """
        Fetch a problem by slug, with caching.

        Args:
            slug: LeetCode problem slug (e.g., 'two-sum')

        Returns:
            Problem dict with content, hints, code templates
        """
        # Check cache first
        cached = self._get_cached(slug)
        if cached and cached.get('content'):
            # Check if cache is still fresh
            if cached.get('fetched_at'):
                age_days = (time.time() - cached['fetched_at']) / 86400
                if age_days < self.CACHE_TTL_DAYS:
                    return {**self._format_problem(cached), 'cached': True}

        # Fetch from LeetCode API
        query = """
        query getQuestionDetail($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            questionId
            title
            titleSlug
            content
            difficulty
            topicTags { name slug }
            codeSnippets { lang langSlug code }
            hints
            exampleTestcases
          }
        }
        """

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://leetcode.com",
            "Origin": "https://leetcode.com"
        }

        try:
            if ASYNC_AVAILABLE:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self.GRAPHQL_URL,
                        json={"query": query, "variables": {"titleSlug": slug}},
                        headers=headers,
                        timeout=15.0
                    )
            else:
                resp = requests.post(
                    self.GRAPHQL_URL,
                    json={"query": query, "variables": {"titleSlug": slug}},
                    headers=headers,
                    timeout=15
                )

            if resp.status_code != 200:
                return {"error": f"LeetCode API error: {resp.status_code}"}

            data = resp.json()
            question = data.get("data", {}).get("question")

            if not question:
                return {"error": f"Problem '{slug}' not found"}

            # Cache the problem
            self._cache_problem(question)

            return {**self._format_problem(question), 'cached': False}

        except Exception as e:
            # Return cached version if available (even if stale)
            if cached:
                return {**self._format_problem(cached), 'cached': True, 'stale': True}
            return {"error": f"Failed to fetch problem: {str(e)}"}

    def fetch_problem_sync(self, slug: str) -> Dict:
        """Synchronous version of fetch_problem."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.fetch_problem(slug))
        finally:
            loop.close()

    def _get_cached(self, slug: str) -> Optional[Dict]:
        """Get problem from cache by slug."""
        return self._fetch_one(
            "SELECT * FROM leetcode_problems WHERE slug = ?",
            [slug]
        )

    def _cache_problem(self, question: Dict):
        """Cache a problem from API response."""
        now = int(time.time())

        # Extract topics as JSON array
        topics = [t['slug'] for t in question.get('topicTags', [])]

        # Extract code templates as JSON object
        templates = {}
        for snippet in question.get('codeSnippets', []):
            templates[snippet['langSlug']] = snippet['code']

        # Build test cases from examples
        test_cases = []
        if question.get('exampleTestcases'):
            # Parse example test cases (simple format)
            examples = question['exampleTestcases'].split('\n')
            test_cases = [{"input": ex, "expected": ""} for ex in examples if ex.strip()]

        self._execute(
            """
            INSERT OR REPLACE INTO leetcode_problems
            (id, slug, title, difficulty, content, code_templates, topics, hints, test_cases, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                int(question.get('questionId', 0)),
                question.get('titleSlug'),
                question.get('title'),
                question.get('difficulty'),
                question.get('content'),
                json.dumps(templates),
                json.dumps(topics),
                json.dumps(question.get('hints', [])),
                json.dumps(test_cases),
                now
            ]
        )

    def _format_problem(self, problem: Dict) -> Dict:
        """Format a problem for API response."""
        # Parse JSON fields if they're strings
        topics = problem.get('topics', '[]')
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except (json.JSONDecodeError, TypeError):
                topics = []

        templates = problem.get('code_templates', '{}')
        if isinstance(templates, str):
            try:
                templates = json.loads(templates)
            except (json.JSONDecodeError, TypeError):
                templates = {}

        hints = problem.get('hints', '[]')
        if isinstance(hints, str):
            try:
                hints = json.loads(hints)
            except (json.JSONDecodeError, TypeError):
                hints = []

        test_cases = problem.get('test_cases', '[]')
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except (json.JSONDecodeError, TypeError):
                test_cases = []

        # Handle topicTags from API response
        if 'topicTags' in problem:
            topics = [t['slug'] for t in problem.get('topicTags', [])]

        # Handle codeSnippets from API response
        if 'codeSnippets' in problem:
            templates = {}
            for snippet in problem.get('codeSnippets', []):
                templates[snippet['langSlug']] = snippet['code']

        return {
            "id": problem.get('id') or problem.get('questionId'),
            "slug": problem.get('slug') or problem.get('titleSlug'),
            "title": problem.get('title'),
            "difficulty": problem.get('difficulty'),
            "content": problem.get('content'),
            "topics": topics,
            "code_templates": templates,
            "hints": hints,
            "test_cases": test_cases,
            "hints_count": len(hints) if hints else 0
        }

    def search_problems(
        self,
        difficulty: str = None,
        topics: List[str] = None,
        jane_street_mode: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> Dict:
        """
        Search cached problems with filters.

        Args:
            difficulty: 'Easy', 'Medium', or 'Hard'
            topics: List of topic slugs to filter by
            jane_street_mode: Filter for quant-relevant problems
            limit: Max results to return
            offset: Pagination offset

        Returns:
            Dict with problems list and total count
        """
        query = "SELECT id, slug, title, difficulty, topics FROM leetcode_problems WHERE 1=1"
        count_query = "SELECT COUNT(*) as total FROM leetcode_problems WHERE 1=1"
        params = []
        count_params = []

        if difficulty:
            query += " AND difficulty = ?"
            count_query += " AND difficulty = ?"
            params.append(difficulty)
            count_params.append(difficulty)

        if jane_street_mode:
            topics = self.JANE_STREET_TOPICS

        if topics:
            topic_conditions = ["topics LIKE ?" for _ in topics]
            topic_clause = f" AND ({' OR '.join(topic_conditions)})"
            query += topic_clause
            count_query += topic_clause
            topic_params = [f'%"{t}"%' for t in topics]
            params.extend(topic_params)
            count_params.extend(topic_params)

        # Get total count
        total = self._fetch_one(count_query, count_params)

        # Get paginated results
        query += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        problems = self._fetch_all(query, params)

        # Parse topics JSON for each problem
        for p in problems:
            if p.get('topics'):
                try:
                    p['topics'] = json.loads(p['topics'])
                except (json.JSONDecodeError, TypeError):
                    p['topics'] = []

        return {
            "problems": problems,
            "total": total['total'] if total else 0,
            "limit": limit,
            "offset": offset
        }

    def get_hints(self, problem_id: int, level: int) -> Dict:
        """
        Get hint at specified level (0-2).

        Args:
            problem_id: LeetCode problem ID
            level: Hint level (0 = first hint, 1 = second, etc.)

        Returns:
            Hint text or error
        """
        problem = self._fetch_one(
            "SELECT hints FROM leetcode_problems WHERE id = ?",
            [problem_id]
        )

        if not problem:
            return {"error": "Problem not found"}

        hints = problem.get('hints', '[]')
        if isinstance(hints, str):
            try:
                hints = json.loads(hints)
            except (json.JSONDecodeError, TypeError):
                hints = []

        if level >= len(hints):
            return {"error": "No more hints available", "total": len(hints)}

        return {
            "hint": hints[level],
            "level": level,
            "total": len(hints),
            "has_more": level < len(hints) - 1
        }

    def save_solution(
        self,
        problem_id: int,
        code: str,
        language: str,
        passed: bool = False,
        time_taken_ms: int = None,
        hints_used: int = 0
    ) -> Dict:
        """
        Save a user's solution.

        Args:
            problem_id: LeetCode problem ID
            code: Solution code
            language: Programming language
            passed: Whether solution passed
            time_taken_ms: Time spent solving
            hints_used: Number of hints used

        Returns:
            Saved solution info
        """
        cursor = self._execute(
            """
            INSERT INTO leetcode_solutions
            (problem_id, code, language, passed, time_taken_ms, hints_used)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [problem_id, code, language, passed, time_taken_ms, hints_used]
        )

        # Update daily activity if passed
        if passed:
            today = datetime.now().strftime('%Y-%m-%d')
            self._execute(
                """
                INSERT INTO daily_activity (date, leetcode_solved)
                VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET leetcode_solved = leetcode_solved + 1
                """,
                [today]
            )

        return {
            "id": cursor.lastrowid,
            "saved": True,
            "passed": passed
        }

    def get_solutions(self, problem_id: int) -> List[Dict]:
        """Get all solutions for a problem."""
        return self._fetch_all(
            """
            SELECT id, code, language, passed, time_taken_ms, hints_used, submitted_at
            FROM leetcode_solutions
            WHERE problem_id = ?
            ORDER BY submitted_at DESC
            """,
            [problem_id]
        )

    def get_progress(self) -> Dict:
        """Get user's LeetCode progress."""
        total_solved = self._fetch_one(
            "SELECT COUNT(DISTINCT problem_id) as count FROM leetcode_solutions WHERE passed = 1"
        )

        by_difficulty = self._fetch_all(
            """
            SELECT p.difficulty, COUNT(DISTINCT s.problem_id) as count
            FROM leetcode_solutions s
            JOIN leetcode_problems p ON s.problem_id = p.id
            WHERE s.passed = 1
            GROUP BY p.difficulty
            """
        )

        recent = self._fetch_all(
            """
            SELECT DISTINCT p.id, p.slug, p.title, p.difficulty, MAX(s.submitted_at) as solved_at
            FROM leetcode_solutions s
            JOIN leetcode_problems p ON s.problem_id = p.id
            WHERE s.passed = 1
            GROUP BY p.id
            ORDER BY solved_at DESC
            LIMIT 10
            """
        )

        difficulty_counts = {d['difficulty']: d['count'] for d in by_difficulty}

        return {
            "total_solved": total_solved['count'] if total_solved else 0,
            "easy_solved": difficulty_counts.get('Easy', 0),
            "medium_solved": difficulty_counts.get('Medium', 0),
            "hard_solved": difficulty_counts.get('Hard', 0),
            "recent_problems": recent
        }

    def get_cached_count(self) -> int:
        """Get number of cached problems."""
        result = self._fetch_one("SELECT COUNT(*) as count FROM leetcode_problems")
        return result['count'] if result else 0
