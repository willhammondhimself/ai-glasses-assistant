"""
Generate test data for pattern analysis testing.

Creates realistic session summaries with patterns:
- Study sessions: Mon-Fri at 2pm
- Poker sessions: Sat-Sun at 7pm
- Focus sessions: Weekdays at 9am
- Random captures: 3-10 per day

Usage:
    python -m phone_client.intelligence.test_data_generator
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import random


def generate_test_sessions(days: int = 30):
    """Generate realistic test session data."""

    output_dir = Path(__file__).parent.parent.parent / "logs" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.now() - timedelta(days=days)

    print(f"Generating {days} days of test data...")

    for day in range(days):
        date = start_date + timedelta(days=day)
        day_of_week = date.weekday()

        # Simulate patterns:
        # - Study sessions: Mon-Fri at 2pm
        # - Poker sessions: Sat-Sun at 7pm
        # - Focus sessions: Weekdays at 9am

        session = {
            'date': date.isoformat(),
            'total_cost': 0.0,
            'captures': [],
            'total_duration_seconds': 0,
            'battery_time_seconds': random.randint(14400, 28800),  # 4-8 hours
        }

        # Focus sessions on weekdays
        if day_of_week < 5:  # Mon-Fri
            session['focus'] = {
                'count': random.randint(1, 3),
                'cost': 0.0,
                'duration_seconds': random.randint(1500, 2700)  # 25-45 min
            }
            session['focus_start_hour'] = 9
            session['total_duration_seconds'] += session['focus']['duration_seconds']

        # Study sessions on weekdays afternoon
        if day_of_week < 5:
            num_problems = random.randint(5, 15)
            session['homework'] = {
                'count': num_problems,
                'cost': random.uniform(0.20, 0.60),
                'duration_seconds': num_problems * random.randint(60, 180),
                'details': {
                    'local_percentage': random.uniform(40, 80),
                    'problem_types': ['algebra', 'calculus', 'physics']
                }
            }
            session['homework_start_hour'] = 14  # 2pm
            session['total_cost'] += session['homework']['cost']
            session['total_duration_seconds'] += session['homework']['duration_seconds']

        # Poker on weekends
        if day_of_week >= 5:  # Sat-Sun
            num_hands = random.randint(30, 80)
            session['poker'] = {
                'count': num_hands,
                'cost': random.uniform(0.80, 1.50),
                'duration_seconds': num_hands * random.randint(60, 120),
                'details': {
                    'profit_bb': random.uniform(-10, 20),
                    'mistakes': random.randint(2, 8)
                }
            }
            session['poker_start_hour'] = 19  # 7pm
            session['total_cost'] += session['poker']['cost']
            session['total_duration_seconds'] += session['poker']['duration_seconds']

        # Random captures throughout the day
        num_captures = random.randint(3, 10)
        for i in range(num_captures):
            capture_time = date.replace(
                hour=random.randint(8, 20),
                minute=random.randint(0, 59),
                second=0,
                microsecond=0
            )
            session['captures'].append({
                'timestamp': capture_time.isoformat(),
                'text': f'Test capture {i+1} for {date.strftime("%Y-%m-%d")}',
                'category': random.choice(['general', 'study', 'poker', 'idea', 'urgent'])
            })

        # Add insights
        if session['total_cost'] > 1.5:
            session['top_insight'] = "High cost day - poker session ran long"
        elif 'homework' in session and session['homework']['count'] > 10:
            session['top_insight'] = "Productive study session"
        else:
            session['top_insight'] = "Regular day"

        # Save session
        filename = output_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)

    print(f"✓ Generated {days} days of test data in {output_dir}")
    print("\nGenerated patterns:")
    print("  - Focus sessions: Weekdays at 9am")
    print("  - Study sessions: Weekdays at 2pm (5-15 problems)")
    print("  - Poker sessions: Weekends at 7pm (30-80 hands)")
    print("  - Captures: 3-10 per day throughout the day")
    print("\nRun pattern analyzer to detect these patterns:")
    print("  python -c \"import asyncio; from phone_client.intelligence.pattern_analyzer import PatternAnalyzer; print(asyncio.run(PatternAnalyzer().analyze_all_patterns()))\"")


if __name__ == '__main__':
    generate_test_sessions(30)
