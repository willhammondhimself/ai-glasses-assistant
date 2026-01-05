#!/usr/bin/env python3
"""
Test script for Phase 7: Homework Vision Copilot.

Tests:
1. Homework history module (JSON logging)
2. API endpoints (solve, history, stats)
3. Integration with academic tools
"""

import asyncio
import sys
import os
import time
import requests
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = "http://localhost:8000/dashboard/api"


def test_homework_history_module():
    """Test the homework history JSON logging module."""
    print("\n=== Testing Homework History Module ===")

    from backend.homework import history

    # Clear history for testing
    history_file = history.HISTORY_FILE
    if history_file.exists():
        history_file.unlink()

    # Test 1: Append entry
    print("1. Testing append_entry...")
    entry_id = history.append_entry({
        "problem_latex": "\\frac{d}{dx}(x^2) = 2x",
        "problem_type": "calculus_derivative",
        "solution_summary": "Apply power rule: bring down exponent, reduce by 1"
    })
    assert entry_id, "Failed to generate entry ID"
    print(f"   Created entry: {entry_id[:8]}...")

    # Test 2: Query history
    print("2. Testing query_history...")
    result = history.query_history(limit=10)
    assert result["total"] == 1, f"Expected 1 entry, got {result['total']}"
    assert result["entries"][0]["id"] == entry_id
    print(f"   Found {result['total']} entry")

    # Test 3: Get single entry
    print("3. Testing get_entry...")
    entry = history.get_entry(entry_id)
    assert entry is not None, "Entry not found"
    assert entry["problem_type"] == "calculus_derivative"
    print(f"   Retrieved: {entry['problem_type']}")

    # Test 4: Update entry (star it)
    print("4. Testing update_entry...")
    success = history.update_entry(entry_id, {"starred": True})
    assert success, "Update failed"
    entry = history.get_entry(entry_id)
    assert entry["starred"] == True, "Star not updated"
    print("   Starred entry")

    # Test 5: Query starred only
    print("5. Testing starred filter...")
    result = history.query_history(starred_only=True)
    assert result["total"] == 1
    print(f"   Found {result['total']} starred entry")

    # Test 6: Get stats
    print("6. Testing get_stats...")
    stats = history.get_stats()
    assert stats["total_count"] == 1
    assert stats["starred_count"] == 1
    print(f"   Stats: {stats['total_count']} total, {stats['starred_count']} starred")

    # Test 7: Delete entry
    print("7. Testing delete_entry...")
    success = history.delete_entry(entry_id)
    assert success, "Delete failed"
    entry = history.get_entry(entry_id)
    assert entry is None, "Entry still exists after delete"
    print("   Deleted entry")

    print("Homework history module tests PASSED")
    return True


def test_homework_api_endpoints():
    """Test the homework API endpoints."""
    print("\n=== Testing Homework API Endpoints ===")

    # Test 1: Get history (should be empty initially)
    print("1. Testing GET /api/homework/history...")
    resp = requests.get(f"{API_BASE}/homework/history")
    if resp.status_code != 200:
        print(f"   FAILED: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"   Got {data['total']} entries")

    # Test 2: Get stats
    print("2. Testing GET /api/homework/stats...")
    resp = requests.get(f"{API_BASE}/homework/stats")
    if resp.status_code != 200:
        print(f"   FAILED: Status {resp.status_code}")
        return False
    stats = resp.json()
    print(f"   Stats: total={stats['total_count']}, starred={stats['starred_count']}")

    # Test 3: Test solve endpoint with mock image
    print("3. Testing POST /api/homework/solve (mock)...")
    # Create a minimal test image (1x1 pixel PNG)
    # This will likely fail vision detection but tests the endpoint structure
    test_image = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x00, 0x00,
        0x01, 0x01, 0x00, 0x05, 0x1C, 0x50, 0x13, 0x5A,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,
        0xAE, 0x42, 0x60, 0x82
    ])

    files = {"file": ("test.png", test_image, "image/png")}
    resp = requests.post(f"{API_BASE}/homework/solve", files=files)

    # We expect either 200 (success) or 503 (Gemini unavailable)
    if resp.status_code == 503:
        print("   Gemini vision not available (expected in test)")
    elif resp.status_code == 200:
        result = resp.json()
        print(f"   Solved: {result.get('success', 'unknown')}")
    else:
        print(f"   Response: {resp.status_code} - {resp.text[:100]}")

    print("API endpoint tests completed")
    return True


def test_homework_models():
    """Test the Pydantic models."""
    print("\n=== Testing Homework Pydantic Models ===")

    from backend.dashboard.models import (
        HomeworkSolution,
        HomeworkHistoryEntry,
        HomeworkHistoryQuery,
        HomeworkHistoryResponse,
        MathDetectionResult
    )

    # Test HomeworkSolution
    print("1. Testing HomeworkSolution model...")
    problem = MathDetectionResult(
        equation="2x + 3 = 7",
        problem_type="algebra",
        variables=["x"],
        confidence=0.95
    )
    solution = HomeworkSolution(
        problem=problem,
        solution_steps=["Subtract 3 from both sides", "Divide by 2"],
        concept_explanation="Basic linear equation",
        tool_used="problem_strategy"
    )
    assert solution.success == True
    print(f"   Created solution with {len(solution.solution_steps)} steps")

    # Test HomeworkHistoryEntry
    print("2. Testing HomeworkHistoryEntry model...")
    entry = HomeworkHistoryEntry(
        problem_latex="2x + 3 = 7",
        problem_type="algebra",
        solution_summary="x = 2"
    )
    assert entry.id  # Should auto-generate UUID
    assert entry.timestamp > 0  # Should auto-generate timestamp
    print(f"   Created entry {entry.id[:8]}...")

    # Test HomeworkHistoryQuery
    print("3. Testing HomeworkHistoryQuery model...")
    query = HomeworkHistoryQuery(limit=50, starred_only=True)
    assert query.limit == 50
    assert query.starred_only == True
    print(f"   Query: limit={query.limit}, starred_only={query.starred_only}")

    # Test HomeworkHistoryResponse
    print("4. Testing HomeworkHistoryResponse model...")
    response = HomeworkHistoryResponse(entries=[entry], total=1)
    assert len(response.entries) == 1
    print(f"   Response: {response.total} entries")

    print("Pydantic model tests PASSED")
    return True


def run_all_tests():
    """Run all homework tests."""
    print("=" * 60)
    print("Phase 7: Homework Vision Copilot - Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Models
    try:
        results.append(("Models", test_homework_models()))
    except Exception as e:
        print(f"Model tests FAILED: {e}")
        results.append(("Models", False))

    # Test 2: History module
    try:
        results.append(("History Module", test_homework_history_module()))
    except Exception as e:
        print(f"History module tests FAILED: {e}")
        results.append(("History Module", False))

    # Test 3: API endpoints (requires server running)
    try:
        # Check if server is running
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        if resp.status_code == 200:
            results.append(("API Endpoints", test_homework_api_endpoints()))
        else:
            print("\nAPI server not responding properly")
            results.append(("API Endpoints", None))
    except requests.exceptions.ConnectionError:
        print("\nAPI server not running - skipping endpoint tests")
        print("Start server with: python -m uvicorn backend.server:app --reload")
        results.append(("API Endpoints", None))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r for _, r in results if r is not None)
    print("\n" + ("All tests PASSED!" if all_passed else "Some tests FAILED"))
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
