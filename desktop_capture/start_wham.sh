#!/bin/bash
# WHAM Desktop App Startup Script
# Starts backend server and desktop capture app

set -e

# Change to project root
cd "$(dirname "$0")/.."

echo "=================================================="
echo "   WHAM Desktop - Starting"
echo "=================================================="
echo ""

# Check if backend is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Backend already running on port 8000"
else
    echo "Starting backend server..."
    uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "✓ Backend started (PID: $BACKEND_PID)"

    # Wait for backend to be ready
    echo "Waiting for backend..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Backend ready"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "Starting desktop app..."
python desktop_capture/main.py

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
