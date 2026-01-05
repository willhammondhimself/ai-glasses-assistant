@echo off
REM WHAM Desktop App Startup Script
REM Starts backend server and desktop capture app

echo ==================================================
echo    WHAM Desktop - Starting
echo ==================================================
echo.

REM Change to project root
cd /d "%~dp0\.."

REM Check if backend is running (simple check)
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel%==0 (
    echo Backend already running on port 8000
) else (
    echo Starting backend server...
    start /B uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
    echo Backend starting...

    REM Wait for backend
    echo Waiting for backend...
    timeout /t 5 /nobreak >nul
)

echo.
echo Starting desktop app...
python desktop_capture\main.py

pause
