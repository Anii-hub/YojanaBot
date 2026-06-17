@echo off
REM ── YojanaBot — Quick Start Script ──────────────────────────────────────
REM Run this from the project root: double-click or `start.bat` in cmd

echo.
echo  ====================================================
echo   🇮🇳  YojanaBot — Government Scheme Eligibility Finder
echo  ====================================================
echo.

REM Check .env exists
if not exist .env (
    echo  [!] .env file not found!
    echo      Copy .env.example to .env and add your GROQ_API_KEY
    echo      Get a free key at https://console.groq.com
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Set encoding
set PYTHONIOENCODING=utf-8

echo  [*] Starting YojanaBot on http://127.0.0.1:8000
echo  [*] Press Ctrl+C to stop
echo.

python manage.py runserver 8000
