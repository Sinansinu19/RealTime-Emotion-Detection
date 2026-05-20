@echo off
REM ============================================================================
REM Streamlit Emotion Detector - Automated Setup Script (Windows)
REM ============================================================================

echo.
echo ============================================================================
echo   ^|^| Streamlit Emotion Detector - Setup Script
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found. Creating virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call .\venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

echo [4/4] Installing required packages...
pip install -r requirements.txt

echo.
echo ============================================================================
echo   Setup Complete!
echo ============================================================================
echo.
echo Next steps:
echo   1. Activate virtual environment:  .\venv\Scripts\Activate.ps1
echo   2. Run the application:            streamlit run streamlit_app.py
echo.
echo The app will open at http://localhost:8501
echo.
pause
