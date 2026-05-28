@echo off
REM 3D Sculptor - Quick Start Script for Windows

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
pip show PyQt6 >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take 10-15 minutes on first run...
    pip install -r requirements.txt
)

REM Run the application
echo Starting 3D Sculptor...
python main.py

REM Deactivate on exit
deactivate
