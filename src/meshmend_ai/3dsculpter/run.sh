#!/bin/bash
# 3D Sculptor - Quick Start Script for Linux/Mac

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.10+ from python.org or your package manager"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! pip show PyQt6 &> /dev/null; then
    echo "Installing dependencies..."
    echo "This may take 10-15 minutes on first run..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting 3D Sculptor..."
python main.py

# Deactivate on exit
deactivate
