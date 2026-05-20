#!/bin/bash

# ============================================================================
# Streamlit Emotion Detector - Automated Setup Script (macOS/Linux)
# ============================================================================

echo ""
echo "============================================================================"
echo "  || Streamlit Emotion Detector - Setup Script"
echo "============================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Python found: $(python3 --version)"
echo "[2/4] Creating virtual environment..."
python3 -m venv venv

echo "[3/4] Activating virtual environment..."
source venv/bin/activate

echo "[4/4] Installing required packages..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

echo ""
echo "============================================================================"
echo "  Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:  source venv/bin/activate"
echo "  2. Run the application:            streamlit run streamlit_app.py"
echo ""
echo "The app will open at http://localhost:8501"
echo ""
