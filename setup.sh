#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-command environment setup for Linux / macOS
# =============================================================================
set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   CineAI — Distributed Movie Recommendation Engine       ║"
echo "║   Environment Setup (Linux / macOS)                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi
echo "[OK] Python 3 found: $(python3 --version)"

# Check Java
if ! command -v java &> /dev/null; then
    echo "[WARNING] java could not be found. PySpark requires Java 8, 11, or 17."
    echo "Install Java using your package manager (e.g. sudo apt install openjdk-11-jdk or brew install openjdk@11)."
else
    echo "[OK] Java found: $(java -version 2>&1 | head -n 1)"
fi

# Create venv if not present
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment ..."
    python3 -m venv venv
fi
echo "[OK] Virtual environment ready."

# Activate and install dependencies
echo "[SETUP] Installing Python dependencies ..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo "[SUCCESS] Setup complete!"
echo "To activate environment: source venv/bin/activate"
echo "To run API: uvicorn api.main:app --reload --port 8000"
echo "To run UI:  streamlit run ui/app.py"
echo ""
