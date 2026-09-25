@echo off
REM ══════════════════════════════════════════════════════════════════════════
REM  setup.bat — One-click environment setup for Recommendation Engine (BDA)
REM  Windows PowerShell / Command Prompt
REM ══════════════════════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════════════════════════╗
echo  ║   CineAI — Distributed Movie Recommendation Engine       ║
echo  ║   BDA Project Setup                                       ║
echo  ╚═══════════════════════════════════════════════════════════╝
echo.

REM ── Step 1: Check Python ──────────────────────────────────────────────────
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found.

REM ── Step 2: Check Java (required for PySpark) ─────────────────────────────
java -version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Java not found!
    echo PySpark requires Java 8 or 11. Download from:
    echo   https://adoptium.net/  (Eclipse Temurin - recommended)
    echo.
    echo After installing Java, re-run this script.
    pause
    exit /b 1
)
echo [OK] Java found.

REM ── Step 3: Create virtual environment ───────────────────────────────────
IF NOT EXIST "venv" (
    echo.
    echo [SETUP] Creating virtual environment ...
    py -3.10 -m venv venv 2>nul || python -m venv venv
)
echo [OK] Virtual environment ready.

REM ── Step 4: Activate and install dependencies ─────────────────────────────
echo.
echo [SETUP] Installing Python dependencies ...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo [OK] Dependencies installed.

REM ── Step 5: Set JAVA_HOME hint ────────────────────────────────────────────
echo.
echo [INFO] If PySpark fails with Java errors, set JAVA_HOME manually:
echo        set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.x.x
echo.

echo.
echo  ══════════════════════════════════════════════════════════
echo   Setup Complete! Run the pipeline in order:
echo  ══════════════════════════════════════════════════════════
echo.
echo   1. Download dataset:
echo      python data\download_data.py --size 1m
echo.
echo   2. Preprocess with PySpark:
echo      python spark\preprocess.py
echo.
echo   3. Train ALS model (quick demo):
echo      python spark\train_als.py --fast
echo.
echo   4. Evaluate model:
echo      python spark\evaluate.py
echo.
echo   5. Generate batch recommendations:
echo      python spark\generate_recs.py
echo.
echo   6. Start FastAPI backend (new terminal):
echo      uvicorn api.main:app --reload --port 8000
echo.
echo   7. Launch Streamlit UI (new terminal):
echo      streamlit run ui\app.py
echo.
echo   Open browser: http://localhost:8501
echo.
pause
