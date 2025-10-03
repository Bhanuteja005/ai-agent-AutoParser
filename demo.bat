@echo off
REM Demo script for Agent-as-Coder (Windows)
REM Demonstrates the complete workflow from setup to testing

echo ==========================================
echo Agent-as-Coder Demo
echo ==========================================
echo.

REM Step 1: Setup virtual environment (if needed)
echo Step 1: Setting up Python environment...
if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Virtual environment ready
echo.

REM Step 2: Install dependencies
echo Step 2: Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
echo Dependencies installed
echo.

REM Step 3: Check for API key
echo Step 3: Checking environment...
if "%OPENAI_API_KEY%"=="" (
    echo Warning: OPENAI_API_KEY not set
    echo   Please set it: set OPENAI_API_KEY=your-key
    echo   Or create a .env file (see .env.example)
    echo.
)

REM Step 4: Run agent
echo Step 4: Running agent to generate parser...
echo Target: icici
echo.
python agent.py --target icici --attempts 3

echo.
echo Parser generated
echo.

REM Step 5: Run tests
echo Step 5: Running tests...
pytest tests/test_parsers.py::test_icici_parser -v

echo.
echo ==========================================
echo Demo completed successfully!
echo ==========================================
echo.
echo Generated parser: custom_parsers\icici_parser.py
echo Run 'pytest -v' to test all parsers
