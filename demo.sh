#!/bin/bash
# Demo script for Agent-as-Coder
# Demonstrates the complete workflow from setup to testing

set -e  # Exit on error

echo "=========================================="
echo "Agent-as-Coder Demo"
echo "=========================================="
echo ""

# Step 1: Setup virtual environment (if needed)
echo "Step 1: Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "✓ Virtual environment ready"
echo ""

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 3: Check for API key
echo "Step 3: Checking environment..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ Warning: OPENAI_API_KEY not set"
    echo "  Please set it: export OPENAI_API_KEY='your-key'"
    echo "  Or create a .env file (see .env.example)"
    echo ""
fi

# Step 4: Run agent
echo "Step 4: Running agent to generate parser..."
echo "Target: icici"
echo ""
python agent.py --target icici --attempts 3

echo ""
echo "✓ Parser generated"
echo ""

# Step 5: Run tests
echo "Step 5: Running tests..."
pytest tests/test_parsers.py::test_icici_parser -v

echo ""
echo "=========================================="
echo "Demo completed successfully!"
echo "=========================================="
echo ""
echo "Generated parser: custom_parsers/icici_parser.py"
echo "Run 'pytest -v' to test all parsers"
