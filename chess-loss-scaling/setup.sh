#!/bin/bash
set -e

echo "========================================"
echo "Chess Loss Scaling - Setup Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_warning "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the shell config to get uv in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually from https://astral.sh/uv/"
        exit 1
    fi
    print_status "uv installed successfully"
else
    print_status "uv already installed"
fi

# Create and activate virtual environment
print_status "Creating virtual environment..."
uv venv

print_status "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    print_error "Could not find virtual environment activation script"
    exit 1
fi

# Install dependencies
print_status "Installing dependencies..."
uv pip install -e .

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/chess_games
mkdir -p weights
mkdir -p results/models

# Download chess games (handled by the script)
print_status "Chess games will be downloaded on first run"

# Check for lc0
print_warning "Checking for Leela Chess Zero (lc0)..."
if command -v lc0 &> /dev/null; then
    print_status "lc0 found! Real Leela Zero will be used."
    USE_MOCK=""
else
    print_warning "lc0 not found. Will use mock Leela Zero."
    print_warning "For real evaluation, install from: https://github.com/LeelaChessZero/lc0/releases"
    USE_MOCK="--mock-leela"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""

# Ask user if they want to run evaluation
read -p "Run evaluation now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_status "Starting evaluation with 5 games, max 10 positions per game..."
    print_status "This is a quick test run. For full evaluation, use more games."
    echo ""

    # Run with limited games for quick test
    python -m src.main \
        --num-games 5 \
        --max-positions 10 \
        $USE_MOCK \
        --models gpt2 pythia-1b

    echo ""
    print_status "Evaluation complete!"
    print_status "Results saved to: results/aggregate_results.csv"
    echo ""
    echo "To run full evaluation on all models:"
    echo "  python -m src.main --num-games 20 $USE_MOCK"
    echo ""
    echo "To evaluate specific models:"
    echo "  python -m src.main --models gpt2 pythia-1b --num-games 10 $USE_MOCK"
    echo ""
else
    echo ""
    print_status "Setup complete! You can now run evaluation:"
    echo ""
    echo "  # Quick test (2 models, 5 games)"
    echo "  python -m src.main --num-games 5 --models gpt2 pythia-1b $USE_MOCK"
    echo ""
    echo "  # Full evaluation (all models, 20 games)"
    echo "  python -m src.main --num-games 20 $USE_MOCK"
    echo ""
    echo "  # Run tests"
    echo "  pytest"
    echo ""
fi

echo "For more options, run:"
echo "  python -m src.main --help"
echo ""
