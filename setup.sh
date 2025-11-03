#!/bin/bash

echo "========================================"
echo "Chess Loss Scaling - Setup Script"
echo "========================================"
echo ""

# Parse command line arguments
QUICK_TEST=false
FULL_EVAL=false
RUN_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --full-eval)
            FULL_EVAL=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --help)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick-test    Run quick test after setup (1 game, gpt2 only)"
            echo "  --full-eval     Run full evaluation after setup (all models, 20 games)"
            echo "  --test          Run pytest after setup"
            echo "  --help          Show this help message"
            echo ""
            echo "If no option is provided, setup will complete and wait for user input."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './setup.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Sync dependencies (idempotent - works even if .venv doesn't exist)
print_status "Syncing dependencies with uv..."
uv sync

# Create necessary directories (idempotent)
print_status "Creating project directories..."
mkdir -p data/chess_games
mkdir -p weights
mkdir -p results/models

# Check for Leela Chess Zero
print_status "Checking for Leela Chess Zero..."

# Check if lc0 is in PATH
if command -v lc0 &> /dev/null; then
    print_status "lc0 found in PATH!"
    USE_MOCK=""

    # Download network weights if needed
    WEIGHTS_FILE="$PWD/weights/lc0_weights.pb.gz"
    if [ ! -f "$WEIGHTS_FILE" ]; then
        print_warning "Downloading lc0 neural network weights (~190MB)..."
        mkdir -p weights
        # Using a medium network from storage.lczero.org
        WEIGHTS_URL="https://storage.lczero.org/files/networks-contrib/BT3-768x15x24h-swa-2790000.pb.gz"
        if curl -fL --progress-bar "$WEIGHTS_URL" -o "$WEIGHTS_FILE"; then
            print_status "lc0 weights downloaded successfully"
        else
            print_error "Failed to download weights. Please download manually from:"
            echo "  https://lczero.org/play/networks/bestnets/"
            print_warning "Will use mock Leela Zero for this run"
            USE_MOCK="--mock-leela"
        fi
    else
        print_status "lc0 weights already present"
    fi
else
    # Try to download for macOS only (Linux users should use package manager)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')

    if [ "$OS" = "darwin" ]; then
        print_warning "lc0 not found. Attempting to download for macOS..."
        LC0_DIR="$PWD/weights/lc0"
        LC0_BINARY="$LC0_DIR/lc0"
        mkdir -p "$LC0_DIR"

        # Download macOS binary
        LC0_URL="https://github.com/LeelaChessZero/lc0/releases/download/v0.32.0/lc0-v0.32.0-macos_12.6.1"
        if curl -fSL "$LC0_URL" -o "$LC0_BINARY" 2>/dev/null; then
            chmod +x "$LC0_BINARY"
            export PATH="$LC0_DIR:$PATH"
            print_status "lc0 downloaded for macOS"
            USE_MOCK=""
        else
            print_warning "Failed to download lc0"
            USE_MOCK="--mock-leela"
        fi
    else
        # Linux - recommend package manager installation
        print_warning "lc0 not found."
        echo ""
        echo "For Linux, install lc0 using your package manager:"
        echo "  Ubuntu/Debian: sudo apt install lc0"
        echo "  Arch: sudo pacman -S lc0"
        echo "  Or build from source: https://github.com/LeelaChessZero/lc0"
        echo ""
        print_warning "Will use mock Leela Zero for this run"
        USE_MOCK="--mock-leela"
    fi
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""

# Run based on flags
if [ "$RUN_TESTS" = true ]; then
    print_status "Running tests..."
    uv run pytest tests/ -v
    exit 0
fi

if [ "$QUICK_TEST" = true ]; then
    print_status "Running quick test (1 game, 5 positions)..."
    uv run python -m chess_loss_scaling.main \
        --num-games 1 \
        --max-positions 5 \
        --models gpt2
    exit 0
fi

if [ "$FULL_EVAL" = true ]; then
    print_status "Starting full evaluation (all models, 20 games)..."
    uv run python -m chess_loss_scaling.main \
        --num-games 20 \
        $USE_MOCK
    exit 0
fi

# Interactive mode (no flags provided)
echo "What would you like to do?"
echo ""
echo "1) Run quick test (1 game, just to verify it works)"
echo "2) Run full evaluation (all 5 models, 20 games)"
echo "3) Run pytest"
echo "4) Skip and exit"
echo ""
read -p "Enter choice [1-4]: " -n 1 -r
echo ""

case $REPLY in
    1)
        echo ""
        print_status "Running quick test..."
        uv run python -m chess_loss_scaling.main \
            --num-games 1 \
            --max-positions 5 \
            --models gpt2
        ;;
    2)
        echo ""
        print_status "Starting full evaluation..."
        uv run python -m chess_loss_scaling.main \
            --num-games 20
        ;;
    3)
        echo ""
        print_status "Running tests..."
        uv run pytest tests/ -v
        ;;
    4)
        print_status "Setup complete!"
        ;;
    *)
        print_warning "Invalid choice. Exiting."
        ;;
esac

echo ""
echo "========================================"
echo "Available Commands"
echo "========================================"
echo ""
echo "Quick test:"
echo "  ./setup.sh --quick-test"
echo ""
echo "Full evaluation:"
echo "  ./setup.sh --full-eval"
echo ""
echo "Run tests:"
echo "  ./setup.sh --test"
echo "  OR: uv run pytest tests/ -v"
echo ""
echo "Custom evaluation:"
echo "  uv run python -m chess_loss_scaling.main --num-games 10 --models gpt2 pythia-1b"
echo ""
echo "For more options:"
echo "  uv run python -m chess_loss_scaling.main --help"
echo ""
