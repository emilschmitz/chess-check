# Chess Loss Scaling

Testing Epoch AI's hypothesis: Does training loss correlate with chess prediction performance?

## Overview

This project evaluates whether general language model training loss correlates with domain-specific performance on chess move prediction. We compare LLM predictions against Leela Chess Zero's superhuman move probability distributions to test the robustness of using training loss as a proxy for AI capabilities.

**Research Question**: Does lower general training loss in LLMs correlate with lower cross-entropy loss on chess move prediction?

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install dependencies with uv
- Download chess games
- Run evaluation (with mock Leela by default)
- Generate results CSV

### Option 2: Manual Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run evaluation
python -m src.main --num-games 5 --mock-leela --max-positions 10
```

## Usage

### Basic Evaluation

Evaluate all models on 10 games:

```bash
python -m src.main --num-games 10 --mock-leela
```

### Evaluate Specific Models

```bash
python -m src.main --models gpt2 pythia-1b --num-games 5 --mock-leela
```

### With Real Leela Zero

If you have lc0 installed:

```bash
python -m src.main --num-games 10
```

### Advanced Options

```bash
python -m src.main \
  --num-games 20 \
  --models pythia-1b pythia-1.4b \
  --max-positions 15 \
  --device cuda \
  --log-level DEBUG
```

### Available Arguments

- `--num-games N`: Number of games to evaluate (default: 10)
- `--models [MODEL ...]`: Specific models to evaluate (default: all)
  - Choices: gpt2, gpt-neo-1.3B, pythia-1b, pythia-1.4b, pythia-2.8b
- `--device {cuda,cpu}`: Device to use (default: auto-detect)
- `--mock-leela`: Use mock Leela Zero (for testing without lc0)
- `--max-positions N`: Max positions per game (default: all)
- `--min-elo N`: Minimum Elo rating for games (default: 2500)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging verbosity (default: INFO)

## Project Structure

```
chess-loss-scaling/
├── src/
│   ├── main.py              # Main execution script
│   ├── models/
│   │   ├── model_config.py  # Model configurations and training losses
│   │   ├── leela_loader.py  # Leela Chess Zero integration
│   │   └── llm_loader.py    # HuggingFace LLM loader
│   ├── data/
│   │   ├── chess_dataset.py # Chess game dataset loader
│   │   └── converters.py    # Format conversion utilities
│   ├── evaluation/
│   │   ├── chess_loss.py    # Loss calculation functions
│   │   └── aggregation.py   # Result aggregation
│   └── utils/
│       ├── logging_config.py
│       └── progress.py
├── tests/                   # Unit and integration tests
├── results/
│   ├── models/             # Per-model JSON results
│   └── aggregate_results.csv
├── writeup.md              # Research paper
└── README.md
```

## Models Evaluated

1. **GPT-2** (124M) - Baseline small model
2. **GPT-Neo-1.3B** - Mid-size open model
3. **Pythia-1B** - Documented training trajectory
4. **Pythia-1.4B** - Larger Pythia variant
5. **Pythia-2.8B** - Largest practical variant

Training losses sourced from original papers (GPT-2, Pythia).

## Requirements

### Python Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- python-chess ≥ 1.999
- pandas, numpy, rich

See `pyproject.toml` for full list.

### Optional: Leela Chess Zero

For real ground truth (not required for testing with mock):

1. Download lc0: https://github.com/LeelaChessZero/lc0/releases
2. Download weights: https://training.lczero.org/
3. Place binary in PATH or specify path

Without lc0, the system uses a mock that generates realistic probability distributions for testing.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_converters.py

# Run with coverage
pytest --cov=src tests/

# Run only fast tests (skip expensive model tests)
pytest -m "not slow"
```

## Expected Output

After running, you'll get:

1. **Per-model JSON files** in `results/models/`:
   - Detailed per-game and per-position losses
   - Model metadata and configuration

2. **Aggregate CSV** at `results/aggregate_results.csv`:
   - Model name, parameters, training loss
   - Chess average loss, std, min, max
   - Number of games and positions evaluated

3. **Console summary**:
   - Table of all models and their losses
   - Correlation coefficient between training loss and chess loss

Example output:

```
================================================================================
CHESS LOSS SCALING RESULTS
================================================================================

model_name      num_parameters  reference_loss  chess_avg_loss  num_games
gpt2            124M            3.31            4.52            10
pythia-1b       1.0B            2.74            3.89            10
pythia-1.4b     1.4B            2.64            3.71            10
pythia-2.8b     2.8B            2.47            3.45            10

Correlation (reference_loss vs chess_avg_loss): 0.987

================================================================================
```

## Memory Requirements

- **GPT-2 (124M)**: ~0.5 GB VRAM
- **Pythia-1B**: ~4 GB VRAM
- **Pythia-2.8B**: ~8 GB VRAM (with 8-bit quantization)

Models are loaded and unloaded sequentially to minimize memory usage.

## Troubleshooting

### Out of Memory

```bash
# Use fewer games
python -m src.main --num-games 5

# Limit positions per game
python -m src.main --max-positions 10

# Use smaller models
python -m src.main --models gpt2 pythia-1b
```

### Slow Evaluation

- Reduce `--num-games`
- Reduce `--max-positions`
- Use `--device cpu` if GPU is slow
- Evaluate fewer models

### Import Errors

```bash
# Ensure you're in virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv pip install -e .
```

## Citation

If you use this code, please cite:

```
@misc{chess-loss-scaling-2024,
  title={Chess Loss Scaling: Testing Training Loss as AI Capability Proxy},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/chess-loss-scaling}
}
```

And the original Epoch AI paper:

```
@article{epoch2024direct,
  title={Direct Approach to AI Forecasting},
  author={Epoch AI},
  year={2024},
  url={https://epoch.ai/files/direct-approach.pdf}
}
```

## Contributing

This is a research project. Suggestions for improvement:

1. Real Leela Zero integration (replace mock)
2. Download actual grandmaster games (replace sample PGN)
3. Expand to more models (Llama 2 7B, etc.)
4. Implement constrained generation for valid moves
5. Add visualization scripts (loss correlation plots)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Epoch AI for the research hypothesis
- Leela Chess Zero project for superhuman chess engine
- EleutherAI for open-source models (Pythia, GPT-Neo)
- HuggingFace for model hosting and transformers library
