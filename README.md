# Chess Loss Scaling

Testing Epoch AI's hypothesis from their [direct approach](https://epoch.ai/files/direct-approach.pdf): Does training loss correlate with chess prediction performance?

## Overview

This project evaluates whether general language model training loss correlates with domain-specific performance on chess move prediction. We compare LLM predictions against Leela Chess Zero's superhuman move probability distributions to test the robustness of using training loss as a proxy for AI capabilities.

**Research Question**: Does lower general training loss in LLMs correlate with lower cross-entropy loss on chess move prediction?

## Quick Start

### 1. Install Leela Chess Zero dependencies

**Ubuntu/Debian:**

```bash
sudo apt install git cmake ninja-build pkg-config g++ libopenblas-dev
```

**Other platforms:** See https://github.com/LeelaChessZero/lc0/blob/master/README.md#building-and-running-lc0

### 2. Run automated setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Install Python dependencies with uv
- Clone and build Leela Chess Zero v0.32.0 into `external/lc0`
- Download neural network weights
- Download chess games
- Run evaluation

**Note:** For testing without lc0 (not recommended for real evaluation), use `--mock-leela` flag

## Usage

### Basic Evaluation

Evaluate all models on 10 games:

```bash
python -m chess_loss_scaling.main --num-games 10
```

### Evaluate Specific Models

```bash
python -m chess_loss_scaling.main --models gpt2 pythia-1b --num-games 5
```

### Testing Without lc0

For development/testing only (not for real evaluation):

```bash
python -m chess_loss_scaling.main --num-games 10 --mock-leela
```

### Advanced Options

```bash
python -m chess_loss_scaling.main \
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
- `--max-positions N`: Max positions per game (default: all)
- `--min-elo N`: Minimum Elo rating for games (default: 2500)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging verbosity (default: INFO)
- `--mock-leela`: Use mock Leela Zero (testing only, not for real evaluation)

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

## Citation

If you use this code, please cite:

```
@misc{chess-loss-scaling-2024,
  title={Chess Loss Scaling: Testing Next-Word Prediction Loss as a General Capability Proxy},
  author={Your Name},
  year={2024},
  url={https://github.com/emilschmitz/chess-check}
}
```

## Ideas for Improvement:

1. Analyze chess loss specifically on models similar to those from the Chinchilla paper (Hoffman 2022). The weights from that paper were never released though. See also [Epoch AI&#39;s replication](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) (2024). For those models we'd have measured next-word loss numbers. We can use OLMO models for example. Another alternative would be Pythia models.
   1. Basically, we need models where we have relevant loss numbers that we can compare. Perhaps, it'd also be nice to have number of train tokens and parameters. Then we could use the Chinchilla formula to also estimate loss. I am unsure if this adds any value. Perhaps it is also useful if the models are all the same except for number of train tokens and number of parameters.
   2. olmo loss number are available on links from the header in [olmo 2 paper](https://arxiv.org/pdf/2501.00656)
2. For the OLMO models: check to what extent it is legitimate to use train loss as an approximation for test loss. To do this, we will have to check if they were trained on more than one epoch. We can also check if there is a holdout validation set we can use to get these numbers.
3. Use self-play games from Leela Zero to test on instead of actual games that might have been part of training data for lc0 (only trained on self-play?) or the LLMs
4. Expand to more models and predict loss with Chinchilla scaling law
5. Implement constrained generation for valid moves
6. Add visualization scripts (loss correlation plots)

## License

MIT License - See LICENSE file for details.
