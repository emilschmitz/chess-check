"""Aggregate evaluation results across games and models."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import chess.pgn
import numpy as np
import pandas as pd

from chess_loss_scaling.data.chess_dataset import ChessGameDataset
from chess_loss_scaling.evaluation.chess_loss import calculate_game_loss
from chess_loss_scaling.models.leela_loader import LeelaZeroModel
from chess_loss_scaling.models.model_config import get_reference_loss
from chess_loss_scaling.utils.logging_config import get_logger
from chess_loss_scaling.utils.progress import ProgressTracker

logger = get_logger()


def evaluate_model_on_dataset(
    model_config: dict,
    dataset: ChessGameDataset,
    leela: LeelaZeroModel,
    output_dir: Path,
    max_positions_per_game: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate one LLM on entire chess dataset.

    Args:
        model_config: Model configuration dict
        dataset: Chess game dataset
        leela: Leela Zero model for ground truth
        output_dir: Directory to save results
        max_positions_per_game: Max positions per game (None = all)

    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating model: {model_config['name']}")

    # Load LLM
    from chess_loss_scaling.models.llm_loader import load_chess_llm
    llm = load_chess_llm(
        model_id=model_config["hf_id"],
        device="auto",
        load_in_8bit=True if "2.8b" in model_config["name"].lower() else False
    )

    # Evaluate on all games
    game_results = []

    with ProgressTracker() as progress:
        # Count games first
        num_games = dataset.count_games()
        game_task = progress.add_task(
            f"Evaluating {model_config['name']}", total=num_games
        )

        # Open PGN file
        with open(dataset.pgn_path) as pgn_file:
            game_index = 0
            while True:
                if dataset.max_games and game_index >= dataset.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Calculate loss for this game
                try:
                    game_loss = calculate_game_loss(
                        llm, leela, game,
                        max_positions=max_positions_per_game
                    )

                    game_results.append({
                        "game_index": game_index,
                        **game_loss
                    })

                    progress.update(game_task)

                except Exception as e:
                    logger.error(f"Error evaluating game {game_index}: {e}")
                    game_results.append({
                        "game_index": game_index,
                        "error": str(e)
                    })

                game_index += 1

    # Aggregate results
    valid_games = [g for g in game_results if "error" not in g]

    if not valid_games:
        result = {
            "model_name": model_config["name"],
            "model_id": model_config["hf_id"],
            "num_parameters": model_config["params"],
            "general_training_loss": model_config.get("training_loss"),
            "general_eval_loss": model_config.get("eval_loss"),
            "reference_loss": get_reference_loss(model_config),
            "chess_avg_loss": float('inf'),
            "chess_std_loss": 0.0,
            "num_games": 0,
            "num_positions": 0,
            "error": "No valid games evaluated"
        }
    else:
        ce_losses = [g["avg_cross_entropy"] for g in valid_games]
        total_positions = sum(g["num_positions"] for g in valid_games)

        result = {
            "model_name": model_config["name"],
            "model_id": model_config["hf_id"],
            "num_parameters": model_config["params"],
            "general_training_loss": model_config.get("training_loss"),
            "general_eval_loss": model_config.get("eval_loss"),
            "reference_loss": get_reference_loss(model_config),
            "chess_avg_loss": float(np.mean(ce_losses)),
            "chess_std_loss": float(np.std(ce_losses)),
            "chess_median_loss": float(np.median(ce_losses)),
            "chess_min_loss": float(np.min(ce_losses)),
            "chess_max_loss": float(np.max(ce_losses)),
            "num_games": len(valid_games),
            "num_positions": total_positions,
            "per_game_results": game_results,
            "timestamp": datetime.now().isoformat(),
        }

    # Save to JSON
    output_path = output_dir / f"{model_config['name']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved results to {output_path}")

    # Unload model
    llm.unload()

    return result


def aggregate_all_results(results_dir: Path, output_file: Path | None = None) -> pd.DataFrame:
    """
    Load all model JSONs and create aggregate CSV.

    Args:
        results_dir: Directory containing model JSON files
        output_file: Where to save CSV (None = don't save)

    Returns:
        DataFrame with all results
    """
    results_dir = Path(results_dir)

    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {results_dir}")
        return pd.DataFrame()

    # Load all results
    all_results = []
    for json_file in json_files:
        with open(json_file) as f:
            result = json.load(f)
            # Remove per_game_results for summary
            result_summary = {k: v for k, v in result.items() if k != "per_game_results"}
            all_results.append(result_summary)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by reference loss
    if "reference_loss" in df.columns:
        df = df.sort_values("reference_loss")

    # Save to CSV
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved aggregate results to {output_file}")

    return df


def print_results_summary(df: pd.DataFrame):
    """Print a nice summary of results."""
    if df.empty:
        logger.warning("No results to summarize")
        return

    print("\n" + "="*80)
    print("CHESS LOSS SCALING RESULTS")
    print("="*80)
    print()

    # Print table
    columns = [
        "model_name",
        "num_parameters",
        "reference_loss",
        "chess_avg_loss",
        "num_games",
        "num_positions"
    ]

    available_columns = [col for col in columns if col in df.columns]
    print(df[available_columns].to_string(index=False))

    # Calculate correlation
    if "reference_loss" in df.columns and "chess_avg_loss" in df.columns:
        # Filter out inf values
        valid_df = df[
            (df["reference_loss"].notna()) &
            (df["chess_avg_loss"].notna()) &
            (df["chess_avg_loss"] != float('inf'))
        ]

        if len(valid_df) >= 2:
            correlation = valid_df["reference_loss"].corr(valid_df["chess_avg_loss"])
            print()
            print(f"Correlation (reference_loss vs chess_avg_loss): {correlation:.3f}")
        else:
            print()
            print("Not enough valid data points to calculate correlation")

    print()
    print("="*80)
