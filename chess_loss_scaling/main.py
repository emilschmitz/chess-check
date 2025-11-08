"""Main execution script for chess loss scaling evaluation."""
import argparse
from pathlib import Path

from chess_loss_scaling.data.chess_dataset import ChessGameDataset, download_chess_games
from chess_loss_scaling.evaluation.aggregation import (
    aggregate_all_results,
    evaluate_model_on_dataset,
    print_results_summary,
)
from chess_loss_scaling.models.leela_loader import load_leela_model
from chess_loss_scaling.models.model_config import MODELS, list_model_names
from chess_loss_scaling.utils.logging_config import setup_logging

logger = setup_logging()


def main(
    num_games: int = 10,
    models: list[str] = None,
    device: str = None,
    max_positions_per_game: int = None,
    min_elo: int = 2500,
):
    """
    Main execution pipeline.

    Args:
        num_games: Number of chess games to evaluate
        models: List of model names to evaluate (None = all)
        device: Device to use ("cuda", "cpu", or None for auto)
        max_positions_per_game: Max positions per game (None = all)
        min_elo: Minimum Elo rating for games
    """
    logger.info("Starting chess loss scaling evaluation")
    logger.info(f"Configuration: {num_games} games, device={device or 'auto'}")

    # Setup paths
    data_dir = Path("data/chess_games")
    results_dir = Path("results/models")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download chess games if needed
    logger.info("Step 1: Preparing chess games dataset")
    pgn_path = download_chess_games(
        output_dir=str(data_dir),
        num_games=num_games,
        min_elo=min_elo
    )

    # Step 2: Load chess dataset
    logger.info("Step 2: Loading chess dataset")
    dataset = ChessGameDataset(
        pgn_path=pgn_path,
        max_games=num_games,
        max_moves_per_game=max_positions_per_game
    )

    num_games_loaded = dataset.count_games()
    logger.info(f"Loaded {num_games_loaded} games")

    # Step 3: Load Leela Zero
    logger.info("Step 3: Loading Leela Chess Zero")
    leela = load_leela_model()

    # Step 4: Select models to evaluate
    if models is None:
        models_to_eval = MODELS
    else:
        models_to_eval = [m for m in MODELS if m["name"] in models]

    logger.info(f"Step 4: Evaluating {len(models_to_eval)} models")

    # Step 5: Evaluate each model
    all_results = []
    for i, model_config in enumerate(models_to_eval, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Model {i}/{len(models_to_eval)}: {model_config['name']}")
        logger.info(f"{'='*60}")

        try:
            result = evaluate_model_on_dataset(
                model_config=model_config,
                dataset=dataset,
                leela=leela,
                output_dir=results_dir,
                max_positions_per_game=max_positions_per_game,
            )
            all_results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate {model_config['name']}: {e}")
            continue

    # Step 6: Aggregate results
    logger.info("\nStep 6: Aggregating results")
    aggregate_csv = Path("results/aggregate_results.csv")
    df = aggregate_all_results(results_dir, output_file=aggregate_csv)

    # Step 7: Print summary
    logger.info("\nStep 7: Results summary")
    print_results_summary(df)

    logger.info(f"\nEvaluation complete! Results saved to {aggregate_csv}")

    # Cleanup
    leela.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM chess prediction performance vs training loss"
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of chess games to evaluate (default: 10)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list_model_names(),
        help="Models to evaluate (default: all)"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        help="Max positions per game to evaluate (default: all)"
    )

    parser.add_argument(
        "--min-elo",
        type=int,
        default=2500,
        help="Minimum Elo rating for games (default: 2500)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging with specified level
    setup_logging(level=args.log_level)

    main(
        num_games=args.num_games,
        models=args.models,
        device=args.device,
        max_positions_per_game=args.max_positions,
        min_elo=args.min_elo,
    )
