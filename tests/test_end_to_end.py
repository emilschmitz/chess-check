"""End-to-end integration tests."""
import json

import chess.pgn
import pytest

from chess_loss_scaling.data.chess_dataset import ChessGameDataset
from chess_loss_scaling.evaluation.aggregation import aggregate_all_results
from chess_loss_scaling.models.leela_loader import MockLeelaZeroModel


@pytest.fixture
def sample_pgn(tmp_path):
    """Create a minimal PGN file for testing."""
    pgn_content = """[Event "Test Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2600"]
[BlackElo "2600"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0

"""
    pgn_file = tmp_path / "test_games.pgn"
    pgn_file.write_text(pgn_content)
    return pgn_file


@pytest.fixture
def mock_leela():
    """Fixture for mock Leela Zero model."""
    return MockLeelaZeroModel()


def test_chess_dataset_loading(sample_pgn):
    """Test that chess dataset loads correctly."""
    dataset = ChessGameDataset(sample_pgn, max_games=1, max_moves_per_game=3)

    # Should be able to iterate
    positions = list(dataset)

    # Should have 3 positions (limited by max_moves_per_game)
    assert len(positions) == 3

    # Each position should have correct structure
    for game_idx, pos_idx, board, move_san, history in positions:
        assert isinstance(game_idx, int)
        assert isinstance(pos_idx, int)
        assert isinstance(board, chess.Board)
        assert isinstance(move_san, str)
        assert isinstance(history, list)


def test_minimal_evaluation_pipeline(sample_pgn, mock_leela, tmp_path):
    """Test minimal evaluation pipeline without loading large models."""
    # Note: This test would need a mock LLM to truly run end-to-end
    # For now, we test that the pipeline structure works

    dataset = ChessGameDataset(sample_pgn, max_games=1, max_moves_per_game=3)

    # Verify dataset is valid
    assert dataset.count_games() == 1

    # Verify mock Leela works
    board = chess.Board()
    probs = mock_leela.get_move_probabilities(board)
    assert len(probs) > 0


def test_results_aggregation(tmp_path):
    """Test that results aggregation works."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create mock result files
    result1 = {
        "model_name": "test-model-1",
        "model_id": "test/model1",
        "num_parameters": "100M",
        "reference_loss": 2.5,
        "chess_avg_loss": 3.5,
        "num_games": 10,
        "num_positions": 100,
    }

    result2 = {
        "model_name": "test-model-2",
        "model_id": "test/model2",
        "num_parameters": "200M",
        "reference_loss": 2.0,
        "chess_avg_loss": 3.0,
        "num_games": 10,
        "num_positions": 100,
    }

    # Save results
    (results_dir / "test-model-1.json").write_text(json.dumps(result1))
    (results_dir / "test-model-2.json").write_text(json.dumps(result2))

    # Aggregate
    csv_path = tmp_path / "aggregate.csv"
    df = aggregate_all_results(results_dir, output_file=csv_path)

    # Check results
    assert len(df) == 2
    assert csv_path.exists()
    assert "model_name" in df.columns
    assert "reference_loss" in df.columns
    assert "chess_avg_loss" in df.columns


def test_dataset_iteration_structure(sample_pgn):
    """Test that dataset iteration provides correct data structure."""
    dataset = ChessGameDataset(sample_pgn)

    for game_idx, pos_idx, board, move_played, history in dataset:
        # Verify types
        assert isinstance(game_idx, int)
        assert isinstance(pos_idx, int)
        assert isinstance(board, chess.Board)
        assert isinstance(move_played, str)
        assert isinstance(history, list)

        # Verify position index matches history length
        assert pos_idx == len(history)

        # Verify board is in a valid state
        assert board.is_valid()

        # Only check first game
        if game_idx > 0:
            break


@pytest.mark.slow
def test_evaluate_single_model_mock(sample_pgn, mock_leela, tmp_path):
    """
    Test evaluating a single model (would require real LLM).

    This test is marked as slow and would need to be run with actual models.
    """
    pytest.skip("Requires real LLM model - expensive to run in CI")

    # This is what a full test would look like:
    # dataset = ChessGameDataset(sample_pgn, max_games=1, max_moves_per_game=5)
    # model_config = MODELS[0]  # GPT-2
    # result = evaluate_model_on_dataset(
    #     model_config, dataset, mock_leela, tmp_path
    # )
    # assert result["num_games"] > 0
