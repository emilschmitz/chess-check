"""Tests for Leela Chess Zero loader."""
import chess

from chess_loss_scaling.models.leela_loader import MockLeelaZeroModel, load_leela_model


def test_mock_leela_initialization():
    """Test mock Leela model initializes correctly."""
    leela = MockLeelaZeroModel()
    assert leela is not None


def test_mock_leela_move_probabilities():
    """Test mock Leela returns valid probability distribution."""
    leela = MockLeelaZeroModel()
    board = chess.Board()

    probs = leela.get_move_probabilities(board)

    # Should return probabilities for all legal moves
    assert len(probs) == 20  # Starting position

    # Probabilities should sum to approximately 1.0
    total_prob = sum(probs.values())
    assert 0.99 < total_prob < 1.01

    # All probabilities should be non-negative
    assert all(p >= 0 for p in probs.values())

    # All moves should be in UCI format
    assert all(len(move) >= 4 for move in probs.keys())


def test_mock_leela_empty_board():
    """Test mock Leela handles positions with no legal moves."""
    leela = MockLeelaZeroModel()

    # Create a checkmated position
    board = chess.Board("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1")  # Black is checkmated

    probs = leela.get_move_probabilities(board)

    # Should return empty dict for no legal moves
    assert len(probs) == 0


def test_load_leela_with_mock():
    """Test loading Leela with mock flag."""
    leela = load_leela_model(mock=True)

    assert isinstance(leela, MockLeelaZeroModel)

    # Should be usable
    board = chess.Board()
    probs = leela.get_move_probabilities(board)
    assert len(probs) > 0


def test_mock_leela_close():
    """Test mock Leela close method doesn't crash."""
    leela = MockLeelaZeroModel()
    leela.close()  # Should not raise


def test_mock_leela_different_positions():
    """Test mock Leela returns different distributions for different positions."""
    leela = MockLeelaZeroModel()

    board1 = chess.Board()
    probs1 = leela.get_move_probabilities(board1)

    board2 = chess.Board()
    board2.push_san("e4")
    probs2 = leela.get_move_probabilities(board2)

    # Different positions, different number of moves
    assert len(probs1) != len(probs2)
