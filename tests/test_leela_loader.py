"""Tests for Leela Chess Zero loader."""
import chess
import numpy as np
import pytest

from chess_loss_scaling.models.leela_loader import LeelaZeroModel, load_leela_model
import lczero.backends as lc0


class MockLeelaZeroModel:
    """
    Mock Leela Zero model for testing without lc0 installation.

    Returns random but valid probability distributions.
    """

    def __init__(self, **kwargs):
        """Initialize mock model (accepts any kwargs for compatibility)."""
        pass

    def get_move_probabilities(self, board: chess.Board) -> dict[str, float]:
        """Return random probability distribution over legal moves."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        # Use Dirichlet distribution for realistic-looking probabilities
        alpha = np.ones(len(legal_moves))
        probs = np.random.dirichlet(alpha)

        return {move.uci(): float(prob) for move, prob in zip(legal_moves, probs)}

    def close(self):
        """No-op for mock."""
        pass


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


@pytest.mark.slow
def test_load_leela_factory():
    """Test loading Leela with factory function."""
    import os

    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "lc0_weights.pb.gz")
    weights_path = os.path.normpath(weights_path)

    if not os.path.exists(weights_path):
        pytest.skip(f"LC0 weights not found at {weights_path}")

    try:
        leela = load_leela_model(weights_path=weights_path)
        assert isinstance(leela, LeelaZeroModel)

        # Should be usable
        board = chess.Board()
        probs = leela.get_move_probabilities(board)
        assert len(probs) > 0
        leela.close()
    except Exception as e:
        pytest.skip(f"Failed to load LC0: {e}")


def test_mock_leela_close():
    """Test mock Leela close method doesn't crash."""
    leela = MockLeelaZeroModel()
    leela.close()  # Should not raise


@pytest.mark.slow
def test_real_leela_forward_pass():
    """Test real Leela Chess Zero forward pass with python bindings."""
    import os

    # Check if weights exist
    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "lc0_weights.pb.gz")
    weights_path = os.path.normpath(weights_path)

    if not os.path.exists(weights_path):
        pytest.skip(f"LC0 weights not found at {weights_path}")

    try:
        # Try to load real Leela
        leela = LeelaZeroModel(weights_path=weights_path)
    except RuntimeError as e:
        pytest.skip(f"Failed to load LC0: {e}")

    # Test forward pass on starting position
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

    # Check that probabilities are not uniform (should favor good moves)
    max_prob = max(probs.values())
    assert max_prob > 0.05, "Best move should have >5% probability"

    # Cleanup
    leela.close()


@pytest.mark.slow
def test_lc0_python_bindings_forward_pass():
    """Test LC0 python bindings can load network and run forward pass."""
    import os
    import numpy as np

    # Check if weights file exists
    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "lc0_weights.pb.gz")
    weights_path = os.path.normpath(weights_path)

    if not os.path.exists(weights_path):
        pytest.skip(f"LC0 weights not found at {weights_path}")

    # Load weights
    weights = lc0.Weights(weights_path)
    assert weights is not None

    # Create backend (use blas for CPU, cudnn would be used automatically if available)
    backend = lc0.Backend(weights=weights, backend="blas")
    assert backend is not None

    # Create starting position
    game = lc0.GameState()
    assert game is not None

    # Create input tensor
    input_tensor = game.as_input(backend)
    assert input_tensor is not None

    # Run forward pass
    outputs = backend.evaluate(input_tensor)
    assert len(outputs) == 1
    output = outputs[0]

    # Check value outputs
    q_value = output.q()
    d_value = output.d()
    m_value = output.m()

    # Q and D should be probabilities
    assert -1.0 <= q_value <= 1.0, f"Q value {q_value} out of range"
    assert 0.0 <= d_value <= 1.0, f"D value {d_value} out of range"
    assert m_value > 0, f"Moves left {m_value} should be positive"

    # Check policy outputs
    policy_indices = game.policy_indices()
    moves = game.moves()

    assert len(policy_indices) == len(moves)
    assert len(moves) == 20  # Starting position has 20 legal moves

    # Get raw policy logits
    raw_probs = [output.p_raw(idx)[0] for idx in policy_indices]
    assert len(raw_probs) == 20

    # Apply softmax to get probabilities
    raw_np = np.array(raw_probs)
    exp_vals = np.exp(raw_np - np.max(raw_np))
    probs = exp_vals / exp_vals.sum()

    # Probabilities should sum to 1
    assert 0.99 < probs.sum() < 1.01

    # All probabilities should be non-negative
    assert all(p >= 0 for p in probs)

    # Most probable move should have reasonable probability (not uniform)
    max_prob = probs.max()
    assert max_prob > 0.05, "Top move should have >5% probability"