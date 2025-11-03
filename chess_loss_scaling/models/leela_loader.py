"""Leela Chess Zero integration using python bindings."""
from typing import Optional

import chess
import numpy as np
import lczero.backends as lc0


class LeelaZeroModel:
    """
    Load and run Leela Chess Zero using python bindings.

    Uses lczero.backends for efficient neural network evaluation.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        backend: str = "blas",
    ):
        """
        Initialize Leela Chess Zero.

        Args:
            weights_path: Path to weights file (e.g., weights.pb.gz)
            backend: Backend to use ("blas", "cudnn", etc.).
                    "blas" works on CPU with OpenBLAS.
                    CUDA backends will be used automatically if available.
        """
        self.weights_path = weights_path or "weights/lc0_weights.pb.gz"
        self.backend_name = backend

        # Load weights and create backend
        self._load_network()

    def _load_network(self):
        """Load the neural network weights and create backend."""
        try:
            self.weights = lc0.Weights(self.weights_path)
            self.backend = lc0.Backend(weights=self.weights, backend=self.backend_name)

        except Exception as e:
            raise RuntimeError(f"Failed to load LC0 weights from {self.weights_path}: {e}")

    def get_move_probabilities(
        self, board: chess.Board, temperature: float = 1.0
    ) -> dict[str, float]:
        """
        Get probability distribution over all legal moves.

        Args:
            board: python-chess Board object
            temperature: Temperature for softmax (1.0 = no adjustment)

        Returns:
            Dict mapping UCI moves (e.g., "e2e4") to probabilities
        """
        # Convert chess board to LC0 game state
        game = self._board_to_gamestate(board)

        # Create input tensor
        input_tensor = game.as_input(self.backend)

        # Run forward pass
        outputs = self.backend.evaluate(input_tensor)
        output = outputs[0]

        # Get policy probabilities
        policy_indices = game.policy_indices()
        moves = game.moves()

        # Get raw logits
        raw_logits = np.array([output.p_raw(idx)[0] for idx in policy_indices])

        # Apply temperature and softmax
        if temperature != 1.0:
            raw_logits = raw_logits / temperature

        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_logits / exp_logits.sum()

        return {move: float(prob) for move, prob in zip(moves, probs)}

    def _board_to_gamestate(self, board: chess.Board) -> lc0.GameState:
        """Convert python-chess Board to LC0 GameState."""
        # Convert board to FEN and create GameState from it
        fen = board.fen()
        return lc0.GameState(fen)

    def close(self):
        """Clean up resources."""
        # Python bindings handle cleanup automatically
        pass


def load_leela_model(
    weights_path: Optional[str] = None,
    backend: str = "blas",
    **kwargs
) -> LeelaZeroModel:
    """
    Factory function to load Leela Zero model.

    Args:
        weights_path: Path to weights file
        backend: Backend to use (default: "blas" for CPU)
        **kwargs: Additional arguments for LeelaZeroModel

    Returns:
        LeelaZeroModel
    """
    return LeelaZeroModel(weights_path=weights_path, backend=backend, **kwargs)
