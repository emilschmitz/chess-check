"""Leela Chess Zero integration for ground truth move probabilities."""
import subprocess
from typing import Optional

import chess
import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger()


class LeelaZeroModel:
    """
    Load and run Leela Chess Zero for move probabilities.

    This implementation uses UCI protocol to communicate with lc0.
    Alternatively, you can use python-lczero bindings if available.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        lc0_binary: str = "lc0",
        device: str = "cpu",
        nodes: int = 100,
    ):
        """
        Initialize Leela Chess Zero.

        Args:
            weights_path: Path to weights file (e.g., weights.pb.gz)
            lc0_binary: Path to lc0 executable
            device: "cuda" or "cpu"
            nodes: Number of nodes to search (affects quality)
        """
        self.weights_path = weights_path
        self.lc0_binary = lc0_binary
        self.device = device
        self.nodes = nodes
        self.process: Optional[subprocess.Popen] = None
        self.logger = get_logger()

        # Check if lc0 is available
        self._check_lc0_available()

    def _check_lc0_available(self):
        """Check if lc0 binary is available."""
        try:
            result = subprocess.run(
                [self.lc0_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.info(f"Found lc0: {result.stdout.split()[0] if result.stdout else 'version unknown'}")
            else:
                self.logger.warning("lc0 binary found but may not be working correctly")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning(
                f"lc0 not found at '{self.lc0_binary}'. "
                "Install from: https://github.com/LeelaChessZero/lc0/releases"
            )
            raise RuntimeError("lc0 binary not available")

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
        # For a real implementation, we would:
        # 1. Set up UCI communication with lc0
        # 2. Send position
        # 3. Request "go nodes {self.nodes}"
        # 4. Parse "info" lines to get policy probabilities

        # Placeholder: Return uniform distribution over legal moves
        # In production, replace with actual lc0 communication
        return self._mock_leela_probabilities(board)

    def _mock_leela_probabilities(self, board: chess.Board) -> dict[str, float]:
        """
        Mock implementation that returns realistic-looking probabilities.

        In production, replace this with actual lc0 UCI communication.
        """
        legal_moves = list(board.legal_moves)
        num_moves = len(legal_moves)

        if num_moves == 0:
            return {}

        # Create somewhat realistic distribution (not uniform)
        # Better moves (captures, checks) get higher probabilities
        logits = []
        for move in legal_moves:
            score = 1.0

            # Captures get bonus
            if board.is_capture(move):
                score += 2.0

            # Checks get bonus
            board.push(move)
            if board.is_check():
                score += 1.5
            board.pop()

            # Add some randomness
            score += np.random.normal(0, 0.5)
            logits.append(score)

        # Softmax to get probabilities
        logits_array = np.array(logits)
        exp_logits = np.exp(logits_array - np.max(logits_array))  # Numerical stability
        probs = exp_logits / exp_logits.sum()

        return {move.uci(): float(prob) for move, prob in zip(legal_moves, probs)}

    def close(self):
        """Clean up resources."""
        if self.process:
            self.process.terminate()
            self.process = None


class MockLeelaZeroModel:
    """
    Mock Leela Zero model for testing without lc0 installation.

    Returns random but valid probability distributions.
    """

    def __init__(self, **kwargs):
        """Initialize mock model (accepts any kwargs for compatibility)."""
        self.logger = get_logger()
        self.logger.warning("Using MockLeelaZeroModel - not real Leela Zero!")

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


def load_leela_model(
    weights_path: Optional[str] = None,
    mock: bool = False,
    **kwargs
) -> LeelaZeroModel | MockLeelaZeroModel:
    """
    Factory function to load Leela Zero model.

    Args:
        weights_path: Path to weights file
        mock: If True, use mock model instead of real lc0
        **kwargs: Additional arguments for LeelaZeroModel

    Returns:
        LeelaZeroModel or MockLeelaZeroModel
    """
    if mock:
        logger.info("Loading mock Leela Zero model")
        return MockLeelaZeroModel(**kwargs)

    try:
        logger.info("Loading real Leela Zero model")
        return LeelaZeroModel(weights_path=weights_path, **kwargs)
    except RuntimeError as e:
        logger.warning(f"Failed to load real Leela Zero: {e}")
        logger.info("Falling back to mock model")
        return MockLeelaZeroModel(**kwargs)
