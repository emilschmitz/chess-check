"""Calculate cross-entropy loss for chess move predictions."""
from typing import Any

import chess
import chess.pgn
import numpy as np

from ..data.converters import board_to_llm_prompt, get_legal_moves_san, normalize_move_format
from ..models.leela_loader import LeelaZeroModel
from ..models.llm_loader import ChessLLM
from ..utils.logging_config import get_logger

logger = get_logger()


def cross_entropy_loss(
    predicted_probs: dict[str, float],
    target_probs: dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate cross-entropy loss H(target, predicted).

    Formula: -Σ target(x) * log(predicted(x))

    Args:
        predicted_probs: LLM's probability distribution
        target_probs: Leela's probability distribution (ground truth)
        epsilon: Small value to avoid log(0)

    Returns:
        Cross-entropy loss value
    """
    # Ensure both distributions have the same moves
    all_moves = set(predicted_probs.keys()) | set(target_probs.keys())

    loss = 0.0
    for move in all_moves:
        target_p = target_probs.get(move, 0.0)
        predicted_p = max(predicted_probs.get(move, epsilon), epsilon)

        # Cross-entropy: -target * log(predicted)
        loss -= target_p * np.log(predicted_p)

    return float(loss)


def kl_divergence(
    predicted_probs: dict[str, float],
    target_probs: dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate KL divergence KL(target || predicted).

    This is another way to measure distribution distance.

    Args:
        predicted_probs: LLM's probability distribution
        target_probs: Leela's probability distribution
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    all_moves = set(predicted_probs.keys()) | set(target_probs.keys())

    kl = 0.0
    for move in all_moves:
        target_p = max(target_probs.get(move, 0.0), epsilon)
        predicted_p = max(predicted_probs.get(move, epsilon), epsilon)

        # KL: target * log(target / predicted)
        kl += target_p * np.log(target_p / predicted_p)

    return float(kl)


def calculate_position_loss(
    llm: ChessLLM,
    leela: LeelaZeroModel,
    board: chess.Board,
    move_history: list[str],
) -> dict[str, Any]:
    """
    Calculate loss for a single position.

    Args:
        llm: Language model
        leela: Leela Chess Zero model
        board: Current board position
        move_history: List of moves in SAN up to this position

    Returns:
        Dict with loss metrics and distributions
    """
    # Get legal moves
    legal_moves_san = get_legal_moves_san(board)

    if not legal_moves_san:
        # No legal moves (checkmate or stalemate)
        return {
            "cross_entropy": 0.0,
            "kl_divergence": 0.0,
            "legal_moves_count": 0,
            "error": "No legal moves"
        }

    # Get LLM probabilities
    prompt = board_to_llm_prompt(board, move_history)
    try:
        llm_probs = llm.get_move_probabilities(prompt, legal_moves_san, board)
    except Exception as e:
        logger.error(f"Error getting LLM probabilities: {e}")
        return {
            "cross_entropy": float('inf'),
            "kl_divergence": float('inf'),
            "legal_moves_count": len(legal_moves_san),
            "error": str(e)
        }

    # Get Leela probabilities (in UCI format)
    try:
        leela_probs_uci = leela.get_move_probabilities(board)
    except Exception as e:
        logger.error(f"Error getting Leela probabilities: {e}")
        return {
            "cross_entropy": float('inf'),
            "kl_divergence": float('inf'),
            "legal_moves_count": len(legal_moves_san),
            "error": str(e)
        }

    # Convert Leela probs from UCI to SAN to match LLM format
    leela_probs_san = {}
    for uci_move, prob in leela_probs_uci.items():
        try:
            san_move = normalize_move_format(uci_move, board, from_format="uci", to_format="san")
            leela_probs_san[san_move] = prob
        except ValueError:
            # Move conversion failed, skip
            continue

    # Calculate losses
    ce_loss = cross_entropy_loss(llm_probs, leela_probs_san)
    kl_div = kl_divergence(llm_probs, leela_probs_san)

    return {
        "cross_entropy": ce_loss,
        "kl_divergence": kl_div,
        "legal_moves_count": len(legal_moves_san),
        "llm_top_move": max(llm_probs.items(), key=lambda x: x[1])[0] if llm_probs else None,
        "leela_top_move": max(leela_probs_san.items(), key=lambda x: x[1])[0] if leela_probs_san else None,
    }


def calculate_game_loss(
    llm: ChessLLM,
    leela: LeelaZeroModel,
    game: chess.pgn.Game,
    max_positions: int | None = None,
) -> dict[str, Any]:
    """
    Calculate average loss for one game.

    Args:
        llm: Language model
        leela: Leela Chess Zero model
        game: PGN game
        max_positions: Maximum positions to evaluate (None = all)

    Returns:
        Dict with game-level statistics
    """
    board = game.board()
    move_history = []
    position_losses = []
    position_index = 0

    for move in game.mainline_moves():
        if max_positions and position_index >= max_positions:
            break

        # Calculate loss for current position
        pos_loss = calculate_position_loss(llm, leela, board, move_history)

        if "error" not in pos_loss:
            position_losses.append(pos_loss)

        # Apply move and continue
        move_san = board.san(move)
        board.push(move)
        move_history.append(move_san)
        position_index += 1

    # Aggregate statistics
    if not position_losses:
        return {
            "num_positions": 0,
            "avg_cross_entropy": float('inf'),
            "avg_kl_divergence": float('inf'),
            "std_cross_entropy": 0.0,
            "error": "No valid positions evaluated"
        }

    ce_losses = [p["cross_entropy"] for p in position_losses]
    kl_losses = [p["kl_divergence"] for p in position_losses]

    return {
        "num_positions": len(position_losses),
        "avg_cross_entropy": float(np.mean(ce_losses)),
        "avg_kl_divergence": float(np.mean(kl_losses)),
        "std_cross_entropy": float(np.std(ce_losses)),
        "std_kl_divergence": float(np.std(kl_losses)),
        "min_cross_entropy": float(np.min(ce_losses)),
        "max_cross_entropy": float(np.max(ce_losses)),
        "per_position_losses": position_losses,
    }
