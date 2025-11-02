"""Tests for chess format converters."""
import chess
import pytest

from src.data.converters import (
    board_to_llm_prompt,
    get_legal_moves_san,
    get_legal_moves_uci,
    normalize_move_format,
)


def test_board_to_llm_prompt_starting_position():
    """Test prompt generation for starting position."""
    board = chess.Board()
    prompt = board_to_llm_prompt(board, [])
    assert "Starting position" in prompt
    assert "What is the best move?" in prompt


def test_board_to_llm_prompt_with_moves():
    """Test prompt generation with move history."""
    board = chess.Board()
    move_history = ["e4", "e5", "Nf3"]

    prompt = board_to_llm_prompt(board, move_history)

    assert "1. e4 e5" in prompt
    assert "2. Nf3" in prompt
    assert "What is the next move?" in prompt


def test_get_legal_moves_san():
    """Test getting legal moves in SAN format."""
    board = chess.Board()
    moves = get_legal_moves_san(board)

    # Starting position has 20 legal moves
    assert len(moves) == 20
    assert "e4" in moves
    assert "Nf3" in moves


def test_get_legal_moves_uci():
    """Test getting legal moves in UCI format."""
    board = chess.Board()
    moves = get_legal_moves_uci(board)

    # Starting position has 20 legal moves
    assert len(moves) == 20
    assert "e2e4" in moves
    assert "g1f3" in moves


def test_normalize_move_format_san_to_uci():
    """Test converting SAN to UCI."""
    board = chess.Board()
    uci_move = normalize_move_format("e4", board, from_format="san", to_format="uci")
    assert uci_move == "e2e4"


def test_normalize_move_format_uci_to_san():
    """Test converting UCI to SAN."""
    board = chess.Board()
    san_move = normalize_move_format("e2e4", board, from_format="uci", to_format="san")
    assert san_move == "e4"


def test_normalize_move_format_same():
    """Test that same format returns unchanged."""
    board = chess.Board()
    move = "e4"
    result = normalize_move_format(move, board, from_format="san", to_format="san")
    assert result == move


def test_normalize_move_format_invalid():
    """Test that invalid move raises error."""
    board = chess.Board()
    with pytest.raises(ValueError):
        normalize_move_format("z9z9", board, from_format="uci", to_format="san")
