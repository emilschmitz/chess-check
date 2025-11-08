"""Convert between chess formats for LLMs and engines."""
import chess


def board_to_llm_prompt(board: chess.Board, move_history: list[str]) -> str:
    """
    Convert chess position to LLM prompt format.

    Args:
        board: Current position
        move_history: List of moves in standard algebraic notation (SAN)

    Returns:
        Formatted prompt string with game history
    """
    if not move_history:
        return "Starting position. What is the best move?"

    # Format moves in pairs (white, black)
    formatted_moves = []
    for i in range(0, len(move_history), 2):
        move_num = (i // 2) + 1
        white_move = move_history[i]
        black_move = move_history[i + 1] if i + 1 < len(move_history) else None

        if black_move:
            formatted_moves.append(f"{move_num}. {white_move} {black_move}")
        else:
            formatted_moves.append(f"{move_num}. {white_move}")

    game_text = " ".join(formatted_moves)
    prompt = f"{game_text}\n\nWhat is the next move?"

    return prompt


def normalize_move_format(
    move: str,
    board: chess.Board,
    from_format: str = "san",
    to_format: str = "uci"
) -> str:
    """
    Convert between move formats.

    Args:
        move: Move string
        board: Board position (needed for SAN conversion)
        from_format: Source format ("uci", "san")
        to_format: Target format ("uci", "san")

    Returns:
        Converted move string
    """
    if from_format == to_format:
        return move

    try:
        if from_format == "san":
            # Parse SAN move to get Move object
            chess_move = board.parse_san(move)
            if to_format == "uci":
                return chess_move.uci()
        elif from_format == "uci":
            # Parse UCI move
            chess_move = chess.Move.from_uci(move)
            if to_format == "san":
                return board.san(chess_move)
    except (ValueError, chess.InvalidMoveError) as e:
        raise ValueError(f"Failed to convert move '{move}' from {from_format} to {to_format}: {e}")

    return move


def get_legal_moves_san(board: chess.Board) -> list[str]:
    """
    Get all legal moves in standard algebraic notation.

    Args:
        board: Current board position

    Returns:
        List of legal moves in SAN format
    """
    return [board.san(move) for move in board.legal_moves]


def get_legal_moves_uci(board: chess.Board) -> list[str]:
    """
    Get all legal moves in UCI notation.

    Args:
        board: Current board position

    Returns:
        List of legal moves in UCI format
    """
    return [move.uci() for move in board.legal_moves]
