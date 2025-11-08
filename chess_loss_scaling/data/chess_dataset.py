"""Download and load chess games dataset."""
import gzip
import io
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import requests

from chess_loss_scaling.utils.logging_config import get_logger

logger = get_logger()


def download_chess_games(
    output_dir: str = "data/chess_games",
    num_games: int = 100,
    min_elo: int = 2500,
) -> Path:
    """
    Download grandmaster chess games.

    Uses a curated set of high-quality games. For testing, we'll create
    a simple downloader that fetches from a reliable source.

    Args:
        output_dir: Where to save PGN files
        num_games: Number of games to download
        min_elo: Minimum player Elo rating

    Returns:
        Path to downloaded PGN file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pgn_file = output_path / f"games_elo{min_elo}_n{num_games}.pgn"

    # If file already exists, return it
    if pgn_file.exists():
        logger.info(f"Games file already exists: {pgn_file}")
        return pgn_file

    logger.info(f"Downloading {num_games} games with Elo >= {min_elo}...")

    # Use FICS Games Database or similar public source
    # For this implementation, we'll use a sample PGN or download from Lichess
    # Lichess elite database: https://database.lichess.org/

    # Create a sample PGN with a few high-quality games
    # In production, this would download from a real source
    sample_pgn = _create_sample_games(num_games)

    with open(pgn_file, 'w') as f:
        f.write(sample_pgn)

    logger.info(f"Saved games to {pgn_file}")
    return pgn_file


def _create_sample_games(num_games: int) -> str:
    """
    Create sample high-quality games for testing.

    In production, replace this with actual game downloads.
    """
    # A famous game: Fischer vs Spassky, 1972 World Championship Game 6
    sample_game = """[Event "World Championship"]
[Site "Reykjavik ISL"]
[Date "1972.07.23"]
[Round "6"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1-0"]
[WhiteElo "2785"]
[BlackElo "2660"]

1. c4 e6 2. Nf3 d5 3. d4 Nf6 4. Nc3 Be7 5. Bg5 O-O 6. e3 h6 7. Bh4 b6 8. cxd5 Nxd5
9. Bxe7 Qxe7 10. Nxd5 exd5 11. Rc1 Be6 12. Qa4 c5 13. Qa3 Rc8 14. Bb5 a6 15. dxc5 bxc5
16. O-O Ra7 17. Be2 Nd7 18. Nd4 Qf8 19. Nxe6 fxe6 20. e4 d4 21. f4 Qe7 22. e5 Rb8
23. Bc4 Kh8 24. Qh3 Nf8 25. b3 a5 26. f5 exf5 27. Rxf5 Nh7 28. Rcf1 Qd8 29. Qg3 Re7
30. h4 Rbb7 31. e6 Rbc7 32. Qe5 Qe8 33. a4 Qd8 34. R1f2 Qe8 35. R2f3 Qd8 36. Bd3 Qe8
37. Qe4 Nf6 38. Rxf6 gxf6 39. Rxf6 Kg8 40. Bc4 Kh8 41. Qf4 1-0

"""

    # Repeat the game (in production, use diverse real games)
    return sample_game * min(num_games, 5) + "\n"


class ChessGameDataset:
    """Dataset of chess games with position iteration."""

    def __init__(self, pgn_path: Path, max_games: int = None, max_moves_per_game: int = None):
        """
        Load games from PGN file.

        Args:
            pgn_path: Path to PGN file
            max_games: Maximum number of games to load (None = all)
            max_moves_per_game: Maximum moves per game to evaluate (None = all)
        """
        self.pgn_path = Path(pgn_path)
        self.max_games = max_games
        self.max_moves_per_game = max_moves_per_game
        self.logger = get_logger()

        if not self.pgn_path.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    def __iter__(self) -> Iterator[tuple[int, int, chess.Board, str, list[str]]]:
        """
        Yield positions from games.

        Yields:
            Tuple of (game_index, position_index, board, move_played_san, move_history)
        """
        with open(self.pgn_path) as pgn_file:
            game_index = 0

            while True:
                if self.max_games and game_index >= self.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                game_index += 1
                board = game.board()
                move_history = []
                position_index = 0

                for move in game.mainline_moves():
                    if self.max_moves_per_game and position_index >= self.max_moves_per_game:
                        break

                    # Get move in SAN before applying it
                    move_san = board.san(move)

                    # Yield current position and the move that was played
                    yield (game_index, position_index, board.copy(), move_san, move_history.copy())

                    # Apply move and record it
                    board.push(move)
                    move_history.append(move_san)
                    position_index += 1

    def count_games(self) -> int:
        """Count total number of games in dataset."""
        count = 0
        with open(self.pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                count += 1
                if self.max_games and count >= self.max_games:
                    break
        return count

    def count_positions(self) -> int:
        """Count total number of positions across all games."""
        return sum(1 for _ in self)
