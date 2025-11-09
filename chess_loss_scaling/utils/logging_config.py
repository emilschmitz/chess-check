"""Logging configuration with rich output."""
import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """
    Configure logging with rich handlers and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file (if None, creates timestamped file in logs/)

    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create log directory and file if not specified
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"chess_loss_scaling_{timestamp}_utc.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with UTC formatter
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        "%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_formatter.converter = lambda *args: datetime.utcnow().timetuple()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(numeric_level)

    # Create handlers
    handlers = [
        # Console handler with rich formatting
        RichHandler(rich_tracebacks=True, tracebacks_show_locals=True),
        # File handler with detailed formatting and UTC timestamps
        file_handler
    ]

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # Get logger for our package
    logger = logging.getLogger("chess_loss_scaling")
    logger.setLevel(numeric_level)

    # Log where we're writing to
    logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str = "chess_loss_scaling") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
