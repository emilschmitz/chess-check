"""Logging configuration with rich output."""
import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with rich handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)]
    )

    # Get logger for our package
    logger = logging.getLogger("chess_loss_scaling")
    logger.setLevel(numeric_level)

    return logger


def get_logger(name: str = "chess_loss_scaling") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
