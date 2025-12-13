"""
Logging Configuration
====================

Professional logging setup using loguru for beautiful, structured logs.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    compression: str = "zip",
) -> None:
    """
    Configure the logging system with file and console output.
    
    Args:
        log_dir: Directory for log files. If None, logs only to console.
        level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        rotation: When to rotate log files ("100 MB", "1 week", etc.).
        retention: How long to keep old logs.
        compression: Compression format for old logs.
    
    Example:
        >>> setup_logger(Path("logs"), level="DEBUG")
        >>> logger.info("Training started")
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file logger if directory specified
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        logger.add(
            log_dir / "combat_racing_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
        )
        
        # Separate error log
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"Logger initialized with level: {level}")


def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module).
    
    Returns:
        Logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
    """
    return logger.bind(name=name)


# Create a default logger instance
setup_logger(level="INFO")
