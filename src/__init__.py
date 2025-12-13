"""
Combat Racing Championship - Main Package
==========================================

A professional reinforcement learning project combining racing and combat.
"""

__version__ = "1.0.0"
__author__ = "Combat Racing RL Team"
__license__ = "MIT"

from .utils import setup_logger, get_logger, ConfigLoader

__all__ = [
    "__version__",
    "setup_logger",
    "get_logger",
    "ConfigLoader",
]
