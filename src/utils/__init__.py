"""
Combat Racing Championship - Utilities Package
==============================================

Core utility functions and helper classes.
"""

from .logger import setup_logger, get_logger
from .config_loader import ConfigLoader
from .helpers import (
    normalize_angle,
    rotate_point,
    distance,
    angle_between,
    clip_value,
    seed_everything,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "ConfigLoader",
    "normalize_angle",
    "rotate_point",
    "distance",
    "angle_between",
    "clip_value",
    "seed_everything",
]
