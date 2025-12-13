"""
Helper Functions
===============

Common utility functions used throughout the project.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import math
import random
from typing import Tuple, List, Any, Optional
import numpy as np
import torch
from .logger import get_logger

logger = get_logger(__name__)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [-π, π].
    
    Args:
        angle: Angle in radians.
    
    Returns:
        Normalized angle in range [-π, π].
    
    Example:
        >>> normalize_angle(3.5 * math.pi)
        -1.5707963267948966
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def rotate_point(
    point: Tuple[float, float],
    angle: float,
    origin: Tuple[float, float] = (0, 0),
) -> Tuple[float, float]:
    """
    Rotate a point around an origin by a given angle.
    
    Args:
        point: Point to rotate (x, y).
        angle: Rotation angle in radians (counter-clockwise).
        origin: Center of rotation (default: (0, 0)).
    
    Returns:
        Rotated point (x, y).
    
    Example:
        >>> rotate_point((1, 0), math.pi/2)
        (0.0, 1.0)
    """
    ox, oy = origin
    px, py = point
    
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    qx = ox + cos_angle * (px - ox) - sin_angle * (py - oy)
    qy = oy + sin_angle * (px - ox) + cos_angle * (py - oy)
    
    return qx, qy


def distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y).
        point2: Second point (x, y).
    
    Returns:
        Distance between points.
    
    Example:
        >>> distance((0, 0), (3, 4))
        5.0
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def angle_between(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """
    Calculate angle from point1 to point2.
    
    Args:
        point1: Starting point (x, y).
        point2: Target point (x, y).
    
    Returns:
        Angle in radians [-π, π].
    
    Example:
        >>> angle_between((0, 0), (1, 1))
        0.7853981633974483  # π/4
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.atan2(dy, dx)


def clip_value(
    value: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Clip value to specified range.
    
    Args:
        value: Value to clip.
        min_val: Minimum value.
        max_val: Maximum value.
    
    Returns:
        Clipped value.
    
    Example:
        >>> clip_value(150, 0, 100)
        100
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor [0, 1].
    
    Returns:
        Interpolated value.
    
    Example:
        >>> lerp(0, 100, 0.5)
        50.0
    """
    return a + (b - a) * t


def smooth_damp(
    current: float,
    target: float,
    current_velocity: float,
    smooth_time: float,
    max_speed: float,
    delta_time: float,
) -> Tuple[float, float]:
    """
    Smoothly damp a value toward a target (like Unity's SmoothDamp).
    
    Args:
        current: Current value.
        target: Target value.
        current_velocity: Current rate of change.
        smooth_time: Approximate time to reach target.
        max_speed: Maximum speed.
        delta_time: Time since last update.
    
    Returns:
        Tuple of (new_value, new_velocity).
    """
    smooth_time = max(0.0001, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * delta_time
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    
    change = current - target
    original_to = target
    
    max_change = max_speed * smooth_time
    change = clip_value(change, -max_change, max_change)
    target = current - change
    
    temp = (current_velocity + omega * change) * delta_time
    current_velocity = (current_velocity - omega * temp) * exp
    output = target + (change + temp) * exp
    
    if (original_to - current > 0) == (output > original_to):
        output = original_to
        current_velocity = (output - original_to) / delta_time
    
    return output, current_velocity


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    
    Example:
        >>> seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {seed}")


def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of a list.
    
    Args:
        data: Input data.
        window_size: Size of moving window.
    
    Returns:
        Smoothed data.
    
    Example:
        >>> moving_average([1, 2, 3, 4, 5], 3)
        [1.0, 1.5, 2.0, 3.0, 4.0]
    """
    if window_size <= 0:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i + 1]
        result.append(sum(window) / len(window))
    
    return result


def exponential_moving_average(
    data: List[float],
    alpha: float = 0.1,
) -> List[float]:
    """
    Calculate exponential moving average.
    
    Args:
        data: Input data.
        alpha: Smoothing factor (0 < alpha <= 1).
    
    Returns:
        Smoothed data.
    
    Example:
        >>> exponential_moving_average([1, 2, 3, 4, 5], 0.3)
        [1.0, 1.3, 1.81, 2.467, 3.2269]
    """
    if not data:
        return []
    
    result = [data[0]]
    for value in data[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    
    return result


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector.
    
    Returns:
        Normalized vector.
    
    Example:
        >>> normalize_vector(np.array([3, 4]))
        array([0.6, 0.8])
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def get_device() -> torch.device:
    """
    Get the best available PyTorch device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device.
    
    Example:
        >>> device = get_device()
        >>> tensor = torch.zeros(10).to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable time string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted string (e.g., "1h 23m 45s").
    
    Example:
        >>> format_time(3725)
        '1h 2m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format.
        precision: Decimal places.
    
    Returns:
        Formatted string.
    
    Example:
        >>> format_number(1234567)
        '1.23M'
    """
    if abs(num) < 1000:
        return f"{num:.{precision}f}"
    elif abs(num) < 1_000_000:
        return f"{num/1_000:.{precision}f}K"
    elif abs(num) < 1_000_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    else:
        return f"{num/1_000_000_000:.{precision}f}B"
