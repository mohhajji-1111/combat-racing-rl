"""
Game Package
===========

Core game engine components for Combat Racing Championship.
"""

from .entities import Car, Projectile, PowerUp
from .physics import PhysicsEngine
from .track import Track, create_oval_track
from .renderer import Renderer
from .engine import GameEngine

__all__ = [
    "Car",
    "Projectile",
    "PowerUp",
    "PhysicsEngine",
    "Track",
    "create_oval_track",
    "Renderer",
    "GameEngine",
]
