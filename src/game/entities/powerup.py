"""
Power-Up System
==============

Collectible power-ups that enhance car abilities.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from enum import Enum
from typing import Tuple, Dict
import numpy as np
from ..physics import PhysicsBody
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PowerUpType(Enum):
    """Power-up type enumeration."""
    SPEED_BOOST = "speed_boost"
    SHIELD = "shield"
    DOUBLE_DAMAGE = "double_damage"
    AMMO_REFILL = "ammo_refill"
    HEALTH_PACK = "health_pack"


class PowerUp(PhysicsBody):
    """
    Collectible power-up.
    
    Power-ups spawn on the track and provide temporary boosts.
    """
    
    # Visual colors for each type
    COLORS = {
        PowerUpType.SPEED_BOOST: (0, 200, 255),      # Cyan
        PowerUpType.SHIELD: (100, 100, 255),          # Blue
        PowerUpType.DOUBLE_DAMAGE: (255, 100, 0),     # Orange
        PowerUpType.AMMO_REFILL: (255, 255, 0),       # Yellow
        PowerUpType.HEALTH_PACK: (0, 255, 0),         # Green
    }
    
    def __init__(
        self,
        position: Tuple[float, float],
        powerup_type: PowerUpType,
        config: Dict,
    ):
        """
        Initialize power-up.
        
        Args:
            position: Spawn position (x, y).
            powerup_type: Type of power-up.
            config: Configuration dictionary with effects.
        """
        super().__init__(
            position=position,
            velocity=(0, 0),
            rotation=0,
            mass=0.1,
        )
        
        # Static object
        self.is_static = True
        self.is_trigger = True
        
        # Collision
        self.collision_shape = "circle"
        self.collision_radius = 20.0
        
        # Power-up properties
        self.powerup_type = powerup_type
        self.config = config
        self.is_collected = False
        self.lifetime = config.get("lifetime", 15.0)
        self.time_alive = 0.0
        self.respawn_delay = config.get("respawn_delay", 3.0)
        
        # Visual
        self.color = self.COLORS.get(powerup_type, (255, 255, 255))
        self.rotation_speed = 2.0  # For visual effect
    
    def update(self, dt: float) -> bool:
        """
        Update power-up.
        
        Returns:
            True if still active, False if expired.
        """
        if self.is_collected:
            return False
        
        self.time_alive += dt
        
        # Rotate for visual effect
        self.rotation += self.rotation_speed * dt
        
        # Check lifetime
        if self.time_alive >= self.lifetime:
            return False
        
        return True
    
    def collect(self, car: any) -> Dict:
        """
        Collect power-up.
        
        Args:
            car: Car that collected the power-up.
        
        Returns:
            Effect parameters to apply to car.
        """
        if self.is_collected:
            return {}
        
        self.is_collected = True
        logger.info(
            f"Car {car.agent_id} collected {self.powerup_type.value} power-up"
        )
        
        # Return effect parameters based on type
        if self.powerup_type == PowerUpType.SPEED_BOOST:
            return {
                "type": "speed_boost",
                "duration": self.config.get("duration", 5.0),
                "multiplier": self.config.get("speed_multiplier", 1.5),
            }
        
        elif self.powerup_type == PowerUpType.SHIELD:
            return {
                "type": "shield",
                "duration": self.config.get("duration", 8.0),
                "reduction": self.config.get("damage_reduction", 0.7),
            }
        
        elif self.powerup_type == PowerUpType.DOUBLE_DAMAGE:
            return {
                "type": "double_damage",
                "duration": self.config.get("duration", 6.0),
                "multiplier": self.config.get("damage_multiplier", 2.0),
            }
        
        elif self.powerup_type == PowerUpType.AMMO_REFILL:
            # Instant effect
            car.add_ammo("laser", self.config.get("laser_amount", 20))
            car.add_ammo("missile", self.config.get("missile_amount", 5))
            car.add_ammo("mine", self.config.get("mine_amount", 3))
            return {}
        
        elif self.powerup_type == PowerUpType.HEALTH_PACK:
            # Instant effect
            car.heal(self.config.get("heal_amount", 50.0))
            return {}
        
        return {}
