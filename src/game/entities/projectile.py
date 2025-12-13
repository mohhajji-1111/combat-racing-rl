"""
Projectile Entities
==================

Weapons and projectiles for combat system.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import math
from typing import Tuple, Optional
from enum import Enum
import numpy as np
from ..physics import PhysicsBody
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ProjectileType(Enum):
    """Projectile type enumeration."""
    LASER = "laser"
    MISSILE = "missile"
    MINE = "mine"


class Projectile(PhysicsBody):
    """Base projectile class."""
    
    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        rotation: float,
        owner: any,
        damage: float,
        lifetime: float,
        projectile_type: ProjectileType,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            rotation=rotation,
            mass=0.1,
            friction=1.0,
        )
        
        self.owner = owner
        self.base_damage = damage
        self.damage_multiplier = 1.0
        self.lifetime = lifetime
        self.time_alive = 0.0
        self.is_active = True
        self.projectile_type = projectile_type
        
        # Collision
        self.collision_shape = "circle"
        self.collision_radius = 5.0
        self.is_trigger = True
    
    def update(self, dt: float) -> bool:
        """
        Update projectile.
        
        Returns:
            True if still active, False if expired.
        """
        if not self.is_active:
            return False
        
        super().update(dt)
        self.time_alive += dt
        
        if self.time_alive >= self.lifetime:
            self.is_active = False
            return False
        
        return True
    
    def get_damage(self) -> float:
        """Get total damage including multipliers."""
        return self.base_damage * self.damage_multiplier
    
    def hit(self, target: any) -> None:
        """Handle hit event."""
        self.is_active = False


class Laser(Projectile):
    """Fast laser projectile."""
    
    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        rotation: float,
        owner: any,
        damage: float = 15.0,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            rotation=rotation,
            owner=owner,
            damage=damage,
            lifetime=2.0,
            projectile_type=ProjectileType.LASER,
        )
        self.collision_radius = 3.0
        self.color = (0, 255, 255)  # Cyan


class Missile(Projectile):
    """Homing missile projectile."""
    
    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        rotation: float,
        owner: any,
        damage: float = 35.0,
        homing_strength: float = 50.0,
    ):
        super().__init__(
            position=position,
            velocity=velocity,
            rotation=rotation,
            owner=owner,
            damage=damage,
            lifetime=3.0,
            projectile_type=ProjectileType.MISSILE,
        )
        self.collision_radius = 5.0
        self.homing_strength = homing_strength
        self.target: Optional[any] = None
        self.explosion_radius = 60.0
        self.color = (255, 100, 0)  # Orange
    
    def update(self, dt: float) -> bool:
        """Update missile with homing behavior."""
        if not super().update(dt):
            return False
        
        # Homing logic
        if self.target and hasattr(self.target, 'position'):
            # Calculate direction to target
            to_target = self.target.position - self.position
            dist = np.linalg.norm(to_target)
            
            if dist > 0:
                desired_dir = to_target / dist
                current_dir = self.velocity / np.linalg.norm(self.velocity)
                
                # Steer towards target
                steer = desired_dir - current_dir
                self.velocity += steer * self.homing_strength * dt
        
        return True


class Mine(Projectile):
    """Stationary mine that explodes on proximity."""
    
    def __init__(
        self,
        position: Tuple[float, float],
        owner: any,
        damage: float = 50.0,
        trigger_radius: float = 50.0,
    ):
        super().__init__(
            position=position,
            velocity=(0, 0),
            rotation=0,
            owner=owner,
            damage=damage,
            lifetime=10.0,
            projectile_type=ProjectileType.MINE,
        )
        self.collision_radius = 8.0
        self.trigger_radius = trigger_radius
        self.explosion_radius = 80.0
        self.is_armed = False
        self.arm_delay = 0.5
        self.color = (255, 0, 0)  # Red
    
    def update(self, dt: float) -> bool:
        """Update mine."""
        if not super().update(dt):
            return False
        
        # Arm after delay
        if not self.is_armed and self.time_alive >= self.arm_delay:
            self.is_armed = True
        
        return True
    
    def check_proximity(self, target: any) -> bool:
        """
        Check if target is in trigger radius.
        
        Returns:
            True if mine should explode.
        """
        if not self.is_armed or target == self.owner:
            return False
        
        dist = np.linalg.norm(target.position - self.position)
        return dist <= self.trigger_radius
