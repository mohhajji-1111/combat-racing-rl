"""
Car Entity
=========

Racing car with combat capabilities.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import math
from typing import Tuple, Dict, Optional, List
from enum import Enum
import numpy as np
from ..physics import PhysicsBody
from ...utils.logger import get_logger
from ...utils.helpers import normalize_angle, rotate_point

logger = get_logger(__name__)


class CarState(Enum):
    """Car state enumeration."""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    RESPAWNING = "respawning"
    FINISHED = "finished"


class PowerUpEffect:
    """Active power-up effect on car."""
    
    def __init__(
        self,
        effect_type: str,
        duration: float,
        **kwargs,
    ):
        self.effect_type = effect_type
        self.duration = duration
        self.time_remaining = duration
        self.params = kwargs
    
    def update(self, dt: float) -> bool:
        """
        Update effect timer.
        
        Returns:
            True if effect is still active, False if expired.
        """
        self.time_remaining -= dt
        return self.time_remaining > 0


class Car(PhysicsBody):
    """
    Racing car with combat capabilities.
    
    Features:
        - Realistic driving physics
        - Health and damage system
        - Weapons and ammo management
        - Power-up effects
        - Checkpoint tracking
        - Statistics tracking
    """
    
    def __init__(
        self,
        position: Tuple[float, float],
        rotation: float = 0.0,
        color: Tuple[int, int, int] = (255, 0, 0),
        config: Optional[Dict] = None,
        agent_id: int = 0,
    ):
        """
        Initialize car.
        
        Args:
            position: Starting position (x, y).
            rotation: Starting rotation in radians.
            color: RGB color tuple.
            config: Configuration dictionary.
            agent_id: Unique agent identifier.
        """
        # Default config
        if config is None:
            config = {
                "size": [40, 30],
                "mass": 1.0,
                "max_speed": 400.0,
                "acceleration": 200.0,
                "deceleration": 150.0,
                "turn_speed": 180.0,
                "max_health": 100.0,
                "health_regen_rate": 2.0,
                "health_regen_delay": 5.0,
            }
        
        # Initialize physics body
        super().__init__(
            position=position,
            velocity=(0, 0),
            rotation=rotation,
            mass=config.get("mass", 1.0),
            friction=0.95,
            bounce=0.3,
        )
        
        # Collision setup
        self.collision_shape = "box"
        self.collision_size = tuple(config.get("size", [40, 30]))
        self.collision_radius = math.sqrt(
            (self.collision_size[0]/2)**2 + (self.collision_size[1]/2)**2
        )
        
        # Identification
        self.agent_id = agent_id
        self.color = color
        
        # Movement parameters
        self.max_speed = config.get("max_speed", 400.0)
        self.acceleration_force = config.get("acceleration", 200.0)
        self.deceleration_force = config.get("deceleration", 150.0)
        self.turn_speed = math.radians(config.get("turn_speed", 180.0))
        
        # Health system
        self.max_health = config.get("max_health", 100.0)
        self.health = self.max_health
        self.health_regen_rate = config.get("health_regen_rate", 2.0)
        self.health_regen_delay = config.get("health_regen_delay", 5.0)
        self.time_since_damage = 0.0
        self.is_alive = True
        
        # Combat
        self.weapons = {
            "laser": {"ammo": 30, "cooldown": 0.0},
            "missile": {"ammo": 10, "cooldown": 0.0},
            "mine": {"ammo": 5, "cooldown": 0.0},
        }
        
        # Power-ups
        self.active_effects: List[PowerUpEffect] = []
        self.shield_active = False
        self.speed_boost_active = False
        self.double_damage_active = False
        
        # Racing state
        self.state = CarState.ACTIVE
        self.checkpoint_index = 0
        self.current_lap = 0
        self.lap_times: List[float] = []
        self.race_time = 0.0
        self.lap_start_time = 0.0
        
        # Statistics
        self.stats = {
            "kills": 0,
            "deaths": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "powerups_collected": 0,
            "collisions": 0,
            "distance_traveled": 0.0,
        }
        
        # Control inputs (for human players or debugging)
        self.throttle_input = 0.0  # -1 to 1
        self.steering_input = 0.0  # -1 to 1
        self.shoot_input = False
        
        logger.debug(f"Car {agent_id} initialized at {position}")
    
    def update(self, dt: float) -> None:
        """
        Update car state.
        
        Args:
            dt: Delta time (seconds).
        """
        if self.state != CarState.ACTIVE:
            return
        
        # Update physics
        super().update(dt, max_velocity=self._get_max_speed())
        
        # Update timers
        self.race_time += dt
        self.time_since_damage += dt
        
        # Health regeneration
        if self.time_since_damage >= self.health_regen_delay:
            self.health = min(self.max_health, self.health + self.health_regen_rate * dt)
        
        # Weapon cooldowns
        for weapon in self.weapons.values():
            if weapon["cooldown"] > 0:
                weapon["cooldown"] -= dt
        
        # Power-up effects
        self._update_powerups(dt)
        
        # Update statistics
        prev_pos = self.position.copy()
        distance = np.linalg.norm(self.position - prev_pos)
        self.stats["distance_traveled"] += distance
    
    def apply_throttle(self, amount: float) -> None:
        """
        Apply throttle input (-1 to 1).
        
        Args:
            amount: Throttle amount (-1=brake, 0=coast, 1=accelerate).
        """
        self.throttle_input = max(-1.0, min(1.0, amount))
        
        # Calculate force in forward direction
        forward = np.array([math.cos(self.rotation), math.sin(self.rotation)])
        
        if amount > 0:
            # Acceleration
            force = forward * self.acceleration_force * amount
        elif amount < 0:
            # Braking/reverse
            force = forward * self.deceleration_force * amount
        else:
            return
        
        self.apply_force(tuple(force))
    
    def apply_steering(self, amount: float) -> None:
        """
        Apply steering input (-1 to 1).
        
        Args:
            amount: Steering amount (-1=left, 0=straight, 1=right).
        """
        self.steering_input = max(-1.0, min(1.0, amount))
        
        # Steering is speed-dependent
        speed = np.linalg.norm(self.velocity)
        speed_factor = min(1.0, speed / (self.max_speed * 0.3))
        
        # Apply torque
        self.angular_velocity = -amount * self.turn_speed * speed_factor
    
    def shoot_weapon(
        self,
        weapon_type: str = "laser",
    ) -> Optional[Dict]:
        """
        Attempt to shoot weapon.
        
        Args:
            weapon_type: Type of weapon ("laser", "missile", "mine").
        
        Returns:
            Projectile spawn data dict if successful, None otherwise.
        """
        if weapon_type not in self.weapons:
            return None
        
        weapon = self.weapons[weapon_type]
        
        # Check ammo and cooldown
        if weapon["ammo"] <= 0 or weapon["cooldown"] > 0:
            return None
        
        # Consume ammo
        weapon["ammo"] -= 1
        
        # Set cooldown
        if weapon_type == "laser":
            weapon["cooldown"] = 0.5
        elif weapon_type == "missile":
            weapon["cooldown"] = 2.0
        elif weapon_type == "mine":
            weapon["cooldown"] = 3.0
        
        # Calculate spawn position (in front of car)
        offset = 30
        spawn_pos = (
            self.position[0] + math.cos(self.rotation) * offset,
            self.position[1] + math.sin(self.rotation) * offset,
        )
        
        # Calculate projectile velocity
        projectile_speed = 600.0 if weapon_type == "laser" else 400.0
        projectile_vel = (
            self.velocity[0] + math.cos(self.rotation) * projectile_speed,
            self.velocity[1] + math.sin(self.rotation) * projectile_speed,
        )
        
        # Damage multiplier from power-ups
        damage_multiplier = 2.0 if self.double_damage_active else 1.0
        
        return {
            "type": weapon_type,
            "position": spawn_pos,
            "velocity": projectile_vel,
            "rotation": self.rotation,
            "owner": self,
            "damage_multiplier": damage_multiplier,
        }
    
    def take_damage(
        self,
        amount: float,
        source: Optional[any] = None,
    ) -> bool:
        """
        Apply damage to car.
        
        Args:
            amount: Damage amount.
            source: Source of damage (e.g., another car or projectile).
        
        Returns:
            True if car was eliminated, False otherwise.
        """
        if not self.is_alive or self.state != CarState.ACTIVE:
            return False
        
        # Shield reduces damage
        if self.shield_active:
            amount *= 0.3
        
        self.health -= amount
        self.time_since_damage = 0.0
        self.stats["damage_taken"] += amount
        
        logger.debug(f"Car {self.agent_id} took {amount:.1f} damage, health: {self.health:.1f}")
        
        # Check elimination
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            self.state = CarState.ELIMINATED
            self.stats["deaths"] += 1
            logger.info(f"Car {self.agent_id} eliminated!")
            return True
        
        return False
    
    def heal(self, amount: float) -> None:
        """Restore health."""
        self.health = min(self.max_health, self.health + amount)
    
    def add_ammo(
        self,
        weapon_type: str,
        amount: int,
    ) -> None:
        """Add ammo to weapon."""
        if weapon_type in self.weapons:
            self.weapons[weapon_type]["ammo"] += amount
    
    def apply_powerup(
        self,
        powerup_type: str,
        duration: float,
        **kwargs,
    ) -> None:
        """
        Apply power-up effect.
        
        Args:
            powerup_type: Type of power-up.
            duration: Effect duration (seconds).
            **kwargs: Additional parameters.
        """
        effect = PowerUpEffect(powerup_type, duration, **kwargs)
        self.active_effects.append(effect)
        
        # Activate effect
        if powerup_type == "speed_boost":
            self.speed_boost_active = True
        elif powerup_type == "shield":
            self.shield_active = True
        elif powerup_type == "double_damage":
            self.double_damage_active = True
        
        self.stats["powerups_collected"] += 1
        logger.info(f"Car {self.agent_id} activated {powerup_type}")
    
    def _update_powerups(self, dt: float) -> None:
        """Update active power-up effects."""
        # Update all effects
        self.active_effects = [e for e in self.active_effects if e.update(dt)]
        
        # Check which effects are still active
        active_types = {e.effect_type for e in self.active_effects}
        
        self.speed_boost_active = "speed_boost" in active_types
        self.shield_active = "shield" in active_types
        self.double_damage_active = "double_damage" in active_types
    
    def _get_max_speed(self) -> float:
        """Get current maximum speed (modified by power-ups)."""
        speed = self.max_speed
        if self.speed_boost_active:
            speed *= 1.5
        return speed
    
    def pass_checkpoint(self, checkpoint_index: int) -> bool:
        """
        Register checkpoint passage.
        
        Args:
            checkpoint_index: Index of passed checkpoint.
        
        Returns:
            True if this completes a lap, False otherwise.
        """
        # Check if this is the next expected checkpoint
        if checkpoint_index == self.checkpoint_index:
            self.checkpoint_index += 1
            logger.debug(f"Car {self.agent_id} passed checkpoint {checkpoint_index}")
            return False
        
        return False
    
    def complete_lap(self, num_checkpoints: int) -> None:
        """Register lap completion."""
        lap_time = self.race_time - self.lap_start_time
        self.lap_times.append(lap_time)
        self.current_lap += 1
        self.checkpoint_index = 0
        self.lap_start_time = self.race_time
        
        logger.info(
            f"Car {self.agent_id} completed lap {self.current_lap} "
            f"in {lap_time:.2f}s"
        )
    
    def finish_race(self) -> None:
        """Register race finish."""
        self.state = CarState.FINISHED
        logger.success(f"Car {self.agent_id} finished race!")
    
    def respawn(
        self,
        position: Tuple[float, float],
        rotation: float = 0.0,
    ) -> None:
        """
        Respawn car at position.
        
        Args:
            position: Respawn position (x, y).
            rotation: Respawn rotation in radians.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.rotation = rotation
        self.angular_velocity = 0.0
        
        self.health = self.max_health
        self.is_alive = True
        self.state = CarState.ACTIVE
        
        # Clear effects
        self.active_effects.clear()
        self.shield_active = False
        self.speed_boost_active = False
        self.double_damage_active = False
        
        logger.info(f"Car {self.agent_id} respawned")
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get state representation for RL agent.
        
        Returns:
            Normalized state vector.
        """
        # Normalize values
        norm_health = self.health / self.max_health
        norm_speed = np.linalg.norm(self.velocity) / self.max_speed
        norm_rotation = self.rotation / math.pi
        
        # Ammo ratios
        laser_ratio = self.weapons["laser"]["ammo"] / 50
        missile_ratio = self.weapons["missile"]["ammo"] / 20
        mine_ratio = self.weapons["mine"]["ammo"] / 10
        
        # Power-up status
        powerup_status = [
            float(self.speed_boost_active),
            float(self.shield_active),
            float(self.double_damage_active),
        ]
        
        state = [
            *self.position / 1000.0,  # Normalized position
            *self.velocity / self.max_speed,  # Normalized velocity
            norm_rotation,
            self.angular_velocity,
            norm_health,
            norm_speed,
            laser_ratio,
            missile_ratio,
            mine_ratio,
            *powerup_status,
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get car's corner vertices (for rendering)."""
        w, h = self.collision_size
        corners = [
            (-w/2, -h/2),
            (w/2, -h/2),
            (w/2, h/2),
            (-w/2, h/2),
        ]
        
        # Rotate and translate
        vertices = []
        for corner in corners:
            rotated = rotate_point(corner, self.rotation)
            vertex = (
                self.position[0] + rotated[0],
                self.position[1] + rotated[1],
            )
            vertices.append(vertex)
        
        return vertices
    
    def __repr__(self) -> str:
        return (
            f"Car(id={self.agent_id}, pos={tuple(self.position)}, "
            f"health={self.health:.1f}, state={self.state.value})"
        )
