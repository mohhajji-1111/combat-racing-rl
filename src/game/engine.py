"""
Game Engine
==========

Main game loop and entity management.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

from .track import Track
from .entities.car import Car, CarState
from .entities.projectile import Projectile, Laser, Missile, Mine
from .entities.powerup import PowerUp, PowerUpType
from .physics import PhysicsEngine
from .renderer import Renderer
from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class GameEngine:
    """
    Main game engine managing all entities and game logic.
    
    Features:
        - Entity management (cars, projectiles, power-ups)
        - Physics simulation
        - Collision detection
        - Game rules (laps, checkpoints, combat)
        - Rendering (optional)
        - Game state management
    """
    
    def __init__(
        self,
        track: Track,
        num_cars: int = 4,
        config_path: Optional[Path] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize game engine.
        
        Args:
            track: Racing track.
            num_cars: Number of cars.
            config_path: Path to game configuration.
            render_mode: Rendering mode ('human', 'rgb_array', or None).
        """
        # Load configuration
        if config_path:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.get_config()
        else:
            self.config = {}
        
        self.track = track
        self.num_cars = num_cars
        self.render_mode = render_mode
        
        # Physics engine
        self.physics = PhysicsEngine(
            width=int(track.width),
            height=int(track.height),
        )
        
        # Entities
        self.cars: List[Car] = []
        self.projectiles: List[Projectile] = []
        self.powerups: List[PowerUp] = []
        
        # Game state
        self.current_step = 0
        self.episode = 0
        self.running = False
        self.max_laps = self.config.get("game", {}).get("max_laps", 3)
        self.max_steps = self.config.get("game", {}).get("max_steps_per_episode", 10000)
        
        # Renderer (optional)
        self.renderer: Optional[Renderer] = None
        if render_mode == "human":
            self.renderer = Renderer(
                width=int(track.width),
                height=int(track.height),
                fps=60
            )
        
        # Initialize entities
        self._initialize_cars()
        self._spawn_initial_powerups()
        
        logger.info(
            f"GameEngine initialized: {num_cars} cars, "
            f"{len(self.powerups)} powerups, render_mode={render_mode}"
        )
    
    def _initialize_cars(self) -> None:
        """Initialize cars at starting positions."""
        self.cars.clear()
        
        car_config = self.config.get("car", {})
        
        for i in range(self.num_cars):
            # Get start position
            start_pos, start_angle = self.track.get_start_position(i)
            
            # Create car configuration
            car_conf = {
                "size": [car_config.get("width", 40), car_config.get("height", 30)],
                "mass": car_config.get("mass", 1.0),
                "max_speed": car_config.get("max_speed", 400.0),
                "acceleration": car_config.get("acceleration", 200.0),
                "max_health": car_config.get("max_health", 100.0),
            }
            
            # Create car
            car = Car(
                position=start_pos.copy(),
                rotation=start_angle,
                config=car_conf,
                agent_id=i,
            )
            
            self.cars.append(car)
            self.physics.add_body(car)
        
        logger.debug(f"Initialized {len(self.cars)} cars")
    
    def _spawn_initial_powerups(self) -> None:
        """Spawn initial power-ups on track."""
        powerup_config = self.config.get("powerups", {})
        num_powerups = powerup_config.get("initial_count", 5)
        
        for _ in range(num_powerups):
            self._spawn_powerup()
    
    def _spawn_powerup(self) -> None:
        """Spawn single power-up at random position."""
        position = self.track.get_random_powerup_position()
        
        if position is None:
            return
        
        # Random power-up type
        powerup_type = np.random.choice(list(PowerUpType))
        
        # Get powerup config
        powerup_config = self.config.get("powerups", {})
        
        powerup = PowerUp(
            position=position,
            powerup_type=powerup_type,
            config=powerup_config,
        )
        
        self.powerups.append(powerup)
        self.physics.add_body(powerup)
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset game to initial state.
        
        Returns:
            Initial observations for all cars.
        """
        self.current_step = 0
        self.running = True
        
        # Reset entities
        self._initialize_cars()
        
        self.projectiles.clear()
        self.powerups.clear()
        self._spawn_initial_powerups()
        
        logger.debug(f"Game reset for episode {self.episode}")
        self.episode += 1
        
        # Return observations
        return [self._get_car_observation(car) for car in self.cars]
    
    def step(
        self,
        actions: List[int],
        dt: float = 1/60,
    ) -> tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Execute one game step.
        
        Args:
            actions: List of actions for each car.
            dt: Time delta in seconds.
        
        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        self.current_step += 1
        
        # Apply actions
        for car, action in zip(self.cars, actions):
            if car.state == CarState.ACTIVE:
                self._apply_action(car, action, dt)
        
        # Update entities
        self._update_cars(dt)
        self._update_projectiles(dt)
        self._update_powerups(dt)
        
        # Check collisions
        self._check_collisions()
        
        # Check checkpoints
        self._check_checkpoints()
        
        # Compute rewards and dones
        observations = []
        rewards = []
        dones = []
        
        for car in self.cars:
            obs = self._get_car_observation(car)
            reward = self._compute_reward(car)
            done = self._is_done(car)
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
        
        # Check episode end
        episode_done = (
            all(dones) or
            self.current_step >= self.max_steps or
            any(car.current_lap >= self.max_laps for car in self.cars)
        )
        
        if episode_done:
            self.running = False
        
        # Info
        info = {
            "episode": self.episode,
            "step": self.current_step,
            "cars_active": sum(1 for car in self.cars if car.state == CarState.ACTIVE),
        }
        
        return observations, rewards, dones, info
    
    def _apply_action(self, car: Car, action: int, dt: float) -> None:
        """
        Apply action to car.
        
        Action space (12 actions):
            0: No-op
            1: Forward
            2: Backward
            3: Turn Left
            4: Turn Right
            5: Forward + Left
            6: Forward + Right
            7: Fire Laser
            8: Fire Missile
            9: Drop Mine
            10: Forward + Fire Laser
            11: Forward + Fire Missile
        """
        # Movement
        if action in [1, 5, 6, 10, 11]:
            car.accelerate(dt)
        elif action == 2:
            car.brake(dt)
        
        if action in [3, 5]:
            car.turn_left(dt)
        elif action in [4, 6]:
            car.turn_right(dt)
        
        # Weapons
        if action in [7, 10] and car.can_fire("laser"):
            projectile = car.fire_weapon("laser")
            if projectile:
                self.projectiles.append(projectile)
                self.physics.add_body(projectile)
        
        elif action in [8, 11] and car.can_fire("missile"):
            projectile = car.fire_weapon("missile")
            if projectile:
                self.projectiles.append(projectile)
                self.physics.add_body(projectile)
        
        elif action == 9 and car.can_fire("mine"):
            projectile = car.fire_weapon("mine")
            if projectile:
                self.projectiles.append(projectile)
                self.physics.add_body(projectile)
    
    def _update_cars(self, dt: float) -> None:
        """Update car states."""
        for car in self.cars:
            if car.state == CarState.ACTIVE:
                car.update(dt)
                
                # Check wall collisions
                collision, normal = self.track.check_wall_collision(
                    car.position,
                    car.collision_radius
                )
                
                if collision:
                    # Bounce off wall
                    car.velocity = car.velocity - 2 * np.dot(car.velocity, normal) * normal
                    car.velocity *= 0.5  # Damping
                    car.take_damage(10)  # Wall damage
    
    def _update_projectiles(self, dt: float) -> None:
        """Update projectile states."""
        for projectile in self.projectiles[:]:
            if not projectile.is_active:
                self.projectiles.remove(projectile)
                self.physics.remove_body(projectile)
                continue
            
            projectile.update(dt)
            
            # Check wall collisions
            collision, _ = self.track.check_wall_collision(
                projectile.position,
                projectile.collision_radius
            )
            
            if collision:
                projectile.is_active = False
    
    def _update_powerups(self, dt: float) -> None:
        """Update power-ups."""
        for powerup in self.powerups[:]:
            if powerup.is_collected:
                self.powerups.remove(powerup)
                self.physics.remove_body(powerup)
                
                # Respawn after delay
                if np.random.random() < 0.01:  # 1% chance per frame
                    self._spawn_powerup()
    
    def _check_collisions(self) -> None:
        """Check collisions between entities."""
        # Car-projectile collisions
        for car in self.cars:
            if car.state != CarState.ACTIVE:
                continue
            
            for projectile in self.projectiles:
                if not projectile.active or projectile.owner == car:
                    continue
                
                # Check collision
                dist = np.linalg.norm(car.position - projectile.position)
                if dist < (car.collision_radius + projectile.collision_radius):
                    # Hit!
                    damage = projectile.damage
                    if car.shield_active:
                        damage *= 0.5  # Shield reduces damage
                    
                    car.take_damage(damage)
                    projectile.is_active = False
                    
                    # Reward shooter
                    if projectile.owner:
                        projectile.owner.stats["hits"] += 1
                        if car.state != CarState.ACTIVE:
                            projectile.owner.stats["kills"] += 1
        
        # Car-powerup collisions
        for car in self.cars:
            if car.state != CarState.ACTIVE:
                continue
            
            for powerup in self.powerups:
                if powerup.is_collected:
                    continue
                
                dist = np.linalg.norm(car.position - powerup.position)
                if dist < (car.collision_radius + powerup.collision_radius):
                    # Collect powerup - TODO: Implement collect_powerup method in Car class
                    # car.collect_powerup(powerup)
                    powerup.is_collected = True
    
    def _check_checkpoints(self) -> None:
        """Check checkpoint crossings."""
        for car in self.cars:
            if car.state != CarState.ACTIVE:
                continue
            
            # Check next checkpoint (simplified - just checking distance)
            next_checkpoint_idx = car.checkpoint_index % len(self.track.checkpoints)
            next_checkpoint = self.track.checkpoints[next_checkpoint_idx]
            
            dist_to_checkpoint = np.linalg.norm(car.position - next_checkpoint.position)
            
            # If close enough to checkpoint, consider it crossed
            if dist_to_checkpoint < 50.0:  # Checkpoint radius
                car.checkpoint_index += 1
                
                # Check lap completion
                if car.checkpoint_index >= len(self.track.checkpoints):
                    car.current_lap += 1
                    car.checkpoint_index = 0
                    logger.debug(f"Car completed lap {car.current_lap}")
    
    def _get_car_observation(self, car: Car) -> np.ndarray:
        """Get observation for car (used by RL agent)."""
        return car.get_state_vector()
    
    def _compute_reward(self, car: Car) -> float:
        """Compute reward for car."""
        reward = 0.0
        
        # Checkpoint rewards
        if car.checkpoint_index != car.stats.get("last_checkpoint", 0):
            reward += 10.0
            car.stats["last_checkpoint"] = car.checkpoint_index
        
        # Lap completion
        if car.current_lap != car.stats.get("last_lap", 0):
            reward += 100.0
            car.stats["last_lap"] = car.current_lap
        
        # Speed reward (encourage fast driving)
        speed = np.linalg.norm(car.velocity)
        reward += speed * 0.01
        
        # Combat rewards
        if car.stats.get("hits", 0) != car.stats.get("last_hits", 0):
            reward += 30.0
            car.stats["last_hits"] = car.stats.get("hits", 0)
        
        if car.stats.get("kills", 0) != car.stats.get("last_kills", 0):
            reward += 100.0
            car.stats["last_kills"] = car.stats.get("kills", 0)
        
        # Damage penalty
        if car.health < car.stats.get("last_health", car.max_health):
            reward -= 10.0
            car.stats["last_health"] = car.health
        
        # Death penalty
        if car.state != CarState.ACTIVE:
            reward -= 100.0
        
        return reward
    
    def _is_done(self, car: Car) -> bool:
        """Check if episode is done for car."""
        return (
            car.state != CarState.ACTIVE or
            car.current_lap >= self.max_laps or
            self.current_step >= self.max_steps
        )
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render current game state.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.renderer is None:
            return None
        
        # Handle events
        if not self.renderer.handle_events():
            self.running = False
            return None
        
        # Render frame
        frame_info = {
            "Episode": self.episode,
            "Step": self.current_step,
        }
        
        self.renderer.render_frame(
            self.track,
            self.cars,
            self.projectiles,
            self.powerups,
            frame_info
        )
        
        if self.render_mode == "rgb_array":
            # Capture frame as RGB array
            import pygame
            rgb_array = pygame.surfarray.array3d(self.renderer.screen)
            return rgb_array.transpose((1, 0, 2))  # Correct orientation
        
        return None
    
    def close(self) -> None:
        """Close game engine and cleanup."""
        if self.renderer:
            self.renderer.close()
        
        logger.info("GameEngine closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get game statistics."""
        return {
            "episode": self.episode,
            "step": self.current_step,
            "num_cars": len(self.cars),
            "cars_active": sum(1 for car in self.cars if car.state == CarState.ACTIVE),
            "num_projectiles": len(self.projectiles),
            "num_powerups": len(self.powerups),
        }
