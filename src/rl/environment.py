"""
Combat Racing Gymnasium Environment
===================================

OpenAI Gym compatible environment for the racing game.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..game.engine import GameEngine
from ..game.entities import Car
from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class CombatRacingEnv(gym.Env):
    """
    Gymnasium environment for Combat Racing Championship.
    
    Observation Space:
        - Car state (position, velocity, rotation, health, ammo)
        - Track sensors (ray distances to walls)
        - Checkpoint information
        - Opponent information (relative positions, health)
        
    Action Space:
        Discrete(12):
            0: No-op
            1: Forward
            2: Backward
            3: Left
            4: Right
            5: Forward + Left
            6: Forward + Right
            7: Backward + Left
            8: Backward + Right
            9: Shoot Laser
            10: Shoot Missile
            11: Drop Mine
    
    Reward:
        - Checkpoint passage: +10
        - Lap completion: +100
        - Race finish: +500
        - Speed reward: +0.01 * speed
        - Hit opponent: +30
        - Eliminate opponent: +100
        - Got hit: -10
        - Death: -100
        - Wall collision: -50
        - Timestep penalty: -0.1
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
        num_opponents: int = 3,
        track_name: str = "medium",
        enable_combat: bool = True,
    ):
        """
        Initialize environment.
        
        Args:
            config: Configuration dictionary.
            render_mode: Render mode ("human", "rgb_array", None).
            num_opponents: Number of opponent cars.
            track_name: Track to use.
            enable_combat: Enable combat mechanics.
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            config_loader = ConfigLoader()
            game_config = config_loader.load("config/game_config.yaml")
            rl_config = config_loader.load("config/rl_config.yaml")
            self.config = config_loader.merge(game_config, rl_config)
        else:
            self.config = config
        
        self.render_mode = render_mode
        self.num_opponents = num_opponents
        self.track_name = track_name
        self.enable_combat = enable_combat
        
        # Create track
        from ..game.track import create_oval_track
        track = create_oval_track()
        
        # Create game engine
        self.game = GameEngine(
            track=track,
            num_cars=num_opponents + 1,
            render_mode=render_mode
        )
        
        # Environment state
        self.agent_car: Optional[Car] = None
        self.opponent_cars: List[Car] = []
        self.episode_steps = 0
        self.max_episode_steps = self.config.get("game", {}).get("max_episode_length", 5000)
        
        # Previous state for rewards
        self.prev_checkpoint_index = 0
        self.prev_lap = 0
        self.prev_position = np.array([0.0, 0.0])
        
        # Ray sensors for track perception
        self.num_rays = 8
        self.ray_length = 200.0
        self.ray_angles = np.linspace(-np.pi/2, np.pi/2, self.num_rays)
        
        # Define observation space
        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Define action space (discrete)
        self.action_space = spaces.Discrete(12)
        
        # Action mapping
        self.action_map = {
            0: ("noop", None),
            1: ("throttle", 1.0),
            2: ("throttle", -1.0),
            3: ("steering", -1.0),
            4: ("steering", 1.0),
            5: ("both", (1.0, -1.0)),  # Forward + Left
            6: ("both", (1.0, 1.0)),   # Forward + Right
            7: ("both", (-1.0, -1.0)), # Backward + Left
            8: ("both", (-1.0, 1.0)),  # Backward + Right
            9: ("shoot", "laser"),
            10: ("shoot", "missile"),
            11: ("shoot", "mine"),
        }
        
        logger.info(
            f"CombatRacingEnv initialized: "
            f"obs_size={obs_size}, actions=12, "
            f"opponents={num_opponents}"
        )
    
    def _calculate_obs_size(self) -> int:
        """Calculate observation vector size."""
        size = 0
        size += 2  # Position (x, y)
        size += 2  # Velocity (vx, vy)
        size += 1  # Rotation
        size += 1  # Angular velocity
        size += 1  # Speed
        size += 1  # Health
        size += 3  # Ammo (laser, missile, mine)
        size += 3  # Power-up status (speed, shield, damage)
        size += self.num_rays  # Track sensors
        size += 2  # Checkpoint direction (x, y)
        size += 1  # Checkpoint distance
        size += self.num_opponents * 5  # Opponent info (x, y, vx, vy, health)
        return size
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
        
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game (this creates cars internally)
        self.game.reset()
        
        # Assign first car as agent, rest as opponents
        if len(self.game.cars) > 0:
            self.agent_car = self.game.cars[0]
            self.opponent_cars = self.game.cars[1:self.num_opponents+1]
        else:
            raise RuntimeError("GameEngine did not create any cars")
        
        # Reset tracking variables
        self.episode_steps = 0
        self.prev_checkpoint_index = 0
        self.prev_lap = 0
        self.prev_position = self.agent_car.position.copy()
        
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug("Environment reset")
        return observation, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return results.
        
        Args:
            action: Action index.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply action to agent car
        self._apply_action(action)
        
        # Apply random actions to opponents (placeholder)
        for opponent in self.opponent_cars:
            if opponent.is_alive:
                self._apply_opponent_action(opponent)
        
        # Update game
        dt = 1.0 / 60.0  # 60 FPS
        self.game.step([], dt)  # Empty actions list since we're applying actions separately
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = not self.agent_car.is_alive or self.agent_car.current_lap >= 3
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Update tracking
        self.episode_steps += 1
        self.prev_checkpoint_index = self.agent_car.checkpoint_index
        self.prev_lap = self.agent_car.current_lap
        self.prev_position = self.agent_car.position.copy()
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: int) -> None:
        """Apply action to agent car."""
        if action not in self.action_map:
            return
        
        action_type, action_value = self.action_map[action]
        
        if action_type == "noop":
            pass
        elif action_type == "throttle":
            self.agent_car.apply_throttle(action_value)
        elif action_type == "steering":
            self.agent_car.apply_steering(action_value)
        elif action_type == "both":
            throttle, steering = action_value
            self.agent_car.apply_throttle(throttle)
            self.agent_car.apply_steering(steering)
        elif action_type == "shoot" and self.enable_combat:
            # Shooting weapons - handle differently as combat system needs further integration
            pass  # TODO: Implement combat system
    
    def _apply_opponent_action(self, opponent: Car) -> None:
        """Apply random action to opponent (placeholder for AI)."""
        # Simple random policy
        action = np.random.choice([1, 5, 6])  # Mostly forward
        action_type, action_value = self.action_map[action]
        
        if action_type == "throttle":
            opponent.apply_throttle(action_value)
        elif action_type == "both":
            throttle, steering = action_value
            opponent.apply_throttle(throttle)
            opponent.apply_steering(steering)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Normalized observation vector.
        """
        if self.agent_car is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = []
        
        # Car state (normalized)
        pos = np.array(self.agent_car.position, dtype=np.float32)
        vel = np.array(self.agent_car.velocity, dtype=np.float32)
        obs.append(float(pos[0]) / 1000.0)
        obs.append(float(pos[1]) / 1000.0)
        obs.append(float(vel[0]) / self.agent_car.max_speed)
        obs.append(float(vel[1]) / self.agent_car.max_speed)
        obs.append(float(self.agent_car.rotation / np.pi))
        obs.append(float(self.agent_car.angular_velocity))
        obs.append(float(np.linalg.norm(self.agent_car.velocity) / self.agent_car.max_speed))
        obs.append(float(self.agent_car.health / self.agent_car.max_health))
        
        # Ammo (normalized)
        obs.append(float(self.agent_car.weapons["laser"]["ammo"]) / 50.0)
        obs.append(float(self.agent_car.weapons["missile"]["ammo"]) / 20.0)
        obs.append(float(self.agent_car.weapons["mine"]["ammo"]) / 10.0)
        
        # Power-up status
        obs.append(float(self.agent_car.speed_boost_active))
        obs.append(float(self.agent_car.shield_active))
        obs.append(float(self.agent_car.double_damage_active))
        
        # Track sensors (ray casting)
        ray_distances = self._cast_rays()
        for dist in ray_distances:
            obs.append(float(dist) / self.ray_length)
        
        # Checkpoint information
        checkpoint_idx = self.agent_car.checkpoint_index % len(self.game.track.checkpoints)
        checkpoint = self.game.track.checkpoints[checkpoint_idx]
        checkpoint_pos = np.array(checkpoint.position, dtype=np.float32)
        checkpoint_dir = checkpoint_pos - np.array(self.agent_car.position, dtype=np.float32)
        checkpoint_dist = float(np.linalg.norm(checkpoint_dir))
        
        if checkpoint_dist > 0:
            checkpoint_dir = checkpoint_dir / checkpoint_dist
        else:
            checkpoint_dir = np.array([0.0, 0.0], dtype=np.float32)
        
        obs.append(float(checkpoint_dir[0]))
        obs.append(float(checkpoint_dir[1]))
        obs.append(float(checkpoint_dist) / 1000.0)
        
        # Opponent information (relative)
        for opponent in self.opponent_cars[:self.num_opponents]:
            if opponent.is_alive:
                rel_pos = np.array(opponent.position, dtype=np.float32) - np.array(self.agent_car.position, dtype=np.float32)
                rel_vel = np.array(opponent.velocity, dtype=np.float32) - np.array(self.agent_car.velocity, dtype=np.float32)
                obs.append(float(rel_pos[0]) / 1000.0)
                obs.append(float(rel_pos[1]) / 1000.0)
                obs.append(float(rel_vel[0]) / self.agent_car.max_speed)
                obs.append(float(rel_vel[1]) / self.agent_car.max_speed)
                obs.append(float(opponent.health / opponent.max_health))
            else:
                obs.extend([0.0] * 5)
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self) -> np.ndarray:
        """
        Cast rays to detect track walls.
        
        Returns:
            Array of ray distances.
        """
        distances = []
        
        for angle_offset in self.ray_angles:
            ray_angle = self.agent_car.rotation + angle_offset
            
            hit = self.game.physics.raycast(
                origin=tuple(self.agent_car.position),
                direction=ray_angle,
                max_distance=self.ray_length,
            )
            
            if hit:
                _, _, distance = hit
                distances.append(distance)
            else:
                distances.append(self.ray_length)
        
        return np.array(distances)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current step."""
        reward = 0.0
        
        # Checkpoint reward
        if self.agent_car.checkpoint_index > self.prev_checkpoint_index:
            reward += 10.0
            logger.debug(f"Checkpoint passed! Reward: +10")
        
        # Lap completion reward
        if self.agent_car.current_lap > self.prev_lap:
            reward += 100.0
            logger.info(f"Lap completed! Reward: +100")
        
        # Speed reward
        speed = np.linalg.norm(self.agent_car.velocity)
        reward += 0.01 * speed
        
        # Progress reward (moving toward checkpoint)
        checkpoint_idx = self.agent_car.checkpoint_index % len(self.game.track.checkpoints)
        checkpoint = self.game.track.checkpoints[checkpoint_idx]
        checkpoint_pos = checkpoint.position
        current_dist = np.linalg.norm(checkpoint_pos - self.agent_car.position)
        prev_dist = np.linalg.norm(checkpoint_pos - self.prev_position)
        progress = prev_dist - current_dist
        reward += 0.1 * progress
        
        # Combat rewards
        damage_delta = self.agent_car.stats["damage_dealt"] - getattr(self, "_prev_damage_dealt", 0.0)
        if damage_delta > 0:
            reward += 30.0
        self._prev_damage_dealt = self.agent_car.stats["damage_dealt"]
        
        # Penalty for taking damage
        damage_taken_delta = self.agent_car.stats["damage_taken"] - getattr(self, "_prev_damage_taken", 0.0)
        if damage_taken_delta > 0:
            reward -= 10.0
        self._prev_damage_taken = self.agent_car.stats["damage_taken"]
        
        # Death penalty
        if not self.agent_car.is_alive:
            reward -= 100.0
            logger.info(f"Agent died! Penalty: -100")
        
        # Collision penalty
        collision_delta = self.agent_car.stats["collisions"] - getattr(self, "_prev_collisions", 0)
        if collision_delta > 0:
            reward -= 50.0
        self._prev_collisions = self.agent_car.stats["collisions"]
        
        # Timestep penalty (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            "episode_steps": self.episode_steps,
            "lap": self.agent_car.current_lap if self.agent_car else 0,
            "checkpoint": self.agent_car.checkpoint_index if self.agent_car else 0,
            "health": self.agent_car.health if self.agent_car else 0,
            "speed": np.linalg.norm(self.agent_car.velocity) if self.agent_car else 0,
            "is_alive": self.agent_car.is_alive if self.agent_car else False,
            "stats": self.agent_car.stats if self.agent_car else {},
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        return self.game.render()
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'game'):
            self.game.close()
        logger.info("Environment closed")
