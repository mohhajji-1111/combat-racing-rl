"""
Base Agent Class
===============

Abstract base class for all RL agents.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from pathlib import Path

from ...utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    
    All agents must implement:
        - select_action: Choose action given state
        - update: Update agent from experience
        - save: Save agent to disk
        - load: Load agent from disk
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            config: Configuration dictionary.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Training state
        self.training = True
        self.total_steps = 0
        self.episodes = 0
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}"
        )
    
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action given state.
        
        Args:
            state: Current state.
            deterministic: If True, select best action (no exploration).
        
        Returns:
            Selected action.
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """
        Update agent from experience.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        
        Returns:
            Dictionary of training metrics.
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save location.
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load from.
        """
        pass
    
    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.training = True
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary of statistics.
        """
        stats = {
            "total_steps": self.total_steps,
            "episodes": self.episodes,
        }
        
        if self.episode_rewards:
            stats.update({
                "mean_reward": np.mean(self.episode_rewards[-100:]),
                "max_reward": np.max(self.episode_rewards),
                "min_reward": np.min(self.episode_rewards),
            })
        
        if self.episode_lengths:
            stats.update({
                "mean_length": np.mean(self.episode_lengths[-100:]),
            })
        
        return stats
    
    def reset_episode_stats(self) -> None:
        """Reset episode statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"steps={self.total_steps})"
        )
