"""
Q-Learning Agent
===============

Tabular Q-Learning with epsilon-greedy exploration.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from .base_agent import BaseAgent
from ...utils.logger import get_logger

logger = get_logger(__name__)


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.
    
    Features:
        - Q-table with state hashing
        - Epsilon-greedy exploration
        - Decaying exploration rate
        - State discretization
    
    Algorithm:
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        discretization_bins: int = 10,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            learning_rate: Learning rate (α).
            discount_factor: Discount factor (γ).
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Exploration decay rate.
            discretization_bins: Number of bins for state discretization.
            config: Configuration dictionary.
        """
        super().__init__(state_dim, action_dim, config)
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discretization_bins = discretization_bins
        
        # Q-table (state -> action -> Q-value)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim)
        )
        
        # Discretization bounds (learned from experience)
        self.state_mins = np.full(state_dim, np.inf)
        self.state_maxs = np.full(state_dim, -np.inf)
        
        logger.info(
            f"QLearningAgent initialized: lr={learning_rate}, "
            f"gamma={discount_factor}, epsilon={epsilon_start}"
        )
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state into bins.
        
        Args:
            state: Continuous state vector.
        
        Returns:
            Tuple of discretized state values.
        """
        # Update bounds
        self.state_mins = np.minimum(self.state_mins, state)
        self.state_maxs = np.maximum(self.state_maxs, state)
        
        # Avoid division by zero
        ranges = self.state_maxs - self.state_mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        
        # Normalize to [0, 1]
        normalized = (state - self.state_mins) / ranges
        normalized = np.clip(normalized, 0, 1)
        
        # Discretize into bins
        discrete = (normalized * (self.discretization_bins - 1)).astype(int)
        
        return tuple(discrete)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            deterministic: If True, always select best action.
        
        Returns:
            Selected action.
        """
        discrete_state = self._discretize_state(state)
        
        # Epsilon-greedy
        if not deterministic and self.training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.action_dim)
        else:
            # Exploit: best action
            q_values = self.q_table[discrete_state]
            action = int(np.argmax(q_values))
        
        return action
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, float]:
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        
        Returns:
            Dictionary of training metrics.
        """
        # Discretize states
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.q_table[discrete_next_state]
            target_q = reward + self.gamma * np.max(next_q_values)
        
        # TD error
        td_error = target_q - current_q
        
        # Update Q-value
        self.q_table[discrete_state][action] += self.lr * td_error
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        
        # Update statistics
        self.total_steps += 1
        
        return {
            "q_value": current_q,
            "td_error": td_error,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Current state.
        
        Returns:
            Array of Q-values for each action.
        """
        discrete_state = self._discretize_state(state)
        return self.q_table[discrete_state].copy()
    
    def save(self, path: Path) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save location.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for pickling
        save_dict = {
            "q_table": dict(self.q_table),
            "state_mins": self.state_mins,
            "state_maxs": self.state_maxs,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "discretization_bins": self.discretization_bins,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load from.
        """
        path = Path(path)
        
        if not path.exists():
            logger.error(f"No saved agent found at {path}")
            raise FileNotFoundError(f"No saved agent found at {path}")
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_dim),
            save_dict["q_table"]
        )
        
        self.state_mins = save_dict["state_mins"]
        self.state_maxs = save_dict["state_maxs"]
        self.epsilon = save_dict["epsilon"]
        self.total_steps = save_dict["total_steps"]
        self.episodes = save_dict["episodes"]
        
        logger.info(f"Agent loaded from {path} (steps={self.total_steps})")
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = super().get_stats()
        stats.update({
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "learning_rate": self.lr,
        })
        return stats
