"""
Experience Replay Buffer
=======================

Replay buffers for DQN and PPO.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions (s, a, r, s', done) and samples random batches.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size.
            state_dim: Dimension of state space.
            device: Device to store tensors ('cpu' or 'cuda').
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add transition to buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors.
        """
        # Random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to TD error.
    Paper: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size.
            state_dim: Dimension of state space.
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized).
            beta_start: Initial importance sampling weight.
            beta_end: Final importance sampling weight.
            device: Device to store tensors.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Priorities
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.position = 0
        self.size = 0
        self.total_steps = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with maximum priority."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # New transitions get max priority
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_steps += 1
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized probabilities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices).
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Compute importance sampling weights
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Convert to tensors
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions.
            priorities: New priorities (TD errors).
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def _get_beta(self) -> float:
        """Get current beta value (annealed)."""
        progress = min(1.0, self.total_steps / 1e6)  # Anneal over 1M steps
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """
    Rollout buffer for PPO.
    
    Stores complete trajectories with advantages and returns.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize rollout buffer.
        
        Args:
            capacity: Maximum buffer size.
            state_dim: Dimension of state space.
            device: Device to store tensors.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        
        # Storage
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        
        # Computed during finalize
        self.advantages: List[float] = []
        self.returns: List[float] = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add transition to buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            value: Value estimate V(s).
            log_prob: Log probability of action.
            done: Whether episode is done.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def finalize(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute advantages and returns using GAE.
        
        Args:
            last_value: Value of final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        # Convert to numpy
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data as tensors.
        
        Returns:
            Dictionary of tensors.
        """
        data = {
            "states": torch.FloatTensor(np.array(self.states)).to(self.device),
            "actions": torch.LongTensor(self.actions).to(self.device),
            "old_log_probs": torch.FloatTensor(self.log_probs).to(self.device),
            "advantages": torch.FloatTensor(self.advantages).to(self.device),
            "returns": torch.FloatTensor(self.returns).to(self.device),
        }
        return data
    
    def clear(self) -> None:
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self) -> int:
        return len(self.states)
