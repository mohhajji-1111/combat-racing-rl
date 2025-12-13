"""
PPO Agent
=========

Proximal Policy Optimization with clipped objective.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .base_agent import BaseAgent
from ..networks import ActorCriticNetwork
from ..replay_buffer import RolloutBuffer
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent.
    
    Features:
        - Actor-Critic architecture
        - Clipped surrogate objective
        - Generalized Advantage Estimation (GAE)
        - Multiple epochs per rollout
        - Value function clipping
    
    Paper: "Proximal Policy Optimization" (Schulman et al., 2017)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        rollout_length: int = 2048,
        batch_size: int = 64,
        epochs: int = 10,
        hidden_dims: tuple = (256, 256),
        device: str = "auto",
        config: Optional[Dict] = None,
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            learning_rate: Learning rate for optimizer.
            discount_factor: Discount factor (γ).
            gae_lambda: GAE lambda parameter.
            clip_ratio: PPO clip ratio (ε).
            value_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Max gradient norm for clipping.
            rollout_length: Length of rollout buffer.
            batch_size: Mini-batch size.
            epochs: Number of optimization epochs per rollout.
            hidden_dims: Hidden layer dimensions.
            device: Device ('cpu', 'cuda', or 'auto').
            config: Configuration dictionary.
        """
        super().__init__(state_dim, action_dim, config)
        
        # Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            rollout_length, state_dim, device=self.device
        )
        
        # Statistics
        self.training_steps = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        logger.info(
            f"PPOAgent initialized: device={self.device}, "
            f"rollout_length={rollout_length}"
        )
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action from policy.
        
        Args:
            state: Current state.
            deterministic: If True, select best action.
        
        Returns:
            Selected action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, _ = self.network(state_tensor)
            
            if deterministic:
                # Greedy action
                action = action_logits.argmax(dim=1).item()
            else:
                # Sample from policy
                probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        
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
        Add transition to rollout buffer.
        
        When buffer is full, train on collected data.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        
        Returns:
            Dictionary of training metrics (empty if not training).
        """
        # Get value and log prob for the state-action pair
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            
            _, log_prob, _, value = self.network.get_action_and_value(
                state_tensor, action_tensor
            )
            
            value = value.item()
            log_prob = log_prob.item()
        
        # Add to buffer
        self.rollout_buffer.add(
            state, action, reward, value, log_prob, done
        )
        
        self.total_steps += 1
        
        # Train when buffer is full
        if len(self.rollout_buffer) >= self.rollout_length:
            return self._train()
        
        return {}
    
    def _train(self) -> Dict[str, float]:
        """
        Train on collected rollout.
        
        Returns:
            Dictionary of training metrics.
        """
        # Get last value for GAE
        # Use zero if episode is done, otherwise bootstrap
        last_state = self.rollout_buffer.states[-1]
        last_done = self.rollout_buffer.dones[-1]
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_value = self.network(state_tensor)
            last_value = last_value.item() * (1 - last_done)
        
        # Finalize buffer (compute advantages and returns)
        self.rollout_buffer.finalize(last_value, self.gamma, self.gae_lambda)
        
        # Get data
        data = self.rollout_buffer.get()
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies_list = []
        
        # Multiple epochs
        for epoch in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(len(self.rollout_buffer))
            
            # Mini-batch updates
            for start in range(0, len(self.rollout_buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_states = data["states"][batch_indices]
                batch_actions = data["actions"][batch_indices]
                batch_old_log_probs = data["old_log_probs"][batch_indices]
                batch_advantages = data["advantages"][batch_indices]
                batch_returns = data["returns"][batch_indices]
                
                # Normalize advantages
                batch_advantages = (
                    (batch_advantages - batch_advantages.mean()) /
                    (batch_advantages.std() + 1e-8)
                )
                
                # Forward pass
                _, log_probs, entropies, values = \
                    self.network.get_action_and_value(batch_states, batch_actions)
                
                values = values.squeeze(-1)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE with optional clipping)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropies.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies_list.append(entropies.mean().item())
                
                self.training_steps += 1
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Statistics
        self.policy_losses.extend(policy_losses)
        self.value_losses.extend(value_losses)
        self.entropies.extend(entropies_list)
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies_list),
            "training_steps": self.training_steps,
        }
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for state.
        
        Args:
            state: Current state.
        
        Returns:
            Value estimate V(s).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.network(state_tensor)
            return value.item()
    
    def save(self, path: Path) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save location.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
            }
        }
        
        torch.save(save_dict, path)
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
        
        save_dict = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(save_dict["network"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        self.total_steps = save_dict["total_steps"]
        self.training_steps = save_dict["training_steps"]
        self.episodes = save_dict["episodes"]
        
        logger.info(f"Agent loaded from {path} (steps={self.total_steps})")
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = super().get_stats()
        stats.update({
            "training_steps": self.training_steps,
            "buffer_size": len(self.rollout_buffer),
        })
        
        if self.policy_losses:
            stats["mean_policy_loss"] = np.mean(self.policy_losses[-100:])
        if self.value_losses:
            stats["mean_value_loss"] = np.mean(self.value_losses[-100:])
        if self.entropies:
            stats["mean_entropy"] = np.mean(self.entropies[-100:])
        
        return stats
