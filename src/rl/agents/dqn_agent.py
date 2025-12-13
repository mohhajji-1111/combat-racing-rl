"""
DQN Agent
=========

Deep Q-Network with experience replay and target network.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .base_agent import BaseAgent
from ..networks import DQNNetwork, DuelingDQNNetwork
from ..replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from ...utils.logger import get_logger

logger = get_logger(__name__)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.
    
    Features:
        - Experience replay
        - Target network
        - Double DQN
        - Dueling architecture (optional)
        - Prioritized replay (optional)
    
    Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        learning_starts: int = 1000,
        hidden_dims: tuple = (256, 256),
        dueling: bool = True,
        double_dqn: bool = True,
        prioritized_replay: bool = False,
        device: str = "auto",
        config: Optional[Dict] = None,
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            learning_rate: Learning rate for optimizer.
            discount_factor: Discount factor (γ).
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Exploration decay rate.
            buffer_size: Replay buffer capacity.
            batch_size: Mini-batch size.
            target_update_freq: Update target network every N steps.
            learning_starts: Start learning after N steps.
            hidden_dims: Hidden layer dimensions.
            dueling: Use dueling architecture.
            double_dqn: Use Double DQN.
            prioritized_replay: Use prioritized experience replay.
            device: Device ('cpu', 'cuda', or 'auto').
            config: Configuration dictionary.
        """
        super().__init__(state_dim, action_dim, config)
        
        # Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.dueling = dueling
        self.double_dqn = double_dqn
        
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Networks
        network_cls = DuelingDQNNetwork if dueling else DQNNetwork
        self.q_network = network_cls(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        self.target_network = network_cls(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, state_dim, device=self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size, state_dim, device=self.device
            )
        
        self.prioritized_replay = prioritized_replay
        
        # Statistics
        self.training_steps = 0
        self.losses = []
        
        logger.info(
            f"DQNAgent initialized: device={self.device}, "
            f"dueling={dueling}, double_dqn={double_dqn}, "
            f"prioritized={prioritized_replay}"
        )
    
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
        # Epsilon-greedy
        if not deterministic and self.training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
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
        Add transition to buffer and train if ready.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        
        Returns:
            Dictionary of training metrics.
        """
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.total_steps += 1
        
        # Don't train until we have enough data
        if len(self.replay_buffer) < self.learning_starts:
            return {"loss": 0.0, "q_value": 0.0}
        
        # Train
        metrics = self._train_step()
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"Target network updated at step {self.total_steps}")
        
        return metrics
    
    def _train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics.
        """
        self.training_steps += 1
        
        # Sample batch
        if self.prioritized_replay:
            (states, actions, rewards, next_states, dones,
             weights, indices) = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones_like(rewards)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # TD errors
        td_errors = torch.abs(target_q - current_q)
        
        # Loss (weighted for prioritized replay)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        if self.prioritized_replay:
            priorities = td_errors.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Statistics
        self.losses.append(loss.item())
        
        return {
            "loss": loss.item(),
            "q_value": current_q.mean().item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.
        
        Args:
            state: Current state.
        
        Returns:
            Array of Q-values.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.squeeze(0).cpu().numpy()
    
    def save(self, path: Path) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save location.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "lr": self.lr,
                "gamma": self.gamma,
                "dueling": self.dueling,
                "double_dqn": self.double_dqn,
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
        
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(save_dict["q_network"])
        self.target_network.load_state_dict(save_dict["target_network"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        self.epsilon = save_dict["epsilon"]
        self.total_steps = save_dict["total_steps"]
        self.training_steps = save_dict["training_steps"]
        self.episodes = save_dict["episodes"]
        
        logger.info(f"Agent loaded from {path} (steps={self.total_steps})")
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = super().get_stats()
        stats.update({
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "training_steps": self.training_steps,
        })
        
        if self.losses:
            stats["mean_loss"] = np.mean(self.losses[-100:])
        
        return stats
