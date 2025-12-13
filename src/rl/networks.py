"""
Neural Networks for DQN
======================

Deep Q-Network architectures.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    Standard DQN network.
    
    Architecture:
        - Input layer
        - 2-3 hidden layers with ReLU
        - Output layer (Q-values for each action)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            hidden_dims: Tuple of hidden layer dimensions.
            activation: Activation function ('relu', 'tanh', 'elu').
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.hidden_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim].
        
        Returns:
            Q-values [batch_size, action_dim].
        """
        x = state
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output layer
        q_values = self.output_layer(x)
        
        return q_values


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN network.
    
    Architecture:
        - Shared feature extractor
        - Value stream: V(s)
        - Advantage stream: A(s,a)
        - Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    
    Paper: "Dueling Network Architectures for Deep RL" (Wang et al., 2016)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        """
        Initialize Dueling DQN network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            hidden_dims: Tuple of hidden layer dimensions.
            activation: Activation function.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared feature extractor
        feature_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.ModuleList(feature_layers)
        
        # Value stream
        self.value_hidden = nn.Linear(prev_dim, hidden_dims[-1])
        self.value_output = nn.Linear(hidden_dims[-1], 1)
        
        # Advantage stream
        self.advantage_hidden = nn.Linear(prev_dim, hidden_dims[-1])
        self.advantage_output = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.feature_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        
        for layer in [self.value_hidden, self.advantage_hidden]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        
        for layer in [self.value_output, self.advantage_output]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim].
        
        Returns:
            Q-values [batch_size, action_dim].
        """
        # Shared features
        x = state
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        # Value stream
        value = self.activation(self.value_hidden(x))
        value = self.value_output(value)  # [batch, 1]
        
        # Advantage stream
        advantage = self.activation(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage)  # [batch, action_dim]
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
        - Shared feature extractor
        - Actor head: policy π(a|s)
        - Critic head: value V(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            hidden_dims: Tuple of hidden layer dimensions.
            activation: Activation function.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared feature extractor
        feature_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.ModuleList(feature_layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(prev_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.feature_layers:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)
        
        # Small initialization for policy head (helps exploration)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        
        # Standard initialization for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim].
        
        Returns:
            Tuple of (action_logits, value).
            - action_logits: [batch_size, action_dim]
            - value: [batch_size, 1]
        """
        # Shared features
        x = state
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        # Actor output (logits)
        action_logits = self.actor(x)
        
        # Critic output (value)
        value = self.critic(x)
        
        return action_logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value.
        
        Args:
            state: State tensor [batch_size, state_dim].
            action: Optional action tensor [batch_size].
        
        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
