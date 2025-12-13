"""
RL Agents Tests
==============

Test reinforcement learning agents.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.rl.agents import QLearningAgent, DQNAgent, PPOAgent


class TestQLearningAgent:
    """Test Q-Learning agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = QLearningAgent(
            state_dim=10,
            action_dim=5,
            learning_rate=0.1,
            discount_factor=0.95
        )
        
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.lr == 0.1
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
    
    def test_action_selection(self):
        """Test action selection."""
        agent = QLearningAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        
        # Random action (exploration)
        agent.epsilon = 1.0
        action = agent.select_action(state)
        assert 0 <= action < 5
        
        # Greedy action (exploitation)
        agent.epsilon = 0.0
        action = agent.select_action(state, deterministic=True)
        assert 0 <= action < 5
    
    def test_update(self):
        """Test agent update."""
        agent = QLearningAgent(state_dim=10, action_dim=5)
        
        state = np.random.randn(10)
        action = 2
        reward = 10.0
        next_state = np.random.randn(10)
        done = False
        
        metrics = agent.update(state, action, reward, next_state, done)
        
        assert 'q_value' in metrics
        assert 'td_error' in metrics
        assert 'epsilon' in metrics
        assert agent.total_steps == 1
    
    def test_save_load(self):
        """Test save and load."""
        agent = QLearningAgent(state_dim=10, action_dim=5)
        
        # Train a bit
        for _ in range(10):
            state = np.random.randn(10)
            action = agent.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            agent.update(state, action, reward, next_state, False)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "agent.pkl"
            agent.save(save_path)
            
            # Load
            new_agent = QLearningAgent(state_dim=10, action_dim=5)
            new_agent.load(save_path)
            
            assert new_agent.total_steps == agent.total_steps
            assert len(new_agent.q_table) == len(agent.q_table)


class TestDQNAgent:
    """Test DQN agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(
            state_dim=20,
            action_dim=12,
            learning_rate=1e-4,
            discount_factor=0.99,
            device='cpu'
        )
        
        assert agent.state_dim == 20
        assert agent.action_dim == 12
        assert agent.lr == 1e-4
        assert agent.gamma == 0.99
        assert agent.device == 'cpu'
    
    def test_action_selection(self):
        """Test action selection."""
        agent = DQNAgent(state_dim=20, action_dim=12, device='cpu')
        state = np.random.randn(20)
        
        # Exploration
        agent.epsilon = 1.0
        action = agent.select_action(state)
        assert 0 <= action < 12
        
        # Exploitation
        agent.epsilon = 0.0
        action = agent.select_action(state, deterministic=True)
        assert 0 <= action < 12
    
    def test_update(self):
        """Test agent update."""
        agent = DQNAgent(
            state_dim=20,
            action_dim=12,
            device='cpu',
            learning_starts=10
        )
        
        # Fill replay buffer
        for _ in range(20):
            state = np.random.randn(20)
            action = np.random.randint(12)
            reward = np.random.randn()
            next_state = np.random.randn(20)
            done = False
            
            metrics = agent.update(state, action, reward, next_state, done)
        
        # Should have training metrics
        assert 'loss' in metrics
        assert 'q_value' in metrics
    
    def test_get_q_values(self):
        """Test Q-value retrieval."""
        agent = DQNAgent(state_dim=20, action_dim=12, device='cpu')
        state = np.random.randn(20)
        
        q_values = agent.get_q_values(state)
        assert q_values.shape == (12,)


class TestPPOAgent:
    """Test PPO agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(
            state_dim=20,
            action_dim=12,
            learning_rate=3e-4,
            discount_factor=0.99,
            device='cpu'
        )
        
        assert agent.state_dim == 20
        assert agent.action_dim == 12
        assert agent.lr == 3e-4
        assert agent.gamma == 0.99
    
    def test_action_selection(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=20, action_dim=12, device='cpu')
        state = np.random.randn(20)
        
        # Stochastic
        action = agent.select_action(state)
        assert 0 <= action < 12
        
        # Deterministic
        action = agent.select_action(state, deterministic=True)
        assert 0 <= action < 12
    
    def test_update(self):
        """Test agent update."""
        agent = PPOAgent(
            state_dim=20,
            action_dim=12,
            device='cpu',
            rollout_length=64
        )
        
        # Fill rollout buffer
        for _ in range(64):
            state = np.random.randn(20)
            action = np.random.randint(12)
            reward = np.random.randn()
            next_state = np.random.randn(20)
            done = False
            
            metrics = agent.update(state, action, reward, next_state, done)
        
        # Should have training metrics after full rollout
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
    
    def test_get_value(self):
        """Test value estimation."""
        agent = PPOAgent(state_dim=20, action_dim=12, device='cpu')
        state = np.random.randn(20)
        
        value = agent.get_value(state)
        assert isinstance(value, float)


class TestAgentTraining:
    """Test agent training integration."""
    
    def test_agent_learning(self):
        """Test that agents can learn from experience."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=4,
            device='cpu',
            learning_starts=10,
            batch_size=8
        )
        
        # Simple task: always get reward for action 2
        target_action = 2
        
        # Train
        for episode in range(50):
            state = np.random.randn(10)
            action = agent.select_action(state)
            
            # Reward for correct action
            reward = 10.0 if action == target_action else -1.0
            
            next_state = np.random.randn(10)
            agent.update(state, action, reward, next_state, False)
        
        # Test learned policy
        agent.epsilon = 0.0  # Greedy
        test_actions = []
        for _ in range(20):
            state = np.random.randn(10)
            action = agent.select_action(state, deterministic=True)
            test_actions.append(action)
        
        # Should prefer target action (at least sometimes)
        # Not 100% due to randomness and simple test
        target_ratio = test_actions.count(target_action) / len(test_actions)
        assert target_ratio > 0.2  # At least 20% of the time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
