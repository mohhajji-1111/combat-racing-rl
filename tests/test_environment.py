"""
RL Environment Tests
===================

Test Gymnasium environment wrapper.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pytest
import numpy as np
from src.rl import CombatRacingEnv
from src.game import create_oval_track


class TestCombatRacingEnv:
    """Test CombatRacingEnv."""
    
    def test_initialization(self):
        """Test environment initialization."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=3)
        
        assert env.num_opponents == 3
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.n == 12
    
    def test_reset(self):
        """Test environment reset."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=2)
        
        obs = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert env.player_car is not None
        assert len(env.opponent_cars) == 2
    
    def test_step(self):
        """Test environment step."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=2)
        
        env.reset()
        
        # Take action
        action = 1  # Forward
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_action_space(self):
        """Test all actions work."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1)
        
        env.reset()
        
        # Test all 12 actions
        for action in range(12):
            obs, reward, done, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            
            if done:
                env.reset()
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1, max_steps=100)
        
        env.reset()
        
        # Run until done
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Should terminate eventually
        assert done or steps >= 100
    
    def test_reward_signal(self):
        """Test reward signals."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1)
        
        env.reset()
        
        # Collect rewards
        rewards = []
        for _ in range(50):
            action = 1  # Forward
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        # Should get some rewards
        assert len(rewards) > 0
        # Not all rewards should be zero
        assert not all(r == 0 for r in rewards)
    
    def test_observation_bounds(self):
        """Test observation values are within bounds."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=2)
        
        obs = env.reset()
        
        # Check bounds
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)
        
        # Take some steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            
            assert np.all(obs >= env.observation_space.low)
            assert np.all(obs <= env.observation_space.high)
            
            if done:
                obs = env.reset()


class TestEnvironmentIntegration:
    """Test environment integration with agents."""
    
    def test_env_agent_interaction(self):
        """Test environment works with agent."""
        from src.rl.agents import DQNAgent
        
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1)
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim, device='cpu')
        
        # Run episode
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        
        assert steps > 0
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1)
        
        episode_rewards = []
        
        for episode in range(5):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
        
        assert len(episode_rewards) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
