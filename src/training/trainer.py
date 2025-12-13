"""
Training Infrastructure
======================

Training loop and utilities for RL agents.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import json

from ..game import GameEngine, Track, create_oval_track
from ..rl import (
    CombatRacingEnv,
    BaseAgent,
    QLearningAgent,
    DQNAgent,
    PPOAgent,
)
from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class Trainer:
    """
    Training loop for RL agents.
    
    Features:
        - Multi-agent training
        - Checkpoint saving
        - Metrics logging
        - Curriculum learning support
        - Early stopping
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        env: CombatRacingEnv,
        config: Optional[Dict] = None,
        save_dir: Path = Path("checkpoints"),
    ):
        """
        Initialize trainer.
        
        Args:
            agent: RL agent to train.
            env: Training environment.
            config: Training configuration.
            save_dir: Directory to save checkpoints.
        """
        self.agent = agent
        self.env = env
        self.config = config or {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.max_episodes = self.config.get("max_episodes", 1000)
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 10000)
        self.save_freq = self.config.get("save_freq", 100)
        self.eval_freq = self.config.get("eval_freq", 50)
        self.log_freq = self.config.get("log_freq", 10)
        
        # Early stopping
        self.patience = self.config.get("patience", 100)
        self.min_improvement = self.config.get("min_improvement", 0.01)
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.best_eval_reward = -float('inf')
        self.episodes_without_improvement = 0
        
        # Timing
        self.start_time = None
        self.episode_times: List[float] = []
        
        logger.info(
            f"Trainer initialized: {self.max_episodes} episodes, "
            f"save_dir={save_dir}"
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Training statistics.
        """
        logger.info("Starting training...")
        self.start_time = time.time()
        
        try:
            for episode in tqdm(range(self.max_episodes), desc="Training"):
                episode_start = time.time()
                
                # Run episode
                episode_stats = self._train_episode(episode)
                
                # Record metrics
                self.episode_rewards.append(episode_stats["total_reward"])
                self.episode_lengths.append(episode_stats["steps"])
                self.episode_times.append(time.time() - episode_start)
                
                # Logging
                if (episode + 1) % self.log_freq == 0:
                    self._log_progress(episode)
                
                # Evaluation
                if (episode + 1) % self.eval_freq == 0:
                    eval_reward = self._evaluate()
                    self.eval_rewards.append(eval_reward)
                    
                    # Check improvement
                    if eval_reward > self.best_eval_reward + self.min_improvement:
                        self.best_eval_reward = eval_reward
                        self.episodes_without_improvement = 0
                        
                        # Save best model
                        self._save_checkpoint(episode, prefix="best")
                    else:
                        self.episodes_without_improvement += 1
                    
                    # Early stopping
                    if self.episodes_without_improvement >= self.patience:
                        logger.info(
                            f"Early stopping at episode {episode + 1}: "
                            f"no improvement for {self.patience} evaluations"
                        )
                        break
                
                # Save checkpoint
                if (episode + 1) % self.save_freq == 0:
                    self._save_checkpoint(episode)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final save
            self._save_checkpoint(self.agent.episodes, prefix="final")
            
            # Save metrics
            self._save_metrics()
        
        # Training summary
        total_time = time.time() - self.start_time
        stats = {
            "total_episodes": len(self.episode_rewards),
            "total_time": total_time,
            "mean_reward": np.mean(self.episode_rewards[-100:]),
            "best_eval_reward": self.best_eval_reward,
            "mean_episode_time": np.mean(self.episode_times),
        }
        
        logger.info(f"Training complete: {stats}")
        return stats
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Args:
            episode: Episode number.
        
        Returns:
            Episode statistics.
        """
        state, _ = self.env.reset()  # Gymnasium API returns (observation, info)
        self.agent.train_mode()
        
        total_reward = 0.0
        steps = 0
        
        done = False
        while not done and steps < self.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Agent update
            update_metrics = self.agent.update(
                state, action, reward, next_state, done
            )
            
            # Next state
            state = next_state
            total_reward += reward
            steps += 1
        
        self.agent.episodes += 1
        
        return {
            "total_reward": total_reward,
            "steps": steps,
            "done": done,
        }
    
    def _evaluate(
        self,
        num_episodes: int = 5,
    ) -> float:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes.
        
        Returns:
            Mean evaluation reward.
        """
        self.agent.eval_mode()
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()  # Gymnasium API returns (observation, info)
            total_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            eval_rewards.append(total_reward)
        
        self.agent.train_mode()
        
        mean_reward = np.mean(eval_rewards)
        logger.info(f"Evaluation: mean_reward={mean_reward:.2f}")
        
        return mean_reward
    
    def _log_progress(self, episode: int) -> None:
        """Log training progress."""
        recent_rewards = self.episode_rewards[-self.log_freq:]
        recent_lengths = self.episode_lengths[-self.log_freq:]
        
        stats = self.agent.get_stats()
        
        logger.info(
            f"Episode {episode + 1}/{self.max_episodes} | "
            f"Mean Reward: {np.mean(recent_rewards):.2f} | "
            f"Mean Length: {np.mean(recent_lengths):.1f} | "
            f"Agent Stats: {stats}"
        )
    
    def _save_checkpoint(
        self,
        episode: int,
        prefix: str = "checkpoint",
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            episode: Episode number.
            prefix: Checkpoint prefix.
        """
        checkpoint_path = self.save_dir / f"{prefix}_episode_{episode}.pt"
        self.agent.save(checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_metrics(self) -> None:
        """Save training metrics to JSON."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "best_eval_reward": self.best_eval_reward,
            "episode_times": self.episode_times,
        }
        
        metrics_path = self.save_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved: {metrics_path}")


def create_trainer(
    agent_type: str,
    config_path: Optional[Path] = None,
    render: bool = False,
) -> Trainer:
    """
    Create trainer with specified agent type.
    
    Args:
        agent_type: Agent type ('qlearning', 'dqn', 'ppo').
        config_path: Path to configuration file.
        render: Enable rendering.
    
    Returns:
        Configured trainer.
    """
    # Load configuration
    if config_path:
        config_loader = ConfigLoader(config_path)
        game_config = config_loader.get_config("game")
        rl_config = config_loader.get_config("rl")
        training_config = config_loader.get_config("training")
    else:
        game_config = {}
        rl_config = {}
        training_config = {}
    
    # Create environment
    render_mode = "human" if render else None
    env = CombatRacingEnv(
        track_name="medium",
        num_opponents=3,
        render_mode=render_mode,
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if agent_type.lower() == "qlearning":
        agent_config = rl_config.get("qlearning", {})
        agent = QLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config
        )
    
    elif agent_type.lower() == "dqn":
        agent_config = rl_config.get("dqn", {})
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config
        )
    
    elif agent_type.lower() == "ppo":
        agent_config = rl_config.get("ppo", {})
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create trainer
    save_dir = Path(training_config.get("save_dir", "checkpoints")) / agent_type
    
    trainer = Trainer(
        agent=agent,
        env=env,
        config=training_config,
        save_dir=save_dir,
    )
    
    logger.info(f"Created trainer for {agent_type} agent")
    return trainer
