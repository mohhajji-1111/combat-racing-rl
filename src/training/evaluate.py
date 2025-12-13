"""
Evaluation Script
================

Evaluate trained RL agents.

Usage:
    python -m src.training.evaluate --agent dqn --checkpoint checkpoints/dqn/best.pt
    python -m src.training.evaluate --agent ppo --checkpoint checkpoints/ppo/final.pt --episodes 10 --render

Author: Combat Racing RL Team
Date: 2024-2025
"""

import argparse
from pathlib import Path
import sys
import numpy as np

from ..game import create_oval_track
from ..rl import CombatRacingEnv, QLearningAgent, DQNAgent, PPOAgent
from ..utils.logger import get_logger
from ..utils.helpers import seed_everything

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["qlearning", "dqn", "ppo"],
        help="Agent type"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to agent checkpoint"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    logger.info(f"Evaluating agent: {args.agent}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Episodes: {args.episodes}")
    
    try:
        # Create environment
        track = create_oval_track()
        render_mode = "human" if args.render else None
        env = CombatRacingEnv(
            track=track,
            num_opponents=3,
            render_mode=render_mode,
        )
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if args.agent == "qlearning":
            agent = QLearningAgent(state_dim, action_dim)
        elif args.agent == "dqn":
            agent = DQNAgent(state_dim, action_dim)
        elif args.agent == "ppo":
            agent = PPOAgent(state_dim, action_dim)
        
        # Load checkpoint
        agent.load(args.checkpoint)
        agent.eval_mode()
        
        logger.info("Agent loaded successfully")
        
        # Evaluation loop
        rewards = []
        lengths = []
        
        for episode in range(args.episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done:
                action = agent.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if args.render:
                    env.render()
            
            rewards.append(total_reward)
            lengths.append(steps)
            
            logger.info(
                f"Episode {episode + 1}/{args.episodes}: "
                f"Reward={total_reward:.2f}, Length={steps}"
            )
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        logger.info(f"Max Reward: {np.max(rewards):.2f}")
        logger.info(f"Min Reward: {np.min(rewards):.2f}")
        logger.info(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        logger.info("=" * 60)
        
        env.close()
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
