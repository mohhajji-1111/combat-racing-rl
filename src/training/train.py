"""
Main Training Script
===================

Train RL agents for Combat Racing Championship.

Usage:
    python -m src.training.train --agent dqn --episodes 1000
    python -m src.training.train --agent ppo --config config/training_config.yaml --render

Author: Combat Racing RL Team
Date: 2024-2025
"""

import argparse
from pathlib import Path
import sys

from ..training import create_trainer
from ..utils.logger import get_logger
from ..utils.helpers import seed_everything

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for Combat Racing Championship"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["qlearning", "dqn", "ppo"],
        help="Agent type to train"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    logger.info(f"Starting training with agent: {args.agent}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Create trainer
        trainer = create_trainer(
            agent_type=args.agent,
            config_path=args.config,
            render=args.render,
        )
        
        # Override episodes if specified
        if args.episodes is not None:
            trainer.max_episodes = args.episodes
        
        # Override save directory
        if args.save_dir:
            trainer.save_dir = args.save_dir / args.agent
            trainer.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        stats = trainer.train()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Episodes: {stats['total_episodes']}")
        logger.info(f"Total Time: {stats['total_time']:.2f}s")
        logger.info(f"Mean Reward (last 100): {stats['mean_reward']:.2f}")
        logger.info(f"Best Eval Reward: {stats['best_eval_reward']:.2f}")
        logger.info(f"Mean Episode Time: {stats['mean_episode_time']:.2f}s")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
