"""
Play Script
==========

Watch trained agents compete in Combat Racing Championship.

Usage:
    python play.py --agent1 checkpoints/dqn/best.pt --agent2 checkpoints/ppo/best.pt
    python play.py --human  # Play as human

Author: Combat Racing RL Team
Date: 2024-2025
"""

import argparse
from pathlib import Path
import sys
import pygame

from src.game import create_oval_track, GameEngine
from src.rl import DQNAgent, PPOAgent, QLearningAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Play Combat Racing Championship")
    
    parser.add_argument("--agent1", type=Path, help="Path to agent 1 checkpoint")
    parser.add_argument("--agent2", type=Path, help="Path to agent 2 checkpoint")
    parser.add_argument("--agent3", type=Path, help="Path to agent 3 checkpoint")
    parser.add_argument("--human", action="store_true", help="Play as human (car 0)")
    parser.add_argument("--fps", type=int, default=60, help="FPS limit")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting Combat Racing Championship...")
    
    # Create track and game
    track = create_oval_track()
    game = GameEngine(track=track, num_cars=4, render_mode="human")
    
    # Load agents
    agents = [None] * 4
    
    if not args.human:
        # All AI
        for i, checkpoint_path in enumerate([args.agent1, args.agent2, args.agent3]):
            if checkpoint_path and checkpoint_path.exists():
                # Determine agent type from checkpoint
                agent = DQNAgent(state_dim=100, action_dim=12)  # Dummy dims
                agent.load(checkpoint_path)
                agent.eval_mode()
                agents[i] = agent
                logger.info(f"Loaded agent {i} from {checkpoint_path}")
    else:
        logger.info("Human player is car 0 (WASD + Space/E/Q)")
    
    # Game loop
    observations = game.reset()
    running = True
    
    logger.info("Game started! Press ESC to quit")
    
    while running:
        # Get actions
        actions = []
        
        for i in range(4):
            if args.human and i == 0:
                # Human control
                action = get_human_action()
            elif agents[i] is not None:
                # AI agent
                action = agents[i].select_action(observations[i], deterministic=True)
            else:
                # Random/no-op
                action = 0
            
            actions.append(action)
        
        # Step game
        observations, rewards, dones, info = game.step(actions)
        
        # Render
        game.render()
        
        # Check if done
        if all(dones) or not game.running:
            logger.info("Episode complete!")
            
            # Show results
            for i, car in enumerate(game.cars):
                logger.info(
                    f"Car {i}: Laps={car.laps_completed}, "
                    f"Checkpoints={car.current_checkpoint}, "
                    f"Kills={car.stats.get('kills', 0)}"
                )
            
            # Reset
            observations = game.reset()
    
    game.close()
    logger.info("Game closed")
    return 0


def get_human_action() -> int:
    """
    Get human player action from keyboard.
    
    Returns:
        Action ID (0-11).
    """
    keys = pygame.key.get_pressed()
    
    # Movement
    forward = keys[pygame.K_w]
    backward = keys[pygame.K_s]
    left = keys[pygame.K_a]
    right = keys[pygame.K_d]
    
    # Weapons
    laser = keys[pygame.K_SPACE]
    missile = keys[pygame.K_e]
    mine = keys[pygame.K_q]
    
    # Map to action
    # 0: No-op
    # 1: Forward, 2: Backward, 3: Left, 4: Right
    # 5: Forward+Left, 6: Forward+Right
    # 7: Laser, 8: Missile, 9: Mine
    # 10: Forward+Laser, 11: Forward+Missile
    
    if forward and left:
        return 5
    elif forward and right:
        return 6
    elif forward and laser:
        return 10
    elif forward and missile:
        return 11
    elif forward:
        return 1
    elif backward:
        return 2
    elif left:
        return 3
    elif right:
        return 4
    elif laser:
        return 7
    elif missile:
        return 8
    elif mine:
        return 9
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
