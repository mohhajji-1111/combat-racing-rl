"""
Play Script - Combat Racing Championship
========================================

Play the game or watch trained agents.

Usage:
    # Play as human
    python scripts/play.py --mode human
    
    # Watch trained agent
    python scripts/play.py --mode agent --agent dqn --checkpoint checkpoints/dqn/best_model.pth

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
sys.path.insert(0, '.')

import argparse
import pygame
import numpy as np
from pathlib import Path

from src.game.track import create_oval_track
from src.rl import CombatRacingEnv
from src.rl.agents import QLearningAgent, DQNAgent, PPOAgent
from src.utils.logger import setup_logger, get_logger

setup_logger("play")
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play Combat Racing Championship")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "agent"],
        help="Play mode: human or agent"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        choices=["qlearning", "dqn", "ppo"],
        help="Agent type (required if mode=agent)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to agent checkpoint (required if mode=agent)"
    )
    
    parser.add_argument(
        "--track",
        type=str,
        default="oval",
        choices=["oval", "figure8"],
        help="Track type"
    )
    
    parser.add_argument(
        "--num-opponents",
        type=int,
        default=3,
        help="Number of opponent cars"
    )
    
    return parser.parse_args()


def load_agent(agent_type: str, checkpoint_path: str, state_dim: int, action_dim: int):
    """Load trained agent."""
    if agent_type == "qlearning":
        agent = QLearningAgent(state_dim, action_dim)
        agent.load(checkpoint_path)
    elif agent_type == "dqn":
        agent = DQNAgent(state_dim, action_dim, device='cpu')
        agent.load(checkpoint_path)
    elif agent_type == "ppo":
        agent = PPOAgent(state_dim, action_dim, device='cpu')
        agent.load(checkpoint_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"Loaded {agent_type} agent from {checkpoint_path}")
    return agent


def get_human_action():
    """Get action from keyboard input."""
    keys = pygame.key.get_pressed()
    
    # Map keys to actions
    accelerate = keys[pygame.K_UP] or keys[pygame.K_w]
    brake = keys[pygame.K_DOWN] or keys[pygame.K_s]
    turn_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    turn_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire_laser = keys[pygame.K_SPACE]
    fire_missile = keys[pygame.K_LSHIFT]
    drop_mine = keys[pygame.K_LCTRL]
    
    # Combine into action
    action = 0  # Default: no action
    
    if accelerate and turn_left:
        action = 5  # Forward + Left
    elif accelerate and turn_right:
        action = 6  # Forward + Right
    elif accelerate:
        action = 1  # Forward
    elif brake and turn_left:
        action = 9  # Backward + Left
    elif brake and turn_right:
        action = 10  # Backward + Right
    elif brake:
        action = 2  # Backward
    elif turn_left:
        action = 3  # Turn Left
    elif turn_right:
        action = 4  # Turn Right
    elif fire_laser:
        action = 7  # Fire Laser
    elif fire_missile:
        action = 8  # Fire Missile
    elif drop_mine:
        action = 11  # Drop Mine
    
    return action


def play_human(env: CombatRacingEnv):
    """Play as human player."""
    logger.info("Starting human play mode")
    logger.info("Controls:")
    logger.info("  ↑/W - Accelerate")
    logger.info("  ↓/S - Brake")
    logger.info("  ←/A - Turn Left")
    logger.info("  →/D - Turn Right")
    logger.info("  SPACE - Fire Laser")
    logger.info("  LSHIFT - Fire Missile")
    logger.info("  LCTRL - Drop Mine")
    logger.info("  ESC - Quit")
    
    state, _ = env.reset()  # Gymnasium API returns (observation, info)
    done = False
    total_reward = 0
    steps = 0
    
    clock = pygame.time.Clock()
    
    while not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        
        # Get action from keyboard
        action = get_human_action()
        
        # Take step
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # Render
        env.render()
        
        # Display stats
        if steps % 60 == 0:  # Every second
            logger.info(f"Steps: {steps}, Reward: {total_reward:.2f}")
        
        clock.tick(60)  # 60 FPS
    
    logger.info(f"Game over! Total reward: {total_reward:.2f}, Steps: {steps}")


def play_agent(env: CombatRacingEnv, agent):
    """Watch trained agent play."""
    logger.info(f"Starting agent play mode with {agent.__class__.__name__}")
    
    state, _ = env.reset()  # Gymnasium API returns (observation, info)
    done = False
    total_reward = 0
    steps = 0
    
    clock = pygame.time.Clock()
    
    while not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        
        # Get action from agent
        action = agent.select_action(state, deterministic=True)
        
        # Take step
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # Render
        env.render()
        
        # Display stats
        if steps % 60 == 0:  # Every second
            logger.info(f"Steps: {steps}, Reward: {total_reward:.2f}")
        
        clock.tick(60)  # 60 FPS
    
    logger.info(f"Game over! Total reward: {total_reward:.2f}, Steps: {steps}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create environment (it creates track internally)
    env = CombatRacingEnv(
        num_opponents=args.num_opponents,
        render_mode="human"
    )
    
    # Play
    if args.mode == "human":
        play_human(env)
    else:
        # Load agent
        if not args.agent or not args.checkpoint:
            logger.error("--agent and --checkpoint required for agent mode")
            return
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = load_agent(args.agent, args.checkpoint, state_dim, action_dim)
        play_agent(env, agent)
    
    pygame.quit()


if __name__ == "__main__":
    main()
