"""
Plotting Utilities
=================

Create training and evaluation plots.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_metrics(
    metrics_path: Path,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot training metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics.json file.
        save_path: Optional path to save figure.
        show: Whether to display plot.
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episode_rewards = metrics.get('episode_rewards', [])
    episode_lengths = metrics.get('episode_lengths', [])
    eval_rewards = metrics.get('eval_rewards', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    episodes = np.arange(len(episode_rewards))
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    window = 50
    if len(episode_rewards) > window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(
            np.arange(window - 1, len(episode_rewards)),
            moving_avg,
            color='red',
            linewidth=2,
            label=f'Moving Average ({window})'
        )
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    ax.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Episode Length')
    
    # Moving average
    if len(episode_lengths) > window:
        moving_avg = np.convolve(
            episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(
            np.arange(window - 1, len(episode_lengths)),
            moving_avg,
            color='orange',
            linewidth=2,
            label=f'Moving Average ({window})'
        )
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Evaluation Rewards
    ax = axes[1, 0]
    if eval_rewards:
        eval_episodes = np.linspace(0, len(episode_rewards), len(eval_rewards))
        ax.plot(eval_episodes, eval_rewards, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Rewards')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
        ax.axis('off')
    
    # 4. Reward Distribution
    ax = axes[1, 1]
    if len(episode_rewards) > 0:
        ax.hist(episode_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()


def plot_agent_comparison(
    agent_metrics: Dict[str, Path],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Compare multiple agents' training performance.
    
    Args:
        agent_metrics: Dictionary mapping agent names to metrics.json paths.
        save_path: Optional path to save figure.
        show: Whether to display plot.
    """
    # Load all metrics
    all_metrics = {}
    for name, path in agent_metrics.items():
        with open(path, 'r') as f:
            all_metrics[name] = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # 1. Episode Rewards Comparison
    ax = axes[0, 0]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        rewards = metrics.get('episode_rewards', [])
        if rewards:
            # Moving average
            window = 50
            if len(rewards) > window:
                moving_avg = np.convolve(
                    rewards,
                    np.ones(window) / window,
                    mode='valid'
                )
                episodes = np.arange(window - 1, len(rewards))
                ax.plot(episodes, moving_avg, color=colors[i % len(colors)],
                       linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Avg)')
    ax.set_title('Training Rewards Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final Performance (Box Plot)
    ax = axes[0, 1]
    final_rewards = []
    labels = []
    for name, metrics in all_metrics.items():
        rewards = metrics.get('episode_rewards', [])
        if rewards:
            # Last 100 episodes
            final_rewards.append(rewards[-100:])
            labels.append(name)
    
    if final_rewards:
        ax.boxplot(final_rewards, labels=labels)
        ax.set_ylabel('Reward')
        ax.set_title('Final Performance (Last 100 Episodes)')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Evaluation Comparison
    ax = axes[1, 0]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        eval_rewards = metrics.get('eval_rewards', [])
        if eval_rewards:
            episodes = np.arange(len(eval_rewards))
            ax.plot(episodes, eval_rewards, marker='o', linewidth=2,
                   color=colors[i % len(colors)], label=name)
    
    ax.set_xlabel('Evaluation #')
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Summary Statistics\n" + "=" * 40 + "\n\n"
    for name, metrics in all_metrics.items():
        rewards = metrics.get('episode_rewards', [])
        if rewards:
            summary_text += f"{name}:\n"
            summary_text += f"  Mean Reward: {np.mean(rewards[-100:]):.2f}\n"
            summary_text += f"  Max Reward: {np.max(rewards):.2f}\n"
            summary_text += f"  Episodes: {len(rewards)}\n\n"
    
    ax.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
           verticalalignment='top')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()


def plot_reward_curve(
    rewards: List[float],
    title: str = "Reward Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot simple reward curve.
    
    Args:
        rewards: List of rewards.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        show: Whether to display plot.
    """
    plt.figure(figsize=(12, 6))
    
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, alpha=0.5, color='blue')
    
    # Moving average
    window = min(50, len(rewards) // 10)
    if len(rewards) > window:
        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )
        plt.plot(
            np.arange(window - 1, len(rewards)),
            moving_avg,
            color='red',
            linewidth=2,
            label=f'Moving Average ({window})'
        )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
