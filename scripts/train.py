"""
Training Script - Combat Racing Championship
===========================================

Train reinforcement learning agents.

Usage:
    python scripts/train.py --agent dqn --episodes 5000
    python scripts/train.py --agent ppo --episodes 10000 --device cuda

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
sys.path.insert(0, '.')

from src.training.train import main

if __name__ == "__main__":
    main()
