"""
Evaluation Script - Combat Racing Championship
=============================================

Evaluate trained agents.

Usage:
    python scripts/evaluate.py --agent dqn --checkpoint checkpoints/dqn/best_model.pth
    python scripts/evaluate.py --agent ppo --checkpoint checkpoints/ppo/best_model.pth --render

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
sys.path.insert(0, '.')

from src.training.evaluate import main

if __name__ == "__main__":
    main()
