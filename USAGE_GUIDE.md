# 🎮 Combat Racing Championship - Complete Usage Guide

## 📋 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Agents](#training-agents)
4. [Evaluating Agents](#evaluating-agents)
5. [Playing the Game](#playing-the-game)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Install Dependencies
```bash
# Navigate to project directory
cd combat-racing-rl

# Install required packages
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### Step 2: Verify Installation
```bash
# Run verification script
python scripts/verify_install.py

# Test components
python scripts/test_components.py
```

---

## 🚀 Quick Start

### Run a Quick Demo
```bash
# Play with random agents (no training needed)
python play.py

# Train a DQN agent for 100 episodes
python -m src.training.train --agent dqn --episodes 100

# Evaluate the trained agent
python -m src.training.evaluate --agent dqn --checkpoint checkpoints/dqn/best_episode_*.pt --render
```

---

## 🎓 Training Agents

### Training Commands

#### **Q-Learning (Tabular RL)**
```bash
# Basic training
python -m src.training.train --agent qlearning --episodes 2000

# With custom config
python -m src.training.train --agent qlearning --config config/rl_config.yaml --episodes 2000

# With rendering (slower but visual)
python -m src.training.train --agent qlearning --episodes 500 --render
```

**Characteristics**:
- ✅ Simple, interpretable
- ✅ No neural networks required
- ❌ Requires state discretization
- 🎯 Good for: Understanding RL basics

#### **DQN (Deep Q-Network)**
```bash
# Basic training
python -m src.training.train --agent dqn --episodes 1000

# With GPU acceleration (if available)
python -m src.training.train --agent dqn --episodes 1000 --config config/rl_config.yaml

# Extended training for better performance
python -m src.training.train --agent dqn --episodes 5000 --save-dir checkpoints/dqn_extended
```

**Characteristics**:
- ✅ Deep learning-based
- ✅ Experience replay
- ✅ Target networks
- 🎯 Good for: High-dimensional state spaces

#### **PPO (Proximal Policy Optimization)**
```bash
# Basic training
python -m src.training.train --agent ppo --episodes 500

# With custom hyperparameters
python -m src.training.train --agent ppo --config config/rl_config.yaml --episodes 1000

# Fast training (no rendering)
python -m src.training.train --agent ppo --episodes 2000 --seed 123
```

**Characteristics**:
- ✅ State-of-the-art policy gradient
- ✅ Stable training
- ✅ Works well with continuous/discrete actions
- 🎯 Good for: Best overall performance

### Training Options

| Option | Description | Example |
|--------|-------------|---------|
| `--agent` | Agent type: qlearning, dqn, ppo | `--agent dqn` |
| `--config` | Path to config file | `--config config/training_config.yaml` |
| `--episodes` | Number of training episodes | `--episodes 1000` |
| `--render` | Enable visualization during training | `--render` |
| `--seed` | Random seed for reproducibility | `--seed 42` |
| `--save-dir` | Directory to save checkpoints | `--save-dir checkpoints/my_agent` |

### Training Tips

1. **Start Small**: Begin with 100-500 episodes to verify everything works
2. **Monitor Progress**: Check `checkpoints/<agent>/metrics.json` for training curves
3. **Use GPU**: For DQN/PPO, ensure CUDA is available for faster training
4. **Experiment with Seeds**: Different seeds can lead to different learning behaviors
5. **Adjust Hyperparameters**: Edit `config/rl_config.yaml` for fine-tuning

---

## 📊 Evaluating Agents

### Evaluation Commands

#### **Basic Evaluation**
```bash
# Evaluate best checkpoint
python -m src.training.evaluate \
    --agent dqn \
    --checkpoint checkpoints/dqn/best_episode_900.pt \
    --episodes 10
```

#### **With Visualization**
```bash
# Watch agent play with rendering
python -m src.training.evaluate \
    --agent ppo \
    --checkpoint checkpoints/ppo/best_episode_450.pt \
    --episodes 5 \
    --render
```

#### **Multiple Evaluations**
```bash
# Compare different checkpoints
for checkpoint in checkpoints/dqn/*.pt; do
    python -m src.training.evaluate \
        --agent dqn \
        --checkpoint $checkpoint \
        --episodes 10
done
```

### Evaluation Metrics

The evaluation script reports:
- **Mean Reward**: Average reward across episodes
- **Std Reward**: Standard deviation (consistency measure)
- **Max/Min Reward**: Best and worst episode performance
- **Mean Length**: Average episode length (steps)

### Interpreting Results

| Metric | Good Performance | Poor Performance |
|--------|------------------|------------------|
| Mean Reward | > 500 | < 0 |
| Std Reward | Low (< 100) | High (> 500) |
| Episode Length | 1000-5000 steps | < 500 steps |

---

## 🎮 Playing the Game

### Interactive Gameplay

#### **Play as Human**
```bash
# You control car 0
python play.py --human
```

**Controls**:
- `W` - Accelerate
- `S` - Brake/Reverse
- `A` - Turn Left
- `D` - Turn Right
- `Space` - Fire Laser
- `E` - Fire Missile
- `Q` - Drop Mine
- `H` - Toggle HUD
- `M` - Toggle Minimap
- `ESC` - Quit

#### **Watch AI Agents Compete**
```bash
# Load trained agents
python play.py \
    --agent1 checkpoints/dqn/best_episode_900.pt \
    --agent2 checkpoints/ppo/best_episode_450.pt \
    --agent3 checkpoints/dqn/best_episode_800.pt
```

#### **Mixed Human + AI**
```bash
# You vs trained agents
python play.py --human \
    --agent1 checkpoints/dqn/best_episode_900.pt \
    --agent2 checkpoints/ppo/best_episode_450.pt
```

### Gameplay Features

- **🏁 Racing**: Complete laps by crossing checkpoints in order
- **💥 Combat**: Shoot opponents to reduce their health
- **⚡ Power-ups**: Collect power-ups for temporary advantages
  - Speed Boost: Increased max speed for 5 seconds
  - Shield: Reduced damage for 10 seconds
  - Double Damage: 2x weapon damage for 8 seconds
  - Ammo Refill: Restore all ammunition
  - Health Pack: Restore 50 HP

---

## ⚙️ Configuration

### Configuration Files

All configs are in `config/` directory:

#### **1. game_config.yaml**
Controls game mechanics:
```yaml
game:
  max_laps: 3                    # Laps to win
  max_steps_per_episode: 10000  # Episode length limit
  
car:
  max_health: 100
  max_speed: 300
  acceleration: 200
  turn_rate: 3.0
  
weapons:
  laser:
    damage: 25
    cooldown: 0.5
    speed: 800
```

#### **2. rl_config.yaml**
RL agent hyperparameters:
```yaml
dqn:
  learning_rate: 0.0001
  discount_factor: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  buffer_size: 100000
  batch_size: 64
  
ppo:
  learning_rate: 0.0003
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
```

#### **3. training_config.yaml**
Training settings:
```yaml
max_episodes: 1000
save_freq: 100          # Save every N episodes
eval_freq: 50           # Evaluate every N episodes
log_freq: 10            # Log every N episodes
patience: 100           # Early stopping patience
```

### Modifying Configurations

1. **Edit YAML files** directly
2. Or **override via code**:
```python
from src.utils.config_loader import ConfigLoader

config_loader = ConfigLoader("config/game_config.yaml")
config = config_loader.get_config()
config["car"]["max_speed"] = 400  # Faster cars!
```

---

## 🐛 Troubleshooting

### Common Issues

#### **1. Import Errors**
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Install package in development mode
```bash
pip install -e .
```

#### **2. Pygame Not Found**
```
ModuleNotFoundError: No module named 'pygame'
```
**Solution**: Install pygame
```bash
pip install pygame
```

#### **3. CUDA Not Available**
```
Training is slow on CPU
```
**Solution**: Install PyTorch with CUDA support
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **4. Memory Error During Training**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config/rl_config.yaml`
```yaml
dqn:
  batch_size: 32  # Reduced from 64
```

#### **5. Agent Not Learning**
**Symptoms**: Reward stays flat or negative

**Solutions**:
1. Train longer (try 2000+ episodes)
2. Adjust learning rate (try 0.001 or 0.00001)
3. Increase exploration (epsilon_end = 0.05)
4. Check reward function in `src/rl/environment.py`

#### **6. Rendering Issues**
```
pygame.error: No available video device
```
**Solution**: Install display drivers or run without rendering
```bash
python -m src.training.train --agent dqn --episodes 1000  # No --render flag
```

### Getting Help

1. **Check logs**: Look in `logs/` directory
2. **Read docstrings**: All functions have detailed documentation
3. **Inspect code**: Clean, commented code throughout
4. **Debug mode**: Set logging level to DEBUG in `src/utils/logger.py`

---

## 📈 Performance Benchmarks

### Expected Training Times (CPU)

| Agent | Episodes | Time | Final Reward |
|-------|----------|------|--------------|
| Q-Learning | 2000 | ~2 hours | 300-500 |
| DQN | 1000 | ~3 hours | 500-800 |
| PPO | 500 | ~2 hours | 600-1000 |

### Expected Training Times (GPU)

| Agent | Episodes | Time | Final Reward |
|-------|----------|------|--------------|
| DQN | 1000 | ~1 hour | 500-800 |
| PPO | 500 | ~45 min | 600-1000 |

*Note: Times vary based on hardware*

---

## 🎯 Next Steps

### For Beginners
1. ✅ Run `play.py --human` to understand the game
2. ✅ Train Q-Learning agent (simplest algorithm)
3. ✅ Watch training progress in logs
4. ✅ Evaluate trained agent
5. ✅ Experiment with hyperparameters

### For Advanced Users
1. ✅ Implement custom reward functions
2. ✅ Create new track layouts
3. ✅ Add new power-up types
4. ✅ Experiment with network architectures
5. ✅ Implement multi-agent training

### For Researchers
1. ✅ Compare algorithm performance
2. ✅ Study exploration strategies
3. ✅ Analyze reward shaping impact
4. ✅ Investigate transfer learning
5. ✅ Publish results!

---

## 📚 Additional Resources

- **README.md**: Project overview and features
- **PROJECT_COMPLETE.md**: Detailed implementation summary
- **CONTRIBUTING.md**: Development guidelines
- **config/**: All configuration files with comments

---

## 🏆 Success Criteria

Your agent is performing well if:
- ✅ Completes at least 1 lap consistently
- ✅ Mean reward > 500 after training
- ✅ Wins battles against random agents
- ✅ Shows strategic behavior (collecting power-ups, avoiding walls)

---

**🎉 You're all set! Start training your champions! 🏁**

For questions or issues, check the logs or review the well-documented code.

**Good luck at ENSAM! 🎓**
