# 🎮 Combat Racing Championship - Quick Commands

## 📦 Installation
```bash
pip install -r requirements.txt
pip install -e .
python scripts/verify_install.py
```

## 🎓 Training (Pick One)

### Q-Learning (Tabular, Simple)
```bash
python -m src.training.train --agent qlearning --episodes 2000
```

### DQN (Deep Learning, Recommended)
```bash
python -m src.training.train --agent dqn --episodes 1000
```

### PPO (State-of-the-Art, Best Performance)
```bash
python -m src.training.train --agent ppo --episodes 500
```

### With Rendering (Slower, Visual)
```bash
python -m src.training.train --agent dqn --episodes 100 --render
```

## 📊 Evaluation
```bash
# Replace <episode_num> with actual checkpoint number
python -m src.training.evaluate \
    --agent dqn \
    --checkpoint checkpoints/dqn/best_episode_<episode_num>.pt \
    --episodes 10 \
    --render
```

## 🎮 Playing

### Play as Human (WASD + Space/E/Q)
```bash
python play.py --human
```

### Watch AI Agents Battle
```bash
python play.py \
    --agent1 checkpoints/dqn/best_episode_900.pt \
    --agent2 checkpoints/ppo/best_episode_450.pt
```

### Human vs AI
```bash
python play.py --human \
    --agent1 checkpoints/dqn/best_episode_900.pt
```

## 🎯 Keyboard Controls (Human Mode)
- `W` - Forward
- `S` - Backward  
- `A` - Turn Left
- `D` - Turn Right
- `Space` - Fire Laser
- `E` - Fire Missile
- `Q` - Drop Mine
- `H` - Toggle HUD
- `M` - Toggle Minimap
- `ESC` - Quit

## 📁 Important Files
- `config/game_config.yaml` - Game mechanics
- `config/rl_config.yaml` - RL hyperparameters
- `config/training_config.yaml` - Training settings
- `checkpoints/<agent>/` - Saved models
- `logs/` - Training logs

## 🚀 Recommended Workflow

1. **Quick Test** (5 minutes)
```bash
python -m src.training.train --agent dqn --episodes 50
python play.py --human
```

2. **Serious Training** (2-3 hours)
```bash
python -m src.training.train --agent dqn --episodes 1000
python -m src.training.train --agent ppo --episodes 500
```

3. **Evaluate & Compare**
```bash
python -m src.training.evaluate --agent dqn --checkpoint checkpoints/dqn/best_*.pt --episodes 10 --render
python -m src.training.evaluate --agent ppo --checkpoint checkpoints/ppo/best_*.pt --episodes 10 --render
```

4. **Tournament**
```bash
python play.py --agent1 checkpoints/dqn/best_*.pt --agent2 checkpoints/ppo/best_*.pt
```

## ⚡ Quick Tips
- Start with 100-500 episodes to test
- Use `--render` only for debugging (slow)
- Check `checkpoints/<agent>/metrics.json` for progress
- GPU makes DQN/PPO 3-5x faster
- Best agents typically appear after 500-1000 episodes

## 🐛 Common Fixes
```bash
# Module not found
pip install -e .

# Pygame issues
pip install pygame --upgrade

# CUDA for GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Expected Performance
- **Q-Learning**: 300-500 reward after 2000 episodes
- **DQN**: 500-800 reward after 1000 episodes  
- **PPO**: 600-1000 reward after 500 episodes

## 🏆 Success = Agent completes 1+ laps consistently!

---
**Need help? Check `USAGE_GUIDE.md` or `README.md`**
