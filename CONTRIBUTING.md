# Combat Racing Championship - Contributing Guide

## 🎯 Project Vision

Create a **production-quality** reinforcement learning project that:
- Demonstrates mastery of RL, Python, and software engineering
- Serves as a reference for other students and researchers
- Is worthy of publication and portfolio inclusion
- Achieves 20/20 grade at ENSAM Morocco

## 🏗️ Development Principles

### Code Quality Standards

1. **PEP 8 Compliance**
   - Use `black` for formatting
   - Use `isort` for import sorting
   - Maximum line length: 100 characters

2. **Type Hints**
   - All function signatures must have type hints
   - Use `mypy` for type checking
   - Example:
     ```python
     def calculate_reward(state: np.ndarray, action: int) -> float:
         ...
     ```

3. **Documentation**
   - Google-style docstrings for all public functions/classes
   - Include: Description, Args, Returns, Raises, Example
   - Example:
     ```python
     def train_agent(env: gym.Env, episodes: int) -> Dict[str, float]:
         """
         Train RL agent on environment.
         
         Args:
             env: Gymnasium environment.
             episodes: Number of training episodes.
         
         Returns:
             Dictionary of training metrics.
         
         Raises:
             ValueError: If episodes < 1.
         
         Example:
             >>> metrics = train_agent(env, episodes=1000)
             >>> print(metrics["mean_reward"])
         """
     ```

4. **Testing**
   - Write unit tests for all core functionality
   - Aim for 70%+ code coverage
   - Use `pytest` framework
   - Example:
     ```python
     def test_car_movement():
         car = Car(position=(0, 0))
         car.apply_throttle(1.0)
         car.update(0.016)
         assert car.position[0] > 0
     ```

5. **Logging**
   - Use loguru for all logging
   - Appropriate log levels: DEBUG, INFO, WARNING, ERROR
   - Never use print() in production code
   - Example:
     ```python
     logger.info(f"Training episode {episode}")
     logger.error(f"Failed to load model: {error}")
     ```

### Architecture Principles

1. **Separation of Concerns**
   - Game engine separate from RL logic
   - Clear interfaces between components
   - Minimal coupling

2. **Design Patterns**
   - Use appropriate patterns (Factory, Strategy, Observer)
   - Document pattern usage

3. **Configuration**
   - All hyperparameters in YAML files
   - No magic numbers in code
   - Environment variables for secrets

4. **Performance**
   - Profile before optimizing
   - Use NumPy for numerical operations
   - Vectorize when possible

## 📝 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples:
```
feat(car): add drift mechanic

Implemented realistic drift physics for cars with configurable
drift factor and tire friction parameters.

Closes #42
```

```
fix(dqn): resolve memory leak in replay buffer

The replay buffer was not clearing old experiences, causing
memory usage to grow unbounded.
```

## 🔄 Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Develop & Test

```bash
# Run tests
pytest tests/ -v

# Check code style
black src/
flake8 src/
mypy src/

# Check coverage
pytest --cov=src --cov-report=html
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(scope): description"
```

### 5. Push & Create PR

```bash
git push origin feature/your-feature-name
# Create Pull Request on GitHub
```

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_car.py
import pytest
from src.game.entities import Car

class TestCar:
    """Test suite for Car entity."""
    
    @pytest.fixture
    def car(self):
        """Create car for testing."""
        return Car(position=(0, 0), agent_id=1)
    
    def test_initialization(self, car):
        """Test car initialization."""
        assert car.health == car.max_health
        assert car.is_alive is True
    
    def test_movement(self, car):
        """Test car movement."""
        car.apply_throttle(1.0)
        car.update(0.016)
        assert car.velocity[0] > 0
```

### Test Coverage Requirements

- **Core game engine**: 90%+
- **RL agents**: 80%+
- **Utilities**: 85%+
- **Overall**: 70%+

## 📚 Documentation Guidelines

### Code Comments

```python
# Good: Explain WHY
# Calculate Q-value using Bellman equation
q_value = reward + gamma * max(next_q_values)

# Bad: Explain WHAT (obvious from code)
# Add 1 to counter
counter += 1
```

### README Structure

Each major component should have README.md:
- Overview
- Usage examples
- API reference
- Configuration options
- Known issues

### API Documentation

Generate with Sphinx:
```bash
cd docs
make html
```

## 🎨 Visual Standards

### UI/UX

- Dark theme (#1a1a2e background)
- Neon accents (cyan #00ffff, magenta #ff00ff)
- Consistent spacing
- Clear visual hierarchy
- Smooth animations

### Graphics

- 60 FPS minimum
- Particle effects for impacts
- Screen shake on collisions
- Glow effects on power-ups
- Motion blur at high speeds

## 🔒 Security & Privacy

1. **Never commit**:
   - API keys
   - Passwords
   - Personal information
   - Large binary files

2. **Use**:
   - Environment variables for secrets
   - .gitignore for sensitive files
   - Encryption for stored credentials

## 📊 Performance Targets

- **Frame rate**: 60 FPS (game)
- **Inference time**: <100ms (agent decision)
- **Training time**: <2 hours for 1000 episodes (GPU)
- **Memory usage**: <4GB RAM

## 🐛 Debugging Guidelines

1. **Reproducibility**
   - Always set random seeds
   - Document system configuration
   - Save logs

2. **Error Handling**
   - Use try/except appropriately
   - Log errors with full traceback
   - Provide helpful error messages

3. **Debugging Tools**
   - Use breakpoints (not print statements)
   - Profile slow code
   - Visualize intermediate results

## 📈 Performance Optimization

1. **Measure first**
   ```python
   import cProfile
   cProfile.run('train_agent(env, 100)')
   ```

2. **Common optimizations**
   - Vectorize NumPy operations
   - Use GPU for neural networks
   - Batch processing
   - Caching expensive computations

3. **Memory optimization**
   - Clear unused variables
   - Use generators for large datasets
   - Monitor memory usage

## 🎓 Academic Standards

### Technical Report Requirements

1. **Structure**:
   - Abstract (200 words)
   - Introduction (2 pages)
   - Related Work (2-3 pages)
   - Methodology (4-5 pages)
   - Experiments & Results (4-5 pages)
   - Discussion (2 pages)
   - Conclusion & Future Work (1 page)
   - References (2+ pages)

2. **Figures & Tables**:
   - High-quality graphics
   - Proper captions
   - Referenced in text

3. **Math Notation**:
   - LaTeX formatting
   - Define all symbols
   - Number important equations

### Presentation Guidelines

1. **Slides**:
   - 15-20 slides for 15-minute talk
   - Minimal text, maximum visuals
   - Live demo if possible

2. **Structure**:
   - Title (1 slide)
   - Motivation (2 slides)
   - Approach (4-5 slides)
   - Results (5-6 slides)
   - Demo (2-3 minutes)
   - Conclusion (1 slide)

## 🤝 Code Review Checklist

Before submitting PR, verify:

- [ ] Code follows PEP 8
- [ ] All functions have docstrings
- [ ] Type hints present
- [ ] Tests written and passing
- [ ] No debug print statements
- [ ] Logging used appropriately
- [ ] Config changes documented
- [ ] README updated if needed
- [ ] No commented-out code
- [ ] Performance acceptable

## 📞 Getting Help

- **Documentation**: Check docs/ folder
- **Examples**: See experiments/analysis/
- **Issues**: Create GitHub issue
- **Discussions**: Use GitHub Discussions

## 🎯 Project Milestones

### Week 1: Foundation
- ✅ Project setup
- ✅ Core utilities
- ✅ Physics engine
- ✅ Game entities

### Week 2: Game Engine
- [ ] Track system
- [ ] Renderer
- [ ] Game loop
- [ ] Basic gameplay

### Week 3: RL Agents
- [ ] Gym environment
- [ ] Q-Learning
- [ ] DQN
- [ ] PPO

### Week 4: Training
- [ ] Training infrastructure
- [ ] Self-play
- [ ] Multi-agent
- [ ] Evaluation

### Week 5: Visualization
- [ ] Dashboard
- [ ] Plotting tools
- [ ] Replay system
- [ ] Attention viz

### Week 6: Polish
- [ ] Testing suite
- [ ] Documentation
- [ ] Technical report
- [ ] Presentation

## 🏆 Quality Goals

- **GitHub Stars**: 100+ (within 6 months)
- **Code Quality**: A+ (SonarQube)
- **Test Coverage**: 75%+
- **Documentation**: 90%+
- **Performance**: 60 FPS, <100ms inference
- **Grade**: 20/20

---

**Remember**: This is not just a project, it's a SHOWCASE of engineering excellence! 🚀

Every line of code should be something you're proud to show in an interview.
