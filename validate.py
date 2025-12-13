"""
Project Validation Script
========================

Validates that all components are correctly installed and functional.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
from pathlib import Path
import importlib


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required = [
        "numpy",
        "torch",
        "gymnasium",
        "pygame",
        "matplotlib",
        "seaborn",
        "pandas",
        "omegaconf",
        "loguru",
        "tqdm"
    ]
    
    missing = []
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_project_structure():
    """Check if all required directories and files exist."""
    print_header("Checking Project Structure")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        "src",
        "src/game",
        "src/game/entities",
        "src/rl",
        "src/rl/agents",
        "src/rl/networks",
        "src/rl/utils",
        "src/training",
        "src/visualization",
        "src/utils",
        "configs",
        "configs/agents",
        "scripts",
        "tests",
        "docs"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "pytest.ini",
        "configs/config.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/play.py"
    ]
    
    all_exist = True
    
    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_exist = False
    
    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files and directories present!")
    else:
        print("\n⚠️  Some files or directories are missing!")
    
    return all_exist


def check_imports():
    """Check if all source modules can be imported."""
    print_header("Checking Module Imports")
    
    modules = [
        "src.game.physics",
        "src.game.entities",
        "src.game.track",
        "src.game.renderer",
        "src.game.engine",
        "src.rl.environment",
        "src.rl.agents.qlearning",
        "src.rl.agents.dqn",
        "src.rl.agents.ppo",
        "src.training.trainer",
        "src.utils.config",
        "src.utils.logger"
    ]
    
    all_imported = True
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module} - ERROR: {e}")
            all_imported = False
    
    if all_imported:
        print("\n✅ All modules can be imported!")
    else:
        print("\n⚠️  Some modules have import errors!")
    
    return all_imported


def check_configs():
    """Check if configuration files are valid."""
    print_header("Checking Configuration Files")
    
    from omegaconf import OmegaConf
    
    config_files = [
        "configs/config.yaml",
        "configs/environment.yaml",
        "configs/agents/qlearning.yaml",
        "configs/agents/dqn.yaml",
        "configs/agents/ppo.yaml"
    ]
    
    project_root = Path(__file__).parent
    all_valid = True
    
    for config_file in config_files:
        config_path = project_root / config_file
        try:
            cfg = OmegaConf.load(config_path)
            print(f"✅ {config_file}")
        except Exception as e:
            print(f"❌ {config_file} - ERROR: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✅ All configuration files are valid!")
    else:
        print("\n⚠️  Some configuration files have errors!")
    
    return all_valid


def run_quick_test():
    """Run a quick functionality test."""
    print_header("Running Quick Functionality Test")
    
    try:
        # Test physics
        from src.game.physics import PhysicsBody
        import numpy as np
        
        body = PhysicsBody(position=np.array([0, 0]))
        body.apply_force(np.array([10, 0]))
        body.update(0.1)
        print("✅ Physics engine")
        
        # Test environment
        from src.rl import CombatRacingEnv
        from src.game import create_oval_track
        
        track = create_oval_track()
        env = CombatRacingEnv(track=track, num_opponents=1)
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("✅ RL environment")
        
        # Test agent
        from src.rl.agents import DQNAgent
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cpu'
        )
        action = agent.select_action(obs)
        print("✅ RL agents")
        
        print("\n✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print(" Combat Racing Championship - Project Validation")
    print("=" * 60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Module Imports": check_imports(),
        "Configuration Files": check_configs(),
        "Functionality": run_quick_test()
    }
    
    print_header("Validation Summary")
    
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" ✅ ALL CHECKS PASSED - PROJECT IS READY!")
    else:
        print(" ⚠️  SOME CHECKS FAILED - PLEASE FIX ISSUES")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
