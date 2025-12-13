#!/usr/bin/env python3
"""
Installation Verification Script
================================

Verify that all dependencies are installed correctly.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
from pathlib import Path


def check_imports():
    """Check if all required packages can be imported."""
    print("="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    packages = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium (OpenAI Gym)"),
        ("pygame", "Pygame"),
        ("yaml", "PyYAML"),
        ("loguru", "Loguru"),
        ("omegaconf", "OmegaConf"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas"),
    ]
    
    failed = []
    
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✅ {display_name:.<40} OK")
        except ImportError:
            print(f"❌ {display_name:.<40} MISSING")
            failed.append((module_name, display_name))
    
    print("="*60)
    
    if failed:
        print("\n⚠️  MISSING DEPENDENCIES:")
        for module_name, display_name in failed:
            print(f"   - {display_name} ({module_name})")
        print("\n📦 Install with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All dependencies installed correctly!")
        return True


def check_python_version():
    """Check Python version."""
    print("\n" + "="*60)
    print("CHECKING PYTHON VERSION")
    print("="*60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_pytorch_gpu():
    """Check if PyTorch can use GPU."""
    print("\n" + "="*60)
    print("CHECKING GPU SUPPORT")
    print("="*60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  No GPU detected - will use CPU")
            print("   Training will be slower without GPU")
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")


def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "src/game",
        "src/game/entities",
        "src/rl",
        "src/utils",
        "config",
        "scripts",
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "config/game_config.yaml",
        "config/rl_config.yaml",
        "config/training_config.yaml",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ MISSING")
            all_ok = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MISSING")
            all_ok = False
    
    if all_ok:
        print("\n✅ Project structure OK")
    else:
        print("\n⚠️  Some files/directories missing")
    
    return all_ok


def main():
    """Run all checks."""
    print("\n" + "🔧" * 30)
    print("COMBAT RACING RL - INSTALLATION CHECK")
    print("🔧" * 30 + "\n")
    
    checks = []
    
    # Run checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Dependencies", check_imports()))
    check_pytorch_gpu()  # Informational only
    checks.append(("Project Structure", check_project_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(result for _, result in checks)
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ Your environment is ready!")
        print("\n📝 Next steps:")
        print("   1. Run: python scripts/test_components.py")
        print("   2. Check: PROJECT_STATUS.md")
        print("   3. Start building remaining components")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n📝 Fix the issues above and run this script again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
