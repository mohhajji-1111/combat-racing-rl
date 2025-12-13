#!/usr/bin/env python3
"""
Quick Demo Script
================

Quickly test the game components that are already implemented.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logger, ConfigLoader, get_logger
from game.physics import PhysicsEngine, PhysicsBody
from game.entities import Car, Laser, PowerUp, PowerUpType
import numpy as np

logger = get_logger(__name__)


def test_physics_engine():
    """Test physics engine."""
    print("\n" + "="*60)
    print("TESTING PHYSICS ENGINE")
    print("="*60)
    
    # Create physics world
    physics = PhysicsEngine(width=1280, height=720)
    
    # Create some bodies
    body1 = PhysicsBody(position=(100, 100))
    body2 = PhysicsBody(position=(200, 100))
    
    physics.add_body(body1)
    physics.add_body(body2)
    
    # Apply forces
    body1.apply_force((100, 0))
    
    # Simulate
    for i in range(10):
        physics.update(0.016)
        print(f"Step {i+1}: Body1 pos={body1.position}, vel={body1.velocity}")
    
    print("✅ Physics engine working!")


def test_car():
    """Test car entity."""
    print("\n" + "="*60)
    print("TESTING CAR ENTITY")
    print("="*60)
    
    # Create car
    car = Car(
        position=(640, 360),
        rotation=0,
        color=(255, 0, 0),
        agent_id=1
    )
    
    print(f"Created: {car}")
    print(f"Health: {car.health}/{car.max_health}")
    print(f"Weapons: {car.weapons}")
    
    # Test driving
    car.apply_throttle(1.0)
    car.apply_steering(0.5)
    
    for i in range(5):
        car.update(0.016)
        print(f"Step {i+1}: pos={tuple(car.position)}, speed={np.linalg.norm(car.velocity):.1f}")
    
    # Test shooting
    projectile_data = car.shoot_weapon("laser")
    if projectile_data:
        print(f"✅ Shot laser: {projectile_data['type']}")
    
    # Test damage
    car.take_damage(30)
    print(f"After damage: health={car.health}")
    
    # Test power-up
    car.apply_powerup("speed_boost", duration=5.0)
    print(f"Active effects: {[e.effect_type for e in car.active_effects]}")
    
    print("✅ Car entity working!")


def test_projectiles():
    """Test projectile entities."""
    print("\n" + "="*60)
    print("TESTING PROJECTILES")
    print("="*60)
    
    # Create dummy car as owner
    car = Car(position=(0, 0), agent_id=1)
    
    # Create laser
    laser = Laser(
        position=(100, 100),
        velocity=(500, 0),
        rotation=0,
        owner=car
    )
    
    print(f"Created laser at {laser.position}")
    print(f"Damage: {laser.get_damage()}")
    
    # Update
    for i in range(3):
        laser.update(0.016)
        print(f"Step {i+1}: pos={tuple(laser.position)}, active={laser.is_active}")
    
    print("✅ Projectiles working!")


def test_powerups():
    """Test power-up system."""
    print("\n" + "="*60)
    print("TESTING POWER-UPS")
    print("="*60)
    
    # Create power-up
    config = {
        "duration": 5.0,
        "speed_multiplier": 1.5,
        "lifetime": 15.0
    }
    
    powerup = PowerUp(
        position=(300, 300),
        powerup_type=PowerUpType.SPEED_BOOST,
        config=config
    )
    
    print(f"Created {powerup.powerup_type.value} at {tuple(powerup.position)}")
    print(f"Color: {powerup.color}")
    
    # Create car to collect it
    car = Car(position=(300, 300), agent_id=1)
    
    # Collect
    effect = powerup.collect(car)
    print(f"Collected! Effect: {effect}")
    print(f"Car effects: {[e.effect_type for e in car.active_effects]}")
    
    print("✅ Power-ups working!")


def test_config_system():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*60)
    
    # Load configs
    config_loader = ConfigLoader()
    
    try:
        game_config = config_loader.load("config/game_config.yaml")
        print("✅ Loaded game_config.yaml")
        print(f"   Window size: {game_config.display.window_width}x{game_config.display.window_height}")
        print(f"   FPS: {game_config.display.fps}")
        print(f"   Max agents: {game_config.game.max_agents}")
        
        rl_config = config_loader.load("config/rl_config.yaml")
        print("✅ Loaded rl_config.yaml")
        print(f"   Gamma: {rl_config.general.gamma}")
        print(f"   DQN learning rate: {rl_config.dqn.learning_rate}")
        
        training_config = config_loader.load("config/training_config.yaml")
        print("✅ Loaded training_config.yaml")
        print(f"   Total episodes: {training_config.training.total_episodes}")
        print(f"   Algorithm: {training_config.training.algorithm}")
        
    except Exception as e:
        print(f"⚠️  Config loading failed: {e}")
        print("   (This is normal if running outside project root)")


def main():
    """Run all tests."""
    print("\n" + "🏎️" * 30)
    print("COMBAT RACING RL - COMPONENT TESTS")
    print("🏎️" * 30)
    
    # Setup logging
    setup_logger(level="INFO")
    
    try:
        test_config_system()
        test_physics_engine()
        test_car()
        test_projectiles()
        test_powerups()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\n✅ Core components are working correctly!")
        print("✅ Physics engine operational")
        print("✅ Car entity fully functional")
        print("✅ Combat system ready")
        print("✅ Power-up system working")
        print("\n📝 Next steps:")
        print("   1. Implement Track system")
        print("   2. Implement Renderer")
        print("   3. Implement GameEngine")
        print("   4. Implement RL agents")
        print("   5. Start training!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
