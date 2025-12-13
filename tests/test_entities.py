"""
Game Entities Tests
==================

Test car, projectile, and power-up entities.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pytest
import numpy as np
from src.game.entities import Car, PowerUp, PowerUpType
from src.game.entities.projectile import Laser, Missile, Mine


class TestCar:
    """Test Car entity."""
    
    def test_initialization(self):
        """Test car initialization."""
        car = Car(
            position=np.array([100, 200]),
            angle=0.5,
            max_health=100,
            max_speed=300
        )
        
        assert np.allclose(car.position, [100, 200])
        assert car.angle == 0.5
        assert car.health == 100
        assert car.max_health == 100
        assert car.max_speed == 300
        assert car.active
    
    def test_acceleration(self):
        """Test car acceleration."""
        car = Car(position=np.array([0, 0]), angle=0)
        
        initial_speed = np.linalg.norm(car.velocity)
        car.accelerate(dt=1.0)
        final_speed = np.linalg.norm(car.velocity)
        
        assert final_speed > initial_speed
    
    def test_braking(self):
        """Test car braking."""
        car = Car(position=np.array([0, 0]), angle=0)
        
        # Accelerate first
        car.velocity = np.array([100, 0])
        initial_speed = np.linalg.norm(car.velocity)
        
        car.brake(dt=1.0)
        final_speed = np.linalg.norm(car.velocity)
        
        assert final_speed < initial_speed
    
    def test_turning(self):
        """Test car turning."""
        car = Car(position=np.array([0, 0]), angle=0)
        
        initial_angle = car.angle
        car.turn_left(dt=0.1)
        
        assert car.angle != initial_angle
    
    def test_damage(self):
        """Test damage system."""
        car = Car(position=np.array([0, 0]), max_health=100)
        
        assert car.health == 100
        assert car.active
        
        car.take_damage(50)
        assert car.health == 50
        assert car.active
        
        car.take_damage(60)
        assert car.health == 0
        assert not car.active
    
    def test_weapon_firing(self):
        """Test weapon system."""
        car = Car(position=np.array([0, 0]), angle=0)
        
        # Fire laser
        projectile = car.fire_weapon("laser")
        assert projectile is not None
        assert isinstance(projectile, Laser)
        assert projectile.owner == car
        
        # Check cooldown
        assert not car.can_fire("laser")
    
    def test_powerup_collection(self):
        """Test power-up collection."""
        car = Car(position=np.array([0, 0]), max_speed=100)
        powerup = PowerUp(
            position=np.array([0, 0]),
            powerup_type=PowerUpType.SPEED_BOOST
        )
        
        initial_speed = car.max_speed
        car.collect_powerup(powerup)
        
        # Speed boost should increase max speed
        assert car.max_speed > initial_speed
        assert not powerup.active
    
    def test_checkpoint_tracking(self):
        """Test checkpoint system."""
        car = Car(position=np.array([0, 0]))
        
        assert car.current_checkpoint == 0
        assert car.laps_completed == 0
        
        car.current_checkpoint = 5
        assert car.current_checkpoint == 5


class TestProjectiles:
    """Test projectile entities."""
    
    def test_laser(self):
        """Test laser projectile."""
        laser = Laser(
            position=np.array([0, 0]),
            angle=0,
            owner=None
        )
        
        assert laser.active
        assert laser.damage == 25
        assert laser.projectile_type == "laser"
    
    def test_missile(self):
        """Test missile projectile."""
        missile = Missile(
            position=np.array([0, 0]),
            angle=0,
            owner=None,
            target=None
        )
        
        assert missile.active
        assert missile.damage == 40
        assert missile.projectile_type == "missile"
    
    def test_mine(self):
        """Test mine projectile."""
        mine = Mine(
            position=np.array([0, 0]),
            owner=None
        )
        
        assert mine.active
        assert mine.damage == 50
        assert mine.projectile_type == "mine"
        assert not mine.armed  # Should start unarmed
    
    def test_mine_activation(self):
        """Test mine activation delay."""
        mine = Mine(position=np.array([0, 0]), owner=None)
        
        # Mine should become armed after delay
        assert not mine.armed
        
        mine.update(dt=0.6)  # 600ms > 500ms activation
        assert mine.armed


class TestPowerUps:
    """Test power-up entities."""
    
    def test_powerup_types(self):
        """Test different power-up types."""
        for powerup_type in PowerUpType:
            powerup = PowerUp(
                position=np.array([0, 0]),
                powerup_type=powerup_type
            )
            
            assert powerup.active
            assert powerup.powerup_type == powerup_type
    
    def test_speed_boost(self):
        """Test speed boost power-up."""
        car = Car(position=np.array([0, 0]), max_speed=100)
        powerup = PowerUp(
            position=np.array([0, 0]),
            powerup_type=PowerUpType.SPEED_BOOST
        )
        
        car.collect_powerup(powerup)
        assert car.max_speed == 150  # 1.5x multiplier
    
    def test_shield(self):
        """Test shield power-up."""
        car = Car(position=np.array([0, 0]))
        powerup = PowerUp(
            position=np.array([0, 0]),
            powerup_type=PowerUpType.SHIELD
        )
        
        car.collect_powerup(powerup)
        assert car.shield_active
    
    def test_health_pack(self):
        """Test health pack power-up."""
        car = Car(position=np.array([0, 0]), max_health=100)
        car.health = 30
        
        powerup = PowerUp(
            position=np.array([0, 0]),
            powerup_type=PowerUpType.HEALTH_PACK
        )
        
        car.collect_powerup(powerup)
        assert car.health == 80  # 30 + 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
