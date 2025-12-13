"""
Physics Engine Tests
===================

Test physics simulation components.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pytest
import numpy as np
from src.game.physics import PhysicsBody, PhysicsEngine, CollisionShape


class TestPhysicsBody:
    """Test PhysicsBody class."""
    
    def test_initialization(self):
        """Test body initialization."""
        body = PhysicsBody(
            position=np.array([100, 200]),
            mass=10.0,
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=20.0
        )
        
        assert np.allclose(body.position, [100, 200])
        assert body.mass == 10.0
        assert body.collision_shape == CollisionShape.CIRCLE
        assert body.collision_radius == 20.0
        assert np.allclose(body.velocity, [0, 0])
    
    def test_apply_force(self):
        """Test force application."""
        body = PhysicsBody(position=np.array([0, 0]), mass=1.0)
        
        force = np.array([10, 0])
        body.apply_force(force)
        
        assert np.allclose(body.force, force)
    
    def test_apply_impulse(self):
        """Test impulse application."""
        body = PhysicsBody(position=np.array([0, 0]), mass=2.0)
        
        impulse = np.array([10, 0])
        body.apply_impulse(impulse)
        
        # impulse = mass * velocity_change
        # velocity_change = impulse / mass
        expected_velocity = impulse / body.mass
        assert np.allclose(body.velocity, expected_velocity)
    
    def test_update(self):
        """Test physics update."""
        body = PhysicsBody(position=np.array([0, 0]), mass=1.0)
        
        # Apply constant force
        force = np.array([10, 0])
        body.apply_force(force)
        
        # Update for 1 second
        dt = 1.0
        body.update(dt)
        
        # acceleration = force / mass = 10 / 1 = 10
        # velocity = acceleration * dt = 10 * 1 = 10
        # position = 0.5 * acceleration * dt^2 = 0.5 * 10 * 1 = 5
        assert np.allclose(body.velocity, [10, 0])
        assert np.allclose(body.position, [5, 0])


class TestPhysicsEngine:
    """Test PhysicsEngine class."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = PhysicsEngine(
            world_width=1920,
            world_height=1080,
            gravity=np.array([0, 9.8])
        )
        
        assert engine.world_width == 1920
        assert engine.world_height == 1080
        assert np.allclose(engine.gravity, [0, 9.8])
        assert len(engine.bodies) == 0
    
    def test_add_remove_body(self):
        """Test adding and removing bodies."""
        engine = PhysicsEngine()
        body = PhysicsBody(position=np.array([0, 0]))
        
        engine.add_body(body)
        assert len(engine.bodies) == 1
        assert body in engine.bodies
        
        engine.remove_body(body)
        assert len(engine.bodies) == 0
    
    def test_circle_collision_detection(self):
        """Test circle-circle collision detection."""
        engine = PhysicsEngine()
        
        body1 = PhysicsBody(
            position=np.array([0, 0]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=10
        )
        
        body2 = PhysicsBody(
            position=np.array([15, 0]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=10
        )
        
        engine.add_body(body1)
        engine.add_body(body2)
        
        # Should collide (distance = 15, radii sum = 20)
        collisions = engine._detect_collisions()
        assert len(collisions) == 1
        
        # Move bodies apart
        body2.position = np.array([25, 0])
        collisions = engine._detect_collisions()
        assert len(collisions) == 0
    
    def test_gravity(self):
        """Test gravity application."""
        engine = PhysicsEngine(gravity=np.array([0, 10]))
        body = PhysicsBody(position=np.array([0, 0]), mass=1.0)
        
        engine.add_body(body)
        engine.step(dt=1.0)
        
        # With gravity 10, velocity should be 10 after 1 second
        assert body.velocity[1] > 0


class TestCollisionDetection:
    """Test collision detection algorithms."""
    
    def test_circle_circle_collision(self):
        """Test circle-circle collision."""
        engine = PhysicsEngine()
        
        # Overlapping circles
        b1 = PhysicsBody(
            position=np.array([0, 0]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=5
        )
        b2 = PhysicsBody(
            position=np.array([8, 0]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=5
        )
        
        engine.add_body(b1)
        engine.add_body(b2)
        
        collisions = engine._detect_collisions()
        assert len(collisions) == 1
        
        # Check collision pair
        pair = collisions[0]
        assert b1 in pair and b2 in pair
    
    def test_no_collision(self):
        """Test non-colliding bodies."""
        engine = PhysicsEngine()
        
        b1 = PhysicsBody(
            position=np.array([0, 0]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=5
        )
        b2 = PhysicsBody(
            position=np.array([100, 100]),
            collision_shape=CollisionShape.CIRCLE,
            collision_radius=5
        )
        
        engine.add_body(b1)
        engine.add_body(b2)
        
        collisions = engine._detect_collisions()
        assert len(collisions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
