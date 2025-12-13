"""
Physics Engine
=============

High-performance 2D physics simulation for racing game.

Features:
    - Velocity and acceleration
    - Friction and drag
    - Collision detection (AABB, Circle, Polygon)
    - Collision response
    - Ray casting for sensors

Author: Combat Racing RL Team
Date: 2024-2025
"""

import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from ..utils.logger import get_logger
from ..utils.helpers import normalize_angle, distance

logger = get_logger(__name__)


@dataclass
class CollisionInfo:
    """Information about a collision event."""
    entity1: any
    entity2: any
    point: Tuple[float, float]
    normal: Tuple[float, float]
    penetration: float


class PhysicsBody:
    """
    Base physics body for all game objects.
    
    Attributes:
        position: (x, y) coordinates
        velocity: (vx, vy) velocity vector
        rotation: Angle in radians
        angular_velocity: Rotation speed in rad/s
        mass: Object mass (kg)
        friction: Friction coefficient [0, 1]
        bounce: Bounciness coefficient [0, 1]
    """
    
    def __init__(
        self,
        position: Tuple[float, float] = (0, 0),
        velocity: Tuple[float, float] = (0, 0),
        rotation: float = 0.0,
        mass: float = 1.0,
        friction: float = 0.95,
        bounce: float = 0.3,
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.rotation = rotation
        self.angular_velocity = 0.0
        self.mass = mass
        self.friction = friction
        self.bounce = bounce
        
        # Forces
        self.force = np.array([0.0, 0.0])
        self.torque = 0.0
        
        # Collision
        self.collision_shape = "circle"  # "circle", "box", "polygon"
        self.collision_radius = 20.0
        self.collision_size = (40, 30)  # width, height for box
        self.collision_layer = 0
        self.collision_mask = 0xFFFFFFFF
        
        # State
        self.is_static = False
        self.is_trigger = False  # Trigger collisions don't apply physics
    
    def apply_force(self, force: Tuple[float, float]) -> None:
        """Apply force vector to body."""
        if not self.is_static:
            self.force += np.array(force)
    
    def apply_impulse(self, impulse: Tuple[float, float]) -> None:
        """Apply instantaneous impulse (change in velocity)."""
        if not self.is_static:
            self.velocity += np.array(impulse) / self.mass
    
    def apply_torque(self, torque: float) -> None:
        """Apply rotational force."""
        if not self.is_static:
            self.torque += torque
    
    def update(self, dt: float, max_velocity: float = 1000.0) -> None:
        """
        Update physics body state.
        
        Args:
            dt: Delta time (seconds).
            max_velocity: Maximum velocity magnitude.
        """
        if self.is_static:
            return
        
        # Apply acceleration from forces
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        
        # Apply friction
        self.velocity *= self.friction
        
        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > max_velocity:
            self.velocity = (self.velocity / speed) * max_velocity
        
        # Update position
        self.position += self.velocity * dt
        
        # Update rotation
        self.angular_velocity += self.torque * dt
        self.angular_velocity *= 0.98  # Angular friction
        self.rotation += self.angular_velocity * dt
        self.rotation = normalize_angle(self.rotation)
        
        # Reset forces
        self.force = np.array([0.0, 0.0])
        self.torque = 0.0


class PhysicsEngine:
    """
    Professional 2D physics engine for the racing game.
    
    Features:
        - Multiple collision shapes
        - Broad-phase collision detection (spatial hashing)
        - Narrow-phase collision detection
        - Collision response with proper physics
        - Ray casting for sensors
        - Collision layers and masks
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        cell_size: int = 100,
    ):
        """
        Initialize physics engine.
        
        Args:
            width: World width in pixels.
            height: World height in pixels.
            cell_size: Size of spatial hash cells for broad-phase.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Bodies registry
        self.bodies: List[PhysicsBody] = []
        self.static_bodies: List[PhysicsBody] = []
        
        # Spatial hash for broad-phase collision detection
        self.spatial_hash: dict[Tuple[int, int], Set[PhysicsBody]] = {}
        
        # Collision tracking
        self.collisions: List[CollisionInfo] = []
        
        logger.info(f"PhysicsEngine initialized: {width}x{height}, cell_size={cell_size}")
    
    def add_body(self, body: PhysicsBody) -> None:
        """Add physics body to simulation."""
        if body.is_static:
            self.static_bodies.append(body)
        else:
            self.bodies.append(body)
        self._add_to_spatial_hash(body)
    
    def remove_body(self, body: PhysicsBody) -> None:
        """Remove physics body from simulation."""
        if body in self.bodies:
            self.bodies.remove(body)
        if body in self.static_bodies:
            self.static_bodies.remove(body)
        self._remove_from_spatial_hash(body)
    
    def _add_to_spatial_hash(self, body: PhysicsBody) -> None:
        """Add body to spatial hash grid."""
        cells = self._get_hash_cells(body)
        for cell in cells:
            if cell not in self.spatial_hash:
                self.spatial_hash[cell] = set()
            self.spatial_hash[cell].add(body)
    
    def _remove_from_spatial_hash(self, body: PhysicsBody) -> None:
        """Remove body from spatial hash grid."""
        cells = self._get_hash_cells(body)
        for cell in cells:
            if cell in self.spatial_hash:
                self.spatial_hash[cell].discard(body)
    
    def _get_hash_cells(self, body: PhysicsBody) -> List[Tuple[int, int]]:
        """Get spatial hash cells occupied by body."""
        x, y = body.position
        radius = body.collision_radius if body.collision_shape == "circle" else max(body.collision_size)
        
        min_x = int((x - radius) // self.cell_size)
        max_x = int((x + radius) // self.cell_size)
        min_y = int((y - radius) // self.cell_size)
        max_y = int((y + radius) // self.cell_size)
        
        cells = []
        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                cells.append((cx, cy))
        
        return cells
    
    def update(self, dt: float) -> None:
        """
        Update all physics bodies and handle collisions.
        
        Args:
            dt: Delta time (seconds).
        """
        # Clear previous collisions
        self.collisions.clear()
        
        # Update spatial hash
        for body in self.bodies:
            self._remove_from_spatial_hash(body)
        
        # Update dynamic bodies
        for body in self.bodies:
            body.update(dt)
            self._add_to_spatial_hash(body)
        
        # Collision detection and response
        self._handle_collisions()
    
    def _handle_collisions(self) -> None:
        """Detect and resolve all collisions."""
        # Broad-phase: Find potential collision pairs
        checked_pairs = set()
        
        for body in self.bodies:
            if body.is_trigger:
                continue
            
            # Get nearby bodies from spatial hash
            cells = self._get_hash_cells(body)
            nearby_bodies = set()
            for cell in cells:
                if cell in self.spatial_hash:
                    nearby_bodies.update(self.spatial_hash[cell])
            
            # Narrow-phase: Check actual collisions
            for other in nearby_bodies:
                if body == other:
                    continue
                
                # Avoid duplicate checks
                pair = (min(id(body), id(other)), max(id(body), id(other)))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Check collision layers
                if not (body.collision_layer & other.collision_mask):
                    continue
                
                # Detect collision
                collision = self._detect_collision(body, other)
                if collision:
                    self.collisions.append(collision)
                    
                    # Resolve collision
                    if not (body.is_trigger or other.is_trigger):
                        self._resolve_collision(collision)
    
    def _detect_collision(
        self,
        body1: PhysicsBody,
        body2: PhysicsBody,
    ) -> Optional[CollisionInfo]:
        """
        Detect collision between two bodies.
        
        Returns:
            CollisionInfo if collision detected, None otherwise.
        """
        # Circle-Circle collision
        if body1.collision_shape == "circle" and body2.collision_shape == "circle":
            return self._circle_circle_collision(body1, body2)
        
        # Box-Box collision (AABB)
        elif body1.collision_shape == "box" and body2.collision_shape == "box":
            return self._box_box_collision(body1, body2)
        
        # Circle-Box collision
        elif body1.collision_shape == "circle" and body2.collision_shape == "box":
            return self._circle_box_collision(body1, body2)
        elif body1.collision_shape == "box" and body2.collision_shape == "circle":
            collision = self._circle_box_collision(body2, body1)
            if collision:
                # Swap entities and reverse normal
                collision.entity1, collision.entity2 = collision.entity2, collision.entity1
                collision.normal = (-collision.normal[0], -collision.normal[1])
            return collision
        
        return None
    
    def _circle_circle_collision(
        self,
        body1: PhysicsBody,
        body2: PhysicsBody,
    ) -> Optional[CollisionInfo]:
        """Circle-circle collision detection."""
        dist = distance(tuple(body1.position), tuple(body2.position))
        radius_sum = body1.collision_radius + body2.collision_radius
        
        if dist < radius_sum:
            # Calculate collision normal
            delta = body2.position - body1.position
            if dist > 0:
                normal = delta / dist
            else:
                normal = np.array([1.0, 0.0])
            
            # Collision point (midpoint between surfaces)
            point = body1.position + normal * body1.collision_radius
            penetration = radius_sum - dist
            
            return CollisionInfo(
                entity1=body1,
                entity2=body2,
                point=tuple(point),
                normal=tuple(normal),
                penetration=penetration,
            )
        
        return None
    
    def _box_box_collision(
        self,
        body1: PhysicsBody,
        body2: PhysicsBody,
    ) -> Optional[CollisionInfo]:
        """AABB box-box collision detection."""
        w1, h1 = body1.collision_size
        w2, h2 = body2.collision_size
        
        # AABB bounds
        left1 = body1.position[0] - w1 / 2
        right1 = body1.position[0] + w1 / 2
        top1 = body1.position[1] - h1 / 2
        bottom1 = body1.position[1] + h1 / 2
        
        left2 = body2.position[0] - w2 / 2
        right2 = body2.position[0] + w2 / 2
        top2 = body2.position[1] - h2 / 2
        bottom2 = body2.position[1] + h2 / 2
        
        # Check overlap
        if (left1 < right2 and right1 > left2 and
            top1 < bottom2 and bottom1 > top2):
            
            # Calculate penetration and normal
            overlap_x = min(right1, right2) - max(left1, left2)
            overlap_y = min(bottom1, bottom2) - max(top1, top2)
            
            if overlap_x < overlap_y:
                normal = (1.0, 0.0) if body1.position[0] < body2.position[0] else (-1.0, 0.0)
                penetration = overlap_x
            else:
                normal = (0.0, 1.0) if body1.position[1] < body2.position[1] else (0.0, -1.0)
                penetration = overlap_y
            
            # Collision point (center of overlap)
            point = (
                (max(left1, left2) + min(right1, right2)) / 2,
                (max(top1, top2) + min(bottom1, bottom2)) / 2,
            )
            
            return CollisionInfo(
                entity1=body1,
                entity2=body2,
                point=point,
                normal=normal,
                penetration=penetration,
            )
        
        return None
    
    def _circle_box_collision(
        self,
        circle: PhysicsBody,
        box: PhysicsBody,
    ) -> Optional[CollisionInfo]:
        """Circle-box collision detection."""
        w, h = box.collision_size
        box_min = box.position - np.array([w/2, h/2])
        box_max = box.position + np.array([w/2, h/2])
        
        # Find closest point on box to circle center
        closest = np.array([
            max(box_min[0], min(circle.position[0], box_max[0])),
            max(box_min[1], min(circle.position[1], box_max[1])),
        ])
        
        # Check if closest point is inside circle
        dist = np.linalg.norm(circle.position - closest)
        
        if dist < circle.collision_radius:
            if dist > 0:
                normal = (circle.position - closest) / dist
            else:
                # Circle center inside box
                normal = np.array([1.0, 0.0])
            
            penetration = circle.collision_radius - dist
            point = closest
            
            return CollisionInfo(
                entity1=circle,
                entity2=box,
                point=tuple(point),
                normal=tuple(normal),
                penetration=penetration,
            )
        
        return None
    
    def _resolve_collision(self, collision: CollisionInfo) -> None:
        """
        Resolve collision with proper physics response.
        
        Uses impulse-based collision response with restitution.
        """
        body1 = collision.entity1
        body2 = collision.entity2
        normal = np.array(collision.normal)
        
        # Separate bodies
        total_mass = body1.mass + body2.mass
        if not body1.is_static and not body2.is_static:
            # Move both bodies
            correction = (collision.penetration / total_mass) * normal
            body1.position -= correction * body2.mass
            body2.position += correction * body1.mass
        elif not body1.is_static:
            # Move only body1
            body1.position -= collision.penetration * normal
        elif not body2.is_static:
            # Move only body2
            body2.position += collision.penetration * normal
        
        # Calculate relative velocity
        rel_velocity = body2.velocity - body1.velocity
        vel_along_normal = np.dot(rel_velocity, normal)
        
        # Don't resolve if bodies are separating
        if vel_along_normal > 0:
            return
        
        # Calculate restitution (bounciness)
        restitution = min(body1.bounce, body2.bounce)
        
        # Calculate impulse
        j = -(1 + restitution) * vel_along_normal
        j /= (1 / body1.mass + 1 / body2.mass) if not (body1.is_static or body2.is_static) else 1
        
        impulse = j * normal
        
        # Apply impulse
        if not body1.is_static:
            body1.velocity -= impulse / body1.mass
        if not body2.is_static:
            body2.velocity += impulse / body2.mass
    
    def raycast(
        self,
        origin: Tuple[float, float],
        direction: float,
        max_distance: float = 1000.0,
        layer_mask: int = 0xFFFFFFFF,
    ) -> Optional[Tuple[Tuple[float, float], PhysicsBody, float]]:
        """
        Cast a ray and return first hit.
        
        Args:
            origin: Ray starting point (x, y).
            direction: Ray angle in radians.
            max_distance: Maximum ray distance.
            layer_mask: Collision layer mask.
        
        Returns:
            Tuple of (hit_point, hit_body, distance) or None if no hit.
        """
        ray_end = (
            origin[0] + math.cos(direction) * max_distance,
            origin[1] + math.sin(direction) * max_distance,
        )
        
        closest_hit = None
        closest_distance = max_distance
        
        # Check all bodies
        all_bodies = self.bodies + self.static_bodies
        for body in all_bodies:
            if not (body.collision_layer & layer_mask):
                continue
            
            # Ray-circle intersection
            if body.collision_shape == "circle":
                hit = self._raycast_circle(origin, ray_end, body)
                if hit:
                    hit_point, hit_distance = hit
                    if hit_distance < closest_distance:
                        closest_hit = (hit_point, body, hit_distance)
                        closest_distance = hit_distance
        
        return closest_hit
    
    def _raycast_circle(
        self,
        ray_start: Tuple[float, float],
        ray_end: Tuple[float, float],
        body: PhysicsBody,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Ray-circle intersection test."""
        # Vector from ray start to circle center
        to_circle = body.position - np.array(ray_start)
        ray_dir = np.array(ray_end) - np.array(ray_start)
        ray_length = np.linalg.norm(ray_dir)
        
        if ray_length == 0:
            return None
        
        ray_dir = ray_dir / ray_length
        
        # Project circle center onto ray
        projection = np.dot(to_circle, ray_dir)
        
        # Closest point on ray to circle center
        if projection < 0:
            closest = np.array(ray_start)
        elif projection > ray_length:
            closest = np.array(ray_end)
        else:
            closest = np.array(ray_start) + ray_dir * projection
        
        # Check if closest point is within circle
        dist_to_center = np.linalg.norm(body.position - closest)
        
        if dist_to_center <= body.collision_radius:
            # Calculate actual hit point
            hit_distance = projection - math.sqrt(body.collision_radius**2 - dist_to_center**2)
            if hit_distance < 0:
                hit_distance = 0
            
            hit_point = np.array(ray_start) + ray_dir * hit_distance
            return (tuple(hit_point), hit_distance)
        
        return None
    
    def get_collisions(self) -> List[CollisionInfo]:
        """Get list of all collisions from last update."""
        return self.collisions.copy()
