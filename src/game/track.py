"""
Track System
===========

Racing track with walls, checkpoints, and zones.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import distance, normalize_angle

logger = get_logger(__name__)


@dataclass
class Checkpoint:
    """Racing checkpoint."""
    position: np.ndarray  # Center position
    angle: float  # Orientation (radians)
    width: float  # Width of checkpoint
    index: int  # Checkpoint number
    is_finish_line: bool = False  # Is this the finish line?


@dataclass
class Wall:
    """Track wall segment."""
    start: np.ndarray  # Start point
    end: np.ndarray  # End point
    
    def get_normal(self) -> np.ndarray:
        """Get wall normal vector."""
        direction = self.end - self.start
        normal = np.array([-direction[1], direction[0]])
        return normal / (np.linalg.norm(normal) + 1e-8)


class Track:
    """
    Racing track with walls and checkpoints.
    
    Features:
        - Procedurally generated or custom tracks
        - Wall collision detection
        - Checkpoint system
        - Power-up spawn zones
        - Start positions
    """
    
    def __init__(
        self,
        name: str = "default",
        width: float = 1920,
        height: float = 1080,
    ):
        """
        Initialize track.
        
        Args:
            name: Track name.
            width: Track area width.
            height: Track area height.
        """
        self.name = name
        self.width = width
        self.height = height
        
        # Track components
        self.walls: List[Wall] = []
        self.checkpoints: List[Checkpoint] = []
        self.powerup_zones: List[Tuple[np.ndarray, float]] = []  # (center, radius)
        self.start_positions: List[Tuple[np.ndarray, float]] = []  # (position, angle)
        
        # Track properties
        self.track_bounds = np.array([width, height])
        
        logger.info(f"Track '{name}' initialized: {width}x{height}")
    
    def add_wall(self, start: np.ndarray, end: np.ndarray) -> None:
        """
        Add wall segment.
        
        Args:
            start: Wall start position.
            end: Wall end position.
        """
        self.walls.append(Wall(start=start, end=end))
    
    def add_checkpoint(
        self,
        position: np.ndarray,
        angle: float,
        width: float,
        is_finish_line: bool = False,
    ) -> None:
        """
        Add checkpoint.
        
        Args:
            position: Checkpoint center.
            angle: Checkpoint orientation.
            width: Checkpoint width.
            is_finish_line: Whether this is the finish line.
        """
        checkpoint = Checkpoint(
            position=position,
            angle=angle,
            width=width,
            index=len(self.checkpoints),
            is_finish_line=is_finish_line,
        )
        self.checkpoints.append(checkpoint)
    
    def add_powerup_zone(self, center: np.ndarray, radius: float) -> None:
        """
        Add power-up spawn zone.
        
        Args:
            center: Zone center.
            radius: Zone radius.
        """
        self.powerup_zones.append((center, radius))
    
    def add_start_position(self, position: np.ndarray, angle: float) -> None:
        """
        Add starting position.
        
        Args:
            position: Start position.
            angle: Start orientation.
        """
        self.start_positions.append((position, angle))
    
    def check_wall_collision(
        self,
        position: np.ndarray,
        radius: float,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if circle collides with walls.
        
        Args:
            position: Circle center.
            radius: Circle radius.
        
        Returns:
            Tuple of (collision, collision_normal).
        """
        for wall in self.walls:
            # Line segment to point distance
            wall_vec = wall.end - wall.start
            wall_length = np.linalg.norm(wall_vec)
            
            if wall_length < 1e-6:
                continue
            
            wall_dir = wall_vec / wall_length
            
            # Project position onto wall
            to_pos = position - wall.start
            projection = np.dot(to_pos, wall_dir)
            projection = np.clip(projection, 0, wall_length)
            
            # Closest point on wall
            closest = wall.start + projection * wall_dir
            
            # Distance to wall
            dist = distance(position, closest)
            
            if dist < radius:
                # Collision!
                normal = (position - closest) / (dist + 1e-8)
                return True, normal
        
        return False, None
    
    def check_checkpoint_crossing(
        self,
        prev_position: np.ndarray,
        current_position: np.ndarray,
        checkpoint_index: int,
    ) -> bool:
        """
        Check if car crossed checkpoint.
        
        Args:
            prev_position: Previous car position.
            current_position: Current car position.
            checkpoint_index: Index of checkpoint to check.
        
        Returns:
            True if checkpoint was crossed.
        """
        if checkpoint_index >= len(self.checkpoints):
            return False
        
        checkpoint = self.checkpoints[checkpoint_index]
        
        # Checkpoint line segment
        cp_angle = checkpoint.angle
        cp_half_width = checkpoint.width / 2
        
        cp_dir = np.array([np.cos(cp_angle), np.sin(cp_angle)])
        cp_perp = np.array([-cp_dir[1], cp_dir[0]])
        
        cp_start = checkpoint.position - cp_perp * cp_half_width
        cp_end = checkpoint.position + cp_perp * cp_half_width
        
        # Check if car trajectory crosses checkpoint line
        return self._line_segments_intersect(
            prev_position, current_position,
            cp_start, cp_end
        )
    
    def _line_segments_intersect(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> bool:
        """
        Check if line segments (p1-p2) and (p3-p4) intersect.
        
        Args:
            p1, p2: First line segment.
            p3, p4: Second line segment.
        
        Returns:
            True if segments intersect.
        """
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p3 - p1
        
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(cross) < 1e-8:
            return False  # Parallel
        
        t1 = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        t2 = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
        
        return 0 <= t1 <= 1 and 0 <= t2 <= 1
    
    def get_random_powerup_position(self) -> Optional[np.ndarray]:
        """
        Get random position in power-up zone.
        
        Returns:
            Random position or None if no zones.
        """
        if not self.powerup_zones:
            return None
        
        # Select random zone
        zone_center, zone_radius = self.powerup_zones[
            np.random.randint(len(self.powerup_zones))
        ]
        
        # Random position in zone
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, zone_radius)
        
        offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])
        
        return zone_center + offset
    
    def get_start_position(self, index: int = 0) -> Tuple[np.ndarray, float]:
        """
        Get starting position.
        
        Args:
            index: Start position index.
        
        Returns:
            Tuple of (position, angle).
        """
        if not self.start_positions:
            # Default to center
            return np.array([self.width / 2, self.height / 2]), 0.0
        
        index = index % len(self.start_positions)
        return self.start_positions[index]
    
    def save(self, path: Path) -> None:
        """
        Save track to file.
        
        Args:
            path: Save path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Track saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'Track':
        """
        Load track from file.
        
        Args:
            path: Load path.
        
        Returns:
            Loaded track.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Track not found: {path}")
        
        with open(path, 'rb') as f:
            track = pickle.load(f)
        
        logger.info(f"Track loaded from {path}")
        return track
    
    def get_stats(self) -> Dict:
        """Get track statistics."""
        return {
            "name": self.name,
            "dimensions": f"{self.width}x{self.height}",
            "walls": len(self.walls),
            "checkpoints": len(self.checkpoints),
            "powerup_zones": len(self.powerup_zones),
            "start_positions": len(self.start_positions),
        }


def create_oval_track(
    width: float = 1920,
    height: float = 1080,
    name: str = "oval",
) -> Track:
    """
    Create simple oval track.
    
    Args:
        width: Track area width.
        height: Track area height.
        name: Track name.
    
    Returns:
        Oval track.
    """
    track = Track(name=name, width=width, height=height)
    
    # Track center and dimensions
    center_x, center_y = width / 2, height / 2
    outer_radius_x, outer_radius_y = width * 0.4, height * 0.4
    inner_radius_x, inner_radius_y = width * 0.25, height * 0.25
    
    # Create walls (outer and inner boundaries)
    num_points = 32
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Outer wall
    outer_points = []
    for angle in angles:
        x = center_x + outer_radius_x * np.cos(angle)
        y = center_y + outer_radius_y * np.sin(angle)
        outer_points.append(np.array([x, y]))
    
    for i in range(len(outer_points)):
        track.add_wall(outer_points[i], outer_points[(i + 1) % len(outer_points)])
    
    # Inner wall
    inner_points = []
    for angle in angles:
        x = center_x + inner_radius_x * np.cos(angle)
        y = center_y + inner_radius_y * np.sin(angle)
        inner_points.append(np.array([x, y]))
    
    for i in range(len(inner_points)):
        track.add_wall(inner_points[i], inner_points[(i + 1) % len(inner_points)])
    
    # Checkpoints
    num_checkpoints = 8
    checkpoint_angles = np.linspace(0, 2 * np.pi, num_checkpoints, endpoint=False)
    
    for i, angle in enumerate(checkpoint_angles):
        checkpoint_radius = (outer_radius_x + inner_radius_x) / 2
        x = center_x + checkpoint_radius * np.cos(angle)
        y = center_y + checkpoint_radius * 0.5 * np.sin(angle)  # Elliptical
        
        track.add_checkpoint(
            position=np.array([x, y]),
            angle=angle + np.pi / 2,
            width=outer_radius_x - inner_radius_x,
            is_finish_line=(i == 0),
        )
    
    # Power-up zones
    for i in range(4):
        angle = i * np.pi / 2
        zone_radius = (outer_radius_x + inner_radius_x) / 2
        x = center_x + zone_radius * np.cos(angle)
        y = center_y + zone_radius * 0.5 * np.sin(angle)
        
        track.add_powerup_zone(
            center=np.array([x, y]),
            radius=50.0
        )
    
    # Start positions (grid formation)
    start_angle = 0
    start_radius = (outer_radius_x + inner_radius_x) / 2
    start_x = center_x + start_radius * np.cos(start_angle)
    start_y = center_y + start_radius * 0.5 * np.sin(start_angle)
    
    for i in range(4):
        offset = (i - 1.5) * 40  # Staggered grid
        perp = np.array([0, offset])
        
        track.add_start_position(
            position=np.array([start_x, start_y]) + perp,
            angle=start_angle + np.pi / 2
        )
    
    logger.info(f"Created oval track: {track.get_stats()}")
    return track
