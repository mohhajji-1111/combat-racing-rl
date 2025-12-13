"""
Game Renderer
============

Pygame-based rendering system for visualization.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pygame
from pathlib import Path

from .track import Track
from .entities.car import Car, CarState
from .entities.projectile import Projectile
from .entities.powerup import PowerUp
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Renderer:
    """
    Pygame renderer for game visualization.
    
    Features:
        - Track rendering
        - Car rendering with health bars
        - Projectile effects
        - Power-up visualization
        - HUD with stats
        - Minimap
        - Camera following
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        fullscreen: bool = False,
    ):
        """
        Initialize renderer.
        
        Args:
            width: Window width.
            height: Window height.
            fps: Target FPS.
            fullscreen: Start in fullscreen mode.
        """
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.fps = fps
        
        # Window scaling (make it smaller for display)
        self.scale = 0.67  # Scale down to ~1280x720
        self.window_width = int(width * self.scale)
        self.window_height = int(height * self.scale)
        
        # Create window
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), flags)
        pygame.display.set_caption("Combat Racing Championship")
        
        # Create render surface at full resolution
        self.render_surface = pygame.Surface((width, height))
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.SysFont('arial', 16)
        self.font_medium = pygame.font.SysFont('arial', 24)
        self.font_large = pygame.font.SysFont('arial', 36, bold=True)
        
        # Camera
        self.camera_position = np.array([0.0, 0.0])
        self.camera_target: Optional[Car] = None
        self.camera_smoothing = 0.1
        
        # Colors
        self.colors = {
            "background": (20, 20, 30),
            "track": (40, 40, 50),
            "wall": (200, 200, 200),
            "checkpoint": (255, 255, 0, 100),
            "finish_line": (255, 255, 255, 150),
            "car_blue": (50, 150, 255),
            "car_red": (255, 50, 50),
            "car_green": (50, 255, 50),
            "car_yellow": (255, 255, 50),
            "health_bar_bg": (100, 100, 100),
            "health_bar_fill": (0, 255, 0),
            "health_bar_damaged": (255, 165, 0),
            "health_bar_critical": (255, 0, 0),
            "powerup": (255, 100, 255),
            "hud_bg": (0, 0, 0, 180),
            "hud_text": (255, 255, 255),
            "minimap_bg": (0, 0, 0, 150),
        }
        
        # HUD settings
        self.show_hud = True
        self.show_minimap = True
        self.minimap_size = 200
        
        logger.info(f"Renderer initialized: {width}x{height} @ {fps}fps (window: {self.window_width}x{self.window_height})")
    
    def set_camera_target(self, car: Optional[Car]) -> None:
        """
        Set car for camera to follow.
        
        Args:
            car: Car to follow (None for fixed camera).
        """
        self.camera_target = car
    
    def update_camera(self) -> None:
        """Update camera position (smooth following)."""
        if self.camera_target is None:
            # Center camera on the track (fixed camera)
            self.camera_position = np.array([self.width / 2, self.height / 2])
            return
        
        target_pos = self.camera_target.position
        
        # Smooth interpolation
        self.camera_position += (target_pos - self.camera_position) * self.camera_smoothing
    
    def world_to_screen(self, position: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            position: World position.
        
        Returns:
            Screen coordinates (x, y).
        """
        screen_x = int(position[0] - self.camera_position[0] + self.width / 2)
        screen_y = int(position[1] - self.camera_position[1] + self.height / 2)
        return screen_x, screen_y
    
    def render_frame(
        self,
        track: Track,
        cars: List[Car],
        projectiles: List[Projectile],
        powerups: List[PowerUp],
        frame_info: Optional[Dict] = None,
    ) -> None:
        """
        Render complete frame.
        
        Args:
            track: Racing track.
            cars: List of cars.
            projectiles: List of active projectiles.
            powerups: List of power-ups.
            frame_info: Additional info to display (episode, step, etc).
        """
        # Update camera
        self.update_camera()
        
        # Clear screen
        self.render_surface.fill(self.colors["background"])
        
        # Render track
        self._render_track(track)
        
        # Render entities
        self._render_powerups(powerups)
        self._render_projectiles(projectiles)
        self._render_cars(cars)
        
        # Render HUD
        if self.show_hud:
            self._render_hud(cars, frame_info)
        
        # Render minimap
        if self.show_minimap:
            self._render_minimap(track, cars)
        
        # Scale and blit to window
        scaled = pygame.transform.scale(self.render_surface, (self.window_width, self.window_height))
        self.screen.blit(scaled, (0, 0))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _render_track(self, track: Track) -> None:
        """Render track walls and checkpoints."""
        # Checkpoints
        for checkpoint in track.checkpoints:
            color = self.colors["finish_line"] if checkpoint.is_finish_line else self.colors["checkpoint"]
            
            # Checkpoint line
            angle = checkpoint.angle
            half_width = checkpoint.width / 2
            perp = np.array([-np.sin(angle), np.cos(angle)])
            
            start = checkpoint.position - perp * half_width
            end = checkpoint.position + perp * half_width
            
            screen_start = self.world_to_screen(start)
            screen_end = self.world_to_screen(end)
            
            # Create surface with alpha
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(surf, color, screen_start, screen_end, 5)
            self.render_surface.blit(surf, (0, 0))
        
        # Walls
        for wall in track.walls:
            screen_start = self.world_to_screen(wall.start)
            screen_end = self.world_to_screen(wall.end)
            pygame.draw.line(self.render_surface, self.colors["wall"], screen_start, screen_end, 3)
    
    def _render_cars(self, cars: List[Car]) -> None:
        """Render cars with health bars."""
        car_colors = [
            self.colors["car_blue"],
            self.colors["car_red"],
            self.colors["car_green"],
            self.colors["car_yellow"],
        ]
        
        for i, car in enumerate(cars):
            if car.state != CarState.ACTIVE:
                continue
            
            color = car_colors[i % len(car_colors)]
            
            # Car body (rectangle rotated)
            screen_pos = self.world_to_screen(car.position)
            
            # Create rotated rectangle
            car_rect = pygame.Surface((int(car.collision_size[0]), int(car.collision_size[1])), pygame.SRCALPHA)
            pygame.draw.rect(car_rect, color, car_rect.get_rect(), border_radius=5)
            
            # Shield effect
            if car.shield_active:
                pygame.draw.rect(
                    car_rect,
                    (100, 200, 255, 100),
                    car_rect.get_rect(),
                    width=3,
                    border_radius=5
                )
            
            # Rotate and blit
            rotated = pygame.transform.rotate(car_rect, -np.degrees(car.rotation))
            rect = rotated.get_rect(center=screen_pos)
            self.render_surface.blit(rotated, rect.topleft)
            
            # Health bar
            self._render_health_bar(car, screen_pos)
            
            # Car name/ID
            name_surf = self.font_small.render(f"Car {i}", True, self.colors["hud_text"])
            name_rect = name_surf.get_rect(center=(screen_pos[0], screen_pos[1] - 40))
            self.render_surface.blit(name_surf, name_rect)
    
    def _render_health_bar(self, car: Car, screen_pos: Tuple[int, int]) -> None:
        """Render health bar above car."""
        bar_width = 50
        bar_height = 6
        bar_x = screen_pos[0] - bar_width // 2
        bar_y = screen_pos[1] - 30
        
        # Background
        pygame.draw.rect(
            self.screen,
            self.colors["health_bar_bg"],
            (bar_x, bar_y, bar_width, bar_height)
        )
        
        # Health fill
        health_percent = car.health / car.max_health
        fill_width = int(bar_width * health_percent)
        
        if health_percent > 0.6:
            fill_color = self.colors["health_bar_fill"]
        elif health_percent > 0.3:
            fill_color = self.colors["health_bar_damaged"]
        else:
            fill_color = self.colors["health_bar_critical"]
        
        if fill_width > 0:
            pygame.draw.rect(
                self.screen,
                fill_color,
                (bar_x, bar_y, fill_width, bar_height)
            )
    
    def _render_projectiles(self, projectiles: List[Projectile]) -> None:
        """Render projectiles."""
        for proj in projectiles:
            if not proj.active:
                continue
            
            screen_pos = self.world_to_screen(proj.position)
            
            # Different rendering for different types
            if proj.projectile_type == "laser":
                pygame.draw.circle(self.render_surface, (255, 0, 0), screen_pos, 4)
            elif proj.projectile_type == "missile":
                pygame.draw.circle(self.render_surface, (255, 165, 0), screen_pos, 6)
                # Exhaust trail
                pygame.draw.circle(self.render_surface, (255, 100, 0), screen_pos, 3)
            elif proj.projectile_type == "mine":
                pygame.draw.circle(self.render_surface, (100, 100, 100), screen_pos, 8)
    
    def _render_powerups(self, powerups: List[PowerUp]) -> None:
        """Render power-ups."""
        for powerup in powerups:
            if powerup.is_collected:
                continue
            
            screen_pos = self.world_to_screen(powerup.position)
            
            # Rotating square
            size = 20
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(surf, self.colors["powerup"], surf.get_rect(), border_radius=3)
            
            # Rotate based on type
            rotation = (pygame.time.get_ticks() / 10) % 360
            rotated = pygame.transform.rotate(surf, rotation)
            rect = rotated.get_rect(center=screen_pos)
            self.render_surface.blit(rotated, rect.topleft)
    
    def _render_hud(self, cars: List[Car], frame_info: Optional[Dict]) -> None:
        """Render heads-up display."""
        # HUD background
        hud_surf = pygame.Surface((self.width, 100), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, self.colors["hud_bg"], hud_surf.get_rect())
        self.render_surface.blit(hud_surf, (0, 0))
        
        # Frame info
        if frame_info:
            y_offset = 10
            for key, value in frame_info.items():
                text = self.font_medium.render(
                    f"{key}: {value}",
                    True,
                    self.colors["hud_text"]
                )
                self.render_surface.blit(text, (10, y_offset))
                y_offset += 25
        
        # Car stats (leaderboard)
        if cars:
            sorted_cars = sorted(
                enumerate(cars),
                key=lambda x: (-x[1].current_lap, -x[1].checkpoint_index),
                reverse=False
            )
            
            x_offset = self.width - 250
            y_offset = 10
            
            title = self.font_large.render("Leaderboard", True, self.colors["hud_text"])
            self.render_surface.blit(title, (x_offset, y_offset))
            y_offset += 40
            
            for rank, (idx, car) in enumerate(sorted_cars[:5], 1):
                if car.state != CarState.ACTIVE:
                    continue
                
                text = self.font_medium.render(
                    f"{rank}. Car {idx} - Lap {car.current_lap + 1}",
                    True,
                    self.colors["hud_text"]
                )
                self.render_surface.blit(text, (x_offset, y_offset))
                y_offset += 25
    
    def _render_minimap(self, track: Track, cars: List[Car]) -> None:
        """Render minimap."""
        # Minimap position (bottom-right)
        minimap_x = self.width - self.minimap_size - 20
        minimap_y = self.height - self.minimap_size - 20
        
        # Background
        minimap_surf = pygame.Surface(
            (self.minimap_size, self.minimap_size),
            pygame.SRCALPHA
        )
        pygame.draw.rect(
            minimap_surf,
            self.colors["minimap_bg"],
            minimap_surf.get_rect(),
            border_radius=10
        )
        
        # Scale factor
        scale_x = self.minimap_size / track.width
        scale_y = self.minimap_size / track.height
        scale = min(scale_x, scale_y) * 0.9
        
        # Draw track outline
        for wall in track.walls:
            start = (
                int(wall.start[0] * scale),
                int(wall.start[1] * scale)
            )
            end = (
                int(wall.end[0] * scale),
                int(wall.end[1] * scale)
            )
            pygame.draw.line(minimap_surf, self.colors["wall"], start, end, 1)
        
        # Draw cars
        car_colors = [
            self.colors["car_blue"],
            self.colors["car_red"],
            self.colors["car_green"],
            self.colors["car_yellow"],
        ]
        
        for i, car in enumerate(cars):
            if car.state != CarState.ACTIVE:
                continue
            
            pos = (
                int(car.position[0] * scale),
                int(car.position[1] * scale)
            )
            color = car_colors[i % len(car_colors)]
            pygame.draw.circle(minimap_surf, color, pos, 3)
        
        self.render_surface.blit(minimap_surf, (minimap_x, minimap_y))
    
    def handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            False if quit event detected, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
        
        return True
    
    def close(self) -> None:
        """Close renderer and cleanup."""
        pygame.quit()
        logger.info("Renderer closed")
