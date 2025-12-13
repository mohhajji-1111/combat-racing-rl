"""
Video Recorder
=============

Record gameplay videos.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from typing import Optional, List
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VideoRecorder:
    """
    Record gameplay videos.
    
    Features:
        - MP4 video export
        - Configurable FPS and quality
        - Frame buffering
        - Timestamp overlay
    """
    
    def __init__(
        self,
        output_path: Path,
        fps: int = 30,
        codec: str = 'mp4v',
    ):
        """
        Initialize video recorder.
        
        Args:
            output_path: Path to save video.
            fps: Frames per second.
            codec: Video codec (mp4v, H264, etc.).
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.codec = codec
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.is_recording = False
        
        logger.info(f"VideoRecorder initialized: {output_path} @ {fps}fps")
    
    def start(self, frame_shape: tuple) -> None:
        """
        Start recording.
        
        Args:
            frame_shape: (height, width, channels) of frames.
        """
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        height, width = frame_shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
        
        self.is_recording = True
        self.frame_count = 0
        
        logger.info(f"Recording started: {width}x{height} @ {self.fps}fps")
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add frame to video.
        
        Args:
            frame: RGB frame array [H, W, 3].
        """
        if not self.is_recording or self.writer is None:
            logger.warning("Not recording")
            return
        
        # Convert RGB to BGR (OpenCV format)
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        # Write frame
        self.writer.write(frame_bgr)
        self.frame_count += 1
    
    def stop(self) -> None:
        """Stop recording and save video."""
        if not self.is_recording:
            return
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        self.is_recording = False
        
        duration = self.frame_count / self.fps
        logger.info(
            f"Recording saved: {self.output_path} "
            f"({self.frame_count} frames, {duration:.2f}s)"
        )
    
    def __enter__(self):
        """Context manager enter."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def record_episode(
    env,
    agent,
    output_path: Path,
    max_steps: int = 5000,
    fps: int = 30,
) -> None:
    """
    Record single episode.
    
    Args:
        env: Environment with render() method.
        agent: Agent to control.
        output_path: Path to save video.
        max_steps: Maximum episode length.
        fps: Video FPS.
    """
    # Reset environment
    state = env.reset()
    
    # Get first frame to determine shape
    frame = env.render()
    if frame is None:
        raise ValueError("Environment must support rgb_array rendering")
    
    # Create recorder
    recorder = VideoRecorder(output_path, fps=fps)
    recorder.start(frame.shape)
    
    # Record episode
    done = False
    steps = 0
    
    try:
        while not done and steps < max_steps:
            # Add current frame
            recorder.add_frame(frame)
            
            # Agent action
            action = agent.select_action(state, deterministic=True)
            
            # Environment step
            state, reward, done, info = env.step(action)
            
            # Render next frame
            frame = env.render()
            if frame is None:
                break
            
            steps += 1
        
        # Add final frame
        recorder.add_frame(frame)
    
    finally:
        recorder.stop()
    
    logger.info(f"Episode recorded: {steps} steps, saved to {output_path}")
