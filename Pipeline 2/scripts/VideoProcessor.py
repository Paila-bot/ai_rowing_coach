import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

class VideoProcessor:
    """Handles video loading and frame processing"""

    def __init__(self):
        self.fps = None
        self.frame_count = None

    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Loading video: {video_path}")
        print(f"FPS: {self.fps}, Frame count: {self.frame_count}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Loaded {frame_idx} frames...")

        cap.release()
        print(f"Successfully loaded {len(frames)} frames")
        return frames

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for analysis"""
        # Resize all frames to a fixed size (e.g., 640x360)
        frame = cv2.resize(frame, (640, 360))
        return frame
