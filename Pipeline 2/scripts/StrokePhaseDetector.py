import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import os

class StrokePhaseDetector:
    """Detects rowing stroke phases from pose data"""

    def __init__(self):
        self.phases = ['catch', 'drive', 'finish', 'recovery']
        self.phase_history = []
        self.current_phase = 'catch'

    def detect_phase(self, body_points: np.ndarray) -> str:
        """Detect current stroke phase based on body position"""
        # Calculate key metrics for phase detection
        arm_extension = self._calculate_arm_extension(body_points)
        leg_compression = self._calculate_leg_compression(body_points)
        trunk_angle = self._calculate_trunk_lean(body_points)

        # Phase detection logic based on biomechanical patterns
        if arm_extension < 0.3 and leg_compression > 0.7:
            phase = 'catch'
        elif arm_extension < 0.6 and leg_compression < 0.3:
            phase = 'drive'
        elif arm_extension > 0.8 and leg_compression < 0.2:
            phase = 'finish'
        else:
            phase = 'recovery'

        # Add temporal consistency
        self.phase_history.append(phase)
        if len(self.phase_history) > 5:
            self.phase_history.pop(0)

        # Use majority vote for stability
        from collections import Counter
        phase_counts = Counter(self.phase_history)
        self.current_phase = phase_counts.most_common(1)[0][0]

        return self.current_phase

    def _calculate_arm_extension(self, points: np.ndarray) -> float:
        """Calculate normalized arm extension (0=compressed, 1=extended)"""
        # Distance from shoulders to wrists
        left_arm_ext = np.linalg.norm(points[5] - points[1])  # left wrist to shoulder
        right_arm_ext = np.linalg.norm(points[6] - points[2])  # right wrist to shoulder

        avg_arm_ext = (left_arm_ext + right_arm_ext) / 2

        # Normalize by shoulder width (approximate max arm extension)
        shoulder_width = np.linalg.norm(points[2] - points[1])
        max_extension = shoulder_width * 2  # Approximate full arm extension

        return min(avg_arm_ext / (max_extension + 1e-8), 1.0)

    def _calculate_leg_compression(self, points: np.ndarray) -> float:
        """Calculate normalized leg compression (0=extended, 1=compressed)"""
        # Distance from hips to ankles
        left_leg_ext = np.linalg.norm(points[11] - points[7])  # left ankle to hip
        right_leg_ext = np.linalg.norm(points[12] - points[8])  # right ankle to hip

        avg_leg_ext = (left_leg_ext + right_leg_ext) / 2

        # Estimate max leg extension (thigh + shin length)
        left_thigh = np.linalg.norm(points[7] - points[9])
        left_shin = np.linalg.norm(points[9] - points[11])
        max_leg_extension = left_thigh + left_shin

        compression = 1.0 - min(avg_leg_ext / (max_leg_extension + 1e-8), 1.0)
        return max(compression, 0.0)

    def _calculate_trunk_lean(self, points: np.ndarray) -> float:
        """Calculate trunk lean angle"""
        shoulder_mid = (points[1] + points[2]) / 2
        hip_mid = (points[7] + points[8]) / 2

        trunk_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, -1])

        cos_angle = np.dot(trunk_vector, vertical_vector) / (
                np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale and normalize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0