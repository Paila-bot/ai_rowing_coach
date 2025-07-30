import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
# Synthetic Data Generator for Demo
class SyntheticDataGenerator:
    """Generate synthetic rowing video data for demonstration"""

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.width = frame_width
        self.height = frame_height
        self.frame_count = 0

    def generate_synthetic_video_frames(self, n_frames: int, technique_quality: str = 'good') -> List[np.ndarray]:
        """Generate synthetic video frames with different technique qualities"""
        frames = []

        for i in range(n_frames):
            frame = self._generate_synthetic_frame(i, technique_quality)
            frames.append(frame)

        return frames

    def _generate_synthetic_frame(self, frame_idx: int, technique_quality: str) -> np.ndarray:
        """Generate a single synthetic frame with a rower"""
        # Create blank frame (simulating gymnasium background)
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200  # Light gray background

        # Add some background noise/texture
        noise = np.random.randint(-30, 30, (self.height, self.width, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Simulate rowing motion cycle
        cycle_position = (frame_idx % 60) / 60.0  # 60 frame cycle

        # Generate rower silhouette based on technique quality and cycle position
        rower_mask = self._generate_rower_silhouette(cycle_position, technique_quality)

        # Add rower to frame (darker than background)
        rower_color = [80, 90, 100]  # Dark blue-gray for rower
        for c in range(3):
            frame[:, :, c] = np.where(rower_mask > 0, rower_color[c], frame[:, :, c])

        # Add some equipment (rowing machine rail)
        self._add_rowing_equipment(frame)

        return frame

    def _generate_rower_silhouette(self, cycle_position: float, technique_quality: str) -> np.ndarray:
        """Generate rower body silhouette for given cycle position"""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Center position
        center_x = self.width // 2
        center_y = self.height // 2

        # Determine stroke phase
        if cycle_position < 0.1:
            phase = 'catch'
        elif cycle_position < 0.4:
            phase = 'drive'
        elif cycle_position < 0.6:
            phase = 'finish'
        else:
            phase = 'recovery'

        # Generate body parts based on phase and technique quality
        body_parts = self._get_body_parts_for_phase(phase, cycle_position, technique_quality)

        # Draw body parts
        for part_name, (x, y, width, height) in body_parts.items():
            x_pos = int(center_x + x)
            y_pos = int(center_y + y)

            # Draw ellipse for body part
            cv2.ellipse(mask, (x_pos, y_pos), (width, height), 0, 0, 360, 255, -1)

        return mask

    def _get_body_parts_for_phase(self, phase: str, cycle_pos: float, quality: str) -> Dict[
        str, Tuple[int, int, int, int]]:
        """Get body part positions for different stroke phases"""
        # Base body proportions
        parts = {}

        # Add technique-specific variations
        arm_asymmetry = 0
        trunk_lean_error = 0
        leg_timing_error = 0

        if quality == 'poor':
            # Add realistic technique errors
            arm_asymmetry = np.sin(cycle_pos * 2 * np.pi) * 15  # Asymmetric arm movement
            trunk_lean_error = np.cos(cycle_pos * 2 * np.pi) * 10  # Poor trunk control
            leg_timing_error = 5 * np.sin(cycle_pos * 4 * np.pi)  # Leg timing issues

        if phase == 'catch':
            # Arms forward, legs compressed
            parts['torso'] = (0, 0, 40, 80)
            parts['head'] = (0, -60, 20, 25)
            parts['left_arm'] = (-60 + arm_asymmetry, -20, 15, 35)
            parts['right_arm'] = (60 - arm_asymmetry, -20, 15, 35)
            parts['left_leg'] = (-15, 40 + leg_timing_error, 20, 40)
            parts['right_leg'] = (15, 40 - leg_timing_error, 20, 40)

        elif phase == 'drive':
            # Arms pulling back, legs extending
            drive_progress = (cycle_pos - 0.1) / 0.3
            arm_pullback = drive_progress * 40
            leg_extension = drive_progress * 30

            parts['torso'] = (trunk_lean_error, -5, 40, 80)
            parts['head'] = (trunk_lean_error, -65, 20, 25)
            parts['left_arm'] = (-40 + arm_pullback + arm_asymmetry, -15, 15, 35)
            parts['right_arm'] = (40 - arm_pullback - arm_asymmetry, -15, 15, 35)
            parts['left_leg'] = (-15, 25 + leg_extension + leg_timing_error, 20, 50)
            parts['right_leg'] = (15, 25 + leg_extension - leg_timing_error, 20, 50)

        elif phase == 'finish':
            # Arms at body, legs extended
            parts['torso'] = (trunk_lean_error, -10, 40, 80)
            parts['head'] = (trunk_lean_error, -70, 20, 25)
            parts['left_arm'] = (-20 + arm_asymmetry, -10, 15, 35)
            parts['right_arm'] = (20 - arm_asymmetry, -10, 15, 35)
            parts['left_leg'] = (-15, 55 + leg_timing_error, 20, 60)
            parts['right_leg'] = (15, 55 - leg_timing_error, 20, 60)

        else:  # recovery
            # Arms extending forward, legs coming back
            recovery_progress = (cycle_pos - 0.6) / 0.4
            arm_extension = recovery_progress * 40
            leg_compression = recovery_progress * 30

            parts['torso'] = (trunk_lean_error * 0.5, -5, 40, 80)
            parts['head'] = (trunk_lean_error * 0.5, -65, 20, 25)
            parts['left_arm'] = (-20 - arm_extension + arm_asymmetry, -15, 15, 35)
            parts['right_arm'] = (20 + arm_extension - arm_asymmetry, -15, 15, 35)
            parts['left_leg'] = (-15, 55 - leg_compression + leg_timing_error, 20, 60 - int(leg_compression))
            parts['right_leg'] = (15, 55 - leg_compression - leg_timing_error, 20, 60 - int(leg_compression))

        return parts

    def _add_rowing_equipment(self, frame: np.ndarray):
        """Add rowing machine equipment to frame"""
        # Add rail/seat track
        rail_y = int(self.height * 0.75)
        cv2.line(frame, (50, rail_y), (self.width - 50, rail_y), (60, 60, 60), 8)

        # Add some equipment details
        cv2.rectangle(frame, (self.width - 100, rail_y - 20), (self.width - 20, rail_y + 20), (100, 100, 100), -1)