import numpy as np
import csv
import os
from typing import Dict, List, Tuple


class StrokeDetector:
    def __init__(self, bins=16, histogram_threshold=0.2, angle_threshold=15.0):
        """
        Enhanced stroke detector for rowing analysis

        Args:
            bins: Number of histogram bins for frame difference detection
            histogram_threshold: Threshold for detecting frame changes
            angle_threshold: Threshold for detecting significant angle changes
        """
        self.prev_hist = None
        self.bins = bins
        self.histogram_threshold = histogram_threshold
        self.angle_threshold = angle_threshold

        # Stroke phase tracking
        self.stroke_transitions = []
        self.stroke_phases = []
        self.prev_torso_angle = None
        self.prev_phase = 'unknown'

        # Joint angle history for stroke analysis
        self.angle_history = {
            'torso_angle': [],
            'knee_angle_left': [],
            'knee_angle_right': [],
            'elbow_angle_left': [],
            'elbow_angle_right': [],
            'hip_knee_angle_left': [],
            'hip_knee_angle_right': []
        }

        # Frame tracking
        self.frame_count = 0

    def calculate_joint_angles(self, joints: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """
        Calculate key joint angles for rowing analysis

        Args:
            joints: Dictionary of joint positions

        Returns:
            Dictionary of calculated angles
        """
        angles = {}

        # Torso angle (already calculated in BackgroundSubtractor)
        if joints['head'] != (0, 0) and joints['hips'] != (0, 0):
            head_y, head_x = joints['head']
            hip_y, hip_x = joints['hips']
            dx = head_x - hip_x
            dy = head_y - hip_y
            angles['torso_angle'] = np.degrees(np.arctan2(dx, -dy))
        else:
            angles['torso_angle'] = 0.0

        # Knee angles (hip-knee-ankle)
        for side in ['left', 'right']:
            hip_pos = joints['hips']
            knee_pos = joints[f'knee_{side}']
            ankle_pos = joints[f'ankle_{side}']

            if all(pos != (0, 0) for pos in [hip_pos, knee_pos, ankle_pos]):
                angles[f'knee_angle_{side}'] = self._calculate_three_point_angle(
                    hip_pos, knee_pos, ankle_pos
                )

                # Hip-knee angle (thigh angle relative to vertical)
                angles[f'hip_knee_angle_{side}'] = self._calculate_angle_from_vertical(
                    hip_pos, knee_pos
                )
            else:
                angles[f'knee_angle_{side}'] = 0.0
                angles[f'hip_knee_angle_{side}'] = 0.0

        # Elbow angles (shoulder-elbow-wrist)
        for side in ['left', 'right']:
            shoulder_pos = joints[f'shoulder_{side}']
            elbow_pos = joints[f'elbow_{side}']
            wrist_pos = joints[f'wrist_{side}']

            if all(pos != (0, 0) for pos in [shoulder_pos, elbow_pos, wrist_pos]):
                angles[f'elbow_angle_{side}'] = self._calculate_three_point_angle(
                    shoulder_pos, elbow_pos, wrist_pos
                )
            else:
                angles[f'elbow_angle_{side}'] = 0.0

        return angles

    def _calculate_three_point_angle(self, p1: Tuple[int, int], p2: Tuple[int, int],
                                     p3: Tuple[int, int]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        # Convert to numpy arrays
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def _calculate_angle_from_vertical(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate angle of line p1-p2 from vertical"""
        dy = p2[0] - p1[0]  # Note: y increases downward in image coordinates
        dx = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dx, dy))
        return angle

    def detect_stroke_phase(self, joints: Dict[str, Tuple[int, int]],
                            joint_angles: Dict[str, float]) -> str:
        """
        Detect current stroke phase based on joint positions and angles

        Args:
            joints: Joint positions
            joint_angles: Calculated joint angles

        Returns:
            String indicating stroke phase
        """
        torso_angle = joint_angles.get('torso_angle', 0.0)

        # Determine stroke phase based on torso angle and leg compression
        avg_knee_angle = (joint_angles.get('knee_angle_left', 180.0) +
                          joint_angles.get('knee_angle_right', 180.0)) / 2.0

        # Stroke phase classification
        if torso_angle > 20:  # Forward lean
            if avg_knee_angle < 120:  # Legs compressed
                phase = 'catch'
            else:
                phase = 'recovery'
        elif torso_angle > -5:  # Upright
            if avg_knee_angle < 140:  # Legs partially extended
                phase = 'drive'
            else:
                phase = 'recovery'
        else:  # Backward lean
            phase = 'finish'

        return phase

    def update(self, frame, joints: Dict[str, Tuple[int, int]], frame_id: int):
        """
        Update stroke detector with new frame and joint data

        Args:
            frame: Current frame (grayscale)
            joints: Joint positions for current frame
            frame_id: Frame identifier
        """
        self.frame_count = frame_id

        # Calculate joint angles
        joint_angles = self.calculate_joint_angles(joints)

        # Update angle history
        for angle_name, angle_value in joint_angles.items():
            if angle_name in self.angle_history:
                self.angle_history[angle_name].append(angle_value)

        # Detect stroke phase
        current_phase = self.detect_stroke_phase(joints, joint_angles)
        self.stroke_phases.append(current_phase)

        # Detect phase transitions
        if self.prev_phase != 'unknown' and current_phase != self.prev_phase:
            transition = {
                'frame_id': frame_id,
                'from_phase': self.prev_phase,
                'to_phase': current_phase,
                'torso_angle': joint_angles.get('torso_angle', 0.0)
            }
            self.stroke_transitions.append(transition)

        self.prev_phase = current_phase

        # Histogram-based change detection
        hist, _ = np.histogram(frame, bins=self.bins, range=(0, 255))
        hist = hist / (hist.sum() + 1e-10)

        if self.prev_hist is not None:
            hist_diff = np.linalg.norm(hist - self.prev_hist)
            if hist_diff > self.histogram_threshold:
                # Significant frame change detected
                pass  # Could be used for additional analysis

        self.prev_hist = hist

        return current_phase, joint_angles

    def get_stroke_metrics(self) -> Dict:
        """
        Calculate stroke-specific metrics from accumulated data

        Returns:
            Dictionary containing stroke analysis metrics
        """
        if not self.angle_history['torso_angle']:
            return {}

        metrics = {}

        # Torso angle analysis
        torso_angles = np.array(self.angle_history['torso_angle'])
        metrics['torso_angle_range'] = float(np.max(torso_angles) - np.min(torso_angles))
        metrics['torso_angle_mean'] = float(np.mean(torso_angles))
        metrics['torso_angle_std'] = float(np.std(torso_angles))

        # Knee angle analysis (averaged left/right)
        left_knee = np.array(self.angle_history['knee_angle_left'])
        right_knee = np.array(self.angle_history['knee_angle_right'])

        if len(left_knee) > 0 and len(right_knee) > 0:
            avg_knee_angles = (left_knee + right_knee) / 2.0
            metrics['knee_angle_range'] = float(np.max(avg_knee_angles) - np.min(avg_knee_angles))
            metrics['knee_angle_mean'] = float(np.mean(avg_knee_angles))

        # Stroke phase analysis
        if self.stroke_phases:
            phase_counts = {}
            for phase in self.stroke_phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            total_frames = len(self.stroke_phases)
            metrics['phase_distribution'] = {
                phase: count / total_frames for phase, count in phase_counts.items()
            }

        # Stroke consistency (transitions)
        metrics['total_transitions'] = len(self.stroke_transitions)
        metrics['stroke_frequency'] = len(self.stroke_transitions) / max(1, self.frame_count)

        return metrics

    def export_stroke_data(self, output_path: str, video_name: str = "unknown"):
        """
        Export stroke analysis data to CSV

        Args:
            output_path: Path to save CSV file
            video_name: Name identifier for the video
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'video_name', 'frame_id', 'stroke_phase',
                'torso_angle', 'knee_angle_left', 'knee_angle_right',
                'elbow_angle_left', 'elbow_angle_right',
                'hip_knee_angle_left', 'hip_knee_angle_right'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Get minimum length across all angle histories
            min_length = min(len(angles) for angles in self.angle_history.values() if angles)

            for i in range(min_length):
                row = {
                    'video_name': video_name,
                    'frame_id': i,
                    'stroke_phase': self.stroke_phases[i] if i < len(self.stroke_phases) else 'unknown'
                }

                # Add angle data
                for angle_name in self.angle_history:
                    if i < len(self.angle_history[angle_name]):
                        row[angle_name] = self.angle_history[angle_name][i]
                    else:
                        row[angle_name] = 0.0

                writer.writerow(row)

    def reset(self):
        """Reset detector state for new video"""
        self.prev_hist = None
        self.stroke_transitions = []
        self.stroke_phases = []
        self.prev_torso_angle = None
        self.prev_phase = 'unknown'
        self.frame_count = 0

        # Clear angle history
        for key in self.angle_history:
            self.angle_history[key] = []