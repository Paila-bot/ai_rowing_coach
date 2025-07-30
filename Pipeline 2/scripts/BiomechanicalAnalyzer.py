import numpy as np
import PoseEstimator as PE
from typing import List, Tuple, Dict, Optional

class BiomechanicalAnalyzer:
    """Extracts biomechanical features from pose estimation results"""

    def __init__(self):
        self.body_points_names = [
            'head', 'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
            'wrist_left', 'wrist_right', 'hip_left', 'hip_right', 'knee_left',
            'knee_right', 'ankle_left', 'ankle_right'
        ]
        self.n_points = len(self.body_points_names)
        self.pose_estimator = PE.PoseEstimator()

    def detect_body_points(self, frame: np.ndarray) -> np.ndarray:
        """Detect body points using pose estimation"""
        body_points = self.pose_estimator.estimate_pose(frame)
        return body_points

    def extract_features(self, body_points: np.ndarray) -> np.ndarray:
        """Extract biomechanical features from body points"""
        features = []

        # Joint angles
        features.extend(self._calculate_joint_angles(body_points))

        # Body segment ratios
        features.extend(self._calculate_segment_ratios(body_points))

        # Posture metrics
        features.extend(self._calculate_posture_metrics(body_points))

        # Symmetry metrics
        features.extend(self._calculate_symmetry_metrics(body_points))

        return np.array(features)

    def _calculate_joint_angles(self, points: np.ndarray) -> List[float]:
        """Calculate key joint angles for rowing technique"""
        angles = []

        # Elbow angles (left and right)
        left_elbow_angle = self._angle_between_points(points[1], points[3], points[5])
        right_elbow_angle = self._angle_between_points(points[2], points[4], points[6])
        angles.extend([left_elbow_angle, right_elbow_angle])

        # Knee angles (left and right)
        left_knee_angle = self._angle_between_points(points[7], points[9], points[11])
        right_knee_angle = self._angle_between_points(points[8], points[10], points[12])
        angles.extend([left_knee_angle, right_knee_angle])

        # Trunk angle (relative to vertical)
        trunk_angle = self._calculate_trunk_angle(points)
        angles.append(trunk_angle)

        return angles

    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def _calculate_trunk_angle(self, points: np.ndarray) -> float:
        """Calculate trunk angle relative to vertical"""
        shoulder_mid = (points[1] + points[2]) / 2
        hip_mid = (points[7] + points[8]) / 2

        trunk_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, -1])  # Pointing up

        cos_angle = np.dot(trunk_vector, vertical_vector) / (
                np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def _calculate_segment_ratios(self, points: np.ndarray) -> List[float]:
        """Calculate body segment length ratios"""
        ratios = []

        # Arm segments
        left_upper_arm = np.linalg.norm(points[1] - points[3])
        left_forearm = np.linalg.norm(points[3] - points[5])
        right_upper_arm = np.linalg.norm(points[2] - points[4])
        right_forearm = np.linalg.norm(points[4] - points[6])

        # Arm ratios
        left_arm_ratio = left_forearm / (left_upper_arm + 1e-8)
        right_arm_ratio = right_forearm / (right_upper_arm + 1e-8)
        ratios.extend([left_arm_ratio, right_arm_ratio])

        # Leg segments
        left_thigh = np.linalg.norm(points[7] - points[9])
        left_shin = np.linalg.norm(points[9] - points[11])
        right_thigh = np.linalg.norm(points[8] - points[10])
        right_shin = np.linalg.norm(points[10] - points[12])

        # Leg ratios
        left_leg_ratio = left_shin / (left_thigh + 1e-8)
        right_leg_ratio = right_shin / (right_thigh + 1e-8)
        ratios.extend([left_leg_ratio, right_leg_ratio])

        return ratios

    def _calculate_posture_metrics(self, points: np.ndarray) -> List[float]:
        """Calculate posture-related metrics"""
        metrics = []

        # Head position relative to shoulders
        head_shoulder_mid = (points[1] + points[2]) / 2
        head_offset = points[0] - head_shoulder_mid
        head_forward = head_offset[0]
        metrics.append(head_forward)

        # Shoulder level (should be horizontal)
        shoulder_slope = (points[2][1] - points[1][1]) / (points[2][0] - points[1][0] + 1e-8)
        metrics.append(abs(shoulder_slope))

        # Hip level
        hip_slope = (points[8][1] - points[7][1]) / (points[8][0] - points[7][0] + 1e-8)
        metrics.append(abs(hip_slope))

        return metrics

    def _calculate_symmetry_metrics(self, points: np.ndarray) -> List[float]:
        """Calculate left-right symmetry metrics"""
        metrics = []

        # Arm symmetry
        left_arm_reach = np.linalg.norm(points[5] - points[1])
        right_arm_reach = np.linalg.norm(points[6] - points[2])
        arm_symmetry = abs(left_arm_reach - right_arm_reach)
        metrics.append(arm_symmetry)

        # Leg symmetry
        left_leg_extension = np.linalg.norm(points[11] - points[7])
        right_leg_extension = np.linalg.norm(points[12] - points[8])
        leg_symmetry = abs(left_leg_extension - right_leg_extension)
        metrics.append(leg_symmetry)

        return metrics