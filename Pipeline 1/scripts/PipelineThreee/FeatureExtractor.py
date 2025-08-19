import numpy as np
from typing import List, Dict, Tuple

class FeatureExtractor:
    def __init__(self, include_velocity=True):
        # Fixed set of features that will always be calculated
        self.feature_names = [
            'left_arm_angle',      # shoulder-elbow-wrist
            'right_arm_angle',     # shoulder-elbow-wrist
            'torso_angle',         # shoulder-shoulder-nose
            'body_lean',           # hip-shoulder angle
            'left_wrist_velocity', # velocity feature
            'right_wrist_velocity' # velocity feature
        ]

        self.n_features = len(self.feature_names)
        self.include_velocity = include_velocity
        self.prev_left_wrist = None
        self.prev_right_wrist = None

    def calculate_angles(self, pose_sequence: List[Dict]) -> np.ndarray:
        """Calculate consistent feature vectors for all frames"""
        features = []
        self.prev_left_wrist = None
        self.prev_right_wrist = None

        for frame_idx, frame in enumerate(pose_sequence):
            frame_features = self._calculate_frame_features(frame)
            features.append(frame_features)

        return np.array(features)

    def _calculate_frame_features(self, frame: Dict) -> List[float]:
        """Calculate exactly 6 features for each frame"""
        features = [0.0] * self.n_features  # Initialize with zeros

        try:
            # Feature 0: Left arm angle (shoulder-elbow-wrist)
            if all(joint in frame for joint in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angle = self._get_angle(
                    frame['left_shoulder'][:3],
                    frame['left_elbow'][:3],
                    frame['left_wrist'][:3]
                )
                if not (np.isnan(angle) or np.isinf(angle)):
                    features[0] = angle

            # Feature 1: Right arm angle (shoulder-elbow-wrist)
            if all(joint in frame for joint in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angle = self._get_angle(
                    frame['right_shoulder'][:3],
                    frame['right_elbow'][:3],
                    frame['right_wrist'][:3]
                )
                if not (np.isnan(angle) or np.isinf(angle)):
                    features[1] = angle

            # Feature 2: Torso orientation (left_shoulder-nose-right_shoulder)
            if all(joint in frame for joint in ['left_shoulder', 'nose', 'right_shoulder']):
                angle = self._get_angle(
                    frame['left_shoulder'][:3],
                    frame['nose'][:3],
                    frame['right_shoulder'][:3]
                )
                if not (np.isnan(angle) or np.isinf(angle)):
                    features[2] = angle

            # Feature 3: Body lean (hip to shoulder angle)
            if all(joint in frame for joint in ['left_hip', 'left_shoulder']):
                # Calculate lean as vertical deviation
                hip_pos = np.array(frame['left_hip'][:3])
                shoulder_pos = np.array(frame['left_shoulder'][:3])
                vertical = np.array([0, -1, 0])  # Downward vertical
                lean_vector = shoulder_pos - hip_pos

                # Calculate angle with vertical
                cos_angle = np.dot(lean_vector, vertical) / (np.linalg.norm(lean_vector) * np.linalg.norm(vertical))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))

                if not (np.isnan(angle) or np.isinf(angle)):
                    features[3] = angle

            # Feature 4: Left wrist velocity
            if 'left_wrist' in frame and self.include_velocity:
                current_pos = np.array(frame['left_wrist'][:3])
                if self.prev_left_wrist is not None:
                    velocity = np.linalg.norm(current_pos - self.prev_left_wrist)
                    if not (np.isnan(velocity) or np.isinf(velocity)):
                        features[4] = velocity
                self.prev_left_wrist = current_pos

            # Feature 5: Right wrist velocity
            if 'right_wrist' in frame and self.include_velocity:
                current_pos = np.array(frame['right_wrist'][:3])
                if self.prev_right_wrist is not None:
                    velocity = np.linalg.norm(current_pos - self.prev_right_wrist)
                    if not (np.isnan(velocity) or np.isinf(velocity)):
                        features[5] = velocity
                self.prev_right_wrist = current_pos

        except Exception as e:
            print(f"Warning: Feature calculation error: {e}")
            # Keep zero values for failed calculations

        return features

    def _get_angle(self, a: Tuple, b: Tuple, c: Tuple) -> float:
        """Calculate 3D angle between three points"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return 90.0  # Default angle for degenerate cases

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))