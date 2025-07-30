import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class PoseEstimator:
    """Classical computer vision pose estimation using only OpenCV and NumPy"""

    def __init__(self, alpha: float = 0.05):
        # Background subtraction parameter
        self.alpha = alpha
        self.background_model = None
        self.frame_count = 0

        # Joint names for 13 key body points
        self.joint_names = [
            'head', 'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
            'wrist_left', 'wrist_right', 'hip_left', 'hip_right', 'knee_left',
            'knee_right', 'ankle_left', 'ankle_right'
        ]
        self.n_joints = len(self.joint_names)

        # Temporal smoothing for joint tracking
        self.joint_history = []
        self.max_history = 5

    def estimate_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Main pose estimation pipeline
        Returns: array of shape (n_joints, 2) with (x, y) coordinates
        """
        # Step 1: Background subtraction to isolate athlete
        athlete_mask = self._extract_athlete_silhouette(frame)

        # Step 2: Contour detection and analysis
        contours = self._detect_contours(athlete_mask)

        # Step 3: Find main body contour
        body_contour = self._find_body_contour(contours, frame.shape)

        if body_contour is None:
            return self._get_fallback_pose(frame.shape)

        # Step 4: Extract skeleton structure
        skeleton_points = self._extract_skeleton_from_contour(body_contour, frame.shape)

        # Step 5: Fit anatomical model
        joint_positions = self._fit_anatomical_model(skeleton_points, body_contour, frame.shape)

        # Step 6: Temporal smoothing
        smoothed_joints = self._temporal_smoothing(joint_positions)

        return smoothed_joints

    def _extract_athlete_silhouette(self, frame: np.ndarray) -> np.ndarray:
        """Extract athlete silhouette using background subtraction"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Initialize background model on first frame
        if self.background_model is None:
            self.background_model = gray_frame.astype(np.float32)
            return np.zeros(gray_frame.shape, dtype=np.uint8)

        # Update background model using running average
        self.background_model = (1 - self.alpha) * self.background_model + self.alpha * gray_frame

        # Calculate absolute difference
        diff = np.abs(gray_frame.astype(np.float32) - self.background_model)

        # Threshold to create binary mask
        threshold = np.mean(diff) + 2 * np.std(diff)
        athlete_mask = (diff > threshold).astype(np.uint8) * 255

        # Morphological operations to clean up mask
        athlete_mask = self._morphological_cleanup(athlete_mask)

        self.frame_count += 1
        return athlete_mask

    def _morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean up binary mask using morphological operations"""
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Remove noise with opening
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        # Fill holes with closing
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)

        return cleaned

    def _detect_contours(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """Detect contours in binary mask"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _find_body_contour(self, contours: List[np.ndarray], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find the main body contour based on size and shape criteria"""
        if not contours:
            return None

        h, w = frame_shape[:2]
        min_area = (h * w) * 0.01  # At least 1% of frame area
        max_area = (h * w) * 0.5  # At most 50% of frame area

        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Filter by aspect ratio
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = ch / (cw + 1e-8)

            if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for seated person
                valid_contours.append((contour, area))

        if not valid_contours:
            return None

        # Return largest valid contour
        return max(valid_contours, key=lambda x: x[1])[0]

    def _extract_skeleton_from_contour(self, contour: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[
        str, np.ndarray]:
        """Extract skeleton points from body contour"""
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return {}

        # Centroid
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        skeleton_points = {}

        # Find extremes of contour
        contour_reshaped = contour.reshape(-1, 2)
        skeleton_points['top'] = contour_reshaped[np.argmin(contour_reshaped[:, 1])]
        skeleton_points['bottom'] = contour_reshaped[np.argmax(contour_reshaped[:, 1])]
        skeleton_points['left'] = contour_reshaped[np.argmin(contour_reshaped[:, 0])]
        skeleton_points['right'] = contour_reshaped[np.argmax(contour_reshaped[:, 0])]
        skeleton_points['centroid'] = np.array([cx, cy])

        return skeleton_points

    def _fit_anatomical_model(self, skeleton_points: Dict[str, np.ndarray],
                              contour: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Fit anatomical model to skeleton points"""
        joints = np.zeros((self.n_joints, 2))

        # Get bounding box of contour
        x, y, cw, ch = cv2.boundingRect(contour)

        # Estimate body dimensions
        body_height = ch
        body_width = cw
        center_x = x + cw // 2
        top_y = y

        # Use anatomical proportions to place joints
        # Head
        joints[0] = skeleton_points.get('top', [center_x, top_y + int(0.125 * body_height / 2)])

        # Shoulders
        shoulder_y = top_y + int(0.2 * body_height)
        shoulder_width = int(0.25 * body_height)
        joints[1] = [center_x - shoulder_width // 2, shoulder_y]  # left shoulder
        joints[2] = [center_x + shoulder_width // 2, shoulder_y]  # right shoulder

        # Hips
        hip_y = shoulder_y + int(0.4 * body_height)
        hip_width = int(0.18 * body_height)
        joints[7] = [center_x - hip_width // 2, hip_y]  # left hip
        joints[8] = [center_x + hip_width // 2, hip_y]  # right hip

        # Elbows (positioned for rowing motion)
        elbow_y = shoulder_y + int(0.15 * body_height)
        joints[3] = [center_x - int(body_width * 0.4), elbow_y]  # left elbow
        joints[4] = [center_x + int(body_width * 0.4), elbow_y]  # right elbow

        # Wrists (extend from elbows toward extremes)
        if 'left' in skeleton_points and 'right' in skeleton_points:
            joints[5] = skeleton_points['left']  # left wrist
            joints[6] = skeleton_points['right']  # right wrist
        else:
            joints[5] = [joints[3][0] - 30, joints[3][1] + 20]
            joints[6] = [joints[4][0] + 30, joints[4][1] + 20]

        # Knees
        knee_y = hip_y + int(0.25 * body_height)
        joints[9] = [joints[7][0], knee_y]  # left knee
        joints[10] = [joints[8][0], knee_y]  # right knee

        # Ankles
        ankle_y = knee_y + int(0.25 * body_height)
        joints[11] = [joints[9][0], ankle_y]  # left ankle
        joints[12] = [joints[10][0], ankle_y]  # right ankle

        return joints

    def _temporal_smoothing(self, current_joints: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce jitter"""
        self.joint_history.append(current_joints.copy())

        if len(self.joint_history) > self.max_history:
            self.joint_history.pop(0)

        if len(self.joint_history) < 3:
            return current_joints

        # Apply exponential moving average
        weights = np.exp(np.linspace(-2, 0, len(self.joint_history)))
        weights /= np.sum(weights)

        smoothed_joints = np.zeros_like(current_joints)
        for i, joints in enumerate(self.joint_history):
            smoothed_joints += weights[i] * joints

        return smoothed_joints

    def _get_fallback_pose(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Return fallback pose when detection fails"""
        if self.joint_history:
            return self.joint_history[-1].copy()
        else:
            h, w = frame_shape[:2]
            return self._generate_default_pose(h, w)

    def _generate_default_pose(self, h: int, w: int) -> np.ndarray:
        """Generate default seated rowing pose"""
        joints = np.zeros((self.n_joints, 2))
        center_x, center_y = w // 2, h // 2

        # Default seated rowing position
        joints[0] = [center_x, center_y - 0.3 * h]  # head
        joints[1] = [center_x - 0.1 * w, center_y - 0.2 * h]  # left shoulder
        joints[2] = [center_x + 0.1 * w, center_y - 0.2 * h]  # right shoulder
        joints[3] = [center_x - 0.15 * w, center_y - 0.1 * h]  # left elbow
        joints[4] = [center_x + 0.15 * w, center_y - 0.1 * h]  # right elbow
        joints[5] = [center_x - 0.2 * w, center_y]  # left wrist
        joints[6] = [center_x + 0.2 * w, center_y]  # right wrist
        joints[7] = [center_x - 0.05 * w, center_y + 0.1 * h]  # left hip
        joints[8] = [center_x + 0.05 * w, center_y + 0.1 * h]  # right hip
        joints[9] = [center_x - 0.05 * w, center_y + 0.25 * h]  # left knee
        joints[10] = [center_x + 0.05 * w, center_y + 0.25 * h]  # right knee
        joints[11] = [center_x - 0.05 * w, center_y + 0.4 * h]  # left ankle
        joints[12] = [center_x + 0.05 * w, center_y + 0.4 * h]  # right ankle

        return joints