import numpy as np
import cv2
from collections import deque
from typing import Dict, Tuple, List


class BackgroundSubtractor:
    def __init__(self, shape=None, alpha=0.01, history_size=50):
        """
        Initialize improved background subtractor with better noise handling

        Args:
            shape: Shape of input frames (height, width) - can be None for dynamic sizing
            alpha: Learning rate for background update (lower = slower adaptation)
            history_size: Number of frames to keep for median background estimation
        """
        self.alpha = alpha
        self.bg = None
        self.bg_variance = None  # Track background variance for adaptive thresholding
        self.frame_history = deque(maxlen=history_size)
        self.initialized = False
        self.frame_count = 0
        self.expected_shape = shape

        # Improved parameters
        self.min_learning_frames = 20  # More frames for initial learning
        self.variance_learning_rate = 0.005  # Slower variance adaptation
        self.shadow_threshold = 0.6  # Shadow detection threshold
        self.ghost_suppression_threshold = 3.0  # Remove ghosting artifacts

    def _ensure_shape_compatibility(self, frame):
        """Ensure frame and background have compatible shapes with better preprocessing"""
        # Convert to float32 and apply slight gaussian blur to reduce noise
        frame = frame.astype(np.float32)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        if self.bg is None:
            # Initialize background and variance with first frame's shape
            self.bg = np.zeros(frame.shape, dtype=np.float32)
            self.bg_variance = np.ones(frame.shape, dtype=np.float32) * 10  # Initial variance
            self.expected_shape = frame.shape
            return frame

        # Check if frame shape matches expected shape
        if frame.shape != self.expected_shape:
            print(f"[Warning] Frame shape {frame.shape} doesn't match expected {self.expected_shape}")
            # Resize frame to match expected shape
            frame = cv2.resize(frame, (self.expected_shape[1], self.expected_shape[0]))
            print(f"[Info] Resized frame to {frame.shape}")

        return frame.astype(np.float32)

    def update(self, frame):
        """
        Update background model with improved statistical modeling

        Args:
            frame: Input frame (grayscale)
        Returns:
            Updated background model
        """
        frame = self._ensure_shape_compatibility(frame)

        # Store frame in history for robust background estimation
        self.frame_history.append(frame.copy())

        if not self.initialized:
            # Extended initialization phase with better statistics
            if self.frame_count == 0:
                self.bg = frame.copy()
                self.bg_variance = np.ones_like(frame) * 10
            else:
                # Use running statistics for better initial background
                n = self.frame_count + 1
                delta = frame - self.bg
                self.bg += delta / n

                # Update variance estimate
                if self.frame_count > 1:
                    self.bg_variance = ((n - 2) * self.bg_variance + (delta ** 2)) / (n - 1)

            self.frame_count += 1

            # Initialize when we have enough frames
            if self.frame_count >= self.min_learning_frames:
                self.initialized = True
                print(f"[Info] Background model initialized after {self.frame_count} frames")

                # Use median of collected frames for robust initial background
                if len(self.frame_history) >= 10:
                    frame_stack = np.array(list(self.frame_history))
                    self.bg = np.median(frame_stack, axis=0)

                    # Calculate more robust variance estimate
                    self.bg_variance = np.var(frame_stack, axis=0) + 1.0  # Add small constant

        else:
            # Adaptive background update with shadow and ghost suppression
            diff = np.abs(frame - self.bg)

            # Create adaptive learning rate based on difference magnitude
            adaptive_alpha = self.alpha * np.exp(-diff / (2 * np.sqrt(self.bg_variance + 1)))

            # Detect potential shadows (darker than background but similar texture)
            shadow_mask = (frame < self.shadow_threshold * self.bg) & (diff < 2 * np.sqrt(self.bg_variance))

            # Detect potential ghosts/static objects (consistently different)
            ghost_mask = diff > self.ghost_suppression_threshold * np.sqrt(self.bg_variance)

            # Update background with reduced learning for shadows and ghosts
            learning_rate = adaptive_alpha.copy()
            learning_rate[shadow_mask] *= 0.1  # Slow learning for shadows
            learning_rate[ghost_mask] *= 0.05  # Very slow learning for potential ghosts

            # Update background model
            self.bg = (1 - learning_rate) * self.bg + learning_rate * frame

            # Update variance with slower adaptation
            variance_diff = (frame - self.bg) ** 2
            self.bg_variance = (1 - self.variance_learning_rate) * self.bg_variance + \
                               self.variance_learning_rate * variance_diff

            # Prevent variance from becoming too small
            self.bg_variance = np.maximum(self.bg_variance, 1.0)

        return self.bg.astype(np.uint8)

    def subtract(self, frame, threshold=30, use_adaptive=True):
        """
        Improved background subtraction with adaptive thresholding and noise reduction

        Args:
            frame: Input frame
            threshold: Base threshold for foreground detection
            use_adaptive: Whether to use adaptive thresholding
        Returns:
            Binary foreground mask
        """
        frame = self._ensure_shape_compatibility(frame)

        # Ensure background is initialized
        if self.bg is None:
            print("[Warning] Background not initialized, using frame as background")
            self.bg = frame.copy()
            self.bg_variance = np.ones_like(frame) * 10
            return np.zeros(frame.shape, dtype=np.uint8)

        # Calculate absolute difference
        diff = np.abs(frame - self.bg)

        if use_adaptive and self.initialized:
            # Adaptive thresholding based on local variance and statistics
            # Create adaptive threshold based on background variance
            adaptive_threshold = threshold + 2.5 * np.sqrt(self.bg_variance)

            # Additional local adaptive thresholding
            kernel_size = 9
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

            # Local statistics
            local_mean = cv2.filter2D(diff, -1, kernel)
            local_var = cv2.filter2D((diff - local_mean) ** 2, -1, kernel)
            local_std = np.sqrt(local_var + 1.0)

            # Combine global and local adaptive thresholds
            final_threshold = np.minimum(adaptive_threshold, threshold + 1.8 * local_std)

            # Apply threshold
            fg_mask = diff > final_threshold

            # Shadow suppression - remove pixels that are just darker
            shadow_mask = (frame < self.shadow_threshold * self.bg) & \
                          (diff < 0.8 * final_threshold)
            fg_mask[shadow_mask] = False

        else:
            # Simple thresholding for non-initialized case
            fg_mask = diff > threshold

        return fg_mask.astype(np.uint8)

    def get_cleaned_mask(self, frame, threshold=30, min_area=100):
        """
        Get cleaned foreground mask with improved morphological operations

        Args:
            frame: Input frame
            threshold: Threshold for foreground detection
            min_area: Minimum area for valid components
        Returns:
            Cleaned binary mask
        """
        # Get initial mask
        fg_mask = self.subtract(frame, threshold)

        # Progressive morphological cleaning
        # Start with small operations and build up
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Remove tiny noise first
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_tiny)

        # Progressive hole filling
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_medium)

        # Final smoothing
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)

        # Remove small components and fill remaining holes
        if min_area > 0:
            fg_mask = self._filter_by_area(fg_mask, min_area)

            # Final hole filling after area filtering
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)

        return fg_mask

    def _filter_by_area(self, binary_mask, min_area):
        """Improved area filtering with hole filling"""
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create new mask with only large enough components
        filtered_mask = np.zeros_like(binary_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Fill the contour completely (removes internal holes)
                cv2.fillPoly(filtered_mask, [contour], 1)

        return filtered_mask

    def get_background_confidence(self, frame):
        """
        Get confidence map showing how well background is learned
        """
        if not self.initialized or self.bg_variance is None:
            return np.zeros_like(frame)

        frame = self._ensure_shape_compatibility(frame)

        # Confidence is inverse of variance (normalized)
        max_var = np.max(self.bg_variance)
        confidence = 1.0 - (self.bg_variance / (max_var + 1))

        return (confidence * 255).astype(np.uint8)

    def get_difference_map(self, frame):
        """
        Get normalized difference map for debugging
        """
        if self.bg is None:
            return np.zeros_like(frame)

        frame = self._ensure_shape_compatibility(frame)
        diff = np.abs(frame - self.bg)

        # Normalize to 0-255 range
        if np.max(diff) > 0:
            diff_norm = (diff / np.max(diff) * 255).astype(np.uint8)
        else:
            diff_norm = diff.astype(np.uint8)

        return diff_norm

    def update_learning_rate(self, new_alpha):
        """Update learning rate during runtime"""
        self.alpha = max(0.001, min(0.1, new_alpha))
        print(f"[Info] Learning rate updated to {self.alpha}")

    def reset_background(self):
        """Reset the background model"""
        self.bg = None
        self.bg_variance = None
        self.frame_history.clear()
        self.initialized = False
        self.frame_count = 0
        self.expected_shape = None
        print("[Info] Background model reset")

    def get_background_image(self):
        """Get current background model as uint8 image"""
        if self.bg is not None:
            return self.bg.astype(np.uint8)
        else:
            return None

    def get_variance_image(self):
        """Get background variance as visualization"""
        if self.bg_variance is not None:
            # Normalize variance for visualization
            var_norm = np.sqrt(self.bg_variance)  # Take sqrt for better visualization
            var_norm = (var_norm / np.max(var_norm) * 255).astype(np.uint8)
            return var_norm
        else:
            return None

    # Keep all the existing joint extraction methods unchanged for compatibility
    def extract_rowing_joints(self, points: List[Tuple[int, int]],
                              frame_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """
        Extract joint positions using rowing-specific anatomical model
        (Keeping existing implementation for now)
        """
        if not points:
            return self._get_rowing_default_joints()

        # Convert to numpy for easier processing
        points_array = np.array(points, dtype=np.int32)
        ys, xs = points_array[:, 0], points_array[:, 1]

        # Basic measurements
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        height = max_y - min_y
        width = max_x - min_x
        center_x = int(np.mean(xs))

        # Step 1: Identify the main body axis and orientation
        body_axis = self._compute_body_axis(points)
        torso_angle = self._compute_torso_angle(body_axis)

        # Step 2: Create detailed horizontal analysis
        num_slices = 20
        slice_profiles = self._analyze_horizontal_slices(points, min_y, height, num_slices)

        # Step 3: Find key anatomical regions using rowing-specific knowledge
        head_region = self._find_head_region_rowing(slice_profiles, min_y, height, center_x)
        shoulder_region = self._find_shoulder_region_rowing(slice_profiles, min_y, height, center_x)
        hip_region = self._find_hip_region_rowing(slice_profiles, min_y, height, center_x)

        # Step 4: Extract arm and leg positions for seated rowing
        arm_joints = self._find_rowing_arm_joints(points, shoulder_region, torso_angle)
        leg_joints = self._find_rowing_leg_joints(points, hip_region)

        # Step 5: Combine all joints with anatomical constraints
        joints = {
            'head': head_region,
            'neck': self._interpolate_joint(head_region, shoulder_region, 0.7),
            'shoulder_left': self._find_left_shoulder_rowing(shoulder_region, slice_profiles),
            'shoulder_right': self._find_right_shoulder_rowing(shoulder_region, slice_profiles),
            'spine_mid': self._interpolate_joint(shoulder_region, hip_region, 0.5),
            'hips': hip_region,
            **arm_joints,
            **leg_joints
        }

        # Step 6: Apply rowing-specific constraints and validation
        joints = self._apply_rowing_constraints(joints, torso_angle, min_y, max_y, min_x, max_x)

        return joints

    # [Keep all the existing joint detection helper methods unchanged]
    def _compute_body_axis(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """Compute the main axis of the body using PCA"""
        points_array = np.array(points, dtype=np.float32)

        # Center the points
        centroid = np.mean(points_array, axis=0)
        centered_points = points_array - centroid

        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)

        # Get principal component (main axis)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]

        return main_axis

    def _compute_torso_angle(self, body_axis: np.ndarray) -> float:
        """Compute torso angle from vertical (positive = forward lean)"""
        # Angle with vertical axis (0, 1)
        vertical = np.array([0, 1])
        cos_angle = np.dot(body_axis, vertical)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angle_degrees = np.degrees(angle)

        # Determine if leaning forward or backward
        if body_axis[0] > 0:  # x-component positive = forward lean
            return angle_degrees
        else:
            return -angle_degrees

    def _analyze_horizontal_slices(self, points: List[Tuple[int, int]],
                                   min_y: int, height: int, num_slices: int) -> List[Dict]:
        """Analyze body shape in horizontal slices for detailed anatomy"""
        slice_height = max(1, height // num_slices)
        slice_profiles = []

        for i in range(num_slices):
            slice_y_min = min_y + i * slice_height
            slice_y_max = min_y + (i + 1) * slice_height

            # Get points in this slice
            slice_points = [p for p in points if slice_y_min <= p[0] < slice_y_max]

            if slice_points:
                slice_xs = [p[1] for p in slice_points]
                profile = {
                    'y_center': (slice_y_min + slice_y_max) // 2,
                    'x_min': min(slice_xs),
                    'x_max': max(slice_xs),
                    'x_center': int(np.mean(slice_xs)),
                    'width': max(slice_xs) - min(slice_xs),
                    'density': len(slice_points),
                    'points': slice_points
                }
            else:
                profile = {
                    'y_center': (slice_y_min + slice_y_max) // 2,
                    'x_min': 0, 'x_max': 0, 'x_center': 0,
                    'width': 0, 'density': 0, 'points': []
                }

            slice_profiles.append(profile)

        return slice_profiles

    def _find_head_region_rowing(self, slice_profiles: List[Dict], min_y: int,
                                 height: int, center_x: int) -> Tuple[int, int]:
        """Find head position for seated rower"""
        # Head is in top 15% and has rounded/compact profile
        head_slices = slice_profiles[:max(1, len(slice_profiles) // 6)]

        if not head_slices:
            return (min_y, center_x)

        # Find slice with good density and reasonable width for head
        valid_slices = [s for s in head_slices if s['density'] > 0 and s['width'] > 0]
        if valid_slices:
            # Look for slice with high density and moderate width (head characteristics)
            best_slice = max(valid_slices,
                             key=lambda x: x['density'] * (1.0 / (1.0 + abs(x['width'] - height * 0.15))))
            return (best_slice['y_center'], best_slice['x_center'])

        return (min_y + height // 10, center_x)

    def _find_shoulder_region_rowing(self, slice_profiles: List[Dict], min_y: int,
                                     height: int, center_x: int) -> Tuple[int, int]:
        """Find shoulder region for seated rower - typically widest in upper body"""
        # Shoulders are 15-25% down from top in seated position
        start_idx = max(1, len(slice_profiles) // 6)  # ~15%
        end_idx = min(len(slice_profiles), len(slice_profiles) // 4)  # ~25%

        shoulder_slices = slice_profiles[start_idx:end_idx]

        if not shoulder_slices:
            return (min_y + int(height * 0.2), center_x)

        # Find widest slice with good density
        valid_slices = [s for s in shoulder_slices if s['density'] > 0]
        if valid_slices:
            best_slice = max(valid_slices, key=lambda x: x['width'] * x['density'])
            return (best_slice['y_center'], best_slice['x_center'])

        return (min_y + int(height * 0.2), center_x)

    def _find_hip_region_rowing(self, slice_profiles: List[Dict], min_y: int,
                                height: int, center_x: int) -> Tuple[int, int]:
        """Find hip region for seated rower"""
        # For seated rower, hips are prominent and stable, around 45-65% down
        start_idx = int(len(slice_profiles) * 0.45)  # 45%
        end_idx = min(len(slice_profiles), int(len(slice_profiles) * 0.65))  # 65%

        hip_slices = slice_profiles[start_idx:end_idx]

        if not hip_slices:
            return (min_y + int(height * 0.55), center_x)

        # For seated position, hips show as wide, stable region
        valid_slices = [s for s in hip_slices if s['density'] > 0 and s['width'] > 0]
        if valid_slices:
            # Look for consistent width and high density (seated hip characteristics)
            best_slice = max(valid_slices,
                             key=lambda x: x['density'] * x['width'] * (1.0 / (1.0 + abs(x['x_center'] - center_x))))
            return (best_slice['y_center'], best_slice['x_center'])

        return (min_y + int(height * 0.55), center_x)

    def _find_left_shoulder_rowing(self, shoulder_center: Tuple[int, int],
                                   slice_profiles: List[Dict]) -> Tuple[int, int]:
        """Find left shoulder for rowing position"""
        shoulder_y, shoulder_x = shoulder_center

        # Find slice closest to shoulder height
        closest_slice = min([s for s in slice_profiles if s['density'] > 0],
                            key=lambda s: abs(s['y_center'] - shoulder_y),
                            default={'y_center': shoulder_y, 'x_min': shoulder_x - 25, 'width': 50})

        # Left shoulder is typically at 20% from left edge
        if closest_slice['width'] > 0:
            left_shoulder_x = closest_slice['x_min'] + int(closest_slice['width'] * 0.2)
        else:
            left_shoulder_x = shoulder_x - 25

        return (shoulder_y, left_shoulder_x)

    def _find_right_shoulder_rowing(self, shoulder_center: Tuple[int, int],
                                    slice_profiles: List[Dict]) -> Tuple[int, int]:
        """Find right shoulder for rowing position"""
        shoulder_y, shoulder_x = shoulder_center

        # Find slice closest to shoulder height
        closest_slice = min([s for s in slice_profiles if s['density'] > 0],
                            key=lambda s: abs(s['y_center'] - shoulder_y),
                            default={'y_center': shoulder_y, 'x_max': shoulder_x + 25, 'width': 50})

        # Right shoulder is typically at 80% from left edge
        if closest_slice['width'] > 0:
            right_shoulder_x = closest_slice['x_min'] + int(closest_slice['width'] * 0.8)
        else:
            right_shoulder_x = shoulder_x + 25

        return (shoulder_y, right_shoulder_x)

    def _find_rowing_arm_joints(self, points: List[Tuple[int, int]],
                                shoulder_pos: Tuple[int, int],
                                torso_angle: float) -> Dict[str, Tuple[int, int]]:
        """Find arm joints specifically for rowing motion analysis"""
        shoulder_y, shoulder_x = shoulder_pos

        # In rowing, arms extend forward and back
        # Separate arm regions based on forward/back position relative to shoulders
        forward_points = []  # Points in front of shoulders (catch position)
        back_points = []  # Points behind shoulders (finish position)

        for y, x in points:
            if y > shoulder_y - 20 and y < shoulder_y + 60:  # Arm height range
                if torso_angle > 0:  # Forward lean
                    if x > shoulder_x:
                        forward_points.append((y, x))
                    else:
                        back_points.append((y, x))
                else:  # Upright or backward lean
                    if x < shoulder_x:
                        forward_points.append((y, x))
                    else:
                        back_points.append((y, x))

        arms = {}

        # Process forward extending arms (catch phase)
        if forward_points:
            # Sort by distance from shoulder
            forward_distances = [(np.sqrt((p[0] - shoulder_y) ** 2 + (p[1] - shoulder_x) ** 2), p)
                                 for p in forward_points]
            forward_sorted = [p for _, p in sorted(forward_distances, reverse=True)]

            if len(forward_sorted) >= 2:
                arms['wrist_forward'] = forward_sorted[0]  # Farthest point
                arms['elbow_forward'] = forward_sorted[len(forward_sorted) // 2]  # Mid-distance
            elif len(forward_sorted) == 1:
                arms['wrist_forward'] = forward_sorted[0]
                arms['elbow_forward'] = self._interpolate_joint(shoulder_pos, forward_sorted[0], 0.6)

        # Process backward extending arms (finish phase)
        if back_points:
            back_distances = [(np.sqrt((p[0] - shoulder_y) ** 2 + (p[1] - shoulder_x) ** 2), p)
                              for p in back_points]
            back_sorted = [p for _, p in sorted(back_distances, reverse=True)]

            if len(back_sorted) >= 2:
                arms['wrist_back'] = back_sorted[0]
                arms['elbow_back'] = back_sorted[len(back_sorted) // 2]
            elif len(back_sorted) == 1:
                arms['wrist_back'] = back_sorted[0]
                arms['elbow_back'] = self._interpolate_joint(shoulder_pos, back_sorted[0], 0.6)

        # Default left/right assignments for compatibility
        arms['elbow_left'] = arms.get('elbow_forward', (0, 0))
        arms['elbow_right'] = arms.get('elbow_back', (0, 0))
        arms['wrist_left'] = arms.get('wrist_forward', (0, 0))
        arms['wrist_right'] = arms.get('wrist_back', (0, 0))

        return arms

    def _find_rowing_leg_joints(self, points: List[Tuple[int, int]],
                                hip_pos: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """Find leg joints for seated rowing position"""
        hip_y, hip_x = hip_pos

        # In seated rowing, legs are bent and extend forward from hips
        leg_points = [p for p in points if p[0] > hip_y]  # Below hip level

        if not leg_points:
            return {'knee_left': (0, 0), 'knee_right': (0, 0),
                    'ankle_left': (0, 0), 'ankle_right': (0, 0)}

        # Separate legs by horizontal position
        left_leg = [p for p in leg_points if p[1] < hip_x - 10]  # Left of center
        right_leg = [p for p in leg_points if p[1] > hip_x + 10]  # Right of center
        center_leg = [p for p in leg_points if hip_x - 10 <= p[1] <= hip_x + 10]  # Center

        # If legs are together (common in rowing), split center points
        if len(center_leg) > len(left_leg) + len(right_leg):
            center_sorted = sorted(center_leg, key=lambda p: p[1])
            mid_idx = len(center_sorted) // 2
            left_leg.extend(center_sorted[:mid_idx])
            right_leg.extend(center_sorted[mid_idx:])

        legs = {}

        # Process each leg
        for side, leg_points_side, suffix in [('left', left_leg, '_left'),
                                              ('right', right_leg, '_right')]:
            if not leg_points_side:
                legs[f'knee{suffix}'] = (0, 0)
                legs[f'ankle{suffix}'] = (0, 0)
                continue

            # Sort by distance from hip (closer = knee, farther = ankle)
            leg_distances = [(np.sqrt((p[0] - hip_y) ** 2 + (p[1] - hip_x) ** 2), p)
                             for p in leg_points_side]
            leg_sorted = [p for _, p in sorted(leg_distances)]

            if len(leg_sorted) >= 2:
                # Knee is closer to hip, ankle is farther
                knee_idx = len(leg_sorted) // 3  # First third
                ankle_idx = min(len(leg_sorted) - 1, int(len(leg_sorted) * 0.8))  # Last part

                legs[f'knee{suffix}'] = leg_sorted[knee_idx]
                legs[f'ankle{suffix}'] = leg_sorted[ankle_idx]
            elif len(leg_sorted) == 1:
                # Only one point - assume it's the knee
                legs[f'knee{suffix}'] = leg_sorted[0]
                legs[f'ankle{suffix}'] = (0, 0)
            else:
                legs[f'knee{suffix}'] = (0, 0)
                legs[f'ankle{suffix}'] = (0, 0)

        return legs

    def _interpolate_joint(self, joint1: Tuple[int, int], joint2: Tuple[int, int],
                           ratio: float) -> Tuple[int, int]:
        """Interpolate between two joint positions"""
        if joint1 == (0, 0) or joint2 == (0, 0):
            return (0, 0)

        y = int(joint1[0] + ratio * (joint2[0] - joint1[0]))
        x = int(joint1[1] + ratio * (joint2[1] - joint1[1]))
        return (y, x)

    def _apply_rowing_constraints(self, joints: Dict[str, Tuple[int, int]],
                                  torso_angle: float, min_y: int, max_y: int,
                                  min_x: int, max_x: int) -> Dict[str, Tuple[int, int]]:
        """Apply rowing-specific anatomical constraints"""

        # Ensure proper joint hierarchy for seated position
        head_y = joints['head'][0] if joints['head'] != (0, 0) else min_y
        shoulder_y = max(joints['shoulder_left'][0], joints['shoulder_right'][0])
        hip_y = joints['hips'][0] if joints['hips'] != (0, 0) else max_y

        # Fix ordering issues
        if shoulder_y <= head_y + 10:
            joints['shoulder_left'] = (head_y + 15, joints['shoulder_left'][1])
            joints['shoulder_right'] = (head_y + 15, joints['shoulder_right'][1])

        if hip_y <= shoulder_y + 20:
            joints['hips'] = (shoulder_y + 25, joints['hips'][1])

        # Constrain all joints to bounding box
        for joint_name, (y, x) in joints.items():
            if (y, x) != (0, 0):
                y = max(min_y, min(max_y, y))
                x = max(min_x, min(max_x, x))
                joints[joint_name] = (y, x)

        return joints

    def _get_rowing_default_joints(self) -> Dict[str, Tuple[int, int]]:
        """Return default joint positions for rowing"""
        return {
            'head': (0, 0),
            'neck': (0, 0),
            'shoulder_left': (0, 0),
            'shoulder_right': (0, 0),
            'elbow_left': (0, 0),
            'elbow_right': (0, 0),
            'wrist_left': (0, 0),
            'wrist_right': (0, 0),
            'spine_mid': (0, 0),
            'hips': (0, 0),
            'knee_left': (0, 0),
            'knee_right': (0, 0),
            'ankle_left': (0, 0),
            'ankle_right': (0, 0)
        }

    def analyze_rowing_stroke_phase(self, joints: Dict[str, Tuple[int, int]]) -> Dict[str, any]:
        """Analyze which phase of the rowing stroke the person is in"""
        analysis = {
            'stroke_phase': 'unknown',
            'torso_angle': 0,
            'leg_compression': 0,
            'arm_extension': 0,
            'form_score': 0,
            'feedback': []
        }

        # Calculate torso angle
        if joints['head'] != (0, 0) and joints['hips'] != (0, 0):
            head_y, head_x = joints['head']
            hip_y, hip_x = joints['hips']

            # Vector from hips to head
            dx = head_x - hip_x
            dy = head_y - hip_y

            # Angle from vertical (positive = forward lean)
            analysis['torso_angle'] = np.degrees(np.arctan2(dx, -dy))

        # Determine stroke phase based on joint positions
        torso_angle = analysis['torso_angle']

        if torso_angle > 20:
            analysis['stroke_phase'] = 'catch'  # Forward lean
        elif torso_angle > -5:
            analysis['stroke_phase'] = 'drive'  # Upright, driving
        else:
            analysis['stroke_phase'] = 'finish'  # Slight backward lean

        # Calculate leg compression (knee angle approximation)
        if (joints['hips'] != (0, 0) and joints['knee_left'] != (0, 0) and
                joints['ankle_left'] != (0, 0)):

            hip_y, hip_x = joints['hips']
            knee_y, knee_x = joints['knee_left']
            ankle_y, ankle_x = joints['ankle_left']

            # Simple leg compression metric
            hip_knee_dist = np.sqrt((hip_y - knee_y) ** 2 + (hip_x - knee_x) ** 2)
            knee_ankle_dist = np.sqrt((knee_y - ankle_y) ** 2 + (knee_x - ankle_x) ** 2)

            if hip_knee_dist > 0:
                analysis['leg_compression'] = knee_ankle_dist / hip_knee_dist

        # Provide rowing-specific feedback
        if analysis['stroke_phase'] == 'catch' and torso_angle > 30:
            analysis['feedback'].append("Too much forward lean at catch")
        elif analysis['stroke_phase'] == 'finish' and torso_angle < -15:
            analysis['feedback'].append("Don't lean back too far at finish")

        return analysis

    # Keep existing methods for compatibility
    @staticmethod
    def extract_joints(points, method='rowing'):
        """
        Backward compatibility wrapper - now defaults to rowing method
        """
        if method == 'rowing':
            # Create temporary instance for static call compatibility
            temp_instance = BackgroundSubtractor((100, 100))  # Dummy shape
            return temp_instance.extract_rowing_joints(points, (100, 100))
        else:
            # Original contour method for compatibility
            return BackgroundSubtractor._extract_joints_from_contour(points)

    @staticmethod
    def connected_components(binary_img, connectivity=8):
        """Find connected components in binary image"""
        if connectivity == 8:
            # Use OpenCV's connected components (more efficient)
            num_labels, labels = cv2.connectedComponents(binary_img, connectivity=8)

            components = {}
            h, w = binary_img.shape

            for label in range(1, num_labels):  # Skip background (label 0)
                y_coords, x_coords = np.where(labels == label)
                if len(y_coords) > 0:  # Only include non-empty components
                    points = list(zip(y_coords, x_coords))
                    components[label] = points

            return components
        else:
            # Use custom flood fill implementation for 4-connectivity
            return BackgroundSubtractor._custom_connected_components(binary_img)

    @staticmethod
    def _custom_connected_components(binary_img):
        """Custom connected components implementation using flood fill"""
        h, w = binary_img.shape
        visited = np.zeros_like(binary_img, dtype=bool)
        label = 1
        components = {}

        for y in range(h):
            for x in range(w):
                if binary_img[y, x] == 1 and not visited[y, x]:
                    # Flood fill to find connected component
                    q = deque([(y, x)])
                    visited[y, x] = True
                    points = []

                    while q:
                        cy, cx = q.popleft()
                        points.append((cy, cx))

                        # Check 8-connected neighbors
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if (0 <= ny < h and 0 <= nx < w and
                                        binary_img[ny, nx] == 1 and not visited[ny, nx]):
                                    visited[ny, nx] = True
                                    q.append((ny, nx))

                    if len(points) > 0:  # Only store non-empty components
                        components[label] = points
                        label += 1

        return components

    @staticmethod
    def _extract_joints_from_contour(points):
        """Original contour method for backward compatibility"""
        if not points:
            return {
                "head": (0, 0),
                "shoulder_left": (0, 0),
                "shoulder_right": (0, 0),
                "hips": (0, 0),
                "knees": (0, 0)
            }

        # Convert to numpy array
        points_array = np.array(points, dtype=np.int32)
        ys, xs = points_array[:, 0], points_array[:, 1]

        # Basic bounds
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))
        height = max_y - min_y
        width = max_x - min_x

        # Create horizontal slices to analyze body structure
        num_slices = 10
        slice_height = height // num_slices
        slice_widths = []
        slice_centers = []

        for i in range(num_slices):
            slice_y_min = min_y + i * slice_height
            slice_y_max = min_y + (i + 1) * slice_height

            slice_points = [p for p in points if slice_y_min <= p[0] < slice_y_max]
            if slice_points:
                slice_xs = [p[1] for p in slice_points]
                slice_width = max(slice_xs) - min(slice_xs)
                slice_center_x = int(np.mean(slice_xs))
                slice_widths.append(slice_width)
                slice_centers.append(slice_center_x)
            else:
                slice_widths.append(0)
                slice_centers.append(center_x)

        # Find head (topmost cluster)
        head_candidates = sorted(points, key=lambda p: p[0])[:max(1, len(points) // 10)]
        head_y = int(np.mean([p[0] for p in head_candidates]))
        head_x = int(np.mean([p[1] for p in head_candidates]))
        head = (head_y, head_x)

        # Find narrowest region (likely waist/hips area)
        if slice_widths:
            waist_slice = slice_widths.index(min([w for w in slice_widths if w > 0]))
            hips_y = min_y + waist_slice * slice_height + slice_height // 2
            hips_x = slice_centers[waist_slice] if waist_slice < len(slice_centers) else center_x
            hips = (hips_y, hips_x)
        else:
            hips = (min_y + int(height * 0.5), center_x)

        # Shoulders: Look for widest region in upper half
        upper_half_slices = slice_widths[:num_slices // 2]
        if upper_half_slices:
            shoulder_slice = upper_half_slices.index(max(upper_half_slices))
            shoulder_y = min_y + shoulder_slice * slice_height + slice_height // 2
            shoulder_x = slice_centers[shoulder_slice] if shoulder_slice < len(slice_centers) else center_x
            shoulders = (shoulder_y, shoulder_x)
        else:
            shoulders = (min_y + int(height * 0.25), center_x)

        # Knees: Look for points in the lower region
        lower_third_y = min_y + int(height * 0.66)
        knee_candidates = [p for p in points if p[0] >= lower_third_y]

        if knee_candidates:
            knee_ys = [p[0] for p in knee_candidates]
            knee_xs = [p[1] for p in knee_candidates]
            knee_y = int(np.mean(knee_ys))
            knee_x = int(np.mean(knee_xs))
            knees = (knee_y, knee_x)
        else:
            knees = (max_y - int(height * 0.1), center_x)

        # Ensure proper ordering
        if head[0] > shoulders[0]:
            head = (min_y, head[1])

        joint_positions = [head[0], shoulders[0], hips[0], knees[0]]
        if not all(joint_positions[i] <= joint_positions[i + 1] for i in range(len(joint_positions) - 1)):
            head = (min_y, head[1])
            shoulders = (min_y + int(height * 0.15), shoulders[1])
            hips = (min_y + int(height * 0.5), hips[1])
            knees = (min_y + int(height * 0.8), knees[1])

        return {
            "head": head,
            "shoulder_left": shoulders,
            "shoulder_right": shoulders,
            "hips": hips,
            "knees": knees
        }

    @staticmethod
    def merge_nearby_components(components, distance_threshold=50):
        """
        Merge components that are likely part of the same person
        """
        if len(components) <= 1:
            return components

        # Calculate centroids for each component
        centroids = {}
        for label, points in components.items():
            if points:
                ys, xs = zip(*points)
                centroids[label] = (np.mean(ys), np.mean(xs))

        # Find components to merge
        merged_components = {}
        used_labels = set()

        for label1, centroid1 in centroids.items():
            if label1 in used_labels:
                continue

            # Start a new merged component
            merged_points = list(components[label1])
            used_labels.add(label1)

            # Find nearby components to merge
            for label2, centroid2 in centroids.items():
                if label2 == label1 or label2 in used_labels:
                    continue

                # Calculate distance between centroids
                distance = np.sqrt((centroid1[0] - centroid2[0]) ** 2 +
                                   (centroid1[1] - centroid2[1]) ** 2)

                if distance < distance_threshold:
                    merged_points.extend(components[label2])
                    used_labels.add(label2)

            if merged_points:
                merged_components[len(merged_components) + 1] = merged_points

        return merged_components

    def set_learning_rate(self, alpha):
        """Update learning rate"""
        self.alpha = max(0.001, min(1.0, alpha))  # Clamp between 0.001 and 1.0