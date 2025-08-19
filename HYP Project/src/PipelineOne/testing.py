import csv
import os
import numpy as np
from collections import defaultdict
import cv2
import warnings

# Suppress overflow warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')


class OptimizedGaussianMixtureBackgroundSubtractor:
    """Memory-optimized version of background subtraction with improved numerical stability"""

    def __init__(self, num_gaussians=3, learning_rate=0.01, background_threshold=0.7, initial_frame=None):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        self.background_threshold = background_threshold

        if initial_frame is not None:
            frame_height, frame_width = initial_frame.shape
            self.means = np.zeros((frame_height, frame_width, num_gaussians), dtype=np.float32)
            self.variances = np.ones((frame_height, frame_width, num_gaussians), dtype=np.float32) * 15.0
            self.weights = np.ones((frame_height, frame_width, num_gaussians), dtype=np.float32) / num_gaussians

            # Initialize first Gaussian with frame data
            self.means[:, :, 0] = initial_frame.astype(np.float32)
            for i in range(1, num_gaussians):
                self.means[:, :, i] = np.random.uniform(0, 255, size=(frame_height, frame_width))

    def apply(self, current_frame):
        if not hasattr(self, 'means'):
            return np.zeros_like(current_frame, dtype=np.uint8)

        current_frame = np.clip(current_frame.astype(np.float32), 0, 255)
        difference = np.abs(current_frame[:, :, None] - self.means)
        matches = difference <= 2.5 * np.sqrt(np.maximum(self.variances, 1.0))
        pixel_matches_any = np.any(matches, axis=2)

        # Vectorized updates with numerical stability
        for g in range(self.num_gaussians):
            matched = matches[:, :, g]
            update_factor = self.learning_rate * self._gaussian_prob(
                current_frame, self.means[:, :, g], self.variances[:, :, g]
            )
            update_factor = np.clip(update_factor, 0, 1)

            self.means[:, :, g] = np.where(matched,
                                           (1 - update_factor) * self.means[:, :, g] + update_factor * current_frame,
                                           self.means[:, :, g])

            squared_diff = (current_frame - self.means[:, :, g]) ** 2
            self.variances[:, :, g] = np.where(matched,
                                               (1 - update_factor) * self.variances[:, :,
                                                                     g] + update_factor * squared_diff,
                                               self.variances[:, :, g])
            # Ensure minimum variance
            self.variances[:, :, g] = np.maximum(self.variances[:, :, g], 1.0)

            self.weights[:, :, g] = np.where(matched,
                                             (1 - self.learning_rate) * self.weights[:, :, g] + self.learning_rate,
                                             (1 - self.learning_rate) * self.weights[:, :, g])

        # Normalize weights with numerical stability
        weight_sum = np.sum(self.weights, axis=2, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-8)
        self.weights = self.weights / weight_sum

        # Determine background - vectorized approach
        sorted_indices = np.argsort(self.weights / np.sqrt(self.variances), axis=2)[:, :, ::-1]
        background_mask = self._determine_background_vectorized(current_frame, sorted_indices)

        return (~background_mask).astype(np.uint8) * 255

    def _gaussian_prob(self, x, mean, variance):
        variance = np.maximum(variance, 1e-8)  # Prevent division by zero
        return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance))

    def _determine_background_vectorized(self, frame, sorted_indices):
        """Vectorized background determination with safety checks"""
        background_mask = np.zeros_like(frame, dtype=bool)
        height, width = frame.shape

        for y in range(height):
            for x in range(width):
                cumulative_weight = 0.0
                pixel_value = frame[y, x]

                for gaussian_idx in sorted_indices[y, x]:
                    cumulative_weight += self.weights[y, x, gaussian_idx]
                    mean = self.means[y, x, gaussian_idx]
                    variance = max(self.variances[y, x, gaussian_idx], 1.0)

                    if abs(pixel_value - mean) <= 2.5 * np.sqrt(variance):
                        background_mask[y, x] = True
                        break

                    if cumulative_weight > self.background_threshold:
                        break

        return background_mask


class OptimizedPoseEstimation:
    """Memory-optimized pose estimation with improved numerical stability"""

    def __init__(self):
        self.background_subtractor = None

    def grayscale(self, frame):
        if frame.ndim == 2:
            return frame.copy()
        return np.dot(frame[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)

    def gaussian_blur(self, sigma, kernel_size):
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    def convolve_frame(self, frame, kernel):
        """Optimized convolution with boundary handling"""
        try:
            if kernel.shape[0] > 7:
                # Use OpenCV for large kernels if available
                return cv2.filter2D(frame, -1, kernel)
            else:
                # Direct convolution for small kernels
                k_h, k_w = kernel.shape
                pad_h, pad_w = k_h // 2, k_w // 2
                padded = np.pad(frame, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

                height, width = frame.shape
                result = np.zeros_like(frame, dtype=np.float32)

                for i in range(height):
                    for j in range(width):
                        result[i, j] = np.sum(padded[i:i + k_h, j:j + k_w] * kernel)

                return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[WARNING] Convolution failed: {e}, using original frame")
            return frame

    def background_subtraction(self, frame):
        try:
            if self.background_subtractor is None:
                gray_frame = self.grayscale(frame)
                self.background_subtractor = OptimizedGaussianMixtureBackgroundSubtractor(
                    num_gaussians=3, learning_rate=0.01, background_threshold=0.7, initial_frame=gray_frame
                )
            return self.background_subtractor.apply(frame)
        except Exception as e:
            print(f"[WARNING] Background subtraction failed: {e}")
            return np.zeros_like(frame, dtype=np.uint8)

    def morphological_operations(self, binary_mask, operation='open', kernel_size=5):
        """Use OpenCV for morphological operations if available"""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if operation == 'open':
                return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            elif operation == 'close':
                return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            else:
                return binary_mask
        except:
            # Fallback to manual implementation
            return self._manual_morphology(binary_mask, operation, kernel_size)

    def _manual_morphology(self, binary_mask, operation, kernel_size):
        """Fallback manual morphological operations"""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pad = kernel_size // 2
        padded = np.pad(binary_mask, pad, mode='constant')

        if operation == 'open':
            # Erosion followed by dilation
            temp = np.zeros_like(binary_mask)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    window = padded[i:i + kernel_size, j:j + kernel_size]
                    temp[i, j] = np.min(window) if np.sum(window) == kernel_size * kernel_size else 0

            padded_temp = np.pad(temp, pad, mode='constant')
            result = np.zeros_like(binary_mask)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    window = padded_temp[i:i + kernel_size, j:j + kernel_size]
                    result[i, j] = np.max(window)
            return result

        return binary_mask

    def connected_components_labeling(self, binary_mask):
        """Use OpenCV connected components if available"""
        try:
            num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
            return labels, num_labels - 1  # Subtract 1 to exclude background
        except:
            return self._manual_connected_components(binary_mask)

    def _manual_connected_components(self, binary_mask):
        """Fallback manual connected components"""
        height, width = binary_mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        current_label = 1

        def flood_fill(start_y, start_x, label):
            stack = [(start_y, start_x)]
            component_size = 0

            while stack:
                y, x = stack.pop()
                if (y < 0 or y >= height or x < 0 or x >= width or
                        binary_mask[y, x] == 0 or labels[y, x] != 0):
                    continue

                labels[y, x] = label
                component_size += 1

                # Add 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy != 0 or dx != 0:
                            stack.append((y + dy, x + dx))

            return component_size

        for y in range(height):
            for x in range(width):
                if binary_mask[y, x] != 0 and labels[y, x] == 0:
                    size = flood_fill(y, x, current_label)
                    if size > 50:  # Filter small components
                        current_label += 1
                    else:
                        # Remove small components
                        labels[labels == current_label] = 0

        return labels, current_label - 1

    def shi_tomasi_corner_detection(self, grayscale_image, window_size=5, min_corner_response=100):
        """Improved corner detection with numerical stability"""
        try:
            # Compute gradients using Scharr operator with float64 for stability
            scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64) / 16
            scharr_y = scharr_x.T

            # Convert to float64 for numerical stability
            img_float = grayscale_image.astype(np.float64)

            Ix = cv2.filter2D(img_float, -1, scharr_x) if 'cv2' in globals() else self._manual_gradient(img_float,
                                                                                                        scharr_x)
            Iy = cv2.filter2D(img_float, -1, scharr_y) if 'cv2' in globals() else self._manual_gradient(img_float,
                                                                                                        scharr_y)

            Ix2, Iy2, IxIy = Ix * Ix, Iy * Iy, Ix * Iy

            height, width = grayscale_image.shape
            half_w = window_size // 2
            keypoints = []

            # Use stride to reduce computation and avoid overflow regions
            stride = max(1, window_size // 2)
            for y in range(half_w, height - half_w, stride):
                for x in range(half_w, width - half_w, stride):
                    # Calculate structure tensor elements
                    window_Ix2 = Ix2[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]
                    window_Iy2 = Iy2[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]
                    window_IxIy = IxIy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]

                    sum_Ix2 = np.sum(window_Ix2)
                    sum_Iy2 = np.sum(window_Iy2)
                    sum_IxIy = np.sum(window_IxIy)

                    # Numerical stability checks
                    max_val = max(abs(sum_Ix2), abs(sum_Iy2), abs(sum_IxIy))
                    if max_val > 1e10 or max_val == 0:  # Avoid overflow and underflow
                        continue

                    # Calculate minimum eigenvalue with numerical stability
                    trace = sum_Ix2 + sum_Iy2
                    det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy

                    if trace <= 0 or det <= 0:
                        continue

                    # Use safer eigenvalue calculation
                    discriminant = trace * trace - 4 * det
                    if discriminant < 0:
                        continue

                    min_eigenvalue = (trace - np.sqrt(discriminant)) / 2

                    if min_eigenvalue > min_corner_response and min_eigenvalue < 1e8:  # Avoid extreme values
                        keypoints.append((y, x))

            return keypoints

        except Exception as e:
            print(f"[WARNING] Corner detection failed: {e}")
            return []

    def _manual_gradient(self, img, kernel):
        """Manual gradient computation fallback"""
        height, width = img.shape
        result = np.zeros_like(img)
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        for y in range(pad_h, height - pad_h):
            for x in range(pad_w, width - pad_w):
                result[y, x] = np.sum(img[y - pad_h:y + pad_h + 1, x - pad_w:x + pad_w + 1] * kernel)

        return result

    def extract_anatomical_joints(self, bbox, keypoints, labels, blob_id):
        """Extract joints for a specific blob with safety checks"""
        joints = {}

        if not keypoints:
            return joints

        # Get blob points
        ys, xs = np.where(labels == blob_id)
        if len(ys) == 0:
            return joints

        blob_points = list(zip(ys, xs))
        height = max(1, bbox[1] - bbox[0])  # Prevent division by zero

        # Head - topmost point
        head = min(blob_points, key=lambda p: p[0])

        # Filter keypoints by relative position in blob
        def get_keypoints_in_range(min_ratio, max_ratio):
            return [kp for kp in keypoints
                    if bbox[0] <= kp[0] <= bbox[1] and  # Within blob bounds
                    min_ratio < (kp[0] - bbox[0]) / height < max_ratio]

        # Extract joints with better error handling
        try:
            # Shoulders (20-40% down from top)
            shoulders = get_keypoints_in_range(0.2, 0.4)
            left_shoulder = min(shoulders, key=lambda p: p[1]) if shoulders else None
            right_shoulder = max(shoulders, key=lambda p: p[1]) if shoulders else None

            # Hips (50-70% down from top)
            hips = get_keypoints_in_range(0.5, 0.7)
            left_hip = min(hips, key=lambda p: p[1]) if hips else None
            right_hip = max(hips, key=lambda p: p[1]) if hips else None

            # Knees (70%+ down from top)
            knees = get_keypoints_in_range(0.7, 1.0)
            left_knee = min(knees, key=lambda p: p[1]) if knees else None
            right_knee = max(knees, key=lambda p: p[1]) if knees else None

            return {
                blob_id: {
                    'Head': (head[1], head[0]),
                    'Left Shoulder': (left_shoulder[1], left_shoulder[0]) if left_shoulder else None,
                    'Right Shoulder': (right_shoulder[1], right_shoulder[0]) if right_shoulder else None,
                    'Left Hip': (left_hip[1], left_hip[0]) if left_hip else None,
                    'Right Hip': (right_hip[1], right_hip[0]) if right_hip else None,
                    'Left Knee': (left_knee[1], left_knee[0]) if left_knee else None,
                    'Right Knee': (right_knee[1], right_knee[0]) if right_knee else None,
                }
            }
        except Exception as e:
            print(f"[WARNING] Joint extraction failed for blob {blob_id}: {e}")
            return {}


class OptimizedBiomechanicalAnalyzer:
    """Enhanced biomechanical analyzer with improved error handling"""

    def __init__(self):
        self.all_frames = []
        self.stroke_phases = []

        # Phase detection thresholds (optimized)
        self.catch_knee_threshold = 120
        self.drive_knee_threshold = 140
        self.finish_knee_threshold = 160
        self.back_angle_threshold = 45

        # Ideal biomechanical ranges for scoring
        self.ideal_ranges = {
            'Catch': {
                'back_angle': (35, 50, 10),
                'knee_angle': (100, 130, 15),
                'elbow_angle': (160, 180, 10)
            },
            'Drive': {
                'back_angle': (40, 60, 10),
                'knee_angle': (130, 160, 15),
                'elbow_angle': (150, 180, 10)
            },
            'Finish': {
                'back_angle': (50, 75, 10),
                'knee_angle': (160, 180, 10),
                'elbow_angle': (90, 120, 15)
            },
            'Recovery': {
                'back_angle': (35, 55, 10),
                'knee_angle': (120, 170, 20),
                'elbow_angle': (160, 180, 10)
            }
        }

    def calculate_angle(self, a, b, c):
        """Calculate angle at point b formed by points a-b-c with improved stability"""
        if None in [a, b, c]:
            return None

        try:
            ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
            bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)

            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)

            if norm_ba < 1e-8 or norm_bc < 1e-8:  # Handle zero-length vectors
                return None

            cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)

            angle_degrees = np.degrees(angle)

            # Sanity check for reasonable angles
            if 0 <= angle_degrees <= 180:
                return angle_degrees
            else:
                return None

        except Exception as e:
            print(f"[WARNING] Angle calculation failed: {e}")
            return None

    def analyze_frame(self, joints_2d):
        """Analyze a single frame and return angles with better error handling"""
        angles = {}

        try:
            # Back angle (torso relative to vertical)
            if joints_2d.get('head') and joints_2d.get('hip'):
                head = joints_2d['head']
                hip = joints_2d['hip']
                vertical_ref = (hip[0], hip[1] - 50)
                back_angle = self.calculate_angle(head, hip, vertical_ref)
                if back_angle is not None:
                    angles['back'] = back_angle

            # Knee angles
            for side in ['left', 'right']:
                hip_key = f'hip_{side}'
                knee_key = f'knee_{side}'
                ankle_key = f'ankle_{side}'

                if all(joints_2d.get(key) for key in [hip_key, knee_key, ankle_key]):
                    knee_angle = self.calculate_angle(
                        joints_2d[hip_key], joints_2d[knee_key], joints_2d[ankle_key]
                    )
                    if knee_angle is not None:
                        angles[f'{side}_knee'] = knee_angle

            # Elbow angles
            for side in ['left', 'right']:
                shoulder_key = f'shoulder_{side}'
                elbow_key = f'elbow_{side}'
                wrist_key = f'wrist_{side}'

                if all(joints_2d.get(key) for key in [shoulder_key, elbow_key, wrist_key]):
                    elbow_angle = self.calculate_angle(
                        joints_2d[shoulder_key], joints_2d[elbow_key], joints_2d[wrist_key]
                    )
                    if elbow_angle is not None:
                        angles[f'{side}_elbow'] = elbow_angle

        except Exception as e:
            print(f"[WARNING] Frame analysis failed: {e}")

        return angles

    def detect_phase(self, angles):
        """Detect rowing phase with improved logic and error handling"""
        try:
            back_angle = angles.get('back')
            left_knee = angles.get('left_knee')
            right_knee = angles.get('right_knee')

            # Calculate average knee angle
            knee_angles = [a for a in [left_knee, right_knee] if a is not None and 0 <= a <= 180]
            if not knee_angles:
                return "Unknown"

            avg_knee = np.mean(knee_angles)

            # Enhanced phase detection logic
            if avg_knee <= self.catch_knee_threshold:
                if back_angle and back_angle <= self.back_angle_threshold:
                    return "Catch"
                else:
                    return "Recovery"
            elif avg_knee <= self.drive_knee_threshold:
                return "Drive"
            elif avg_knee >= self.finish_knee_threshold:
                if back_angle and back_angle > self.back_angle_threshold + 10:
                    return "Finish"
                else:
                    return "Recovery"
            else:
                return "Drive"

        except Exception as e:
            print(f"[WARNING] Phase detection failed: {e}")
            return "Unknown"

    def segment_into_strokes(self, all_frame_data):
        """Segment frames into individual strokes with improved validation"""
        strokes = []
        current_stroke = []
        last_phase = None

        for frame_num, joints in all_frame_data:
            try:
                angles = self.analyze_frame(joints)
                phase = self.detect_phase(angles)

                # Detect stroke boundaries
                if (last_phase in ["Finish", "Recovery"] and phase == "Catch" and
                        len(current_stroke) > 15):  # Minimum frames per stroke

                    # Validate stroke has proper sequence
                    phases_in_stroke = [frame['phase'] for frame in current_stroke]
                    if self._validate_stroke_sequence(phases_in_stroke):
                        strokes.append(current_stroke)

                    current_stroke = []

                current_stroke.append({
                    'frame': frame_num,
                    'phase': phase,
                    'angles': angles
                })
                last_phase = phase

            except Exception as e:
                print(f"[WARNING] Error processing frame {frame_num}: {e}")
                continue

        # Add final stroke if valid
        if current_stroke and len(current_stroke) > 15:
            phases_in_stroke = [frame['phase'] for frame in current_stroke]
            if self._validate_stroke_sequence(phases_in_stroke):
                strokes.append(current_stroke)

        return strokes

    def _validate_stroke_sequence(self, phases):
        """Validate that stroke contains proper phase sequence"""
        phase_set = set(phases)
        required_phases = {'Catch', 'Drive', 'Finish'}
        return len(required_phases.intersection(phase_set)) >= 2

    def calculate_stroke_score(self, stroke_data):
        """Calculate comprehensive 100-point score for a stroke with error handling"""
        try:
            phase_scores = {}
            phase_data = defaultdict(list)

            # Group frames by phase
            for frame_data in stroke_data:
                phase = frame_data['phase']
                if phase != "Unknown":
                    phase_data[phase].append(frame_data['angles'])

            # Score each phase
            total_weighted_score = 0
            total_weight = 0

            phase_weights = {'Catch': 0.3, 'Drive': 0.3, 'Finish': 0.25, 'Recovery': 0.15}

            for phase, frames_angles in phase_data.items():
                if phase in self.ideal_ranges and frames_angles:
                    phase_score = self._score_phase(phase, frames_angles)
                    if 0 <= phase_score <= 100:  # Sanity check
                        phase_scores[phase] = phase_score

                        weight = phase_weights.get(phase, 0.1)
                        total_weighted_score += phase_score * weight
                        total_weight += weight

            # Overall stroke score
            if total_weight > 0:
                base_score = total_weighted_score / total_weight
            else:
                base_score = 50  # Default score if no valid phases

            # Apply consistency bonus/penalty
            consistency_score = self._calculate_consistency_score(stroke_data)
            final_score = base_score * (0.85 + 0.15 * consistency_score / 100)

            return {
                'overall_score': min(100, max(0, final_score)),
                'phase_scores': phase_scores,
                'consistency_score': consistency_score,
                'stroke_length': len(stroke_data)
            }

        except Exception as e:
            print(f"[WARNING] Stroke scoring failed: {e}")
            return {
                'overall_score': 0,
                'phase_scores': {},
                'consistency_score': 0,
                'stroke_length': len(stroke_data) if stroke_data else 0
            }

    def _score_phase(self, phase_name, angles_list):
        """Score a specific phase with improved error handling"""
        if phase_name not in self.ideal_ranges or not angles_list:
            return 50

        try:
            ideal = self.ideal_ranges[phase_name]
            scores = []

            # Calculate average angles for this phase
            avg_angles = {}
            for angle_type in ['back', 'left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
                values = []
                for angles in angles_list:
                    if (angle_type in angles and angles[angle_type] is not None and
                            0 <= angles[angle_type] <= 180):  # Sanity check
                        values.append(angles[angle_type])
                if values:
                    avg_angles[angle_type] = np.mean(values)

            # Score back angle
            if 'back' in avg_angles:
                back_score = self._score_angle(avg_angles['back'], ideal['back_angle'])
                scores.append(back_score)

            # Score knee angle (average of left/right)
            knee_values = [avg_angles[f'{side}_knee'] for side in ['left', 'right']
                           if f'{side}_knee' in avg_angles]
            if knee_values:
                avg_knee = np.mean(knee_values)
                knee_score = self._score_angle(avg_knee, ideal['knee_angle'])
                scores.append(knee_score)

            # Score elbow angle (average of left/right)
            elbow_values = [avg_angles[f'{side}_elbow'] for side in ['left', 'right']
                            if f'{side}_elbow' in avg_angles]
            if elbow_values:
                avg_elbow = np.mean(elbow_values)
                elbow_score = self._score_angle(avg_elbow, ideal['elbow_angle'])
                scores.append(elbow_score)

            return np.mean(scores) if scores else 50

        except Exception as e:
            print(f"[WARNING] Phase scoring failed for {phase_name}: {e}")
            return 50

    def _score_angle(self, measured_angle, ideal_range):
        """Score individual angle against ideal range"""
        try:
            min_ideal, max_ideal, tolerance = ideal_range

            if min_ideal <= measured_angle <= max_ideal:
                return 100

            # Calculate deviation from ideal range
            if measured_angle < min_ideal:
                deviation = min_ideal - measured_angle
            else:
                deviation = measured_angle - max_ideal

            # Score decreases linearly with deviation
            score = max(0, 100 - (deviation / tolerance) * 50)
            return score

        except Exception as e:
            print(f"[WARNING] Angle scoring failed: {e}")
            return 50

    def _calculate_consistency_score(self, stroke_data):
        """Calculate consistency score based on phase transitions"""
        try:
            phases = [frame['phase'] for frame in stroke_data if frame['phase'] != "Unknown"]

            if len(phases) < 2:
                return 50

            # Count phase transitions
            transitions = 0
            last_phase = None

            for phase in phases:
                if last_phase and last_phase != phase:
                    transitions += 1
                last_phase = phase

            # Ideal number of transitions (Catch->Drive->Finish->Recovery)
            expected_transitions = 3

            # Score based on transition smoothness
            if transitions <= expected_transitions + 2:
                return max(0, 100 - abs(transitions - expected_transitions) * 10)
            else:
                return max(0, 100 - (transitions - expected_transitions) * 15)

        except Exception as e:
            print(f"[WARNING] Consistency scoring failed: {e}")
            return 50

    def analyze_session(self, all_frame_data):
        """Analyze complete session and return comprehensive results"""
        try:
            strokes = self.segment_into_strokes(all_frame_data)

            if not strokes:
                return {
                    'overall_score': 0,
                    'stroke_scores': [],
                    'average_stroke_score': 0,
                    'total_strokes': 0,
                    'session_consistency': 0,
                    'stroke_rate': None
                }

            # Score each stroke
            stroke_scores = []
            for i, stroke in enumerate(strokes):
                try:
                    score_data = self.calculate_stroke_score(stroke)
                    stroke_scores.append(score_data)
                except Exception as e:
                    print(f"[WARNING] Failed to score stroke {i + 1}: {e}")
                    continue

            if not stroke_scores:
                return {
                    'overall_score': 0,
                    'stroke_scores': [],
                    'average_stroke_score': 0,
                    'total_strokes': 0,
                    'session_consistency': 0,
                    'stroke_rate': None
                }

            # Calculate session metrics
            overall_scores = [s['overall_score'] for s in stroke_scores if 0 <= s['overall_score'] <= 100]
            average_score = np.mean(overall_scores) if overall_scores else 0

            # Session consistency (coefficient of variation)
            if len(overall_scores) > 1:
                cv = np.std(overall_scores) / np.mean(overall_scores) if np.mean(overall_scores) > 0 else 1
                session_consistency = max(0, 100 - cv * 100)
            else:
                session_consistency = 100

            return {
                'overall_score': average_score,
                'stroke_scores': stroke_scores,
                'average_stroke_score': average_score,
                'total_strokes': len(strokes),
                'session_consistency': session_consistency,
                'stroke_rate': self._calculate_stroke_rate(strokes)
            }

        except Exception as e:
            print(f"[WARNING] Session analysis failed: {e}")
            return {
                'overall_score': 0,
                'stroke_scores': [],
                'average_stroke_score': 0,
                'total_strokes': 0,
                'session_consistency': 0,
                'stroke_rate': None
            }

    def _calculate_stroke_rate(self, strokes):
        """Calculate strokes per minute"""
        try:
            if len(strokes) < 2:
                return None

            total_frames = sum(len(stroke) for stroke in strokes)
            # Assuming 30 fps (adjust based on actual frame rate)
            total_seconds = total_frames / 30.0
            total_minutes = total_seconds / 60.0

            return len(strokes) / total_minutes if total_minutes > 0 else None

        except Exception as e:
            print(f"[WARNING] Stroke rate calculation failed: {e}")
            return None

    def load_joints_from_csv(self, csv_path):
        """Load joint data from CSV - optimized for memory with error handling"""
        joint_map = {
            'head': 'head',
            'left_shoulder': 'shoulder_left',
            'right_shoulder': 'shoulder_right',
            'left_hip': 'hip_left',
            'right_hip': 'hip_right',
            'left_knee': 'knee_left',
            'right_knee': 'knee_right',
            'left_elbow': 'elbow_left',
            'right_elbow': 'elbow_right',
            'left_wrist': 'wrist_left',
            'right_wrist': 'wrist_right',
            'left_ankle': 'ankle_left',
            'right_ankle': 'ankle_right',
        }

        def normalize_joint_name(s):
            return s.strip().lower().replace(" ", "_")

        # Load all data into memory at once
        frames_data = defaultdict(dict)

        try:
            with open(csv_path, newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f, skipinitialspace=True)
                try:
                    first_row = next(reader)
                except StopIteration:
                    print("[WARNING] Empty CSV file")
                    return

                # Check if first row is header
                is_header = first_row[0].strip().lower() == 'frame'
                if not is_header:
                    rows = [first_row] + list(reader)
                else:
                    rows = list(reader)

                # Process all rows in memory
                for row_num, row in enumerate(rows, start=2):
                    if not row or len(row) < 4:
                        continue

                    try:
                        # Handle both formats
                        if len(row) >= 5:
                            frame_str, _, joint_str, x_str, y_str = row[:5]
                        else:
                            frame_str, joint_str, x_str, y_str = row[:4]

                        frame = int(frame_str)
                        x, y = float(x_str), float(y_str)

                        # Sanity check coordinates
                        if not (0 <= x <= 10000 and 0 <= y <= 10000):  # Reasonable bounds
                            continue

                    except ValueError as e:
                        print(f"[WARNING] Invalid data in CSV row {row_num}: {e}")
                        continue

                    joint_name = normalize_joint_name(joint_str)
                    if joint_name in joint_map:
                        mapped_name = joint_map[joint_name]
                        frames_data[frame][mapped_name] = (x, y)

        except FileNotFoundError:
            print(f"[ERROR] CSV file not found: {csv_path}")
            return
        except Exception as e:
            print(f"[ERROR] Failed to read CSV file: {e}")
            return

        # Process and yield frame data
        for frame_num in sorted(frames_data.keys()):
            joints = frames_data[frame_num]

            # Compute hip center if both hips available
            if 'hip_left' in joints and 'hip_right' in joints:
                left_hip = joints['hip_left']
                right_hip = joints['hip_right']
                joints['hip'] = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )

            yield frame_num, joints


class OptimizedVideoProcessor:
    """Process video entirely in memory with minimal disk access and improved error handling"""

    def __init__(self):
        self.pose_estimator = OptimizedPoseEstimation()
        self.biomech_analyzer = OptimizedBiomechanicalAnalyzer()

    def process_video_to_joints(self, video_path, frame_skip=3):
        """Process entire video and extract joints, keeping everything in memory"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        all_joints = []  # Store all joint data in memory
        frame_count = 0
        processed_count = 0

        print("[INFO] Processing video frames...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    joints = self._process_single_frame(frame, processed_count)
                    if joints:
                        all_joints.extend(joints)
                    processed_count += 1

                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"[INFO] Processed {frame_count} frames, extracted joints from {processed_count} frames...")

        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
        finally:
            cap.release()

        print(f"[INFO] Completed processing {frame_count} frames")
        print(f"[INFO] Successfully processed {processed_count} frames")
        print(f"[INFO] Extracted {len(all_joints)} joint data points")

        return all_joints

    def _process_single_frame(self, frame, frame_num):
        """Process a single frame and extract joints with comprehensive error handling"""
        try:
            if frame is None or frame.size == 0:
                return []

            # Convert to grayscale
            gray = self.pose_estimator.grayscale(frame)

            # Apply gaussian blur
            try:
                kernel = self.pose_estimator.gaussian_blur(3.0, 5)
                blur = self.pose_estimator.convolve_frame(gray, kernel)
            except Exception as e:
                print(f"[WARNING] Blur failed for frame {frame_num}: {e}")
                blur = gray

            # Background subtraction
            try:
                fg_mask = self.pose_estimator.background_subtraction(gray)
            except Exception as e:
                print(f"[WARNING] Background subtraction failed for frame {frame_num}: {e}")
                return []

            # Morphological operations
            try:
                fg_mask = self.pose_estimator.morphological_operations(fg_mask, 'open', 3)
                fg_mask = self.pose_estimator.morphological_operations(fg_mask, 'close', 5)
            except Exception as e:
                print(f"[WARNING] Morphological operations failed for frame {frame_num}: {e}")

            # Connected components
            try:
                labels, num_labels = self.pose_estimator.connected_components_labeling(fg_mask)
            except Exception as e:
                print(f"[WARNING] Connected components failed for frame {frame_num}: {e}")
                return []

            if num_labels == 0:
                return []

            # Extract keypoints on foreground
            try:
                blur_fg = np.where(fg_mask > 0, blur, 0)
                keypoints = self.pose_estimator.shi_tomasi_corner_detection(blur_fg)
            except Exception as e:
                print(f"[WARNING] Corner detection failed for frame {frame_num}: {e}")
                keypoints = []

            # Extract joints for each blob
            joints_data = []
            for blob_id in range(1, min(num_labels + 1, 10)):  # Limit to 10 blobs max
                try:
                    ys, xs = np.where(labels == blob_id)
                    if len(ys) == 0:
                        continue

                    bbox = (min(ys), max(ys))
                    joints = self.pose_estimator.extract_anatomical_joints(bbox, keypoints, labels, blob_id)

                    if joints and blob_id in joints:
                        # Convert to the format expected by biomechanical analyzer
                        joint_dict = joints[blob_id]
                        for joint_name, coords in joint_dict.items():
                            if coords is not None:
                                # Map joint names to standard format
                                standard_name = self._map_joint_name(joint_name)
                                joints_data.append((frame_num, standard_name, coords[0], coords[1]))

                except Exception as e:
                    print(f"[WARNING] Joint extraction failed for blob {blob_id} in frame {frame_num}: {e}")
                    continue

            return joints_data

        except Exception as e:
            print(f"[WARNING] Error processing frame {frame_num}: {e}")
            return []

    def _map_joint_name(self, joint_name):
        """Map joint names to standard format"""
        name_map = {
            'Head': 'head',
            'Left Shoulder': 'left_shoulder',
            'Right Shoulder': 'right_shoulder',
            'Left Hip': 'left_hip',
            'Right Hip': 'right_hip',
            'Left Knee': 'left_knee',
            'Right Knee': 'right_knee',
        }
        return name_map.get(joint_name, joint_name.lower().replace(' ', '_'))

    def save_joints_to_csv(self, joints_data, csv_path):
        """Save joints to CSV file once at the end"""
        print(f"[INFO] Saving {len(joints_data)} joint points to {csv_path}")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'joint_name', 'x', 'y'])

                for frame_num, joint_name, x, y in joints_data:
                    writer.writerow([frame_num, joint_name, x, y])

            print(f"[INFO] Successfully saved joints to {csv_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    def process_and_analyze_video(self, video_path, output_csv=None, frame_skip=3):
        """Complete pipeline: video -> joints -> biomechanical analysis"""
        try:
            # Extract joints from video (all in memory)
            joints_data = self.process_video_to_joints(video_path, frame_skip)

            if not joints_data:
                print("[WARNING] No joint data extracted from video")
                return {
                    'overall_score': 0,
                    'stroke_scores': [],
                    'average_stroke_score': 0,
                    'total_strokes': 0,
                    'session_consistency': 0,
                    'stroke_rate': None
                }, []

            # Save joints if path provided
            if output_csv:
                self.save_joints_to_csv(joints_data, output_csv)

            # Convert to format expected by analyzer
            frame_data = self._convert_joints_for_analysis(joints_data)

            if not frame_data:
                print("[WARNING] No frame data for analysis")
                return {
                    'overall_score': 0,
                    'stroke_scores': [],
                    'average_stroke_score': 0,
                    'total_strokes': 0,
                    'session_consistency': 0,
                    'stroke_rate': None
                }, joints_data

            # Perform biomechanical analysis
            print("[INFO] Performing biomechanical analysis...")
            analysis_results = self.biomech_analyzer.analyze_session(frame_data)

            return analysis_results, joints_data

        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            return {
                'overall_score': 0,
                'stroke_scores': [],
                'average_stroke_score': 0,
                'total_strokes': 0,
                'session_consistency': 0,
                'stroke_rate': None
            }, []

    def _convert_joints_for_analysis(self, joints_data):
        """Convert joint data to format expected by biomechanical analyzer"""
        try:
            frames_dict = defaultdict(dict)

            # Group joints by frame
            for frame_num, joint_name, x, y in joints_data:
                frames_dict[frame_num][joint_name] = (x, y)

            # Convert to list of (frame_num, joints_dict) tuples
            frame_data = []
            for frame_num in sorted(frames_dict.keys()):
                joints = frames_dict[frame_num]

                # Add hip center if both hips exist
                if 'left_hip' in joints and 'right_hip' in joints:
                    left_hip = joints['left_hip']
                    right_hip = joints['right_hip']
                    joints['hip'] = (
                        (left_hip[0] + right_hip[0]) / 2,
                        (left_hip[1] + right_hip[1]) / 2
                    )

                frame_data.append((frame_num, joints))

            return frame_data

        except Exception as e:
            print(f"[ERROR] Joint data conversion failed: {e}")
            return []


class ActionQualityReporter:
    """Generate comprehensive quality assessment reports with improved error handling"""

    def __init__(self):
        self.grade_thresholds = {
            'A+': 95, 'A': 90, 'A-': 85,
            'B+': 82, 'B': 78, 'B-': 75,
            'C+': 72, 'C': 68, 'C-': 65,
            'D+': 62, 'D': 58, 'D-': 55,
            'F': 0
        }

    def generate_report(self, analysis_results, output_path):
        """Generate comprehensive action quality report"""
        try:
            score = analysis_results.get('overall_score', 0)
            grade = self._get_grade(score)

            report = f"""
=== ROWING ACTION QUALITY ASSESSMENT REPORT ===
Generated: {self._get_timestamp()}

OVERALL PERFORMANCE
==================
Final Score: {score:.1f}/100
Grade: {grade}
Performance Level: {self._get_performance_level(score)}

SESSION OVERVIEW
===============
Total Strokes Analyzed: {analysis_results.get('total_strokes', 0)}
Average Stroke Score: {analysis_results.get('average_stroke_score', 0):.1f}
Session Consistency: {analysis_results.get('session_consistency', 0):.1f}/100
Stroke Rate: {analysis_results.get('stroke_rate', 'N/A')} strokes/min

INDIVIDUAL STROKE ANALYSIS
=========================
"""

            # Add individual stroke scores
            stroke_scores = analysis_results.get('stroke_scores', [])
            if stroke_scores:
                for i, stroke in enumerate(stroke_scores, 1):
                    try:
                        stroke_score = stroke.get('overall_score', 0)
                        consistency = stroke.get('consistency_score', 0)
                        report += f"Stroke {i:2d}: {stroke_score:5.1f}/100 "
                        report += f"(Consistency: {consistency:3.0f}) "
                        report += f"[{self._get_grade(stroke_score)}]\n"

                        # Add phase breakdown for detailed strokes
                        phase_scores = stroke.get('phase_scores', {})
                        if phase_scores:
                            for phase, phase_score in phase_scores.items():
                                report += f"    {phase:8s}: {phase_score:5.1f}/100\n"
                            report += "\n"
                    except Exception as e:
                        print(f"[WARNING] Error formatting stroke {i}: {e}")
                        continue

            # Performance recommendations
            report += f"""
PERFORMANCE ANALYSIS
===================
{self._get_detailed_feedback(analysis_results)}

IMPROVEMENT RECOMMENDATIONS
==========================
{self._get_improvement_recommendations(analysis_results)}

TECHNICAL NOTES
===============
- Scoring based on biomechanical analysis of joint angles
- Phase detection: Catch, Drive, Finish, Recovery
- Consistency measured across stroke-to-stroke variation
- Ideal angle ranges based on rowing biomechanics research

"""

            # Save report
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"[INFO] Report saved successfully to {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save report: {e}")

            return report

        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
            return "Error generating report"

    def _get_grade(self, score):
        """Convert numerical score to letter grade"""
        try:
            for grade, threshold in self.grade_thresholds.items():
                if score >= threshold:
                    return grade
            return 'F'
        except:
            return 'F'

    def _get_performance_level(self, score):
        """Get performance level description"""
        try:
            if score >= 90:
                return "Elite/Competitive"
            elif score >= 80:
                return "Advanced"
            elif score >= 70:
                return "Intermediate"
            elif score >= 60:
                return "Beginner+"
            else:
                return "Novice"
        except:
            return "Unknown"

    def _get_detailed_feedback(self, results):
        """Generate detailed performance feedback"""
        try:
            score = results.get('overall_score', 0)
            consistency = results.get('session_consistency', 0)
            stroke_count = results.get('total_strokes', 0)

            feedback = []

            # Overall performance feedback
            if score >= 85:
                feedback.append("Excellent technique demonstrated throughout the session.")
            elif score >= 70:
                feedback.append("Good technique with room for refinement in specific areas.")
            else:
                feedback.append("Technique needs significant improvement across multiple phases.")

            # Consistency feedback
            if consistency >= 90:
                feedback.append("Outstanding consistency across all strokes.")
            elif consistency >= 75:
                feedback.append("Good consistency with minor variations between strokes.")
            else:
                feedback.append("Inconsistent technique - focus on stroke-to-stroke repeatability.")

            # Session volume feedback
            if stroke_count >= 20:
                feedback.append("Good session volume for comprehensive analysis.")
            elif stroke_count >= 10:
                feedback.append("Moderate session volume - longer sessions provide better analysis.")
            else:
                feedback.append("Short session - consider longer training sessions for better assessment.")

            return "\n".join(f"• {item}" for item in feedback)

        except Exception as e:
            print(f"[WARNING] Feedback generation failed: {e}")
            return "• Unable to generate detailed feedback"

    def _get_improvement_recommendations(self, results):
        """Generate specific improvement recommendations"""
        try:
            recommendations = []
            score = results.get('overall_score', 0)
            consistency = results.get('session_consistency', 0)

            # Score-based recommendations
            if score < 70:
                recommendations.extend([
                    "Focus on basic rowing technique fundamentals",
                    "Work with a qualified coach to address form issues",
                    "Practice phase transitions (Catch -> Drive -> Finish -> Recovery)",
                    "Concentrate on proper body positioning throughout the stroke"
                ])
            elif score < 85:
                recommendations.extend([
                    "Fine-tune body positioning in specific stroke phases",
                    "Work on consistency between strokes",
                    "Focus on smooth phase transitions",
                    "Practice maintaining form at different intensities"
                ])
            else:
                recommendations.extend([
                    "Maintain excellent form while increasing stroke rate",
                    "Focus on competition-specific scenarios",
                    "Work on advanced technique refinements",
                    "Consider power and endurance training while maintaining technique"
                ])

            # Consistency-based recommendations
            if consistency < 80:
                recommendations.extend([
                    "Practice stroke repeatability drills",
                    "Focus on rhythm and timing consistency",
                    "Use mirrors or video feedback during training",
                    "Work on muscle memory development"
                ])

            return "\n".join(f"• {item}" for item in recommendations)

        except Exception as e:
            print(f"[WARNING] Recommendations generation failed: {e}")
            return "• Unable to generate specific recommendations"

    def _get_timestamp(self):
        """Get current timestamp for report"""
        try:
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "Unknown"


# Main execution function
def main():
    """Main execution pipeline with comprehensive error handling"""

    # Configuration
    video_path = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset\VID-20250609-WA0017_1 Paida.mp4"
    output_csv = r"C:\Users\brigh\Documents\Honours\HYP\HYP Project\src\PipelineOne\csv files\optimized_joints.csv"
    report_path = r"C:\Users\brigh\Documents\Honours\HYP\HYP Project\src\PipelineOne\rowing_quality_report.txt"

    # Initialize processor
    processor = OptimizedVideoProcessor()
    reporter = ActionQualityReporter()

    try:
        print("=== STARTING ROWING ACTION QUALITY ASSESSMENT ===")

        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return None

        # Process video and analyze
        analysis_results, joints_data = processor.process_and_analyze_video(
            video_path, output_csv, frame_skip=3
        )

        # Generate comprehensive report
        print("[INFO] Generating quality assessment report...")
        report = reporter.generate_report(analysis_results, report_path)

        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Overall Score: {analysis_results.get('overall_score', 0):.1f}/100")
        print(f"Grade: {reporter._get_grade(analysis_results.get('overall_score', 0))}")
        print(f"Total Strokes: {analysis_results.get('total_strokes', 0)}")
        print(f"Report saved to: {report_path}")

        return analysis_results

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
        return None
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()