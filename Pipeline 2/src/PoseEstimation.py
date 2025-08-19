import os
import numpy as np
import cv2
from scripts import FrameExtractor as fc
from scripts.BackgroundSubtractor import BackgroundSubtractor as bs



class PoseEstimator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.stroke_phase_history = []
        self.joint_consistency_tracker = {}
        self.previous_joints = {}
        self.joint_smoothing_factor = 0.6  # Reduced for more responsiveness

        # Enhanced head detection parameters
        self.head_detection_params = {
            'min_compactness': 0.15,  # Minimum compactness for head region
            'max_width_ratio': 0.25,  # Max width relative to total body width
            'min_density': 0.3,  # Minimum point density in head region
            'top_region_ratio': 0.25  # Look in top 25% of body
        }

    def run(self):
        # Initialize components
        frame_extractor = fc.FrameExtractor(self.video_path)
        frame_extractor.extract_frames()

        first_frame_path = f"{frame_extractor.output_dir}/frame_0000.png"
        frame = frame_extractor.load_grayscale_frame(first_frame_path)

        if frame is None or frame.size == 0:
            print("Error: Could not load first frame")
            return

        print(f"[Info] First frame shape: {frame.shape}")

        # Initialize background subtractor with better parameters
        bg_subtractor = bs(shape=None, alpha=0.003, history_size=40)

        import glob
        frame_paths = sorted(glob.glob(f"{frame_extractor.output_dir}/frame_*.png"))

        # Build background model with fewer frames but better selection
        print("Building enhanced background model...")
        background_frames = min(30, len(frame_paths) // 3)

        # Skip some frames to get more diverse background
        step = max(1, len(frame_paths[:background_frames * 2]) // background_frames)

        for i, frame_idx in enumerate(range(0, background_frames * step, step)):
            if frame_idx < len(frame_paths):
                frame = frame_extractor.load_grayscale_frame(frame_paths[frame_idx])
                if frame is not None and frame.size > 0:
                    bg_subtractor.update(frame)
                    if (i + 1) % 10 == 0:
                        print(f"Processing background frame {i + 1}/{background_frames}")

        print("Processing frames with improved pose estimation...")
        stroke_phases = []

        for frame_idx, frame_path in enumerate(frame_paths):
            frame = frame_extractor.load_grayscale_frame(frame_path)

            if frame is None or frame.size == 0:
                stroke_phases.append('unknown')
                continue

            try:
                # Enhanced foreground mask with better parameters
                fg_mask = self._get_improved_foreground_mask(bg_subtractor, frame)

                # Get debug visualizations
                diff_map = bg_subtractor.get_difference_map(frame)
                confidence_map = bg_subtractor.get_background_confidence(frame)
                variance_map = bg_subtractor.get_variance_image()

            except Exception as e:
                print(f"[Error] Failed to process frame {frame_idx}: {e}")
                stroke_phases.append('unknown')
                continue

            # Improved component analysis
            components = self._get_improved_components(fg_mask)

            # Visualization
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Find main rower with better heuristics
            main_component = self._find_improved_main_component(components, frame.shape)

            if main_component is not None:
                label, points = main_component

                # Enhanced joint estimation
                joints = self._estimate_improved_joints(points, frame.shape, frame)

                # Apply temporal smoothing
                joints = self._smooth_joints_improved(joints)

                # Enhanced stroke analysis
                stroke_analysis = self._analyze_stroke_enhanced(joints, frame.shape)

                # Draw improved visualization
                self._draw_improved_tracking(frame_vis, joints, stroke_analysis, points)

                # Print analysis
                self._print_improved_analysis(frame_idx, joints, stroke_analysis)

                stroke_phases.append(stroke_analysis['stroke_phase'])

            else:
                print(f"Frame {frame_idx}: No rower detected")
                stroke_phases.append('unknown')

            # Create debug visualization
            debug_vis = self._create_improved_debug_visualization(
                frame, fg_mask, diff_map, confidence_map, variance_map, frame_vis
            )

            # Show visualizations
            cv2.imshow("Improved Debug", debug_vis)
            cv2.imshow("Improved Tracking", frame_vis)

            # Stroke timeline
            timeline_img = self._create_stroke_timeline(stroke_phases, frame_idx)
            cv2.imshow("Stroke Timeline", timeline_img)

            # Save outputs
            try:
                out_name = os.path.basename(frame_path).replace("frame_", "")
                cv2.imwrite(f"{frame_extractor.output_dir}/improved_mask_{out_name}", fg_mask * 255)
                cv2.imwrite(f"{frame_extractor.output_dir}/improved_tracking_{out_name}", frame_vis)
            except Exception as e:
                print(f"[Warning] Could not save images: {e}")

            key = cv2.waitKey(100)
            if key == 27:  # Esc
                break
            elif key == ord('p'):  # Pause
                cv2.waitKey(0)
            elif key == ord('r'):  # Reset background
                print("Resetting background...")
                bg_subtractor.reset_background()

        self._print_session_summary(stroke_phases)
        cv2.destroyAllWindows()

    def _get_improved_foreground_mask(self, bg_subtractor, frame):
        """Get improved foreground mask with better preprocessing"""
        # Apply slight blur to reduce noise before background subtraction
        blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        # Get mask with adjusted threshold
        base_mask = bg_subtractor.get_cleaned_mask(blurred_frame, threshold=22, min_area=800)

        # Additional morphological cleaning
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Remove small noise
        cleaned = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel3)
        # Fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5)
        # Final smoothing
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel7)

        return cleaned

    def _get_improved_components(self, fg_mask):
        """Improved component analysis with better filtering"""
        # Use existing connected components method
        components = bs.connected_components(fg_mask, connectivity=8)

        # Filter components by size and shape
        filtered_components = {}
        h, w = fg_mask.shape
        min_area = (h * w) * 0.005  # At least 0.5% of frame
        max_area = (h * w) * 0.6  # At most 60% of frame

        for label, points in components.items():
            if len(points) < min_area or len(points) > max_area:
                continue

            # Calculate basic shape metrics
            ys, xs = zip(*points) if points else ([], [])
            if not ys or not xs:
                continue

            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Filter by aspect ratio (should be taller than wide for person)
            if height > 0 and width > 0:
                aspect_ratio = height / width
                if 0.8 < aspect_ratio < 4.0:  # Reasonable human proportions
                    filtered_components[label] = points

        return filtered_components

    def _find_improved_main_component(self, components, frame_shape):
        """Find main component with better scoring"""
        if not components:
            return None

        h, w = frame_shape
        best_component = None
        best_score = 0

        for label, points in components.items():
            if not points:
                continue

            # Calculate metrics
            ys, xs = zip(*points)
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            area = len(points)

            # Scoring factors
            size_score = min(1.0, area / 3000)  # Prefer larger components

            # Aspect ratio score (prefer taller shapes)
            aspect_ratio = height / width if width > 0 else 0
            aspect_score = 1.0 if 1.2 < aspect_ratio < 2.5 else 0.7

            # Position score (prefer center-positioned components)
            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2
            y_pos_score = 1.0 if 0.2 < center_y / h < 0.8 else 0.8
            x_pos_score = 1.0 if 0.2 < center_x / w < 0.8 else 0.9

            # Compactness score (prefer more compact shapes)
            bounding_area = width * height
            compactness = area / bounding_area if bounding_area > 0 else 0
            compact_score = min(1.0, compactness * 3)  # Boost compactness

            total_score = size_score * aspect_score * y_pos_score * x_pos_score * compact_score

            if total_score > best_score:
                best_score = total_score
                best_component = (label, points)

        return best_component

    def _estimate_improved_joints(self, points, frame_shape, frame):
        """Improved joint estimation with better head detection"""
        if not points:
            return self._get_default_joints()

        h, w = frame_shape
        points_array = np.array(points, dtype=np.int32)
        ys, xs = points_array[:, 0], points_array[:, 1]

        # Basic measurements
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        height = max_y - min_y
        width = max_x - min_x
        center_x = int(np.mean(xs))

        # Enhanced horizontal slice analysis
        joints = self._analyze_body_slices_improved(points, min_y, max_y, min_x, max_x, height, width)

        # Validate and correct joint positions
        joints = self._validate_joint_positions(joints, min_y, max_y, min_x, max_x, height)

        return joints

    def _analyze_body_slices_improved(self, points, min_y, max_y, min_x, max_x, height, width):
        """Improved body slice analysis for better joint detection"""
        # Create more detailed slices
        num_slices = 25
        slice_height = max(1, height // num_slices)
        slice_data = []

        for i in range(num_slices):
            slice_y_min = min_y + i * slice_height
            slice_y_max = min_y + (i + 1) * slice_height

            # Get points in this slice
            slice_points = [p for p in points if slice_y_min <= p[0] < slice_y_max]

            if slice_points:
                slice_xs = [p[1] for p in slice_points]
                slice_width = max(slice_xs) - min(slice_xs)
                slice_center_x = int(np.mean(slice_xs))
                slice_density = len(slice_points)

                # Calculate compactness
                if slice_width > 0:
                    compactness = slice_density / slice_width
                else:
                    compactness = slice_density

                slice_data.append({
                    'y_center': (slice_y_min + slice_y_max) // 2,
                    'x_center': slice_center_x,
                    'width': slice_width,
                    'density': slice_density,
                    'compactness': compactness,
                    'points': slice_points,
                    'x_min': min(slice_xs),
                    'x_max': max(slice_xs)
                })
            else:
                slice_data.append({
                    'y_center': (slice_y_min + slice_y_max) // 2,
                    'x_center': min_x + width // 2,
                    'width': 0,
                    'density': 0,
                    'compactness': 0,
                    'points': [],
                    'x_min': min_x + width // 2,
                    'x_max': min_x + width // 2
                })

        # Find head with improved detection
        head_pos = self._find_improved_head(slice_data, height, width)

        # Find shoulders (widest region in upper body)
        shoulder_pos = self._find_improved_shoulders(slice_data, head_pos, height)

        # Find hips (stable wide region in lower torso)
        hip_pos = self._find_improved_hips(slice_data, shoulder_pos, height)

        # Find arms and legs
        arm_joints = self._find_improved_arms(points, shoulder_pos, hip_pos)
        leg_joints = self._find_improved_legs(points, hip_pos, max_y)

        joints = {
            'head': head_pos,
            'neck': self._interpolate_point(head_pos, shoulder_pos, 0.6),
            'shoulder_left': (shoulder_pos[0], shoulder_pos[1] - width // 6),
            'shoulder_right': (shoulder_pos[0], shoulder_pos[1] + width // 6),
            'spine_mid': self._interpolate_point(shoulder_pos, hip_pos, 0.5),
            'hips': hip_pos,
            **arm_joints,
            **leg_joints
        }

        return joints

    def _find_improved_head(self, slice_data, height, width):
        """Improved head detection using multiple criteria"""
        # Look in top portion of body
        top_region_end = len(slice_data) // 4  # Top 25%
        head_candidates = slice_data[:top_region_end]

        valid_candidates = [s for s in head_candidates if s['density'] > 0]

        if not valid_candidates:
            # Fallback: use topmost slice with any points
            for s in slice_data:
                if s['density'] > 0:
                    return (s['y_center'], s['x_center'])
            return (slice_data[0]['y_center'], slice_data[0]['x_center'])

        # Score candidates based on head-like characteristics
        best_slice = None
        best_score = 0

        for s in valid_candidates:
            # Head should be compact and reasonably sized
            size_score = 1.0 if width * 0.15 < s['width'] < width * 0.4 else 0.5

            # Moderate density (not too sparse, not too dense)
            density_normalized = min(1.0, s['density'] / 100)
            density_score = density_normalized

            # Compactness score
            compact_score = min(1.0, s['compactness'] / 5)

            # Position score (prefer higher positions)
            position_score = 1.0 - (slice_data.index(s) / len(head_candidates))

            total_score = size_score * density_score * compact_score * position_score

            if total_score > best_score:
                best_score = total_score
                best_slice = s

        if best_slice:
            return (best_slice['y_center'], best_slice['x_center'])
        else:
            return (valid_candidates[0]['y_center'], valid_candidates[0]['x_center'])

    def _find_improved_shoulders(self, slice_data, head_pos, height):
        """Find shoulders as widest region below head"""
        head_y = head_pos[0]

        # Look for shoulders in region below head
        shoulder_start_idx = 0
        for i, s in enumerate(slice_data):
            if s['y_center'] > head_y:
                shoulder_start_idx = i
                break

        shoulder_end_idx = min(len(slice_data), shoulder_start_idx + len(slice_data) // 3)
        shoulder_candidates = slice_data[shoulder_start_idx:shoulder_end_idx]

        valid_candidates = [s for s in shoulder_candidates if s['density'] > 0]

        if valid_candidates:
            # Find widest region with good density
            best_slice = max(valid_candidates, key=lambda s: s['width'] * (s['density'] / 100))
            return (best_slice['y_center'], best_slice['x_center'])
        else:
            # Fallback
            fallback_y = head_y + height // 8
            return (fallback_y, head_pos[1])

    def _find_improved_hips(self, slice_data, shoulder_pos, height):
        """Find hips in middle-lower torso region"""
        shoulder_y = shoulder_pos[0]

        # Look for hips in middle region
        hip_start_idx = len(slice_data) // 3  # Start from 1/3 down
        hip_end_idx = int(len(slice_data) * 0.7)  # End at 70% down

        hip_candidates = slice_data[hip_start_idx:hip_end_idx]
        valid_candidates = [s for s in hip_candidates if s['density'] > 0 and s['y_center'] > shoulder_y]

        if valid_candidates:
            # Find region with good width and density (sitting position)
            best_slice = max(valid_candidates,
                             key=lambda s: s['width'] * min(1.0, s['density'] / 50))
            return (best_slice['y_center'], best_slice['x_center'])
        else:
            # Fallback
            fallback_y = shoulder_y + height // 3
            return (fallback_y, shoulder_pos[1])

    def _find_improved_arms(self, points, shoulder_pos, hip_pos):
        """Find arm joints with better detection"""
        shoulder_y, shoulder_x = shoulder_pos
        hip_y = hip_pos[0]

        # Look for arm points in the region around shoulder level
        arm_region_points = []
        arm_y_range = abs(hip_y - shoulder_y) // 2

        for y, x in points:
            if shoulder_y - arm_y_range // 2 <= y <= hip_y + arm_y_range // 2:
                arm_region_points.append((y, x))

        if not arm_region_points:
            return {'elbow_left': (0, 0), 'elbow_right': (0, 0),
                    'wrist_left': (0, 0), 'wrist_right': (0, 0)}

        # Separate left and right arms
        left_arm = [(y, x) for y, x in arm_region_points if x < shoulder_x - 20]
        right_arm = [(y, x) for y, x in arm_region_points if x > shoulder_x + 20]

        arms = {}

        # Process left arm
        if left_arm:
            # Sort by distance from shoulder
            left_distances = [((y - shoulder_y) ** 2 + (x - shoulder_x) ** 2, (y, x)) for y, x in left_arm]
            left_sorted = [p for _, p in sorted(left_distances, reverse=True)]

            if len(left_sorted) >= 2:
                arms['wrist_left'] = left_sorted[0]  # Farthest
                arms['elbow_left'] = left_sorted[len(left_sorted) // 2]  # Middle
            elif len(left_sorted) == 1:
                arms['wrist_left'] = left_sorted[0]
                arms['elbow_left'] = self._interpolate_point(shoulder_pos, left_sorted[0], 0.6)
            else:
                arms['wrist_left'] = (0, 0)
                arms['elbow_left'] = (0, 0)
        else:
            arms['wrist_left'] = (0, 0)
            arms['elbow_left'] = (0, 0)

        # Process right arm
        if right_arm:
            right_distances = [((y - shoulder_y) ** 2 + (x - shoulder_x) ** 2, (y, x)) for y, x in right_arm]
            right_sorted = [p for _, p in sorted(right_distances, reverse=True)]

            if len(right_sorted) >= 2:
                arms['wrist_right'] = right_sorted[0]
                arms['elbow_right'] = right_sorted[len(right_sorted) // 2]
            elif len(right_sorted) == 1:
                arms['wrist_right'] = right_sorted[0]
                arms['elbow_right'] = self._interpolate_point(shoulder_pos, right_sorted[0], 0.6)
            else:
                arms['wrist_right'] = (0, 0)
                arms['elbow_right'] = (0, 0)
        else:
            arms['wrist_right'] = (0, 0)
            arms['elbow_right'] = (0, 0)

        return arms

    def _find_improved_legs(self, points, hip_pos, max_y):
        """Find leg joints with better detection"""
        hip_y, hip_x = hip_pos

        # Look for leg points below hips
        leg_points = [(y, x) for y, x in points if y > hip_y]

        if not leg_points:
            return {'knee_left': (0, 0), 'knee_right': (0, 0),
                    'ankle_left': (0, 0), 'ankle_right': (0, 0)}

        # Separate legs
        left_leg = [(y, x) for y, x in leg_points if x < hip_x - 10]
        right_leg = [(y, x) for y, x in leg_points if x > hip_x + 10]
        center_leg = [(y, x) for y, x in leg_points if hip_x - 10 <= x <= hip_x + 10]

        # Distribute center points
        if center_leg and len(center_leg) > len(left_leg) + len(right_leg):
            center_sorted = sorted(center_leg, key=lambda p: p[1])  # Sort by x
            mid_idx = len(center_sorted) // 2
            left_leg.extend(center_sorted[:mid_idx])
            right_leg.extend(center_sorted[mid_idx:])

        legs = {}

        # Process each leg
        for side, leg_points_side, suffix in [('left', left_leg, '_left'), ('right', right_leg, '_right')]:
            if not leg_points_side:
                legs[f'knee{suffix}'] = (0, 0)
                legs[f'ankle{suffix}'] = (0, 0)
                continue

            # Sort by y-coordinate (top to bottom)
            leg_sorted = sorted(leg_points_side, key=lambda p: p[0])

            if len(leg_sorted) >= 2:
                # Knee is in upper part, ankle in lower part
                knee_idx = len(leg_sorted) // 3
                ankle_idx = min(len(leg_sorted) - 1, int(len(leg_sorted) * 0.8))

                legs[f'knee{suffix}'] = leg_sorted[knee_idx]
                legs[f'ankle{suffix}'] = leg_sorted[ankle_idx]
            elif len(leg_sorted) == 1:
                legs[f'knee{suffix}'] = leg_sorted[0]
                legs[f'ankle{suffix}'] = (0, 0)
            else:
                legs[f'knee{suffix}'] = (0, 0)
                legs[f'ankle{suffix}'] = (0, 0)

        return legs

    def _interpolate_point(self, point1, point2, ratio):
        """Interpolate between two points"""
        if point1 == (0, 0) or point2 == (0, 0):
            return (0, 0)

        y = int(point1[0] + ratio * (point2[0] - point1[0]))
        x = int(point1[1] + ratio * (point2[1] - point1[1]))
        return (y, x)

    def _validate_joint_positions(self, joints, min_y, max_y, min_x, max_x, height):
        """Validate and correct unrealistic joint positions"""
        # Ensure proper anatomical ordering
        head = joints.get('head', (0, 0))
        shoulders = ((joints.get('shoulder_left', (0, 0))[0] + joints.get('shoulder_right', (0, 0))[0]) // 2,
                     (joints.get('shoulder_left', (0, 0))[1] + joints.get('shoulder_right', (0, 0))[1]) // 2)
        hips = joints.get('hips', (0, 0))

        # Fix head position if it's below shoulders
        if head != (0, 0) and shoulders != (0, 0) and head[0] >= shoulders[0]:
            joints['head'] = (max(min_y, shoulders[0] - height // 10), head[1])

        # Fix shoulder position if it's below hips
        if shoulders != (0, 0) and hips != (0, 0) and shoulders[0] >= hips[0]:
            new_shoulder_y = hips[0] - height // 8
            joints['shoulder_left'] = (new_shoulder_y, joints['shoulder_left'][1])
            joints['shoulder_right'] = (new_shoulder_y, joints['shoulder_right'][1])

        # Ensure all joints are within bounds
        for joint_name, (y, x) in joints.items():
            if (y, x) != (0, 0):
                y = max(min_y, min(max_y, y))
                x = max(min_x, min(max_x, x))
                joints[joint_name] = (y, x)

        return joints

    def _smooth_joints_improved(self, joints):
        """Improved temporal smoothing with validation"""
        smoothed_joints = {}

        for joint_name, (y, x) in joints.items():
            if (y, x) != (0, 0):
                if joint_name in self.previous_joints:
                    prev_y, prev_x = self.previous_joints[joint_name]
                    if (prev_y, prev_x) != (0, 0):
                        # Check if movement is reasonable (not too large)
                        distance = ((y - prev_y) ** 2 + (x - prev_x) ** 2) ** 0.5

                        if distance < 100:  # Reasonable movement
                            smooth_y = int(self.joint_smoothing_factor * prev_y +
                                           (1 - self.joint_smoothing_factor) * y)
                            smooth_x = int(self.joint_smoothing_factor * prev_x +
                                           (1 - self.joint_smoothing_factor) * x)
                            smoothed_joints[joint_name] = (smooth_y, smooth_x)
                        else:
                            # Large movement - use current position
                            smoothed_joints[joint_name] = (y, x)
                    else:
                        smoothed_joints[joint_name] = (y, x)
                else:
                    smoothed_joints[joint_name] = (y, x)

                self.previous_joints[joint_name] = smoothed_joints[joint_name]
            else:
                smoothed_joints[joint_name] = (0, 0)

        return smoothed_joints

    def _analyze_stroke_enhanced(self, joints, frame_shape):
        """Enhanced stroke phase analysis"""
        analysis = {
            'stroke_phase': 'unknown',
            'torso_angle': 0,
            'leg_extension': 0,
            'arm_extension': 0,
            'confidence': 0
        }

        # Calculate torso angle
        head = joints.get('head', (0, 0))
        hips = joints.get('hips', (0, 0))

        if head != (0, 0) and hips != (0, 0):
            dx = head[1] - hips[1]
            dy = head[0] - hips[0]  # Note: y increases downward

            # Angle from vertical
            angle_rad = np.arctan2(dx, -dy)  # Negative dy for correct orientation
            analysis['torso_angle'] = np.degrees(angle_rad)
            analysis['confidence'] = 0.8

        # Determine stroke phase based on torso angle and arm positions
        torso_angle = analysis['torso_angle']

        # Get arm positions for additional analysis
        wrist_left = joints.get('wrist_left', (0, 0))
        wrist_right = joints.get('wrist_right', (0, 0))
        shoulder_left = joints.get('shoulder_left', (0, 0))
        shoulder_right = joints.get('shoulder_right', (0, 0))

        # Calculate arm extension (approximate)
        arm_extended = False
        if (wrist_left != (0, 0) and shoulder_left != (0, 0)) or (wrist_right != (0, 0) and shoulder_right != (0, 0)):
            # Check if arms are extended forward
            if wrist_left != (0, 0) and shoulder_left != (0, 0):
                arm_extension_left = wrist_left[1] - shoulder_left[1]  # Positive = forward
            else:
                arm_extension_left = 0

            if wrist_right != (0, 0) and shoulder_right != (0, 0):
                arm_extension_right = wrist_right[1] - shoulder_right[1]
            else:
                arm_extension_right = 0

            # Arms are extended if they're significantly forward of shoulders
            arm_extended = (arm_extension_left > 30) or (arm_extension_right > 30)

        # Determine stroke phase
        if torso_angle > 15:  # Forward lean
            if arm_extended:
                analysis['stroke_phase'] = 'catch'
            else:
                analysis['stroke_phase'] = 'drive'
        elif torso_angle > -10:  # Upright
            analysis['stroke_phase'] = 'drive'
        else:  # Backward lean
            if arm_extended:
                analysis['stroke_phase'] = 'recovery'
            else:
                analysis['stroke_phase'] = 'finish'

        return analysis

    def _draw_improved_tracking(self, frame_vis, joints, stroke_analysis, points):
        """Draw improved tracking visualization"""
        # Get bounding box of points
        if points:
            ys, xs = zip(*points)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Draw main bounding box
            cv2.rectangle(frame_vis, (min_x - 5, min_y - 5), (max_x + 5, max_y + 5), (0, 255, 0), 3)

            # Draw stroke phase
            phase_colors = {
                'catch': (0, 255, 255),  # Yellow
                'drive': (0, 255, 0),  # Green
                'finish': (255, 0, 0),  # Blue
                'recovery': (255, 0, 255),  # Magenta
                'unknown': (128, 128, 128)  # Gray
            }

            phase_color = phase_colors.get(stroke_analysis['stroke_phase'], (255, 255, 255))
            phase_text = stroke_analysis['stroke_phase'].upper()

            # Background for text
            text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame_vis, (min_x - 5, min_y - 40), (min_x + text_size[0] + 10, min_y - 5), (0, 0, 0), -1)
            cv2.putText(frame_vis, phase_text, (min_x, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

        # Draw skeleton with improved visualization
        self._draw_improved_skeleton(frame_vis, joints)

        # Add torso angle info
        if points:
            angle_text = f"Torso: {stroke_analysis['torso_angle']:.1f}°"
            cv2.putText(frame_vis, angle_text, (max_x - 150, max_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_improved_skeleton(self, frame_vis, joints):
        """Draw improved skeleton with better joint visualization"""
        # Define joint colors and sizes
        joint_config = {
            'head': {'color': (0, 0, 255), 'size': 10, 'show_label': True},  # Red
            'neck': {'color': (255, 255, 0), 'size': 6, 'show_label': False},  # Cyan
            'shoulder_left': {'color': (255, 255, 0), 'size': 8, 'show_label': False},
            'shoulder_right': {'color': (255, 255, 0), 'size': 8, 'show_label': False},
            'elbow_left': {'color': (255, 0, 255), 'size': 6, 'show_label': False},  # Magenta
            'elbow_right': {'color': (255, 0, 255), 'size': 6, 'show_label': False},
            'wrist_left': {'color': (0, 255, 0), 'size': 6, 'show_label': False},  # Green
            'wrist_right': {'color': (0, 255, 0), 'size': 6, 'show_label': False},
            'spine_mid': {'color': (255, 255, 255), 'size': 6, 'show_label': False},  # White
            'hips': {'color': (0, 255, 255), 'size': 10, 'show_label': True},  # Yellow
            'knee_left': {'color': (255, 0, 0), 'size': 7, 'show_label': False},  # Blue
            'knee_right': {'color': (255, 0, 0), 'size': 7, 'show_label': False},
            'ankle_left': {'color': (128, 255, 0), 'size': 6, 'show_label': False},  # Light green
            'ankle_right': {'color': (128, 255, 0), 'size': 6, 'show_label': False}
        }

        # Define skeleton connections
        connections = [
            ('head', 'neck'),
            ('neck', 'shoulder_left'),
            ('neck', 'shoulder_right'),
            ('shoulder_left', 'elbow_left'),
            ('shoulder_right', 'elbow_right'),
            ('elbow_left', 'wrist_left'),
            ('elbow_right', 'wrist_right'),
            ('neck', 'spine_mid'),
            ('spine_mid', 'hips'),
            ('hips', 'knee_left'),
            ('hips', 'knee_right'),
            ('knee_left', 'ankle_left'),
            ('knee_right', 'ankle_right')
        ]

        # Draw connections first (so joints appear on top)
        for joint1_name, joint2_name in connections:
            joint1 = joints.get(joint1_name, (0, 0))
            joint2 = joints.get(joint2_name, (0, 0))

            if joint1 != (0, 0) and joint2 != (0, 0):
                cv2.line(frame_vis, (joint1[1], joint1[0]), (joint2[1], joint2[0]),
                         (255, 255, 255), 2)

        # Draw joints
        for joint_name, config in joint_config.items():
            joint_pos = joints.get(joint_name, (0, 0))
            if joint_pos != (0, 0):
                y, x = joint_pos
                color = config['color']
                size = config['size']

                # Draw filled circle
                cv2.circle(frame_vis, (x, y), size, color, -1)
                # Draw white border
                cv2.circle(frame_vis, (x, y), size + 1, (255, 255, 255), 1)

                # Add label for key joints
                if config['show_label']:
                    label = joint_name.upper().replace('_', ' ')
                    cv2.putText(frame_vis, label, (x + 15, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw handle estimation
        self._draw_handle_estimation(frame_vis, joints)

    def _draw_handle_estimation(self, frame_vis, joints):
        """Draw rowing handle estimation"""
        wrist_left = joints.get('wrist_left', (0, 0))
        wrist_right = joints.get('wrist_right', (0, 0))

        if wrist_left != (0, 0) and wrist_right != (0, 0):
            # Handle as midpoint between wrists
            handle_y = (wrist_left[0] + wrist_right[0]) // 2
            handle_x = (wrist_left[1] + wrist_right[1]) // 2

            # Draw handle
            cv2.circle(frame_vis, (handle_x, handle_y), 10, (0, 165, 255), -1)  # Orange
            cv2.circle(frame_vis, (handle_x, handle_y), 11, (255, 255, 255), 2)

            # Draw lines from wrists to handle
            cv2.line(frame_vis, (wrist_left[1], wrist_left[0]), (handle_x, handle_y), (0, 165, 255), 3)
            cv2.line(frame_vis, (wrist_right[1], wrist_right[0]), (handle_x, handle_y), (0, 165, 255), 3)

            # Label
            cv2.putText(frame_vis, "HANDLE", (handle_x + 15, handle_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif wrist_left != (0, 0):
            # Only left wrist visible
            cv2.circle(frame_vis, (wrist_left[1], wrist_left[0]), 8, (0, 165, 255), -1)
            cv2.putText(frame_vis, "HANDLE", (wrist_left[1] + 15, wrist_left[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        elif wrist_right != (0, 0):
            # Only right wrist visible
            cv2.circle(frame_vis, (wrist_right[1], wrist_right[0]), 8, (0, 165, 255), -1)
            cv2.putText(frame_vis, "HANDLE", (wrist_right[1] + 15, wrist_right[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _create_improved_debug_visualization(self, original, mask, diff_map, confidence_map, variance_map, tracking):
        """Create improved debug visualization grid"""
        h, w = original.shape[:2]
        target_size = (w // 2, h // 2)

        # Resize all images
        orig_small = cv2.resize(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), target_size)
        mask_small = cv2.resize(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), target_size)
        diff_small = cv2.resize(cv2.cvtColor(diff_map, cv2.COLOR_GRAY2BGR), target_size)
        tracking_small = cv2.resize(tracking, target_size)

        # Handle optional maps
        if confidence_map is not None:
            conf_small = cv2.resize(cv2.cvtColor(confidence_map, cv2.COLOR_GRAY2BGR), target_size)
        else:
            conf_small = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        if variance_map is not None:
            var_small = cv2.resize(cv2.cvtColor(variance_map, cv2.COLOR_GRAY2BGR), target_size)
        else:
            var_small = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Add labels with better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Add black background for text
        def add_label(img, text, pos):
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(img, pos, (pos[0] + text_size[0] + 10, pos[1] + 25), (0, 0, 0), -1)
            cv2.putText(img, text, (pos[0] + 5, pos[1] + 18), font, font_scale, (255, 255, 255), thickness)

        add_label(orig_small, "Original", (5, 5))
        add_label(mask_small, "Improved Mask", (5, 5))
        add_label(diff_small, "Difference", (5, 5))
        add_label(tracking_small, "Improved Tracking", (5, 5))
        add_label(conf_small, "BG Confidence", (5, 5))
        add_label(var_small, "BG Variance", (5, 5))

        # Create 2x3 grid
        top_row = np.hstack([orig_small, mask_small, diff_small])
        bottom_row = np.hstack([tracking_small, conf_small, var_small])

        return np.vstack([top_row, bottom_row])

    def _create_stroke_timeline(self, stroke_phases, current_frame):
        """Create stroke phase timeline visualization"""
        timeline_height = 80
        timeline_width = 1000
        timeline_img = np.zeros((timeline_height, timeline_width, 3), dtype=np.uint8)

        if not stroke_phases:
            return timeline_img

        phase_colors = {
            'catch': (0, 255, 255),  # Yellow
            'drive': (0, 255, 0),  # Green
            'finish': (255, 0, 0),  # Blue
            'recovery': (255, 0, 255),  # Magenta
            'unknown': (64, 64, 64)  # Dark gray
        }

        # Draw timeline bars
        bar_width = timeline_width / max(len(stroke_phases), 1)

        for i, phase in enumerate(stroke_phases):
            x1 = int(i * bar_width)
            x2 = int((i + 1) * bar_width)
            color = phase_colors.get(phase, (128, 128, 128))

            # Draw phase bar
            cv2.rectangle(timeline_img, (x1, 15), (x2, 55), color, -1)

            # Highlight current frame
            if i == current_frame:
                cv2.rectangle(timeline_img, (x1, 15), (x2, 55), (255, 255, 255), 3)

        # Add frame markers every 10 frames
        for i in range(0, len(stroke_phases), 10):
            x = int(i * bar_width)
            cv2.line(timeline_img, (x, 55), (x, 65), (255, 255, 255), 1)
            if i % 20 == 0:  # Label every 20 frames
                cv2.putText(timeline_img, str(i), (x - 10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Add title and current frame info
        cv2.putText(timeline_img, "Stroke Phase Timeline", (10, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(timeline_img, f"Frame: {current_frame}/{len(stroke_phases) - 1}",
                    (timeline_width - 150, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return timeline_img

    def _print_improved_analysis(self, frame_idx, joints, stroke_analysis):
        """Print improved frame analysis"""
        if frame_idx % 10 == 0:  # Print every 10th frame to reduce clutter
            print(f"\nFrame {frame_idx} Analysis:")
            print(f"  Stroke Phase: {stroke_analysis['stroke_phase']}")
            print(f"  Torso Angle: {stroke_analysis['torso_angle']:.1f}°")
            print(f"  Detected Joints: {sum(1 for pos in joints.values() if pos != (0, 0))}/14")

            # Print key joint positions
            key_joints = ['head', 'hips', 'shoulder_left', 'shoulder_right']
            detected_key = [name for name in key_joints if joints.get(name, (0, 0)) != (0, 0)]
            print(f"  Key Joints Detected: {', '.join(detected_key) if detected_key else 'None'}")

    def _get_default_joints(self):
        """Return default joint dictionary"""
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

    def _print_session_summary(self, stroke_phases):
        """Print session summary"""
        print("\n" + "=" * 70)
        print("IMPROVED POSE ESTIMATION SESSION SUMMARY")
        print("=" * 70)

        if stroke_phases:
            # Phase distribution
            phase_counts = {}
            for phase in stroke_phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            print(f"\nTotal Frames Processed: {len(stroke_phases)}")
            print("\nStroke Phase Distribution:")
            total_frames = len(stroke_phases)
            for phase, count in sorted(phase_counts.items()):
                percentage = (count / total_frames) * 100
                print(f"  {phase.capitalize():<10}: {count:>4} frames ({percentage:>5.1f}%)")

            # Calculate detection success rate
            valid_frames = sum(1 for phase in stroke_phases if phase != 'unknown')
            success_rate = (valid_frames / total_frames) * 100
            print(f"\nDetection Success Rate: {success_rate:.1f}% ({valid_frames}/{total_frames})")

            # Phase transitions
            transitions = []
            for i in range(1, len(stroke_phases)):
                if stroke_phases[i] != stroke_phases[i - 1] and stroke_phases[i] != 'unknown':
                    transitions.append((stroke_phases[i - 1], stroke_phases[i]))

            if transitions:
                print(f"\nPhase Transitions Detected: {len(transitions)}")
                # Count transition types
                transition_counts = {}
                for trans in transitions:
                    key = f"{trans[0]} → {trans[1]}"
                    transition_counts[key] = transition_counts.get(key, 0) + 1

                print("Most Common Transitions:")
                for trans, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {trans:<20}: {count} times")

        print(f"\nKey Improvements Made:")
        print(f"  • Enhanced head detection with compactness analysis")
        print(f"  • Better component filtering and selection")
        print(f"  • Improved joint validation and smoothing")
        print(f"  • More robust stroke phase classification")
        print(f"  • Enhanced debug visualization")

        print(f"\nDebug Controls:")
        print(f"  'p' - Pause/unpause playback")
        print(f"  'r' - Reset background model")
        print(f"  'Esc' - Exit application")
        print("=" * 70)


if __name__ == "__main__":
    estimator = PoseEstimator(
        r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset\VID-20250609-WA0017_1 Paida.mp4")
    estimator.run()