import cv2
import numpy as np
import os
import json
from collections import defaultdict


class AutomatedJointTracker:
    """Fully automated joint tracking using only OpenCV and numpy"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []
        self.joint_positions = defaultdict(dict)
        self.joint_labels = ["head", "shoulder", "hip", "knee", "handle"]

        # Color detection parameters for different body parts
        self.color_ranges = {
            "skin": {
                "lower": np.array([0, 20, 70]),
                "upper": np.array([20, 255, 255])
            },
            "clothing": {
                "lower": np.array([100, 50, 50]),
                "upper": np.array([130, 255, 255])
            }
        }

        # Template matching parameters
        self.template_size = (30, 30)
        self.search_region = 50

    def load_video(self):
        """Load video frames into memory"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

        print(f"Loaded {len(self.frames)} frames")

    def detect_person_silhouette(self, frame):
        """Detect person silhouette using background subtraction and contours"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold to separate person from background
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (person)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Minimum size filter
                return largest_contour

        return None

    def extract_skeleton_points(self, contour, frame):
        """Extract key skeleton points from person contour"""
        if contour is None:
            return {}

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate approximate joint positions based on human proportions
        joints = {}

        # Head (top of contour)
        head_y = y + int(h * 0.1)  # 10% from top
        head_x = x + w // 2
        joints['head'] = (head_x, head_y)

        # Shoulder (approximately 20% from top)
        shoulder_y = y + int(h * 0.2)
        shoulder_x = x + w // 2
        joints['shoulder'] = (shoulder_x, shoulder_y)

        # Hip (approximately 50% from top)
        hip_y = y + int(h * 0.5)
        hip_x = x + w // 2
        joints['hip'] = (hip_x, hip_y)

        # Knee (approximately 75% from top)
        knee_y = y + int(h * 0.75)
        knee_x = x + w // 2
        joints['knee'] = (knee_x, knee_y)

        # Handle detection using edge detection
        handle_pos = self.detect_handle(frame, contour)
        if handle_pos:
            joints['handle'] = handle_pos
        else:
            # Fallback: estimate handle position relative to shoulder
            joints['handle'] = (shoulder_x + int(w * 0.3), shoulder_y)

        return joints

    def detect_handle(self, frame, person_contour):
        """Detect rowing handle using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)

        if lines is not None:
            # Find the most horizontal line (likely the handle)
            best_line = None
            min_angle = float('inf')

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Look for nearly horizontal lines
                if angle < 30 or angle > 150:
                    if angle < min_angle:
                        min_angle = angle
                        best_line = line[0]

            if best_line is not None:
                x1, y1, x2, y2 = best_line
                # Return center of the handle
                return ((x1 + x2) // 2, (y1 + y2) // 2)

        return None

    def refine_joint_positions(self, joints, frame):
        """Refine joint positions using local feature matching"""
        refined_joints = {}

        for joint_name, (x, y) in joints.items():
            # Extract small region around estimated position
            roi_size = 20
            x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
            x2, y2 = min(frame.shape[1], x + roi_size), min(frame.shape[0], y + roi_size)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                # Find the most distinct point in the region
                refined_pos = self.find_feature_point(roi, joint_name)
                if refined_pos:
                    refined_x = x1 + refined_pos[0]
                    refined_y = y1 + refined_pos[1]
                    refined_joints[joint_name] = (refined_x, refined_y)
                else:
                    refined_joints[joint_name] = (x, y)
            else:
                refined_joints[joint_name] = (x, y)

        return refined_joints

    def find_feature_point(self, roi, joint_type):
        """Find the most distinctive point in a region"""
        if roi.size == 0:
            return None

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Different strategies for different joint types
        if joint_type == "head":
            # Look for circular features (head outline)
            circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=30, minRadius=5, maxRadius=15)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    return (circles[0][0], circles[0][1])

        elif joint_type == "handle":
            # Look for horizontal edges
            edges = cv2.Canny(gray_roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                    minLineLength=10, maxLineGap=5)
            if lines is not None:
                # Return center of the most horizontal line
                best_line = None
                min_angle = float('inf')
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < min_angle:
                        min_angle = angle
                        best_line = line[0]

                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    return ((x1 + x2) // 2, (y1 + y2) // 2)

        else:
            # For other joints, look for corner features
            corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=1,
                                              qualityLevel=0.01, minDistance=10)
            if corners is not None and len(corners) > 0:
                return (int(corners[0][0][0]), int(corners[0][0][1]))

        # Fallback: return center of ROI
        return (gray_roi.shape[1] // 2, gray_roi.shape[0] // 2)

    def track_between_frames(self, frame1, frame2, joints1):
        """Track joints between consecutive frames using optical flow"""
        if not joints1:
            return {}

        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Prepare points for optical flow
        points1 = np.array([[x, y] for x, y in joints1.values()], dtype=np.float32)
        joint_names = list(joints1.keys())

        # Calculate optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        points2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)

        # Build tracked joints dictionary
        tracked_joints = {}
        for i, (joint_name, (x, y)) in enumerate(zip(joint_names, points2)):
            if status[i] and error[i] < 50:  # Good tracking
                tracked_joints[joint_name] = (int(x), int(y))
            else:
                # Fallback to previous position
                tracked_joints[joint_name] = joints1[joint_name]

        return tracked_joints

    def process_all_frames(self):
        """Process all frames automatically"""
        print("Processing frames automatically...")

        for frame_idx, frame in enumerate(self.frames):
            if frame_idx % 10 == 0:  # Progress update
                print(f"Processing frame {frame_idx}/{len(self.frames)}")

            if frame_idx == 0:
                # First frame: detect from scratch
                contour = self.detect_person_silhouette(frame)
                joints = self.extract_skeleton_points(contour, frame)
                joints = self.refine_joint_positions(joints, frame)
                self.joint_positions[frame_idx] = joints
            else:
                # Subsequent frames: track from previous frame
                prev_joints = self.joint_positions[frame_idx - 1]
                tracked_joints = self.track_between_frames(
                    self.frames[frame_idx - 1], frame, prev_joints)

                # Optionally refine tracked positions
                if frame_idx % 5 == 0:  # Refine every 5th frame
                    tracked_joints = self.refine_joint_positions(tracked_joints, frame)

                self.joint_positions[frame_idx] = tracked_joints

        print("✅ Automatic tracking complete!")

    def validate_and_smooth_tracking(self):
        """Validate tracking results and smooth trajectories"""
        print("Validating and smoothing tracking...")

        # Smooth trajectories using moving average
        window_size = 5

        for joint_name in self.joint_labels:
            x_coords = []
            y_coords = []
            frame_indices = []

            # Collect all coordinates for this joint
            for frame_idx in sorted(self.joint_positions.keys()):
                if joint_name in self.joint_positions[frame_idx]:
                    x, y = self.joint_positions[frame_idx][joint_name]
                    x_coords.append(x)
                    y_coords.append(y)
                    frame_indices.append(frame_idx)

            # Apply moving average smoothing
            if len(x_coords) >= window_size:
                smoothed_x = self.moving_average(x_coords, window_size)
                smoothed_y = self.moving_average(y_coords, window_size)

                # Update positions with smoothed values
                for i, frame_idx in enumerate(frame_indices):
                    if i < len(smoothed_x):
                        self.joint_positions[frame_idx][joint_name] = (
                            int(smoothed_x[i]), int(smoothed_y[i]))

    def moving_average(self, data, window_size):
        """Apply moving average smoothing"""
        if len(data) < window_size:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            avg = np.mean(data[start_idx:end_idx])
            smoothed.append(avg)

        return smoothed

    def save_tracking_results(self):
        """Save tracking results to files"""
        # Save all joint positions
        with open('automatic_joint_positions.csv', 'w') as f:
            f.write('frame,joint_name,x,y\n')
            for frame_idx in sorted(self.joint_positions.keys()):
                for joint_name, (x, y) in self.joint_positions[frame_idx].items():
                    f.write(f'{frame_idx},{joint_name},{x},{y}\n')

        # Save as JSON for easy loading
        json_data = {}
        for frame_idx in self.joint_positions:
            json_data[str(frame_idx)] = self.joint_positions[frame_idx]

        with open('automatic_joint_positions.json', 'w') as f:
            json.dump(json_data, f, indent=2)

        print("✅ Saved automatic_joint_positions.csv and automatic_joint_positions.json")

    def visualize_tracking(self, output_path='tracking_visualization.mp4'):
        """Create a video showing the tracked joints"""
        if not self.frames:
            print("No frames loaded")
            return

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        colors = {
            "head": (0, 255, 255),  # Yellow
            "shoulder": (255, 0, 0),  # Blue
            "hip": (0, 255, 0),  # Green
            "knee": (255, 0, 255),  # Magenta
            "handle": (0, 165, 255)  # Orange
        }

        skeleton_connections = [
            ("head", "shoulder"),
            ("shoulder", "hip"),
            ("hip", "knee"),
            ("shoulder", "handle")
        ]

        for frame_idx, frame in enumerate(self.frames):
            vis_frame = frame.copy()

            # Draw joints
            if frame_idx in self.joint_positions:
                joints = self.joint_positions[frame_idx]

                # Draw skeleton connections
                for joint1, joint2 in skeleton_connections:
                    if joint1 in joints and joint2 in joints:
                        pt1 = joints[joint1]
                        pt2 = joints[joint2]
                        cv2.line(vis_frame, pt1, pt2, (255, 255, 255), 2)

                # Draw joint points
                for joint_name, (x, y) in joints.items():
                    color = colors.get(joint_name, (255, 255, 255))
                    cv2.circle(vis_frame, (x, y), 6, color, -1)
                    cv2.putText(vis_frame, joint_name, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add frame number
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(vis_frame)

        out.release()
        print(f"✅ Saved tracking visualization: {output_path}")

    def run(self):
        """Run the complete automated tracking pipeline"""
        print("=== AUTOMATED JOINT TRACKING SYSTEM ===")
        print("Using only OpenCV and NumPy - no external ML libraries!")

        # Load video
        self.load_video()

        # Process all frames automatically
        self.process_all_frames()

        # Validate and smooth results
        self.validate_and_smooth_tracking()

        # Save results
        self.save_tracking_results()

        # Create visualization
        self.visualize_tracking()

        print("\n✅ Automated tracking complete!")
        print("Generated files:")
        print("  - automatic_joint_positions.csv")
        print("  - automatic_joint_positions.json")
        print("  - tracking_visualization.mp4")

        return self.joint_positions

