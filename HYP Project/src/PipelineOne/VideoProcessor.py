import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from src.PipelineOne.BiomechanicalAnalyser import BiomechanicalAnalyzer
from src.PipelineOne.ClassicalPoseEstimation import ClassicalPoseEstimation


class VideoProcessor:
    """Process video entirely in memory with minimal disk access and improved error handling"""

    def __init__(self):
        self.pose_estimator = ClassicalPoseEstimation()
        self.biomech_analyzer = BiomechanicalAnalyzer()

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
        """Process a single frame and extract joints with robust error handling"""
        if frame is None or frame.size == 0:
            return []

        try:
            # --- Preprocessing ---
            gray = self.pose_estimator.grayscale(frame)

            # Blur
            blur = gray
            try:
                kernel = self.pose_estimator.gaussian_blur(3.0, 5)
                blur = self.pose_estimator.convolve_frame(gray, kernel)
            except Exception as e:
                print(f"[WARNING] Blur failed for frame {frame_num}: {e}")

            # Foreground segmentation
            try:
                fg_mask = self.pose_estimator.background_subtraction(gray)
                fg_mask = self.pose_estimator.morphological_operations(fg_mask, 'open', 3)
                fg_mask = self.pose_estimator.morphological_operations(fg_mask, 'close', 5)
            except Exception as e:
                print(f"[WARNING] Foreground extraction failed for frame {frame_num}: {e}")
                return []

            # Connected components
            try:
                labels, num_labels = self.pose_estimator.connected_components_labeling(fg_mask)
            except Exception as e:
                print(f"[WARNING] Connected components failed for frame {frame_num}: {e}")
                return []

            if num_labels == 0:
                return []

            # Keypoints
            keypoints = []
            try:
                blur_fg = np.where(fg_mask > 0, blur, 0)
                keypoints = self.pose_estimator.shi_tomasi_corner_detection(blur_fg)
            except Exception as e:
                print(f"[WARNING] Corner detection failed for frame {frame_num}: {e}")

            # --- Joint Extraction ---
            joints_data = []
            for blob_id in range(1, min(num_labels + 1, 10)):  # process up to 10 blobs
                try:
                    ys, xs = np.where(labels == blob_id)
                    if len(ys) == 0:
                        continue

                    bbox = (ys.min(), ys.max(), xs.min(), xs.max())
                    joints = self.pose_estimator.extract_anatomical_joints(bbox, keypoints, labels, blob_id)

                    if joints and blob_id in joints:
                        for joint_name, coords in joints[blob_id].items():
                            if coords is not None:
                                standard_name = self._map_joint_name(joint_name)
                                joints_data.append((frame_num, standard_name, coords[0], coords[1]))

                except Exception as e:
                    print(f"[WARNING] Joint extraction failed for blob {blob_id} in frame {frame_num}: {e}")
                    continue

            return joints_data

        except Exception as e:
            print(f"[WARNING] Unexpected error processing frame {frame_num}: {e}")
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