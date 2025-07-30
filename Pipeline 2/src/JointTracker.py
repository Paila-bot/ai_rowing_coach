import numpy as np
import csv
import os
from typing import Dict, List, Tuple


class JointTracker:
    def __init__(self, window_size=5, max_displacement=100):
        """
        Enhanced joint tracker with CSV export and outlier detection

        Args:
            window_size: Number of frames to use for smoothing
            max_displacement: Maximum allowed displacement between frames (outlier detection)
        """
        self.history = {}
        self.window = window_size
        self.max_displacement = max_displacement
        self.frame_count = 0

        # Storage for all joint data (for CSV export)
        self.joint_data = []
        self.smoothed_data = []

        # Tracking statistics
        self.tracking_stats = {
            'total_frames': 0,
            'valid_detections': {},
            'outliers_removed': {},
            'interpolated_frames': {}
        }

    def update(self, joints: Dict[str, Tuple[int, int]], frame_id: int = None) -> Dict[str, Tuple[int, int]]:
        """
        Update joint tracker with new joint positions

        Args:
            joints: Dictionary of joint positions
            frame_id: Frame identifier (optional)

        Returns:
            Dictionary of smoothed joint positions
        """
        if frame_id is None:
            frame_id = self.frame_count

        self.frame_count = frame_id
        self.tracking_stats['total_frames'] = max(self.tracking_stats['total_frames'], frame_id + 1)

        # Store raw joint data
        raw_frame_data = {'frame_id': frame_id}
        raw_frame_data.update(joints)
        self.joint_data.append(raw_frame_data)

        # Process each joint
        smoothed = {}
        for joint_name, position in joints.items():
            # Initialize history for new joints
            if joint_name not in self.history:
                self.history[joint_name] = []
                self.tracking_stats['valid_detections'][joint_name] = 0
                self.tracking_stats['outliers_removed'][joint_name] = 0
                self.tracking_stats['interpolated_frames'][joint_name] = 0

            # Outlier detection and correction
            corrected_position = self._detect_and_correct_outlier(joint_name, position)

            # Add to history
            self.history[joint_name].append(corrected_position)

            # Maintain window size
            if len(self.history[joint_name]) > self.window:
                self.history[joint_name].pop(0)

            # Calculate smoothed position
            valid_positions = [pos for pos in self.history[joint_name] if pos != (0, 0)]

            if valid_positions:
                # Use temporal smoothing with valid positions
                smoothed[joint_name] = self._temporal_smooth(valid_positions)
                self.tracking_stats['valid_detections'][joint_name] += 1
            else:
                # No valid positions, try interpolation
                smoothed[joint_name] = self._interpolate_missing_joint(joint_name, frame_id)
                if smoothed[joint_name] != (0, 0):
                    self.tracking_stats['interpolated_frames'][joint_name] += 1

        # Store smoothed data
        smoothed_frame_data = {'frame_id': frame_id}
        smoothed_frame_data.update(smoothed)
        self.smoothed_data.append(smoothed_frame_data)

        return smoothed

    def _detect_and_correct_outlier(self, joint_name: str, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Detect and correct outlier joint positions

        Args:
            joint_name: Name of the joint
            position: Current joint position

        Returns:
            Corrected position
        """
        if position == (0, 0) or len(self.history[joint_name]) < 2:
            return position

        # Get recent valid positions
        recent_positions = [pos for pos in self.history[joint_name][-3:] if pos != (0, 0)]

        if not recent_positions:
            return position

        # Calculate average recent position
        avg_y = np.mean([pos[0] for pos in recent_positions])
        avg_x = np.mean([pos[1] for pos in recent_positions])
        avg_position = (avg_y, avg_x)

        # Calculate displacement
        displacement = np.sqrt((position[0] - avg_position[0]) ** 2 +
                               (position[1] - avg_position[1]) ** 2)

        if displacement > self.max_displacement:
            # This is likely an outlier
            self.tracking_stats['outliers_removed'][joint_name] += 1

            # Return interpolated position instead
            if len(recent_positions) >= 2:
                # Use linear prediction
                vel_y = recent_positions[-1][0] - recent_positions[-2][0]
                vel_x = recent_positions[-1][1] - recent_positions[-2][1]

                predicted_y = recent_positions[-1][0] + vel_y
                predicted_x = recent_positions[-1][1] + vel_x

                return (int(predicted_y), int(predicted_x))
            else:
                return recent_positions[-1]

        return position

    def _temporal_smooth(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Apply temporal smoothing to joint positions

        Args:
            positions: List of recent valid positions

        Returns:
            Smoothed position
        """
        if len(positions) == 1:
            return positions[0]

        # Use weighted average with more weight on recent positions
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights = weights / np.sum(weights)

        ys = [pos[0] for pos in positions]
        xs = [pos[1] for pos in positions]

        smoothed_y = int(np.sum(np.array(ys) * weights))
        smoothed_x = int(np.sum(np.array(xs) * weights))

        return (smoothed_y, smoothed_x)

    def _interpolate_missing_joint(self, joint_name: str, frame_id: int) -> Tuple[int, int]:
        """
        Interpolate missing joint position based on recent history

        Args:
            joint_name: Name of the joint
            frame_id: Current frame ID

        Returns:
            Interpolated position or (0, 0) if interpolation not possible
        """
        if len(self.history[joint_name]) < 2:
            return (0, 0)

        # Find last two valid positions
        valid_positions = [(i, pos) for i, pos in enumerate(self.history[joint_name])
                           if pos != (0, 0)]

        if len(valid_positions) < 2:
            return (0, 0)

        # Use linear interpolation
        (idx1, pos1), (idx2, pos2) = valid_positions[-2:]

        if idx1 == idx2:
            return pos2

        # Linear interpolation
        t = (len(self.history[joint_name]) - 1 - idx2) / max(1, idx2 - idx1)

        interp_y = int(pos2[0] + t * (pos2[0] - pos1[0]))
        interp_x = int(pos2[1] + t * (pos2[1] - pos1[1]))

        return (interp_y, interp_x)

    def export_to_csv(self, output_path: str, video_name: str = "unknown",
                      export_smoothed: bool = True):
        """
        Export joint tracking data to CSV

        Args:
            output_path: Path to save CSV file
            video_name: Video identifier
            export_smoothed: Whether to export smoothed or raw data
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data_to_export = self.smoothed_data if export_smoothed else self.joint_data

        if not data_to_export:
            print(f"[Warning] No data to export for {video_name}")
            return

        # Get all joint names from the data
        all_joints = set()
        for frame_data in data_to_export:
            all_joints.update(key for key in frame_data.keys() if key != 'frame_id')

        # Create CSV fieldnames
        fieldnames = ['video_name', 'frame_id']
        for joint in sorted(all_joints):
            fieldnames.extend([f'{joint}_y', f'{joint}_x'])

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for frame_data in data_to_export:
                row = {
                    'video_name': video_name,
                    'frame_id': frame_data['frame_id']
                }

                # Add joint position data
                for joint in all_joints:
                    if joint in frame_data and frame_data[joint] != (0, 0):
                        row[f'{joint}_y'] = frame_data[joint][0]
                        row[f'{joint}_x'] = frame_data[joint][1]
                    else:
                        row[f'{joint}_y'] = 0
                        row[f'{joint}_x'] = 0

                writer.writerow(row)

        print(f"[Info] Exported {len(data_to_export)} frames of joint data to {output_path}")

    def get_tracking_quality_report(self) -> Dict:
        """
        Generate a quality report for joint tracking

        Returns:
            Dictionary containing tracking quality metrics
        """
        report = {
            'total_frames': self.tracking_stats['total_frames'],
            'joint_quality': {}
        }

        for joint_name in self.tracking_stats['valid_detections']:
            valid_detections = self.tracking_stats['valid_detections'][joint_name]
            outliers_removed = self.tracking_stats['outliers_removed'][joint_name]
            interpolated = self.tracking_stats['interpolated_frames'][joint_name]

            total_frames = self.tracking_stats['total_frames']

            if total_frames > 0:
                quality_score = valid_detections / total_frames
                outlier_rate = outliers_removed / max(1, valid_detections + outliers_removed)
                interpolation_rate = interpolated / total_frames

                report['joint_quality'][joint_name] = {
                    'detection_rate': quality_score,
                    'outlier_rate': outlier_rate,
                    'interpolation_rate': interpolation_rate,
                    'valid_detections': valid_detections,
                    'outliers_removed': outliers_removed,
                    'interpolated_frames': interpolated
                }

        return report

    def calculate_joint_velocities(self) -> Dict[str, List[float]]:
        """
        Calculate joint velocities over time

        Returns:
            Dictionary of joint velocities for each frame
        """
        velocities = {}

        for joint_name in self.history:
            velocities[joint_name] = []

            if len(self.smoothed_data) < 2:
                continue

            for i in range(1, len(self.smoothed_data)):
                prev_frame = self.smoothed_data[i - 1]
                curr_frame = self.smoothed_data[i]

                if (joint_name in prev_frame and joint_name in curr_frame and
                        prev_frame[joint_name] != (0, 0) and curr_frame[joint_name] != (0, 0)):

                    prev_pos = prev_frame[joint_name]
                    curr_pos = curr_frame[joint_name]

                    # Calculate Euclidean velocity
                    velocity = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 +
                                       (curr_pos[1] - prev_pos[1]) ** 2)
                    velocities[joint_name].append(velocity)
                else:
                    velocities[joint_name].append(0.0)

        return velocities

    def reset(self):
        """Reset tracker state for new video"""
        self.history = {}
        self.frame_count = 0
        self.joint_data = []
        self.smoothed_data = []
        self.tracking_stats = {
            'total_frames': 0,
            'valid_detections': {},
            'outliers_removed': {},
            'interpolated_frames': {}
        }