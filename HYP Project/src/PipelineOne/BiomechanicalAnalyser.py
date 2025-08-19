import csv
import time
import numpy as np
from collections import defaultdict


class BiomechanicalAnalyser:
    def __init__(self):
        self.all_frames = []
        self.stroke_phases = []
        self.current_stroke = []

        # Phase detection thresholds
        self.catch_knee_threshold = 120  # degrees
        self.drive_knee_threshold = 140  # degrees
        self.finish_knee_threshold = 160  # degrees
        self.back_angle_threshold = 45  # degrees

    def calculate_angle(self, a, b, c):
        """Calculate angle at point b formed by points a-b-c"""
        if None in [a, b, c]:
            return None
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_frame(self, joints_2d):
        """Analyze a single frame and return angles"""
        angles = {}

        # Calculate back angle (torso relative to vertical)
        if joints_2d.get('head') and joints_2d.get('hip'):
            head = joints_2d['head']
            hip = joints_2d['hip']
            vertical_ref = (hip[0], hip[1] - 50)  # Point above hip
            angles['back'] = self.calculate_angle(head, hip, vertical_ref)

        # Calculate knee angles
        if all(joints_2d.get(j) for j in ['hip_left', 'knee_left', 'ankle_left']):
            angles['left_knee'] = self.calculate_angle(
                joints_2d['hip_left'], joints_2d['knee_left'], joints_2d['ankle_left']
            )

        if all(joints_2d.get(j) for j in ['hip_right', 'knee_right', 'ankle_right']):
            angles['right_knee'] = self.calculate_angle(
                joints_2d['hip_right'], joints_2d['knee_right'], joints_2d['ankle_right']
            )

        # Calculate elbow angles
        if all(joints_2d.get(j) for j in ['shoulder_left', 'elbow_left', 'wrist_left']):
            angles['left_elbow'] = self.calculate_angle(
                joints_2d['shoulder_left'], joints_2d['elbow_left'], joints_2d['wrist_left']
            )

        if all(joints_2d.get(j) for j in ['shoulder_right', 'elbow_right', 'wrist_right']):
            angles['right_elbow'] = self.calculate_angle(
                joints_2d['shoulder_right'], joints_2d['elbow_right'], joints_2d['wrist_right']
            )

        return angles

    def detect_phase(self, angles):
        """Detect rowing phase based on joint angles"""
        back_angle = angles.get('back')
        left_knee = angles.get('left_knee')
        right_knee = angles.get('right_knee')

        # Use average knee angle if both available
        knee_angles = [a for a in [left_knee, right_knee] if a is not None]
        avg_knee = np.mean(knee_angles) if knee_angles else None

        if avg_knee is None:
            return "Unknown"

        # Phase detection logic based on biomechanics
        if avg_knee <= self.catch_knee_threshold:
            # Very compressed legs = Catch
            if back_angle and back_angle <= self.back_angle_threshold:
                return "Catch"
            else:
                return "Recovery"  # Legs compressed but body not forward

        elif avg_knee <= self.drive_knee_threshold:
            # Legs extending = Drive
            return "Drive"

        elif avg_knee >= self.finish_knee_threshold:
            # Legs extended = Finish or Recovery
            if back_angle and back_angle > self.back_angle_threshold + 10:
                return "Finish"  # Body leaning back
            else:
                return "Recovery"  # Body coming forward

        else:
            # Mid-range angles
            return "Drive"

    def segment_into_strokes(self, all_frame_data):
        """Segment continuous frame data into individual strokes"""
        strokes = []
        current_stroke = []
        last_phase = None

        for frame_num, angles in all_frame_data:
            phase = self.detect_phase(angles)

            # Detect stroke boundaries (when we go from Finish/Recovery back to Catch)
            if (last_phase in ["Finish", "Recovery"] and phase == "Catch" and
                    len(current_stroke) > 10):  # Minimum stroke length

                if current_stroke:
                    strokes.append(current_stroke)
                current_stroke = []

            current_stroke.append({
                'frame': frame_num,
                'phase': phase,
                'angles': angles
            })
            last_phase = phase

        # Add final stroke
        if current_stroke:
            strokes.append(current_stroke)

        return strokes

    def analyze_stroke_phases(self, stroke_data):
        """Analyze each phase within a stroke"""
        phase_data = defaultdict(list)

        # Group frames by phase
        for frame_data in stroke_data:
            phase = frame_data['phase']
            phase_data[phase].append(frame_data)

        phase_analysis = {}

        for phase, frames in phase_data.items():
            if not frames:
                continue

            # Collect all angles for this phase
            all_angles = defaultdict(list)
            for frame in frames:
                for angle_name, angle_value in frame['angles'].items():
                    if angle_value is not None:
                        all_angles[angle_name].append(angle_value)

            # Calculate phase statistics
            phase_stats = {}
            for angle_name, values in all_angles.items():
                if values:
                    phase_stats[angle_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values)
                    }

            phase_analysis[phase] = {
                'frame_count': len(frames),
                'angle_stats': phase_stats
            }

        return phase_analysis

    def load_joints_from_csv(self, csv_path):
        """Load joints from CSV file"""
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

        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f, skipinitialspace=True)
            try:
                first_row = next(reader)
            except StopIteration:
                return

            # Check if first row is header
            is_header = first_row[0].strip().lower() == 'frame'
            if not is_header:
                # Put first row back
                rows = [first_row] + list(reader)
            else:
                rows = list(reader)

            # Group by frame
            frames_data = defaultdict(dict)

            for row in rows:
                if not row or len(row) < 4:
                    continue

                # Handle both 4-column and 5-column formats
                if len(row) >= 5:
                    frame_str, _, joint_str, x_str, y_str = row[:5]
                else:
                    frame_str, joint_str, x_str, y_str = row[:4]

                try:
                    frame = int(frame_str)
                    x, y = float(x_str), float(y_str)
                except ValueError:
                    continue

                joint_name = normalize_joint_name(joint_str)
                if joint_name in joint_map:
                    mapped_name = joint_map[joint_name]
                    frames_data[frame][mapped_name] = (x, y)

            # Process frames and add computed hip center
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