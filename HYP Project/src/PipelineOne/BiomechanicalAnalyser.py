import csv
from collections import defaultdict

import numpy as np

class BiomechanicalAnalyzer:
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