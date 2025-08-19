import os
import joblib
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class PipelineThreeAnalyzer:
    """Complete working analyzer that implements all functionality directly"""

    def __init__(self, video_path: str, model_dir: str = None):
        self.video_path = video_path

        if model_dir is None:
            self.model_dir = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 1\models\Pipeline 3 models"
        else:
            self.model_dir = model_dir

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            smooth_landmarks=True
        )

        # Joint mapping for rowing analysis
        self.joint_mapping = {
            'nose': mp.solutions.pose.PoseLandmark.NOSE,
            'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
            'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            'left_knee': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            'right_knee': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        }

    def extract_joints(self, results):
        """Extract joint positions from MediaPipe results"""
        try:
            landmarks = results.pose_landmarks.landmark
            joints = {}

            for joint_name, joint_landmark in self.joint_mapping.items():
                landmark = landmarks[joint_landmark]
                joints[joint_name] = (landmark.x, landmark.y, landmark.z)

            return joints
        except Exception as e:
            return None

    def process_video(self):
        """Process video and extract pose data"""
        if not os.path.exists(self.video_path):
            return []

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return []

        joint_data = []
        frame_count = 0
        processed_frames = 0
        successful_detections = 0

        while cap.isOpened() and processed_frames < 500:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:  # Skip frames for efficiency
                processed_frames += 1

                # Resize if too large
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = 1280
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    joints = self.extract_joints(results)
                    if joints:
                        joint_data.append(joints)
                        successful_detections += 1

            frame_count += 1

        cap.release()
        return joint_data

    def calculate_features(self, pose_sequence):
        """Calculate rowing features from pose data"""
        features = []
        prev_left_wrist = None
        prev_right_wrist = None

        for frame in pose_sequence:
            frame_features = [0.0] * 6  # 6 features as expected

            try:
                # Feature 0: Left arm angle
                if all(joint in frame for joint in ['left_shoulder', 'left_elbow', 'left_wrist']):
                    angle = self._get_angle(
                        frame['left_shoulder'][:3],
                        frame['left_elbow'][:3],
                        frame['left_wrist'][:3]
                    )
                    if not (np.isnan(angle) or np.isinf(angle)):
                        frame_features[0] = angle

                # Feature 1: Right arm angle
                if all(joint in frame for joint in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    angle = self._get_angle(
                        frame['right_shoulder'][:3],
                        frame['right_elbow'][:3],
                        frame['right_wrist'][:3]
                    )
                    if not (np.isnan(angle) or np.isinf(angle)):
                        frame_features[1] = angle

                # Feature 2: Torso angle
                if all(joint in frame for joint in ['left_shoulder', 'nose', 'right_shoulder']):
                    angle = self._get_angle(
                        frame['left_shoulder'][:3],
                        frame['nose'][:3],
                        frame['right_shoulder'][:3]
                    )
                    if not (np.isnan(angle) or np.isinf(angle)):
                        frame_features[2] = angle

                # Feature 3: Body lean
                if all(joint in frame for joint in ['left_hip', 'left_shoulder']):
                    hip_pos = np.array(frame['left_hip'][:3])
                    shoulder_pos = np.array(frame['left_shoulder'][:3])
                    vertical = np.array([0, -1, 0])
                    lean_vector = shoulder_pos - hip_pos

                    cos_angle = np.dot(lean_vector, vertical) / (np.linalg.norm(lean_vector) * np.linalg.norm(vertical))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))

                    if not (np.isnan(angle) or np.isinf(angle)):
                        frame_features[3] = angle

                # Feature 4: Left wrist velocity
                if 'left_wrist' in frame:
                    current_pos = np.array(frame['left_wrist'][:3])
                    if prev_left_wrist is not None:
                        velocity = np.linalg.norm(current_pos - prev_left_wrist)
                        if not (np.isnan(velocity) or np.isinf(velocity)):
                            frame_features[4] = velocity
                    prev_left_wrist = current_pos

                # Feature 5: Right wrist velocity
                if 'right_wrist' in frame:
                    current_pos = np.array(frame['right_wrist'][:3])
                    if prev_right_wrist is not None:
                        velocity = np.linalg.norm(current_pos - prev_right_wrist)
                        if not (np.isnan(velocity) or np.isinf(velocity)):
                            frame_features[5] = velocity
                    prev_right_wrist = current_pos

            except Exception as e:
                print(f"Feature calculation error: {e}")

            features.append(frame_features)

        return np.array(features)

    def _get_angle(self, a, b, c):
        """Calculate angle between three points"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return 90.0

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))

    def generate_feedback(self, analysis_results):
        """Generate human-like coaching feedback"""
        total_frames = len(analysis_results['phases'])
        critical_count = len(analysis_results['critical_frames'])
        error_rate = (critical_count / total_frames) * 100 if total_frames > 0 else 0

        # Calculate score
        score = max(0, int(100 - (error_rate * 2)))

        # Generate summary
        if error_rate < 10:
            summary = f"Excellent technique! Only {critical_count} minor issues detected across {total_frames} frames. Your rowing form is very consistent."
        elif error_rate < 25:
            summary = f"Good rowing with room for improvement. {critical_count} technical issues found ({error_rate:.1f}% of strokes). Focus on consistency."
        else:
            summary = f"Significant technique issues detected. {critical_count} problems found ({error_rate:.1f}% of strokes). Let's work on the fundamentals."

        # Identify most problematic phase
        phases = analysis_results['phases']
        phase_names = {0: 'Catch', 1: 'Drive', 2: 'Finish', 3: 'Recovery'}

        # Group critical frames by phase
        phase_issues = {}
        for frame_idx in analysis_results['critical_frames']:
            phase = phases[frame_idx]
            phase_name = phase_names.get(phase, 'Unknown')
            if phase_name not in phase_issues:
                phase_issues[phase_name] = []
            phase_issues[phase_name].append(frame_idx)

        # Generate phase-specific feedback
        technical_analysis = []
        for phase_name, problematic_frames in phase_issues.items():
            issue_desc = self._get_phase_issue_description(phase_name)
            technical_analysis.append({
                'phase': phase_name,
                'issue': issue_desc,
                'frames': problematic_frames[:3],
                'ideal_technique': self._get_ideal_technique(phase_name)
            })

        # Generate recommendations
        recommendations = self._generate_recommendations(error_rate, phase_issues)

        return {
            'summary': summary,
            'technical_analysis': technical_analysis,
            'recommendations': recommendations,
            'score': score
        }

    def _get_phase_issue_description(self, phase_name):
        """Get likely issue description for each phase"""
        issues = {
            'Catch': "Body position and timing at catch entry",
            'Drive': "Power transfer and sequencing during drive phase",
            'Finish': "Body position and handle height at stroke completion",
            'Recovery': "Movement timing and preparation for next stroke"
        }
        return issues.get(phase_name, "Technique consistency issue")

    def _get_ideal_technique(self, phase_name):
        """Get ideal technique points for each phase"""
        techniques = {
            'Catch': [
                "Arms straight but relaxed",
                "Shins vertical, body forward",
                "Blade entry clean and controlled"
            ],
            'Drive': [
                "Legs drive first, then back",
                "Arms stay straight initially",
                "Smooth power transfer"
            ],
            'Finish': [
                "Handle to lower ribs",
                "Slight lean back (11 o'clock)",
                "Clean blade extraction"
            ],
            'Recovery': [
                "Hands away first",
                "Body swing controlled",
                "Slide speed consistent"
            ]
        }
        return techniques.get(phase_name, ["Focus on smooth, controlled movement"])

    def _generate_recommendations(self, error_rate, phase_issues):
        """Generate specific drill recommendations"""
        recommendations = []

        if error_rate < 10:
            recommendations.append("Continue refining technique with video analysis")
            recommendations.append("Focus on racing pace consistency")
        elif error_rate < 25:
            recommendations.append("Practice stroke sequencing with pause drills")
            recommendations.append("Use mirror for visual feedback")
        else:
            recommendations.append("Break down stroke into individual components")
            recommendations.append("Start with basic catch and drive drills")

        # Phase-specific recommendations
        if 'Catch' in phase_issues:
            recommendations.append("Practice catch position holds")
        if 'Drive' in phase_issues:
            recommendations.append("Focus on leg drive isolation drills")
        if 'Finish' in phase_issues:
            recommendations.append("Work on finish position stability")
        if 'Recovery' in phase_issues:
            recommendations.append("Practice recovery timing with metronome")

        return recommendations[:5]  # Limit to top 5

    def run(self) -> str:
        """Run complete analysis pipeline"""
        try:
            # Load models
            autoencoder_path = os.path.join(self.model_dir, 'autoencoder.keras')
            sklearn_models_path = os.path.join(self.model_dir, 'scikit_models.pkl')

            if not os.path.exists(autoencoder_path) or not os.path.exists(sklearn_models_path):
                return "❌ Model files not found. Please ensure models are trained and saved."

            autoencoder = load_model(autoencoder_path)
            sklearn_models = joblib.load(sklearn_models_path)

            scaler = sklearn_models['scaler']
            kmeans = sklearn_models['phase_detector_kmeans']
            sequence_length = sklearn_models['sequence_length']

            # Process video
            print("🎥 Processing video...")
            pose_data = self.process_video()

            if not pose_data:
                return "❌ No valid poses detected in the video. Please ensure the video shows a clear side view of the rower."

            print(f"✅ Extracted {len(pose_data)} pose frames")

            # Calculate features
            print("🔢 Calculating features...")
            features = self.calculate_features(pose_data)

            if len(features) == 0:
                return "❌ No valid features could be extracted from the poses."

            print(f"✅ Calculated features for {len(features)} frames")

            # Predict phases
            phases = kmeans.predict(features)

            # Prepare for autoencoder
            features_scaled = scaler.transform(features)

            # Pad sequence if needed
            if len(features_scaled) < sequence_length:
                padding = np.repeat([features_scaled[-1]], sequence_length - len(features_scaled), axis=0)
                padded_features = np.vstack([features_scaled, padding])
            else:
                padded_features = features_scaled[:sequence_length]

            # Reshape for autoencoder
            input_features = padded_features.reshape(1, sequence_length, 6)

            # Get anomaly scores
            reconstructed = autoencoder.predict(input_features, verbose=0)[0]
            actual_length = len(features_scaled)

            errors = np.mean(np.square(
                features_scaled - reconstructed[:actual_length]
            ), axis=1)

            # Analysis results
            results = {
                'phases': phases,
                'anomaly_scores': errors,
                'critical_frames': np.where(errors > np.percentile(errors, 90))[0]
            }

            # Generate feedback
            print("📝 Generating feedback...")
            report = self.generate_feedback(results)

            # Format output
            output = []
            output.append("🏆 **Expert Rowing Analysis Report** 🏆\n")
            output.append(f"📊 **Summary (Score: {report['score']}/100):**")
            output.append(f"{report['summary']}\n")

            output.append("🔍 **Technical Analysis:**")
            for analysis in report['technical_analysis']:
                output.append(f"\n**{analysis['phase'].upper()} PHASE:**")
                output.append(f"- Main issue: {analysis['issue']}")
                output.append(f"- Problematic frames: {', '.join(map(str, analysis['frames']))}")
                output.append("- Ideal technique:")
                for point in analysis['ideal_technique']:
                    output.append(f"  • {point}")

            output.append("\n💡 **Recommended Drills:**")
            for i, rec in enumerate(report['recommendations'], 1):
                output.append(f"{i}. {rec}")

            output.append(f"\n📈 **Technical Summary:**")
            output.append(f"- Analyzed {len(features)} frames of rowing video")
            output.append(f"- Detected {len(results['critical_frames'])} technique issues")
            output.append(f"- Average error score: {np.mean(errors):.2f}")
            output.append(f"- Most problematic frame: {np.argmax(errors)} (error: {np.max(errors):.2f})")

            return "\n".join(output)

        except Exception as e:
            return f"❌ Analysis failed: {str(e)}\n\nPlease check:\n- Video file is valid\n- All model files are present\n- Video shows clear side view of rower"