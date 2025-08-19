import cv2
import mediapipe as mp
import os
import joblib

class PoseDetector:
    def __init__(self, model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )

        # Rowing-specific joint mapping (joints typically visible during rowing)
        self.joint_mapping = {
            'nose': mp.solutions.pose.PoseLandmark.NOSE,
            'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            # Try to get these but don't require them
            'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
            'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            'left_knee': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            'right_knee': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        }

        # Minimum required joints for rowing analysis
        self.required_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']

    def extract_joints(self, results):
        """Extract joints with flexible missing joint handling"""
        try:
            landmarks = results.pose_landmarks.landmark
            joints = {}

            for joint_name, joint_landmark in self.joint_mapping.items():
                landmark = landmarks[joint_landmark]

                # Store all joints, even with low visibility
                joints[joint_name] = (landmark.x, landmark.y, landmark.z)

            return joints

        except Exception as e:
            print(f"Error extracting joints: {str(e)}")
            return None

    def process_video(self, video_path, skip_frames=5, max_frames=500, debug=True):
        """Process video with rowing-specific validation"""
        if debug:
            print(f"Processing: {os.path.basename(video_path)}")

        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {os.path.basename(video_path)}")
            return []

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if debug:
            print(f"📹 Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")

        frame_count = 0
        processed_frames = 0
        successful_detections = 0
        joint_data = []

        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % skip_frames == 0:
                processed_frames += 1

                if frame is None or frame.size == 0:
                    frame_count += 1
                    continue

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
                        # Check if we have minimum required joints for rowing
                        required_found = sum(1 for joint in self.required_joints if joint in joints)
                        if required_found >= 3:  # Need at least 3 out of 4 key joints
                            joint_data.append(joints)
                            successful_detections += 1

                if debug and processed_frames % 50 == 0:
                    detection_rate = (successful_detections / processed_frames) * 100
                    print(f"📊 Processed {processed_frames} frames, {successful_detections} detections ({detection_rate:.1f}%)")

            frame_count += 1

        cap.release()

        detection_rate = (successful_detections / max(processed_frames, 1)) * 100
        if debug:
            print(f"✅ Final: {successful_detections}/{processed_frames} frames ({detection_rate:.1f}% success)")

        return joint_data