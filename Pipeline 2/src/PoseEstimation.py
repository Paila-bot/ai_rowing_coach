import os
import numpy as np
import cv2
from PIL import Image
import FrameExtractor as fc
from src.BackgroundSubtractor import BackgroundSubtractor as bs
import StrokeDetector as sd
import JointTracker as jt
import HOGFeatureExtractor as hfe
from src.HOGFeatureExtractor import HOGFeatureExtractor


class PoseEstimator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.stroke_phase_history = []
        self.joint_consistency_tracker = {}

    def run(self):
        # Step 1: Create the extractor object
        frame_extractor = fc.FrameExtractor(self.video_path)

        # Step 2: Extract frames
        frame_extractor.extract_frames()

        # Step 3: Load the first frame to get image shape
        first_frame_path = f"{frame_extractor.output_dir}/frame_0000.png"
        frame = frame_extractor.load_grayscale_frame(first_frame_path)

        if frame is None or frame.size == 0:
            print("Error: Could not load first frame")
            return

        print(f"[Info] First frame shape: {frame.shape}")

        # Step 4: Initialize background subtractor with None shape for dynamic sizing
        bg_subtractor = bs(shape=None, alpha=0.003, history_size=40)

        import glob
        # Step 5: Iterate over all extracted frames
        frame_paths = sorted(glob.glob(f"{frame_extractor.output_dir}/frame_*.png"))
        min_size = 64  # Minimum size for HOG features

        # Build background model first
        print("Building background model for pose estimation...")
        for i, frame_path in enumerate(frame_paths[:30]):  # Use more frames for background
            frame = frame_extractor.load_grayscale_frame(frame_path)
            if frame is None or frame.size == 0:
                print(f"[Warning] Could not load frame {frame_path}")
                continue

            print(f"[Debug] Frame {i} shape: {frame.shape}")
            bg_subtractor.update(frame)
            if i % 5 == 0:
                print(f"Processing background frame {i + 1}/30")

        print("Processing frames for pose estimation...")

        # Initialize stroke analysis
        stroke_phases = []

        for frame_idx, frame_path in enumerate(frame_paths):
            frame = frame_extractor.load_grayscale_frame(frame_path)

            if frame is None or frame.size == 0:
                print(f"[Warning] Could not load frame {frame_path}")
                stroke_phases.append('unknown')
                continue

            try:
                # Get cleaned foreground mask
                fg_mask = bg_subtractor.get_cleaned_mask(frame, threshold=35, min_area=800)
            except Exception as e:
                print(f"[Error] Failed to process frame {frame_idx}: {e}")
                stroke_phases.append('unknown')
                continue

            # Find connected components
            components = bg_subtractor.connected_components(fg_mask, connectivity=8)

            # Merge nearby components
            components = bg_subtractor.merge_nearby_components(components, distance_threshold=120)

            # For visualization
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Create overlay for pose visualization
            pose_overlay = frame_vis.copy()

            valid_components = 0
            frame_analysis = []

            for label, points in components.items():
                if len(points) < 200:  # Skip small components
                    continue

                ys, xs = zip(*points)
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                width = max_x - min_x + 1
                height = max_y - min_y + 1

                # Skip components too small for meaningful analysis
                if width < min_size or height < min_size:
                    continue

                roi = frame[min_y:max_y + 1, min_x:max_x + 1]

                if roi.size == 0:
                    continue

                # Pad ROI to ensure minimum size
                roi = HOGFeatureExtractor.pad_roi(roi, min_size, min_size)

                try:
                    # Extract joint positions
                    joints = bg_subtractor.extract_rowing_joints(points, frame.shape)

                    # Analyze stroke phase (for visualization only)
                    stroke_analysis = bg_subtractor.analyze_rowing_stroke_phase(joints)

                    # Compute HOG features
                    hog_features = HOGFeatureExtractor.extract(roi)

                    # Track this component's analysis
                    component_analysis = {
                        'component_id': label,
                        'joints': joints,
                        'stroke_phase': stroke_analysis['stroke_phase'],
                        'torso_angle': stroke_analysis['torso_angle'],
                        'hog_features': hog_features
                    }
                    frame_analysis.append(component_analysis)

                    # Draw pose visualization (no feedback)
                    self._draw_pose_estimation(frame_vis, pose_overlay, joints, stroke_analysis,
                                             min_x, min_y, max_x, max_y, label)

                    # Print pose information only
                    print(f"Frame {frame_idx}, Person {label}:")
                    print(f"  Stroke Phase: {stroke_analysis['stroke_phase']}")
                    print(f"  Torso Angle: {stroke_analysis['torso_angle']:.1f}°")
                    print(f"  ROI: {width}x{height}, HOG vector: {hog_features.shape}")

                    valid_components += 1

                except Exception as e:
                    print(f"Error processing component {label}: {e}")
                    import traceback
                    traceback.print_exc()

            # Track overall frame statistics
            if frame_analysis:
                avg_torso_angle = np.mean([a['torso_angle'] for a in frame_analysis])
                dominant_phase = max(set([a['stroke_phase'] for a in frame_analysis]),
                                     key=[a['stroke_phase'] for a in frame_analysis].count)
                stroke_phases.append(dominant_phase)

                print(f"Frame {frame_idx} Summary: {valid_components} person(s), "
                      f"Phase: {dominant_phase}, Avg Torso: {avg_torso_angle:.1f}°")
            else:
                print(f"Frame {frame_idx}: No valid persons detected")
                stroke_phases.append('unknown')

            # Create blended visualization
            alpha = 0.7
            blended = cv2.addWeighted(frame_vis, alpha, pose_overlay, 1 - alpha, 0)

            # Show visualizations
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Foreground Mask", fg_mask * 255)
            cv2.imshow("Pose Estimation", blended)

            # Add stroke phase timeline at bottom
            timeline_img = self._create_stroke_timeline(stroke_phases, frame_idx)
            cv2.imshow("Stroke Phase Timeline", timeline_img)

            # Save pose estimation images
            try:
                out_name = os.path.basename(frame_path).replace("frame_", "")
                cv2.imwrite(f"{frame_extractor.output_dir}/mask_{out_name}", fg_mask * 255)
                cv2.imwrite(f"{frame_extractor.output_dir}/pose_estimation_{out_name}", blended)
            except Exception as e:
                print(f"[Warning] Could not save output images: {e}")

            key = cv2.waitKey(50)
            if key == 27:  # Esc to exit early
                break
            elif key == ord('p'):  # Pause
                cv2.waitKey(0)

        # Final analysis summary (no feedback)
        self._print_session_summary(stroke_phases)
        cv2.destroyAllWindows()

    def _draw_pose_estimation(self, frame_vis, overlay, joints, stroke_analysis,
                            min_x, min_y, max_x, max_y, component_label):
        """Draw pose estimation visualization without feedback"""

        # Draw bounding box with stroke phase color
        phase_colors = {
            'catch': (0, 255, 255),  # Yellow - preparation
            'drive': (0, 255, 0),  # Green - power phase
            'finish': (255, 0, 0),  # Blue - completion
            'recovery': (255, 0, 255),  # Magenta - return
            'unknown': (128, 128, 128)  # Gray - uncertain
        }

        phase_color = phase_colors.get(stroke_analysis['stroke_phase'], (255, 255, 255))
        cv2.rectangle(frame_vis, (min_x, min_y), (max_x, max_y), phase_color, 3)

        # Add stroke phase label
        cv2.putText(frame_vis, f"P{component_label}: {stroke_analysis['stroke_phase'].upper()}",
                    (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)

        # Draw joint skeleton
        joint_connections = [
            ('head', 'neck'),
            ('neck', 'shoulder_left'),
            ('neck', 'shoulder_right'),
            ('shoulder_left', 'elbow_left'),
            ('elbow_left', 'wrist_left'),
            ('shoulder_right', 'elbow_right'),
            ('elbow_right', 'wrist_right'),
            ('neck', 'spine_mid'),
            ('spine_mid', 'hips'),
            ('hips', 'knee_left'),
            ('knee_left', 'ankle_left'),
            ('hips', 'knee_right'),
            ('knee_right', 'ankle_right')
        ]

        # Draw skeleton connections
        for joint1_name, joint2_name in joint_connections:
            joint1 = joints.get(joint1_name, (0, 0))
            joint2 = joints.get(joint2_name, (0, 0))

            if joint1 != (0, 0) and joint2 != (0, 0):
                cv2.line(overlay, (joint1[1], joint1[0]), (joint2[1], joint2[0]),
                         (255, 255, 255), 2)

        # Draw joints with different colors and sizes
        joint_colors = {
            'head': (0, 0, 255),  # Red
            'neck': (0, 100, 255),  # Orange
            'shoulder_left': (0, 255, 0),  # Green
            'shoulder_right': (0, 255, 0),  # Green
            'elbow_left': (255, 255, 0),  # Cyan
            'elbow_right': (255, 255, 0),  # Cyan
            'wrist_left': (255, 0, 255),  # Magenta
            'wrist_right': (255, 0, 255),  # Magenta
            'spine_mid': (128, 0, 128),  # Purple
            'hips': (0, 255, 255),  # Yellow
            'knee_left': (255, 0, 0),  # Blue
            'knee_right': (255, 0, 0),  # Blue
            'ankle_left': (128, 128, 0),  # Dark cyan
            'ankle_right': (128, 128, 0)  # Dark cyan
        }

        joint_sizes = {
            'head': 8, 'neck': 6, 'hips': 8, 'spine_mid': 6,
            'shoulder_left': 7, 'shoulder_right': 7,
            'elbow_left': 6, 'elbow_right': 6,
            'wrist_left': 5, 'wrist_right': 5,
            'knee_left': 6, 'knee_right': 6,
            'ankle_left': 5, 'ankle_right': 5
        }

        for joint_name, (y, x) in joints.items():
            if (y, x) != (0, 0):
                color = joint_colors.get(joint_name, (255, 255, 255))
                size = joint_sizes.get(joint_name, 5)

                # Draw filled circle for joint
                cv2.circle(frame_vis, (x, y), size, color, -1)
                # Draw white border
                cv2.circle(frame_vis, (x, y), size + 1, (255, 255, 255), 1)

        # Draw torso angle indicator
        if joints['head'] != (0, 0) and joints['hips'] != (0, 0):
            head_y, head_x = joints['head']
            hip_y, hip_x = joints['hips']

            # Draw torso line
            cv2.line(overlay, (hip_x, hip_y), (head_x, head_y), (0, 255, 255), 3)

            # Add angle text
            angle_text = f"{stroke_analysis['torso_angle']:.1f}°"
            cv2.putText(overlay, angle_text, (hip_x + 10, hip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _create_stroke_timeline(self, stroke_phases, current_frame):
        """Create a timeline visualization of stroke phases"""
        timeline_height = 60
        timeline_width = 800
        timeline_img = np.zeros((timeline_height, timeline_width, 3), dtype=np.uint8)

        if not stroke_phases:
            return timeline_img

        phase_colors = {
            'catch': (0, 255, 255),
            'drive': (0, 255, 0),
            'finish': (255, 0, 0),
            'recovery': (255, 0, 255),
            'unknown': (128, 128, 128)
        }

        # Draw timeline
        pixels_per_frame = timeline_width / max(len(stroke_phases), 1)

        for i, phase in enumerate(stroke_phases):
            x1 = int(i * pixels_per_frame)
            x2 = int((i + 1) * pixels_per_frame)
            color = phase_colors.get(phase, (255, 255, 255))

            cv2.rectangle(timeline_img, (x1, 10), (x2, 50), color, -1)

            # Highlight current frame
            if i == current_frame:
                cv2.rectangle(timeline_img, (x1, 10), (x2, 50), (255, 255, 255), 2)

        # Add labels
        cv2.putText(timeline_img, "Stroke Timeline", (10, timeline_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(timeline_img, f"Frame: {current_frame}", (timeline_width - 100, timeline_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return timeline_img

    def _print_session_summary(self, stroke_phases):
        """Print pose estimation summary without feedback"""
        print("\n" + "=" * 60)
        print("POSE ESTIMATION SESSION SUMMARY")
        print("=" * 60)

        if stroke_phases:
            # Phase distribution
            phase_counts = {}
            for phase in stroke_phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            print("\nStroke Phase Distribution:")
            total_frames = len(stroke_phases)
            for phase, count in sorted(phase_counts.items()):
                percentage = (count / total_frames) * 100
                print(f"  {phase.capitalize()}: {count} frames ({percentage:.1f}%)")

            # Phase transitions
            phase_transitions = []
            for i in range(1, len(stroke_phases)):
                if stroke_phases[i] != stroke_phases[i - 1]:
                    phase_transitions.append((stroke_phases[i - 1], stroke_phases[i]))

            if phase_transitions:
                print(f"\nPhase Transitions: {len(phase_transitions)} detected")
                transition_counts = {}
                for trans in phase_transitions:
                    trans_str = f"{trans[0]} → {trans[1]}"
                    transition_counts[trans_str] = transition_counts.get(trans_str, 0) + 1

                print("Most Common Transitions:")
                for trans, count in sorted(transition_counts.items(),
                                           key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {trans}: {count} times")

        print("=" * 60)


if __name__ == "__main__":
    estimator = PoseEstimator(
        r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset\VID-20250609-WA0017_1 Paida.mp4")
    estimator.run()