# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ClassicalPoseEstimation import ClassicalPoseEstimation
# import csv
#
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from ClassicalPoseEstimation import ClassicalPoseEstimation
#



# import os
# import cv2
# import numpy as np
# import csv
# from ClassicalPoseEstimation import ClassicalPoseEstimation
#
# class PoseExtractor:
#     def __init__(self, video_path, frame_skip=3, csv_path="joints.csv"):
#         self.pose = ClassicalPoseEstimation()
#         print("[INFO] Extracting frames...")
#         self.frame_files = self.pose.extract_frames(video_path, frame_skip)
#         self.folder = self.pose.output_folder
#         self.csv_path = csv_path
#
#         self.prev_gray = None
#         self.prev_keypoints = None
#
#         self.extract_and_save_joints()
#
#     def extract_and_save_joints(self):
#         # Create CSV with header if it doesn't exist
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(["frame", "blob_id", "joint_name", "x", "y"])
#
#         for idx, filename in enumerate(self.frame_files):
#             frame_path = os.path.join(self.folder, filename)
#             frame = cv2.imread(frame_path)
#             if frame is None:
#                 print(f"[WARNING] Could not read frame {frame_path}")
#                 continue
#
#             gray = self.pose.grayscale(frame)
#             blur = self.pose.convolve_frame(gray, self.pose.gaussian_blur(3.0, 5))
#
#             mask = self.pose.background_subtraction(gray)
#             fg_mask = self.pose.morphological_operations(mask, operation='open', kernel_size=3)
#             fg_mask = self.pose.morphological_operations(fg_mask, operation='close', kernel_size=5)
#
#             # Connected components labeling
#             labels, num_labels = self.pose.connected_components_labeling(fg_mask)
#
#             # Detect keypoints on foreground area
#             blur_fg = np.where(fg_mask > 0, blur, 0)
#             keypoints = self.pose.shi_tomasi_corner_detection(blur_fg)
#
#             # Extract joints per blob
#             all_joints = {}
#             for blob_id in range(1, num_labels + 1):
#                 # Find bounding box for the blob
#                 ys, xs = np.where(labels == blob_id)
#                 if len(ys) == 0 or len(xs) == 0:
#                     continue
#                 bbox = (min(ys), max(ys))  # top and bottom row of the blob
#
#                 joints = self.pose.extract_anatomical_joints(bbox, keypoints, labels, blob_id)
#                 if joints:
#                     all_joints.update(joints)
#
#             # Write joints to CSV
#             with open(self.csv_path, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 for blob_id, joints_dict in all_joints.items():
#                     for joint_name, coords in joints_dict.items():
#                         if coords is not None:
#                             x, y = coords
#                             writer.writerow([idx, blob_id, joint_name, x, y])
#
#             print(f"[INFO] Processed frame {idx + 1}/{len(self.frame_files)} with {num_labels} blobs.")
#
#             # Update previous frame data for optical flow if you want to keep it (optional)
#             self.prev_gray = gray
#             self.prev_keypoints = keypoints
#
# if __name__ == "__main__":
#     video_file = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset\VID-20250609-WA0017_1 Paida.mp4"
#     PoseExtractor(video_file, frame_skip=3, csv_path=r"C:\Users\brigh\Documents\Honours\HYP\HYP Project\src\PipelineOne\csv files\keypoints.csv")

from BiomechanicalAnalyser import BiomechanicalAnalyser  # Adjust import as needed
from ActionQualityReporter import PostSessionFeedbackGenerator

def score_technique(analyzer):
    # Simple scoring example based on average angles completeness and stroke rate
    valid_angles = 0
    total_angles = 0

    all_angles = analyzer.left_knee_angles + analyzer.right_knee_angles + analyzer.left_elbow_angles + analyzer.right_elbow_angles
    for angle in all_angles:
        if angle is not None:
            valid_angles += 1
        total_angles += 1

    angle_score = (valid_angles / total_angles) * 100 if total_angles > 0 else 0
    stroke_rate_score = (analyzer.stroke_rate or 0) * 2  # arbitrary scaling

    final_score = min(angle_score + stroke_rate_score, 100)
    return final_score


def main(csv_path, output_path):
    analyzer = BiomechanicalAnalyser()
    feedback_gen = PostSessionFeedbackGenerator()

    print("Loading joint data from CSV...")
    all_frame_data = list(analyzer.load_joints_from_csv(csv_path))

    print(f"Loaded {len(all_frame_data)} frames")

    # Analyze each frame
    analyzed_frames = []
    for frame_num, joints in all_frame_data:
        angles = analyzer.analyze_frame(joints)
        analyzed_frames.append((frame_num, angles))

    print("Segmenting into strokes...")
    strokes = analyzer.segment_into_strokes(analyzed_frames)
    print(f"Found {len(strokes)} strokes")

    # Analyze each stroke
    all_stroke_analyses = []
    for i, stroke_data in enumerate(strokes):
        print(f"Analyzing stroke {i + 1}/{len(strokes)}")
        stroke_analysis = analyzer.analyze_stroke_phases(stroke_data)
        all_stroke_analyses.append(stroke_analysis)

    # Generate final feedback
    print("Generating feedback...")
    summary = feedback_gen.generate_summary(all_stroke_analyses)
    feedback_gen.save_feedback_to_text(summary, output_path)

    print(f"Feedback saved to {output_path}")
    return summary

if __name__ == "__main__":
    csv_file = r"C:\Users\brigh\Documents\Honours\HYP\HYP Project\src\PipelineOne\csv files\keypoints.csv"
    output_file = r"C:\Users\brigh\Documents\Honours\HYP\HYP Project\src\PipelineOne\feedback"
    main(csv_file, output_file)



