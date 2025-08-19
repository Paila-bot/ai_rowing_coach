import os
import numpy as np
import matplotlib.pyplot as plt
from ClassicalPoseEstimation import ClassicalPoseEstimation

class PoseVisualizer:
    def __init__(self, video_path, frame_skip=6):
        self.pose = ClassicalPoseEstimation()
        print("[INFO] Extracting frames...")
        self.frame_files = self.pose.extract_frames(video_path, frame_skip)
        self.current_frame_idx = 0
        self.folder = self.pose.output_folder
        self.prev_gray = None
        self.prev_keypoints = None
        self.show_all_frames()
        self.frames = [plt.imread(os.path.join(self.folder, f)) for f in self.frame_files]

    def visualize_keypoints(self, image, keypoints):
        """Draw green circles for Shi–Tomasi keypoints."""
        img_vis = image.copy()
        if img_vis.ndim == 2:  # grayscale → RGB
            img_vis = np.stack([img_vis] * 3, axis=-1)

        for y, x in keypoints:
            if 0 <= y < img_vis.shape[0] and 0 <= x < img_vis.shape[1]:
                yy, xx = np.ogrid[-2:3, -2:3]  # small square brush
                mask = (y + yy >= 0) & (y + yy < img_vis.shape[0]) & \
                       (x + xx >= 0) & (x + xx < img_vis.shape[1])
                img_vis[y + yy[mask], x + xx[mask]] = [0, 255, 0]  # green

        return img_vis

    def visualize_optical_flow(self, image, keypoints, flow_vectors):
        """Draw yellow lines for optical flow vectors."""
        img_vis = image.copy()
        if img_vis.ndim == 2:  # grayscale → RGB
            img_vis = np.stack([img_vis] * 3, axis=-1)

        for (y, x), (dy, dx) in zip(keypoints, flow_vectors):
            if dx != 0 or dy != 0:
                steps = max(abs(dx), abs(dy))
                for t in np.linspace(0, 1, max(1, int(steps))):
                    yy = int(y + dy * t)
                    xx = int(x + dx * t)
                    if 0 <= yy < img_vis.shape[0] and 0 <= xx < img_vis.shape[1]:
                        img_vis[yy, xx] = [255, 255, 0]  # yellow

        return img_vis


    def pose_pipeline(self):
        prev_frame = None
        prev_keypoints = None

        for idx, frame in enumerate(self.frames):
            # grayscale
            gray = self.pose.grayscale(frame)

            # blur
            blur = self.pose.convolve_frame(gray, self.pose.gaussian_blur(3.0, 5))

            # background subtraction
            fg_mask = self.pose.background_subtraction(blur)

            # morphological cleanup
            fg_mask_clean = self.pose.morphological_operations(fg_mask, operation='open', kernel_size=3)
            fg_mask_cleaner = self.pose.morphological_operations(fg_mask_clean, operation='close', kernel_size=5)

            # detect keypoints (first frame) or track (subsequent frames)
            if prev_frame is None:
                keypoints = self.pose.shi_tomasi_corner_detection(fg_mask_cleaner)
            else:
                flow_vectors = self.pose.lucas_kanade_optical_flow(prev_frame, gray, prev_keypoints)
                keypoints = [(int(y + v[1]), int(x + v[0]))
                             for (y, x), v in zip(prev_keypoints, flow_vectors)]

            # compute bounding box
            ys, xs = np.where(fg_mask_cleaner > 0)
            if len(ys) > 0 and len(xs) > 0:
                bbox = (ys.min(), ys.max())
            else:
                continue  # skip frame if nothing detected

            # classify by body regions
            head_pts, torso_pts, lower_pts = self.pose.classify_keypoints_by_region(bbox, keypoints)

            # build skeleton
            skeleton = self.pose.build_pose_skeleton(head_pts, torso_pts, lower_pts)

            # draw skeleton
            frame_with_skeleton = self.pose.draw_skeleton(frame, skeleton)

            plt.imshow(frame_with_skeleton, cmap="gray")
            plt.show()

            # update for next iteration
            prev_frame = gray
            prev_keypoints = keypoints

    def plot_results(self, frame, gray, blur, fg_mask, keypoints_img, optical_flow_img, idx, num_labels, keypoints):
        """Display all intermediate results for the given frame."""
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Frame {idx + 1}/{len(self.frame_files)} - {num_labels} connected components", fontsize=14)

        axs[0, 0].imshow(frame)
        axs[0, 0].set_title("Original")

        axs[0, 1].imshow(gray, cmap='gray')
        axs[0, 1].set_title("Grayscale")

        axs[0, 2].imshow(blur, cmap='gray')
        axs[0, 2].set_title("Gaussian Blur")

        axs[1, 0].imshow(fg_mask, cmap='gray')
        axs[1, 0].set_title("Foreground Mask")

        axs[1, 1].imshow(keypoints_img)
        axs[1, 1].set_title(f"Shi–Tomasi Keypoints ({len(keypoints)} pts)")

        axs[1, 2].imshow(optical_flow_img)
        axs[1, 2].set_title("Optical Flow")

        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    video_file = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset\VID-20250609-WA0017_1 Paida.mp4"
    PoseVisualizer(video_file, frame_skip=3)