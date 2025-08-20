import os
import numpy as np
import matplotlib.pyplot as plt
from ClassicalPoseEstimation import ClassicalPoseEstimation

class PoseVisualizer:
    def __init__(self, video_path, frame_skip=6):
        self.pose = ClassicalPoseEstimation()
        print("[INFO] Extracting frames...")
        self.frame_files = self.pose.extract_frames(video_path, frame_skip)
        self.folder = self.pose.output_folder
        # load frames as uint8 RGB
        self.frames = []
        for f in self.frame_files:
            frame = plt.imread(os.path.join(self.folder, f))
            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=-1)
            if frame.dtype != np.uint8:
                frame = (frame*255).astype(np.uint8)
            self.frames.append(frame)

    def visualize_keypoints(self, image, keypoints):
        img_vis = image.copy()
        for y, x in keypoints:
            if 0 <= y < img_vis.shape[0] and 0 <= x < img_vis.shape[1]:
                y_min, y_max = max(0, y-2), min(img_vis.shape[0], y+3)
                x_min, x_max = max(0, x-2), min(img_vis.shape[1], x+3)
                img_vis[y_min:y_max, x_min:x_max] = [0, 255, 0]  # green
        return img_vis

    def visualize_optical_flow(self, image, keypoints, flow_vectors):
        img_vis = image.copy()
        for (y, x), (dx, dy) in zip(keypoints, flow_vectors):
            x0, y0 = int(x), int(y)
            x1, y1 = int(x + dx), int(y + dy)
            # interpolate line
            num_steps = max(abs(x1 - x0), abs(y1 - y0))
            for t in np.linspace(0, 1, num_steps + 1):
                xx = int(x0 + (x1 - x0) * t)
                yy = int(y0 + (y1 - y0) * t)
                if 0 <= xx < img_vis.shape[1] and 0 <= yy < img_vis.shape[0]:
                    img_vis[yy, xx] = [255, 255, 0]  # yellow
            # small tip
            if 0 <= x1 < img_vis.shape[1] and 0 <= y1 < img_vis.shape[0]:
                img_vis[y1, x1] = [255, 200, 0]
        return img_vis

    def pose_pipeline(self):
        prev_gray = None
        prev_keypoints = None
        reinit_interval = 30  # re-detect keypoints every 30 frames

        for idx, frame in enumerate(self.frames):
            # grayscale
            gray = self.pose.grayscale(frame)
            # blur
            blur = self.pose.convolve_frame(gray, self.pose.gaussian_blur(3.0, 5))
            # background subtraction
            fg_mask = self.pose.background_subtraction(gray)
            fg_mask_clean = self.pose.morphological_operations(fg_mask, operation='open', kernel_size=3)
            fg_mask_clean = self.pose.morphological_operations(fg_mask_clean, operation='close', kernel_size=5)

            # initialize or optical flow
            if prev_gray is None or idx % reinit_interval == 0:
                keypoints = self.pose.shi_tomasi_corner_detection(fg_mask_clean)
                flow_vectors = [(0,0)] * len(keypoints)
            else:
                flow_vectors = self.pose.lucas_kanade_optical_flow(prev_gray, gray, prev_keypoints)
                keypoints = []
                for (y, x), (dx, dy) in zip(prev_keypoints, flow_vectors):
                    new_x = max(0, min(int(x + dx), frame.shape[1]-1))
                    new_y = max(0, min(int(y + dy), frame.shape[0]-1))
                    keypoints.append((new_y, new_x))

            # visualize keypoints and optical flow
            keypoints_img = self.visualize_keypoints(frame, keypoints)
            optical_flow_img = self.visualize_optical_flow(frame, keypoints, flow_vectors)

            # bounding box
            ys, xs = np.where(fg_mask_clean > 0)
            if len(ys) == 0 or len(xs) == 0:
                prev_gray = gray
                prev_keypoints = keypoints
                continue
            bbox = (ys.min(), ys.max(), xs.min(), xs.max())

            # classify and skeleton
            head_pts, torso_pts, lower_pts = self.pose.classify_keypoints_by_region(bbox, keypoints)
            skeleton = self.pose.build_pose_skeleton(head_pts, torso_pts, lower_pts)
            frame_with_skeleton = self.pose.draw_skeleton(frame.copy(), skeleton)

            # connected components
            num_labels = np.max(fg_mask_clean) if np.any(fg_mask_clean) else 0

            # plot
            self.plot_results(frame_with_skeleton, gray, blur, fg_mask_clean,
                              keypoints_img, optical_flow_img, idx, num_labels, keypoints)

            # update
            prev_gray = gray
            prev_keypoints = keypoints

    def plot_results(self, frame, gray, blur, fg_mask, keypoints_img, optical_flow_img, idx, num_labels, keypoints):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Frame {idx + 1}/{len(self.frame_files)} - {num_labels} connected components", fontsize=14)

        axs[0,0].imshow(frame)
        axs[0,0].set_title("Original")

        axs[0,1].imshow(gray, cmap='gray')
        axs[0,1].set_title("Grayscale")

        axs[0,2].imshow(blur, cmap='gray')
        axs[0,2].set_title("Gaussian Blur")

        axs[1,0].imshow(fg_mask, cmap='gray')
        axs[1,0].set_title("Foreground Mask")

        axs[1,1].imshow(keypoints_img)
        axs[1,1].set_title(f"Shi–Tomasi Keypoints ({len(keypoints)} pts)")

        axs[1,2].imshow(optical_flow_img)
        axs[1,2].set_title("Optical Flow")

        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    video_file = r"C:\Users\paida\Documents\Rowing Dataset\row_1.mp4"
    pose = PoseVisualizer(video_file, frame_skip=1)
    pose.pose_pipeline()
