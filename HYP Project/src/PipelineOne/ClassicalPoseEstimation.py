import numpy as np
import os
from GaussianMixtureBackgroundSubtractor import GaussianMixtureBackgroundSubtractor

class ClassicalPoseEstimation:
    def __init__(self, output_frames="frames"):
        self.output_folder = output_frames
        os.makedirs(self.output_folder, exist_ok=True)
        self.background_subtractor = None

    def grayscale(self, frame):
        if frame.ndim == 2:
            return frame.copy()
        return np.dot(frame[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)

    def extract_frames(self, video_path, frame_skip):
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame_path = os.path.join(self.output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(os.path.basename(frame_path))

            frame_count += 1

        cap.release()
        return extracted_frames

    def gaussian_blur(self, sigma, kernel_size):
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    def convolve_frame(self, frame, kernel):
        if kernel.shape[0] > 7:
            from numpy.fft import fft2, ifft2
            pad_size = kernel.shape[0] // 2
            padded = np.pad(frame, pad_size, mode='edge')
            kernel_padded = np.zeros_like(padded)
            kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
            return np.real(ifft2(fft2(padded) * fft2(np.rot90(kernel_padded, 2))))[pad_size:-pad_size,
                   pad_size:-pad_size]
        else:
            return self._convolve_frame_strided(frame, kernel)

    def _convolve_frame_strided(self, frame, kernel):
        if frame.ndim == 2:
            height, width = frame.shape
            channels = 1
        else:
            height, width, channels = frame.shape

        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        if channels == 1:
            padded = np.pad(frame, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            shape = (height, width, k_h, k_w)
            strides = padded.strides * 2
            windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
            result = np.einsum('ijkl,kl->ij', windows, kernel)
        else:
            padded = np.pad(frame, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
            shape = (height, width, k_h, k_w, channels)
            strides = padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1], padded.strides[2]
            windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
            result = np.einsum('ijklc,kl->ijc', windows, kernel)

        return np.clip(result, 0, 255).astype(np.uint8)

    def background_subtraction(self, frame):
        if self.background_subtractor is None:
            self.background_subtractor = GaussianMixtureBackgroundSubtractor(num_gaussians=5,
            learning_rate=0.01,
            background_threshold=0.7,initial_frame=frame)
        return self.background_subtractor.apply(frame)

    def morphological_operations(self, binary_mask, operation='open', kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size))
        pad = kernel_size // 2
        padded = np.pad(binary_mask, pad, mode='constant')
        output = np.zeros_like(binary_mask)

        if operation == 'open':

            temp = np.zeros_like(binary_mask)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    temp[i, j] = np.min(padded[i:i + kernel_size, j:j + kernel_size] * kernel)

            padded_temp = np.pad(temp, pad, mode='constant')
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    output[i, j] = np.max(padded_temp[i:i + kernel_size, j:j + kernel_size] * kernel)

        elif operation == 'close':
            temp = np.zeros_like(binary_mask)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    temp[i, j] = np.max(padded[i:i + kernel_size, j:j + kernel_size] * kernel)

            padded_temp = np.pad(temp, pad, mode='constant')
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    output[i, j] = np.min(padded_temp[i:i + kernel_size, j:j + kernel_size] * kernel)

        return output

    #
    def connected_components_labeling(self, binary_mask):
        height, width = binary_mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        current_label = 1

        def get_neighbors(y, x):
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if ny != y or nx != x:
                        yield ny, nx

        for y in range(height):
            for x in range(width):
                if binary_mask[y, x] != 0 and labels[y, x] == 0:
                    # Flood fill
                    queue = [(y, x)]
                    labels[y, x] = current_label
                    while queue:
                        cy, cx = queue.pop()
                        for ny, nx in get_neighbors(cy, cx):
                            if binary_mask[ny, nx] != 0 and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))
                    current_label += 1

        return labels, current_label - 1

    def compute_image_gradients(self, grayscale_image):
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 16
        scharr_y = scharr_x.T

        Ix = self.convolve_frame(grayscale_image, scharr_x)
        Iy = self.convolve_frame(grayscale_image, scharr_y)
        return Ix, Iy

    def shi_tomasi_corner_detection(self, grayscale_image, window_size=5, min_corner_response=100, step=1):
        Ix, Iy = self.compute_image_gradients(grayscale_image)
        Ix2, Iy2, IxIy = Ix * Ix, Iy * Iy, Ix * Iy

        height, width = grayscale_image.shape
        half_w = window_size // 2
        keypoints = []

        for y in range(half_w, height - half_w, step):
            for x in range(half_w, width - half_w, step):
                sum_Ix2 = float(np.sum(Ix2[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]))
                sum_Iy2 = float(np.sum(Iy2[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]))
                sum_IxIy = float(np.sum(IxIy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]))


                if abs(sum_Ix2) > 1e4 or abs(sum_Iy2) > 1e4 or abs(sum_IxIy) > 1e4:
                    continue


                trace = sum_Ix2 + sum_Iy2
                det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy

                # Skip if determinant is negative (shouldn't happen with real data)
                if det < 0:
                    continue

                discriminant = trace ** 2 - 4 * det
                if discriminant < 0:
                    continue

                min_eigen = (trace - np.sqrt(discriminant)) / 2

                if min_eigen > min_corner_response:
                    keypoints.append((y, x))

        return keypoints

    def classify_keypoints_by_region(self, bbox, keypoints):
        """Classify keypoints into head, torso, and lower body regions"""
        height = bbox[1] - bbox[0]
        head_pts = []
        torso_pts = []
        lower_pts = []

        for y, x in keypoints:

            rel_y = (y - bbox[0]) / height

            if rel_y < 0.3:
                head_pts.append((y, x))
            elif rel_y < 0.6:
                torso_pts.append((y, x))
            else:
                lower_pts.append((y, x))

        return head_pts, torso_pts, lower_pts

    def build_pose_skeleton(self, head_pts, torso_pts, lower_pts):
        """Create a simple skeleton connection between keypoints"""
        skeleton = []

        # Find center points for each region
        head_center = np.mean(head_pts, axis=0) if head_pts else None
        torso_center = np.mean(torso_pts, axis=0) if torso_pts else None
        lower_center = np.mean(lower_pts, axis=0) if lower_pts else None

        # Connect regions if they exist
        if head_center is not None and torso_center is not None:
            skeleton.append((head_center, torso_center))
        if torso_center is not None and lower_center is not None:
            skeleton.append((torso_center, lower_center))

        return skeleton

    def extract_anatomical_joints(self, bbox, keypoints, labels, blob_id):
        """Estimate key joints for a given blob using keypoints and bounding box."""

        joints = {}

        if not keypoints:
            return joints

        # Points belonging to the blob (where label == blob_id)
        ys, xs = np.where(labels == blob_id)
        blob_points = list(zip(ys, xs))

        if not blob_points:
            return joints

        # Head - top-most point in blob
        head = min(blob_points, key=lambda p: p[0])


        shoulders = [p for p in keypoints if 0.2 < (p[0] - bbox[0]) / (bbox[1] - bbox[0]) < 0.4]
        left_shoulder = min(shoulders, key=lambda p: p[1]) if shoulders else None
        right_shoulder = max(shoulders, key=lambda p: p[1]) if shoulders else None

        # Hips: keypoints between 50-70% height
        hips = [p for p in keypoints if 0.5 < (p[0] - bbox[0]) / (bbox[1] - bbox[0]) < 0.7]
        left_hip = min(hips, key=lambda p: p[1]) if hips else None
        right_hip = max(hips, key=lambda p: p[1]) if hips else None

        # Knees: keypoints below 70% height
        knees = [p for p in keypoints if (p[0] - bbox[0]) / (bbox[1] - bbox[0]) > 0.7]
        left_knee = min(knees, key=lambda p: p[1]) if knees else None
        right_knee = max(knees, key=lambda p: p[1]) if knees else None

        joints[blob_id] = {
            'Head': (head[1], head[0]),
            'Left Shoulder': (left_shoulder[1], left_shoulder[0]) if left_shoulder else None,
            'Right Shoulder': (right_shoulder[1], right_shoulder[0]) if right_shoulder else None,
            'Left Hip': (left_hip[1], left_hip[0]) if left_hip else None,
            'Right Hip': (right_hip[1], right_hip[0]) if right_hip else None,
            'Left Knee': (left_knee[1], left_knee[0]) if left_knee else None,
            'Right Knee': (right_knee[1], right_knee[0]) if right_knee else None
        }

        return joints

    def draw_skeleton(self, frame, skeleton):
        """Draw skeleton lines on the frame"""
        output = frame.copy()
        for (p1, p2) in skeleton:
            y1, x1 = map(int, p1)
            y2, x2 = map(int, p2)

            # Simple line drawing algorithm
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for t in np.linspace(0, 1, steps):
                y = int(y1 + (y2 - y1) * t)
                x = int(x1 + (x2 - x1) * t)
                if 0 <= y < output.shape[0] and 0 <= x < output.shape[1]:
                    output[y, x] = [255, 0, 0]  # Red line

        return output

    def lucas_kanade_optical_flow(self, prev_frame, curr_frame, prev_keypoints, window_size=5):
        """Simple Lucas-Kanade optical flow implementation"""
        if not prev_keypoints:
            return []

        # Compute gradients
        Ix, Iy = self.compute_image_gradients(prev_frame)
        It = curr_frame - prev_frame

        flow_vectors = []

        for y, x in prev_keypoints:
            # Get local window
            half_w = window_size // 2
            ix_window = Ix[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
            iy_window = Iy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
            it_window = It[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()

            # Solve least squares problem
            A = np.vstack((ix_window, iy_window)).T
            b = -it_window

            try:
                v, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                flow_vectors.append(v)
            except np.linalg.LinAlgError:
                flow_vectors.append((0, 0))

        return flow_vectors
