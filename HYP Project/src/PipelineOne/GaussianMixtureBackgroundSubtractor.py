import numpy as np


class GaussianMixtureBackgroundSubtractor:
    def __init__(self, num_gaussians=3, learning_rate=0.01, background_threshold=0.7, initial_frame=None):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        self.background_threshold = background_threshold

        frame_height, frame_width = initial_frame.shape

        # Means of Gaussians for each pixel (height x width x num_gaussians)
        self.means = np.zeros((frame_height, frame_width, num_gaussians), dtype=np.float32)
        # Variances of Gaussians for each pixel
        self.variances = np.ones((frame_height, frame_width, num_gaussians), dtype=np.float32) * 15.0
        # Weights of each Gaussian component per pixel
        self.weights = np.ones((frame_height, frame_width, num_gaussians), dtype=np.float32) / num_gaussians

        # Initialize the first Gaussian mean to initial frame pixel values, others random noise
        for gaussian_index in range(num_gaussians):
            if gaussian_index == 0:
                self.means[:, :, gaussian_index] = initial_frame
            else:
                self.means[:, :, gaussian_index] = np.random.uniform(0, 255, size=(frame_height, frame_width))

    def apply(self, current_frame):
        """
        Process the current frame and update GMM parameters.
        Returns a binary foreground mask: 255 for foreground, 0 for background.
        """
        frame_height, frame_width = current_frame.shape
        current_frame = current_frame.astype(np.float32)

        # Calculate absolute difference between current pixel and Gaussian means for all components
        difference = np.abs(current_frame[:, :, None] - self.means)  # shape: (height, width, num_gaussians)

        # Check which Gaussians match pixel within 2.5 standard deviations
        matches = difference <= 2.5 * np.sqrt(self.variances)

        # For each pixel, does it match any Gaussian component?
        pixel_matches_any = np.any(matches, axis=2)

        # Update parameters for each Gaussian component
        for gaussian_index in range(self.num_gaussians):
            matched_pixels = matches[:, :, gaussian_index]  # boolean mask of pixels matching this Gaussian

            # Compute update factor rho based on Gaussian probability density
            update_factor = self.learning_rate * self.gaussian_probability(
                current_frame,
                self.means[:, :, gaussian_index],
                self.variances[:, :, gaussian_index]
            )

            # Update mean only where matched
            self.means[:, :, gaussian_index] = np.where(
                matched_pixels,
                (1 - update_factor) * self.means[:, :, gaussian_index] + update_factor * current_frame,
                self.means[:, :, gaussian_index]
            )

            # Calculate squared difference for variance update
            squared_difference = (current_frame - self.means[:, :, gaussian_index]) ** 2

            # Update variance only where matched
            self.variances[:, :, gaussian_index] = np.where(
                matched_pixels,
                (1 - update_factor) * self.variances[:, :, gaussian_index] + update_factor * squared_difference,
                self.variances[:, :, gaussian_index]
            )

            # Update weights: increase for matched pixels, decrease otherwise
            self.weights[:, :, gaussian_index] = np.where(
                matched_pixels,
                (1 - self.learning_rate) * self.weights[:, :, gaussian_index] + self.learning_rate,
                (1 - self.learning_rate) * self.weights[:, :, gaussian_index]
            )

        # For pixels with no matching Gaussian, replace the least probable Gaussian component
        pixels_without_match = ~pixel_matches_any
        for y in range(frame_height):
            for x in range(frame_width):
                if pixels_without_match[y, x]:
                    least_weight_index = np.argmin(self.weights[y, x, :])
                    self.means[y, x, least_weight_index] = current_frame[y, x]
                    self.variances[y, x, least_weight_index] = 15.0
                    self.weights[y, x, least_weight_index] = 0.05

        # Normalize weights so sum of weights at each pixel equals 1
        weight_sum_per_pixel = np.sum(self.weights, axis=2, keepdims=True)
        self.weights /= weight_sum_per_pixel

        # Determine background pixels by checking if pixel fits any background Gaussian
        # Background Gaussians are those with cumulative weights up to background_threshold
        sorted_indices = np.argsort(self.weights / np.sqrt(self.variances), axis=2)[:, :, ::-1]  # descending order

        background_mask = np.zeros_like(current_frame, dtype=bool)

        for y in range(frame_height):
            for x in range(frame_width):
                cumulative_weight = 0.0
                pixel_value = current_frame[y, x]
                pixel_is_background = False

                # Check Gaussians in order of importance
                for gaussian_idx in sorted_indices[y, x]:
                    cumulative_weight += self.weights[y, x, gaussian_idx]
                    mean = self.means[y, x, gaussian_idx]
                    variance = self.variances[y, x, gaussian_idx]

                    # If pixel fits this Gaussian within 2.5 std dev, mark as background
                    if abs(pixel_value - mean) <= 2.5 * np.sqrt(variance):
                        pixel_is_background = True
                        break

                    # Stop if cumulative weight exceeds threshold
                    if cumulative_weight > self.background_threshold:
                        break

                background_mask[y, x] = pixel_is_background

        # Foreground mask is inverse of background mask; convert boolean to 0 or 255
        foreground_mask = (~background_mask).astype(np.uint8) * 255

        return foreground_mask

    @staticmethod
    def gaussian_probability(x, mean, variance):
        """Calculate Gaussian probability density for scalar input x."""
        return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance))