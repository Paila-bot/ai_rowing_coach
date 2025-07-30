import numpy as np
import GradientComputer as gc
from src.GradientComputer import GradientComputer


class HOGFeatureExtractor:
    @staticmethod
    def compute_hog_block(mag, angle, cell_size=8, bin_count=9):
        h, w = mag.shape
        hog = []
        bin_width = 180 // bin_count

        for i in range(0, h - cell_size + 1, cell_size):
            row = []
            for j in range(0, w - cell_size + 1, cell_size):
                cell_mag = mag[i:i + cell_size, j:j + cell_size]
                cell_angle = angle[i:i + cell_size, j:j + cell_size]
                hist = np.zeros(bin_count)
                for m in range(cell_size):
                    for n in range(cell_size):
                        bin_idx = int(cell_angle[m, n] // bin_width) % bin_count
                        hist[bin_idx] += cell_mag[m, n]
                row.append(hist)
            hog.append(row)

        return np.array(hog)

    @staticmethod  # Added missing decorator
    def normalize_blocks(hog_cells, block_size=2, epsilon=1e-5):
        # Check if hog_cells is valid
        if hog_cells.size == 0:
            return np.array([])

        if hog_cells.ndim != 3:
            print(f"[Warning] Invalid hog_cells shape: {hog_cells.shape}, skipping this ROI.")
            return np.array([])

        n_cells_y, n_cells_x, bin_count = hog_cells.shape
        blocks = []

        # Check if we have enough cells for at least one block
        if n_cells_y < block_size or n_cells_x < block_size:
            print(f"[Warning] Not enough cells ({n_cells_y}x{n_cells_x}) for block size {block_size}")
            return np.array([])

        for i in range(n_cells_y - block_size + 1):
            for j in range(n_cells_x - block_size + 1):
                block = hog_cells[i:i + block_size, j:j + block_size, :].ravel()
                norm = np.linalg.norm(block) + epsilon
                normalized = block / norm
                blocks.append(normalized)

        return np.array(blocks)  # shape: (num_blocks, block_size*block_size*bin_count)

    @staticmethod
    def extract(image, cell_size=8, bin_count=9, block_size=2):
        # Ensure image is valid
        if image.size == 0:
            print("[Warning] Empty image provided to HOG extractor")
            return np.array([])

        # Ensure image dimensions are compatible with cell_size
        h, w = image.shape
        if h < cell_size or w < cell_size:
            print(f"[Warning] Image too small ({h}x{w}) for cell size {cell_size}")
            return np.array([])

        try:
            mag, angle = GradientComputer.compute_gradients(image)
            hog_cells = HOGFeatureExtractor.compute_hog_block(mag, angle, cell_size, bin_count)

            if hog_cells.size == 0:
                return np.array([])

            hog_blocks = HOGFeatureExtractor.normalize_blocks(hog_cells, block_size)

            if hog_blocks.size == 0:
                return np.array([])

            return hog_blocks.ravel()  # final HOG feature vector
        except Exception as e:
            print(f"[Error] HOG extraction failed: {e}")
            return np.array([])

    @staticmethod  # Added missing decorator
    def pad_roi(roi, min_height, min_width):
        if roi.size == 0:
            return np.zeros((min_height, min_width), dtype=roi.dtype)

        h, w = roi.shape
        pad_h = max(0, min_height - h)
        pad_w = max(0, min_width - w)
        return np.pad(roi, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)