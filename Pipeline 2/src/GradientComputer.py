import numpy as np

class GradientComputer:
    @staticmethod
    def compute_gradients(image):
        dx = np.zeros_like(image, dtype=float)
        dy = np.zeros_like(image, dtype=float)
        dx[:, :-1] = image[:, 1:] - image[:, :-1]
        dy[:-1, :] = image[1:, :] - image[:-1, :]

        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * (180 / np.pi) % 180
        return mag, angle
