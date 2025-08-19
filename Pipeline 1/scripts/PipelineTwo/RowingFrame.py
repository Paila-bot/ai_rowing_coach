import numpy as np
from dataclasses import dataclass

@dataclass
class RowingFrame:
    """Data structure to hold rowing frame information"""
    frame_idx: int
    timestamp: float
    body_points: np.ndarray  # Key body points (x, y) coordinates
    features: np.ndarray  # Extracted biomechanical features