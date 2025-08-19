import numpy as np
from typing import List, Tuple, Dict, Optional

class TechniqueModel:
    """Models good rowing technique using statistical methods"""

    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.covariance_matrix = None
        self.inv_covariance = None
        self.n_features = None
        self.is_trained = False

    def train(self, good_technique_features: List[np.ndarray]):
        """Train the model on examples of good rowing technique"""
        if not good_technique_features:
            raise ValueError("No training data provided")

        # Stack all feature vectors
        features_matrix = np.vstack(good_technique_features)
        self.n_features = features_matrix.shape[1]

        print(f"Training technique model with {len(good_technique_features)} samples")
        print(f"Feature dimensionality: {self.n_features}")

        # Calculate statistics
        self.feature_means = np.mean(features_matrix, axis=0)
        self.feature_stds = np.std(features_matrix, axis=0)

        # Calculate covariance matrix
        centered_features = features_matrix - self.feature_means
        self.covariance_matrix = np.cov(centered_features.T)

        # Calculate inverse covariance for Mahalanobis distance
        try:
            self.inv_covariance = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.inv_covariance = np.linalg.pinv(self.covariance_matrix)

        self.is_trained = True
        print("Technique model training completed")

    def detect_deviations(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect deviations from good technique using Mahalanobis distance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting deviations")

        # Center the features
        centered_features = features - self.feature_means

        # Calculate Mahalanobis distance
        mahalanobis_dist = np.sqrt(
            np.dot(np.dot(centered_features, self.inv_covariance), centered_features))

        # Calculate individual feature deviations (z-scores)
        z_scores = np.abs(centered_features) / (self.feature_stds + 1e-8)

        return mahalanobis_dist, z_scores
