import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # Added import
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector
import os
import joblib

class PhaseDetector:
    def __init__(self, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.fitted = False
        self.feature_dim = None

    def fit(self, feature_sequences: List[np.ndarray]):
        """Fit using pre-computed features"""
        all_features = []
        valid_sequences = 0

        for features in feature_sequences:
            if len(features) > 5:  # Need minimum sequence length
                all_features.append(features)
                valid_sequences += 1

        if not all_features:
            raise ValueError("No features provided for phase detection")

        print(f"📊 Phase detector: {valid_sequences} sequences")

        # All sequences now have same feature dimension
        self.feature_dim = all_features[0].shape[1]
        print(f"📊 Feature dimensions: {self.feature_dim} per frame")

        # Concatenate all features for clustering
        features_concat = np.vstack(all_features)
        print(f"📊 Clustering {features_concat.shape[0]} frames with {features_concat.shape[1]} features")

        # Fit k-means
        self.kmeans.fit(features_concat)
        self.fitted = True

        print("✅ Phase detector trained")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict phases for feature sequence"""
        if not self.fitted:
            raise ValueError("Phase detector not fitted")

        if len(features) == 0:
            return np.array([])

        return self.kmeans.predict(features)

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)

    # Decoder
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    outputs = Dense(input_shape[1])(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model