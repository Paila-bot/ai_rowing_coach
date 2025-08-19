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
import FeatureExtractor
import PoseDetector
import PhaseDetector

def auto_label_clusters(cluster_centers: np.ndarray) -> Dict[int, str]:
    """Map clusters to phases based on biomechanics"""
    # Sort clusters by knee angle (catch -> drive -> finish -> recovery)
    sorted_clusters = np.argsort(cluster_centers[:, 0])  # Column 0 = knee angle

    labels = {
        0: "catch",     # Minimal knee angle
        1: "drive",     # Increasing angle
        2: "finish",    # Max torso lean
        3: "recovery"   # Decreasing angle
    }
    return {k: labels[v] for k, v in enumerate(sorted_clusters)}

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

class RowingCoach:
    def __init__(self):
        self.pose_processor = PoseDetector()
        self.feature_extractor = FeatureExtractor()
        self.phase_detector = PhaseDetector(n_clusters=4)

        self.scaler = StandardScaler()
        self.autoencoder = None
        self.sequence_length = None
        self.expected_features = 6  # Now fixed at 6 features

    def _pad_sequences(self, sequences):
        """Pad sequences to same length"""
        if not sequences:
            return np.array([])

        padded = []
        for seq in sequences:
            if len(seq) < self.sequence_length:
                # Pad with last frame repeated
                padding = np.repeat([seq[-1]], self.sequence_length - len(seq), axis=0)
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq[:self.sequence_length]
            padded.append(padded_seq)
        return np.array(padded)

    def train(self, video_paths):
        print(f"Processing {len(video_paths)} videos...")
        all_sequences = []
        all_feature_sequences = []

        # Process all videos
        for path in video_paths:
            if not os.path.exists(path):
                print(f"Video not found: {path}")
                continue

            try:
                pose_data = self.pose_processor.process_video(path, debug=False)
                if not pose_data:
                    print(f"No poses detected in: {os.path.basename(path)}")
                    continue

                angles = self.feature_extractor.calculate_angles(pose_data)
                if len(angles) > 10:  # Need minimum sequence length
                    all_sequences.append(angles)
                    all_feature_sequences.append(angles)
                    print(f"✅ {os.path.basename(path)}: {len(angles)} frames, {angles.shape[1]} features")
                else:
                    print(f"⚠️ {os.path.basename(path)}: Too short ({len(angles)} frames)")

            except Exception as e:
                print(f"⚠️ Error processing {os.path.basename(path)}: {str(e)}")
                continue

        if not all_sequences:
            raise ValueError("No valid training data found")

        print(f"\n🎯 Training with {len(all_sequences)} valid sequences")
        print(f"📊 All sequences have {self.expected_features} features per frame")

        # Train phase detector with feature sequences (not pose sequences)
        print("🎯 Training phase detector...")
        self.phase_detector.fit(all_feature_sequences)

        # Train autoencoder
        print("📊 Training autoencoder...")
        flattened = np.vstack(all_sequences)
        self.scaler.fit(flattened)

        self.sequence_length = max(len(seq) for seq in all_sequences)
        self.autoencoder = build_autoencoder((self.sequence_length, self.expected_features))

        padded = self._pad_sequences(all_sequences)
        scaled_padded = np.array([self.scaler.transform(seq) for seq in padded])

        self.autoencoder.fit(
            scaled_padded, scaled_padded,
            epochs=50,
            batch_size=min(8, len(scaled_padded)),
            shuffle=True,
            verbose=1
        )

        print("✅ Training completed!")

    def analyze(self, new_video_path):
        try:
            pose_data = self.pose_processor.process_video(new_video_path, debug=False)
            if not pose_data:
                raise ValueError("No poses detected in video")

            angles = self.feature_extractor.calculate_angles(pose_data)
            if len(angles) == 0:
                raise ValueError("No valid features extracted")

            # Get phases using feature sequences
            phases = self.phase_detector.predict(angles)

            angles_scaled = self.scaler.transform(angles)
            padded = self._pad_sequences([angles_scaled])[0:1]
            reconstructed = self.autoencoder.predict(padded)[0]

            actual_length = len(angles)
            errors = np.mean(np.square(
                angles_scaled - reconstructed[:actual_length]
            ), axis=1)

            return {
                'phases': phases,
                'anomaly_scores': errors,
                'critical_frames': np.where(errors > np.percentile(errors, 90))[0],
                'reconstructed': reconstructed
            }
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return None

    def save_models(self, output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Save autoencoder separately
            if self.autoencoder is not None:
                self.autoencoder.save(os.path.join(output_dir, 'autoencoder.keras'))
                print("✅ Autoencoder saved")

            # Save other components without circular references
            models_to_save = {
                'phase_detector_kmeans': self.phase_detector.kmeans,
                'phase_detector_fitted': self.phase_detector.fitted,
                'phase_detector_feature_dim': self.phase_detector.feature_dim,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'expected_features': self.expected_features
            }

            joblib.dump(models_to_save, os.path.join(output_dir, 'scikit_models.pkl'))
            print("✅ Supporting models saved")

        except Exception as e:
            print(f"Save error: {str(e)}")

    @classmethod
    def load_models(cls, model_dir):
        try:
            coach = cls()

            # Load autoencoder
            autoencoder_path = os.path.join(model_dir, 'autoencoder.keras')
            if os.path.exists(autoencoder_path):
                coach.autoencoder = load_model(autoencoder_path)
                print("✅ Autoencoder loaded")

            # Load other components
            sklearn_path = os.path.join(model_dir, 'scikit_models.pkl')
            if os.path.exists(sklearn_path):
                models = joblib.load(sklearn_path)

                # Reconstruct phase detector
                coach.phase_detector.kmeans = models['phase_detector_kmeans']
                coach.phase_detector.fitted = models['phase_detector_fitted']
                coach.phase_detector.feature_dim = models['phase_detector_feature_dim']

                coach.scaler = models['scaler']
                coach.sequence_length = models['sequence_length']
                coach.expected_features = models.get('expected_features', 6)
                print("✅ Supporting models loaded")

            return coach

        except Exception as e:
            print(f"Load error: {str(e)}")
            return cls()