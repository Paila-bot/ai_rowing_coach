import numpy as np
import BiomechanicalAnalyzer as BA
import VideoProcessor as VP
import TechniqueModel as TM
import EvolutionaryOptimizer as EVO
import FeedbackNeuralNetwork as FNN
from typing import List, Tuple, Dict, Optional
import json
import os
import cv2


class RowingAnalysisSystem:
    """Main system that orchestrates the entire rowing analysis pipeline"""

    def __init__(self):
        self.video_processor = VP.VideoProcessor()
        self.biomech_analyzer = BA.BiomechanicalAnalyzer()
        self.technique_model = TM.TechniqueModel()
        self.feedback_model = None
        self.is_trained = False

        self.feature_names = [
            'left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
            'trunk_angle', 'left_arm_ratio', 'right_arm_ratio', 'left_leg_ratio',
            'right_leg_ratio', 'head_forward', 'shoulder_slope', 'hip_slope',
            'arm_symmetry', 'leg_symmetry'
        ]

    def train_system(self, good_videos_dir: str, max_videos: int = None):
        print(f"Training system with videos from: {good_videos_dir}")

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []

        if os.path.isdir(good_videos_dir):
            for file in os.listdir(good_videos_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(good_videos_dir, file))
        elif os.path.exists(good_videos_dir):
            video_files = [good_videos_dir]

        if max_videos:
            video_files = video_files[:max_videos]

        if not video_files:
            raise ValueError("No video files found in the specified directory")

        good_features = []
        for i, video_path in enumerate(video_files):
            print(f"\nProcessing video {i + 1}/{len(video_files)}: {os.path.basename(video_path)}")
            try:
                features = self._extract_video_features(video_path)
                good_features.extend(features)
                print(f"Extracted {len(features)} feature vectors")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

        if not good_features:
            raise ValueError("No features extracted from videos")

        print(f"\nTotal features extracted: {len(good_features)}")

        self.technique_model.train(good_features)

        # Create synthetic labels with some variety for neural network training
        # Instead of all zeros, create labels based on feature quality
        good_features_array = np.array(good_features)
        labels = self._create_synthetic_labels(good_features_array)

        self._optimize_feedback_network(good_features_array, labels)

        self.is_trained = True
        print("System training completed!")
        return len(good_features)

    def _create_synthetic_labels(self, features: np.ndarray) -> np.ndarray:
        """Create synthetic labels based on feature quality for training"""
        n_samples = features.shape[0]

        # Calculate a simple quality score based on feature variance
        # Features with extreme values get higher class labels
        feature_scores = np.abs(features - np.mean(features, axis=0))
        overall_scores = np.mean(feature_scores, axis=1)

        # Divide into 4 quality categories (0=excellent, 3=poor)
        percentiles = np.percentile(overall_scores, [25, 50, 75])
        labels = np.zeros(n_samples, dtype=int)

        for i, score in enumerate(overall_scores):
            if score <= percentiles[0]:
                labels[i] = 0  # Excellent
            elif score <= percentiles[1]:
                labels[i] = 1  # Good
            elif score <= percentiles[2]:
                labels[i] = 2  # Moderate
            else:
                labels[i] = 3  # Poor

        return labels

    def _optimize_feedback_network(self, X: np.ndarray, y: np.ndarray):
        """Use Evolutionary Algorithm to find best NN hyperparameters"""

        # Sanity check: make sure X is (n_samples, 14) and y is (n_samples,)
        if X.ndim != 2 or X.shape[1] != 14:
            raise ValueError(f"Expected X shape (n_samples, 14), got {X.shape}")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError(f"Expected y shape ({X.shape[0]},), got {y.shape}")

        def fitness(params):
            h1 = int(params['h1'])
            h2 = int(params['h2'])
            lr = params['lr']

            try:
                model = FNN.FeedbackNeuralNetwork(X.shape[1], [h1, h2], 4)
                model.train(X, y, epochs=50, learning_rate=lr)  # Reduced epochs for faster optimization

                correct = 0
                for i, x in enumerate(X):
                    pred = model.predict(x)
                    if pred == y[i]:
                        correct += 1

                return correct / len(X)
            except Exception as e:
                print(f"Error in fitness evaluation: {e}")
                return 0.0  # Return 0 fitness for failed configurations

        evo = EVO.EvolutionaryOptimizer(param_ranges={
            'h1': (8, 32),
            'h2': (4, 16),
            'lr': (0.001, 0.05)
        })

        best = evo.optimize(fitness, n_generations=5)  # Reduced generations for faster training
        print("Best neural net parameters:", best)

        # Train final model with best parameters
        h1 = int(best['h1'])
        h2 = int(best['h2'])
        lr = best['lr']

        self.feedback_model = FNN.FeedbackNeuralNetwork(X.shape[1], [h1, h2], 4)
        self.feedback_model.train(X, y, epochs=10, learning_rate=lr)

    def analyze_video(self, video_path: str) -> Dict[str, any]:
        if not self.is_trained:
            raise ValueError("System must be trained before analysis")

        print(f"\nAnalyzing video: {os.path.basename(video_path)}")
        features = self._extract_video_features(video_path)
        if not features:
            return {"error": "No features could be extracted from video"}

        frame_analyses = []
        overall_deviations = []

        for i, frame_features in enumerate(features):
            mahalanobis_dist, z_scores = self.technique_model.detect_deviations(frame_features)
            feedback_category = self._categorize_technique(z_scores)

            frame_analyses.append({
                'frame_idx': i,
                'mahalanobis_distance': float(mahalanobis_dist),
                'feature_deviations': z_scores.tolist(),
                'feedback_category': feedback_category
            })
            overall_deviations.append(mahalanobis_dist)

        return {
            'video_path': video_path,
            'overall_score': float(self._calculate_overall_score(overall_deviations)),
            'frame_analyses': frame_analyses,
            'detailed_feedback': self._generate_detailed_feedback(frame_analyses),
            'average_deviation': float(np.mean(overall_deviations)),
            'consistency_score': float(1.0 / (1.0 + np.std(overall_deviations))),
            'total_frames_analyzed': len(frame_analyses)
        }

    def _extract_video_features(self, video_path: str) -> List[np.ndarray]:
        frames = self.video_processor.load_video(video_path)
        sample_frames = frames[::5] if len(frames) > 100 else frames
        features = []

        for i, frame in enumerate(sample_frames):
            try:
                processed = self.video_processor.preprocess_frame(frame)
                body_points = self.biomech_analyzer.detect_body_points(processed)
                feature_vector = self.biomech_analyzer.extract_features(body_points)

                # Ensure feature vector is the right size
                if len(feature_vector) == 14:
                    features.append(feature_vector)
                else:
                    print(f"Warning: Feature vector has {len(feature_vector)} dimensions, expected 14")

            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue

        return features

    def _categorize_technique(self, z_scores: np.ndarray) -> str:
        if self.feedback_model:
            try:
                prediction = self.feedback_model.predict(z_scores)
                return [
                    "Excellent technique",
                    "Good technique with minor issues",
                    "Moderate technique issues",
                    "Significant technique problems"
                ][prediction]
            except Exception as e:
                print(f"Error in technique categorization: {e}")
                return "Uncategorized"
        else:
            return "Uncategorized"

    def _calculate_overall_score(self, deviations: List[float]) -> float:
        if not deviations:
            return 0.0

        avg = np.mean(deviations)
        consistency = 1.0 / (1.0 + np.std(deviations))
        base = max(0, 100 - avg * 15)
        return min(100, base * consistency)

    def _generate_detailed_feedback(self, frame_analyses: List[Dict]) -> List[str]:
        if not frame_analyses:
            return ["No frame data available for analysis"]

        feedback = []
        issue_counts = {}
        total = len(frame_analyses)

        for analysis in frame_analyses:
            cat = analysis['feedback_category']
            issue_counts[cat] = issue_counts.get(cat, 0) + 1

        for cat, count in issue_counts.items():
            if count > total * 0.3:
                pct = (count / total) * 100
                if "minor" in cat.lower():
                    feedback.append(f"Minor technique issues in {pct:.1f}% of frames")
                elif "moderate" in cat.lower():
                    feedback.append(f"Moderate issues in {pct:.1f}% - improve consistency")
                elif "significant" in cat.lower():
                    feedback.append(f"Major technique problems in {pct:.1f}% - technique review advised")

        # Analyze feature deviations
        try:
            avg_dev = np.mean([a['feature_deviations'] for a in frame_analyses], axis=0)
            problems = [(self.feature_names[i], d) for i, d in enumerate(avg_dev) if d > 2.0]
            problems.sort(key=lambda x: x[1], reverse=True)

            if problems:
                feedback.append("Problematic areas: " + ", ".join(p[0] for p in problems[:3]))
        except Exception as e:
            print(f"Error analyzing feature deviations: {e}")

        if not feedback:
            feedback.append("Technique is consistent throughout the video")

        return feedback

    def evaluate_system(self, test_videos_dir: str, max_videos: int = None) -> Dict[str, any]:
        if not self.is_trained:
            raise ValueError("System must be trained before evaluation")

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []

        if os.path.isdir(test_videos_dir):
            for file in os.listdir(test_videos_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(test_videos_dir, file))
        elif os.path.exists(test_videos_dir):
            video_files = [test_videos_dir]

        if max_videos:
            video_files = video_files[:max_videos]

        evaluation_results = []
        total_scores = []
        total_deviations = []

        for i, video_path in enumerate(video_files):
            print(f"\nEvaluating video {i + 1}/{len(video_files)}: {os.path.basename(video_path)}")
            try:
                analysis = self.analyze_video(video_path)
                if 'error' not in analysis:
                    evaluation_results.append(analysis)
                    total_scores.append(analysis['overall_score'])
                    total_deviations.append(analysis['average_deviation'])
                else:
                    print(f"Skipping video due to error: {analysis['error']}")
            except Exception as e:
                print(f"Error evaluating {video_path}: {e}")

        return {
            'total_videos_evaluated': len(evaluation_results),
            'average_score': float(np.mean(total_scores)) if total_scores else 0,
            'score_std': float(np.std(total_scores)) if total_scores else 0,
            'average_deviation': float(np.mean(total_deviations)) if total_deviations else 0,
            'deviation_std': float(np.std(total_deviations)) if total_deviations else 0,
            'individual_results': evaluation_results
        }

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'technique_model': {
                'feature_means': self.technique_model.feature_means.tolist(),
                'feature_stds': self.technique_model.feature_stds.tolist(),
                'covariance_matrix': self.technique_model.covariance_matrix.tolist(),
                'inv_covariance': self.technique_model.inv_covariance.tolist(),
                'n_features': self.technique_model.n_features
            },
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.technique_model.feature_means = np.array(model_data['technique_model']['feature_means'])
        self.technique_model.feature_stds = np.array(model_data['technique_model']['feature_stds'])
        self.technique_model.covariance_matrix = np.array(model_data['technique_model']['covariance_matrix'])
        self.technique_model.inv_covariance = np.array(model_data['technique_model']['inv_covariance'])
        self.technique_model.n_features = model_data['technique_model']['n_features']
        self.technique_model.is_trained = True

        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")