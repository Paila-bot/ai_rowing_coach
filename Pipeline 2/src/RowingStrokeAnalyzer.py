import numpy as np
import csv
import os
import glob
from typing import Dict, List, Tuple, Optional
import cv2


class RowingStrokeAnalyzer:
    """
    Main class for training on multiple rowing videos and analyzing stroke patterns
    """

    def __init__(self, output_dir: str = "stroke_analysis"):
        """
        Initialize the rowing stroke analyzer

        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Training data storage
        self.training_data = []
        self.angle_profiles = {}
        self.stroke_templates = {}

        # Analysis parameters
        self.angle_features = [
            'torso_angle', 'knee_angle_left', 'knee_angle_right',
            'elbow_angle_left', 'elbow_angle_right',
            'hip_knee_angle_left', 'hip_knee_angle_right'
        ]

    def train_on_videos(self, video_paths: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        Train the analyzer on multiple rowing videos

        Args:
            video_paths: List of paths to training videos
            labels: Optional labels for each video (e.g., skill level, technique type)

        Returns:
            Dictionary containing training summary
        """
        if labels is None:
            labels = [f"video_{i}" for i in range(len(video_paths))]

        training_summary = {
            'videos_processed': 0,
            'total_frames': 0,
            'stroke_cycles_detected': 0,
            'videos_failed': []
        }

        print(f"[Info] Starting training on {len(video_paths)} videos...")

        for i, video_path in enumerate(video_paths):
            try:
                print(f"\n[Info] Processing training video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")

                # Process single video
                video_data = self._process_single_video(video_path, labels[i])

                if video_data:
                    self.training_data.append(video_data)
                    training_summary['videos_processed'] += 1
                    training_summary['total_frames'] += video_data['total_frames']
                    training_summary['stroke_cycles_detected'] += video_data['stroke_cycles']

                    print(f"[Success] Video processed: {video_data['stroke_cycles']} stroke cycles detected")
                else:
                    training_summary['videos_failed'].append(video_path)
                    print(f"[Error] Failed to process video: {video_path}")

            except Exception as e:
                print(f"[Error] Exception processing {video_path}: {e}")
                training_summary['videos_failed'].append(video_path)

        # Build stroke templates from training data
        if self.training_data:
            self._build_stroke_templates()
            self._save_training_results()

        print(f"\n[Info] Training completed:")
        print(f"  Videos processed: {training_summary['videos_processed']}")
        print(f"  Total frames: {training_summary['total_frames']}")
        print(f"  Stroke cycles: {training_summary['stroke_cycles_detected']}")
        print(f"  Failed videos: {len(training_summary['videos_failed'])}")

        return training_summary

    def _process_single_video(self, video_path: str, label: str) -> Optional[Dict]:
        """
        Process a single video and extract stroke data

        Args:
            video_path: Path to video file
            label: Label for this video

        Returns:
            Dictionary containing extracted video data
        """
        try:
            # Import required modules
            from FrameExtractor import FrameExtractor
            from BackgroundSubtractor import BackgroundSubtractor
            from StrokeDetector import StrokeDetector
            from JointTracker import JointTracker

            # Create temporary directory for frames
            temp_dir = os.path.join(self.output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)

            # Extract frames
            extractor = FrameExtractor(video_path, temp_dir)
            extractor.extract_frames()

            # Initialize components
            bg_subtractor = BackgroundSubtractor(shape=None, alpha=0.003, history_size=40)
            stroke_detector = StrokeDetector()
            joint_tracker = JointTracker(window_size=5)

            # Get frame paths
            frame_paths = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

            if not frame_paths:
                print(f"[Error] No frames extracted from {video_path}")
                return None

            # Build background model
            print("Building background model...")
            for i, frame_path in enumerate(frame_paths[:30]):
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    bg_subtractor.update(frame)
                    if i % 10 == 0:
                        print(f"  Background frame {i + 1}/30")

            # Process all frames
            print("Extracting joint data and stroke patterns...")
            all_joint_data = []
            all_stroke_data = []

            for frame_idx, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    continue

                try:
                    # Get foreground mask
                    fg_mask = bg_subtractor.get_cleaned_mask(frame, threshold=35, min_area=800)

                    # Find connected components
                    components = bg_subtractor.connected_components(fg_mask, connectivity=8)
                    components = bg_subtractor.merge_nearby_components(components, distance_threshold=120)

                    # Process each person detected
                    for label_id, points in components.items():
                        if len(points) < 200:
                            continue

                        # Extract joint positions
                        joints = bg_subtractor.extract_rowing_joints(points, frame.shape)

                        # Track and smooth joints
                        smoothed_joints = joint_tracker.update(joints, frame_idx)

                        # Analyze stroke
                        stroke_phase, joint_angles = stroke_detector.update(frame, smoothed_joints, frame_idx)

                        # Store data
                        joint_data = {
                            'frame_id': frame_idx,
                            'person_id': label_id,
                            'joints': smoothed_joints,
                            'joint_angles': joint_angles,
                            'stroke_phase': stroke_phase
                        }
                        all_joint_data.append(joint_data)

                except Exception as e:
                    print(f"[Warning] Error processing frame {frame_idx}: {e}")
                    continue

                if frame_idx % 50 == 0:
                    print(f"  Processed frame {frame_idx}/{len(frame_paths)}")

            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            # Analyze stroke cycles
            stroke_cycles = self._detect_stroke_cycles(all_joint_data)

            # Create video data summary
            video_data = {
                'video_path': video_path,
                'label': label,
                'total_frames': len(frame_paths),
                'joint_data': all_joint_data,
                'stroke_cycles': len(stroke_cycles),
                'cycle_data': stroke_cycles,
                'stroke_metrics': stroke_detector.get_stroke_metrics()
            }

            # Export individual video data
            self._export_video_data(video_data)

            return video_data

        except Exception as e:
            print(f"[Error] Failed to process video {video_path}: {e}")
            return None

    def _detect_stroke_cycles(self, joint_data: List[Dict]) -> List[Dict]:
        """
        Detect complete stroke cycles from joint data

        Args:
            joint_data: List of frame data with joints and angles

        Returns:
            List of detected stroke cycles
        """
        if not joint_data:
            return []

        cycles = []
        current_cycle = []

        # Extract torso angles
        torso_angles = []
        for data in joint_data:
            if 'joint_angles' in data and 'torso_angle' in data['joint_angles']:
                torso_angles.append(data['joint_angles']['torso_angle'])
            else:
                torso_angles.append(0.0)

        if len(torso_angles) < 10:
            return cycles

        # Find peaks and valleys in torso angle (stroke cycle indicators)
        peaks = self._find_peaks(torso_angles, min_distance=20)
        valleys = self._find_valleys(torso_angles, min_distance=20)

        # Combine and sort turning points
        turning_points = [(idx, 'peak') for idx in peaks] + [(idx, 'valley') for idx in valleys]
        turning_points.sort(key=lambda x: x[0])

        # Extract cycles (peak to peak or valley to valley)
        cycle_starts = []
        for i, (idx, point_type) in enumerate(turning_points):
            if point_type == 'peak':  # Start cycles at forward lean peaks
                cycle_starts.append(idx)

        # Create cycles
        for i in range(len(cycle_starts) - 1):
            start_idx = cycle_starts[i]
            end_idx = cycle_starts[i + 1]

            if end_idx - start_idx > 10:  # Minimum cycle length
                cycle_data = joint_data[start_idx:end_idx]

                cycle_info = {
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'duration': end_idx - start_idx,
                    'data': cycle_data,
                    'angle_profile': self._extract_angle_profile(cycle_data)
                }
                cycles.append(cycle_info)

        return cycles

    def _find_peaks(self, data: List[float], min_distance: int = 10) -> List[int]:
        """Find peaks in data"""
        data = np.array(data)
        peaks = []

        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                # Check minimum distance constraint
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)

        return peaks

    def _find_valleys(self, data: List[float], min_distance: int = 10) -> List[int]:
        """Find valleys in data"""
        data = np.array(data)
        valleys = []

        for i in range(1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                # Check minimum distance constraint
                if not valleys or i - valleys[-1] >= min_distance:
                    valleys.append(i)

        return valleys

    def _extract_angle_profile(self, cycle_data: List[Dict]) -> Dict[str, List[float]]:
        """Extract angle profiles for a stroke cycle"""
        profile = {angle: [] for angle in self.angle_features}

        for frame_data in cycle_data:
            if 'joint_angles' in frame_data:
                for angle in self.angle_features:
                    value = frame_data['joint_angles'].get(angle, 0.0)
                    profile[angle].append(value)
            else:
                for angle in self.angle_features:
                    profile[angle].append(0.0)

        return profile

    def _build_stroke_templates(self):
        """Build stroke templates from training data"""
        print("\n[Info] Building stroke templates...")

        # Collect all cycles by label
        cycles_by_label = {}
        for video_data in self.training_data:
            label = video_data['label']
            if label not in cycles_by_label:
                cycles_by_label[label] = []
            cycles_by_label[label].extend(video_data['cycle_data'])

        # Create templates for each label
        for label, cycles in cycles_by_label.items():
            if not cycles:
                continue

            print(f"  Creating template for '{label}' from {len(cycles)} cycles")

            # Normalize cycle lengths and average
            template = self._create_average_template(cycles)
            self.stroke_templates[label] = template

    def _create_average_template(self, cycles: List[Dict], target_length: int = 50) -> Dict:
        """Create average template from multiple stroke cycles"""
        if not cycles:
            return {}

        # Resample all cycles to same length
        resampled_profiles = []
        for cycle in cycles:
            if 'angle_profile' in cycle and cycle['duration'] > 5:
                resampled = self._resample_profile(cycle['angle_profile'], target_length)
                resampled_profiles.append(resampled)

        if not resampled_profiles:
            return {}

        # Average across all cycles
        template = {}
        for angle in self.angle_features:
            angle_data = []
            for profile in resampled_profiles:
                if angle in profile and len(profile[angle]) == target_length:
                    angle_data.append(profile[angle])

            if angle_data:
                template[angle] = {
                    'mean': np.mean(angle_data, axis=0).tolist(),
                    'std': np.std(angle_data, axis=0).tolist(),
                    'min': np.min(angle_data, axis=0).tolist(),
                    'max': np.max(angle_data, axis=0).tolist()
                }

        return template

    def _resample_profile(self, profile: Dict[str, List[float]], target_length: int) -> Dict[str, List[float]]:
        """Resample angle profile to target length"""
        resampled = {}

        for angle, values in profile.items():
            if len(values) < 2:
                resampled[angle] = [0.0] * target_length
                continue

            # Linear interpolation to target length
            original_indices = np.linspace(0, len(values) - 1, len(values))
            target_indices = np.linspace(0, len(values) - 1, target_length)
            resampled_values = np.interp(target_indices, original_indices, values)
            resampled[angle] = resampled_values.tolist()

        return resampled

    def _export_video_data(self, video_data: Dict):
        """Export individual video analysis data"""
        video_name = os.path.splitext(os.path.basename(video_data['video_path']))[0]

        # Export joint tracking data
        joint_csv_path = os.path.join(self.output_dir, f"{video_name}_joints.csv")
        with open(joint_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_id', 'person_id', 'stroke_phase'] + \
                         [f'{joint}_{coord}' for joint in [
                             'head', 'neck', 'shoulder_left', 'shoulder_right',
                             'elbow_left', 'elbow_right', 'wrist_left', 'wrist_right',
                             'spine_mid', 'hips', 'knee_left', 'knee_right',
                             'ankle_left', 'ankle_right'
                         ] for coord in ['y', 'x']] + \
                         list(self.angle_features)

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in video_data['joint_data']:
                row = {
                    'frame_id': data['frame_id'],
                    'person_id': data['person_id'],
                    'stroke_phase': data['stroke_phase']
                }

                # Add joint positions
                for joint, pos in data['joints'].items():
                    row[f'{joint}_y'] = pos[0] if pos != (0, 0) else 0
                    row[f'{joint}_x'] = pos[1] if pos != (0, 0) else 0

                # Add joint angles
                for angle, value in data['joint_angles'].items():
                    if angle in self.angle_features:
                        row[angle] = value

                writer.writerow(row)

        print(f"[Info] Exported joint data to {joint_csv_path}")

    def _save_training_results(self):
        """Save training results and templates"""
        # Save stroke templates
        template_path = os.path.join(self.output_dir, "stroke_templates.csv")
        with open(template_path, 'w', newline='') as csvfile:
            fieldnames = ['template_label', 'angle_type', 'time_step'] + \
                         ['mean', 'std', 'min', 'max']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for label, template in self.stroke_templates.items():
                for angle, stats in template.items():
                    for i in range(len(stats['mean'])):
                        row = {
                            'template_label': label,
                            'angle_type': angle,
                            'time_step': i,
                            'mean': stats['mean'][i],
                            'std': stats['std'][i],
                            'min': stats['min'][i],
                            'max': stats['max'][i]
                        }
                        writer.writerow(row)

        print(f"[Info] Saved stroke templates to {template_path}")

        # Save training summary
        summary_path = os.path.join(self.output_dir, "training_summary.csv")
        with open(summary_path, 'w', newline='') as csvfile:
            fieldnames = ['video_path', 'label', 'total_frames', 'stroke_cycles',
                          'avg_torso_range', 'avg_knee_range']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for video_data in self.training_data:
                metrics = video_data.get('stroke_metrics', {})
                row = {
                    'video_path': video_data['video_path'],
                    'label': video_data['label'],
                    'total_frames': video_data['total_frames'],
                    'stroke_cycles': video_data['stroke_cycles'],
                    'avg_torso_range': metrics.get('torso_angle_range', 0.0),
                    'avg_knee_range': metrics.get('knee_angle_range', 0.0)
                }
                writer.writerow(row)

        print(f"[Info] Saved training summary to {summary_path}")

    def evaluate_video(self, video_path: str, comparison_label: str = None) -> Dict:
        """
        Evaluate a new video against trained templates

        Args:
            video_path: Path to video to evaluate
            comparison_label: Template label to compare against (if None, compare to all)

        Returns:
            Dictionary containing evaluation results
        """
        if not self.stroke_templates:
            print("[Error] No trained templates available. Run train_on_videos() first.")
            return {}

        print(f"\n[Info] Evaluating video: {os.path.basename(video_path)}")

        # Process the video
        video_data = self._process_single_video(video_path, "evaluation")

        if not video_data:
            print("[Error] Failed to process evaluation video")
            return {}

        # Compare against templates
        evaluation_results = {}

        templates_to_compare = [comparison_label] if comparison_label else list(self.stroke_templates.keys())

        for template_label in templates_to_compare:
            if template_label not in self.stroke_templates:
                continue

            print(f"  Comparing against template: {template_label}")

            # Calculate similarity scores
            similarity_scores = self._calculate_template_similarity(
                video_data['cycle_data'],
                self.stroke_templates[template_label]
            )

            evaluation_results[template_label] = similarity_scores

        # Save evaluation results
        self._save_evaluation_results(video_data, evaluation_results)

        return evaluation_results

    def _calculate_template_similarity(self, cycles: List[Dict], template: Dict) -> Dict:
        """Calculate similarity between video cycles and template"""
        if not cycles or not template:
            return {'error': 'No data to compare'}

        similarities = []

        for cycle in cycles:
            if 'angle_profile' not in cycle:
                continue

            # Resample cycle to template length
            template_length = len(next(iter(template.values()))['mean'])
            resampled_cycle = self._resample_profile(cycle['angle_profile'], template_length)

            # Calculate similarity for each angle
            cycle_similarity = {}
            for angle in self.angle_features:
                if angle in template and angle in resampled_cycle:
                    template_mean = np.array(template[angle]['mean'])
                    cycle_values = np.array(resampled_cycle[angle])

                    # Calculate correlation and RMSE
                    correlation = np.corrcoef(template_mean, cycle_values)[0, 1]
                    rmse = np.sqrt(np.mean((template_mean - cycle_values) ** 2))

                    cycle_similarity[angle] = {
                        'correlation': correlation if not np.isnan(correlation) else 0.0,
                        'rmse': rmse
                    }

            similarities.append(cycle_similarity)

        # Average across all cycles
        avg_similarity = {}
        for angle in self.angle_features:
            correlations = [s[angle]['correlation'] for s in similarities if angle in s]
            rmses = [s[angle]['rmse'] for s in similarities if angle in s]

            if correlations:
                avg_similarity[angle] = {
                    'avg_correlation': np.mean(correlations),
                    'avg_rmse': np.mean(rmses),
                    'std_correlation': np.std(correlations)
                }

        # Overall similarity score
        overall_correlation = np.mean([
            avg_similarity[angle]['avg_correlation']
            for angle in avg_similarity if 'avg_correlation' in avg_similarity[angle]
        ])

        return {
            'overall_similarity': overall_correlation,
            'angle_similarities': avg_similarity,
            'cycles_analyzed': len(similarities)
        }

    def _save_evaluation_results(self, video_data: Dict, evaluation_results: Dict):
        """Save evaluation results to CSV"""
        video_name = os.path.splitext(os.path.basename(video_data['video_path']))[0]
        results_path = os.path.join(self.output_dir, f"{video_name}_evaluation.csv")

        with open(results_path, 'w', newline='') as csvfile:
            fieldnames = ['template_label', 'overall_similarity', 'cycles_analyzed'] + \
                         [f'{angle}_{metric}' for angle in self.angle_features
                          for metric in ['correlation', 'rmse', 'std_correlation']]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for template_label, results in evaluation_results.items():
                if 'error' in results:
                    continue

                row = {
                    'template_label': template_label,
                    'overall_similarity': results.get('overall_similarity', 0.0),
                    'cycles_analyzed': results.get('cycles_analyzed', 0)
                }

                # Add angle-specific metrics
                for angle in self.angle_features:
                    if angle in results.get('angle_similarities', {}):
                        angle_data = results['angle_similarities'][angle]
                        row[f'{angle}_correlation'] = angle_data.get('avg_correlation', 0.0)
                        row[f'{angle}_rmse'] = angle_data.get('avg_rmse', 0.0)
                        row[f'{angle}_std_correlation'] = angle_data.get('std_correlation', 0.0)
                    else:
                        row[f'{angle}_correlation'] = 0.0
                        row[f'{angle}_rmse'] = 0.0
                        row[f'{angle}_std_correlation'] = 0.0

                writer.writerow(row)

        print(f"[Info] Saved evaluation results to {results_path}")


# Example usage and main training script
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RowingStrokeAnalyzer("rowing_analysis_output")

    # Example training videos (replace with actual paths)
    training_videos = [
        "path/to/expert_rower_1.mp4",
        "path/to/expert_rower_2.mp4",
        "path/to/beginner_rower_1.mp4",
        "path/to/beginner_rower_2.mp4"
    ]

    # Labels for training videos
    training_labels = [
        "expert",
        "expert",
        "beginner",
        "beginner"
    ]

    # Train on videos
    print("Starting training phase...")
    training_summary = analyzer.train_on_videos(training_videos, training_labels)

    # Evaluate a new video
    print("\nStarting evaluation phase...")
    evaluation_video = "path/to/new_rower.mp4"
    evaluation_results = analyzer.evaluate_video(evaluation_video)

    print("\nAnalysis complete! Check the output directory for results.")