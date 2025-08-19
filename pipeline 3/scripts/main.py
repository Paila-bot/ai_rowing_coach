import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from deap import base, creator, tools, algorithms
from mediapipe.python.solutions import pose as mp_pose
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models


# ================ 1. Video Processing & Feature Extraction ================
class VideoProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.joint_mapping = {
            'LEFT_SHOULDER': 'shoulder',
            'RIGHT_SHOULDER': 'shoulder',
            'LEFT_ELBOW': 'elbow',
            'RIGHT_ELBOW': 'elbow',
            'LEFT_WRIST': 'wrist',
            'RIGHT_WRIST': 'wrist',
            'LEFT_HIP': 'hip',
            'RIGHT_HIP': 'hip',
            'LEFT_KNEE': 'knee',
            'RIGHT_KNEE': 'knee',
            'LEFT_ANKLE': 'ankle',
            'RIGHT_ANKLE': 'ankle'
        }

    def process_video(self, video_path):
        """Process video and extract joint positions"""
        cap = cv2.VideoCapture(video_path)
        joint_data = []
        frame_count = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Process frame with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                frame_joints = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp_pose.PoseLandmark(idx).name
                    if landmark_name in self.joint_mapping:
                        prefix = self.joint_mapping[landmark_name]
                        frame_joints[f'{prefix}_x'] = landmark.x
                        frame_joints[f'{prefix}_y'] = landmark.y
                joint_data.append(frame_joints)

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        return pd.DataFrame(joint_data)

    @staticmethod
    def calculate_angles(joints):
        """Compute biomechanical joint angles"""
        angles = pd.DataFrame()

        # Shoulder-Elbow-Wrist angle (arm bend)
        angles['elbow'] = np.degrees(
            np.arctan2(
                joints['wrist_y'] - joints['elbow_y'],
                joints['wrist_x'] - joints['elbow_x']
            ) - np.arctan2(
                joints['shoulder_y'] - joints['elbow_y'],
                joints['shoulder_x'] - joints['elbow_x']
            )
        )

        # Hip-Knee-Ankle angle (leg drive)
        angles['knee'] = np.degrees(
            np.arctan2(
                joints['ankle_y'] - joints['knee_y'],
                joints['ankle_x'] - joints['knee_x']
            ) - np.arctan2(
                joints['hip_y'] - joints['knee_y'],
                joints['hip_x'] - joints['knee_x']
            )
        )

        # Shoulder-Hip-Knee angle (back posture)
        angles['back'] = np.degrees(
            np.arctan2(
                joints['hip_y'] - joints['shoulder_y'],
                joints['hip_x'] - joints['shoulder_x']
            ) - np.arctan2(
                joints['knee_y'] - joints['hip_y'],
                joints['knee_x'] - joints['hip_x']
            )
        )

        return angles

    def detect_stroke_phases(self, angles, min_cycle_length=20):
        """Identify stroke phases using biomechanical heuristics"""
        phases = []
        current_phase = "recovery"
        phase_counter = 0

        for i in range(1, len(angles)):
            # Catch: Knee flexion increases and back angle forward
            if (angles['knee'].iloc[i] > angles['knee'].iloc[i - 1] and
                    angles['back'].iloc[i] < 10 and
                    phase_counter > min_cycle_length):
                current_phase = "catch"
                phase_counter = 0

            # Drive: Rapid back angle change
            elif current_phase == "catch" and abs(angles['back'].iloc[i] - angles['back'].iloc[i - 1]) > 0.5:
                current_phase = "drive"

            # Finish: Elbow flexion peaks
            elif current_phase == "drive" and angles['elbow'].iloc[i] < 90:
                current_phase = "finish"

            # Recovery: Transition back to catch
            elif current_phase == "finish" and phase_counter > 10:
                current_phase = "recovery"

            phases.append(current_phase)
            phase_counter += 1

        # Add first phase
        phases.insert(0, current_phase)
        return phases

    def preprocess_video(self, video_path, window_size=45):
        """Full processing pipeline for a video"""
        print(f"Processing {os.path.basename(video_path)}...")
        joints = self.process_video(video_path)

        if len(joints) < window_size:
            print(f"Video too short ({len(joints)} frames). Skipping.")
            return []

        angles = self.calculate_angles(joints)
        velocities = angles.diff().fillna(0)
        phases = self.detect_stroke_phases(angles)

        # Create feature matrix
        features = pd.concat([
            angles.add_prefix('angle_'),
            velocities.add_prefix('vel_'),
            pd.get_dummies(phases, prefix='phase')
        ], axis=1)

        # Segment into stroke windows
        strokes = []
        for i in range(0, len(features) - window_size + 1, window_size // 2):
            stroke = features.iloc[i:i + window_size].copy()
            if 'phase_catch' in stroke.columns and stroke['phase_catch'].sum() > 0:
                strokes.append(stroke.values)

        print(f"Extracted {len(strokes)} strokes from video")
        return strokes


# ================ 2. Hybrid Model Architecture ================
class BiomechanicsEncoder(tf.keras.Model):
    """LSTM-based encoder with attention for biomechanical patterns"""

    def __init__(self, input_shape, latent_dim=16):
        super().__init__()
        self.lstm1 = layers.LSTM(64, return_sequences=True, input_shape=input_shape)
        self.attention = layers.Attention()
        self.lstm2 = layers.LSTM(32)
        self.dense = layers.Dense(latent_dim)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        x = self.lstm1(inputs)
        context = self.attention([x, x])
        return self.dense(self.lstm2(context))

    def train_step(self, data):
        # Contrastive learning: Augment good strokes
        anchor = data
        positive = anchor + tf.random.normal(tf.shape(anchor), 0, 0.05)

        with tf.GradientTape() as tape:
            anchor_embed = self(anchor)
            positive_embed = self(positive)

            # Distance metric
            distance = tf.reduce_mean(tf.square(anchor_embed - positive_embed), axis=1)
            loss = tf.reduce_mean(distance)  # Minimize distance between variations

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# ================ Evolutionary Algorithm Components ================
def evaluate_rule(individual, encoder, good_strokes):
    """Evaluate EA rule fitness on biomechanical parameters"""
    # Decode thresholds from individual [elbow_min, elbow_max, knee_min, ...]
    thresholds = np.array(individual[:6])

    # Apply thresholds to good strokes
    violations = 0
    for stroke in good_strokes:
        # Check joint angle rules
        if (np.min(stroke[:, 0]) < thresholds[0] or  # Elbow min
                np.max(stroke[:, 0]) > thresholds[1] or  # Elbow max
                np.min(stroke[:, 1]) < thresholds[2] or  # Knee min
                np.max(stroke[:, 1]) > thresholds[3] or  # Knee max
                np.min(stroke[:, 2]) < thresholds[4] or  # Back min
                np.max(stroke[:, 2]) > thresholds[5]):  # Back max
            violations += 1

    # Fitness: Minimize false positives on good strokes
    return (violations / len(good_strokes)),


def evolve_biomechanical_rules(encoder, good_strokes, ngen=15):
    """Evolve biomechanical thresholds using genetic algorithm"""
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 180)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=6)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_rule, encoder=encoder, good_strokes=good_strokes)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=15, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=ngen,
        stats=stats, halloffame=hof, verbose=True
    )

    return hof[0]


# ================ 3. Training Pipeline ================
class RowingCoachTrainer:
    def __init__(self, video_dir, output_dir="model_output"):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.processor = VideoProcessor()
        os.makedirs(output_dir, exist_ok=True)

    def load_training_data(self):
        """Process all videos in directory and extract strokes"""
        all_strokes = []
        video_files = [f for f in os.listdir(self.video_dir)
                       if f.endswith(('.mp4', '.avi', '.mov'))]

        print(f"Found {len(video_files)} videos for training")

        for video_file in video_files:
            video_path = os.path.join(self.video_dir, video_file)
            strokes = self.processor.preprocess_video(video_path)
            all_strokes.extend(strokes)

        print(f"Total strokes extracted: {len(all_strokes)}")
        return np.array(all_strokes)

    def train_hybrid_model(self, strokes):
        """Train hybrid DL+EA model using good strokes"""
        # Normalize data
        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack(strokes))
        scaled_strokes = np.array([self.scaler.transform(s) for s in strokes])

        # Train deep learning encoder
        input_shape = (scaled_strokes[0].shape[0], scaled_strokes[0].shape[1])
        self.encoder = BiomechanicsEncoder(input_shape)
        self.encoder.compile(optimizer=tf.keras.optimizers.Adam(0.001))

        print("Training biomechanics encoder...")
        history = self.encoder.fit(
            scaled_strokes,
            epochs=100,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )

        # Generate embeddings for good strokes
        print("Generating reference embeddings...")
        good_embeddings = np.array([self.encoder.predict(np.expand_dims(s, 0))[0]
                                    for s in scaled_strokes])

        # Create reference distributions
        self.mean_good = np.mean(good_embeddings, axis=0)
        self.cov_good = np.cov(good_embeddings.T) + 1e-6 * np.eye(good_embeddings.shape[1])

        # Train EA component
        print("Evolving biomechanical rules...")
        self.thresholds = evolve_biomechanical_rules(self.encoder, scaled_strokes)

        # Save model components
        self.save_model()
        return history

    def save_model(self):
        """Save all model components"""
        # Save encoder
        self.encoder.save(os.path.join(self.output_dir, "biomechanics_encoder.h5"))

        # Save scaler
        np.savez(
            os.path.join(self.output_dir, "scaler.npz"),
            mean=self.scaler.mean_,
            scale=self.scaler.scale_
        )

        # Save reference distribution and thresholds
        np.savez(
            os.path.join(self.output_dir, "reference.npz"),
            mean_good=self.mean_good,
            cov_good=self.cov_good,
            thresholds=self.thresholds
        )
        print(f"Model saved to {self.output_dir}")


# ================ 4. Inference & Feedback System ================
class RowingCoach:
    def __init__(self, model_dir):
        self.load_model(model_dir)
        self.processor = VideoProcessor()

    def load_model(self, model_dir):
        """Load trained model components"""
        # Load encoder
        self.encoder = tf.keras.models.load_model(
            os.path.join(model_dir, "biomechanics_encoder.h5"),
            custom_objects={"BiomechanicsEncoder": BiomechanicsEncoder}
        )

        # Load scaler
        scaler_data = np.load(os.path.join(model_dir, "scaler.npz"))
        self.scaler = StandardScaler()
        self.scaler.mean_ = scaler_data['mean']
        self.scaler.scale_ = scaler_data['scale']

        # Load reference distribution and thresholds
        ref_data = np.load(os.path.join(model_dir, "reference.npz"))
        self.mean_good = ref_data['mean_good']
        self.cov_good = ref_data['cov_good']
        self.thresholds = ref_data['thresholds']

    def analyze_video(self, video_path):
        """Full analysis pipeline for a new video"""
        # Process video and extract strokes
        strokes = self.processor.preprocess_video(video_path)
        if len(strokes) == 0:
            return {"error": "No valid strokes detected"}

        results = []
        for i, stroke in enumerate(strokes):
            # Preprocess stroke
            scaled_stroke = self.scaler.transform(stroke)
            stroke_tensor = np.expand_dims(scaled_stroke, 0)

            # Get embedding
            embedding = self.encoder.predict(stroke_tensor, verbose=0)[0]

            # Calculate deviation metrics
            try:
                mahalanobis_dist = mahalanobis(embedding, self.mean_good,
                                               np.linalg.inv(self.cov_good))
            except:
                mahalanobis_dist = 100  # Large error if calculation fails

            # Detect phase-specific errors
            feedback = self.generate_feedback(stroke)

            # Compile results
            results.append({
                "stroke_id": i + 1,
                "score": max(0, 10 - min(mahalanobis_dist, 10)),
                "deviation": float(mahalanobis_dist),
                "feedback": feedback
            })

        return results

    def generate_feedback(self, stroke):
        """Generate coach-like feedback based on biomechanical analysis"""
        feedback = []
        stroke_df = pd.DataFrame(stroke, columns=[
            'angle_elbow', 'angle_knee', 'angle_back',
            'vel_elbow', 'vel_knee', 'vel_back',
            'phase_catch', 'phase_drive', 'phase_finish', 'phase_recovery'
        ])

        # EA threshold violations
        thresholds = self.thresholds
        if np.min(stroke_df['angle_elbow']) < thresholds[0]:
            feedback.append("Elbow angle too small during drive phase")
        if np.max(stroke_df['angle_elbow']) > thresholds[1]:
            feedback.append("Elbow over-extension during recovery")
        if np.min(stroke_df['angle_knee']) < thresholds[2]:
            feedback.append("Insufficient knee bend at catch position")
        if np.max(stroke_df['angle_knee']) > thresholds[3]:
            feedback.append("Over-extension during leg drive")
        if np.min(stroke_df['angle_back']) < thresholds[4]:
            feedback.append("Excessive forward lean at catch")
        if np.max(stroke_df['angle_back']) > thresholds[5]:
            feedback.append("Excessive backward lean at finish")

        # Phase-specific analysis
        if 'phase_catch' in stroke_df.columns:
            catch_frames = stroke_df[stroke_df['phase_catch'] > 0.5]
            if len(catch_frames) > 0:
                catch_idx = catch_frames.index[0]
                if stroke_df.loc[catch_idx, 'angle_knee'] > 45:
                    feedback.append("Insufficient knee bend at catch")
                if abs(stroke_df.loc[catch_idx, 'angle_back']) > 15:
                    feedback.append("Improper back angle at catch")

        if 'phase_drive' in stroke_df.columns:
            drive_frames = stroke_df[stroke_df['phase_drive'] > 0.5]
            if len(drive_frames) > 0:
                elbow_vel = np.mean(drive_frames['vel_elbow'])
                if elbow_vel > 0.5:
                    feedback.append("Arms pulling too early during drive")

        if 'phase_recovery' in stroke_df.columns:
            recovery_frames = stroke_df[stroke_df['phase_recovery'] > 0.5]
            if len(recovery_frames) > 0:
                avg_knee_angle = np.mean(recovery_frames['angle_knee'])
                if avg_knee_angle < 100:
                    feedback.append("Insufficient leg recovery extension")

        # Add positive reinforcement if no errors
        if not feedback:
            feedback.append("Excellent form! Maintain this technique")

        return feedback


# ================ Main Execution ================
if __name__ == "__main__":
    # Configuration
    TRAIN_VIDEO_DIR = "rowing_videos/good_technique"
    MODEL_OUTPUT_DIR = "trained_model"
    TEST_VIDEO = "test_video.mp4"

    # Step 1: Train the model
    print("===== TRAINING PHASE =====")
    trainer = RowingCoachTrainer(TRAIN_VIDEO_DIR, MODEL_OUTPUT_DIR)
    training_strokes = trainer.load_training_data()

    if len(training_strokes) > 0:
        train_hybrid_model(training_strokes)
        print("Training completed successfully!")
    else:
        print("Insufficient training data. Add more videos.")
        exit()

    # Step 2: Analyze new video
    print("\n===== ANALYSIS PHASE =====")
    coach = RowingCoach(MODEL_OUTPUT_DIR)
    analysis_results = coach.analyze_video(TEST_VIDEO)

    # Display results
    print("\nStroke Analysis Report:")
    for result in analysis_results:
        print(f"\nStroke #{result['stroke_id']} - Score: {result['score']:.1f}/10")
        print("Feedback:")
        for i, feedback in enumerate(result['feedback'], 1):
            print(f"{i}. {feedback}")