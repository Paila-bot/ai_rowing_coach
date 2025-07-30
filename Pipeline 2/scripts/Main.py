import os
import RowingAnalysisSystem as RAS

def main():
    """Main function to demonstrate the rowing analysis system"""
    print("=== Rowing Technique Analysis System ===")
    print("Using only OpenCV and NumPy for pose estimation and analysis\n")

    # Initialize the system
    system = RAS.RowingAnalysisSystem()

    # Configuration
    TRAIN_DIR = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Rowing Dataset"  # Directory with good technique videos
    TEST_DIR = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\data\Test Data"  # Directory with test videos
    MODEL_PATH = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 2\models\model.pkl"

    # Example usage - modify paths as needed
    try:
        # Training phase
        if os.path.exists(TRAIN_DIR):
            print("1. Training the system...")
            num_features = system.train_system(TRAIN_DIR, max_videos=10)
            print(f"Training completed with {num_features} feature samples")

            # Save the trained model
            system.save_model(MODEL_PATH)

        # If no training directory, try to load existing model
        elif os.path.exists(MODEL_PATH):
            print("1. Loading existing model...")
            system.load_model(MODEL_PATH)

        else:
            print("ERROR: No training data or existing model found!")
            print(f"Please create '{TRAIN_DIR}' directory with video files or provide '{MODEL_PATH}'")
            return

        # Testing/Evaluation phase
        if os.path.exists(TEST_DIR):
            print("\n2. Evaluating system performance...")
            evaluation = system.evaluate_system(TEST_DIR, max_videos=5)

            # Print detailed results
            print(f"\nDetailed Results:")
            for result in evaluation['individual_results']:
                print(f"  {os.path.basename(result['video_path'])}: {result['overall_score']:.1f}")
                for feedback in result['detailed_feedback']:
                    print(f"    - {feedback}")

        # Single video analysis example
        sample_video = None
        if os.path.exists(TEST_DIR):
            for file in os.listdir(TEST_DIR):
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    sample_video = os.path.join(TEST_DIR, file)
                    break

        if sample_video:
            print(f"\n3. Analyzing single video: {os.path.basename(sample_video)}")
            analysis = system.analyze_video(sample_video)

            print(f"Overall Score: {analysis['overall_score']:.1f}/100")
            print(f"Consistency Score: {analysis['consistency_score']:.3f}")
            print("Feedback:")
            for feedback in analysis['detailed_feedback']:
                print(f"  - {feedback}")

            # Visualize pose estimation (uncomment to enable)
            # print("\n4. Generating pose visualization...")
            # system.visualize_pose(sample_video, "pose_visualization.mp4", max_frames=50)

        print("\n=== Analysis Complete ===")

    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Video files in the specified directories")
        print("2. OpenCV installed (pip install opencv-python)")
        print("3. NumPy installed (pip install numpy)")


if __name__ == "__main__":
    main()