import os

import joblib

import RowingCoach
import FeedbackGenerator
import FeatureExtractor
# 1. INITIALIZE AND TRAIN
coach = RowingCoach()

# Process directory AND train (now combined)
try:
    print("Processing training videos...")
    video_paths = [
        os.path.join("/content/drive/My Drive/Rowing Dataset/data/Rowing Dataset", f)
        for f in os.listdir("/content/drive/My Drive/Rowing Dataset/data/Rowing Dataset")
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]

    if not video_paths:
        raise ValueError("No video files found in training directory")

    # This now automatically trains on processed data
    coach.train(video_paths)
    coach.save_models("/content/drive/My Drive/Rowing Dataset/Pipeline 3 models")
    print("Training completed and models saved successfully")

except Exception as e:
    print(f"Training failed: {str(e)}")
    # Handle error (e.g., skip to loading pre-trained if available)

# 2. LOAD AND ANALYZE (alternative if training fails)
try:
    print("\nLoading pre-trained models...")
    pretrained_coach = RowingCoach.load_models("/content/drive/My Drive/Rowing Dataset/Pipeline 3 models")

    test_video = "/content/drive/My Drive/Rowing Dataset/data/Test Data/3.mp4"
    if not os.path.exists(test_video):
        raise FileNotFoundError(f"Test video not found at {test_video}")

    print("Analyzing test video...")
    results = pretrained_coach.analyze(test_video)

    # 3. GENERATE ENHANCED FEEDBACK
    if results:
        # Initialize feedback generator (try to load if exists)
        feedback_gen_path = "/content/drive/My Drive/Rowing Dataset/Pipeline 3 models/feedback_generator.pkl"
        try:
            feedback_gen = joblib.load(feedback_gen_path)
            print("✅ Loaded existing feedback generator with learning history")
        except:
            feedback_gen = FeedbackGenerator()
            print("✅ Created new feedback generator")

        # Generate comprehensive report
        report = feedback_gen.generate_feedback(results)

        # Print the enhanced report
        print("\n" + "="*50)
        print(f"{'🏆 EXPERT ROWING ANALYSIS REPORT 🏆':^50}")
        print("="*50)

        print(f"\n📊 SUMMARY (Score: {report['score']}/100):")
        print(report['summary'])

        print("\n🔍 TECHNICAL ANALYSIS:")
        for analysis in report['technical_analysis']:
            print(f"\n{analysis['phase'].upper()} PHASE:")
            print(f"• Main issue: {analysis['issue']}")
            print(f"• Most problematic frames: {', '.join(map(str, analysis['frames'][:3]))}")
            print("• Ideal technique:")
            for point in analysis['ideal_technique']:
                print(f"  - {point}")

        print("\n💡 RECOMMENDED DRILLS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

        # Save the updated feedback generator
        joblib.dump(feedback_gen, feedback_gen_path)

        # Keep original technical output for reference
        print("\n" + "-"*50)
        print("🧾 DETAILED TECHNICAL OUTPUT:")
        print(f"- Detected {len(results['critical_frames'])} technique anomalies")
        print(f"- Average reconstruction error: {np.mean(results['anomaly_scores']):.2f}")

        print("\nCritical Frames:")
        for frame in results['critical_frames']:
            phase_name = {
                0: "Catch",
                1: "Drive",
                2: "Finish",
                3: "Recovery"
            }.get(results['phases'][frame], "Unknown")

            print(f"Frame {frame:4d} | {phase_name:8s} | Error: {results['anomaly_scores'][frame]:.2f}")

    else:
        print("No valid poses detected in test video")

except Exception as e:
    print(f"Analysis failed: {str(e)}")