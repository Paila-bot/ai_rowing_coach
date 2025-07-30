import cv2
import os
import json
import AutomatedJointTracker as jt
import AutomatedRowingAnalyzer as ra

def main():
    """Main function demonstrating the complete automated pipeline"""
    video_path = r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 1\data\videos\1.mp4"

    if not os.path.exists(video_path):
        print("❌ Video file does not exist.")
        print("Please update the video_path variable with the correct path to your video file.")
        return

    print("=== FULLY AUTOMATED ROWING COACH ===")
    print("This system requires NO manual input!")
    print("Using only OpenCV and NumPy for complete automation.\n")

    try:
        # Step 1: Automated joint tracking
        print("STEP 1: Automatic Joint Tracking")
        print("-" * 40)
        tracker = jt.AutomatedJointTracker(video_path)
        joint_positions = tracker.run()

        print(f"\n✅ Tracked {len(joint_positions)} frames automatically")

        # Step 2: Automated analysis
        print("\nSTEP 2: Automated Technique Analysis")
        print("-" * 40)
        analyzer = ra.AutomatedRowingAnalyzer()
        strokes, feedback = analyzer.run_analysis()

        print(f"\n✅ Analyzed {len(strokes)} strokes automatically")

        print("\n" + "=" * 60)
        print("🎉 FULLY AUTOMATED ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  📊 automatic_joint_positions.csv - Joint tracking data")
        print("  📊 automatic_joint_positions.json - Joint tracking data (JSON)")
        print("  🎥 tracking_visualization.mp4 - Visual tracking overlay")
        print("  📝 automated_analysis_report.txt - Detailed technique report")
        print("  📊 automated_stroke_analysis.json - Stroke analysis data")

        print("\n💡 This system eliminates ALL manual work!")
        print("   • No manual joint labeling required")
        print("   • No template creation needed")
        print("   • Fully automated tracking and analysis")
        print("   • Works with any rowing video")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the video file exists and is accessible.")


if __name__ == "__main__":
    main()