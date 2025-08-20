
import os

from src.PipelineOne.ActionQualityReporter import ActionQualityReporter
from src.PipelineOne.VideoProcessor import VideoProcessor
# Main execution function
def main():

    video_path = r"C:\Users\paida\Documents\Rowing Dataset\row_1.mp4"
    output_csv = r"C:\Users\paida\Documents\GitHub\ai_rowing_coach\HYP Project\src\PipelineOne\csv files\joints.csv"
    report_path = r"C:\Users\paida\Documents\GitHub\ai_rowing_coach\HYP Project\src\PipelineOne\feedback\rowing_quality_report.txt"

    # Initialize processor
    processor = VideoProcessor()
    reporter = ActionQualityReporter()

    try:
        print("=== STARTING ROWING ACTION QUALITY ASSESSMENT ===")

        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return None

        # Process video and analyze
        analysis_results, joints_data = processor.process_and_analyze_video(
            video_path, output_csv, frame_skip=3
        )

        # Generate comprehensive report
        print("[INFO] Generating quality assessment report...")
        report = reporter.generate_report(analysis_results, report_path)

        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Overall Score: {analysis_results.get('overall_score', 0):.1f}/100")
        print(f"Grade: {reporter._get_grade(analysis_results.get('overall_score', 0))}")
        print(f"Total Strokes: {analysis_results.get('total_strokes', 0)}")
        print(f"Report saved to: {report_path}")

        return analysis_results

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
        return None
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()