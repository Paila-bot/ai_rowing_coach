import os
import json
from . import RowingAnalysisSystem as RAS


class RowingEvaluator:
    """Simple interface for evaluating rowing videos using trained model"""

    def __init__(self, model_path: str):
        """Initialize evaluator with trained model"""
        self.system = RAS.RowingAnalysisSystem()
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.system.load_model(self.model_path)
                print(f"✓ Model loaded successfully from: {self.model_path}")
                print(f"✓ System is ready for video analysis")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def analyze_single_video(self, video_path: str, show_detailed: bool = True):
        """Analyze a single video and display results"""
        if not os.path.exists(video_path):
            print(f"✗ Video file not found: {video_path}")
            return None

        print(f"\n{'=' * 60}")
        print(f"ANALYZING VIDEO: {os.path.basename(video_path)}")
        print(f"{'=' * 60}")

        try:
            # Analyze the video
            result = self.system.analyze_video(video_path)

            if 'error' in result:
                print(f"✗ Analysis failed: {result['error']}")
                return None

            # Display basic results
            print(f"\n📊 OVERALL RESULTS:")
            print(f"   Overall Score: {result['overall_score']:.1f}/100")
            print(f"   Consistency Score: {result['consistency_score']:.3f}")
            print(f"   Average Deviation: {result['average_deviation']:.3f}")
            print(f"   Frames Analyzed: {result['total_frames_analyzed']}")

            # Display feedback
            print(f"\n💬 FEEDBACK:")
            for i, feedback in enumerate(result['detailed_feedback'], 1):
                print(f"   {i}. {feedback}")

            if show_detailed and len(result['frame_analyses']) > 0:
                self._show_detailed_analysis(result['frame_analyses'])

            return result

        except Exception as e:
            print(f"✗ Error during analysis: {e}")
            return None

    def _show_detailed_analysis(self, frame_analyses):
        """Show detailed frame-by-frame analysis"""
        print(f"\n📋 DETAILED FRAME ANALYSIS:")
        print(f"{'Frame':<8} {'Score':<8} {'Category':<30}")
        print(f"{'-' * 50}")


        # Show first 10 and last 5 frames if more than 15 total
        total_frames = len(frame_analyses)
        if total_frames <= 15:
            frames_to_show = frame_analyses
        elif total_frames <= 30:
            frames_to_show = frame_analyses[:10] + frame_analyses[-5:]
            show_gap = True
        else:
            show_gap = False

        shown_count = 0
        for i, analysis in enumerate(frames_to_show):
            if show_gap and shown_count == 10:
                print(f"{'...':<8} {'...':<8} {'...':<30}")

            frame_idx = analysis['frame_idx']
            score = 100 - (analysis['mahalanobis_distance'] * 10)  # Convert to 0-100 scale
            category = analysis['feedback_category']

            print(f"{frame_idx:<8} {score:<8.1f} {category:<30}")
            shown_count += 1


    def analyze_multiple_videos(self, video_directory: str):
        """Analyze all videos in a directory"""
        if not os.path.isdir(video_directory):
            print(f"✗ Directory not found: {video_directory}")
            return

        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []

        for file in os.listdir(video_directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(video_directory, file))

        if not video_files:
            print(f"✗ No video files found in: {video_directory}")
            return

        print(f"\n🎬 Found {len(video_files)} video files")

        # Analyze each video
        results = []
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
            result = self.analyze_single_video(video_path, show_detailed=False)
            if result:
                results.append({
                    'filename': os.path.basename(video_path),
                    'result': result
                })

        # Summary report
        if results:
            self._show_summary_report(results)


    def _show_summary_report(self, results):
        """Show summary report for multiple videos"""
        print(f"\n{'=' * 80}")
        print(f"SUMMARY REPORT - {len(results)} Videos Analyzed")
        print(f"{'=' * 80}")

        # Sort by overall score (best first)
        sorted_results = sorted(results, key=lambda x: x['result']['overall_score'], reverse=True)

        print(f"\n🏆 RANKING BY OVERALL SCORE:")
        print(f"{'Rank':<6} {'Score':<8} {'Consistency':<12} {'Filename':<30}")
        print(f"{'-' * 70}")

        total_score = 0
        for i, item in enumerate(sorted_results, 1):
            result = item['result']
            filename = item['filename']
            score = result['overall_score']
            consistency = result['consistency_score']
            total_score += score

            # Add medal emoji for top 3
            if i == 1:
                rank = "🥇 1"
            elif i == 2:
                rank = "🥈 2"
            elif i == 3:
                rank = "🥉 3"
            else:
                rank = f"   {i}"

            print(f"{rank:<6} {score:<8.1f} {consistency:<12.3f} {filename:<30}")

        # Statistics
        avg_score = total_score / len(results)
        print(f"\n📈 STATISTICS:")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Best Score: {sorted_results[0]['result']['overall_score']:.1f}")
        print(f"   Worst Score: {sorted_results[-1]['result']['overall_score']:.1f}")


    def save_analysis_report(self, video_path: str, output_path: str = None):
        """Analyze video and save detailed report to file"""
        result = self.analyze_single_video(video_path, show_detailed=False)

        if not result:
            return

        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"analysis_report_{base_name}.txt"

        try:
            with open(output_path, 'w') as f:
                f.write(f"ROWING TECHNIQUE ANALYSIS REPORT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Video: {os.path.basename(video_path)}\n")
                f.write(f"Analysis Date: {self._get_current_time()}\n\n")

                f.write(f"OVERALL RESULTS:\n")
                f.write(f"Overall Score: {result['overall_score']:.1f}/100\n")
                f.write(f"Consistency Score: {result['consistency_score']:.3f}\n")
                f.write(f"Average Deviation: {result['average_deviation']:.3f}\n")
                f.write(f"Frames Analyzed: {result['total_frames_analyzed']}\n\n")

                f.write(f"FEEDBACK:\n")
                for i, feedback in enumerate(result['detailed_feedback'], 1):
                    f.write(f"{i}. {feedback}\n")

                f.write(f"\nDETAILED FRAME ANALYSIS:\n")
                f.write(f"{'Frame':<8} {'Distance':<12} {'Category':<30}\n")
                f.write(f"{'-' * 55}\n")

                for analysis in result['frame_analyses']:
                    frame_idx = analysis['frame_idx']
                    distance = analysis['mahalanobis_distance']
                    category = analysis['feedback_category']
                    f.write(f"{frame_idx:<8} {distance:<12.3f} {category:<30}\n")

            print(f"✓ Detailed report saved to: {output_path}")

        except Exception as e:
            print(f"✗ Error saving report: {e}")


    def _get_current_time(self):
        """Get current time as string (without using datetime library)"""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


    def interactive_mode(self):
        """Run interactive command-line interface"""
        print(f"\n{'=' * 60}")
        print(f"🚣 ROWING TECHNIQUE ANALYZER")
        print(f"{'=' * 60}")
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Status: Ready for analysis")

        while True:
            print(f"\n{'=' * 40}")
            print(f"OPTIONS:")
            print(f"1. Analyze single video")
            print(f"2. Analyze all videos in directory")
            print(f"3. Analyze video and save report")
            print(f"4. Exit")

            try:
                choice = input(f"\nEnter your choice (1-4): ").strip()

                if choice == '1':
                    self._handle_single_video()
                elif choice == '2':
                    self._handle_directory_analysis()
                elif choice == '3':
                    self._handle_save_report()
                elif choice == '4':
                    print(f"\n👋 Goodbye!")
                    break
                else:
                    print(f"✗ Invalid choice. Please enter 1, 2, 3, or 4.")

            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"✗ Error: {e}")


    def _handle_single_video(self):
        """Handle single video analysis in interactive mode"""
        video_path = input(f"\nEnter video file path: ").strip().strip('"')
        if video_path:
            self.analyze_single_video(video_path)
        else:
            print(f"✗ No path entered")


    def _handle_directory_analysis(self):
        """Handle directory analysis in interactive mode"""
        directory = input(f"\nEnter directory path: ").strip().strip('"')
        if directory:
            self.analyze_multiple_videos(directory)
        else:
            print(f"✗ No directory entered")


    def _handle_save_report(self):
        """Handle save report in interactive mode"""
        video_path = input(f"\nEnter video file path: ").strip().strip('"')
        if not video_path:
            print(f"✗ No path entered")
            return

        output_path = input(f"Enter output file path (or press Enter for auto): ").strip().strip('"')
        if not output_path:
            output_path = None

        self.save_analysis_report(video_path, output_path)


def main():
    """Main function to run the evaluator"""
    print("🚣 Rowing Video Evaluator")
    print("=" * 40)

    # Get model path
    default_model = "models/model.pkl"

    if os.path.exists(default_model):
        print(f"Found model at: {default_model}")
        use_default = input(f"Use this model? (y/n): ").strip().lower()

        if use_default in ['y', 'yes', '']:
            model_path = default_model
        else:
            model_path = input(f"Enter model path: ").strip().strip('"')
    else:
        model_path = input(f"Enter model path: ").strip().strip('"')

    if not model_path:
        print("✗ No model path provided")
        return

    try:
        # Initialize evaluator
        evaluator = RowingEvaluator(model_path)

        # Run interactive mode
        evaluator.interactive_mode()

    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {e}")


if __name__ == "__main__":
    main()