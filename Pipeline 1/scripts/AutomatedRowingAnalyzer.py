import cv2
import numpy as np
import json
from collections import defaultdict

class AutomatedRowingAnalyzer:
    """Rowing analysis using automatically tracked joints"""

    def __init__(self):
        self.joint_data = {}
        self.stroke_analysis = {}

    def load_automatic_tracking(self, filename='automatic_joint_positions.json'):
        """Load automatically tracked joint data"""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Convert string keys back to integers
        self.joint_data = {int(k): v for k, v in data.items()}
        print(f"Loaded automatic tracking data for {len(self.joint_data)} frames")

    def analyze_rowing_technique(self):
        """Analyze rowing technique from tracked joints"""
        print("Analyzing rowing technique...")

        # Calculate key metrics for each frame
        frame_analysis = {}

        for frame_idx, joints in self.joint_data.items():
            if len(joints) >= 4:  # Need at least 4 joints for analysis
                analysis = self.analyze_frame(joints)
                frame_analysis[frame_idx] = analysis

        # Detect strokes and phases
        strokes = self.detect_stroke_cycles(frame_analysis)

        # Analyze each stroke
        stroke_feedback = []
        for stroke in strokes:
            feedback = self.analyze_stroke_quality(stroke, frame_analysis)
            stroke_feedback.append(feedback)

        return strokes, stroke_feedback

    def analyze_frame(self, joints):
        """Analyze a single frame's joint positions"""
        analysis = {}

        if 'shoulder' in joints and 'hip' in joints:
            # Torso angle
            shoulder = joints['shoulder']
            hip = joints['hip']
            torso_angle = np.arctan2(shoulder[1] - hip[1], shoulder[0] - hip[0]) * 180 / np.pi
            analysis['torso_angle'] = torso_angle

        if 'hip' in joints and 'knee' in joints:
            # Leg compression
            hip = joints['hip']
            knee = joints['knee']
            leg_compression = np.sqrt((knee[0] - hip[0]) ** 2 + (knee[1] - hip[1]) ** 2)
            analysis['leg_compression'] = leg_compression

        if 'handle' in joints:
            # Handle position
            handle = joints['handle']
            analysis['handle_x'] = handle[0]
            analysis['handle_y'] = handle[1]

        return analysis

    def detect_stroke_cycles(self, frame_analysis):
        """Detect rowing stroke cycles from handle movement"""
        if not frame_analysis:
            return []

        # Extract handle positions
        frames = sorted(frame_analysis.keys())
        handle_positions = []

        for frame in frames:
            if 'handle_x' in frame_analysis[frame]:
                handle_positions.append(frame_analysis[frame]['handle_x'])
            else:
                handle_positions.append(0)  # Fallback

        # Find stroke cycles (local minima in handle position)
        stroke_starts = []
        for i in range(1, len(handle_positions) - 1):
            if (handle_positions[i] < handle_positions[i - 1] and
                    handle_positions[i] < handle_positions[i + 1]):
                stroke_starts.append(frames[i])

        # Create stroke objects
        strokes = []
        for i in range(len(stroke_starts) - 1):
            stroke = {
                'id': i + 1,
                'start_frame': stroke_starts[i],
                'end_frame': stroke_starts[i + 1],
                'duration': stroke_starts[i + 1] - stroke_starts[i]
            }
            strokes.append(stroke)

        return strokes

    def analyze_stroke_quality(self, stroke, frame_analysis):
        """Analyze the quality of a single stroke"""
        feedback = {
            'stroke_id': stroke['id'],
            'score': 100,
            'faults': [],
            'recommendations': []
        }

        # Get stroke data
        stroke_frames = range(stroke['start_frame'], stroke['end_frame'] + 1)
        stroke_data = {f: frame_analysis[f] for f in stroke_frames if f in frame_analysis}

        if not stroke_data:
            return feedback

        # Check for common faults

        # 1. Handle height variation (early arm bend)
        handle_ys = [data.get('handle_y', 0) for data in stroke_data.values()]
        if handle_ys:
            handle_variation = max(handle_ys) - min(handle_ys)
            if handle_variation > 30:  # Adjust threshold as needed
                feedback['faults'].append("Excessive handle height variation")
                feedback['score'] -= 15
                feedback['recommendations'].append("Keep arms straight longer during the drive")

        # 2. Stroke length (handle travel)
        handle_xs = [data.get('handle_x', 0) for data in stroke_data.values()]
        if handle_xs:
            stroke_length = max(handle_xs) - min(handle_xs)
            if stroke_length < 50:  # Adjust threshold as needed
                feedback['faults'].append("Short stroke length")
                feedback['score'] -= 20
                feedback['recommendations'].append("Extend further forward at the catch")

        # 3. Torso angle consistency
        torso_angles = [data.get('torso_angle', 0) for data in stroke_data.values()]
        if torso_angles:
            angle_range = max(torso_angles) - min(torso_angles)
            if angle_range > 40:  # Adjust threshold as needed
                feedback['faults'].append("Excessive torso movement")
                feedback['score'] -= 10
                feedback['recommendations'].append("Control torso swing, avoid over-reaching")

        # 4. Stroke timing (too fast/slow)
        if stroke['duration'] < 20:  # Adjust based on your frame rate
            feedback['faults'].append("Rushed stroke")
            feedback['score'] -= 10
            feedback['recommendations'].append("Slow down, focus on technique over speed")
        elif stroke['duration'] > 60:
            feedback['faults'].append("Very slow stroke")
            feedback['score'] -= 5
            feedback['recommendations'].append("Maintain consistent stroke rate")

        return feedback

    def generate_session_report(self, strokes, stroke_feedback):
        """Generate comprehensive session report"""
        report = []
        report.append("=== AUTOMATED ROWING ANALYSIS REPORT ===\n")

        if not strokes:
            report.append("No complete strokes detected in this session.")
            return "\n".join(report)

        # Session overview
        total_strokes = len(strokes)
        avg_score = np.mean([feedback['score'] for feedback in stroke_feedback])

        report.append(f"Session Overview:")
        report.append(f"  Total Strokes: {total_strokes}")
        report.append(f"  Average Score: {avg_score:.1f}/100")

        # Grade the session
        if avg_score >= 90:
            grade = "Excellent"
        elif avg_score >= 80:
            grade = "Good"
        elif avg_score >= 70:
            grade = "Fair"
        else:
            grade = "Needs Improvement"

        report.append(f"  Session Grade: {grade}\n")

        # Common faults analysis
        all_faults = []
        for feedback in stroke_feedback:
            all_faults.extend(feedback['faults'])

        if all_faults:
            fault_counts = {}
            for fault in all_faults:
                fault_counts[fault] = fault_counts.get(fault, 0) + 1

            report.append("Most Common Issues:")
            for fault, count in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_strokes) * 100
                report.append(f"  {fault}: {count} strokes ({percentage:.1f}%)")

        report.append("\n" + "=" * 50 + "\n")

        # Individual stroke analysis
        report.append("Individual Stroke Analysis:")
        for feedback in stroke_feedback:
            report.append(f"\nStroke {feedback['stroke_id']}: {feedback['score']}/100")

            if feedback['faults']:
                report.append(f"  Issues: {', '.join(feedback['faults'])}")
            else:
                report.append("  Clean technique!")

            if feedback['recommendations']:
                report.append(f"  Recommendations: {'; '.join(feedback['recommendations'])}")

        return "\n".join(report)

    def run_analysis(self):
        """Run complete automated analysis"""
        print("=== AUTOMATED ROWING ANALYSIS ===")

        # Load tracking data
        self.load_automatic_tracking()

        # Analyze technique
        strokes, stroke_feedback = self.analyze_rowing_technique()

        # Generate report
        report = self.generate_session_report(strokes, stroke_feedback)

        # Save results
        with open('automated_analysis_report.txt', 'w') as f:
            f.write(report)

        with open('automated_stroke_analysis.json', 'w') as f:
            json.dump({
                'strokes': strokes,
                'feedback': stroke_feedback
            }, f, indent=2)

        print("✅ Automated analysis complete!")
        print("\nGenerated files:")
        print("  - automated_analysis_report.txt")
        print("  - automated_stroke_analysis.json")

        print("\n" + "=" * 60)
        print(report)

        return strokes, stroke_feedback