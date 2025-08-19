import numpy as np
from collections import defaultdict


class PostSessionFeedbackGenerator:
    def __init__(self):
        # Ideal angle ranges for each phase
        self.phase_ideal_ranges = {
            'Catch': {
                'back': (35, 50),      # Forward lean
                'knee': (100, 130),    # Compressed
                'elbow': (160, 180)    # Extended
            },
            'Drive': {
                'back': (40, 60),      # Opening up
                'knee': (130, 160),    # Extending
                'elbow': (150, 180)    # Still extended
            },
            'Finish': {
                'back': (50, 75),      # Leaning back
                'knee': (160, 180),    # Extended
                'elbow': (90, 120)     # Flexed (pulling)
            },
            'Recovery': {
                'back': (35, 55),      # Coming forward
                'knee': (120, 170),    # Variable
                'elbow': (160, 180)    # Extended again
            }
        }

    def evaluate_phase_technique(self, phase_analysis, phase_name):
        """Evaluate technique for a specific phase"""
        if phase_name not in self.phase_ideal_ranges:
            return None

        ideal_ranges = self.phase_ideal_ranges[phase_name]
        phase_data = phase_analysis.get(phase_name, {})
        angle_stats = phase_data.get('angle_stats', {})

        scores = {}
        feedback = {}

        # Evaluate back angle
        if 'back' in angle_stats:
            back_mean = angle_stats['back']['mean']
            ideal_back = ideal_ranges['back']
            if ideal_back[0] <= back_mean <= ideal_back[1]:
                scores['back'] = 100
                feedback['back'] = "Excellent back angle"
            else:
                deviation = min(abs(back_mean - ideal_back[0]), abs(back_mean - ideal_back[1]))
                scores['back'] = max(0, 100 - deviation * 2)
                if back_mean < ideal_back[0]:
                    feedback['back'] = f"Back too forward ({back_mean:.1f}°)"
                else:
                    feedback['back'] = f"Back too upright ({back_mean:.1f}°)"

        # Evaluate knee angles (average of left/right)
        knee_angles = []
        for knee in ['left_knee', 'right_knee']:
            if knee in angle_stats:
                knee_angles.append(angle_stats[knee]['mean'])

        if knee_angles:
            avg_knee = np.mean(knee_angles)
            ideal_knee = ideal_ranges['knee']
            if ideal_knee[0] <= avg_knee <= ideal_knee[1]:
                scores['legs'] = 100
                feedback['legs'] = "Good leg position"
            else:
                deviation = min(abs(avg_knee - ideal_knee[0]), abs(avg_knee - ideal_knee[1]))
                scores['legs'] = max(0, 100 - deviation * 1.5)
                if avg_knee < ideal_knee[0]:
                    feedback['legs'] = f"Legs too compressed ({avg_knee:.1f}°)"
                else:
                    feedback['legs'] = f"Legs too extended ({avg_knee:.1f}°)"

        # Evaluate elbow angles (average of left/right)
        elbow_angles = []
        for elbow in ['left_elbow', 'right_elbow']:
            if elbow in angle_stats:
                elbow_angles.append(angle_stats[elbow]['mean'])

        if elbow_angles:
            avg_elbow = np.mean(elbow_angles)
            ideal_elbow = ideal_ranges['elbow']
            if ideal_elbow[0] <= avg_elbow <= ideal_elbow[1]:
                scores['arms'] = 100
                feedback['arms'] = "Good arm position"
            else:
                deviation = min(abs(avg_elbow - ideal_elbow[0]), abs(avg_elbow - ideal_elbow[1]))
                scores['arms'] = max(0, 100 - deviation * 1.5)
                if avg_elbow < ideal_elbow[0]:
                    feedback['arms'] = f"Arms too bent ({avg_elbow:.1f}°)"
                else:
                    feedback['arms'] = f"Arms too straight ({avg_elbow:.1f}°)"

        overall_score = np.mean(list(scores.values())) if scores else 0

        return {
            'scores': scores,
            'overall_score': overall_score,
            'feedback': feedback,
            'frame_count': phase_data.get('frame_count', 0)
        }

    def generate_summary(self, all_strokes_analysis):
        """Generate comprehensive feedback summary"""
        phase_summaries = defaultdict(lambda: {'scores': [], 'feedback': []})

        # Aggregate across all strokes
        for stroke_analysis in all_strokes_analysis:
            for phase_name in ['Catch', 'Drive', 'Finish', 'Recovery']:
                phase_eval = self.evaluate_phase_technique(stroke_analysis, phase_name)
                if phase_eval:
                    phase_summaries[phase_name]['scores'].append(phase_eval['overall_score'])
                    phase_summaries[phase_name]['feedback'].append(phase_eval['feedback'])

        # Calculate averages and generate final feedback
        final_summary = {}
        overall_scores = []

        for phase_name, data in phase_summaries.items():
            if data['scores']:
                avg_score = np.mean(data['scores'])
                overall_scores.append(avg_score)

                # Generate phase-specific feedback
                common_issues = defaultdict(int)
                for feedback_dict in data['feedback']:
                    for joint, message in feedback_dict.items():
                        if 'too' in message.lower() or 'excellent' not in message.lower():
                            common_issues[joint] += 1

                phase_feedback = []
                if avg_score >= 90:
                    phase_feedback.append(f"Excellent {phase_name.lower()} technique!")
                elif avg_score >= 75:
                    phase_feedback.append(f"Good {phase_name.lower()} technique with room for improvement.")
                else:
                    phase_feedback.append(f"{phase_name} needs significant improvement.")

                # Add specific joint feedback
                for joint, count in common_issues.items():
                    if count > len(data['scores']) * 0.5:  # Issue in >50% of strokes
                        phase_feedback.append(f"Consistent {joint} positioning issues in {phase_name.lower()}.")

                final_summary[phase_name] = {
                    'score': avg_score,
                    'feedback': ' '.join(phase_feedback),
                    'stroke_count': len(data['scores'])
                }

        overall_score = np.mean(overall_scores) if overall_scores else 0

        return {
            'overall_score': overall_score,
            'phase_analysis': final_summary,
            'total_strokes_analyzed': len(all_strokes_analysis)
        }

    def save_feedback_to_text(self, summary, filepath):
        """Save feedback to text file"""
        import os

        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, "rowing_feedback.txt")

        parent = os.path.dirname(filepath)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("=== ROWING TECHNIQUE ANALYSIS REPORT ===\n\n")
            f.write(f"Overall Technique Score: {summary['overall_score']:.1f}/100\n")
            f.write(f"Total Strokes Analyzed: {summary['total_strokes_analyzed']}\n\n")

            f.write("PHASE-BY-PHASE ANALYSIS:\n")
            f.write("=" * 40 + "\n\n")

            for phase in ['Catch', 'Drive', 'Finish', 'Recovery']:
                if phase in summary['phase_analysis']:
                    phase_data = summary['phase_analysis'][phase]
                    f.write(f"{phase.upper()} PHASE:\n")
                    f.write(f"  Score: {phase_data['score']:.1f}/100\n")
                    f.write(f"  Strokes analyzed: {phase_data['stroke_count']}\n")
                    f.write(f"  Feedback: {phase_data['feedback']}\n\n")
                else:
                    f.write(f"{phase.upper()} PHASE:\n")
                    f.write("  No data available\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("=" * 40 + "\n")
            if summary['overall_score'] >= 85:
                f.write("Excellent technique! Focus on consistency.\n")
            elif summary['overall_score'] >= 70:
                f.write("Good overall technique. Work on the specific issues mentioned above.\n")
            else:
                f.write("Significant technique improvements needed. Consider working with a coach.\n")
