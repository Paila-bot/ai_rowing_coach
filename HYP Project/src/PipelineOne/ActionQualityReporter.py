import os


class ActionQualityReporter:
    def __init__(self):
        self.grade_thresholds = {
            'A+': 95, 'A': 90, 'A-': 85,
            'B+': 82, 'B': 78, 'B-': 75,
            'C+': 72, 'C': 68, 'C-': 65,
            'D+': 62, 'D': 58, 'D-': 55,
            'F': 0
        }

    def generate_report(self, analysis_results, output_path):
        """Generate comprehensive action quality report"""
        try:
            score = analysis_results.get('overall_score', 0)
            grade = self._get_grade(score)

            report = f"""
            === ROWING ACTION QUALITY ASSESSMENT REPORT ===
            Generated: {self._get_timestamp()}
            
            OVERALL PERFORMANCE
            ==================
            Final Score: {score:.1f}/100
            Grade: {grade}
            Performance Level: {self._get_performance_level(score)}
            
            SESSION OVERVIEW
            ===============
            Total Strokes Analyzed: {analysis_results.get('total_strokes', 0)}
            Average Stroke Score: {analysis_results.get('average_stroke_score', 0):.1f}
            Session Consistency: {analysis_results.get('session_consistency', 0):.1f}/100
            Stroke Rate: {analysis_results.get('stroke_rate', 'N/A')} strokes/min
            
            INDIVIDUAL STROKE ANALYSIS
            =========================
            """

            # Add individual stroke scores
            stroke_scores = analysis_results.get('stroke_scores', [])
            if stroke_scores:
                for i, stroke in enumerate(stroke_scores, 1):
                    try:
                        stroke_score = stroke.get('overall_score', 0)
                        consistency = stroke.get('consistency_score', 0)
                        report += f"Stroke {i:2d}: {stroke_score:5.1f}/100 "
                        report += f"(Consistency: {consistency:3.0f}) "
                        report += f"[{self._get_grade(stroke_score)}]\n"

                        # Add phase breakdown for detailed strokes
                        phase_scores = stroke.get('phase_scores', {})
                        if phase_scores:
                            for phase, phase_score in phase_scores.items():
                                report += f"    {phase:8s}: {phase_score:5.1f}/100\n"
                            report += "\n"
                    except Exception as e:
                        print(f"[WARNING] Error formatting stroke {i}: {e}")
                        continue

            # Performance recommendations
            report += f"""
PERFORMANCE ANALYSIS
===================
{self._get_detailed_feedback(analysis_results)}

IMPROVEMENT RECOMMENDATIONS
==========================
{self._get_improvement_recommendations(analysis_results)}

TECHNICAL NOTES
===============
- Scoring based on biomechanical analysis of joint angles
- Phase detection: Catch, Drive, Finish, Recovery
- Consistency measured across stroke-to-stroke variation
- Ideal angle ranges based on rowing biomechanics research

"""

            # Save report
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"[INFO] Report saved successfully to {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save report: {e}")

            return report

        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
            return "Error generating report"

    def _get_grade(self, score):
        """Convert numerical score to letter grade"""
        try:
            for grade, threshold in self.grade_thresholds.items():
                if score >= threshold:
                    return grade
            return 'F'
        except:
            return 'F'

    def _get_performance_level(self, score):
        """Get performance level description"""
        try:
            if score >= 90:
                return "Elite/Competitive"
            elif score >= 80:
                return "Advanced"
            elif score >= 70:
                return "Intermediate"
            elif score >= 60:
                return "Beginner+"
            else:
                return "Novice"
        except:
            return "Unknown"

    def _get_detailed_feedback(self, results):
        """Generate detailed performance feedback"""
        try:
            score = results.get('overall_score', 0)
            consistency = results.get('session_consistency', 0)
            stroke_count = results.get('total_strokes', 0)

            feedback = []

            # Overall performance feedback
            if score >= 85:
                feedback.append("Excellent technique demonstrated throughout the session.")
            elif score >= 70:
                feedback.append("Good technique with room for refinement in specific areas.")
            else:
                feedback.append("Technique needs significant improvement across multiple phases.")

            # Consistency feedback
            if consistency >= 90:
                feedback.append("Outstanding consistency across all strokes.")
            elif consistency >= 75:
                feedback.append("Good consistency with minor variations between strokes.")
            else:
                feedback.append("Inconsistent technique - focus on stroke-to-stroke repeatability.")

            # Session volume feedback
            if stroke_count >= 20:
                feedback.append("Good session volume for comprehensive analysis.")
            elif stroke_count >= 10:
                feedback.append("Moderate session volume - longer sessions provide better analysis.")
            else:
                feedback.append("Short session - consider longer training sessions for better assessment.")

            return "\n".join(f"• {item}" for item in feedback)

        except Exception as e:
            print(f"[WARNING] Feedback generation failed: {e}")
            return "• Unable to generate detailed feedback"

    def _get_improvement_recommendations(self, results):
        """Generate specific improvement recommendations"""
        try:
            recommendations = []
            score = results.get('overall_score', 0)
            consistency = results.get('session_consistency', 0)

            # Score-based recommendations
            if score < 70:
                recommendations.extend([
                    "Focus on basic rowing technique fundamentals",
                    "Work with a qualified coach to address form issues",
                    "Practice phase transitions (Catch -> Drive -> Finish -> Recovery)",
                    "Concentrate on proper body positioning throughout the stroke"
                ])
            elif score < 85:
                recommendations.extend([
                    "Fine-tune body positioning in specific stroke phases",
                    "Work on consistency between strokes",
                    "Focus on smooth phase transitions",
                    "Practice maintaining form at different intensities"
                ])
            else:
                recommendations.extend([
                    "Maintain excellent form while increasing stroke rate",
                    "Focus on competition-specific scenarios",
                    "Work on advanced technique refinements",
                    "Consider power and endurance training while maintaining technique"
                ])

            # Consistency-based recommendations
            if consistency < 80:
                recommendations.extend([
                    "Practice stroke repeatability drills",
                    "Focus on rhythm and timing consistency",
                    "Use mirrors or video feedback during training",
                    "Work on muscle memory development"
                ])

            return "\n".join(f"• {item}" for item in recommendations)

        except Exception as e:
            print(f"[WARNING] Recommendations generation failed: {e}")
            return "• Unable to generate specific recommendations"

    def _get_timestamp(self):
        """Get current timestamp for report"""
        try:
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except:
            return "Unknown"