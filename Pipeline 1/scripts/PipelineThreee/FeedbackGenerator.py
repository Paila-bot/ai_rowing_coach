
import numpy as np

class FeedbackGenerator:
    def __init__(self):
        # Knowledge base for rowing technique
        self.rowing_knowledge = {
            'catch': {
                'common_issues': [
                    "overreaching at the catch",
                    "early arm bend before leg drive",
                    "poor shin angle at entry"
                ],
                'ideal_technique': [
                    "shoulders relaxed and squared",
                    "arms straight but not locked",
                    "shins vertical at entry"
                ]
            },
            'drive': {
                'common_issues': [
                    "legs and back not sequencing properly",
                    "opening the back too early",
                    "inconsistent handle height"
                ],
                'ideal_technique': [
                    "legs initiate the drive",
                    "smooth transition to back swing",
                    "maintain level handle path"
                ]
            },
            'finish': {
                'common_issues': [
                    "leaning back too far",
                    "pulling hands too high",
                    "breaking at wrists"
                ],
                'ideal_technique': [
                    "slight lean back (11 o'clock position)",
                    "hands come to lower ribs",
                    "flat wrists at release"
                ]
            },
            'recovery': {
                'common_issues': [
                    "rushing the slide",
                    "poor body preparation",
                    "irconsistent hand speed"
                ],
                'ideal_technique': [
                    "hands away first, then body swing",
                    "controlled slide speed",
                    "ready position before catch"
                ]
            }
        }

        # Learning components
        self.issue_frequency = {}  # Tracks how often issues occur
        self.user_terminology = {}  # Learns user's preferred terms
        self.coaching_style = "technical"  # Can adapt to 'encouraging', 'directive', etc.

        # Initialize issue frequency tracking
        for phase in self.rowing_knowledge:
            for issue in self.rowing_knowledge[phase]['common_issues']:
                self.issue_frequency[issue] = 0

    def generate_feedback(self, analysis_results):
        """Generate human-like coaching feedback"""
        # Update learning from current analysis
        self._update_knowledge(analysis_results)

        # Generate base technical feedback
        technical_feedback = self._generate_technical_feedback(analysis_results)

        # Generate human-like summary
        summary = self._generate_summary(analysis_results)

        # Generate personalized recommendations
        recommendations = self._generate_recommendations(analysis_results)

        # Compile full report
        report = {
            'summary': summary,
            'technical_analysis': technical_feedback,
            'recommendations': recommendations,
            'score': self._calculate_score(analysis_results)
        }

        return report

    def _update_knowledge(self, analysis_results):
        """Learn from current analysis session"""
        # Count frequency of issues in this session
        current_issues = {}
        for frame in analysis_results['critical_frames']:
            phase = self._get_phase_name(analysis_results['phases'][frame])
            if phase in self.rowing_knowledge:
                # Simple pattern - assume most common issue for phase
                issue = self.rowing_knowledge[phase]['common_issues'][0]
                current_issues[issue] = current_issues.get(issue, 0) + 1

        # Update long-term issue frequency
        for issue, count in current_issues.items():
            self.issue_frequency[issue] = self.issue_frequency.get(issue, 0) + count

        # Identify most common persistent issues
        self.persistent_issues = [
            issue for issue, count in sorted(self.issue_frequency.items(),
                                          key=lambda x: x[1], reverse=True)[:3]
        ]

    def _generate_summary(self, analysis_results):
        """Generate human-like summary of performance"""
        total_frames = len(analysis_results['phases'])
        critical_count = len(analysis_results['critical_frames'])
        error_rate = (critical_count / total_frames) * 100 if total_frames > 0 else 0

        # Phase distribution analysis
        phases = analysis_results['phases']
        phase_dist = {
            'catch': np.mean(phases == 0),
            'drive': np.mean(phases == 1),
            'finish': np.mean(phases == 2),
            'recovery': np.mean(phases == 3)
        }
        dominant_phase = max(phase_dist.items(), key=lambda x: x[1])[0]

        # Generate summary based on performance
        if error_rate < 10:
            summary = (f"Good session overall! Your technique was solid with only {critical_count} "
                      f"minor issues detected across {total_frames} frames. Your {dominant_phase} "
                      "phase looked particularly strong today.")
        elif error_rate < 25:
            summary = (f"Some good elements but room for improvement. We detected {critical_count} "
                     f"technical issues ({error_rate:.1f}% of strokes). Focus on your {dominant_phase} "
                     "phase where most issues occurred.")
        else:
            summary = (f"Let's work on consistency. We found {critical_count} issues ({error_rate:.1f}% "
                     f"of strokes), particularly during the {dominant_phase}. Don't worry - we'll "
                     "break this down with specific drills.")

        # Add persistent issue context if available
        if hasattr(self, 'persistent_issues') and self.persistent_issues:
            if len(self.persistent_issues) == 1:
                summary += f"\n\nWe're still seeing some {self.persistent_issues[0]} - let's focus on this next session."
            else:
                summary += "\n\nYour main areas to focus on remain: " + ", ".join(
                    self.persistent_issues[:-1]) + f", and {self.persistent_issues[-1]}."

        return summary

    def _generate_technical_feedback(self, analysis_results):
        """Generate detailed technical feedback"""
        feedback = []

        # Group critical frames by phase
        phase_frames = {}
        for frame in analysis_results['critical_frames']:
            phase = self._get_phase_name(analysis_results['phases'][frame])
            if phase not in phase_frames:
                phase_frames[phase] = []
            phase_frames[phase].append({
                'frame': frame,
                'error': analysis_results['anomaly_scores'][frame]
            })

        # Generate phase-specific feedback
        for phase, frames in phase_frames.items():
            if phase not in self.rowing_knowledge:
                continue

            # Get top 3 worst frames in this phase
            worst_frames = sorted(frames, key=lambda x: x['error'], reverse=True)[:3]

            # Select most likely issue based on error pattern
            likely_issue = self._identify_likely_issue(phase, worst_frames)

            feedback.append({
                'phase': phase.capitalize(),
                'issue': likely_issue,
                'frames': [f['frame'] for f in worst_frames],
                'average_error': np.mean([f['error'] for f in frames]),
                'ideal_technique': self._get_ideal_for_phase(phase)
            })

        return feedback

    def _identify_likely_issue(self, phase, frames):
        """Determine most likely technical issue based on error patterns"""
        # Simple implementation - can be enhanced with more sophisticated pattern matching
        if phase == 'catch':
            if any(f['error'] > 8 for f in frames):
                return "Overreaching at the catch (reaching too far forward)"
            return "Timing issue at catch (legs/arms coordination)"
        elif phase == 'drive':
            return "Power transfer issue (legs to back to arms)"
        elif phase == 'finish':
            return "Body position at release (lean angle)"
        return "Recovery sequencing (hands/body/slide)"

    def _get_ideal_for_phase(self, phase):
        """Get ideal technique points for a phase"""
        return self.rowing_knowledge.get(phase, {}).get('ideal_technique', [])

    def _generate_recommendations(self, analysis_results):
        """Generate personalized drills and exercises"""
        recommendations = []

        # Add general recommendations based on overall performance
        avg_error = np.mean(analysis_results['anomaly_scores'])
        if avg_error < 2:
            recommendations.append("Focus on refining technique with video review and mirror drills")
        elif avg_error < 5:
            recommendations.append("Practice stroke sequencing with pause drills at each phase")
        else:
            recommendations.append("Break down the stroke into components with isolation drills")

        # Add phase-specific drills
        phase_feedback = self._generate_technical_feedback(analysis_results)
        for feedback in phase_feedback:
            if feedback['average_error'] > 3:
                recs = {
                    'catch': [
                        "Catch position drills with mirror feedback",
                        "Pause drills at the catch (2 seconds)"
                    ],
                    'drive': [
                        "Leg drive isolation on erg (no arms)",
                        "Sequencing drills (legs-back-arms)"
                    ],
                    'finish': [
                        "Finish position holds with focus on body angle",
                        "Towel drill for consistent hand height"
                    ],
                    'recovery': [
                        "Hands-away-body-slow-slide sequence",
                        "Recovery timing with metronome"
                    ]
                }.get(feedback['phase'].lower(), [])

                recommendations.extend(recs)

        # Add recommendations for persistent issues
        if hasattr(self, 'persistent_issues'):
            for issue in self.persistent_issues:
                if "overreach" in issue.lower():
                    recommendations.append("Box drill to limit catch position")
                elif "sequencing" in issue.lower():
                    recommendations.append("Reverse pick drill for sequencing")
                elif "timing" in issue.lower():
                    recommendations.append("Pause drills at each phase transition")

        return list(set(recommendations))[:5]  # Return top 5 unique recommendations

    def _calculate_score(self, analysis_results):
        """Calculate performance score (0-100)"""
        total_frames = len(analysis_results['phases'])
        if total_frames == 0:
            return 0

        critical_count = len(analysis_results['critical_frames'])
        error_rate = (critical_count / total_frames) * 100

        # Base score (100% for 0 errors, 50% for 25% error rate)
        score = max(0, 100 - (error_rate * 2))

        # Adjust based on severity of errors
        avg_error = np.mean(analysis_results['anomaly_scores'])
        severity_adjustment = 1 - (min(avg_error, 10) / 20)  # Up to 5% adjustment
        score *= severity_adjustment

        return int(np.clip(score, 0, 100))

    def _get_phase_name(self, phase_num):
        phases = {0: 'catch', 1: 'drive', 2: 'finish', 3: 'recovery'}
        return phases.get(phase_num, 'unknown')