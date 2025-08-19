import streamlit as st
import tempfile
import os
import json
import numpy as np

# --------------------------------------------------------
# Detect Pipeline Two availability
# --------------------------------------------------------
PIPELINE_2_AVAILABLE = False
try:
    from PipelineTwo.RowingAnalysisSystem import RowingAnalysisSystem as RAS
    PIPELINE_2_AVAILABLE = True
except ImportError:
    PIPELINE_2_AVAILABLE = False

# Pipeline One modules (assumed always available)
from PipelineOne.AutomatedJointTracker import AutomatedJointTracker
from PipelineOne.AutomatedRowingAnalyzer import AutomatedRowingAnalyzer

# --------------------------------------------------------
# Streamlit page config
# --------------------------------------------------------
st.set_page_config(page_title="Feather Rowing Coach", layout="wide")

# --------------------------------------------------------
# Custom CSS
# --------------------------------------------------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #00B4D8;
        }
        .stTextInput, .stFileUploader, .stTextArea, .stButton button {
            background-color: #1e222a;
            border: 1px solid #333;
            color: #fff;
        }
        .stButton>button {
            background-color: #0077b6;
            border: none;
            color: white;
            padding: 0.5rem 1.2rem;
            border-radius: 5px;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0096c7;
        }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1e222a;
            color: #fff;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        .section {
            margin-bottom: 2rem;
        }
        .analysis-card {
            background-color: #1e222a;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #00B4D8;
        }
        .score-display {
            font-size: 2rem;
            font-weight: bold;
            color: #00B4D8;
            text-align: center;
        }
        .feedback-item {
            background-color: #262730;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            border-left: 3px solid #0096c7;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    pipeline = st.selectbox("Choose Pipeline", ["Pipeline One", "Pipeline Two", "Pipeline Three"])

    if pipeline == "Pipeline Two":
        st.subheader("Pipeline 2 Settings")
        model_path = st.text_input(
            "Model Path",
            value=r"C:\Users\brigh\Documents\Honours\HYP\Project Implementation\ai_rowing_coach\Pipeline 1\models\Pipeline 2 models\model.pkl"
        )
        show_detailed = st.checkbox("Show Detailed Analysis", value=True)

        if os.path.exists(model_path):
            st.success("✅ Model found")
        else:
            st.error("❌ Model not found")

    st.button("Performance Evaluation")

# --------------------------------------------------------
# Main Header
# --------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>FEATHER</h1>", unsafe_allow_html=True)

# --------------------------------------------------------
# Upload Rowing Video
# --------------------------------------------------------
st.markdown("## Upload Rowing Video")
uploaded_video = st.file_uploader("Choose a rowing video", type=["mp4", "mov", "m4v", "avi", "mkv", "wmv"])

feedback_text = "Feedback will appear here after processing..."

# --------------------------------------------------------
# Process Video
# --------------------------------------------------------
if uploaded_video:
    st.video(uploaded_video)

    if st.button("Process Video"):
        st.success("Processing started...")

        if pipeline == "Pipeline One":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                tracker = AutomatedJointTracker(video_path)
                joint_positions = tracker.run()
                st.success(f"✅ Tracked {len(joint_positions)} frames")

                analyzer = AutomatedRowingAnalyzer()
                strokes, feedback = analyzer.run_analysis()
                st.success(f"✅ Analyzed {len(strokes)} strokes")

                if os.path.exists("tracking_visualization.mp4"):
                    st.subheader("📹 Tracking Visualization")
                    st.video("tracking_visualization.mp4")

                if os.path.exists("automated_analysis_report.txt"):
                    with open("automated_analysis_report.txt", "r") as f:
                        feedback_text = f.read()
                else:
                    feedback_text = feedback if isinstance(feedback, str) else str(feedback)

            except Exception as e:
                st.error(f"Error during Pipeline One processing: {e}")

        elif pipeline == "Pipeline Two":
            if not PIPELINE_2_AVAILABLE:
                st.error("Pipeline Two is not installed or could not be loaded.")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_video.read())
                        video_path = tmp.name

                    if not os.path.exists(model_path):
                        st.error(f"Model file not found at: {model_path}")
                        st.info("Please ensure you have trained and saved a model first.")
                    else:
                        with st.spinner("Initializing analysis system..."):
                            system = RAS()
                            system.load_model(model_path)

                        st.success("✅ Model loaded successfully")

                        with st.spinner("Analyzing video... This may take a few minutes..."):
                            analysis_result = system.analyze_video(video_path)

                        if 'error' in analysis_result:
                            st.error(f"Analysis failed: {analysis_result['error']}")
                            feedback_text = f"Analysis Error: {analysis_result['error']}"
                        else:
                            st.success("✅ Analysis completed successfully!")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                <div class="analysis-card">
                                    <h3>Overall Score</h3>
                                    <div class="score-display">{analysis_result['overall_score']:.1f}/100</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class="analysis-card">
                                    <h3>Consistency</h3>
                                    <div class="score-display">{analysis_result['consistency_score']:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div class="analysis-card">
                                    <h3>Frames Analyzed</h3>
                                    <div class="score-display">{analysis_result['total_frames_analyzed']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("### 💬 Coach Feedback")
                            for i, feedback_item in enumerate(analysis_result['detailed_feedback'], 1):
                                st.markdown(f"""
                                <div class="feedback-item">
                                    <strong>{i}.</strong> {feedback_item}
                                </div>
                                """, unsafe_allow_html=True)

                            if show_detailed and len(analysis_result['frame_analyses']) > 0:
                                st.markdown("### 📊 Detailed Frame Analysis")
                                frame_scores = []
                                categories = {}
                                for frame_analysis in analysis_result['frame_analyses']:
                                    score = max(0, 100 - (frame_analysis['mahalanobis_distance'] * 10))
                                    frame_scores.append(score)
                                    category = frame_analysis['feedback_category']
                                    categories[category] = categories.get(category, 0) + 1

                                st.markdown("**Technique Category Distribution:**")
                                total_frames = len(frame_scores)
                                for category, count in categories.items():
                                    percentage = (count / total_frames) * 100
                                    st.write(f"• {category}: {count} frames ({percentage:.1f}%)")

                                if frame_scores:
                                    avg_score = np.mean(frame_scores)
                                    min_score = np.min(frame_scores)
                                    max_score = np.max(frame_scores)
                                    st.markdown(f"""
                                    **Frame Score Statistics:**
                                    - Average Score: {avg_score:.1f}
                                    - Best Frame: {max_score:.1f}
                                    - Worst Frame: {min_score:.1f}
                                    """)

                            feedback_lines = [
                                f"🚣 ROWING TECHNIQUE ANALYSIS RESULTS",
                                "=" * 50,
                                "",
                                f"📊 OVERALL METRICS:",
                                f"   Overall Score: {analysis_result['overall_score']:.1f}/100",
                                f"   Consistency Score: {analysis_result['consistency_score']:.3f}",
                                f"   Average Deviation: {analysis_result['average_deviation']:.3f}",
                                f"   Frames Analyzed: {analysis_result['total_frames_analyzed']}",
                                "",
                                f"💬 COACH FEEDBACK:",
                            ]
                            for i, feedback_item in enumerate(analysis_result['detailed_feedback'], 1):
                                feedback_lines.append(f"   {i}. {feedback_item}")
                            feedback_text = "\n".join(feedback_lines)

                    try:
                        os.unlink(video_path)
                    except:
                        pass

                except Exception as e:
                    st.error(f"Error during Pipeline Two processing: {str(e)}")
                    feedback_text = f"Pipeline Two Error: {str(e)}"

        elif pipeline == "Pipeline Three":
            try:
                from PipelineThreeAnalyzer import PipelineThreeAnalyzer
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name
                analyzer = PipelineThreeAnalyzer(video_path)
                feedback_text = analyzer.run()
                st.success("✅ Pipeline Three analysis complete.")
            except Exception as e:
                st.error(f"❌ Error with main Pipeline Three: {e}")
                try:
                    from PipelineThreeAnalyzer import PipelineThreeAnalyzer
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_video.read())
                        video_path = tmp.name
                    simple_analyzer = PipelineThreeAnalyzer(video_path)
                    feedback_text = simple_analyzer.run()
                    st.info("🔍 Running diagnostic analysis...")
                except Exception as fallback_error:
                    st.error(f"❌ Diagnostic analysis also failed: {fallback_error}")
                    feedback_text = f"❌ Both main and diagnostic analysis failed.\n\nMain error: {e}\nDiagnostic error: {fallback_error}"

# --------------------------------------------------------
# Coach Feedback Section
# --------------------------------------------------------
st.markdown("## Coach Feedback")
st.text_area("Feedback", value=feedback_text, height=200)

# --------------------------------------------------------
# Pipeline 2 Info Section
# --------------------------------------------------------
if pipeline == "Pipeline Two" and PIPELINE_2_AVAILABLE:
    st.markdown("---")
    st.markdown("### 📋 Pipeline 2 Information")
    st.markdown("""
    **Pipeline 2: Statistical Technique Analysis**

    This pipeline uses:
    - Classical computer vision for pose estimation
    - Statistical modeling of good technique
    - Biomechanical feature extraction (14 features)
    - Mahalanobis distance for deviation detection
    - Neural network for feedback categorization

    **Features analyzed:**
    - Joint angles (elbows, knees, trunk)
    - Body segment ratios
    - Posture metrics
    - Symmetry measurements
    """)

    if os.path.exists(model_path):
        try:
            with open(model_path, 'r') as f:
                model_info = json.load(f)
                st.markdown(f"**Model trained on:** {model_info.get('feature_names', 'N/A')}")
        except Exception:
            st.markdown("**Model:** Loaded successfully")
elif pipeline == "Pipeline Two" and not PIPELINE_2_AVAILABLE:
    st.warning("⚠ Pipeline 2 is not installed or could not be loaded.")
