from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import joblib

from src.nlp_utils import clean_text, extract_indicators, SimpleTokenizer
from src.model_utils import TextCNN1D, load_metadata

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

st.set_page_config(
    page_title="AI Mental Health Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4a90e2;
        margin-bottom: 10px;
    }
    .disclaimer-card {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 12px;
        color: #856404;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }
    .high-risk-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 15px;
        color: #721c24;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_models():
    """Load PyTorch CNN, Tokenizer, TF-IDF Vectorizer, and Metadata."""
    cnn_path = MODELS / "cnn_model.pt"
    tok_path = MODELS / "tokenizer.joblib"
    vec_path = MODELS / "tfidf_vectorizer.joblib"
    meta_path = MODELS / "metadata.json"

    if not (cnn_path.exists() and tok_path.exists() and vec_path.exists() and meta_path.exists()):
        return None

    try:
        metadata = load_metadata(meta_path)
        tokenizer = joblib.load(tok_path)
        vectorizer = joblib.load(vec_path)

        cnn = TextCNN1D(
            vocab_size=metadata["vocab_size"],
            feature_dim=metadata["feature_dim"],
            embed_dim=64,
            num_risk_classes=len(metadata["risk_classes"]),
            num_emotion_classes=len(metadata["emotion_classes"]),
            kernel_sizes=[2, 3, 4],
            num_filters=32,
            dropout=0.2
        )
        cnn.load_state_dict(torch.load(cnn_path, map_location=torch.device("cpu")))
        cnn.eval()

        return cnn, tokenizer, vectorizer, metadata
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None


def run_inference(text: str):
    """Run full neural network inference with calibrated clinical distress integration."""
    models = load_trained_models()
    indicators = extract_indicators(text)

    if models is None:
        # Fallback Mode
        t = text.lower()
        score = 25
        if indicators["high"]:
            score += len(indicators["high"]) * 22
        if indicators["medium"]:
            score += len(indicators["medium"]) * 10
        if indicators["positive"]:
            score -= len(indicators["positive"]) * 7
        score = max(5, min(95, score))

        if score >= 70 or indicators["high"]:
            risk = "High"
        elif score >= 45 or indicators["medium"]:
            risk = "Medium"
        else:
            risk = "Low"

        if indicators["high"] or any(w in t for w in ["crying", "cry", "failure", "lonely", "sad", "down", "exhausted"]):
            emotion = "Sad"
        elif indicators["medium"] or any(w in t for w in ["worried", "stress", "anxious"]):
            emotion = "Anxious"
        elif indicators["positive"]:
            emotion = "Positive"
        else:
            emotion = "Neutral"

        probs_risk = {"Low": 0.15 if risk != "Low" else 0.70,
                      "Medium": 0.60 if risk == "Medium" else 0.20,
                      "High": 0.85 if risk == "High" else 0.10}
        return {
            "mode": "Explainable Fallback",
            "risk": risk,
            "emotion": emotion,
            "risk_score": score,
            "risk_probs": probs_risk,
            "indicators": indicators["all_indicators"]
        }

    # Model Mode
    cnn, tokenizer, vectorizer, metadata = models
    cleaned = clean_text(text)

    # 1. Prepare sequence input
    seq = tokenizer.texts_to_sequences([cleaned])
    seq_padded = tokenizer.pad_sequences(seq, max_len=metadata["max_len"])
    x_seq = torch.tensor(seq_padded, dtype=torch.long)

    # 2. Prepare TF-IDF feature input
    x_tfidf = vectorizer.transform([cleaned]).toarray()
    x_feat = torch.tensor(x_tfidf, dtype=torch.float32)

    with torch.no_grad():
        r_logits, e_logits = cnn(x_seq, x_feat)
        r_probs = F.softmax(r_logits, dim=1).numpy()[0]
        e_probs = F.softmax(e_logits, dim=1).numpy()[0]

    risk_classes = metadata["risk_classes"]
    emotion_classes = metadata["emotion_classes"]

    prob_dict = {cls.lower(): float(prob) for cls, prob in zip(risk_classes, r_probs)}
    emo_dict = {cls.lower(): float(prob) for cls, prob in zip(emotion_classes, e_probs)}

    # Clinical Distress Calibration (Clinical Decision Support Prior)
    h_cnt = indicators["high_count"]
    m_cnt = indicators["medium_count"]
    p_cnt = indicators["positive_count"]

    if h_cnt >= 1:
        # Strong crisis/hopelessness/failure cues detected
        prob_dict["high"] = max(prob_dict.get("high", 0.0), 0.75 + min(0.20, h_cnt * 0.08))
        prob_dict["low"] = min(prob_dict.get("low", 0.0), 0.05)
        prob_dict["medium"] = max(0.0, 1.0 - prob_dict["high"] - prob_dict["low"])
        # Bias emotion towards sad/anxious
        emo_dict["sad"] = max(emo_dict.get("sad", 0.0), 0.70)
        emo_dict["positive"] = min(emo_dict.get("positive", 0.0), 0.05)
    elif m_cnt >= 1 and p_cnt == 0:
        # Moderate stress/anxiety cues detected
        if prob_dict.get("low", 0.0) > 0.50:
            prob_dict["medium"] = max(prob_dict.get("medium", 0.0), 0.65)
            prob_dict["low"] = 0.20
            prob_dict["high"] = max(0.0, 1.0 - prob_dict["medium"] - prob_dict["low"])
        if emo_dict.get("positive", 0.0) > 0.40:
            emo_dict["anxious"] = max(emo_dict.get("anxious", 0.0), 0.60)
            emo_dict["positive"] = 0.10

    # Normalize probabilities
    r_total = sum(prob_dict.values())
    prob_dict = {k: v / r_total for k, v in prob_dict.items()}

    e_total = sum(emo_dict.values())
    emo_dict = {k: v / e_total for k, v in emo_dict.items()}

    pred_risk_key = max(prob_dict, key=prob_dict.get)
    pred_emo_key = max(emo_dict, key=emo_dict.get)

    pred_risk = pred_risk_key.title()
    pred_emotion = pred_emo_key.title()

    p_low = prob_dict.get("low", 0.0)
    p_med = prob_dict.get("medium", 0.0)
    p_high = prob_dict.get("high", 0.0)

    continuous_score = int(round(p_low * 15 + p_med * 55 + p_high * 92))
    continuous_score = max(5, min(98, continuous_score))

    display_probs = {cls.title(): float(prob_dict.get(cls.lower(), 0.0)) for cls in risk_classes}

    return {
        "mode": "1D CNN + GAN Pipeline",
        "risk": pred_risk,
        "emotion": pred_emotion,
        "risk_score": continuous_score,
        "risk_probs": display_probs,
        "indicators": indicators["all_indicators"]
    }


# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mental-health.png", width=70)
    st.title("Monitor Settings")
    st.markdown("**AI Screening Framework:**")
    st.caption("• PyTorch 1D Multi-Branch CNN\n• Feature-Space GAN Augmentation\n• Longitudinal Session Tracking")

    st.divider()
    if st.button("🗑️ Reset Check-in History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🆘 Emergency Crisis Helplines")
    st.markdown("""
    - **Tele-MANAS (India):** `14416` or `1800 891 4416`
    - **KIRAN Helpline:** `1800-599-0019`
    - **Vandrevala Foundation:** `+91 9999 666 555`
    - **US/Canada 988 Lifeline:** `988`
    """)


# Main Interface Header
st.title("🧠 AI Mental Health Risk Screening & Monitoring System")
st.caption("College Mini-Project Demonstration • Multi-Scale 1D CNN + Feature-Space GAN")

st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Educational & Academic Screening Notice:</strong> This software is a student project demonstration developed for educational and screening research. It is <strong>NOT a diagnostic tool</strong> and cannot replace consultation with a licensed clinical psychologist or medical professional.
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_screen, tab_trends, tab_models = st.tabs([
    "📝 Screening & Check-in",
    "📈 Longitudinal Trends",
    "🔬 Model Architecture & Performance"
])

# ==========================================
# TAB 1: SCREENING & CHECK-IN
# ==========================================
with tab_screen:
    st.subheader("Daily Journal / Self-Assessment Check-in")
    st.write("Share how you have been feeling, your recent thoughts, academic stress, or daily experiences:")

    # Quick demo fill buttons
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if st.button("🟢 Sample: Positive Day", use_container_width=True):
            st.session_state.sample_text = "I had a really productive day studying with friends and feel motivated for tomorrow."
    with col_d2:
        if st.button("🟡 Sample: Exam Anxiety", use_container_width=True):
            st.session_state.sample_text = "I am constantly stressed about project deadlines and my heart races when thinking about exams."
    with col_d3:
        if st.button("🔴 Sample: Severe Distress", use_container_width=True):
            st.session_state.sample_text = "I felt very bad today, I was crying at a time too and I don't know what to do, I think I'm a failure."

    default_input = st.session_state.get("sample_text", "")
    user_text = st.text_area(
        "Your Check-in Text:",
        value=default_input,
        height=140,
        placeholder="Type a few sentences describing your current mental state, academic workload, or mood..."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Mental State", type="primary", use_container_width=True)

    if analyze_clicked:
        if len(user_text.strip()) < 8:
            st.error("Please enter a longer journal entry (at least 8 characters) for meaningful analysis.")
        else:
            with st.spinner("Processing NLP tokens & Deep Learning inference..."):
                res = run_inference(user_text)

            # Record in session state history
            st.session_state.history.append({
                "Session": len(st.session_state.history) + 1,
                "Risk Score": res["risk_score"],
                "Risk Level": res["risk"],
                "Emotion": res["emotion"],
                "Indicators Count": len(res["indicators"])
            })

            st.success(f"Analysis Complete • Pipeline: **{res['mode']}**")

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                emotion_icon = {"Positive": "😊", "Neutral": "😐", "Anxious": "😰", "Sad": "😔"}.get(res["emotion"], "🧠")
                st.metric("Detected Emotion", f"{emotion_icon} {res['emotion']}")
            with m2:
                risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(res["risk"], "⚪")
                st.metric("Screening Risk Level", f"{risk_icon} {res['risk']}")
            with m3:
                st.metric("Screening Risk Score", f"{res['risk_score']} / 100")

            # Softmax Probability Distribution
            st.markdown("##### 📊 Neural Network Probability Distribution")
            p_cols = st.columns(len(res["risk_probs"]))
            for idx, (label, prob) in enumerate(res["risk_probs"].items()):
                with p_cols[idx]:
                    st.write(f"**{label} Risk**: `{prob * 100:.1f}%`")
                    st.progress(float(prob))

            # Explainability / Indicator tags
            if res["indicators"]:
                st.markdown("##### 🔍 Detected Salience Indicators in Text")
                st.write(", ".join([f"`{w}`" for w in res["indicators"]]))

            # Contextual Clinical Guidance
            if res["risk_score"] >= 70:
                st.markdown("""
                <div class="high-risk-alert">
                    <strong>🚨 Elevated Risk Screening Detected:</strong><br>
                    The analysis indicates persistent distress or severe emotional strain. Please consider reaching out to a mental health professional, university counselor, or trusted individual. You do not have to carry this alone.
                </div>
                """, unsafe_allow_html=True)
            elif res["risk_score"] >= 45:
                st.warning(
                    "⚠️ **Moderate Screening Range:** Signs of stress, anxiety, or fatigue detected. "
                    "Prioritize regular sleep, physical breaks, academic pacing, and connecting with supportive peers."
                )
            else:
                st.info(
                    "✅ **Low Risk Range:** Your entry reflects stable or positive emotional wellbeing. "
                    "Keep up your healthy routines, self-care, and balanced schedule."
                )

# ==========================================
# TAB 2: LONGITUDINAL TRENDS
# ==========================================
with tab_trends:
    st.subheader("📈 Longitudinal Risk & Wellbeing Trajectory")
    st.write("Tracking check-in risk scores across multiple sessions provides insights into persistent vs. transient distress.")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        # Overview KPI metrics
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Total Check-ins", len(hist_df))
        with k2:
            st.metric("Average Risk Score", f"{hist_df['Risk Score'].mean():.1f} / 100")
        with k3:
            st.metric("Peak Risk Score", f"{hist_df['Risk Score'].max()} / 100")

        # Line Chart
        st.markdown("##### Risk Score Progression Over Sessions")
        chart_data = hist_df.set_index("Session")[["Risk Score"]]
        st.line_chart(chart_data)

        # Data Table
        st.markdown("##### Session Records")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export History as CSV",
            data=csv_data,
            file_name="mental_health_checkin_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No check-in entries logged yet. Perform an analysis in the Screening tab to view longitudinal trends.")

# ==========================================
# TAB 3: MODEL ARCHITECTURE & EVALUATION
# ==========================================
with tab_models:
    st.subheader("🔬 Machine Learning & Deep Learning Specifications")
    st.write("Overview of the system architecture, mathematical formulations, and evaluation metrics for college project presentation.")

    metrics_file = MODELS / "evaluation_metrics.json"
    if metrics_file.exists():
        metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1D CNN Risk Accuracy", f"{metrics_data.get('cnn_risk_accuracy', 0) * 100:.1f}%")
        c2.metric("Emotion Accuracy", f"{metrics_data.get('cnn_emotion_accuracy', 0) * 100:.1f}%")
        c3.metric("Baseline F1 Score", f"{metrics_data.get('baseline_f1', 0):.3f}")
        c4.metric("GAN-Augmented F1", f"{metrics_data.get('gan_augmented_f1', 0):.3f}")

        st.markdown("---")
        st.markdown("##### 🔲 Confusion Matrix (Risk Classification)")
        cm = np.array(metrics_data.get("risk_confusion_matrix", []))
        classes = [c.title() for c in metrics_data.get("risk_classes", ["Low", "Medium", "High"])]
        if len(cm) > 0:
            cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
            st.dataframe(cm_df, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📉 GAN Training Convergence (Discriminator & Generator Loss)")
        gan_hist = metrics_data.get("gan_history", {})
        if "d_loss" in gan_hist and "g_loss" in gan_hist:
            gan_df = pd.DataFrame({
                "Discriminator Loss": gan_hist["d_loss"],
                "Generator Loss": gan_hist["g_loss"]
            })
            st.line_chart(gan_df)

        st.markdown("---")
        st.markdown("##### 🏛️ Academic Architecture & Viva Defense Summary")
        st.markdown("""
        1. **NLP Representation Layer:** Dual representation utilizing Tokenized Sequence Embeddings for 1D CNN + TF-IDF n-grams for global contextual feature modeling.
        2. **Generative Adversarial Network (GAN):** A deep generator-discriminator pair trained on minority high-risk feature vectors to mitigate class imbalance in mental health data.
        3. **Multi-Scale 1D CNN:** Parallel convolution kernels ($k=2, 3, 4$) extracting n-gram features combined with dense contextual layers for joint **Risk** and **Emotion** multi-head classification.
        4. **Dynamic Continuous Scoring:** Computes risk score $S = \sum P(C_i) \cdot W_i \in [0, 100]$ avoiding abrupt discrete jumps.
        """)
    else:
        st.warning("Model evaluation metrics not found. Run `python train_model.py` to generate complete benchmark results.")

st.markdown("---")
st.caption("AI-Based Mental Health Monitoring System • Built with PyTorch, Scikit-Learn, and Streamlit")
