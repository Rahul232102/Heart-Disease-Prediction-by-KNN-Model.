import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #fff5f5 0%, #fff0f0 50%, #ffe8e8 100%);
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 50%, #ef4444 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(185, 28, 28, 0.25);
    }
    .hero-banner h1 {
        font-family: 'DM Serif Display', serif;
        color: #ffffff !important;
        font-size: 2.4rem;
        margin: 0 0 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-banner .subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0 0 0.4rem;
    }
    .hero-banner .byline {
        color: rgba(255,255,255,0.65);
        font-size: 0.82rem;
    }
    .hero-pulse {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: pulse 1.8s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.15); }
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #b91c1c;
        margin: 1.8rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #fca5a5;
    }

    /* ── Info card strip ── */
    .info-strip {
        background: #fff7f7;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #7f1d1d;
    }

    /* ── Metric chips ── */
    .chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .chip {
        background: #fee2e2;
        color: #991b1b;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* ── Result cards ── */
    .result-high {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-low {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-icon { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.4rem; }
    .result-sub { font-size: 0.9rem; opacity: 0.75; }

    /* ── Predict button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 2.5rem !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(185,28,28,0.35) !important;
        transition: all 0.2s !important;
        letter-spacing: 0.03em;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(185,28,28,0.45) !important;
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: #fff5f5;
        border-right: 1px solid #fecaca;
    }

    /* ── Slider accent ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #dc2626 !important;
        border-color: #dc2626 !important;
    }

    /* ── Input label color ── */
    label { color: #374151 !important; font-weight: 500 !important; }

    /* ── Divider ── */
    hr { border-color: #fecaca !important; margin: 1.5rem 0 !important; }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #fecaca;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("KNN_heart_desease.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_model()

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <span class="hero-pulse">❤️</span>
    <h1>Heart Risk Predictor</h1>
    <p class="subtitle">AI-powered cardiovascular risk screening tool</p>
    <p class="byline">Built by <strong>Rahul</strong> · KNN Classification Model</p>
</div>
""", unsafe_allow_html=True)

# ── Info strip ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-strip">
    ℹ️ &nbsp;Fill in your health details below and click <strong>Predict</strong>.
    This tool is for educational use only — always consult a qualified doctor.
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Basic Info
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧍 Basic Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("🎂 Age", 18, 100, 45, help="Your current age in years")
with col2:
    sex = st.selectbox("⚧ Biological Sex", ["M", "F"],
                       format_func=lambda x: "👨 Male" if x == "M" else "👩 Female")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Vitals
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🩺 Vital Signs</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    resting_bp = st.number_input(
        "🩸 Resting Blood Pressure (mm Hg)",
        min_value=80, max_value=200, value=120,
        help="Blood pressure measured at rest"
    )
with col4:
    cholesterol = st.number_input(
        "🧪 Cholesterol (mg/dL)",
        min_value=100, max_value=600, value=200,
        help="Serum cholesterol level"
    )

col5, col6 = st.columns(2)
with col5:
    max_hr = st.slider(
        "💓 Max Heart Rate (bpm)",
        60, 220, 150,
        help="Maximum heart rate achieved during exercise"
    )
with col6:
    fasting_bs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "✅ No (≤ 120)" if x == 0 else "⚠️ Yes (> 120)"
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Clinical Details
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Clinical Details</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    chest_pain = st.selectbox(
        "😣 Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "💛 ATA — Atypical Angina",
            "NAP": "🩶 NAP — Non-Anginal Pain",
            "TA":  "🧡 TA — Typical Angina",
            "ASY": "❤️ ASY — Asymptomatic",
        }[x]
    )
with col8:
    resting_ecg = st.selectbox(
        "📈 Resting ECG Result",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "✅ Normal",
            "ST":     "⚠️ ST–T Abnormality",
            "LVH":    "❗ Left Ventricular Hypertrophy",
        }[x]
    )

col9, col10 = st.columns(2)
with col9:
    exercise_angina = st.selectbox(
        "🏃 Exercise-Induced Angina",
        ["Y", "N"],
        format_func=lambda x: "⚠️ Yes" if x == "Y" else "✅ No"
    )
with col10:
    st_slope = st.selectbox(
        "📉 ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up":   "📈 Upsloping",
            "Flat": "➡️ Flat",
            "Down": "📉 Downsloping",
        }[x]
    )

oldpeak = st.slider(
    "🔻 Oldpeak — ST Depression (relative to rest)",
    0.0, 6.0, 1.0, step=0.1,
    help="ST depression induced by exercise relative to rest"
)

# ── Live summary chips ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="chip-row">
    <span class="chip">🎂 Age: {age}</span>
    <span class="chip">⚧ {'Male' if sex=='M' else 'Female'}</span>
    <span class="chip">🩸 BP: {resting_bp} mm Hg</span>
    <span class="chip">🧪 Chol: {cholesterol} mg/dL</span>
    <span class="chip">💓 Max HR: {max_hr} bpm</span>
    <span class="chip">🔻 Oldpeak: {oldpeak}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ═════════════════════════════════════════════════════════════════════════════
predict_clicked = st.button("🔍 Analyse My Heart Risk", use_container_width=True)

if predict_clicked:
    with st.spinner("🫀 Analysing your cardiac profile..."):
        # Build raw input dict
        raw_input = {
            'Age':        age,
            'RestingBP':  resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS':  fasting_bs,
            'MaxHR':      max_hr,
            'Oldpeak':    oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1,
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction   = model.predict(scaled_input)[0]

    # ── Result display ────────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown("""
        <div class="result-high">
            <span class="result-icon">⚠️</span>
            <div class="result-title" style="color:#991b1b;">High Risk Detected</div>
            <div class="result-sub" style="color:#7f1d1d;">
                The model indicates an elevated risk of heart disease.<br>
                Please consult a cardiologist or your doctor promptly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Recommended Next Steps")
        st.error("🏥 Book an appointment with a cardiologist soon.")
        st.warning("🥗 Review your diet — reduce saturated fats and sodium.")
        st.warning("🚶 Aim for at least 150 min of moderate exercise per week.")
        st.info("💊 Discuss medication or further tests with your physician.")

    else:
        st.markdown("""
        <div class="result-low">
            <span class="result-icon">✅</span>
            <div class="result-title" style="color:#166534;">Low Risk — Great News!</div>
            <div class="result-sub" style="color:#14532d;">
                Based on your inputs, the model predicts a low risk of heart disease.<br>
                Keep maintaining a healthy lifestyle to stay protected.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Keep Up the Good Work")
        st.success("🥦 Maintain a balanced diet rich in fruits and vegetables.")
        st.success("🏃 Stay active — regular exercise keeps your heart strong.")
        st.info("🩺 Schedule routine check-ups every 1–2 years.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    ❤️ Heart Risk Predictor · Built with Streamlit & KNN · For educational use only<br>
    <strong>Always seek professional medical advice.</strong>
</div>
""", unsafe_allow_html=True)