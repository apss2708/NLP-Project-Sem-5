"""Streamlit GUI for Sentiment Analysis (Project 1) - Polished Website UI."""

import html
import os
import pandas as pd
import streamlit as st

from src.inference import get_predictor


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer | NLP Project 1",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --teal: #0d9488;
        --teal-bright: #14b8a6;
        --cyan: #0891b2;
        --ink: #0b1220;
        --muted: #5b6b81;
        --line: #e6edf5;
        --positive: #10b981;
        --negative: #ef4444;
        --neutral: #f59e0b;
        --radius: 22px;
        --shadow: 0 22px 45px -30px rgba(15, 23, 42, 0.28);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .stApp {
        background:
            radial-gradient(1000px 520px at 8% -12%, rgba(13, 148, 136, 0.10), transparent 60%),
            radial-gradient(900px 480px at 95% -5%, rgba(37, 99, 235, 0.08), transparent 55%),
            #f6f8fb;
    }

    .main .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    #MainMenu, footer, header {visibility: hidden;}

    p, span, label, li { color: var(--ink); }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #070e1a 0%, #0c2430 52%, #08332e 100%);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 26px;
        padding: 2.75rem 2.75rem 2.4rem;
        margin-bottom: 1.9rem;
        box-shadow: 0 34px 70px -34px rgba(3, 16, 26, 0.55);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 460px;
        height: 460px;
        right: -140px;
        top: -190px;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.32), transparent 65%);
        pointer-events: none;
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        left: -120px;
        bottom: -190px;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.22), transparent 65%);
        pointer-events: none;
    }

    .hero-content { position: relative; z-index: 1; }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.35);
        color: #9ff0e4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.1rem;
    }

    .hero h1 {
        margin: 0 0 0.7rem;
        font-size: 3.1rem;
        font-weight: 700;
        line-height: 1.04;
        color: #f8fafc;
    }

    .hero h1 .accent {
        background: linear-gradient(92deg, #2dd4bf 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }

    .hero .subtitle {
        margin: 0 0 1.4rem;
        max-width: 640px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #b8c7d9;
    }

    .hero .team {
        margin: 0 0 1.2rem;
        font-size: 0.85rem;
        color: #8ea3ba;
        line-height: 1.6;
    }

    .hero .team strong { color: #dbe7f3; font-weight: 600; }

    .hero-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; }

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.78rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #d7e3f1;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* ---------- Cards ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 0.35rem;
    }

    .card-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 0.15rem;
    }

    .card-sub {
        font-size: 0.82rem;
        color: var(--muted);
        margin: 0 0 0.9rem;
    }

    /* ---------- Result badge ---------- */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .result-Positive { background: #ecfdf5; color: #065f46; border: 1px solid #86efac; }
    .result-Negative { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
    .result-Neutral  { background: #fffbeb; color: #92400e; border: 1px solid #fde047; }

    /* ---------- Confidence bar ---------- */
    .conf-container {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .conf-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .conf-label {
        width: 110px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
    }

    .conf-bar {
        flex: 1;
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }

    .conf-fill {
        height: 100%;
        border-radius: 999px;
    }

    .conf-val {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--ink);
        min-width: 40px;
        text-align: right;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.9rem 0 0.5rem;
    }

    /* ---------- Inputs & buttons ---------- */
    .stTextArea textarea {
        background: #fbfdff !important;
        border: 1px solid var(--line) !important;
        border-radius: 13px !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        padding: 0.9rem !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.14) !important;
    }

    .stTextArea textarea::placeholder { color: #93a3b8 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        letter-spacing: 0.01em !important;
        width: 100% !important;
        box-shadow: 0 16px 30px -16px rgba(13, 148, 136, 0.75) !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 22px 36px -16px rgba(13, 148, 136, 0.85) !important;
    }

    .stButton > button p { color: #ffffff !important; }

    /* ---------- File uploader ---------- */
    .stFileUploader > section {
        border: 2px dashed #cbd8e6 !important;
        border-radius: 14px !important;
        background: #fbfdff !important;
        padding: 1.9rem !important;
    }

    .stFileUploader > section:hover {
        border-color: var(--teal) !important;
        background: #f0fdfa !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: #eaf1f8;
        border: 1px solid #dde7f1;
        border-radius: 13px;
        padding: 0.32rem;
        margin-bottom: 1.1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border: none;
        border-radius: 10px;
        padding: 0.52rem 1.1rem;
        background: transparent;
        font-weight: 600;
        font-size: 0.88rem;
    }

    .stTabs [data-baseweb="tab"] p { color: #40506a !important; font-weight: 600 !important; }

    .stTabs [aria-selected="true"] {
        background: #ffffff;
        box-shadow: 0 3px 10px -3px rgba(15, 23, 42, 0.18);
    }

    .stTabs [aria-selected="true"] p { color: var(--teal) !important; font-weight: 700 !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070e1a 0%, #0c1a2b 100%);
        border-right: 1px solid #12233a;
    }

    section[data-testid="stSidebar"] * { color: #dbe7f3; }

    section[data-testid="stSidebar"] h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7f95ad !important;
        margin: 0 0 0.7rem;
    }

    section[data-testid="stSidebar"] hr { border-color: #17293f; }

    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #a9bbcd !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #101d30 !important;
        border: 1px solid #20344c !important;
        border-radius: 11px !important;
    }

    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] input {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stSelectbox svg { fill: #5eead4 !important; color: #5eead4 !important; }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: #0c1a2b !important;
        border: 1px solid #20344c !important;
        border-radius: 11px !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div[role="option"] {
        background-color: #0c1a2b !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: #16283f !important;
        color: #ffffff !important;
    }

    .model-state {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        background: rgba(45, 212, 191, 0.08);
        border: 1px solid rgba(45, 212, 191, 0.22);
        font-size: 0.74rem;
        color: #9ff0e4;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .model-state .pulse {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #2dd4bf;
        box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.15);
    }

    /* ---------- Info boxes ---------- */
    .info-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 11px;
        padding: 0.9rem 1rem;
        color: #115e59;
        font-size: 0.85rem;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde047;
        border-radius: 11px;
        padding: 0.9rem 1rem;
        color: #92400e;
        font-size: 0.85rem;
    }

    /* ---------- Example chips ---------- */
    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.6rem;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        font-size: 0.78rem;
        color: #334155;
        font-weight: 500;
        cursor: pointer;
    }

    .chip:hover { background: #e2e8f0; }

    /* ---------- Footer ---------- */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid var(--line);
        color: #475569;
        font-size: 0.8rem;
        text-align: center;
    }

    .footer .brand {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_predictor():
    model_path = os.path.join("models", "sentiment_logistic_regression.joblib")
    if not os.path.exists(model_path):
        return get_predictor(None)
    return get_predictor(model_path)


import os
predictor = load_predictor()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <span class="hero-badge">💬 Sentiment Analysis · Project 1</span>
            <h1>Sentiment <span class="accent">Analyzer</span></h1>
            <p class="subtitle">
                Transformer-style ML pipeline that classifies movie or product reviews
                as Positive, Negative, or Neutral using TF-IDF and classical classifiers.
            </p>
            <p class="team">
                <strong>Group 8</strong> &nbsp;·&nbsp; Ayush Singh (2401730015) &nbsp;·&nbsp;
                Krish Jaidka (2401730188) &nbsp;·&nbsp; Sahil Shrivastava (2401730016)
            </p>
            <div class="hero-pills">
                <span class="pill">TF-IDF</span>
                <span class="pill">Logistic Regression</span>
                <span class="pill">Naive Bayes</span>
                <span class="pill">SVM</span>
                <span class="pill">Streamlit</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<h4>⚙️ Model Settings</h4>', unsafe_allow_html=True)

    model_option = st.selectbox(
        "Model",
        ["Logistic Regression", "Naive Bayes", "SVM", "Rule-Based Demo"],
        help="ML models require trained weights in models/. Rule-Based Demo runs instantly.",
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Predictions below this confidence are flagged as Neutral.",
    )

    st.markdown(
        '<div class="model-state"><span class="pulse"></span>'
        f"{html.escape(model_option)} · ready</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<h4>🏷️ Sentiment Classes</h4>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
            <div style="background:#ecfdf5; border:1px solid #86efac; border-radius:8px; padding:0.5rem; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#065f46; text-transform:uppercase; letter-spacing:0.5px;">Positive</div>
                <div style="font-size:0.75rem; color:#065f46;">😊</div>
            </div>
            <div style="background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:0.5rem; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#991b1b; text-transform:uppercase; letter-spacing:0.5px;">Negative</div>
                <div style="font-size:0.75rem; color:#991b1b;">😞</div>
            </div>
            <div style="background:#fffbeb; border:1px solid #fde047; border-radius:8px; padding:0.5rem; text-align:center;">
                <div style="font-size:0.7rem; font-weight:700; color:#92400e; text-transform:uppercase; letter-spacing:0.5px;">Neutral</div>
                <div style="font-size:0.75rem; color:#92400e;">😐</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<h4>ℹ️ About</h4>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem; color:#93a8bf; line-height:1.5; margin:0;">'
        "TF-IDF + ML classification for sentiment analysis. "
        "Extracts sentiment from movie or product reviews."
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Single Review", "📄 Upload File"])

# ---------------------------- Single Text ----------------------------
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">📝 Enter Review</p>'
                '<p class="card-sub">Type a movie or product review to analyze its sentiment.</p>',
                unsafe_allow_html=True,
            )
            default_text = (
                "The movie was fantastic and I loved every minute of it. "
                "Outstanding direction and brilliant performances!"
            )
            text = st.text_area(
                "Review text",
                value=default_text,
                height=280,
                label_visibility="collapsed",
                placeholder="e.g. The product exceeded my expectations. Highly recommended!",
            )
            analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

            st.markdown('<p class="section-label">Try these examples:</p>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="example-chips">
                    <span class="chip">The movie was fantastic and I loved every minute of it.</span>
                    <span class="chip">Terrible plot, terrible acting, complete waste of time.</span>
                    <span class="chip">It was okay, nothing special but watchable.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">🔍 Prediction Result</p>'
                '<p class="card-sub">Sentiment label, confidence score, and class probabilities.</p>',
                unsafe_allow_html=True,
            )
            if analyze_btn:
                if not text.strip():
                    st.markdown(
                        '<div class="warning-box">⚠️ Please enter some text to analyze.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    with st.spinner("Analyzing sentiment..."):
                        result = predictor.predict(text)
                    label = result["label"]
                    confidence = result["confidence"]
                    probs = result["probabilities"]

                    css_class = f"result-{label}"
                    emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(label, "😐")

                    st.markdown(
                        f'<div class="result-badge {css_class}">'
                        f"{emoji} {label}"
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown('<p class="section-label">Confidence Breakdown</p>', unsafe_allow_html=True)
                    for cls in ["Positive", "Negative", "Neutral"]:
                        p = probs.get(cls, 0.0)
                        color = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}[cls]
                        st.markdown(
                            f"""
                            <div class="conf-row">
                                <span class="conf-label">{cls}</span>
                                <div class="conf-bar">
                                    <div class="conf-fill" style="width:{p*100}%; background:{color};"></div>
                                </div>
                                <span class="conf-val">{p:.0%}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    '<div class="info-box">Enter a review and press '
                    "<b>Analyze Sentiment</b> — results will appear here.</div>",
                    unsafe_allow_html=True,
                )

# -------------------------- Upload File --------------------------
with tab2:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">📄 Upload Reviews File</p>'
                '<p class="card-sub">Upload a CSV with a "text" column for batch analysis.</p>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                label_visibility="collapsed",
                help="Upload a .csv file containing a 'text' column.",
            )

            if uploaded is not None:
                df = pd.read_csv(uploaded)
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    st.text_area(
                        "Preview",
                        df.head(5).to_string(),
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    analyze_file_btn = st.button(
                        "Analyze All Reviews",
                        type="primary",
                        use_container_width=True,
                    )
            else:
                analyze_file_btn = False
                st.info("Upload a .csv file to begin.")

    with right:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">🔍 Batch Results</p>'
                '<p class="card-sub">Sentiment predictions for all uploaded reviews.</p>',
                unsafe_allow_html=True,
            )
            if uploaded is not None and analyze_file_btn:
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    with st.spinner("Processing reviews..."):
                        results = [predictor.predict(t) for t in df["text"].astype(str)]
                    df["predicted_sentiment"] = [r["label"] for r in results]
                    df["confidence"] = [r["confidence"] for r in results]

                    # Summary stats
                    total = len(df)
                    pos = (df["predicted_sentiment"] == "Positive").sum()
                    neg = (df["predicted_sentiment"] == "Negative").sum()
                    neu = (df["predicted_sentiment"] == "Neutral").sum()

                    st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Total Reviews", total)
                    with c2:
                        st.metric("Positive", int(pos))
                    with c3:
                        st.metric("Negative", int(neg))
                    with c4:
                        st.metric("Neutral", int(neu))

                    st.markdown('<p class="section-label">Predictions Table</p>', unsafe_allow_html=True)
                    st.dataframe(
                        df[["text", "predicted_sentiment", "confidence"]],
                        use_container_width=True,
                        hide_index=True,
                        height=350,
                    )

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.markdown(
                    '<div class="info-box">Upload a CSV and press '
                    "<b>Analyze All Reviews</b> — results will appear here.</div>",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <div class="brand">💬 Project 1 – Sentiment Analysis | Group 8</div>
        <p style="margin:0; line-height:1.6;">
            Ayush Singh (2401730015) · Krish Jaidka (2401730188) · Sahil Shrivastava (2401730016)<br>
            Educational capstone project. Model predictions are assistive, not clinical diagnoses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
