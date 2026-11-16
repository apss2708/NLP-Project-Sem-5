"""BioNER Studio — Domain-Specific NER with Transformers (Project 16).

Premium Streamlit interface: dark biotech hero, teal palette, entity chips.
"""

import html

import pandas as pd
import streamlit as st

from src.inference import NERPredictor


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BioNER Studio | Domain-Specific NER",
    page_icon="🧬",
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
        --danger: #ef4444;
        --success: #10b981;
        --blue: #2563eb;
        --amber: #f59e0b;
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
            radial-gradient(1000px 520px at 8% -12%, rgba(13, 148, 136, 0.12), transparent 60%),
            radial-gradient(900px 480px at 95% -5%, rgba(37, 99, 235, 0.10), transparent 55%),
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

    /* ---------- Cards (bordered containers) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 22px 45px -30px rgba(15, 23, 42, 0.28);
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

    /* ---------- Stats ---------- */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
        gap: 0.7rem;
        margin: 0.85rem 0 1rem;
    }

    .stat-card {
        position: relative;
        overflow: hidden;
        background: #fbfdff;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 0.95rem 0.7rem;
        text-align: center;
    }

    .stat-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: var(--accent);
    }

    .stat-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0 0 0.1rem;
        color: var(--accent);
    }

    .stat-card .label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 0.3rem;
    }

    .stat-card .conf {
        font-size: 0.66rem;
        color: #7c8da3;
        margin: 0;
    }

    /* ---------- Entity chips ---------- */
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.15rem 0 1rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid;
    }

    .chip .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
    }

    .chip .score {
        font-size: 0.66rem;
        font-weight: 700;
        opacity: 0.75;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.9rem 0 0.5rem;
    }

    /* ---------- Highlighted text ---------- */
    .highlight-container {
        background: #fbfdff;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1.35rem;
        min-height: 180px;
        line-height: 2;
        font-size: 1rem;
        color: var(--ink);
        overflow-x: auto;
    }

    mark {
        border-radius: 5px;
        padding: 0.13rem 0.36rem;
        font-weight: 700;
        margin: 0 1px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.10);
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

    .stDownloadButton > button {
        background: #ffffff !important;
        color: #0d9488 !important;
        border: 1px solid #99f6e4 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: none !important;
    }

    .stDownloadButton > button:hover {
        background: #f0fdfa !important;
        border-color: #2dd4bf !important;
    }

    .stDownloadButton > button p { color: #0d9488 !important; }

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

    .stFileUploader section span,
    .stFileUploader section small { color: #40506a !important; }

    /* ---------- Tabs: segmented control ---------- */
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

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

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

    /* Dark selectbox */
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

    .legend-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.45rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.8rem;
        color: #c6d5e6;
        font-weight: 500;
    }

    .legend-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .sidebar-about {
        font-size: 0.8rem;
        color: #93a8bf;
        line-height: 1.55;
        margin: 0;
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

    /* Subtle scrollbar for highlighted box */
    .highlight-container::-webkit-scrollbar { height: 6px; }
    .highlight-container::-webkit-scrollbar-thumb {
        background: #cbd8e6; border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Entity palette
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "DISEASE": "#ef4444",
    "DRUG": "#10b981",
    "PROCEDURE": "#2563eb",
    "ANATOMY": "#f59e0b",
}

LABEL_BG = {
    "DISEASE": "#fef2f2",
    "DRUG": "#ecfdf5",
    "PROCEDURE": "#eff6ff",
    "ANATOMY": "#fffbeb",
}

LABEL_TEXT = {
    "DISEASE": "#991b1b",
    "DRUG": "#065f46",
    "PROCEDURE": "#1e40af",
    "ANATOMY": "#92400e",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def highlight_entities(text: str, entities: list) -> str:
    """Inline colored spans with confidence pills."""
    if not entities:
        return html.escape(text).replace("\n", "<br>")

    # Escape the whole text first, then map offsets of the escaped string back
    # to the original by processing entities in reverse on the raw text, and
    # escaping individual segments instead.
    ordered = sorted(entities, key=lambda e: e["start"])
    parts = []
    cursor = 0
    for ent in ordered:
        start, end = ent["start"], ent["end"]
        label, conf = ent["entity"], ent["confidence"]
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        conf_color = "#166534" if conf >= 0.7 else ("#92400e" if conf >= 0.4 else "#991b1b")
        parts.append(
            f'<mark style="background-color:{LABEL_BG.get(label, "#e2e8f0")};'
            f"color:{LABEL_TEXT.get(label, "#334155")};"
            f'border-left:3px solid {COLOR_MAP.get(label, "#64748b")};" '
            f'title="{label} · {conf:.0%}">'
            f"{html.escape(text[start:end])}"
            f'<span style="font-size:0.63rem;padding:0.04rem 0.28rem;border-radius:4px;'
            f'font-weight:700;margin-left:4px;background:{conf_color};color:#ffffff;">'
            f"{conf:.0%}</span></mark>"
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def get_entity_stats(entities: list) -> dict:
    stats = {}
    for ent in entities:
        label = ent["entity"]
        stats.setdefault(label, {"count": 0, "confidences": []})
        stats[label]["count"] += 1
        stats[label]["confidences"].append(ent["confidence"])
    return stats


def render_stats(entities: list) -> None:
    stats = get_entity_stats(entities)
    if not stats:
        return
    cards = '<div class="stats-grid">'
    for label in ["DISEASE", "DRUG", "PROCEDURE", "ANATOMY"]:
        if label in stats:
            count = stats[label]["count"]
            avg_conf = sum(stats[label]["confidences"]) / len(stats[label]["confidences"])
            cards += (
                f'<div class="stat-card" style="--accent:{COLOR_MAP.get(label, "#0d9488")};">'
                f'<p class="value">{count}</p>'
                f'<p class="label">{label}</p>'
                f'<p class="conf">Avg confidence {avg_conf:.0%}</p>'
                f"</div>"
            )
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)


def render_chips(entities: list) -> None:
    if not entities:
        return
    chips = '<div class="chips">'
    for ent in entities:
        label = ent["entity"]
        chips += (
            f'<span class="chip" style="background:{LABEL_BG.get(label, "#f1f5f9")};'
            f'color:{LABEL_TEXT.get(label, "#334155")};'
            f'border-color:{COLOR_MAP.get(label, "#94a3b8")}33;">'
            f'<span class="dot" style="background:{COLOR_MAP.get(label, "#94a3b8")};"></span>'
            f"{html.escape(ent['word'])}"
            f'<span class="score">{ent["confidence"]:.0%}</span></span>'
        )
    chips += "</div>"
    st.markdown(chips, unsafe_allow_html=True)


def render_results(text: str, entities: list) -> None:
    render_stats(entities)

    if entities:
        st.markdown('<p class="section-label">Detected entities</p>', unsafe_allow_html=True)
        render_chips(entities)

    st.markdown('<p class="section-label">Highlighted text</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="highlight-container">{highlight_entities(text, entities)}</div>',
        unsafe_allow_html=True,
    )

    if entities:
        df = pd.DataFrame(entities)[["entity", "word", "start", "end", "confidence"]]
        df.columns = ["Category", "Text", "Start", "End", "Confidence"]
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.0%}")

        def color_cat(val):
            return f"color: {LABEL_TEXT.get(val, '#334155')}; font-weight: 700;"

        st.markdown('<p class="section-label">Entity table</p>', unsafe_allow_html=True)
        st.dataframe(
            df.style.map(color_cat, subset=["Category"]),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download entities as CSV",
            data=csv,
            file_name="extracted_entities.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="info-box">No entities crossed the confidence threshold. '
            "Try lowering the threshold or adding more clinical detail.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NER model...")
def load_predictor(option: str):
    if option == "BioBERT · fine-tuned":
        return NERPredictor(
            model_path="./models/biobert_ner",
            model_name="dmis-lab/biobert-base-cased-v1.1",
        )
    if option == "BioBERT · base":
        return NERPredictor(
            model_path=None,
            model_name="dmis-lab/biobert-base-cased-v1.1",
        )
    return NERPredictor(model_path=None, model_name=None, rule_based=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h4>⚙️ Model settings</h4>',
        unsafe_allow_html=True,
    )

    model_option = st.selectbox(
        "Model",
        ["BioBERT · fine-tuned", "BioBERT · base", "Rule-based demo"],
        help="Fine-tuned uses the weights in models/biobert_ner. "
        "Rule-based demo runs instantly without downloading anything.",
    )

    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Entities below this confidence are hidden.",
    )

    st.markdown(
        f'<div class="model-state"><span class="pulse"></span>'
        f"{html.escape(model_option)} · ready</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<h4>🏷️ Entity types</h4>', unsafe_allow_html=True)
    st.markdown(
        '<div class="legend-grid">'
        '<div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span>DISEASE</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#10b981;"></span>DRUG</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#2563eb;"></span>PROCEDURE</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span>ANATOMY</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<h4>ℹ️ About</h4>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-about">Fine-tuned BioBERT for biomedical NER. '
        "Extracts diseases, drugs, procedures and anatomy from clinical text.</p>",
        unsafe_allow_html=True,
    )

predictor = load_predictor(model_option)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <span class="hero-badge">🧬 Biomedical NLP · Project 16</span>
            <h1>BioNER <span class="accent">Studio</span></h1>
            <p class="subtitle">
                Transformer-powered entity extraction that turns clinical notes into
                structured insight — diseases, drugs, procedures and anatomy in one pass.
            </p>
            <p class="team">
                <strong>Group 8</strong> &nbsp;·&nbsp; Ayush Singh (2401730015) &nbsp;·&nbsp;
                Krish Jaidka (2401730188) &nbsp;·&nbsp; Sahil Shrivastava (2401730016)
            </p>
            <div class="hero-pills">
                <span class="pill">BioBERT</span>
                <span class="pill">Token classification</span>
                <span class="pill">8 BIO labels</span>
                <span class="pill">Streamlit</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Single text", "📄 Upload document"])

# ---------------------------- Single text ----------------------------
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">📝 Clinical text</p>'
                '<p class="card-sub">Paste a note, report, or any biomedical excerpt.</p>',
                unsafe_allow_html=True,
            )
            text = st.text_area(
                "Clinical text",
                value=(
                    "The patient was diagnosed with type 2 diabetes mellitus and "
                    "prescribed metformin. She also has hypertension and takes lisinopril."
                ),
                height=280,
                label_visibility="collapsed",
                placeholder="e.g. The patient presented with chest pain and was started on aspirin...",
            )
            extract_btn = st.button("Extract entities", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">🔍 Extracted insight</p>'
                '<p class="card-sub">Entity summary, highlighted spans and downloadable table.</p>',
                unsafe_allow_html=True,
            )
            if extract_btn:
                if not text.strip():
                    st.warning("Please enter some text first.")
                else:
                    with st.spinner("Running BioBERT inference..."):
                        entities = predictor.predict(text, conf_thresh=conf_thresh)
                    render_results(text, entities)
            else:
                st.markdown(
                    '<div class="info-box">Add text and press '
                    "<b>Extract entities</b> — results will appear here.</div>",
                    unsafe_allow_html=True,
                )

# -------------------------- Upload document --------------------------
with tab2:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">📄 Upload a document</p>'
                '<p class="card-sub">Plain text (.txt) files up to a few hundred KB.</p>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload document",
                type=["txt"],
                label_visibility="collapsed",
                help="Upload a .txt file containing biomedical or clinical text.",
            )

            if uploaded is not None:
                doc_text = uploaded.read().decode("utf-8")
                st.text_area(
                    "Document preview",
                    doc_text,
                    height=220,
                    disabled=True,
                    label_visibility="collapsed",
                )
                extract_doc_btn = st.button(
                    "Extract entities from document",
                    type="primary",
                    use_container_width=True,
                )
            else:
                extract_doc_btn = False
                st.info("Upload a .txt file to begin.")

    with right:
        with st.container(border=True):
            st.markdown(
                '<p class="card-title">🔍 Extracted insight</p>'
                '<p class="card-sub">Entity summary, highlighted spans and downloadable table.</p>',
                unsafe_allow_html=True,
            )
            if uploaded is not None and extract_doc_btn:
                with st.spinner("Processing document..."):
                    entities = predictor.predict(doc_text, conf_thresh=conf_thresh)
                render_results(doc_text, entities)
            else:
                st.markdown(
                    '<div class="info-box">Upload a document and press '
                    "<b>Extract entities</b> — results will appear here.</div>",
                    unsafe_allow_html=True,
                )
