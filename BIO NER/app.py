"""BioNER Clinical — Domain-Specific NER with Transformers (Project 16).

A clinical-journal interface for biomedical entity extraction: porcelain paper,
deep pine accents, serif display type and annotation-style entity spans.
"""

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import NERPredictor


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BioNER Clinical | Biomedical NER",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "biobert_ner"
FINE_TUNED_AVAILABLE = (MODEL_DIR / "config.json").exists()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap');

    :root {
        --paper: #F6F7F3;
        --surface: #FFFFFF;
        --surface-2: #FBFCFA;
        --ink: #10201F;
        --ink-2: #3B4D49;
        --muted: #566661;
        --line: #DFE6E1;
        --line-strong: #AEBAB4;
        --teal: #0B4F52;
        --teal-deep: #083D40;
        --teal-tint: #E7F0EE;
        --disease: #A13B3B;
        --disease-bg: #FAEDEC;
        --drug: #2C6A4D;
        --drug-bg: #ECF5EF;
        --procedure: #2B5B8C;
        --procedure-bg: #ECF2F9;
        --anatomy: #8A6420;
        --anatomy-bg: #FAF3E4;
    }

    html { scroll-behavior: smooth; }
    html, body, [class*="css"] {
        font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--paper); }

    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1260px !important;
        padding: 0.35rem 2rem 2.5rem 2rem !important;
    }

    #MainMenu, [data-testid="stFooter"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    header[data-testid="stHeader"] { background: transparent; }

    body, p, li, span, label { color: var(--ink); }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em;
        color: var(--ink);
    }

    ::selection { background: var(--teal-tint); color: var(--teal-deep); }

    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #EDF2F0; }
    ::-webkit-scrollbar-thumb { background: #C5D2CE; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #A9BBB6; }

    /* ------------------------------------------------------------------
       Compliance strip, site header
       ------------------------------------------------------------------ */
    .compliance {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.42rem 0.85rem;
        background: var(--teal-deep);
        color: #E8F2EF;
        border-radius: 2px 2px 0 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        line-height: 1.4;
    }

    .compliance .c-left { display: flex; align-items: center; gap: 0.5rem; }
    .compliance .c-dot {
        width: 5px; height: 5px; border-radius: 50%;
        background: #9CC9BE; flex: none;
    }
    .compliance .c-right { color: #A9C6BF; white-space: nowrap; }

    .site-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 0.9rem 0.1rem 0.95rem;
        border-bottom: 1px solid var(--line);
    }

    .brand { display: flex; align-items: center; gap: 0.65rem; }

    .brand-mark {
        width: 34px; height: 34px;
        display: grid; place-items: center;
        background: var(--teal);
        border: 1px solid var(--teal);
        border-radius: 5px;
        box-shadow: inset 0 0 0 3px rgba(255, 255, 255, 0.10);
        flex: none;
    }
    .brand-mark svg { width: 18px; height: 18px; stroke: #FFFFFF; }

    .brand-name {
        display: block;
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.18rem;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -0.01em;
    }
    .brand-name em { color: var(--teal); font-style: italic; }
    .brand-sub {
        display: block;
        margin-top: 0.28rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.52rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .head-right { display: flex; align-items: center; gap: 1.5rem; }
    .head-nav { display: flex; align-items: center; gap: 1.25rem; }
    .head-nav a {
        font-size: 0.76rem;
        font-weight: 500;
        color: var(--ink-2);
        text-decoration: none;
        padding-bottom: 2px;
        border-bottom: 1px solid transparent;
    }
    .head-nav a:hover { color: var(--teal); border-color: var(--teal); }

    .sys-state {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.28rem 0.62rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 999px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-2);
        white-space: nowrap;
    }
    .sys-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--drug);
        box-shadow: 0 0 0 3px rgba(44, 106, 77, 0.12);
        flex: none;
    }
    .sys-dot.fallback {
        background: var(--anatomy);
        box-shadow: 0 0 0 3px rgba(138, 100, 32, 0.14);
    }

    /* ------------------------------------------------------------------
       Hero
       ------------------------------------------------------------------ */
    .hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(285px, 0.55fr);
        gap: 2.5rem;
        align-items: center;
        padding: 2.25rem 0 2rem;
        border-bottom: 1px solid var(--line);
        background-image:
            linear-gradient(rgba(14, 90, 84, 0.032) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 90, 84, 0.032) 1px, transparent 1px);
        background-size: 28px 28px;
        background-position: -1px -1px;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: 5%;
        top: 8%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(14, 90, 84, 0.07), transparent 66%);
        pointer-events: none;
    }

    .hero-copy { position: relative; z-index: 1; }

    .eyebrow {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0 0 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--teal);
        line-height: 1.4;
    }
    .eyebrow::before {
        content: "";
        width: 22px; height: 1px;
        background: var(--teal);
        flex: none;
    }

    .hero h1 {
        margin: 0 0 1.05rem;
        font-size: clamp(2.6rem, 4.4vw, 3.85rem);
        line-height: 0.99;
        font-weight: 400 !important;
        letter-spacing: -0.035em;
    }
    .hero h1 em { color: var(--teal); font-style: italic; }

    .hero .lede {
        max-width: 540px;
        margin: 0 0 1.2rem;
        font-size: 0.95rem;
        line-height: 1.65;
        color: var(--ink-2);
    }

    .hero-tags { display: flex; flex-wrap: wrap; gap: 0.42rem; }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.28rem 0.55rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.07em;
        color: var(--ink-2);
    }
    .hero-tag i {
        width: 5px; height: 5px; border-radius: 50%;
        background: var(--c); flex: none;
    }

    .model-card {
        position: relative;
        z-index: 1;
        padding: 1.1rem 1.1rem 0.95rem;
        background: var(--surface);
        border: 1px solid var(--line-strong);
        border-radius: 6px;
        box-shadow: 0 26px 54px -36px rgba(18, 42, 44, 0.42);
    }
    .model-card::before {
        content: "";
        position: absolute;
        inset: 5px;
        border: 1px solid rgba(14, 90, 84, 0.07);
        border-radius: 3px;
        pointer-events: none;
    }

    .mc-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.53rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .mc-live { display: flex; align-items: center; gap: 0.35rem; color: var(--drug); }
    .mc-live.fallback { color: var(--anatomy); }
    .mc-live::before {
        content: "";
        width: 5px; height: 5px; border-radius: 50%;
        background: currentColor;
    }

    .mc-title {
        margin: 0.5rem 0 0.05rem;
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2rem;
        font-weight: 500;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .mc-sub { margin: 0 0 0.85rem; font-size: 0.72rem; color: var(--muted); }

    .mc-rows { border-top: 1px solid var(--line); }
    .mc-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.6rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.66rem;
    }
    .mc-row span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.5rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        white-space: nowrap;
    }
    .mc-row b {
        font-weight: 600;
        color: var(--ink);
        text-align: right;
        font-size: 0.66rem;
        font-variant-numeric: tabular-nums;
    }

    .mc-foot { display: flex; align-items: center; gap: 0.7rem; margin-top: 0.8rem; }
    .barcode {
        flex: 1;
        height: 24px;
        background: repeating-linear-gradient(
            90deg,
            var(--ink) 0 1px,
            transparent 1px 3px,
            var(--ink) 3px 5px,
            transparent 5px 8px
        );
        opacity: 0.68;
    }
    .mc-foot span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.08em;
        color: var(--muted);
        white-space: nowrap;
    }

    /* ------------------------------------------------------------------
       Protocol strip
       ------------------------------------------------------------------ */
    .protocol {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border-bottom: 1px solid var(--line);
    }
    .protocol-item {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        padding: 0.9rem 1.3rem 0.9rem 0;
        border-right: 1px solid var(--line);
    }
    .protocol-item:last-child { border-right: 0; padding-right: 0; }
    .protocol-num {
        display: grid;
        place-items: center;
        width: 26px; height: 26px;
        flex: none;
        background: var(--teal-tint);
        border: 1px solid rgba(14, 90, 84, 0.22);
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.55rem;
        color: var(--teal);
    }
    .protocol-item h4 {
        margin: 0.08rem 0 0.18rem;
        font-family: 'Public Sans', sans-serif !important;
        font-size: 0.7rem;
        font-weight: 700 !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .protocol-item p {
        margin: 0;
        font-size: 0.74rem;
        line-height: 1.5;
        color: var(--muted);
    }

    /* ------------------------------------------------------------------
       Section headers and tabs
       ------------------------------------------------------------------ */
    .sec-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.75rem 0 1.05rem;
    }
    .sec-head h2 {
        margin: 0.3rem 0 0;
        font-size: 2.1rem;
        font-weight: 400 !important;
        letter-spacing: -0.025em;
    }
    .sec-head .eyebrow { margin-bottom: 0.35rem; }
    .sec-note {
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        text-align: right;
        line-height: 1.7;
    }

    .stTabs [data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        gap: 1.7rem;
        background: transparent;
        border-bottom: 1px solid var(--line);
        border-radius: 0;
        padding: 0;
        margin-bottom: 1.1rem;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [role="tab"] {
        padding: 0.58rem 0;
        background: transparent;
        border: none;
        border-radius: 0;
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [role="tab"] p {
        color: var(--muted) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.6rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] { background: transparent; box-shadow: none; }
    .stTabs [aria-selected="true"] p { color: var(--teal) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--teal); height: 2px; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-panel"], .stTabs [role="tabpanel"] { padding-top: 0.15rem; }

    /* ------------------------------------------------------------------
       Panels
       ------------------------------------------------------------------ */
    .st-key-panel_text_intake,
    .st-key-panel_text_report,
    .st-key-panel_doc_intake,
    .st-key-panel_doc_report {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 6px;
        box-shadow: 0 20px 44px -38px rgba(18, 42, 44, 0.55);
        padding: 1.1rem 1.15rem;
    }

    .panel-head {
        padding-bottom: 0.8rem;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid var(--line);
    }
    .panel-kicker {
        display: block;
        margin-bottom: 0.32rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.53rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--teal);
    }
    .panel-head h3 {
        margin: 0 0 0.16rem;
        font-size: 1.38rem;
        font-weight: 500 !important;
        letter-spacing: -0.015em;
    }
    .panel-head p {
        margin: 0;
        font-size: 0.76rem !important;
        line-height: 1.5 !important;
        color: var(--muted) !important;
    }

    .panel-note {
        margin-top: 0.8rem;
        padding: 0.6rem 0.7rem;
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 4px;
        font-size: 0.68rem !important;
        line-height: 1.55 !important;
        color: var(--muted) !important;
    }
    .panel-note b { color: var(--ink-2); font-weight: 600; }

    .input-meta {
        margin: 0.2rem 0 0;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.5rem !important;
        letter-spacing: 0.04em;
        color: var(--muted) !important;
        line-height: 1.5 !important;
    }
    .input-meta.right { text-align: right; }

    /* ------------------------------------------------------------------
       Widgets
       ------------------------------------------------------------------ */
    .stTextArea textarea {
        background: var(--surface-2) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        color: var(--ink) !important;
        font-family: 'Public Sans', sans-serif !important;
        font-size: 0.84rem !important;
        line-height: 1.7 !important;
        padding: 0.8rem 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(14, 90, 84, 0.11) !important;
    }
    .stTextArea textarea::placeholder { color: #8A9B9B !important; }

    .stTextArea label, .stSlider label, .stSelectbox label, .stFileUploader label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.55rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        font-weight: 500 !important;
    }
    .stTextArea label p, .stSlider label p, .stSelectbox label p,
    .stFileUploader label p {
        font-size: 0.55rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        font-weight: 500 !important;
    }

    .stButton > button,
    [data-testid="stBaseButton-secondary"] {
        min-height: 2.45rem !important;
        padding: 0.4rem 0.85rem !important;
        background: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        font-family: 'Public Sans', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease !important;
    }
    .stButton > button:hover {
        background: var(--teal-tint) !important;
        border-color: var(--teal) !important;
        color: var(--teal) !important;
        transform: none !important;
    }
    .stButton > button p {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: inherit !important;
    }

    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: var(--teal) !important;
        border-color: var(--teal) !important;
        color: #FFFFFF !important;
        box-shadow: 0 12px 26px -16px rgba(14, 90, 84, 0.68) !important;
    }
    .stButton > button[kind="primary"] p { color: #FFFFFF !important; }
    .stButton > button[kind="primary"]:hover {
        background: var(--teal-deep) !important;
        border-color: var(--teal-deep) !important;
    }
    .stButton > button[kind="primary"]:hover p { color: #FFFFFF !important; }
    .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(14, 90, 84, 0.2) !important;
    }

    .stDownloadButton > button,
    [data-testid="stDownloadButton"] button {
        min-height: 2.35rem !important;
        background: var(--surface) !important;
        color: var(--teal) !important;
        border: 1px solid rgba(14, 90, 84, 0.35) !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button p,
    [data-testid="stDownloadButton"] button p {
        color: var(--teal) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background: var(--teal-tint) !important;
        border-color: var(--teal) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        min-height: 2.45rem !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] input {
        color: var(--ink) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox svg { fill: var(--teal) !important; color: var(--teal) !important; }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] div[role="listbox"] {
        background: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div[role="option"] {
        background: var(--surface) !important;
        color: var(--ink) !important;
        font-size: 0.78rem !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover {
        background: var(--teal-tint) !important;
        color: var(--teal-deep) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--teal) !important;
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(14, 90, 84, 0.12) !important;
    }

    [data-testid="stFileUploaderDropzone"],
    .stFileUploader > section {
        background: var(--surface-2) !important;
        border: 1px dashed var(--line-strong) !important;
        border-radius: 4px !important;
        padding: 1.25rem 0.9rem !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    .stFileUploader > section:hover {
        border-color: var(--teal) !important;
        background: var(--teal-tint) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    .stFileUploader section span,
    .stFileUploader section small {
        color: var(--ink-2) !important;
        font-size: 0.7rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        color: var(--teal) !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 4px !important;
        background: var(--surface-2) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.55rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--ink-2) !important;
    }

    [data-testid="stAlert"] {
        border: 1px solid #E4D3A6 !important;
        border-radius: 4px !important;
        background: #FFF9EA !important;
        color: #5D4A1B !important;
        font-size: 0.76rem !important;
    }
    [data-testid="stAlert"] p { color: #5D4A1B !important; font-size: 0.76rem !important; }

    /* ------------------------------------------------------------------
       Sidebar
       ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background: var(--surface-2);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] > div { padding-top: 0.35rem; }
    section[data-testid="stSidebar"] * { color: var(--ink); }
    section[data-testid="stSidebar"] .stSelectbox label p,
    section[data-testid="stSidebar"] .stSlider label p {
        color: var(--muted) !important;
    }

    .side-brand {
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        padding: 0.95rem 0.9rem 0.9rem;
        background: var(--teal-deep);
        border-radius: 5px;
    }
    .side-brand::after {
        content: "";
        position: absolute;
        right: -34px; top: -42px;
        width: 128px; height: 128px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 50%;
        pointer-events: none;
    }
    .side-brand .sb-kicker {
        display: block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A9C6BF;
    }
    .side-brand .sb-name {
        display: block;
        margin: 0.32rem 0 0.18rem;
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.32rem;
        line-height: 1.1;
        color: #FFFFFF;
    }
    .side-brand .sb-name em { color: #BFE0D7; font-style: italic; }
    .side-brand .sb-sub {
        position: relative;
        z-index: 1;
        display: block;
        font-size: 0.66rem;
        line-height: 1.45;
        color: #C4D8D2;
    }

    .side-kicker {
        margin: 1.1rem 0 0.35rem;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.52rem !important;
        letter-spacing: 0.13em !important;
        line-height: 1.4 !important;
        text-transform: uppercase !important;
        color: var(--teal) !important;
    }

    .engine-card {
        margin: 0.35rem 0 0.55rem;
        padding: 0.65rem 0.7rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    .engine-state {
        display: flex;
        align-items: center;
        gap: 0.42rem;
        margin-bottom: 0.35rem;
        font-size: 0.68rem !important;
        font-weight: 600;
        color: var(--ink);
    }
    .engine-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--drug);
        box-shadow: 0 0 0 3px rgba(44, 106, 77, 0.12);
        flex: none;
    }
    .engine-dot.fallback {
        background: var(--anatomy);
        box-shadow: 0 0 0 3px rgba(138, 100, 32, 0.14);
    }
    .engine-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.5rem;
        padding: 0.18rem 0;
        font-size: 0.64rem !important;
    }
    .engine-row span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .engine-row b { font-weight: 600; text-align: right; font-size: 0.64rem !important; }

    .side-note {
        margin: 0.2rem 0 0.7rem;
        padding: 0.55rem 0.6rem;
        background: #FFF9EA;
        border: 1px solid #E7D6A9;
        border-radius: 4px;
        font-size: 0.63rem !important;
        line-height: 1.5 !important;
        color: #65511D !important;
    }
    .side-note code {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.56rem;
        color: #4E3F16;
        background: rgba(138, 100, 32, 0.08);
        padding: 0.05rem 0.18rem;
        border-radius: 2px;
    }

    .side-rule { height: 1px; background: var(--line); margin: 0.9rem 0 0.1rem; }

    .legend-row {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.42rem 0;
        border-bottom: 1px solid var(--line);
    }
    .legend-row:last-child { border-bottom: 0; }
    .legend-code {
        width: 26px;
        flex: none;
        padding: 0.12rem 0;
        text-align: center;
        background: var(--bg);
        border: 1px solid var(--line);
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.5rem;
        font-weight: 600;
        color: var(--c);
    }
    .legend-row b {
        display: block;
        font-size: 0.62rem !important;
        letter-spacing: 0.06em;
        color: var(--ink);
    }
    .legend-row p {
        margin: 0.06rem 0 0;
        font-size: 0.58rem !important;
        line-height: 1.4 !important;
        color: var(--muted) !important;
    }

    .about-box {
        margin: 0 0 0.55rem;
        font-size: 0.66rem !important;
        line-height: 1.55 !important;
        color: var(--muted) !important;
    }
    .about-box b { color: var(--ink); font-weight: 600; }

    .side-stamp {
        margin-top: 1rem;
        padding: 0.5rem 0.55rem;
        border: 1px solid var(--line-strong);
        border-radius: 3px;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
        line-height: 1.5;
    }

    /* Keep the sidebar expand control reachable when the toolbar is hidden. */
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        width: 32px;
        height: 32px;
        padding: 2px !important;
        background: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        box-shadow: 0 5px 14px -8px rgba(16, 32, 31, 0.35);
        color: var(--teal) !important;
    }
    [data-testid="stExpandSidebarButton"] button {
        width: 30px;
        height: 30px;
        min-height: 30px;
        padding: 0 !important;
        background: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 4px !important;
        color: var(--teal) !important;
    }
    [data-testid="stExpandSidebarButton"] svg {
        fill: var(--teal) !important;
        color: var(--teal) !important;
    }

    /* ------------------------------------------------------------------
       Report
       ------------------------------------------------------------------ */
    .report-meta {
        display: grid;
        grid-template-columns: 1.45fr 0.85fr 0.9fr 1.15fr;
        margin-bottom: 0.85rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }
    .report-meta > div {
        padding: 0.5rem 0.6rem;
        border-right: 1px solid var(--line);
        min-width: 0;
    }
    .report-meta > div:last-child { border-right: 0; }
    .report-meta span {
        display: block;
        margin-bottom: 0.16rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.46rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .report-meta b {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.65rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--ink);
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-bottom: 0.5rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }
    .metric {
        padding: 0.55rem 0.6rem 0.5rem;
        background: var(--surface);
        border-right: 1px solid var(--line);
        border-top: 2px solid var(--c);
    }
    .metric:last-child { border-right: 0; }
    .metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.28rem;
    }
    .metric-code {
        padding: 0.08rem 0.2rem;
        background: var(--bg);
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        font-weight: 600;
        color: var(--c);
    }
    .metric-label {
        overflow: hidden;
        text-overflow: ellipsis;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.45rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
        white-space: nowrap;
    }
    .metric-value {
        margin: 0.3rem 0 0.25rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        line-height: 1;
        color: var(--ink);
    }
    .metric-foot {
        display: flex;
        flex-direction: column;
        gap: 0.16rem;
    }
    .metric-foot span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.45rem;
        letter-spacing: 0.02em;
        color: var(--muted);
    }
    .metric-bar {
        display: block;
        width: 100%;
        height: 2px;
        background: var(--line);
        border-radius: 2px;
        overflow: hidden;
    }
    .metric-bar i { display: block; height: 100%; background: var(--c); }

    .report-tally {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.32rem 0.7rem;
        padding: 0.3rem 0 0.8rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.85rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.51rem;
        letter-spacing: 0.02em;
        color: var(--muted);
    }
    .report-tally b { color: var(--ink); font-weight: 600; }
    .report-tally .t-dot {
        width: 3px; height: 3px; border-radius: 50%;
        background: var(--line-strong);
    }

    .impression {
        margin-bottom: 0.9rem;
        padding: 0.65rem 0.75rem;
        background: var(--teal-tint);
        border-left: 2px solid var(--teal);
        border-radius: 0 3px 3px 0;
    }
    .impression span {
        display: block;
        margin-bottom: 0.2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--teal);
    }
    .impression p {
        margin: 0;
        font-size: 0.72rem;
        line-height: 1.55;
        color: var(--ink-2);
    }

    .control-note {
        margin-bottom: 0.7rem;
        padding: 0.5rem 0.6rem;
        background: #FFF9EA;
        border: 1px solid #E7D6A9;
        border-radius: 3px;
        font-size: 0.65rem;
        line-height: 1.45;
        color: #65511D;
    }

    .block-label {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.8rem;
        margin: 0.85rem 0 0.35rem;
    }
    .block-label span:first-child {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.52rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-2);
    }
    .block-label span:last-child {
        font-size: 0.58rem;
        color: var(--muted);
        text-align: right;
    }

    .ent-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin: 0.05rem 0 0.45rem;
    }
    .ent-legend span {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .ent-legend i {
        width: 5px; height: 5px; border-radius: 50%;
        background: var(--c); flex: none;
    }
    .ent-legend .legend-neg i {
        width: 9px; height: 5px;
        border-radius: 1px;
        background: transparent;
        border-bottom: 1px dashed currentColor;
    }

    .source-box {
        max-height: 330px;
        overflow: auto;
        padding: 0.9rem 0.95rem;
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 4px;
        font-size: 0.92rem;
        line-height: 2.05;
        color: var(--ink);
        white-space: pre-wrap;
    }
    .source-box mark {
        padding: 0.03rem 0.12rem;
        background: var(--bg);
        border-bottom: 2px solid var(--c);
        border-radius: 1px 1px 0 0;
        color: var(--ink);
        font-weight: 600;
    }
    .source-box mark.is-negated {
        background: transparent;
        border-bottom: 1px dashed var(--muted);
        color: var(--ink-2);
        font-weight: 500;
    }
    .source-box mark.is-negated::after {
        content: "neg";
        margin-left: 2px;
        vertical-align: super;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.42rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        color: var(--muted);
    }
    .source-box sup {
        margin-left: 1px;
        vertical-align: super;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.45rem;
        font-weight: 600;
        letter-spacing: 0;
        color: var(--c);
    }
    .source-box mark.is-negated sup { color: var(--muted); }

    .table-wrap {
        max-height: 350px;
        overflow: auto;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    table.findings {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.72rem;
        font-variant-numeric: tabular-nums;
    }
    table.findings thead th {
        position: sticky;
        top: 0;
        z-index: 1;
        padding: 0.42rem 0.45rem;
        background: #F1F5F3;
        border-bottom: 1px solid var(--line);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        font-weight: 500;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        text-align: left;
        white-space: nowrap;
    }
    table.findings td {
        padding: 0.38rem 0.45rem;
        border-bottom: 1px solid var(--line);
        color: var(--ink-2);
        vertical-align: middle;
    }
    table.findings tr:last-child td { border-bottom: 0; }
    table.findings tbody tr:hover td { background: #FAFCFB; }

    .cat {
        display: inline-block;
        padding: 0.1rem 0.26rem;
        background: var(--bg);
        border: 1px solid var(--line);
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.05em;
        color: var(--c);
        white-space: nowrap;
    }
    .find-word { font-weight: 600; color: var(--ink); }
    .neg-chip,
    .ctx-chip {
        display: inline-block;
        padding: 0.06rem 0.24rem;
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.42rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .neg-chip {
        background: #FAF3E4;
        border: 1px solid #E7D7B4;
        color: #7A5614;
    }
    .ctx-chip {
        background: transparent;
        border: 1px solid var(--line);
        color: var(--muted);
    }
    .span-range {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.54rem;
        color: var(--muted);
        white-space: nowrap;
    }
    .conf { display: flex; align-items: center; gap: 0.36rem; min-width: 82px; }
    .conf-track {
        display: block;
        width: 40px; height: 3px;
        background: var(--line);
        border-radius: 2px;
        overflow: hidden;
        flex: none;
    }
    .conf-track i { display: block; height: 100%; background: var(--c); }
    .conf b {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.52rem;
        font-weight: 500;
        color: var(--ink-2);
    }

    .confidence-note {
        margin: 0.45rem 0 0;
        padding-top: 0.4rem;
        border-top: 1px solid var(--line);
        font-size: 0.58rem !important;
        line-height: 1.5 !important;
        color: var(--muted) !important;
    }
    .confidence-note b { color: var(--ink-2); font-weight: 600; }

    .export-caption {
        margin: 0.5rem 0 0;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.47rem !important;
        letter-spacing: 0.03em;
        color: var(--muted) !important;
        text-align: center;
    }

    .meta-list { margin: 0; }
    .meta-item {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.68rem;
    }
    .meta-item:last-child { border-bottom: 0; }
    .meta-item span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.5rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .meta-item b { font-weight: 600; color: var(--ink); text-align: right; }

    .empty {
        padding: 2.1rem 1.1rem;
        background: var(--surface-2);
        border: 1px dashed var(--line-strong);
        border-radius: 5px;
        text-align: center;
    }
    .empty svg {
        width: 42px; height: 42px;
        margin-bottom: 0.65rem;
        color: var(--line-strong);
    }
    .empty-title {
        margin: 0 0 0.22rem;
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.22rem;
        color: var(--ink);
    }
    .empty-text {
        max-width: 310px;
        margin: 0 auto;
        font-size: 0.72rem;
        line-height: 1.55;
        color: var(--muted);
    }
    .empty-meta {
        margin-top: 0.85rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--teal);
    }

    .file-card {
        margin: 0.1rem 0 0.7rem;
        padding: 0.6rem 0.7rem;
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 4px;
    }
    .file-card .f-name {
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--ink);
        word-break: break-all;
    }
    .file-card .f-meta {
        margin-top: 0.18rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.5rem;
        letter-spacing: 0.03em;
        color: var(--muted);
    }
    .doc-preview {
        max-height: 185px;
        overflow: auto;
        margin-bottom: 0.7rem;
        padding: 0.7rem;
        background: var(--surface-2);
        border: 1px solid var(--line);
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        line-height: 1.65;
        color: var(--ink-2);
        white-space: pre-wrap;
    }

    /* ------------------------------------------------------------------
       Entity schema
       ------------------------------------------------------------------ */
    .schema-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin-bottom: 2.6rem;
    }
    .schema-card {
        padding: 0.8rem 0.8rem 0.75rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 2px solid var(--c);
        border-radius: 3px;
    }
    .schema-code {
        display: inline-block;
        padding: 0.08rem 0.24rem;
        background: var(--bg);
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem;
        font-weight: 600;
        color: var(--c);
    }
    .schema-card h3 {
        margin: 0.5rem 0 0.2rem;
        font-family: 'Public Sans', sans-serif !important;
        font-size: 0.65rem;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
    }
    .schema-card p {
        margin: 0 0 0.45rem;
        font-size: 0.66rem;
        line-height: 1.5;
        color: var(--muted);
    }
    .schema-ex {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem;
        line-height: 1.55;
        color: var(--ink-2);
    }

    /* ------------------------------------------------------------------
       Footer
       ------------------------------------------------------------------ */
    .site-foot {
        margin-top: 0.6rem;
        padding: 1.4rem 0 0.8rem;
        border-top: 1px solid var(--line);
    }
    .foot-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 2rem;
    }
    .foot-brand {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .foot-brand span {
        display: block;
        margin-top: 0.18rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.48rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .foot-meta {
        font-size: 0.6rem;
        line-height: 1.7;
        color: var(--muted);
        text-align: right;
    }
    .foot-bottom {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1.5rem;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--line);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.47rem;
        letter-spacing: 0.03em;
        line-height: 1.6;
        color: var(--muted);
    }
    .foot-bottom .disclaimer { max-width: 760px; }
    .foot-bottom .foot-tools { white-space: nowrap; }

    /* ------------------------------------------------------------------
       Responsive
       ------------------------------------------------------------------ */
    @media (max-width: 980px) {
        .hero {
            grid-template-columns: 1fr;
            gap: 1.6rem;
            padding: 2.3rem 0 2rem;
        }
        .model-card { max-width: 430px; }
        .protocol { grid-template-columns: 1fr; }
        .protocol-item {
            border-right: 0;
            border-bottom: 1px solid var(--line);
            padding: 1rem 0;
        }
        .protocol-item:last-child { border-bottom: 0; }
        .schema-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .head-nav { display: none; }
        .head-right { gap: 0.8rem; }
    }

    @media (max-width: 900px) {
        /* Anchor targets need clearance for the floating sidebar control. */
        #workspace,
        #schema,
        #method {
            scroll-margin-top: 3.6rem;
        }
        /* Streamlit keeps columns side-by-side by default; a clinical
           workbench needs the report to sit below its source on narrow screens. */
        [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem !important;
        }
    }

    @media (max-width: 640px) {
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            /* Clearance for the floating sidebar control above the strip. */
            padding: 3.05rem 1rem 2.2rem 1rem !important;
        }
        .metric-code,
        .metric-label,
        .metric-foot span,
        .table.findings thead th,
        .cat,
        .conf b {
            font-size: 0.55rem !important;
        }
        .report-tally,
        .block-label span:first-child,
        .block-label span:last-child,
        .input-meta,
        .confidence-note {
            font-size: 0.58rem !important;
        }
        .compliance,
        .sys-state {
            font-size: 0.55rem !important;
        }
        .compliance { font-size: 0.47rem; padding: 0.38rem 0.55rem; letter-spacing: 0.08em; }
        .compliance .c-right { display: none; }
        .site-head { padding: 0.7rem 0 0.8rem; }
        .brand-sub { display: none; }
        .brand-name { font-size: 1.05rem; }
        .brand-mark { width: 30px; height: 30px; }
        .sys-state { font-size: 0.47rem; padding: 0.23rem 0.45rem; letter-spacing: 0.05em; }
        .hero { padding: 1.55rem 0 1.35rem; }
        .hero h1 { font-size: 2.45rem; }
        .hero .lede { font-size: 0.9rem; }
        .model-card { padding: 0.95rem; }
        .sec-head {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.4rem;
            padding: 1.8rem 0 0.9rem;
        }
        .sec-head .sec-note { text-align: left; }
        .sec-head h2 { font-size: 1.75rem; }
        .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .report-meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .report-meta > div:nth-child(2) { border-right: 0; }
        .report-meta > div:nth-child(3) { border-top: 1px solid var(--line); }
        .report-meta > div:nth-child(4) { border-top: 1px solid var(--line); }
        .block-label { flex-direction: column; gap: 0.12rem; }
        .block-label span:last-child { text-align: left; }
        .source-box { font-size: 0.75rem; padding: 0.75rem; }
        .schema-grid { grid-template-columns: 1fr; }
        .foot-top { flex-direction: column; gap: 0.7rem; }
        .foot-meta { text-align: left; }
        .foot-bottom { flex-direction: column; gap: 0.45rem; }
        .export-caption { text-align: left; }
        .site-foot { margin-top: 0.3rem; }
    }

    @media (max-width: 380px) {
        .hero h1 { font-size: 2.4rem; }
        .brand-name { font-size: 0.98rem; }
        .sys-state { display: none; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Entity schema and example notes
# ---------------------------------------------------------------------------
ENTITY = {
    "DISEASE": {
        "code": "Dx",
        "name": "Disease",
        "desc": "Conditions, disorders and clinical diagnoses.",
        "examples": "diabetes · hypertension · carcinoma",
        "color": "#A13B3B",
        "bg": "#FAEDEC",
    },
    "DRUG": {
        "code": "Rx",
        "name": "Drug",
        "desc": "Medications, biologics and pharmaceutical agents.",
        "examples": "metformin · insulin · aspirin",
        "color": "#2C6A4D",
        "bg": "#ECF5EF",
    },
    "PROCEDURE": {
        "code": "Px",
        "name": "Procedure",
        "desc": "Tests, imaging, surgeries and therapeutic procedures.",
        "examples": "MRI · biopsy · dialysis",
        "color": "#2B5B8C",
        "bg": "#ECF2F9",
    },
    "ANATOMY": {
        "code": "An",
        "name": "Anatomy",
        "desc": "Organs, tissues and anatomical structures.",
        "examples": "heart · liver · kidney",
        "color": "#8A6420",
        "bg": "#FAF3E4",
    },
}

LABEL_ORDER = ["DISEASE", "DRUG", "PROCEDURE", "ANATOMY"]

EXAMPLES = [
    (
        "Endocrine",
        "The patient was diagnosed with type 2 diabetes mellitus and chronic "
        "kidney disease. Metformin was adjusted, and insulin was added. Blood "
        "work showed elevated glucose; an ultrasound of the kidneys was ordered "
        "and dialysis was discussed.",
    ),
    (
        "Oncology",
        "A 61-year-old patient presented with a tumor in the left lung. A biopsy "
        "confirmed carcinoma, and chemotherapy with cisplatin was initiated. CT "
        "scan showed no metastasis to the liver; the pancreas and kidneys were "
        "unremarkable.",
    ),
    (
        "Cardiology",
        "The patient reported chest pain and shortness of breath. An ECG showed "
        "no acute myocardial infarction, but an MRI of the heart was ordered. "
        "Aspirin and atorvastatin were started, and surgery was deferred.",
    ),
]

DEFAULT_NOTE = (
    "A 58-year-old patient with a history of type 2 diabetes mellitus and "
    "hypertension presented with chest pain and shortness of breath. Metformin "
    "and lisinopril were continued. An ECG showed no acute myocardial "
    "infarction, but an MRI of the heart was ordered to evaluate the "
    "myocardium. The patient was started on aspirin and atorvastatin; surgery "
    "was deferred."
)

MAX_DOC_CHARS = 200_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    """Escape text for safe inline HTML."""
    return html.escape(str(value))


def entity_stats(entities: list) -> dict:
    """Count entities and collect confidences per category."""
    stats = {label: {"count": 0, "conf": []} for label in LABEL_ORDER}
    for ent in entities:
        label = ent.get("entity")
        if label in stats:
            stats[label]["count"] += 1
            stats[label]["conf"].append(float(ent.get("confidence", 0.0)))
    return stats


def highlight_entities(text: str, entities: list) -> str:
    """Render annotation-style spans with a category superscript."""
    if not entities:
        return esc(text).replace("\n", "<br>")

    ordered = sorted(entities, key=lambda e: e["start"])
    parts = []
    cursor = 0

    for ent in ordered:
        start = max(0, min(int(ent["start"]), len(text)))
        end = max(start, min(int(ent["end"]), len(text)))
        if start < cursor:
            continue

        label = ent.get("entity", "DISEASE")
        meta = ENTITY.get(label, ENTITY["DISEASE"])
        conf = float(ent.get("confidence", 0.0))
        negated = bool(ent.get("negated"))
        neg_class = ' class="is-negated"' if negated else ""
        neg_note = " · negated context" if negated else ""

        parts.append(esc(text[cursor:start]))
        parts.append(
            f"<mark{neg_class} style=\"--c:{meta['color']};--bg:{meta['bg']};\" "
            f'title="{esc(label)} · {conf:.0%} confidence{neg_note}">'
            f"{esc(text[start:end])}<sup>{meta['code']}</sup></mark>"
        )
        cursor = end

    parts.append(esc(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def create_report(
    text: str,
    entities: list,
    model_requested: str,
    backend_label: str,
    threshold: float,
    source: str,
) -> dict:
    """Build the report record held in session state."""
    now = datetime.now()
    report_id = (
        "BNR-"
        + now.strftime("%Y%m%d")
        + "-"
        + hashlib.sha1(f"{text}{now.isoformat()}".encode("utf-8")).hexdigest()[:6].upper()
    )

    # Normalize every surface form to the exact source slice so the annotated
    # text, finding table and exports can never disagree about spelling.
    normalized = []
    for ent in entities:
        item = dict(ent)
        start = max(0, min(int(item.get("start", 0)), len(text)))
        end = max(start, min(int(item.get("end", start)), len(text)))
        item["start"] = start
        item["end"] = end
        item["word"] = text[start:end]
        normalized.append(item)

    return {
        "id": report_id,
        "created": now.strftime("%d %b %Y · %H:%M"),
        "created_iso": now.astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "text": text,
        "entities": normalized,
        "model_requested": model_requested,
        "backend": backend_label,
        "threshold": float(threshold),
        "source": source,
        "chars": len(text),
    }


def report_payload(report: dict) -> dict:
    """Structured export payload for JSON and CSV."""
    return {
        "report_id": report["id"],
        "generated_at": report["created"],
        "generated_at_iso": report["created_iso"],
        "source": report["source"],
        "engine_requested": report["model_requested"],
        "engine_active": report["backend"],
        "confidence_floor": report["threshold"],
        "characters": report["chars"],
        "entities": [
            {
                "category": ent["entity"],
                "text": ent["word"],
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "confidence": round(float(ent["confidence"]), 4),
                "negated": bool(ent.get("negated")),
            }
            for ent in report["entities"]
        ],
    }


def build_impression(entities: list, threshold: float) -> str:
    """A short, factual reading of the entity index."""
    if not entities:
        return (
            "No spans met the configured confidence floor. Lower the floor in the "
            "model console or add more clinical detail to the note."
        )

    stats = entity_stats(entities)
    counts = {label: data["count"] for label, data in stats.items() if data["count"]}
    if not counts:
        return (
            "Entities were returned outside the supported four-class schema. "
            "Review the engine label configuration before using this report."
        )
    top_label = max(counts, key=counts.get)
    unique = len({ent["word"].strip().lower() for ent in entities})
    mean_conf = sum(float(ent["confidence"]) for ent in entities) / len(entities)
    class_count = len(counts)

    class_phrase = "1 class" if class_count == 1 else f"{class_count} classes"
    top_phrase = "1 mention" if counts[top_label] == 1 else f"{counts[top_label]} mentions"
    negated = sum(1 for ent in entities if ent.get("negated"))
    negation_phrase = ""
    if negated:
        verb = "mention falls" if negated == 1 else "mentions fall"
        negation_phrase = (
            f" {negated} {verb} inside a negated context and "
            "should be reviewed against the source."
        )

    return (
        f"{len(entities)} entity mentions ({unique} unique surface forms) were "
        f"identified across {class_phrase}. {top_label.title()} is the most "
        f"frequent class with {top_phrase}. Mean span confidence is "
        f"{mean_conf:.0%}; all displayed spans meet the {threshold:.0%} floor."
        f"{negation_phrase}"
    )


def render_metrics(entities: list) -> None:
    """Four-category entity index."""
    stats = entity_stats(entities)
    cards = []

    for label in LABEL_ORDER:
        meta = ENTITY[label]
        data = stats[label]
        count = data["count"]
        avg = sum(data["conf"]) / len(data["conf"]) if data["conf"] else 0.0
        bar = f"{avg:.0%}" if count else "0%"
        foot = f"Avg confidence {avg:.0%}" if count else "No matches"

        cards.append(
            f'<div class="metric" style="--c:{meta["color"]};--bg:{meta["bg"]};">'
            f'<div class="metric-top">'
            f'<span class="metric-code">{meta["code"]}</span>'
            f'<span class="metric-label">{label}</span>'
            f"</div>"
            f'<div class="metric-value">{count}</div>'
            f'<div class="metric-foot"><span>{foot}</span>'
            f'<span class="metric-bar"><i style="width:{bar};"></i></span></div>'
            f"</div>"
        )

    st.markdown(f'<div class="metrics">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_empty_report() -> None:
    st.markdown(
        """
        <div class="empty">
            <svg viewBox="0 0 48 48" fill="none" stroke="currentColor"
                 stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true">
                <path d="M12 5h17l7 7v31H12z"/>
                <path d="M29 5v7h7"/>
                <path d="M18 21h13M18 27h13M18 33h8"/>
            </svg>
            <p class="empty-title">No report on file</p>
            <p class="empty-text">Generate a report to see the entity index,
            annotated source and exportable findings for this note.</p>
            <div class="empty-meta">Awaiting clinical text</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_report(
    report: dict,
    current_threshold: float,
    current_backend: str,
    current_text: str | None = None,
) -> None:
    """Render a persisted clinical report."""
    entities = report["entities"]
    changed = []
    if abs(float(report["threshold"]) - float(current_threshold)) > 1e-9:
        changed.append("confidence floor")
    if report["backend"] != current_backend:
        changed.append("engine")
    if current_text is not None and report["text"] != current_text:
        changed.append("source text")

    st.markdown(
        f"""
        <div class="report-meta">
            <div><span>Report ID</span><b>{esc(report["id"])}</b></div>
            <div><span>Engine</span><b>{esc(report["backend"])}</b></div>
            <div><span>Floor</span><b>{report["threshold"]:.0%}</b></div>
            <div><span>Generated</span><b>{esc(report["created"])}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if changed:
        st.markdown(
            f'<div class="control-note">Controls changed since this report was '
            f'generated ({esc(", ".join(changed))}). Regenerate to apply the new '
            f"settings.</div>",
            unsafe_allow_html=True,
        )

    if report["backend"] != "Rule-based demo":
        st.markdown(
            '<p class="confidence-note"><b>Sequence limit.</b> Transformer '
            "inference is bounded by the model maximum sequence length; the tail "
            "of very long inputs may not be analyzed.</p>",
            unsafe_allow_html=True,
        )

    render_metrics(entities)

    stats = entity_stats(entities)
    unique = len({ent["word"].strip().lower() for ent in entities})
    mean_conf = (
        sum(float(ent["confidence"]) for ent in entities) / len(entities)
        if entities
        else 0.0
    )
    class_count = sum(1 for data in stats.values() if data["count"])
    negated_count = sum(1 for ent in entities if ent.get("negated"))
    negation_tally = (
        f'<span class="t-dot"></span><span><b>{negated_count}</b> negated</span>'
        if negated_count
        else ""
    )

    st.markdown(
        f"""
        <div class="report-tally">
            <span><b>{len(entities)}</b> mentions</span>
            <span class="t-dot"></span>
            <span><b>{unique}</b> unique</span>
            <span class="t-dot"></span>
            <span><b>{class_count}</b> classes</span>
            <span class="t-dot"></span>
            <span><b>{mean_conf:.0%}</b> mean confidence</span>
            {negation_tally}
            <span class="t-dot"></span>
            <span><b>{report["chars"]:,}</b> characters in source</span>
        </div>
        <div class="impression">
            <span>Impression</span>
            <p>{esc(build_impression(entities, report["threshold"]))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not entities:
        st.markdown(
            '<div class="panel-note"><b>No qualifying spans.</b> The text was '
            "processed, but no entity crossed the confidence floor. Lower the "
            "floor in the model console or add more clinical detail.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="block-label"><span>Annotated source</span>'
            "<span>Span-level extraction · category code shown in superscript</span></div>",
            unsafe_allow_html=True,
        )
        legend = "".join(
            f'<span style="--c:{ENTITY[label]["color"]};">'
            f'<i></i>{ENTITY[label]["code"]} · {ENTITY[label]["name"]}</span>'
            for label in LABEL_ORDER
        )
        if negated_count:
            legend += (
                '<span class="legend-neg" style="--c:var(--muted);">'
                "<i></i>Dashed · negated context</span>"
            )
        st.markdown(f'<div class="ent-legend">{legend}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="source-box">{highlight_entities(report["text"], entities)}</div>',
            unsafe_allow_html=True,
        )

        rows = []
        for index, ent in enumerate(entities, 1):
            label = ent.get("entity", "DISEASE")
            meta = ENTITY.get(label, ENTITY["DISEASE"])
            conf = float(ent.get("confidence", 0.0))
            negated = bool(ent.get("negated"))
            context = (
                '<span class="neg-chip">Negated</span>'
                if negated
                else '<span class="ctx-chip">Asserted</span>'
            )
            rows.append(
                f"<tr>"
                f'<td class="span-range">{index:02d}</td>'
                f'<td><span class="cat" style="--c:{meta["color"]};--bg:{meta["bg"]};">'
                f"{esc(label)}</span></td>"
                f'<td class="find-word">{esc(ent["word"])}</td>'
                f'<td class="span-range">{int(ent["start"])}–{int(ent["end"])}</td>'
                f'<td><div class="conf"><span class="conf-track" '
                f'style="--c:{meta["color"]};"><i style="width:{conf:.0%};"></i></span>'
                f"<b>{conf:.0%}</b></div></td>"
                f"<td>{context}</td>"
                f"</tr>"
            )

        st.markdown(
            '<div class="block-label"><span>Finding index</span>'
            f"<span>{len(entities)} annotated spans</span></div>"
            '<div class="table-wrap"><table class="findings">'
            "<thead><tr>"
            '<th scope="col">#</th><th scope="col">Category</th>'
            '<th scope="col">Surface form</th><th scope="col">Span</th>'
            '<th scope="col">Confidence</th><th scope="col">Context</th>'
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
            # Confidence semantics are stated explicitly so the score is never
            # mistaken for a calibrated probability of clinical correctness.
            '<p class="confidence-note"><b>Confidence bands:</b> High ≥ 90% · '
            "Review 70–89% · Low &lt; 70%. A score is model or rule confidence, "
            "not a calibrated probability of clinical correctness."
            + (
                " Negation flags are a demonstration heuristic, not a clinical "
                "assertion classifier."
                if negated_count
                else ""
            )
            + "</p>",
            unsafe_allow_html=True,
        )

    # Exports and audit metadata stay available even when no span qualifies.
    payload = report_payload(report)
    export_rows = [
        {
            "report_id": report["id"],
            "category": ent["entity"],
            "text": ent["word"],
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "confidence": round(float(ent["confidence"]), 4),
            "negated": bool(ent.get("negated")),
        }
        for ent in entities
    ]
    frame = pd.DataFrame(
        export_rows,
        columns=[
            "report_id",
            "category",
            "text",
            "start",
            "end",
            "confidence",
            "negated",
        ],
    )
    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Export CSV",
            data=csv_bytes,
            file_name=f"{report['id']}.csv",
            mime="text/csv",
            key=f"csv_{report['id']}",
            width="stretch",
        )
    with export_right:
        st.download_button(
            "Export JSON",
            data=json_bytes,
            file_name=f"{report['id']}.json",
            mime="application/json",
            key=f"json_{report['id']}",
            width="stretch",
        )
    st.markdown(
        '<p class="export-caption">CSV for spreadsheets · JSON for downstream review</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Report metadata"):
        st.markdown(
            f"""
            <div class="meta-list">
                <div class="meta-item"><span>Report ID</span><b>{esc(report["id"])}</b></div>
                <div class="meta-item"><span>Source</span><b>{esc(report["source"])}</b></div>
                <div class="meta-item"><span>Engine requested</span><b>{esc(report["model_requested"])}</b></div>
                <div class="meta-item"><span>Engine active</span><b>{esc(report["backend"])}</b></div>
                <div class="meta-item"><span>Confidence floor</span><b>{report["threshold"]:.0%}</b></div>
                <div class="meta-item"><span>Characters in source</span><b>{report["chars"]:,}</b></div>
                <div class="meta-item"><span>Annotated spans</span><b>{len(entities)}</b></div>
                <div class="meta-item"><span>Negated spans</span><b>{negated_count}</b></div>
                <div class="meta-item"><span>Generated</span><b>{esc(report["created"])}</b></div>
                <div class="meta-item"><span>Recorded (ISO)</span><b>{esc(report["created_iso"])}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View structured payload"):
        st.code(json.dumps(payload, indent=2), language="json")


@st.cache_resource(show_spinner="Preparing clinical engine…")
def load_predictor(option: str):
    if option == "BioBERT · fine-tuned" and FINE_TUNED_AVAILABLE:
        return NERPredictor(
            model_path=str(MODEL_DIR),
            model_name="dmis-lab/biobert-base-cased-v1.1",
        )
    return NERPredictor(model_path=None, model_name=None, rule_based=True)


# ---------------------------------------------------------------------------
# Session callbacks
# ---------------------------------------------------------------------------
def load_example(text: str) -> None:
    st.session_state["clinical_text"] = text
    st.session_state.pop("single_report", None)


def clear_single() -> None:
    st.session_state["clinical_text"] = ""
    st.session_state.pop("single_report", None)


def clear_document_report() -> None:
    st.session_state.pop("doc_report", None)


if "clinical_text" not in st.session_state:
    st.session_state["clinical_text"] = DEFAULT_NOTE


# ---------------------------------------------------------------------------
# Sidebar — model console
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <span class="sb-kicker">Biomedical Language Laboratory</span>
            <span class="sb-name">BioNER <em>Clinical</em></span>
            <span class="sb-sub">Model console for biomedical entity extraction.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="side-kicker">Model console</p>', unsafe_allow_html=True)

    model_option = st.selectbox(
        "Engine",
        ["BioBERT · fine-tuned", "Rule-based demo"],
        index=0 if FINE_TUNED_AVAILABLE else 1,
        key="engine_select",
        help="Fine-tuned uses weights in models/biobert_ner. The rule-based demo "
        "runs instantly and is the intended fallback when weights are absent.",
    )

    conf_pct = st.slider(
        "Confidence floor",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        format="%d%%",
        key="confidence_floor",
        help="Spans below this floor are excluded from the report.",
    )
    conf_thresh = conf_pct / 100.0

    predictor = load_predictor(model_option)
    is_fallback = bool(predictor.rule_based)
    backend_label = "Rule-based demo" if is_fallback else "BioBERT · fine-tuned"
    engine_state = "Rule-based fallback" if is_fallback else "Transformer loaded"
    dot_class = "fallback" if is_fallback else ""

    st.markdown(
        f"""
        <div class="engine-card">
            <div class="engine-state">
                <span class="engine-dot {dot_class}"></span>{engine_state}
            </div>
            <div class="engine-row"><span>Backend</span><b>{esc(backend_label)}</b></div>
            <div class="engine-row"><span>Runtime</span><b>Local · CPU</b></div>
            <div class="engine-row"><span>Confidence floor</span><b>{conf_pct}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_fallback and model_option == "BioBERT · fine-tuned":
        if not FINE_TUNED_AVAILABLE:
            st.markdown(
                '<div class="side-note">Fine-tuned weights were not found at '
                "<code>models/biobert_ner</code>. The rule-based demo is active. "
                "Train the model to enable transformer inference.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="side-note">The fine-tuned model could not be loaded. '
                "The rule-based demo is active for this session.</div>",
                unsafe_allow_html=True,
            )
    elif is_fallback:
        st.markdown(
            '<div class="side-note">Rule-based demo: deterministic pattern matching. '
            "Scores reflect pattern specificity (0.90–0.96), not calibrated model "
            "probability.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown('<p class="side-kicker">Entity schema</p>', unsafe_allow_html=True)

    legend_rows = "".join(
        f'<div class="legend-row" style="--c:{ENTITY[label]["color"]};'
        f'--bg:{ENTITY[label]["bg"]};">'
        f'<span class="legend-code">{ENTITY[label]["code"]}</span>'
        f"<div><b>{label}</b><p>{ENTITY[label]['desc']}</p></div></div>"
        for label in LABEL_ORDER
    )
    st.markdown(legend_rows, unsafe_allow_html=True)

    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown('<p class="side-kicker">About this interface</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="about-box"><b>Project 16</b> · Biomedical named entity recognition. '
        "Extracts diseases, drugs, procedures and anatomy from clinical text. Built "
        "with BioBERT and Streamlit for the NLP Capstone, Semester 5.</p>"
        '<p class="about-box"><b>Group 8</b><br>Ayush Singh · Krish Jaidka · '
        "Sahil Shrivastava</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-stamp">Research use only<br>Not a medical device</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Site header
# ---------------------------------------------------------------------------
engine_short = "Rule-based" if is_fallback else "BioBERT"
st.markdown(
    f"""
    <div class="compliance">
        <div class="c-left"><span class="c-dot"></span>
            Research interface · outputs require clinical review
        </div>
        <div class="c-right">Do not enter PHI</div>
    </div>
    <div class="site-head">
        <div class="brand">
            <div class="brand-mark">
                <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5"
                     stroke-linecap="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="8.5"/>
                    <path d="M12 3v18M3 12h18"/>
                </svg>
            </div>
            <div>
                <span class="brand-name">BioNER <em>Clinical</em></span>
                <span class="brand-sub">Biomedical Language Laboratory</span>
            </div>
        </div>
        <div class="head-right">
            <nav class="head-nav">
                <a href="#workspace">Workspace</a>
                <a href="#schema">Entity schema</a>
                <a href="#method">Method</a>
            </nav>
            <span class="sys-state">
                <span class="sys-dot {dot_class}"></span>Local engine · {engine_short}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hero_tags = "".join(
    f'<span class="hero-tag" style="--c:{ENTITY[label]["color"]};">'
    f"<i></i>{label}</span>"
    for label in LABEL_ORDER
)

if is_fallback:
    card_status = "Fallback"
    card_status_class = " fallback"
    card_title = "Rule-based"
    card_sub = "Deterministic demo engine"
    card_arch = "Pattern lexicon"
    card_task = "Pattern matching"
else:
    card_status = "Active"
    card_status_class = ""
    card_title = "BioBERT"
    card_sub = "Biomedical token classification"
    card_arch = "BioBERT-base"
    card_task = "Token classification"

st.markdown(
    f"""
    <section class="hero">
        <div class="hero-copy">
            <p class="eyebrow">Biomedical named entity recognition</p>
            <h1>Every entity,<br><em>with its evidence.</em></h1>
            <p class="lede">
                Paste a clinical note or abstract. Inspect spans, labels,
                confidence and provenance — then export a structured report
                for review.
            </p>
            <div class="hero-tags">{hero_tags}</div>
        </div>
        <aside class="model-card">
            <div class="mc-head"><span>Model card</span>
                <span class="mc-live{card_status_class}">{card_status}</span></div>
            <div class="mc-title">{card_title}</div>
            <div class="mc-sub">{card_sub}</div>
            <div class="mc-rows">
                <div class="mc-row"><span>Architecture</span><b>{card_arch}</b></div>
                <div class="mc-row"><span>Task</span><b>{card_task}</b></div>
                <div class="mc-row"><span>Label set</span><b>4 classes · 8 BIO</b></div>
                <div class="mc-row"><span>Output</span><b>Report · CSV · JSON</b></div>
                <div class="mc-row"><span>Engine</span><b>{esc(backend_label)}</b></div>
            </div>
            <div class="mc-foot">
                <div class="barcode" aria-hidden="true"></div>
                <span>BN-016 · REV 1.0</span>
            </div>
        </aside>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Protocol strip
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="protocol" id="method">
        <div class="protocol-item">
            <span class="protocol-num">01</span>
            <div><h4>Intake</h4>
            <p>Paste a clinical note or upload a plain-text document.</p></div>
        </div>
        <div class="protocol-item">
            <span class="protocol-num">02</span>
            <div><h4>Annotate</h4>
            <p>Extract span-level entities with category and confidence.</p></div>
        </div>
        <div class="protocol-item">
            <span class="protocol-num">03</span>
            <div><h4>Review</h4>
            <p>Inspect the finding index, then export CSV or JSON.</p></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Workspace heading
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="sec-head" id="workspace">
        <div>
            <p class="eyebrow">Clinical workspace</p>
            <h2>Build a structured report</h2>
        </div>
        <p class="sec-note">Local session<br>Local inference · {conf_pct}% floor</p>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_text, tab_doc = st.tabs(["Text intake", "Document intake"])

# ---------------------------------------------------------------------------
# Tab 1 — single clinical text
# ---------------------------------------------------------------------------
with tab_text:
    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        with st.container(key="panel_text_intake"):
            st.markdown(
                """
                <div class="panel-head">
                    <span class="panel-kicker">01 · Intake</span>
                    <h3>Clinical text</h3>
                    <p>Paste a note, discharge summary or biomedical excerpt.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            example_cols = st.columns(3)
            for index, (label, sample) in enumerate(EXAMPLES):
                with example_cols[index]:
                    st.button(
                        label,
                        key=f"example_{index}",
                        on_click=load_example,
                        args=(sample,),
                        width="stretch",
                    )

            text = st.text_area(
                "Clinical note",
                height=250,
                key="clinical_text",
                label_visibility="collapsed",
                placeholder="e.g. The patient presented with chest pain and was started on aspirin...",
            )

            st.markdown(
                f'<p class="input-meta">{len(text):,} characters · plain text · '
                "local session · no retention</p>",
                unsafe_allow_html=True,
            )

            button_left, button_right = st.columns([2.15, 1])
            with button_left:
                generate_single = st.button(
                    "Generate report",
                    type="primary",
                    key="generate_single",
                    width="stretch",
                )
            with button_right:
                st.button(
                    "Clear",
                    key="clear_single",
                    on_click=clear_single,
                    width="stretch",
                )

            if generate_single:
                if not text.strip():
                    st.warning("Enter a clinical note before generating a report.")
                else:
                    with st.spinner("Annotating clinical spans…"):
                        entities = predictor.predict(text, conf_thresh=conf_thresh)
                    st.session_state["single_report"] = create_report(
                        text=text,
                        entities=entities,
                        model_requested=model_option,
                        backend_label=backend_label,
                        threshold=conf_thresh,
                        source="Clinical note",
                    )

            st.markdown(
                f'<div class="panel-note"><b>Confidence floor {conf_pct}%.</b> '
                "Every span in the report clears this floor. Adjust it in the model "
                "console. <b>Do not enter protected health information.</b> Text is "
                "processed locally and is not retained outside this session.</div>",
                unsafe_allow_html=True,
            )

    with right:
        with st.container(key="panel_text_report"):
            st.markdown(
                """
                <div class="panel-head">
                    <span class="panel-kicker">02 · Report</span>
                    <h3>Structured findings</h3>
                    <p>Entity index, annotated source and exportable findings.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            single_report = st.session_state.get("single_report")
            if single_report is None:
                render_empty_report()
            else:
                render_report(single_report, conf_thresh, backend_label, text)


# ---------------------------------------------------------------------------
# Tab 2 — document intake
# ---------------------------------------------------------------------------
with tab_doc:
    doc_text = None
    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        with st.container(key="panel_doc_intake"):
            st.markdown(
                """
                <div class="panel-head">
                    <span class="panel-kicker">01 · Document intake</span>
                    <h3>Upload a document</h3>
                    <p>Plain-text clinical notes or biomedical research excerpts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Upload document",
                type=["txt"],
                key="doc_upload",
                label_visibility="collapsed",
                max_upload_size=5,
                help="Upload a UTF-8 plain-text file up to 5 MB containing clinical or biomedical text.",
            )

            if uploaded is not None:
                raw_bytes = uploaded.getvalue()
                doc_text = raw_bytes.decode("utf-8", errors="replace")
                replacement_chars = doc_text.count("\ufffd")
                encoding_label = (
                    "UTF-8" if replacement_chars == 0 else "UTF-8 · replacement characters"
                )
                size_kb = len(raw_bytes) / 1024
                size_label = (
                    f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
                )

                st.markdown(
                    f"""
                    <div class="file-card">
                        <div class="f-name">{esc(uploaded.name)}</div>
                        <div class="f-meta">{size_label} · {len(doc_text):,} characters · {encoding_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if replacement_chars:
                    st.markdown(
                        f'<p class="input-meta">{replacement_chars:,} bytes were not '
                        "valid UTF-8 and were replaced. Check the source encoding "
                        "before relying on this report.</p>",
                        unsafe_allow_html=True,
                    )

                preview = doc_text[:1400]
                st.markdown(
                    f'<div class="doc-preview">{esc(preview).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                if len(doc_text) > len(preview):
                    st.markdown(
                        f'<p class="input-meta">Preview truncated · showing first '
                        f"{len(preview):,} of {len(doc_text):,} characters</p>",
                        unsafe_allow_html=True,
                    )

                generate_doc = st.button(
                    "Generate report",
                    type="primary",
                    key="generate_doc",
                    width="stretch",
                )

                if generate_doc:
                    if not doc_text.strip():
                        st.warning("The uploaded document is empty.")
                    elif len(doc_text) > MAX_DOC_CHARS:
                        st.warning(
                            f"The document exceeds {MAX_DOC_CHARS:,} characters. "
                            "Upload a shorter excerpt."
                        )
                    else:
                        with st.spinner("Processing the document…"):
                            entities = predictor.predict(doc_text, conf_thresh=conf_thresh)
                        st.session_state["doc_report"] = create_report(
                            text=doc_text,
                            entities=entities,
                            model_requested=model_option,
                            backend_label=backend_label,
                            threshold=conf_thresh,
                            source=f"Document · {uploaded.name}",
                        )

                if st.session_state.get("doc_report") is not None:
                    st.button(
                        "Clear report",
                        key="clear_doc",
                        on_click=clear_document_report,
                        width="stretch",
                    )
            else:
                st.markdown(
                    '<div class="panel-note"><b>Accepted format.</b> UTF-8 plain text '
                    f"(.txt), up to {MAX_DOC_CHARS:,} characters. Documents are "
                    "processed locally and are not retained outside this session. "
                    "<b>Do not upload protected health information.</b></div>",
                    unsafe_allow_html=True,
                )
                if st.session_state.get("doc_report") is not None:
                    st.button(
                        "Clear previous report",
                        key="clear_doc_empty",
                        on_click=clear_document_report,
                        width="stretch",
                    )

            st.markdown(
                f'<div class="panel-note"><b>Confidence floor {conf_pct}%.</b> '
                "Every span in the report clears this floor. Adjust it in the model "
                "console.</div>",
                unsafe_allow_html=True,
            )

    with right:
        with st.container(key="panel_doc_report"):
            st.markdown(
                """
                <div class="panel-head">
                    <span class="panel-kicker">02 · Report</span>
                    <h3>Structured findings</h3>
                    <p>Entity index, annotated source and exportable findings.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            doc_report = st.session_state.get("doc_report")
            if doc_report is None:
                render_empty_report()
            else:
                if uploaded is not None:
                    expected_source = f"Document · {uploaded.name}"
                    if doc_report["source"] != expected_source:
                        st.markdown(
                            '<div class="control-note">This report belongs to a '
                            f'different document. Generate a new report to replace it.'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                render_report(doc_report, conf_thresh, backend_label, doc_text)


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------
schema_cards = "".join(
    f'<div class="schema-card" style="--c:{ENTITY[label]["color"]};'
    f'--bg:{ENTITY[label]["bg"]};">'
    f'<span class="schema-code">{ENTITY[label]["code"]}</span>'
    f"<h3>{label}</h3>"
    f"<p>{ENTITY[label]['desc']}</p>"
    f'<div class="schema-ex">{ENTITY[label]["examples"]}</div>'
    f"</div>"
    for label in LABEL_ORDER
)

st.markdown(
    f"""
    <div class="sec-head" id="schema">
        <div>
            <p class="eyebrow">Entity schema</p>
            <h2>Four clinical classes</h2>
        </div>
        <p class="sec-note">8 BIO transition tags<br>Span-level annotation</p>
    </div>
    <div class="schema-grid">{schema_cards}</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <footer class="site-foot">
        <div class="foot-top">
            <div class="foot-brand">BioNER Clinical
                <span>Biomedical Language Laboratory</span>
            </div>
            <div class="foot-meta">
                Project 16 · Group 8<br>
                Ayush Singh · Krish Jaidka · Sahil Shrivastava
            </div>
        </div>
        <div class="foot-bottom">
            <span class="disclaimer">Research and educational use only. Not a
            medical device. Do not enter protected health information. Outputs are
            assistive and must be reviewed by a qualified professional before any
            clinical use.</span>
            <span class="foot-tools">{esc(backend_label)} · Streamlit</span>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
