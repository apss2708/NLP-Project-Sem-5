"""Streamlit GUI for Sentiment Analysis (Project 1).

Visual identity: "Reel Verdict" — a cinematic editorial screening room.
Near-black film print, antique gold, serif display type, scorecards and
critic-consensus meters. Deliberately distinct from the BioNER Studio theme.
"""

import html
import os

import pandas as pd
import streamlit as st

from src.inference import get_predictor


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reel Verdict — Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300..700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --rv-bg: #0a0908;
        --rv-panel: #12100d;
        --rv-panel-2: #14110e;
        --rv-cream: #f3ede2;
        --rv-muted: #a69b8a;
        --rv-dim: #9a8e7c;
        --rv-gold: #d9b36c;
        --rv-gold-bright: #f0d29a;
        --rv-line: rgba(217, 179, 108, 0.16);
        --rv-line-soft: rgba(217, 179, 108, 0.09);
        --rv-pos: #8fb573;
        --rv-neg: #d15a4e;
        --rv-neu: #d8ac57;
        --rv-r: 3px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1100px 620px at 82% -12%, rgba(217, 179, 108, 0.10), transparent 62%),
            radial-gradient(900px 520px at -5% 25%, rgba(209, 90, 78, 0.05), transparent 60%),
            var(--rv-bg);
        color: var(--rv-cream);
    }

    /* Film grain over the whole print */
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 3;
        pointer-events: none;
        opacity: 0.032;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23g)' opacity='0.6'/%3E%3C/svg%3E");
    }

    .main .block-container {
        max-width: 1220px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
    }

    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    html { color-scheme: dark; }
    body, p, li, span, label { color: var(--rv-cream); }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-weight: 400 !important;
    }

    ::selection { background: rgba(217, 179, 108, 0.28); color: #fffaf1; }

    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: #0c0a08; }
    ::-webkit-scrollbar-thumb { background: #2e271d; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #453a29; }

    /* ------------------------------------------------------------------
       Top bar + ticker
    ------------------------------------------------------------------ */
    .rv-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 0.35rem 0 0.95rem;
        border-bottom: 1px solid var(--rv-line);
    }

    .rv-brand {
        font-family: 'DM Mono', monospace;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.32em;
        color: var(--rv-gold-bright);
        white-space: nowrap;
    }

    .rv-brand span { color: #5f5547; padding: 0 0.25rem; }

    .rv-topbar-meta {
        display: flex;
        align-items: center;
        gap: 1.6rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.2em;
        color: var(--rv-dim);
        text-transform: uppercase;
    }

    .rv-live { display: inline-flex; align-items: center; gap: 0.45rem; color: #bbae9b; }
    .rv-live i {
        width: 5px; height: 5px; border-radius: 50%;
        background: var(--rv-neg);
        box-shadow: 0 0 0 4px rgba(209, 90, 78, 0.12);
        animation: rvPulse 2.6s ease-in-out infinite;
    }

    @keyframes rvPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    .rv-ticker {
        overflow: hidden;
        border-bottom: 1px solid var(--rv-line-soft);
        padding: 0.55rem 0;
        -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 4%, #000 96%, transparent 100%);
        mask-image: linear-gradient(90deg, transparent 0, #000 4%, #000 96%, transparent 100%);
    }

    .rv-ticker-track {
        display: flex;
        align-items: center;
        gap: 2rem;
        width: max-content;
        white-space: nowrap;
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.26em;
        text-transform: uppercase;
        color: #857967;
        animation: rvMarquee 46s linear infinite;
    }

    .rv-ticker-track em { color: #8a785c; font-style: normal; font-size: 0.5rem; }
    .rv-ticker-track b { color: var(--rv-gold); font-weight: 400; }

    @keyframes rvMarquee {
        from { transform: translateX(0); }
        to { transform: translateX(-50%); }
    }

    /* ------------------------------------------------------------------
       Hero
    ------------------------------------------------------------------ */
    .rv-hero {
        display: grid;
        grid-template-columns: 1.4fr 0.82fr;
        gap: 2.7rem;
        align-items: center;
        padding: 2.1rem 0 2rem;
        border-bottom: 1px solid var(--rv-line);
    }

    .rv-eyebrow, .rv-kicker {
        margin: 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 0.26em;
        text-transform: uppercase;
        color: var(--rv-gold);
    }

    .rv-hero h1 {
        margin: 0.7rem 0 1.1rem;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: clamp(2.9rem, 4.4vw, 4.15rem);
        font-weight: 400 !important;
        line-height: 0.98;
        letter-spacing: -0.015em;
        color: #fbf6ec;
    }

    .rv-hero h1 em {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-style: italic;
        color: var(--rv-gold-bright);
    }

    .rv-lede {
        max-width: 41rem;
        margin: 0 0 1.35rem;
        font-size: 0.98rem;
        font-weight: 300;
        line-height: 1.7;
        color: #bdb2a1;
    }

    .rv-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 2.4rem;
        padding-top: 1rem;
        border-top: 1px solid var(--rv-line-soft);
    }

    .rv-meta span {
        display: block;
        margin-bottom: 0.3rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.52rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    .rv-meta b {
        font-family: 'DM Mono', monospace;
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #e4d9c7;
    }

    /* Poster card */
    .rv-poster {
        position: relative;
        padding: 1.15rem 1.3rem 1.25rem;
        border: 1px solid rgba(217, 179, 108, 0.28);
        border-radius: 2px;
        background:
            radial-gradient(400px 260px at 90% 0%, rgba(217, 179, 108, 0.10), transparent 70%),
            linear-gradient(158deg, #1a1610 0%, #0d0b09 72%);
        box-shadow: 0 46px 90px -46px rgba(0, 0, 0, 0.95);
        transform: rotate(1.1deg);
    }

    .rv-poster::before {
        content: "";
        position: absolute;
        inset: 7px;
        border: 1px solid rgba(217, 179, 108, 0.12);
        pointer-events: none;
    }

    .rv-perf {
        height: 7px;
        margin: 0.35rem 0;
        background-image: repeating-linear-gradient(
            90deg,
            rgba(217, 179, 108, 0.42) 0 5px,
            transparent 5px 12px
        );
        opacity: 0.4;
    }

    .rv-poster-top {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8b7f6c;
    }

    .rv-poster h2 {
        margin: 0 0 0.55rem;
        font-size: 2.25rem;
        line-height: 1;
        color: #f8f1e5;
    }

    .rv-poster h2 em { color: var(--rv-gold); font-style: italic; }

    .rv-stars {
        font-size: 0.95rem;
        letter-spacing: 0.24em;
        color: var(--rv-gold);
        line-height: 1;
    }

    .rv-stars.small { font-size: 0.8rem; letter-spacing: 0.2em; margin: 0.35rem 0 0.15rem; }

    .rv-poster-rule {
        height: 1px;
        margin: 1.1rem 0 0.7rem;
        background: var(--rv-line);
    }

    .rv-poster-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.28rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #8a7e6c;
    }

    .rv-poster-row span:last-child { color: #ded3c1; text-align: right; }

    .rv-poster-foot {
        display: flex;
        justify-content: space-between;
        font-family: 'DM Mono', monospace;
        font-size: 0.52rem;
        letter-spacing: 0.16em;
        color: #938875;
    }

    .rv-stamp {
        display: block;
        width: fit-content;
        margin: 1rem 0 0.15rem auto;
        transform: rotate(-4deg);
        padding: 0.4rem 0.6rem;
        border: 1.5px solid rgba(209, 90, 78, 0.65);
        border-radius: 2px;
        background: rgba(10, 9, 8, 0.82);
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        line-height: 1.5;
        letter-spacing: 0.2em;
        text-align: center;
        text-transform: uppercase;
        color: #de7a6e;
    }

    /* ------------------------------------------------------------------
       Section head
    ------------------------------------------------------------------ */
    .rv-section-head {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin: 1.8rem 0 0.4rem;
    }

    .rv-section-head .rv-num {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        color: var(--rv-gold);
    }

    .rv-section-head h2 {
        margin: 0;
        font-size: 1.65rem;
        line-height: 1;
        color: #f7f0e4;
    }

    .rv-section-head .rv-rule {
        flex: 1;
        height: 1px;
        background: var(--rv-line);
    }

    .rv-section-head .rv-note {
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    /* ------------------------------------------------------------------
       Tabs — editorial underlines, no pills
    ------------------------------------------------------------------ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.2rem;
        background: transparent;
        border-bottom: 1px solid var(--rv-line);
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 0.7rem 0 0.75rem;
        height: auto;
    }

    .stTabs [data-baseweb="tab"] p {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.62rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase;
        color: #938875 !important;
    }

    .stTabs [aria-selected="true"] { background: transparent; }

    .stTabs [aria-selected="true"] p { color: var(--rv-gold-bright) !important; }

    .stTabs [data-baseweb="tab-highlight"] { background: var(--rv-gold); height: 1px; }

    .stTabs [data-baseweb="tab-border"] { background: transparent; height: 1px; }

    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

    /* ------------------------------------------------------------------
       Panels — dark, sharp, gold hairline. No white cards.
    ------------------------------------------------------------------ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        position: relative;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.018), transparent 45%),
            linear-gradient(180deg, #13110d 0%, #0e0c09 100%);
        border: 1px solid #2a241c;
        border-radius: var(--rv-r);
        box-shadow: 0 32px 60px -48px rgba(0, 0, 0, 0.9);
        padding: 1.05rem 1.1rem 1.15rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, var(--rv-gold), rgba(217, 179, 108, 0.18) 45%, transparent 85%);
        opacity: 0.6;
        pointer-events: none;
    }

    .rv-panel-title {
        margin: 0.15rem 0 0.2rem;
        font-size: 1.5rem;
        line-height: 1.1;
        color: #f8f1e5;
    }

    .rv-panel-sub {
        margin: 0 0 1rem;
        font-size: 0.83rem;
        font-weight: 300;
        line-height: 1.6;
        color: var(--rv-muted);
    }

    .rv-kicker { margin-bottom: 0.2rem; }

    .rv-hint {
        margin: 0.65rem 0 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #938875;
    }

    .rv-code {
        padding: 0.05rem 0.3rem;
        border: 1px solid #352d22;
        border-radius: 2px;
        background: #0d0b09;
        font-family: 'DM Mono', monospace;
        font-size: 0.74rem;
        color: var(--rv-gold-bright);
    }

    /* ------------------------------------------------------------------
       Inputs
    ------------------------------------------------------------------ */
    .stTextArea textarea {
        background: #0b0907 !important;
        border: 1px solid #2c251b !important;
        border-radius: 2px !important;
        color: var(--rv-cream) !important;
        caret-color: var(--rv-gold) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.96rem !important;
        font-weight: 300 !important;
        line-height: 1.75 !important;
        padding: 0.95rem 1rem !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .stTextArea textarea::placeholder {
        color: #6c6152 !important;
        font-style: italic;
    }

    .stTextArea textarea:focus {
        border-color: rgba(217, 179, 108, 0.75) !important;
        box-shadow: 0 0 0 1px rgba(217, 179, 108, 0.25) !important;
    }

    .stTextArea label p,
    .stSelectbox label p,
    .stSlider label p,
    .stFileUploader label p {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.58rem !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase;
        color: #9b8f7d !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 2px !important;
        transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease;
        box-shadow: none !important;
    }

    .stButton > button[kind="primary"] {
        width: 100%;
        background: var(--rv-gold) !important;
        border: 1px solid var(--rv-gold) !important;
        padding: 0.72rem 1.1rem !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.64rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase;
        color: #0a0908 !important;
    }

    .stButton > button[kind="primary"] p {
        color: #0a0908 !important;
        font-weight: 500 !important;
        letter-spacing: 0.2em !important;
        font-size: 0.64rem !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--rv-gold-bright) !important;
        border-color: var(--rv-gold-bright) !important;
        box-shadow: 0 16px 36px -20px rgba(217, 179, 108, 0.65) !important;
    }

    .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 1px #0a0908, 0 0 0 2px var(--rv-gold-bright) !important;
    }

    .stButton > button[kind="secondary"] {
        width: 100%;
        height: auto;
        min-height: 0;
        justify-content: flex-start;
        white-space: normal;
        text-align: left;
        padding: 0.62rem 0.78rem;
        border: 1px solid #2b241b !important;
        border-radius: 2px !important;
        background: #0f0d0a !important;
        color: #c5b9a7 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 300 !important;
        line-height: 1.5 !important;
    }

    .stButton > button[kind="secondary"] p {
        text-align: left;
        color: #c5b9a7 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 300 !important;
        line-height: 1.5 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: rgba(217, 179, 108, 0.5) !important;
        background: #15120d !important;
    }

    .stButton > button[kind="secondary"]:hover p { color: var(--rv-cream) !important; }

    .stButton > button[kind="secondary"]:focus {
        border-color: var(--rv-gold) !important;
        box-shadow: 0 0 0 1px rgba(217, 179, 108, 0.35) !important;
    }

    [data-testid="stDownloadButton"] button {
        width: 100%;
        border: 1px solid rgba(217, 179, 108, 0.4) !important;
        border-radius: 2px !important;
        background: transparent !important;
        padding: 0.65rem 1rem !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.6rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase;
        color: var(--rv-gold-bright) !important;
    }

    [data-testid="stDownloadButton"] button p {
        color: var(--rv-gold-bright) !important;
        font-size: 0.6rem !important;
        letter-spacing: 0.18em !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--rv-gold) !important;
        background: rgba(217, 179, 108, 0.07) !important;
    }

    /* ------------------------------------------------------------------
       File uploader
    ------------------------------------------------------------------ */
    [data-testid="stFileUploaderDropzone"] {
        background: #0d0b08 !important;
        border: 1px dashed #372f23 !important;
        border-radius: 2px !important;
        padding: 1.4rem !important;
        transition: border-color 0.16s ease, background 0.16s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(217, 179, 108, 0.5) !important;
        background: #12100c !important;
    }

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #a19583 !important;
        font-size: 0.78rem !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid rgba(217, 179, 108, 0.38) !important;
        border-radius: 2px !important;
        background: transparent !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.58rem !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase;
        color: var(--rv-gold-bright) !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: var(--rv-gold) !important;
        background: rgba(217, 179, 108, 0.08) !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #100d0a !important;
        border: 1px solid #2b241b !important;
        border-radius: 2px !important;
    }

    [data-testid="stFileUploaderFile"] * { color: #cbbfab !important; }

    [data-testid="stFileUploaderFile"] button svg { fill: var(--rv-gold) !important; }

    .rv-filemeta {
        margin: 0.8rem 0 0.55rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--rv-gold);
    }

    /* ------------------------------------------------------------------
       Verdict scorecard
    ------------------------------------------------------------------ */
    .rv-awaiting {
        padding: 2.5rem 1.4rem;
        border: 1px dashed #322a20;
        border-radius: 2px;
        background: radial-gradient(circle at 50% 0%, rgba(217, 179, 108, 0.055), transparent 72%);
        text-align: center;
    }

    .rv-awaiting-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        margin-bottom: 0.9rem;
        border: 1px solid #3d3426;
        border-radius: 50%;
        font-size: 1rem;
        color: var(--rv-gold);
    }

    .rv-awaiting-title {
        margin: 0 0 0.35rem;
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.45rem;
        color: #eadfce;
    }

    .rv-awaiting p {
        max-width: 21rem;
        margin: 0 auto;
        font-size: 0.8rem;
        font-weight: 300;
        line-height: 1.65;
        color: #8d8271;
    }

    .rv-verdict {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 1.15rem;
        align-items: center;
        padding: 0.35rem 0 1.1rem;
        border-bottom: 1px solid var(--rv-line);
    }

    .rv-score {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 4.1rem;
        line-height: 0.85;
        color: var(--rv-gold-bright);
    }

    .rv-score small {
        display: block;
        margin-top: 0.45rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.55rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    .rv-verdict-kicker {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin: 0 0 0.25rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #9d9180;
    }

    .rv-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }

    .rv-dot.pos { background: var(--rv-pos); box-shadow: 0 0 0 3px rgba(143, 181, 115, 0.14); }
    .rv-dot.neg { background: var(--rv-neg); box-shadow: 0 0 0 3px rgba(209, 90, 78, 0.14); }
    .rv-dot.neu { background: var(--rv-neu); box-shadow: 0 0 0 3px rgba(216, 172, 87, 0.14); }

    .rv-verdict-word {
        margin: 0;
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.85rem;
        line-height: 1.05;
        color: #f7f0e4;
    }

    .rv-verdict-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.9rem;
        margin-top: 0.15rem;
    }

    .rv-verdict-note {
        margin: 0.2rem 0 0;
        font-size: 0.78rem;
        font-weight: 300;
        line-height: 1.5;
        color: var(--rv-muted);
    }

    .rv-verdict-stamp {
        transform: rotate(-6deg);
        padding: 0.35rem 0.55rem;
        border: 1.5px solid currentColor;
        border-radius: 2px;
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        opacity: 0.9;
        flex: 0 0 auto;
    }

    .rv-verdict-stamp.pos { color: var(--rv-pos); }
    .rv-verdict-stamp.neg { color: var(--rv-neg); }
    .rv-verdict-stamp.neu { color: var(--rv-neu); }

    .rv-quote {
        margin: 0.95rem 0 1.05rem;
        padding-left: 0.85rem;
        border-left: 1px solid rgba(217, 179, 108, 0.35);
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.02rem;
        font-style: italic;
        line-height: 1.55;
        color: #cdc0ad;
    }

    .rv-consensus-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.7rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    .rv-consensus-head b { color: var(--rv-gold); font-weight: 400; }

    .rv-meter { margin-bottom: 0.62rem; }

    .rv-meter-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.28rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #a39785;
    }

    .rv-meter-head span:last-child { color: #e2d6c2; }

    .rv-meter-track {
        height: 3px;
        background: #262019;
        overflow: hidden;
    }

    .rv-meter-fill { height: 100%; }

    .rv-footnote {
        margin: 0.9rem 0 0;
        padding-top: 0.75rem;
        border-top: 1px solid var(--rv-line-soft);
        font-family: 'DM Mono', monospace;
        font-size: 0.56rem;
        line-height: 1.7;
        letter-spacing: 0.06em;
        color: #948875;
    }

    .rv-footnote b { color: #baa98f; font-weight: 500; }

    .rv-warning {
        margin: 0.2rem 0 0.6rem;
        padding: 0.75rem 0.9rem;
        border: 1px solid #4a3628;
        border-left: 2px solid var(--rv-neu);
        border-radius: 2px;
        background: rgba(216, 172, 87, 0.055);
        font-size: 0.82rem;
        line-height: 1.55;
        color: #dcc9a8;
    }

    /* ------------------------------------------------------------------
       Batch tallies + ratio bar
    ------------------------------------------------------------------ */
    .rv-tallies {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        margin: 0.4rem 0 1rem;
        border-top: 1px solid var(--rv-line);
        border-bottom: 1px solid var(--rv-line);
    }

    .rv-tally {
        padding: 0.8rem 0.9rem 0.75rem;
        border-left: 1px solid var(--rv-line-soft);
    }

    .rv-tally:first-child { padding-left: 0; border-left: none; }

    .rv-tally-k {
        display: block;
        margin-bottom: 0.35rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        letter-spacing: 0.17em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    .rv-tally-v {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 2.1rem;
        line-height: 1;
        color: #f3ecdf;
    }

    .rv-tally-v.pos { color: var(--rv-pos); }
    .rv-tally-v.neg { color: var(--rv-neg); }
    .rv-tally-v.neu { color: var(--rv-neu); }

    .rv-ratio {
        display: flex;
        height: 5px;
        margin-bottom: 1.15rem;
        overflow: hidden;
        background: #221d16;
    }

    .rv-ratio span { display: block; height: 100%; }

    /* Custom ledger — avoids Streamlit's generic grid and column clipping */
    .rv-ledger {
        border: 1px solid var(--rv-line);
        border-radius: 2px;
        overflow: hidden;
        background: #0d0b08;
    }

    .rv-ledger-head {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 92px 78px;
        gap: 0.6rem;
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid var(--rv-line);
        background: #100d0a;
        font-family: 'DM Mono', monospace;
        font-size: 0.48rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--rv-dim);
    }

    .rv-ledger-head span:last-child { text-align: right; }

    .rv-ledger-body { max-height: 322px; overflow-y: auto; }

    .rv-ledger-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 92px 78px;
        gap: 0.6rem;
        align-items: center;
        padding: 0.48rem 0.7rem;
        border-bottom: 1px solid var(--rv-line-soft);
    }

    .rv-ledger-row:last-child { border-bottom: none; }

    .rv-ledger-rev {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.74rem;
        font-weight: 300;
        color: #c9bdab;
    }

    .rv-ledger-verdict {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.5rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #d8cec0;
        white-space: nowrap;
    }

    .rv-ledger-conf {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.28rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.52rem;
        color: #d8cec0;
    }

    .rv-ledger-track {
        display: block;
        width: 100%;
        height: 2px;
        background: #262019;
    }

    .rv-ledger-track span { display: block; height: 100%; }

    /* ------------------------------------------------------------------
       Sidebar — projection room
    ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(520px 280px at 50% -5%, rgba(217, 179, 108, 0.07), transparent 70%),
            #0c0a08;
        border-right: 1px solid var(--rv-line-soft);
    }

    section[data-testid="stSidebar"] * { color: #d8cec0; }

    section[data-testid="stSidebar"] .rv-side-head h2 {
        margin: 0.2rem 0 0.45rem;
        font-size: 1.55rem;
        line-height: 1.05;
        color: #f6efe2;
    }

    section[data-testid="stSidebar"] .rv-side-head h2 em { color: var(--rv-gold); font-style: italic; }

    section[data-testid="stSidebar"] .rv-side-head p {
        margin: 0 0 1.1rem;
        font-size: 0.78rem;
        font-weight: 300;
        line-height: 1.6;
        color: #938875;
    }

    section[data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid var(--rv-line-soft);
        margin: 0.4rem 0 1.15rem;
    }

    section[data-testid="stSidebar"] .rv-kicker { color: var(--rv-gold); margin-bottom: 0.35rem; }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #12100c !important;
        border: 1px solid #322a20 !important;
        border-radius: 2px !important;
        min-height: 2.55rem;
    }

    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] input {
        color: #f4ede1 !important;
        font-size: 0.83rem !important;
        font-weight: 400 !important;
    }

    .stSelectbox svg { fill: var(--rv-gold) !important; color: var(--rv-gold) !important; }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: #12100c !important;
        border: 1px solid #3a3023 !important;
        border-radius: 2px !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div[role="option"] {
        background-color: #12100c !important;
        color: #f4ede1 !important;
        font-size: 0.82rem !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: #1d1811 !important;
        color: #ffffff !important;
    }

    .rv-engine {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0.45rem 0 1.35rem;
        padding: 0.5rem 0.6rem;
        border: 1px solid #2b241b;
        border-radius: 2px;
        background: #100d0a;
        font-family: 'DM Mono', monospace;
        font-size: 0.54rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #a29683;
    }

    .rv-engine i {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        flex: 0 0 auto;
    }

    .rv-engine.b-live i { background: var(--rv-pos); box-shadow: 0 0 0 3px rgba(143, 181, 115, 0.13); }
    .rv-engine.b-rule i { background: var(--rv-neu); box-shadow: 0 0 0 3px rgba(216, 172, 87, 0.13); }

    .rv-engine b {
        margin-left: auto;
        font-weight: 400;
        font-size: 0.51rem;
        letter-spacing: 0.1em;
        color: #d9c8ab;
    }

    .rv-threshold-note {
        margin: -0.25rem 0 1.1rem;
        font-size: 0.69rem;
        font-weight: 300;
        line-height: 1.55;
        color: #938875;
    }

    .rv-ticket {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        margin-bottom: 0.38rem;
        padding: 0.48rem 0.6rem;
        border: 1px dashed #332b20;
        border-radius: 2px;
        background: #0f0c09;
        font-family: 'DM Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #c7baa6;
    }

    .rv-ticket > span { display: inline-flex; align-items: center; gap: 0.45rem; }
    .rv-ticket em { font-style: normal; font-size: 0.52rem; letter-spacing: 0.1em; color: #9a8e7c; }

    .rv-about {
        margin: 0;
        font-size: 0.76rem;
        font-weight: 300;
        line-height: 1.7;
        color: #8d8271;
    }

    .rv-about b { color: #c8bcaa; font-weight: 500; }

    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapsedControl"] svg { fill: var(--rv-gold) !important; color: var(--rv-gold) !important; }

    /* Keep the sidebar expand control reachable on phones even though the toolbar is hidden. */
    [data-testid="stExpandSidebarButton"] { visibility: visible !important; }

    [data-testid="stExpandSidebarButton"] button {
        width: 30px;
        height: 30px;
        min-height: 30px;
        padding: 0 !important;
        background: #12100d !important;
        border: 1px solid var(--rv-line) !important;
        border-radius: 2px !important;
        color: var(--rv-gold) !important;
    }

    [data-testid="stExpandSidebarButton"] svg {
        fill: var(--rv-gold) !important;
        color: var(--rv-gold) !important;
    }

    [data-testid="stTooltipIcon"] svg { fill: #8a7f6e !important; }

    /* ------------------------------------------------------------------
       Alerts, dataframes
    ------------------------------------------------------------------ */
    div[data-testid="stAlert"] {
        border: 1px solid #433126 !important;
        border-radius: 2px !important;
        background: #17110c !important;
    }

    div[data-testid="stAlert"] p { color: #e5d5c0 !important; font-size: 0.82rem !important; }

    div[data-testid="stDataFrame"] { border: 1px solid var(--rv-line-soft); }

    /* ------------------------------------------------------------------
       Footer
    ------------------------------------------------------------------ */
    .rv-footer {
        margin-top: 3.2rem;
        padding-top: 1.6rem;
        border-top: 1px solid var(--rv-line);
        text-align: center;
    }

    .rv-footer-mark {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.34em;
        color: var(--rv-gold);
        margin-bottom: 0.8rem;
    }

    .rv-footer-meta {
        font-family: 'DM Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8f8372;
        margin-bottom: 0.7rem;
    }

    .rv-footer p {
        max-width: 44rem;
        margin: 0 auto;
        font-size: 0.72rem;
        font-weight: 300;
        line-height: 1.7;
        color: #948875;
    }

    .rv-footer .rv-fin {
        margin-top: 1rem;
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.05rem;
        font-style: italic;
        letter-spacing: 0.28em;
        color: #5f5547;
    }

    /* ------------------------------------------------------------------
       Responsive
    ------------------------------------------------------------------ */
    @media (max-width: 960px) {
        .rv-hero { grid-template-columns: 1fr; gap: 2.2rem; padding: 2.3rem 0 2.2rem; }
        .rv-poster { transform: none; max-width: 420px; }
        .rv-meta { gap: 1.5rem; }
        .rv-verdict { grid-template-columns: 1fr; }
        .rv-verdict-stamp { display: none; }
    }

    @media (max-height: 950px) and (min-width: 961px) {
        .rv-hero { padding: 1.15rem 0 1.1rem; }
        .rv-hero h1 { font-size: 3.45rem; margin: 0.5rem 0 0.85rem; }
        .rv-lede { font-size: 0.9rem; margin-bottom: 1rem; }
        .rv-meta { padding-top: 0.8rem; }
        .rv-poster { padding: 0.95rem 1.1rem 1rem; }
        .rv-poster h2 { font-size: 1.8rem; }
        .rv-poster-top { margin-bottom: 1rem; }
        .rv-poster-row { padding: 0.2rem 0; }
        .rv-poster-rule { margin: 0.75rem 0 0.5rem; }
        .rv-stamp { margin-top: 0.7rem; }
        .rv-section-head { margin-top: 1.2rem; }
    }

    @media (max-width: 640px) {
        .rv-hero h1 { font-size: 2.7rem; }
        .rv-topbar-meta .rv-season { display: none; }
        .rv-tallies { grid-template-columns: repeat(2, 1fr); }
        .rv-tally:nth-child(3) { padding-left: 0; border-left: none; border-top: 1px solid var(--rv-line-soft); }
        .rv-tally:nth-child(4) { border-top: 1px solid var(--rv-line-soft); }
        .rv-score { font-size: 3.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Logistic Regression": "sentiment_logistic_regression.joblib",
    "Naive Bayes": "sentiment_naive_bayes.joblib",
    "Support Vector Machine": "sentiment_svm.joblib",
    "Rule-Based Demo": None,
}

VERDICT_COPY = {
    "Positive": {
        "phrase": "A favourable notice",
        "note": "The consensus leans warmly favourable.",
        "stamp": "FRESH",
        "cls": "pos",
    },
    "Negative": {
        "phrase": "An unfavourable notice",
        "note": "The consensus is firmly against it.",
        "stamp": "ROTTEN",
        "cls": "neg",
    },
    "Neutral": {
        "phrase": "Mixed notices",
        "note": "The critics cannot agree on this one.",
        "stamp": "MIXED",
        "cls": "neu",
    },
}

CLASS_COLORS = {"Positive": "#8fb573", "Neutral": "#d8ac57", "Negative": "#d15a4e"}

DEFAULT_REVIEW = (
    "The movie was fantastic and I loved every minute of it. "
    "Outstanding direction and brilliant performances!"
)

EXAMPLES = [
    "A heartwarming story with excellent acting. Loved every minute.",
    "Boring from start to finish. The worst film I have seen all year.",
    "It was fine. Nothing memorable, but not terrible either.",
]

TICKER_TITLES = [
    "THE LAST PROJECTION",
    "GOLDEN HOUR",
    "MIDNIGHT CUT",
    "A QUIET FRAME",
    "THE CRITIC",
    "SILVER HALIDE",
    "REEL ONE",
    "THE FINAL TAKE",
]


@st.cache_resource(show_spinner=False)
def load_predictor(model_label):
    """Load the selected model, or the rule-based critic if weights are absent."""
    filename = MODEL_OPTIONS.get(model_label)
    if filename:
        path = os.path.join("models", filename)
        if os.path.exists(path):
            return get_predictor(path), True
    return get_predictor(None), False


def resolve_label(raw_label, confidence, threshold):
    """Apply the confidence floor: low-confidence calls read as Neutral."""
    if raw_label != "Neutral" and confidence < threshold:
        return "Neutral", True
    return raw_label, False


def sentiment_index(probabilities):
    """Weighted 0–100 index: Negative 0, Neutral 50, Positive 100."""
    pos = float(probabilities.get("Positive", 0.0))
    neu = float(probabilities.get("Neutral", 0.0))
    return max(0, min(100, int(round(pos * 100 + neu * 50))))


def stars_for(index):
    filled = max(0, min(5, int(round(index / 20))))
    return "★" * filled + "☆" * (5 - filled)


def excerpt(text, limit=240):
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def use_example(text):
    """Callback: load a pull quote into the review desk and clear old verdict."""
    st.session_state["review_text"] = text
    st.session_state.pop("verdict", None)
    st.session_state.pop("verdict_error", None)


st.session_state.setdefault("review_text", DEFAULT_REVIEW)


# ---------------------------------------------------------------------------
# Top bar + ticker + hero
# ---------------------------------------------------------------------------
ticker_items = "".join(
    f'<span>{title}</span><em>◆</em><b>NOW SCREENING</b><em>◆</em>' for title in TICKER_TITLES
)

st.markdown(
    f"""
    <header class="rv-topbar">
        <div class="rv-brand">REEL<span>//</span>VERDICT</div>
        <div class="rv-topbar-meta">
            <span class="rv-live"><i></i>Now screening</span>
            <span class="rv-season">Sem V · Group 8</span>
        </div>
    </header>
    <div class="rv-ticker" aria-hidden="true">
        <div class="rv-ticker-track">{ticker_items}{ticker_items}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="rv-hero">
        <div class="rv-hero-copy">
            <p class="rv-eyebrow">Project 01 — Sentiment Analysis</p>
            <h1>Every review<br>has a <em>verdict.</em></h1>
            <p class="rv-lede">
                A TF-IDF screening room for movie reviews. File a review with the guest
                critic and receive a verdict, a 0–100 sentiment index, and the full
                critic consensus — no red carpet required.
            </p>
            <div class="rv-meta">
                <div><span>The ensemble</span><b>LR · NB · SVM</b></div>
                <div><span>The method</span><b>TF-IDF · N-grams</b></div>
                <div><span>The season</span><b>Group 8 · Sem V</b></div>
            </div>
        </div>
        <aside class="rv-poster">
            <div class="rv-perf"></div>
            <div class="rv-poster-top"><span>Feature presentation</span><span>No. 001</span></div>
            <h2>The <em>Verdict</em></h2>
            <div class="rv-stars">★★★★★</div>
            <div class="rv-poster-rule"></div>
            <div class="rv-poster-row"><span>Directed by</span><span>Group 8</span></div>
            <div class="rv-poster-row"><span>Cinematography</span><span>TF-IDF</span></div>
            <div class="rv-poster-row"><span>Running time</span><span>3 classes</span></div>
            <div class="rv-poster-row"><span>Format</span><span>Streamlit · 70mm</span></div>
            <div class="rv-poster-rule"></div>
            <div class="rv-poster-foot"><span>Positive</span><span>Neutral</span><span>Negative</span></div>
            <div class="rv-stamp">Press<br>screening</div>
            <div class="rv-perf"></div>
        </aside>
    </section>
    <div class="rv-section-head">
        <span class="rv-num">01</span>
        <h2>The screening room</h2>
        <span class="rv-rule"></span>
        <span class="rv-note">File a review</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — projection room
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="rv-side-head">
            <p class="rv-kicker">Projection room</p>
            <h2>Tonight's <em>controls</em></h2>
            <p>Choose the critic, set the confidence floor, and read the ticket key.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="rv-kicker">Select critic</p>', unsafe_allow_html=True)
    model_option = st.selectbox(
        "Select critic",
        list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed",
        help="ML models need trained weights in models/. Without them, the rule-based critic steps in.",
    )

    predictor, weights_loaded = load_predictor(model_option)
    engine_class = "b-live" if weights_loaded else "b-rule"
    engine_status = "Weights loaded" if weights_loaded else "No weights · rule-based"
    st.markdown(
        f'<div class="rv-engine {engine_class}"><i></i>{html.escape(model_option)}'
        f"<b>{engine_status}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="rv-kicker">Confidence floor</p>', unsafe_allow_html=True)
    conf_thresh = st.slider(
        "Confidence floor",
        min_value=0.0,
        max_value=0.95,
        value=0.45,
        step=0.05,
        label_visibility="collapsed",
        help="Calls below this confidence are read as Mixed. Set to 0 to trust every call.",
    )
    st.markdown(
        '<p class="rv-threshold-note">Verdicts under this confidence are downgraded to '
        f'<b style="color:#d8ac57;">Mixed</b>. Currently {conf_thresh:.0%}.</p>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<p class="rv-kicker">Ticket key</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rv-ticket"><span><i class="rv-dot pos"></i>Positive</span><em>favourable</em></div>
        <div class="rv-ticket"><span><i class="rv-dot neu"></i>Neutral</span><em>divided</em></div>
        <div class="rv-ticket"><span><i class="rv-dot neg"></i>Negative</span><em>unfavourable</em></div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<p class="rv-kicker">The project</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rv-about"><b>Project 1 — Sentiment Analysis.</b> TF-IDF features '
        "with classical classifiers, served through Streamlit. Trained on a small "
        "IMDb sample; retrain on a larger corpus for tougher critics.</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Screening modes
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["01 · Single review", "02 · Batch screening"])


# ------------------------------- Single review ------------------------------
with tab_single:
    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <p class="rv-kicker">Guest critic</p>
                <h3 class="rv-panel-title">File your review</h3>
                <p class="rv-panel-sub">Paste a movie or product review. The projectionist reads every word.</p>
                """,
                unsafe_allow_html=True,
            )

            text = st.text_area(
                "Review text",
                key="review_text",
                height=250,
                label_visibility="collapsed",
                placeholder="e.g. A heartwarming story with excellent acting. Loved every minute.",
            )

            screen_btn = st.button(
                "Screen this review",
                type="primary",
                width="stretch",
                key="screen_single",
            )

            st.markdown(
                '<p class="rv-hint" style="margin-bottom:0.55rem;">Or pull a quote from the archive</p>',
                unsafe_allow_html=True,
            )
            for idx, quote in enumerate(EXAMPLES):
                st.button(
                    f"“{quote}”",
                    key=f"example_{idx}",
                    on_click=use_example,
                    args=(quote,),
                    width="stretch",
                )

            st.markdown(
                '<p class="rv-hint">Screening runs locally · no review leaves this machine</p>',
                unsafe_allow_html=True,
            )

        if screen_btn:
            if not text.strip():
                st.session_state["verdict_error"] = "Please file a review before requesting a verdict."
                st.session_state.pop("verdict", None)
            else:
                with st.spinner("The projectionist is reading…"):
                    result = predictor.predict(text)
                st.session_state["verdict"] = {
                    "text": text,
                    "result": result,
                    "model": model_option,
                    "threshold": conf_thresh,
                }
                st.session_state.pop("verdict_error", None)

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <p class="rv-kicker">The verdict</p>
                <h3 class="rv-panel-title">Critic's scorecard</h3>
                <p class="rv-panel-sub">Score, stamp and the full consensus from the model.</p>
                """,
                unsafe_allow_html=True,
            )

            verdict = st.session_state.get("verdict")
            verdict_error = st.session_state.get("verdict_error")

            if verdict_error:
                st.markdown(
                    f'<div class="rv-warning">{html.escape(verdict_error)}</div>',
                    unsafe_allow_html=True,
                )

            if verdict:
                result = verdict["result"]
                raw_label = result["label"]
                raw_conf = float(result["confidence"])
                probabilities = result["probabilities"] or {}
                threshold = verdict["threshold"]
                label, floored = resolve_label(raw_label, raw_conf, threshold)
                meta = VERDICT_COPY[label]
                index = sentiment_index(probabilities)
                stars = stars_for(index)
                shown_conf = raw_conf if not floored else float(probabilities.get(label, 0.0))

                verdict_kicker = f"{label} · confidence {shown_conf:.0%}"
                if floored:
                    verdict_kicker = f"{label} · below the {threshold:.0%} floor"

                bars = ""
                consensus_bits = []
                for cls in ["Positive", "Neutral", "Negative"]:
                    prob = float(probabilities.get(cls, 0.0))
                    consensus_bits.append(f"{cls[:3].upper()} {prob:.0%}")
                    bars += (
                        '<div class="rv-meter">'
                        '<div class="rv-meter-head">'
                        f"<span>{cls}</span><span>{prob:.0%}</span>"
                        "</div>"
                        '<div class="rv-meter-track">'
                        f'<div class="rv-meter-fill" style="width:{prob * 100:.1f}%;'
                        f'background:{CLASS_COLORS[cls]};"></div>'
                        "</div>"
                        "</div>"
                    )

                footnote = (
                    f"Screened with <b>{html.escape(verdict['model'])}</b> · "
                    f"confidence floor <b>{threshold:.0%}</b> · "
                    "index weighted Positive 100 / Neutral 50 / Negative 0."
                )
                if floored:
                    footnote = (
                        f"The model leaned <b>{html.escape(raw_label)}</b> at "
                        f"<b>{raw_conf:.0%}</b>, below your {threshold:.0%} confidence floor, "
                        f"so the verdict reads <b>Mixed</b>.<br>" + footnote
                    )

                st.markdown(
                    f"""
                    <div class="rv-verdict">
                        <div class="rv-score">{index}<small>/ 100 · index</small></div>
                        <div>
                            <p class="rv-verdict-kicker">
                                <span class="rv-dot {meta['cls']}"></span>{verdict_kicker}
                            </p>
                            <h4 class="rv-verdict-word">{meta['phrase']}</h4>
                            <div class="rv-verdict-meta">
                                <div class="rv-stars small">{stars}</div>
                                <span class="rv-verdict-stamp {meta['cls']}">{meta['stamp']}</span>
                            </div>
                            <p class="rv-verdict-note">{meta['note']}</p>
                        </div>
                    </div>
                    <div class="rv-quote">“{html.escape(excerpt(verdict['text']))}”</div>
                    <div class="rv-consensus-head">
                        <span>Critic consensus</span>
                        <b>{' · '.join(consensus_bits)}</b>
                    </div>
                    {bars}
                    <p class="rv-footnote">{footnote}</p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="rv-awaiting">
                        <div class="rv-awaiting-mark">✦</div>
                        <p class="rv-awaiting-title">The screen is dark.</p>
                        <p>File a review on the left and the verdict will be projected here —
                        score, stamp and full critic consensus.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ------------------------------- Batch screening -----------------------------
with tab_batch:
    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <p class="rv-kicker">Box office</p>
                <h3 class="rv-panel-title">Submit the docket</h3>
                <p class="rv-panel-sub">Upload a CSV with a <span class="rv-code">text</span> column.
                Every row gets its own screening.</p>
                """,
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Upload a review CSV",
                type=["csv"],
                label_visibility="collapsed",
                help="A .csv file with a 'text' column.",
            )

            df = None
            batch_ready = False
            file_key = None

            if uploaded is not None:
                file_key = getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
                try:
                    df = pd.read_csv(uploaded)
                except Exception as exc:  # noqa: BLE001 - surface any parse failure
                    st.markdown(
                        f'<div class="rv-warning">Could not read that CSV — {html.escape(str(exc))}</div>',
                        unsafe_allow_html=True,
                    )
                    df = None

                if df is not None:
                    if "text" not in df.columns:
                        st.markdown(
                            '<div class="rv-warning">This docket needs a <b>text</b> column. '
                            "Found: "
                            + html.escape(", ".join(str(c) for c in df.columns))
                            + ".</div>",
                            unsafe_allow_html=True,
                        )
                    elif df.empty:
                        st.markdown(
                            '<div class="rv-warning">The docket is empty — add at least one review row.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        batch_ready = True
                        st.markdown(
                            f'<p class="rv-filemeta">{len(df):,} rows · {len(df.columns)} columns · '
                            f"{html.escape(uploaded.name)}</p>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            df.head(4),
                            hide_index=True,
                            height=190,
                            width="stretch",
                        )
                        analyze_all = st.button(
                            "Screen all reviews",
                            type="primary",
                            width="stretch",
                            key="screen_batch",
                        )
                        if analyze_all:
                            texts = df["text"].fillna("").astype(str).tolist()
                            with st.spinner("Screening the docket…"):
                                results = [predictor.predict(t) for t in texts]
                            out = df.copy()
                            out["Verdict"] = [
                                resolve_label(r["label"], float(r["confidence"]), conf_thresh)[0]
                                for r in results
                            ]
                            out["Confidence"] = [
                                int(round(float(r["confidence"]) * 100)) for r in results
                            ]
                            st.session_state["batch"] = {
                                "key": file_key,
                                "df": out,
                                "model": model_option,
                                "threshold": conf_thresh,
                            }

            if uploaded is None:
                st.markdown(
                    '<p class="rv-hint" style="margin-top:0.9rem;">No file at the door? '
                    'A sample waits at <span class="rv-code">data/sample_imdb.csv</span>.</p>',
                    unsafe_allow_html=True,
                )

        batch_state = st.session_state.get("batch")
        if not batch_ready and uploaded is not None:
            batch_state = None

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <p class="rv-kicker">The dailies</p>
                <h3 class="rv-panel-title">Batch results</h3>
                <p class="rv-panel-sub">Every review from the docket, screened and tallied.</p>
                """,
                unsafe_allow_html=True,
            )

            if batch_ready and batch_state and batch_state["key"] == file_key:
                out = batch_state["df"]
                total = len(out)
                pos = int((out["Verdict"] == "Positive").sum())
                neg = int((out["Verdict"] == "Negative").sum())
                neu = int((out["Verdict"] == "Neutral").sum())

                def pct(value):
                    return (value / total * 100) if total else 0.0

                st.markdown(
                    f"""
                    <div class="rv-tallies">
                        <div class="rv-tally"><span class="rv-tally-k">Features</span><span class="rv-tally-v">{total:,}</span></div>
                        <div class="rv-tally"><span class="rv-tally-k">Positive</span><span class="rv-tally-v pos">{pos:,}</span></div>
                        <div class="rv-tally"><span class="rv-tally-k">Negative</span><span class="rv-tally-v neg">{neg:,}</span></div>
                        <div class="rv-tally"><span class="rv-tally-k">Mixed</span><span class="rv-tally-v neu">{neu:,}</span></div>
                    </div>
                    <div class="rv-ratio" aria-hidden="true">
                        <span style="width:{pct(pos):.2f}%; background:#8fb573;"></span>
                        <span style="width:{pct(neu):.2f}%; background:#d8ac57;"></span>
                        <span style="width:{pct(neg):.2f}%; background:#d15a4e;"></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<p class="rv-kicker" style="margin-bottom:0.5rem;">The ledger</p>', unsafe_allow_html=True)

                ledger_rows = []
                for _, row in out.head(300).iterrows():
                    verdict = str(row["Verdict"])
                    conf = int(row["Confidence"])
                    cls = {"Positive": "pos", "Negative": "neg", "Neutral": "neu"}.get(verdict, "neu")
                    color = CLASS_COLORS.get(verdict, "#d8ac57")
                    full_text = str(row.get("text", ""))
                    preview_text = html.escape(excerpt(full_text, 150))
                    title_text = html.escape(excerpt(full_text, 300), quote=True)
                    ledger_rows.append(
                        '<div class="rv-ledger-row">'
                        f'<span class="rv-ledger-rev" title="{title_text}">{preview_text}</span>'
                        f'<span class="rv-ledger-verdict"><i class="rv-dot {cls}"></i>'
                        f"{html.escape(verdict)}</span>"
                        f'<span class="rv-ledger-conf"><span>{conf}%</span>'
                        f'<span class="rv-ledger-track"><span style="width:{conf}%;'
                        f'background:{color};"></span></span></span>'
                        "</div>"
                    )

                st.markdown(
                    '<div class="rv-ledger">'
                    '<div class="rv-ledger-head"><span>Review</span><span>Verdict</span>'
                    '<span>Conf.</span></div>'
                    '<div class="rv-ledger-body">'
                    + "".join(ledger_rows)
                    + "</div></div>",
                    unsafe_allow_html=True,
                )

                shown_note = (
                    f" Showing the first 300 of {total:,} rows; the export includes all."
                    if total > 300
                    else ""
                )

                st.download_button(
                    "Export ledger (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="reel_verdict_results.csv",
                    mime="text/csv",
                    width="stretch",
                    key="download_batch",
                )

                st.markdown(
                    f'<p class="rv-footnote">Screened with <b>{html.escape(batch_state["model"])}</b> · '
                    f'confidence floor <b>{batch_state["threshold"]:.0%}</b>. '
                    "Change the critic or floor in the projection room, then screen again to refresh."
                    f"{shown_note}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="rv-awaiting">
                        <div class="rv-awaiting-mark">▤</div>
                        <p class="rv-awaiting-title">Nothing in the dailies yet.</p>
                        <p>Upload a CSV on the left and screen the docket — counts, consensus
                        and a downloadable ledger will appear here.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="rv-footer">
        <div class="rv-footer-mark">REEL VERDICT</div>
        <div class="rv-footer-meta">Ayush Singh · Krish Jaidka · Sahil Shrivastava</div>
        <p>Project 1 — Sentiment Analysis · Group 8 · Semester V. Built with TF-IDF,
        scikit-learn and Streamlit. Predictions are assistive reading notes, not final reviews.</p>
        <div class="rv-fin">FIN.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
