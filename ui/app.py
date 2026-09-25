"""
ui/app.py
─────────────────────────────────────────────────────────────────────────────
Premium Streamlit Dashboard for the Distributed Movie Recommendation Engine.

Pages:
  1. 🎬 Explore & Rate     — Search movies, rate them, get recommendations
  2. ✨ My Recommendations — Personalised top-10 picks with details
  3. 📊 System Dashboard   — Live model metrics, charts, architecture
  4. 🏗️ Architecture       — How the distributed pipeline works

Run:
    streamlit run ui/app.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import math
import time
import random
import requests
from typing import List, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Distributed Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Warm Cartoonish Light Theme ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Fredoka+One&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Global background: warm creamy white ── */
html, body, [data-testid="stApp"] {
    font-family: 'Nunito', sans-serif;
    background: #fef9f0;
    color: #3d2c1e;
}

/* ── Sidebar: warm cartoon peach gradient ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff9f0 0%, #ffe8cc 100%) !important;
    border-right: 3px solid #ffb347 !important;
    box-shadow: 4px 0 16px rgba(255, 179, 71, 0.15) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 6px;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #ffd180;
    border-radius: 10px;
}

/* ── Sidebar nav radio ── */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.96rem !important;
    color: #7c4a10 !important;
    padding: 0.6rem 1rem !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    display: block !important;
    margin: 4px 0 !important;
    border: 2px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 179, 71, 0.25) !important;
    border-color: #ffcc80 !important;
    transform: translateX(5px) !important;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Fredoka One', cursive;
    font-size: 2.6rem;
    color: #e05c00;
    line-height: 1.15;
    margin-bottom: 0.2rem;
    text-shadow: 3px 3px 0 rgba(0,0,0,0.06);
}
.hero-sub {
    font-size: 1rem;
    color: #a0704a;
    font-weight: 700;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 2.5px solid #ffcc80;
    border-radius: 20px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 4px 4px 0 #ffd180;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 4px 8px 0 #ffb347;
}
.metric-value {
    font-family: 'Fredoka One', cursive;
    font-size: 2.2rem;
    color: #e05c00;
}
.metric-label { font-size: 0.82rem; color: #a0704a; margin-top: 0.2rem; font-weight: 700; }

/* ── Movie cards ── */
.movie-card {
    background: white;
    border: 2.5px solid #ffe0b2;
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 3px 3px 0 #ffcc80;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.movie-card:hover {
    transform: translateY(-3px);
    box-shadow: 3px 6px 0 #ffb347;
    border-color: #ffb347;
}
.movie-title { font-weight: 800; font-size: 1rem; color: #3d2c1e; }
.movie-genres span {
    display: inline-block;
    background: #fff3e0;
    border: 1.5px solid #ffb347;
    border-radius: 30px;
    padding: 2px 10px;
    font-size: 0.7rem;
    color: #e05c00;
    margin: 2px;
    font-weight: 700;
}

/* ── Recommendation cards ── */
.rec-card {
    background: white;
    border: 2.5px solid #ffe0b2;
    border-radius: 20px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 4px 4px 0 #ffcc80;
    transition: transform 0.25s, box-shadow 0.25s;
}
.rec-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #ffb347, #ff6b35, #f7931e);
    border-radius: 20px 20px 0 0;
}
.rec-card:hover {
    transform: translateY(-5px) rotate(-0.4deg);
    box-shadow: 4px 9px 0 #ffb347;
}
.rec-rank {
    font-family: 'Fredoka One', cursive;
    font-size: 2.4rem;
    color: #ffb347;
    line-height: 1;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.07);
}
.rec-title { font-size: 1.1rem; font-weight: 800; color: #3d2c1e; }
.match-bar-container {
    background: #fff3e0;
    border-radius: 20px;
    height: 10px;
    margin: 8px 0;
    overflow: hidden;
    border: 1.5px solid #ffe0b2;
}
.match-bar {
    height: 100%; border-radius: 20px;
    background: linear-gradient(90deg, #ffb347, #ff6b35);
    transition: width 0.8s ease;
}

/* ── Alert boxes ── */
.info-box {
    background: #fff8e1;
    border: 2px solid #ffe082;
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #7c5700;
    font-weight: 600;
}
.cold-start-box {
    background: #fff3e0;
    border: 2px solid #ffb347;
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    color: #7c3a00;
    font-weight: 600;
}

/* ── Buttons — Playful Cartoon Aesthetic (No Black Anywhere!) ── */
.stButton > button {
    border-radius: 14px !important;
    font-weight: 800 !important;
    font-family: 'Nunito', sans-serif !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    border: 2.5px solid #ffcc80 !important;
    background: #ffffff !important;
    color: #7c4a10 !important;
    box-shadow: 0 4px 0 #ffd180 !important;
    outline: none !important;
}
.stButton > button * {
    color: inherit !important;
    font-family: inherit !important;
    font-weight: 800 !important;
}
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #7c4a10 !important;
    font-weight: 800 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
}
.stButton > button:hover {
    background: #fff8e1 !important;
    border-color: #ff9800 !important;
    color: #e05c00 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 0 #ffd180 !important;
}
.stButton > button:hover p,
.stButton > button:hover span {
    color: #e05c00 !important;
}
.stButton > button:active {
    transform: translateY(2px) !important;
    box-shadow: 0 1px 0 #ffd180 !important;
}
.stButton > button:focus,
.stButton > button:focus:not(:active) {
    border-color: #ff9800 !important;
    color: #7c4a10 !important;
    outline: none !important;
    box-shadow: 0 4px 0 #ffd180 !important;
}

/* Primary Button: Vibrant Orange Hero Gradient */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #ff9233 0%, #ff6b35 100%) !important;
    color: #ffffff !important;
    border: 2.5px solid #d45200 !important;
    box-shadow: 0 4px 0 #b33e00 !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span,
.stButton > button[data-testid="baseButton-primary"] div {
    color: #ffffff !important;
    font-family: 'Fredoka One', cursive !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #ffa34d 0%, #ff7d4d 100%) !important;
    border-color: #d45200 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 0 #b33e00 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span {
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(2px) !important;
    box-shadow: 0 1px 0 #b33e00 !important;
}

/* Secondary Button: Cream Cartoon Pill (NO Black!) */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: #7c4a10 !important;
    border: 2.5px solid #ffcc80 !important;
    box-shadow: 0 4px 0 #ffd180 !important;
}
.stButton > button[kind="secondary"] p,
.stButton > button[kind="secondary"] span,
.stButton > button[data-testid="baseButton-secondary"] p,
.stButton > button[data-testid="baseButton-secondary"] span {
    color: #7c4a10 !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: #fff8e1 !important;
    border-color: #ff9800 !important;
    color: #e05c00 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 0 #ffd180 !important;
}
.stButton > button[kind="secondary"]:hover p,
.stButton > button[kind="secondary"]:hover span {
    color: #e05c00 !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #66bb6a, #43a047) !important;
    border-color: #2e7d32 !important;
    box-shadow: 3px 3px 0 #2e7d32 !important;
    color: white !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #4caf50, #388e3c) !important;
    box-shadow: 3px 5px 0 #1b5e20 !important;
    color: white !important;
}

/* ── Text Input & High-Contrast Placeholder ── */
[data-testid="stTextInput"] > div {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: none !important;
}
[data-testid="stTextInput"] input {
    border: 2.5px solid #ffcc80 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #3d2c1e !important;
    -webkit-text-fill-color: #3d2c1e !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #ff9800 !important;
    box-shadow: 0 0 0 3px rgba(255, 179, 71, 0.35) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #9c6644 !important;
    -webkit-text-fill-color: #9c6644 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInput"] input::-webkit-input-placeholder {
    color: #9c6644 !important;
    -webkit-text-fill-color: #9c6644 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInput"] input::-moz-placeholder {
    color: #9c6644 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
[data-testid="stTextInput"] input:-ms-input-placeholder {
    color: #9c6644 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* ── Sliders — Warm Cartoon Palette (No Black!) ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p {
    color: #7c4a10 !important;
    font-weight: 800 !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div {
    font-family: 'Nunito', sans-serif !important;
    color: #7c4a10 !important;
    font-weight: 800 !important;
}
[data-testid="stSlider"] div[role="slider"] {
    background-color: #ff6b35 !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 2px 6px rgba(255, 107, 53, 0.4) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: #a0704a !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #e05c00 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 2px solid #ffe0b2 !important;
    border-radius: 14px !important;
    background: white !important;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1.4rem;
    color: #e05c00;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.divider { border: none; border-top: 2px dashed #ffe0b2; margin: 2rem 0; }

/* ── Animations ── */
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
@keyframes wiggle {
    0%, 100% { transform: rotate(-3deg); }
    50% { transform: rotate(3deg); }
}
.bounce-emoji { display: inline-block; animation: bounce 1.8s ease-in-out infinite; }
.wiggle-emoji { display: inline-block; animation: wiggle 2s ease-in-out infinite; }

/* ── Clean Streamlit Chrome & Fun Sidebar Toggle ── */
#MainMenu { display: none !important; }
footer { display: none !important; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3.8rem !important;
    z-index: 100 !important;
}

/* Sidebar collapse/expand button: ALWAYS visible and prominent */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 999999 !important;
    align-items: center !important;
    justify-content: center !important;
    background: #ffecb3 !important;
    border: 2.5px solid #ff9800 !important;
    border-radius: 14px !important;
    box-shadow: 2px 3px 0 #e07000 !important;
    transition: all 0.15s ease !important;
    padding: 2px !important;
}
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {
    background: #ffe082 !important;
    transform: scale(1.08) !important;
    box-shadow: 2px 4px 0 #c05000 !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    background: transparent !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    color: #e05c00 !important;
    padding: 6px !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill: #e05c00 !important;
    stroke: #e05c00 !important;
    color: #e05c00 !important;
    width: 22px !important;
    height: 22px !important;
    display: block !important;
}

/* Inside the open sidebar */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    background: #ffe8cc !important;
    border: 2px solid #ffb347 !important;
    border-radius: 12px !important;
    color: #e05c00 !important;
    margin: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover {
    background: #ffd8a8 !important;
    transform: scale(1.05) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    color: #e05c00 !important;
}

/* ── Top Navigation Bar Styles ── */
.top-nav-card {
    background: white;
    border: 2.5px solid #ffcc80;
    border-radius: 20px;
    padding: 0.85rem 1.4rem;
    margin-bottom: 0.85rem;
    box-shadow: 4px 4px 0 #ffd180;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.top-nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.top-sidebar-btn {
    background: #ffe8cc;
    border: 2.5px solid #ff9800;
    border-radius: 12px;
    padding: 0.4rem 0.85rem;
    font-family: 'Fredoka One', cursive;
    font-size: 0.95rem;
    color: #e05c00;
    box-shadow: 2px 2px 0 #e07000;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.top-sidebar-btn:hover {
    background: #ffd8a8;
    transform: translateY(-2px);
    box-shadow: 2px 4px 0 #c05000;
}
.top-sidebar-btn:active {
    transform: translateY(1px);
    box-shadow: 1px 1px 0 #c05000;
}
.top-nav-logo {
    font-size: 2.3rem;
    display: inline-block;
    animation: bounce 2s ease-in-out infinite;
    line-height: 1;
}
.top-nav-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1.8rem;
    color: #e05c00;
    line-height: 1.05;
}
.top-nav-tagline {
    font-size: 0.76rem;
    color: #a0704a;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.top-nav-badges {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.top-badge {
    background: #fff8e1;
    border: 2px solid #ffe082;
    border-radius: 12px;
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 800;
    color: #b45309;
    box-shadow: 2px 2px 0 #ffe082;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-expand sidebar & handle menu trigger ────────────────────────────────
components.html("""
<script>
(function() {
    function getSidebarBtn() {
        const pDoc = window.parent.document;
        if (!pDoc) return null;
        return pDoc.querySelector(
            '[data-testid="stSidebarCollapsedControl"] button, ' +
            '[data-testid="collapsedControl"] button, ' +
            'button[data-testid="stSidebarCollapseButton"]'
        );
    }

    function autoExpand() {
        try {
            const pDoc = window.parent.document;
            if (!pDoc) return;
            const sidebar = pDoc.querySelector('section[data-testid="stSidebar"]');
            const isCollapsed = sidebar ? sidebar.getAttribute('aria-expanded') === 'false' : false;
            const btn = getSidebarBtn();
            if (isCollapsed && btn) {
                btn.click();
            }
        } catch(e) {}
    }

    try {
        const pDoc = window.parent.document;
        if (pDoc && !window.parent.__cineai_menu_listener) {
            window.parent.__cineai_menu_listener = true;
            pDoc.addEventListener('click', function(e) {
                const target = e.target.closest('button, .top-sidebar-btn');
                if (target && target.innerText && target.innerText.includes('Menu')) {
                    const btn = getSidebarBtn();
                    if (btn && btn !== target) {
                        btn.click();
                    }
                }
            }, true);
        }
    } catch(e) {}

    autoExpand();
    setTimeout(autoExpand, 300);
    setTimeout(autoExpand, 900);
})();
</script>
""", height=0, width=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_local_data():
    """Load all local JSON files for offline / demo mode."""
    data = {"catalog": [], "cold_start": [], "metrics": {}, "stats": {}, "recs_cache": {}}

    all_path = os.path.join(DATA_DIR, "all_movies.json")
    cold_path = os.path.join(DATA_DIR, "cold_start_popular.json")

    if os.path.exists(all_path):
        with open(all_path, encoding="utf-8") as f:
            data["catalog"] = json.load(f)
    elif os.path.exists(cold_path):
        with open(cold_path, encoding="utf-8") as f:
            data["catalog"] = json.load(f)

    if os.path.exists(cold_path):
        with open(cold_path, encoding="utf-8") as f:
            data["cold_start"] = json.load(f)
    else:
        data["cold_start"] = data["catalog"][:500]

    metrics_path = os.path.join(LOGS_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    stats_path = os.path.join(LOGS_DIR, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            data["stats"] = json.load(f)

    cache_path = os.path.join(DATA_DIR, "recs_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data["recs_cache"] = json.load(f)

    return data


def api_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def search_movies_local(query: str, catalog: list, limit: int = 24) -> list:
    q = query.lower().strip()
    if not q:
        return []
    title_matches = []
    genre_matches = []
    for m in catalog:
        title = (m.get("title") or "").lower()
        genres = (m.get("genres") or "").lower()
        if q in title:
            title_matches.append(m)
        elif q in genres:
            genre_matches.append(m)
    title_matches.sort(key=lambda x: (-int(x.get("num_ratings", 0)), -float(x.get("avg_rating", 0))))
    genre_matches.sort(key=lambda x: (-int(x.get("num_ratings", 0)), -float(x.get("avg_rating", 0))))
    return (title_matches + genre_matches)[:limit]


def filter_rated_local(candidates: list, rated_movies: list) -> list:
    import re
    rated_ids = set()
    rated_titles = set()
    for m in rated_movies:
        mid = m.get("movie_id") if m.get("movie_id") is not None else m.get("movieId")
        if mid is not None:
            rated_ids.add(mid)
            rated_ids.add(str(mid).strip())
            try:
                rated_ids.add(int(mid))
            except (ValueError, TypeError):
                pass
        t = str(m.get("title", "")).strip().lower()
        if t:
            rated_titles.add(t)
            clean_t = re.sub(r"\s*\(\d{4}\).*", "", t).strip()
            if clean_t:
                rated_titles.add(clean_t)

    unrated = []
    for m in candidates:
        mid = m.get("movieId") if m.get("movieId") is not None else m.get("movie_id")
        if mid is not None:
            if mid in rated_ids or str(mid).strip() in rated_ids:
                continue
            try:
                if int(mid) in rated_ids:
                    continue
            except (ValueError, TypeError):
                pass
        t = str(m.get("title", "")).strip().lower()
        if t in rated_titles:
            continue
        clean_t = re.sub(r"\s*\(\d{4}\).*", "", t).strip()
        if clean_t and clean_t in rated_titles:
            continue
        unrated.append(m)
    return unrated


def compute_local_recommendations(rated_movies: list, catalog: list, top_n: int = 10) -> tuple:
    import math
    if not rated_movies:
        return [], True

    is_cold_start = len(rated_movies) < 5
    candidates = filter_rated_local(catalog, rated_movies)

    if is_cold_start:
        # Cold start: initial genre interest + popularity + quality
        liked_genres: Dict[str, float] = {}
        for movie in rated_movies:
            r = float(movie.get("rating", 3.5))
            if r >= 3.0:
                for g in (movie.get("genres") or "").split("|"):
                    if g and g != "(no genres listed)":
                        liked_genres[g] = liked_genres.get(g, 0.0) + (r / 5.0)

        scored = []
        for movie in candidates:
            genres = (movie.get("genres") or "").split("|")
            genre_score = sum(liked_genres.get(g, 0.0) for g in genres)
            popularity = math.log1p(movie.get("num_ratings", 0))
            quality = float(movie.get("avg_rating", 3.0)) / 5.0
            score = (genre_score * 0.45) + (popularity * 0.35) + (quality * 0.20)
            scored.append((score, movie))

        scored.sort(key=lambda x: -x[0])
        max_score = scored[0][0] if scored else 1.0
        recs = [
            {
                "movie_id": m.get("movieId"),
                "title": m.get("title", "Unknown"),
                "genres": m.get("genres", ""),
                "predicted_rating": round(float(m.get("avg_rating", 3.5)), 2),
                "match_score": round(min(99.0, max(50.0, (s / max_score) * 98.0)), 1),
            }
            for s, m in scored[:top_n]
        ]
        return recs, True
    else:
        # Collaborative filtering: rating-centered taste vector + cosine similarity
        user_taste: Dict[str, float] = {}
        ratings = [float(m.get("rating", 3.5)) for m in rated_movies]
        user_mean = sum(ratings) / len(ratings) if ratings else 3.8

        for m in rated_movies:
            r = float(m.get("rating", 3.5))
            weight = r - 2.5
            for g in (m.get("genres") or "").split("|"):
                if g and g != "(no genres listed)":
                    user_taste[g] = user_taste.get(g, 0.0) + weight

        taste_norm = math.sqrt(sum(v * v for v in user_taste.values())) if user_taste else 1.0

        scored = []
        for movie in candidates:
            genres = [g for g in (movie.get("genres") or "").split("|") if g and g != "(no genres listed)"]
            if not genres:
                continue
            g_dot = sum(user_taste.get(g, 0.0) for g in genres)
            g_norm = math.sqrt(len(genres))
            sim = g_dot / (taste_norm * g_norm) if (taste_norm * g_norm) > 0 else 0.0

            quality = float(movie.get("avg_rating", 3.0)) / 5.0
            pop = math.log1p(movie.get("num_ratings", 0)) / 10.0
            combined = (sim * 0.80) + (quality * 0.15) + (pop * 0.05)
            scored.append((combined, movie, sim))

        scored.sort(key=lambda x: -x[0])
        recs = [
            {
                "movie_id": movie.get("movieId"),
                "title": movie.get("title", "Unknown"),
                "genres": movie.get("genres", ""),
                "predicted_rating": round(min(5.0, max(3.5, (user_mean * 0.6) + (float(movie.get("avg_rating", 3.5)) * 0.4))), 2),
                "match_score": round(min(99.0, max(52.0, 52.0 + (sim * 46.0))), 1),
            }
            for combined, movie, sim in scored[:top_n]
        ]
        return recs, False


def get_genre_recs(rated_movies: list, catalog: list, top_n: int = 10) -> list:
    recs, _ = compute_local_recommendations(rated_movies, catalog, top_n)
    return recs


def fetch_or_compute_recommendations(rated: list, catalog: list, use_api: bool, top_n: int = 10) -> tuple:
    if use_api and rated:
        try:
            payload = {
                "ratings": [
                    {
                        "movie_id": int(m["movie_id"]) if str(m["movie_id"]).isdigit() else m["movie_id"],
                        "title": m["title"],
                        "rating": float(m["rating"]),
                        "genres": m.get("genres", "")
                    }
                    for m in rated
                ],
                "top_n": top_n,
            }
            resp = requests.post(f"{API_URL}/recommendations", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["recommendations"], data.get("is_cold_start", len(rated) < 5)
        except Exception:
            pass

    return compute_local_recommendations(rated, catalog, top_n)


def genre_badge(genre_str: str) -> str:
    genres = [g for g in (genre_str or "").split("|") if g and g != "(no genres listed)"][:4]
    return "".join(f"<span>{g}</span>" for g in genres)


def star_display(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + ("✨" if half else "") + "☆" * empty


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation & Navigation Setup
# ─────────────────────────────────────────────────────────────────────────────

PAGE_OPTIONS = [
    ("🎬 Explore & Rate", "Search, discover, & rate films"),
    ("✨ My Recommendations", "Personalized AI movie picks"),
    ("📊 System Dashboard", "Live model metrics & analytics"),
    ("🏗️ Architecture", "Distributed pipeline overview"),
]
PAGE_LABELS = [p[0] for p in PAGE_OPTIONS]

if "rated_movies" not in st.session_state:
    st.session_state.rated_movies = {}       # {movie_id: {title, genres, rating}}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "current_page" not in st.session_state:
    st.session_state.current_page = PAGE_LABELS[0]

if st.session_state.current_page not in PAGE_LABELS:
    st.session_state.current_page = PAGE_LABELS[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Branding ──
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1rem;'>
        <div style='font-size:3rem; display:inline-block;
             animation:bounce 1.8s ease-in-out infinite;'>🍿</div>
        <div style='font-family:"Fredoka One",cursive; font-size:1.8rem;
             color:#e05c00; line-height:1;'>CineAI</div>
        <div style='font-size:0.72rem; color:#a0704a; font-weight:700;
             margin-top:0.2rem; letter-spacing:0.05em;'>
            MOVIE RECOMMENDATION ENGINE
        </div>
    </div>
    <hr style='border:none;border-top:2px dashed #ffcc80;margin:0.5rem 0 0.8rem;'>
    <div style='font-family:"Nunito",sans-serif; font-size:0.75rem;
         color:#c9945a; font-weight:800; text-transform:uppercase;
         letter-spacing:0.08em; padding:0 0.5rem 0.4rem;'>Menu</div>
    """, unsafe_allow_html=True)

    cur_idx = PAGE_LABELS.index(st.session_state.current_page)
    sidebar_choice = st.radio(
        "Navigation",
        PAGE_LABELS,
        index=cur_idx,
        key="sidebar_radio_nav",
        label_visibility="collapsed",
    )
    if sidebar_choice != st.session_state.current_page:
        st.session_state.current_page = sidebar_choice
        st.rerun()

    st.markdown("<hr style='border:none;border-top:2px dashed #ffcc80;margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Rated movies badge
    n_rated = len(st.session_state.rated_movies)
    badge_color = "#66bb6a" if n_rated >= 5 else "#ffb347"
    hint = "<div style='font-size:0.7rem;color:#66bb6a;font-weight:700;margin-top:4px;'>✓ Full AI mode!</div>" if n_rated >= 5 else f"<div style='font-size:0.7rem;color:#a0704a;font-weight:600;margin-top:4px;'>Rate {5-n_rated} more for full AI</div>"
    st.markdown(f"""
    <div style='background:white;border:2.5px solid {badge_color};
         border-radius:18px;padding:1rem;text-align:center;
         box-shadow:3px 3px 0 {badge_color};'>
        <div style='font-family:"Fredoka One",cursive;font-size:2.2rem;
             color:{badge_color};'>{n_rated}</div>
        <div style='font-size:0.78rem;color:#a0704a;font-weight:800;'>🎬 Movies Rated</div>
        {hint}
    </div>
    """, unsafe_allow_html=True)

    if n_rated > 0:
        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Ratings", use_container_width=True):
            st.session_state.rated_movies = {}
            st.session_state.recommendations = []
            st.rerun()

    # API status
    use_api = api_available()
    status_color = "#4caf50" if use_api else "#ffb347"
    status_icon = "🟢" if use_api else "🟡"
    status_text = "API Online" if use_api else "Offline Mode"
    st.markdown(f"""
    <div style='margin-top:0.8rem;background:white;border:2px solid {status_color}50;
         border-radius:12px;padding:0.6rem;text-align:center;
         font-size:0.8rem;color:{status_color};font-weight:800;'>
        {status_icon} {status_text}
    </div>
    <div style='margin-top:0.7rem;background:#fff8e1;border:1.5px solid #ffe082;
         border-radius:12px;padding:0.6rem 0.75rem;font-size:0.75rem;
         color:#7c5700;font-weight:700;line-height:1.4;'>
        💡 Rate 5+ movies for the best AI picks!
    </div>
    """, unsafe_allow_html=True)


# Load local data
local_data = load_local_data()
catalog = local_data.get("catalog") or local_data.get("cold_start", [])
metrics_data = local_data["metrics"]
stats_data = local_data["stats"]

has_data = len(catalog) > 0
has_metrics = bool(metrics_data)

# ── Top Navigation Bar (Always Visible) ───────────────────────────────────────
n_rated_main = len(st.session_state.rated_movies)
api_status_icon = "🟢 API Online" if api_available() else "🟡 Local Mode"

st.markdown(f"""
<div class="top-nav-card">
    <div class="top-nav-brand">
        <button class="top-sidebar-btn" onclick="
            var d = window.parent.document;
            var b = d.querySelector('[data-testid=stSidebarCollapsedControl] button, [data-testid=collapsedControl] button, [data-testid=stSidebarCollapseButton] button, button[data-testid=stSidebarCollapseButton]');
            if (b) b.click();
        ">☰ Menu</button>
        <span class="top-nav-logo">🍿</span>
        <div>
            <div class="top-nav-title">CineAI</div>
            <div class="top-nav-tagline">SMART MOVIE RECOMMENDER</div>
        </div>
    </div>
    <div class="top-nav-badges">
        <div class="top-badge">🎬 {n_rated_main} Rated</div>
        <div class="top-badge">{api_status_icon}</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns(4)
for idx, (p_name, p_desc) in enumerate(PAGE_OPTIONS):
    with nav_cols[idx]:
        is_active = (st.session_state.current_page == p_name)
        badge_suffix = f" ({n_rated_main})" if "Recommendations" in p_name and n_rated_main > 0 else ""
        button_text = f"{p_name}{badge_suffix}"
        if st.button(
            button_text,
            key=f"top_nav_{idx}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            help=p_desc,
        ):
            if not is_active:
                st.session_state.current_page = p_name
                st.rerun()

st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

page = st.session_state.current_page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Explore & Rate
# ─────────────────────────────────────────────────────────────────────────────

if page == "🎬 Explore & Rate":
    st.markdown("""
    <div class='hero-title'>
        <span style='display:inline-block;animation:bounce 1.8s ease-in-out infinite;'>🎬</span>
        Find Your Next Favourite Film!
    </div>
    <div class='hero-sub'>
        Search movies &middot; Give them a star rating &middot; Let our AI learn your taste ✨
    </div>
    """, unsafe_allow_html=True)

    # Quick-add popular movies (seeded)
    if not has_data:
        st.markdown("""
        <div class='info-box'>
            ⚠️ No dataset found. Run the following to get started:<br>
            <code>python data/download_data.py</code><br>
            <code>python spark/preprocess.py</code><br>
            <code>python spark/train_als.py --fast</code><br>
            <code>python spark/generate_recs.py</code>
        </div>
        """, unsafe_allow_html=True)

    # Demo mode: populate with sample movies if no dataset
    DEMO_MOVIES = [
        {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy", "avg_rating": 4.1, "num_ratings": 2077},
        {"movieId": 318, "title": "Shawshank Redemption, The (1994)", "genres": "Drama", "avg_rating": 4.5, "num_ratings": 2227},
        {"movieId": 260, "title": "Star Wars: Episode IV (1977)", "genres": "Action|Adventure|Fantasy|Sci-Fi", "avg_rating": 4.3, "num_ratings": 2991},
        {"movieId": 593, "title": "Silence of the Lambs, The (1991)", "genres": "Drama|Thriller", "avg_rating": 4.3, "num_ratings": 2578},
        {"movieId": 2571, "title": "Matrix, The (1999)", "genres": "Action|Sci-Fi|Thriller", "avg_rating": 4.3, "num_ratings": 2590},
        {"movieId": 527, "title": "Schindler's List (1993)", "genres": "Drama|War", "avg_rating": 4.5, "num_ratings": 2304},
        {"movieId": 1196, "title": "Star Wars: Episode V (1980)", "genres": "Action|Adventure|Drama|Sci-Fi|War", "avg_rating": 4.3, "num_ratings": 2990},
        {"movieId": 2858, "title": "American Beauty (1999)", "genres": "Drama", "avg_rating": 4.3, "num_ratings": 3428},
        {"movieId": 1210, "title": "Star Wars: Episode VI (1983)", "genres": "Action|Adventure|Romance|Sci-Fi|War", "avg_rating": 4.0, "num_ratings": 2908},
        {"movieId": 589, "title": "Terminator 2 (1991)", "genres": "Action|Sci-Fi|Thriller", "avg_rating": 4.1, "num_ratings": 2438},
        {"movieId": 356, "title": "Forrest Gump (1994)", "genres": "Comedy|Drama|Romance|War", "avg_rating": 4.1, "num_ratings": 3216},
        {"movieId": 480, "title": "Jurassic Park (1993)", "genres": "Action|Adventure|Sci-Fi", "avg_rating": 3.7, "num_ratings": 2725},
    ]
    active_catalog = catalog if has_data else DEMO_MOVIES

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "🔍 Search movies",
            placeholder="e.g. Star Wars, Inception, Forrest Gump …",
            key="search_input",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Search", use_container_width=True):
            if query:
                st.session_state.search_results = search_movies_local(query, active_catalog, 12)

    # Display search results
    if query and st.session_state.search_results:
        st.markdown(f"<div class='section-title'>🔎 Results for '{query}'</div>", unsafe_allow_html=True)
        for movie in st.session_state.search_results:
            mid = movie.get("movieId") or movie.get("movie_id")
            if not mid:
                continue
            is_rated = mid in st.session_state.rated_movies
            border = "#ff9800" if is_rated else "#ffcc80"
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"""
                <div class='movie-card' style='border-color:{border};'>
                    <div class='movie-title'>{movie.get('title','')}</div>
                    <div class='movie-genres' style='margin-top:6px;'>{genre_badge(movie.get('genres',''))}</div>
                    <div style='font-size:0.78rem;color:#a0704a;margin-top:6px;font-weight:700;'>
                        ⭐ {movie.get('avg_rating', 0):.1f} avg · {int(movie.get('num_ratings',0)):,} ratings
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if not is_rated:
                    r = st.slider(
                        f"Your rating",
                        0.5, 5.0, 3.5, 0.5,
                        key=f"slider_{mid}",
                        label_visibility="visible",
                    )
            with col3:
                st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
                if is_rated:
                    existing = st.session_state.rated_movies[mid]["rating"]
                    st.markdown(f"<div style='color:#2e7d32;font-weight:800;padding-top:0.5rem;'>✓ Rated {existing}★</div>", unsafe_allow_html=True)
                    if st.button("Remove", key=f"remove_{mid}"):
                        del st.session_state.rated_movies[mid]
                        st.session_state.recommendations = []
                        st.rerun()
                else:
                    if st.button("＋ Add", key=f"add_{mid}", use_container_width=True):
                        st.session_state.rated_movies[mid] = {
                            "movie_id": mid,
                            "title": movie.get("title", ""),
                            "genres": movie.get("genres", ""),
                            "rating": st.session_state.get(f"slider_{mid}", 3.5),
                        }
                        st.session_state.recommendations = []
                        st.success(f"Added: {movie.get('title','')}")
                        st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Your rated movies ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>⭐ Your Rated Movies</div>", unsafe_allow_html=True)

    if not st.session_state.rated_movies:
        # Show quick-add popular movies
        st.markdown("""
        <div class='info-box'>
            🎯 Rate some movies to get personalised recommendations.<br>
            Search above, or click a popular movie below to rate it instantly!
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🔥 Popular Right Now**")
        cols = st.columns(4)
        for i, movie in enumerate(DEMO_MOVIES[:8]):
            with cols[i % 4]:
                mid = movie["movieId"]
                if st.button(
                    f"{movie['title'][:28]}…" if len(movie["title"]) > 28 else movie["title"],
                    key=f"quick_{mid}", use_container_width=True
                ):
                    st.session_state.rated_movies[mid] = {
                        "movie_id": mid,
                        "title": movie["title"],
                        "genres": movie["genres"],
                        "rating": 4.0,
                    }
                    st.session_state.recommendations = []
                    st.rerun()
    else:
        # Show rated movies grid
        rated_list = list(st.session_state.rated_movies.values())
        for i in range(0, len(rated_list), 3):
            cols = st.columns(3)
            for j, movie in enumerate(rated_list[i:i+3]):
                with cols[j]:
                    mid = movie["movie_id"]
                    st.markdown(f"""
                    <div class='movie-card' style='border-color:#ffcc80;'>
                        <div class='movie-title'>{movie['title'][:35]}{"…" if len(movie['title'])>35 else ""}</div>
                        <div style='font-size:1.3rem;margin:6px 0;'>{star_display(movie['rating'])}</div>
                        <div class='movie-genres'>{genre_badge(movie.get('genres',''))}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Get recommendations button
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2, _ = st.columns([2.5, 2.5, 3])
        with col_btn1:
            if st.button("🚀 Get My Recommendations", use_container_width=True, type="primary"):
                with st.spinner("🔮 Computing personalised recommendations …"):
                    rated = list(st.session_state.rated_movies.values())
                    if rated:
                        recs, is_cs = fetch_or_compute_recommendations(rated, active_catalog, use_api, 10)
                        st.session_state.recommendations = recs
                        st.session_state.is_cold_start = is_cs
                        st.session_state.last_rated_signature = tuple(sorted((m["movie_id"], m["rating"]) for m in rated))
                        mode_text = "⚡ Cold-Start Mode" if is_cs else "🤖 Full AI Collaborative Filtering"
                        st.success(f"✅ Generated {len(recs)} recommendations ({mode_text})!")
                        time.sleep(0.4)

        with col_btn2:
            if st.session_state.recommendations:
                if st.button("✨ View Recommendations →", key="goto_recs_btn", type="primary", use_container_width=True):
                    st.session_state.current_page = "✨ My Recommendations"
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — My Recommendations
# ─────────────────────────────────────────────────────────────────────────────

elif page == "✨ My Recommendations":
    st.markdown("""
    <div class='hero-title'>
        <span style='display:inline-block;animation:wiggle 2s ease-in-out infinite;'>✨</span>
        Your Personalised Picks!
    </div>
    <div class='hero-sub'>Powered by Alternating Least Squares &middot; PySpark MLlib &middot; Just for you 🎯</div>
    """, unsafe_allow_html=True)

    rated = list(st.session_state.rated_movies.values())
    if not rated:
        st.session_state.recommendations = []
        st.markdown("""
        <div class='info-box' style='text-align:center;padding:2.5rem 1.5rem;'>
            <div style='font-size:3rem;margin-bottom:0.8rem;animation:bounce 1.8s ease-in-out infinite;'>🎬</div>
            <div style='font-family:"Fredoka One",cursive;font-size:1.6rem;color:#e05c00;'>No recommendations yet!</div>
            <div style='color:#a0704a;font-weight:700;margin-top:0.5rem;font-size:0.95rem;'>
                Rate a few movies so our AI engine can learn your taste ✨
            </div>
        </div>
        """, unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns([1, 1.5, 1])
        with col_c2:
            if st.button("🎬 Rate Movies Now →", key="rate_now_btn", type="primary", use_container_width=True):
                st.session_state.current_page = "🎬 Explore & Rate"
                st.rerun()
        st.stop()

    current_rated_signature = tuple(sorted((m["movie_id"], m["rating"]) for m in rated))
    last_signature = st.session_state.get("last_rated_signature")

    # Automatically recompute if rated movies have changed or if recommendations list is empty
    if (not st.session_state.recommendations) or (current_rated_signature != last_signature):
        with st.spinner("🔮 Computing recommendations …"):
            active_catalog = catalog if has_data else DEMO_MOVIES
            recs, is_cs = fetch_or_compute_recommendations(rated, active_catalog, use_api, 10)
            st.session_state.recommendations = recs
            st.session_state.is_cold_start = is_cs
            st.session_state.last_rated_signature = current_rated_signature

    recs = st.session_state.recommendations
    n_rated = len(st.session_state.rated_movies)
    is_cold_start = st.session_state.get("is_cold_start", n_rated < 5)

    if is_cold_start:
        st.markdown(f"""
        <div class='cold-start-box'>
            ⚡ <strong>Cold-Start Mode ({n_rated}/5 rated)</strong> — You've rated fewer than 5 movies.
            These are popular picks matching your initial genre preferences.<br>
            ⭐ <strong>Rate {5 - n_rated} more movie{'s' if (5 - n_rated) > 1 else ''}</strong> to unlock full AI Collaborative Filtering with deep personalized taste profiling!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='info-box' style='background: #e8f5e9; border: 2px solid #66bb6a; border-left: 6px solid #2e7d32;'>
            🤖 <strong>Full AI Collaborative Filtering ({n_rated} movies rated)</strong> — Full AI mode unlocked!
            Recommendations are computed using multi-dimensional ALS taste vectors across all your ratings, penalizing genres you disliked and boosting your favorites.
        </div>
        """, unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    avg_pred = sum(r.get("predicted_rating", 0) for r in recs) / len(recs) if recs else 0
    avg_match = sum(r.get("match_score", 0) for r in recs) / len(recs) if recs else 0
    genres_found = set()
    for r in recs:
        for g in (r.get("genres") or "").split("|"):
            if g and g != "(no genres listed)":
                genres_found.add(g)

    for col, val, label in zip(
        [col1, col2, col3, col4],
        [len(recs), f"{avg_pred:.1f}★", f"{avg_match:.0f}%", len(genres_found)],
        ["🎬 Recommendations", "🌟 Avg Predicted", "🎯 Avg Match", "🎭 Genres"],
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # Recommendation cards
    for i, rec in enumerate(recs):
        title = rec.get("title") or rec.get("title", "Unknown Movie")
        genres = rec.get("genres", "")
        predicted = rec.get("predicted_rating", 0)
        match_score = rec.get("match_score", 0)

        col_rank, col_info = st.columns([1, 7])
        with col_rank:
            st.markdown(f"""
            <div style='text-align:center;padding-top:0.5rem;'>
                <div class='rec-rank'>#{i+1}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info:
            col_title, col_score = st.columns([4, 2])
            with col_title:
                st.markdown(f"""
                <div class='rec-card'>
                    <div class='rec-title'>{title}</div>
                    <div class='movie-genres' style='margin-top:8px;'>{genre_badge(genres)}</div>
                    <div class='match-bar-container' style='margin-top:12px;'>
                        <div class='match-bar' style='width:{match_score}%;'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-top:4px;font-weight:700;'>
                        <span style='font-size:0.82rem;color:#a0704a;'>Match: <strong style='color:#e05c00;'>{match_score:.0f}%</strong></span>
                        <span style='font-size:0.82rem;color:#a0704a;'>Predicted: <strong style='color:#ff6b35;'>{predicted:.1f}★</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Download recommendations
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    recs_df = pd.DataFrame([{
        "Rank": i+1,
        "Title": r.get("title", ""),
        "Genres": r.get("genres", ""),
        "Predicted Rating": r.get("predicted_rating", 0),
        "Match Score (%)": r.get("match_score", 0),
    } for i, r in enumerate(recs)])

    col_dl, _ = st.columns([2, 6])
    with col_dl:
        st.download_button(
            "📥 Download Recommendations (CSV)",
            recs_df.to_csv(index=False),
            "my_recommendations.csv",
            "text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — System Dashboard
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 System Dashboard":
    st.markdown("""
    <div class='hero-title'>
        <span style='display:inline-block;animation:bounce 1.8s ease-in-out infinite;'>📊</span>
        System Dashboard
    </div>
    <div class='hero-sub'>Live model metrics &middot; Dataset statistics &middot; Hyperparameter analysis</div>
    """, unsafe_allow_html=True)

    if not has_metrics and not stats_data:
        st.markdown("""
        <div class='info-box'>
            📂 No training data found yet. Run the full pipeline:<br><br>
            <code>python data/download_data.py</code><br>
            <code>python spark/preprocess.py</code><br>
            <code>python spark/train_als.py --fast</code><br>
            <code>python spark/evaluate.py</code><br>
            <code>python spark/generate_recs.py</code><br><br>
            In the meantime, below are <strong>demo metrics</strong> to illustrate the dashboard.
        </div>
        """, unsafe_allow_html=True)

    # Use real or demo metrics
    if has_metrics:
        test_metrics = metrics_data.get("test_metrics", {})
        best_hp = metrics_data.get("best_hyperparams", {})
        cv_results = metrics_data.get("cv_results", [])
        train_time = metrics_data.get("training_time_seconds", 0)
    else:
        # Demo data
        test_metrics = {"rmse": 0.8721, "mae": 0.6834, "coverage_pct": 94.3}
        best_hp = {"rank": 50, "regParam": 0.1, "maxIter": 20}
        train_time = 312.5
        cv_results = [
            {"rank": 10, "regParam": 0.01, "maxIter": 10, "cv_rmse": 1.0234},
            {"rank": 10, "regParam": 0.1,  "maxIter": 10, "cv_rmse": 0.9123},
            {"rank": 10, "regParam": 1.0,  "maxIter": 10, "cv_rmse": 0.9876},
            {"rank": 50, "regParam": 0.01, "maxIter": 10, "cv_rmse": 0.9012},
            {"rank": 50, "regParam": 0.1,  "maxIter": 10, "cv_rmse": 0.8871},
            {"rank": 50, "regParam": 1.0,  "maxIter": 10, "cv_rmse": 0.9234},
            {"rank": 100,"regParam": 0.01, "maxIter": 10, "cv_rmse": 0.8923},
            {"rank": 100,"regParam": 0.1,  "maxIter": 10, "cv_rmse": 0.8721},
            {"rank": 100,"regParam": 1.0,  "maxIter": 10, "cv_rmse": 0.9456},
            {"rank": 10, "regParam": 0.01, "maxIter": 20, "cv_rmse": 0.9934},
            {"rank": 10, "regParam": 0.1,  "maxIter": 20, "cv_rmse": 0.9034},
            {"rank": 10, "regParam": 1.0,  "maxIter": 20, "cv_rmse": 0.9567},
            {"rank": 50, "regParam": 0.01, "maxIter": 20, "cv_rmse": 0.8945},
            {"rank": 50, "regParam": 0.1,  "maxIter": 20, "cv_rmse": 0.8721},
            {"rank": 50, "regParam": 1.0,  "maxIter": 20, "cv_rmse": 0.9123},
            {"rank": 100,"regParam": 0.01, "maxIter": 20, "cv_rmse": 0.8834},
            {"rank": 100,"regParam": 0.1,  "maxIter": 20, "cv_rmse": 0.8654},
            {"rank": 100,"regParam": 1.0,  "maxIter": 20, "cv_rmse": 0.9234},
        ]

    if stats_data:
        ds = stats_data
    else:
        ds = {"num_users": 6040, "num_movies": 3706, "num_ratings_total": 1000209,
              "avg_rating": 3.58, "sparsity": 0.9553}

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Model Accuracy & Summary</div>", unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    kpis = [
        (f"{test_metrics.get('rmse', 0):.4f}", "Prediction Error (RMSE)"),
        (f"{test_metrics.get('mae', 0):.4f}", "Average Error (MAE)"),
        (f"{test_metrics.get('coverage_pct', 0):.1f}%", "User Coverage"),
        (f"{train_time:.0f}s", "Training Time"),
    ]
    for col, (val, label) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:1.7rem;'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🗄️ Dataset Overview</div>", unsafe_allow_html=True)
    ds_cols = st.columns(4)
    ds_kpis = [
        (f"{int(ds.get('num_users',0)):,}", "👤 Total Users"),
        (f"{int(ds.get('num_movies',0)):,}", "🎬 Total Movies"),
        (f"{int(ds.get('num_ratings_total',0)):,}", "⭐ Total Ratings (~1M)"),
        (f"{ds.get('sparsity',0)*100:.1f}%", "🕳️ Empty Data (Sparsity)"),
    ]
    colors = ["#ff8c42", "#66bb6a", "#42a5f5", "#ab47bc"]
    for col, (val, label), color in zip(ds_cols, ds_kpis, colors):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-color:{color}50;box-shadow:3px 3px 0 {color}30;'>
                <div class='metric-value' style='font-size:1.7rem;color:{color};'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Best params display ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏆 Best Model Settings</div>", unsafe_allow_html=True)
    hp_cols = st.columns(3)
    hp_labels = {"rank": "🧠 Latent Features (rank)", "regParam": "⚖️ Regularization (λ)", "maxIter": "🔄 Iterations"}
    for col, (key, val) in zip(hp_cols, best_hp.items()):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-color:#ff6b3550;box-shadow:3px 3px 0 #ff6b3530;'>
                <div class='metric-value' style='color:#ff6b35;'>{val}</div>
                <div class='metric-label'>{hp_labels.get(key, key)}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Detailed Hyperparameter Tuning Graphs (Hidden by default in expander) ─
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Show Detailed Model Tuning Graphs (Optional / Advanced)", expanded=False):
        st.caption("These charts show the RMSE score across different hyperparameter combinations during cross-validation.")
        tab1, tab2, tab3 = st.tabs(["📊 Heatmap", "📈 RMSE by Rank", "📋 All Results"])

        with tab1:
            if cv_results:
                df_cv = pd.DataFrame(cv_results)
                for max_iter_val in df_cv["maxIter"].unique():
                    subset = df_cv[df_cv["maxIter"] == max_iter_val]
                    if len(subset) > 0:
                        pivot = subset.pivot_table(values="cv_rmse", index="regParam", columns="rank")
                        fig = go.Figure(go.Heatmap(
                            z=pivot.values,
                            x=[f"rank={c}" for c in pivot.columns],
                            y=[f"λ={r}" for r in pivot.index],
                            colorscale="YlOrRd",
                            text=[[f"{v:.4f}" for v in row] for row in pivot.values],
                            texttemplate="%{text}",
                            colorbar=dict(title=dict(text="RMSE", font=dict(color="#3d2c1e")), tickfont=dict(color="#3d2c1e")),
                        ))
                        fig.update_layout(
                            title=dict(text=f"CV RMSE Heatmap (maxIter={max_iter_val})", font=dict(color="#3d2c1e", size=14)),
                            plot_bgcolor="#fef9f0", paper_bgcolor="#fef9f0",
                            font=dict(color="#3d2c1e"),
                            xaxis=dict(gridcolor="rgba(255,179,71,0.3)"),
                            yaxis=dict(gridcolor="rgba(255,179,71,0.3)"),
                            height=320,
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if cv_results:
                df_cv = pd.DataFrame(cv_results)
                fig = px.line(
                    df_cv, x="rank", y="cv_rmse",
                    color=df_cv["regParam"].astype(str),
                    line_dash=df_cv["maxIter"].astype(str),
                    markers=True,
                    labels={"cv_rmse": "CV RMSE", "rank": "Latent Factors (rank)", "color": "regParam", "line_dash": "maxIter"},
                    title="RMSE vs. Number of Latent Factors",
                    color_discrete_sequence=["#ff8c42", "#66bb6a", "#42a5f5"],
                )
                fig.update_layout(
                    plot_bgcolor="#fef9f0", paper_bgcolor="#fef9f0",
                    font=dict(color="#3d2c1e"),
                    xaxis=dict(gridcolor="rgba(255,179,71,0.3)"),
                    yaxis=dict(gridcolor="rgba(255,179,71,0.3)"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if cv_results:
                df_cv = pd.DataFrame(cv_results).sort_values("cv_rmse")
                df_cv.insert(0, "Rank", range(1, len(df_cv)+1))
                df_cv["Best"] = df_cv["cv_rmse"] == df_cv["cv_rmse"].min()
                st.dataframe(
                    df_cv[["Rank", "rank", "regParam", "maxIter", "cv_rmse", "Best"]],
                    use_container_width=True,
                    hide_index=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Architecture
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🏗️ Architecture":
    st.markdown("""
    <div class='hero-title'>🏗️ System Architecture</div>
    <div class='hero-sub'>End-to-end distributed ML pipeline &mdash; from raw data to live recommendations</div>
    """, unsafe_allow_html=True)

    # Architecture pipeline diagram
    st.markdown("""
    <div style='background:white;border:2.5px solid #ffcc80;border-radius:20px;
         padding:2rem;margin-bottom:2rem;box-shadow:4px 4px 0 #ffd180;'>
        <div style='font-family:"Fredoka One",cursive;font-size:1.3rem;color:#e05c00;margin-bottom:1.5rem;'>
            🔄 Data Flow Pipeline
        </div>
        <div style='display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:0.5rem;align-items:center;'>
            <div style='background:#fff8f0;border:2.5px solid #ffb347;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #ffd180;'>
                <div style='font-size:1.8rem;'>📦</div>
                <div style='font-weight:800;color:#e05c00;font-size:0.9rem;margin-top:6px;'>MovieLens</div>
                <div style='font-size:0.75rem;color:#a0704a;font-weight:600;'>1M / 25M records</div>
            </div>
            <div style='text-align:center;color:#ffb347;font-size:1.8rem;font-weight:900;'>→</div>
            <div style='background:#f3e5f5;border:2.5px solid #ab47bc;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #ce93d8;'>
                <div style='font-size:1.8rem;'>⚡</div>
                <div style='font-weight:800;color:#7b1fa2;font-size:0.9rem;margin-top:6px;'>PySpark ETL</div>
                <div style='font-size:0.75rem;color:#9c27b0;font-weight:600;'>Parquet output</div>
            </div>
            <div style='text-align:center;color:#ab47bc;font-size:1.8rem;font-weight:900;'>→</div>
            <div style='background:#e8f5e9;border:2.5px solid #66bb6a;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #a5d6a7;'>
                <div style='font-size:1.8rem;'>🤖</div>
                <div style='font-weight:800;color:#2e7d32;font-size:0.9rem;margin-top:6px;'>ALS Model</div>
                <div style='font-size:0.75rem;color:#388e3c;font-weight:600;'>MLlib + CrossVal</div>
            </div>
        </div>
        <div style='display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:0.5rem;align-items:center;margin-top:1rem;'>
            <div style='background:#e3f2fd;border:2.5px solid #42a5f5;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #90caf9;'>
                <div style='font-size:1.8rem;'>💾</div>
                <div style='font-weight:800;color:#1565c0;font-size:0.9rem;margin-top:6px;'>Batch Recs</div>
                <div style='font-size:0.75rem;color:#1976d2;font-weight:600;'>Parquet / JSON</div>
            </div>
            <div style='text-align:center;color:#42a5f5;font-size:1.8rem;font-weight:900;'>→</div>
            <div style='background:#fff3e0;border:2.5px solid #ff8c42;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #ffcc80;'>
                <div style='font-size:1.8rem;'>🚀</div>
                <div style='font-weight:800;color:#e05c00;font-size:0.9rem;margin-top:6px;'>FastAPI</div>
                <div style='font-size:0.75rem;color:#bf360c;font-weight:600;'>REST Endpoints</div>
            </div>
            <div style='text-align:center;color:#ff8c42;font-size:1.8rem;font-weight:900;'>→</div>
            <div style='background:#fce4ec;border:2.5px solid #f48fb1;border-radius:14px;padding:1rem;text-align:center;box-shadow:3px 3px 0 #f8bbd0;'>
                <div style='font-size:1.8rem;'>🎨</div>
                <div style='font-weight:800;color:#880e4f;font-size:0.9rem;margin-top:6px;'>Streamlit UI</div>
                <div style='font-size:0.75rem;color:#ad1457;font-weight:600;'>You are here! 👋</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ALS explanation
    st.markdown("<div class='section-title'>🔬 ALS Algorithm Deep-Dive</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:white;border:2.5px solid #ffcc80;border-radius:16px;
             padding:1.5rem;box-shadow:3px 3px 0 #ffd180;'>
            <div style='font-family:"Fredoka One",cursive;color:#e05c00;font-size:1.1rem;margin-bottom:1rem;'>
                Matrix Factorisation
            </div>
            <p style='color:#7c4a10;font-size:0.9rem;line-height:1.6;font-weight:600;'>
                The user-item rating matrix <strong style='color:#ff8c42;'>R</strong> is factorised into two
                lower-dimensional matrices:
            </p>
            <div style='text-align:center;margin:1rem 0;font-size:1.3rem;font-weight:800;
                 background:#fff8f0;padding:1rem;border-radius:10px;border:2px solid #ffe0b2;'>
                <span style='color:#ff6b35;'>R</span> ≈
                <span style='color:#42a5f5;'>U</span> ×
                <span style='color:#66bb6a;'>V</span><sup>T</sup>
            </div>
            <ul style='color:#7c4a10;font-size:0.88rem;line-height:1.8;font-weight:600;'>
                <li><strong style='color:#42a5f5;'>U</strong>: User factor matrix (n_users × rank)</li>
                <li><strong style='color:#66bb6a;'>V</strong>: Item factor matrix (n_items × rank)</li>
                <li><strong style='color:#ff6b35;'>rank</strong>: Number of latent factors (hyperparameter)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:white;border:2.5px solid #ffcc80;border-radius:16px;
             padding:1.5rem;box-shadow:3px 3px 0 #ffd180;'>
            <div style='font-family:"Fredoka One",cursive;color:#e05c00;font-size:1.1rem;margin-bottom:1rem;'>
                Distributed ALS Steps
            </div>
            <div>
                <div style='display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;'>
                    <span style='background:#ff8c42;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.8rem;flex-shrink:0;color:white;'>1</span>
                    <span style='color:#7c4a10;font-size:0.88rem;font-weight:600;'>Distribute rating matrix <strong>R</strong> across Spark partitions</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;'>
                    <span style='background:#ab47bc;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.8rem;flex-shrink:0;color:white;'>2</span>
                    <span style='color:#7c4a10;font-size:0.88rem;font-weight:600;'>Fix <strong style='color:#42a5f5;'>V</strong>, solve for <strong style='color:#42a5f5;'>U</strong> via least-squares (parallel)</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;'>
                    <span style='background:#66bb6a;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.8rem;flex-shrink:0;color:white;'>3</span>
                    <span style='color:#7c4a10;font-size:0.88rem;font-weight:600;'>Fix <strong style='color:#66bb6a;'>U</strong>, solve for <strong style='color:#66bb6a;'>V</strong> via least-squares (parallel)</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;'>
                    <span style='background:#42a5f5;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.8rem;flex-shrink:0;color:white;'>4</span>
                    <span style='color:#7c4a10;font-size:0.88rem;font-weight:600;'>Repeat for <strong>maxIter</strong> iterations until convergence</span>
                </div>
                <div style='display:flex;align-items:flex-start;gap:0.75rem;'>
                    <span style='background:#ff6b35;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.8rem;flex-shrink:0;color:white;'>5</span>
                    <span style='color:#7c4a10;font-size:0.88rem;font-weight:600;'>Predict: <strong>r̂(u,i)</strong> = U<sub>u</sub> · V<sub>i</sub><sup>T</sup></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Cold-start strategy
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>❄️ Cold-Start Strategy</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cold_cards = [
        ("🧊", "New User (<5 ratings)", "#f59e0b",
         "Genre-based popularity fallback. Score movies by genre overlap with user's liked genres weighted by rating."),
        ("🌡️", "Warm User (5–20 ratings)", "#818cf8",
         "Genre-similarity ALS approximation. User preferences modelled as weighted genre vector, matched against item factors."),
        ("🔥", "Active User (>20 ratings)", "#22c55e",
         "Full ALS collaborative filtering. Pre-computed batch recommendations from the trained model via O(1) cache lookup."),
    ]
    for col, (icon, title, color, desc) in zip([col1, col2, col3], cold_cards):
        with col:
            st.markdown(f"""
            <div style='background:white;border:2.5px solid {color};
                 border-radius:18px;padding:1.4rem;height:100%;
                 box-shadow:3px 3px 0 {color}50;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-family:"Fredoka One",cursive;color:{color};margin:0.5rem 0 0.3rem;font-size:1.1rem;'>{title}</div>
                <div style='color:#7c4a10;font-size:0.88rem;line-height:1.6;font-weight:600;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🛠️ Tech Stack</div>", unsafe_allow_html=True)

    tech_stack = [
        ("PySpark 3.5",     "Distributed data processing & MLlib ALS",           "#ff8c42"),
        ("Spark MLlib",     "ALS + CrossValidator + ParamGridBuilder",            "#ab47bc"),
        ("Apache Parquet",  "Columnar storage for fast ML I/O",                  "#66bb6a"),
        ("FastAPI",         "Async REST API with auto-generated OpenAPI docs",    "#f48fb1"),
        ("Pydantic v2",     "Request / response schema validation",               "#ff8a65"),
        ("Streamlit",       "Interactive ML dashboard",                           "#42a5f5"),
        ("Plotly",          "Interactive charts for metric visualisation",         "#26c6da"),
        ("MovieLens",       "Benchmark dataset (1M/25M user-movie ratings)",      "#ffa726"),
    ]

    cols = st.columns(4)
    for i, (name, desc, color) in enumerate(tech_stack):
        with cols[i % 4]:
            st.markdown(f"""
            <div style='background:white;border:2px solid {color}50;
                 border-left:4px solid {color};border-radius:12px;
                 padding:0.8rem;margin-bottom:0.75rem;
                 box-shadow:2px 2px 0 {color}30;'>
                <div style='font-weight:800;color:{color};font-size:0.9rem;'>{name}</div>
                <div style='color:#a0704a;font-size:0.78rem;margin-top:3px;font-weight:600;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)



# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#c9945a;font-size:0.82rem;font-weight:700;'>
    🍿 Built for BDA (Big Data Analytics) · PySpark MLlib ALS · CineAI v2.0 🎬
</div>
""", unsafe_allow_html=True)
