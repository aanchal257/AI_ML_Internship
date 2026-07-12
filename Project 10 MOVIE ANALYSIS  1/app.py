"""
Dhurandhar 2 — Review Intelligence Studio
A Streamlit dashboard with dataset insights and instant review summarization,
built on Hugging Face Transformers.
"""
 
import os
import time
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
 
# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dhurandhar 2 | Review Intelligence Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
 
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
 
    .stApp {
        background: radial-gradient(circle at 10% 0%, #1a0a12 0%, #0b0710 45%, #050308 100%);
        color: #f2eef5;
    }
 
    #MainMenu, footer, header {visibility: hidden;}
 
    .hero {
        padding: 2.4rem 2.6rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #3b0a2e 0%, #6b0f2a 45%, #b5182c 100%);
        box-shadow: 0 20px 60px rgba(181, 24, 44, 0.35);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after{
        content:"";
        position:absolute; top:-40%; right:-10%;
        width:420px; height:420px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.05rem;
        color: #f4d9df;
        max-width: 720px;
        margin: 0;
    }
    .badge-row { margin-top: 1.1rem; display:flex; gap:0.6rem; flex-wrap:wrap; }
    .badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        backdrop-filter: blur(6px);
    }
 
    .metric-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        transition: all 0.25s ease;
    }
    .metric-card:hover { border-color: rgba(255,255,255,0.28); transform: translateY(-3px); }
    .metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #d79bb0; font-weight: 600;}
    .metric-value { font-family:'Poppins', sans-serif; font-size: 2rem; font-weight: 700; margin-top: 2px;}
    .metric-sub { font-size: 0.78rem; color: #b8adbf; margin-top: 2px;}
 
    .review-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid #b5182c;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .review-card.pos { border-left-color: #34c98e; }
    .review-card.neg { border-left-color: #ff5470; }
    .rc-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;}
    .rc-tag { font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:999px; letter-spacing:0.5px;}
    .rc-tag.pos { background: rgba(52,201,142,0.15); color:#34c98e; border:1px solid rgba(52,201,142,0.4);}
    .rc-tag.neg { background: rgba(255,84,112,0.15); color:#ff5470; border:1px solid rgba(255,84,112,0.4);}
    .rc-conf { font-size:0.75rem; color:#b8adbf; }
    .rc-text { font-size:0.88rem; color:#e6dfe9; line-height:1.5; }
    .match-yes { color:#34c98e; font-weight:700;}
    .match-no { color:#ff5470; font-weight:700;}
 
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12060d 0%, #0b0710 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
 
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: #cfc3d3;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #b5182c, #6b0f2a);
        color: white !important;
    }
 
    .stButton>button {
        background: linear-gradient(120deg, #b5182c, #8a1224);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 8px 20px rgba(181,24,44,0.3);
    }
    .stButton>button:hover { filter: brightness(1.12); }
 
    .section-title {
        font-family:'Poppins', sans-serif;
        font-weight:700;
        font-size:1.35rem;
        margin: 0.4rem 0 1rem 0;
        color: #f2eef5;
    }
    .footer-note { color:#7d7284; font-size:0.78rem; text-align:center; margin-top:2.5rem; }
</style>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# DATA LOADING — CSV must sit in the SAME FOLDER as app.py (repo root).
# No subfolders, no upload widget — just keep the two files together.
# --------------------------------------------------------------------------------------
DATA_FILENAME = "netflix_movie_dhurandhar_2.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILENAME)
 
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Could not find '{DATA_FILENAME}'. Make sure it's uploaded to the same "
            f"folder as app.py (repo root) — no subfolders."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH, delimiter=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df
 
df = load_data()
reviews = df["Review"].tolist()
real_labels = df["Class"].tolist()
 
# --------------------------------------------------------------------------------------
# MODEL LOADING (cached) — pinned to CPU explicitly + low_cpu_mem_usage to keep memory
# footprint small on resource-constrained hosting (e.g. Streamlit Community Cloud)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_summarization_pipeline():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        device=-1,
        model_kwargs={"low_cpu_mem_usage": True},
    )
 
def safe_load(loader_fn, friendly_name):
    """Load a pipeline and surface a clean, actionable error instead of a raw traceback."""
    import traceback
    try:
        return loader_fn()
    except Exception as e:
        st.error(f"⚠️ Couldn't load the {friendly_name} model.")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        st.stop()
 
def safe_call(fn, *args, friendly_name="model", **kwargs):
    """Run inference and surface a clean, actionable error instead of a raw traceback."""
    import traceback
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ The {friendly_name} step failed while running.")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        st.stop()
 
# --------------------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 Dhurandhar 2 — Review Intelligence Studio</h1>
    <p>Step inside the audience reaction — every review, distilled to what matters most.</p>
    <div class="badge-row">
        <span class="badge">⚡ Smart Summaries</span>
        <span class="badge">📊 Audience Insights</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 Dhurandhar 2")
    st.caption("Audience reviews, decoded.")
    st.markdown("---")
    pos_count = real_labels.count("POSITIVE")
    neg_count = real_labels.count("NEGATIVE")
    st.markdown(f"**Total reviews:** {len(df)}")
    st.markdown(f"**Positive:** {pos_count}  |  **Negative:** {neg_count}")
    st.markdown("---")
    st.caption("Made with 🩷")
 
# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Overview", "📝 Summarization"])
 
# ========================================================================================
# TAB 1 — OVERVIEW
# ========================================================================================
with tab1:
    st.markdown('<div class="section-title">Dataset at a Glance</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    avg_len = int(df["Review"].str.split().apply(len).mean())
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Reviews</div>
        <div class="metric-value">{len(df)}</div><div class="metric-sub">Curated critic & audience reviews</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Positive Reviews</div>
        <div class="metric-value" style="color:#34c98e;">{pos_count}</div><div class="metric-sub">{pos_count/len(df)*100:.0f}% of dataset</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Negative Reviews</div>
        <div class="metric-value" style="color:#ff5470;">{neg_count}</div><div class="metric-sub">{neg_count/len(df)*100:.0f}% of dataset</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Review Length</div>
        <div class="metric-value">{avg_len}</div><div class="metric-sub">words per review</div></div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.4])
 
    with left:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=["Positive", "Negative"],
            values=[pos_count, neg_count],
            hole=0.62,
            marker=dict(colors=["#34c98e", "#ff5470"]),
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300,
            font=dict(color="#f2eef5"),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with right:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Review Length Distribution</div>', unsafe_allow_html=True)
        lengths = df["Review"].str.split().apply(len)
        fig2 = px.histogram(x=lengths, nbins=10, color_discrete_sequence=["#b5182c"])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), height=300,
            xaxis_title="Words per review", yaxis_title="Count",
            font=dict(color="#f2eef5"),
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown('<div class="section-title">Browse Reviews</div>', unsafe_allow_html=True)
    for i, (rev, lab) in enumerate(zip(reviews, real_labels)):
        cls = "pos" if lab == "POSITIVE" else "neg"
        snippet = rev if len(rev) < 380 else rev[:380].rsplit(" ", 1)[0] + " …"
        st.markdown(f"""
        <div class="review-card {cls}">
            <div class="rc-head">
                <span class="rc-tag {cls}">{lab}</span>
                <span class="rc-conf">Review #{i+1}</span>
            </div>
            <div class="rc-text">{snippet}</div>
        </div>
        """, unsafe_allow_html=True)
 
# ========================================================================================
# TAB 2 — SUMMARIZATION
# ========================================================================================
with tab2:
    st.markdown('<div class="section-title">Review Summarization</div>', unsafe_allow_html=True)
    st.caption("Condenses long reviews into concise summaries, generated instantly as you pick a review.")
 
    sum_idx = st.selectbox(
        "Select a review to summarize:",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review #{i+1} ({real_labels[i]}) — {reviews[i][:70]}...",
        key="sum_select",
    )
    full_review = reviews[sum_idx]
    st.markdown(f'<div class="review-card">{full_review}</div>', unsafe_allow_html=True)
 
    max_len = st.slider("Max summary length (tokens):", 30, 150, 80)
    min_len = st.slider("Min summary length (tokens):", 10, 60, 25)
 
    if min_len >= max_len:
        st.warning("Min length must be smaller than max length — adjust the sliders.")
    else:
        with st.spinner("Generating summary..."):
            summarizer = safe_load(load_summarization_pipeline, "summarization")
            summary = safe_call(
                summarizer, full_review, max_length=max_len, min_length=min_len,
                do_sample=False, truncation=True, friendly_name="summarization"
            )[0]["summary_text"]
        orig_words = len(full_review.split())
        summ_words = len(summary.split())
        st.markdown(f"""
        <div class="review-card pos">
            <div class="rc-head">
                <span class="rc-tag pos">Summary</span>
                <span class="rc-conf">{orig_words} words → {summ_words} words ({(1 - summ_words/orig_words)*100:.0f}% shorter)</span>
            </div>
            <div class="rc-text">{summary}</div>
        </div>
        """, unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown('<div class="footer-note">Dhurandhar 2 Review Intelligence Studio · Streamlit + Hugging Face Transformers</div>', unsafe_allow_html=True)
 
