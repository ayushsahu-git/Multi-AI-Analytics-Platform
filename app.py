# """
# app.py  —  Multi-AI Analytics Platform v2.1 (Clean Edition)
# Run:  streamlit run app.py

# Requirements (install before running):
#     pip install streamlit pandas numpy pillow plotly scikit-learn xgboost lightgbm
#     pip install opencv-python-headless transformers torch torchvision
#     pip install openai google-generativeai anthropic   # optional, for API providers
#     pip install spacy && python -m spacy download en_core_web_sm   # optional NER

# Place these files in the SAME folder as app.py:
#     generative_ai.py
#     nlp_module.py
#     ml_models.py        (your existing file — unchanged)
#     dl_module.py        (your existing file — unchanged)
#     data/data_loader.py
#     data/powerbi_export.py
#     config.py
#     utils/helpers.py
# """
# import os
# port = int(os.environ.get("PORT", 10000))

# import warnings
# warnings.filterwarnings("ignore")

# import os
# import sys
# import json
# from pathlib import Path
# from typing import Any, Optional

# import streamlit as st
# import pandas as pd
# import numpy as np
# from PIL import Image

# # ── Root path setup ───────────────────────────────────────────────────────────
# ROOT = Path(__file__).parent
# sys.path.insert(0, str(ROOT))

# # ── Core imports ──────────────────────────────────────────────────────────────
# try:
#     from config import Config, OUTPUT_DIR
# except ImportError:
#     OUTPUT_DIR = ROOT / "outputs"
#     OUTPUT_DIR.mkdir(exist_ok=True)

# from data.data_loader import DataLoader
# from data.powerbi_export import PowerBIExporter
# from models.ml_models import MLPipeline, XGBoostPipeline, EnsemblePipeline
# from models.generative_ai import GenerativeAI, OPENAI_OK, GOOGLE_OK, ANTHROPIC_OK
# from utils.helpers import (
#     create_feature_importance_chart, create_metrics_dashboard,
#     create_confusion_matrix, create_correlation_heatmap,
#     create_class_distribution, create_actual_vs_predicted,
# )

# try:
#     from models.ml_models import LightGBMPipeline
#     LGB_OK = True
# except Exception:
#     LightGBMPipeline = None
#     LGB_OK = False


# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Multi-AI Analytics Platform",
#     page_icon="⚡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # ══════════════════════════════════════════════════════════════════════════════
# #  Global CSS — Aurora Dark theme
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

# *, *::before, *::after { box-sizing: border-box; }
# html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

# .stApp {
#     background: #080b14;
#     background-image:
#         radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.18) 0%, transparent 60%),
#         radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20,184,166,0.14) 0%, transparent 55%),
#         radial-gradient(ellipse 50% 60% at 50% 50%, rgba(139,92,246,0.05) 0%, transparent 70%);
#     color: #e2e8f0;
# }

# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #0c0f1d 0%, #0e1120 60%, #0a0d18 100%) !important;
#     border-right: 1px solid rgba(99,102,241,0.2);
# }
# section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

# .hero-wrap {
#     background: linear-gradient(135deg,
#         rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 40%, rgba(20,184,166,0.1) 100%);
#     border: 1px solid rgba(99,102,241,0.25);
#     border-radius: 20px;
#     padding: 2.4rem 2rem 2rem;
#     margin-bottom: 1.8rem;
#     position: relative; overflow: hidden;
# }
# .hero-title {
#     font-size: 2.6rem; font-weight: 800; line-height: 1.2;
#     background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #2dd4bf 100%);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#     background-clip: text; margin-bottom: 0.5rem;
# }
# .hero-sub { font-size: 1rem; color: #94a3b8; letter-spacing: 0.06em; }
# .hero-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1.2rem; }
# .badge {
#     background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
#     color: #a78bfa; padding: 4px 14px; border-radius: 999px;
#     font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
#     font-family: 'JetBrains Mono', monospace;
# }
# .badge.teal { background: rgba(20,184,166,0.12); border-color: rgba(20,184,166,0.3); color: #2dd4bf; }
# .badge.blue { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.3); color: #60a5fa; }

# .mod-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.2rem 0; }
# .mod-card {
#     background: linear-gradient(135deg, rgba(15,18,35,0.9) 0%, rgba(20,24,42,0.9) 100%);
#     border: 1px solid rgba(99,102,241,0.2); border-radius: 16px;
#     padding: 1.4rem 1.2rem; transition: border-color 0.25s, transform 0.2s;
#     position: relative; overflow: hidden;
# }
# .mod-card:hover { border-color: rgba(139,92,246,0.55); transform: translateY(-2px); }
# .mod-card .icon { font-size: 2.2rem; margin-bottom: 0.7rem; }
# .mod-card .title { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem; }
# .mod-card .desc  { font-size: 0.82rem; color: #64748b; line-height: 1.5; }
# .mod-card .glow {
#     position: absolute; top: -40px; right: -40px;
#     width: 100px; height: 100px; border-radius: 50%;
#     background: radial-gradient(circle, var(--gc) 0%, transparent 70%); opacity: 0.35;
# }

# .stat-row { display: flex; gap: 1rem; margin: 1rem 0; }
# .stat-card {
#     flex: 1;
#     background: linear-gradient(135deg, rgba(15,18,35,0.95) 0%, rgba(20,24,45,0.95) 100%);
#     border: 1px solid rgba(99,102,241,0.18); border-radius: 14px;
#     padding: 1.1rem 1rem; text-align: center;
# }
# .stat-val { font-size: 1.7rem; font-weight: 700; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
# .stat-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.07em; }

# .sec-head {
#     font-size: 1.35rem; font-weight: 700;
#     background: linear-gradient(90deg, #a78bfa, #60a5fa);
#     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#     background-clip: text; margin: 1rem 0 0.6rem;
#     display: flex; align-items: center; gap: 0.5rem;
# }

# .info-box {
#     background: rgba(99,102,241,0.08); border-left: 4px solid #6366f1;
#     border-radius: 0 12px 12px 0; padding: 0.9rem 1.1rem; margin: 0.7rem 0;
#     color: #cbd5e1; line-height: 1.6;
# }
# .success-box {
#     background: rgba(20,184,166,0.08); border-left: 4px solid #14b8a6;
#     border-radius: 0 12px 12px 0; padding: 0.9rem 1.1rem; margin: 0.7rem 0;
#     color: #99f6e4;
# }
# .result-box {
#     background: linear-gradient(135deg, rgba(15,18,35,0.98) 0%, rgba(20,24,48,0.98) 100%);
#     border: 1px solid rgba(99,102,241,0.22); border-radius: 14px;
#     padding: 1.3rem 1.4rem; margin: 0.8rem 0; color: #e2e8f0; line-height: 1.75;
# }

# .stButton > button {
#     background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
#     color: white !important; border: none !important; border-radius: 10px !important;
#     font-weight: 600 !important; font-family: 'Outfit', sans-serif !important;
#     letter-spacing: 0.02em; transition: opacity 0.2s, transform 0.15s !important;
#     padding: 0.55rem 1.4rem !important;
# }
# .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

# .stTabs [data-baseweb="tab-list"] {
#     gap: 6px; background: rgba(8,11,20,0.6); border-radius: 12px; padding: 4px;
#     border: 1px solid rgba(99,102,241,0.15);
# }
# .stTabs [data-baseweb="tab"] {
#     border-radius: 9px; padding: 0.5rem 1.2rem; font-weight: 600; color: #64748b;
# }
# .stTabs [aria-selected="true"] {
#     background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3)) !important;
#     color: #a78bfa !important;
# }

# .stTextArea textarea, .stTextInput input, .stSelectbox select {
#     background: rgba(15,18,35,0.9) !important; border: 1px solid rgba(99,102,241,0.2) !important;
#     border-radius: 10px !important; color: #e2e8f0 !important;
# }
# .stDataFrame { border: 1px solid rgba(99,102,241,0.2) !important; border-radius: 12px !important; }
# [data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700; }
# [data-testid="stMetricLabel"] { color: #64748b !important; }
# [data-testid="stChatMessage"] {
#     background: rgba(15,18,35,0.7) !important;
#     border: 1px solid rgba(99,102,241,0.15) !important; border-radius: 14px !important;
# }
# .footer {
#     text-align: center; color: #334155; font-size: 0.78rem;
#     margin-top: 2.5rem; padding: 1rem;
#     border-top: 1px solid rgba(99,102,241,0.1); letter-spacing: 0.06em;
# }
# .footer span { color: #6366f1; font-family: 'JetBrains Mono', monospace; font-weight: 600; }
# .stSpinner > div { border-top-color: #6366f1 !important; }
# hr { border-color: rgba(99,102,241,0.15) !important; }
# ::-webkit-scrollbar { width: 6px; height: 6px; }
# ::-webkit-scrollbar-track { background: #080b14; }
# ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
# </style>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Sidebar
# # ══════════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("""
#     <div style="text-align:center;padding:1rem 0 0.5rem">
#         <div style="font-size:2.4rem">⚡</div>
#         <div style="font-size:1.1rem;font-weight:800;
#                     background:linear-gradient(135deg,#a78bfa,#2dd4bf);
#                     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
#             AI Platform
#         </div>
#         <div style="font-size:0.72rem;color:#475569;letter-spacing:0.08em;margin-top:2px">
#             v2.1 · CLEAN EDITION
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.divider()

#     st.markdown("### 🔑 Generative AI")
#     provider_choice = st.selectbox(
#         "Provider",
#         ["smart", "openai", "google", "anthropic"],
#         format_func=lambda x: {
#             "smart":     "⚡ Smart AI (Instant · No API Key)",
#             "openai":    "🟢 OpenAI GPT-4o",
#             "google":    "🔵 Google Gemini",
#             "anthropic": "🟣 Anthropic Claude",
#         }[x],
#     )
#     if provider_choice == "smart":
#         st.caption("✅ Instant responses — no API key, no downloads")
#         api_key_input = ""
#     else:
#         api_key_input = st.text_input("API Key", type="password", placeholder="Paste key here…")
#     st.divider()

#     st.markdown("### ⚙️ System Status")
#     for lib, label, emoji in [
#         ("torch",       "PyTorch",      "🔥"),
#         ("sklearn",     "sklearn",      "🤖"),
#         ("xgboost",     "XGBoost",      "⚡"),
#         ("transformers","Transformers", "🤗"),
#         ("cv2",         "OpenCV",       "📷"),
#         ("lightgbm",    "LightGBM",     "🌿"),
#     ]:
#         try:
#             mod = __import__(lib)
#             ver = getattr(mod, "__version__", "✓")
#             st.markdown(f"{emoji} {label} `{ver}`")
#         except ImportError:
#             st.markdown(f"{emoji} {label} ❌")
#     st.divider()

#     if st.button("🔄 Reset Session", use_container_width=True):
#         for k in list(st.session_state.keys()):
#             del st.session_state[k]
#         st.rerun()

#     st.markdown("""
#     <div style="text-align:center;margin-top:1rem;color:#334155;font-size:0.7rem">
#         Multi-AI Analytics Platform<br>
#         <span style="color:#6366f1;font-family:'JetBrains Mono',monospace">KYOTO-Z</span>
#     </div>""", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Session state initialisation
# # ══════════════════════════════════════════════════════════════════════════════
# def _init():
#     defaults = {
#         "data_loader":    DataLoader(),
#         "powerbi_exp":    PowerBIExporter(OUTPUT_DIR),
#         "df":             None,
#         "data_summary":   None,
#         "ml_pipeline":    None,
#         "ml_results":     None,
#         "ml_metrics":     None,
#         "gen_ai":         GenerativeAI(provider="smart"),
#         "uploaded_image": None,
#         "target_column":  None,
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# _init()

# # Always rebuild gen_ai when provider/key change
# st.session_state.gen_ai = GenerativeAI(api_key=api_key_input, provider=provider_choice)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Hero Banner
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div class="hero-wrap">
#   <div class="hero-title">⚡ Multi-AI Analytics Platform</div>
#   <div class="hero-sub">Machine Learning &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp; NLP &nbsp;·&nbsp; Generative AI &nbsp;·&nbsp; Power BI Export</div>
#   <div class="hero-badges">
#     <span class="badge">scikit-learn</span>
#     <span class="badge">XGBoost</span>
#     <span class="badge teal">HuggingFace</span>
#     <span class="badge blue">OpenCV</span>
#     <span class="badge">PyTorch / TF</span>
#     <span class="badge teal">Plotly</span>
#     <span class="badge blue">Power BI</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Main tabs
# # ══════════════════════════════════════════════════════════════════════════════
# tab_home, tab_data, tab_ml, tab_dl, tab_nlp, tab_genai, tab_pbi = st.tabs([
#     "🏠 Home", "📊 Data", "🤖 ML Pipeline",
#     "🧠 Deep Learning", "📝 NLP", "💡 Generative AI", "📤 Power BI",
# ])


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 0 · HOME
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_home:
#     st.markdown("""
#     <div class="mod-grid">
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(99,102,241,0.5)"></div>
#         <div class="icon">🤖</div>
#         <div class="title">ML Pipeline</div>
#         <div class="desc">9+ models — Random Forest, XGBoost, LightGBM, Ensemble. Full metrics, ROC, feature importance.</div>
#       </div>
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(20,184,166,0.5)"></div>
#         <div class="icon">📊</div>
#         <div class="title">Data Explorer</div>
#         <div class="desc">Upload CSV/Excel/JSON. Auto EDA, correlation heatmaps, distributions, scatter builder.</div>
#       </div>
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(139,92,246,0.5)"></div>
#         <div class="icon">🧠</div>
#         <div class="title">Deep Learning</div>
#         <div class="desc">MobileNetV2, ResNet50, VGG16 classification. Grad-CAM, face/edge detection, image filters.</div>
#       </div>
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(59,130,246,0.5)"></div>
#         <div class="icon">📝</div>
#         <div class="title">NLP Suite</div>
#         <div class="desc">Sentiment analysis, NER, zero-shot classification, summarization — all via HuggingFace.</div>
#       </div>
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(236,72,153,0.5)"></div>
#         <div class="icon">💡</div>
#         <div class="title">Generative AI</div>
#         <div class="desc">GPT-4 · Gemini · Claude. Context-aware chatbot, Q&A, code gen, image generation, reports.</div>
#       </div>
#       <div class="mod-card">
#         <div class="glow" style="--gc:rgba(245,158,11,0.5)"></div>
#         <div class="icon">📤</div>
#         <div class="title">Power BI Export</div>
#         <div class="desc">Export datasets, feature importance tables, and predictions as CSV/Parquet for Power BI.</div>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.markdown('<div class="stat-card"><div class="stat-val">9+</div><div class="stat-lbl">ML Models</div></div>', unsafe_allow_html=True)
#     c2.markdown('<div class="stat-card"><div class="stat-val">4</div><div class="stat-lbl">CNN Backbones</div></div>', unsafe_allow_html=True)
#     c3.markdown('<div class="stat-card"><div class="stat-val">5</div><div class="stat-lbl">NLP Tasks</div></div>', unsafe_allow_html=True)
#     c4.markdown('<div class="stat-card"><div class="stat-val">3</div><div class="stat-lbl">Gen AI Providers</div></div>', unsafe_allow_html=True)

#     st.markdown("""
#     <div class="info-box">
#     💡 <strong>Quick Start:</strong> Head to <em>📊 Data</em> to upload your dataset,
#     then use <em>🤖 ML Pipeline</em> to train models, or jump to <em>🧠 Deep Learning</em>
#     for image analysis. Use <em>💡 Generative AI</em> for automated insights.
#     </div>""", unsafe_allow_html=True)


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 1 · DATA
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_data:
#     st.markdown('<div class="sec-head">📊 Data Loading & Exploration</div>', unsafe_allow_html=True)

#     col_up, col_sum = st.columns([2, 1])
#     with col_up:
#         uploaded_file = st.file_uploader(
#             "Upload CSV, Excel, JSON or Image",
#             type=["csv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "bmp", "webp"],
#         )
#         if uploaded_file:
#             try:
#                 if uploaded_file.type.startswith("image"):
#                     img = Image.open(uploaded_file).convert("RGB")
#                     st.image(img, caption=uploaded_file.name, width=420)
#                     st.session_state.uploaded_image = img
#                 else:
#                     ext = Path(uploaded_file.name).suffix.lower()
#                     if ext == ".csv":
#                         df = pd.read_csv(uploaded_file)
#                     elif ext in [".xlsx", ".xls"]:
#                         df = pd.read_excel(uploaded_file)
#                     elif ext == ".json":
#                         df = pd.read_json(uploaded_file)
#                     else:
#                         df = pd.read_csv(uploaded_file)

#                     st.session_state.df = df
#                     st.session_state.data_summary = st.session_state.data_loader.get_data_summary(df)
#                     st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
#             except Exception as e:
#                 st.error(f"❌ {e}")

#     if st.session_state.df is not None:
#         df = st.session_state.df
#         with col_sum:
#             st.markdown(f'<div class="stat-card" style="margin-bottom:8px"><div class="stat-val">{df.shape[0]:,}</div><div class="stat-lbl">Rows</div></div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="stat-card" style="margin-bottom:8px"><div class="stat-val">{df.shape[1]}</div><div class="stat-lbl">Columns</div></div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="stat-card"><div class="stat-val">{df.isnull().sum().sum()}</div><div class="stat-lbl">Missing</div></div>', unsafe_allow_html=True)

#         dtabs = st.tabs(["🔍 Preview", "📈 Statistics", "📊 Charts", "🌡️ Correlations"])

#         with dtabs[0]:
#             st.dataframe(df.head(50), use_container_width=True)

#         with dtabs[1]:
#             st.dataframe(df.describe(include="all").T, use_container_width=True)
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.markdown("**Data Types**")
#                 st.dataframe(df.dtypes.rename("Type").reset_index().rename(columns={"index": "Column"}), use_container_width=True)
#             with col_b:
#                 st.markdown("**Missing Values**")
#                 miss = df.isnull().sum().reset_index()
#                 miss.columns = ["Column", "Missing"]
#                 miss["Pct"] = (miss["Missing"] / len(df) * 100).round(2)
#                 st.dataframe(miss[miss["Missing"] > 0], use_container_width=True)

#         with dtabs[2]:
#             import plotly.express as px
#             num_cols = df.select_dtypes(include=np.number).columns.tolist()
#             cat_cols = df.select_dtypes(include="object").columns.tolist()
#             chart_type = st.selectbox("Chart Type", ["Histogram", "Bar Chart", "Scatter", "Box Plot", "Line"])

#             if chart_type == "Histogram" and num_cols:
#                 col = st.selectbox("Column", num_cols)
#                 fig = px.histogram(df, x=col, template="plotly_dark", color_discrete_sequence=["#6366f1"])
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,18,35,0.6)")
#                 st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Bar Chart" and cat_cols and num_cols:
#                 xc = st.selectbox("X (categorical)", cat_cols)
#                 yc = st.selectbox("Y (numeric)", num_cols)
#                 fig = px.bar(df.groupby(xc)[yc].mean().reset_index(), x=xc, y=yc,
#                              template="plotly_dark", color=yc, color_continuous_scale="Viridis")
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Scatter" and len(num_cols) >= 2:
#                 xc = st.selectbox("X", num_cols, key="sc_x")
#                 yc = st.selectbox("Y", num_cols, index=1, key="sc_y")
#                 cc = st.selectbox("Color", ["None"] + cat_cols)
#                 fig = px.scatter(df, x=xc, y=yc, color=None if cc == "None" else cc,
#                                  template="plotly_dark", opacity=0.7)
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Box Plot" and num_cols:
#                 yc = st.selectbox("Value", num_cols, key="bp_y")
#                 gc = st.selectbox("Group", ["None"] + cat_cols)
#                 fig = px.box(df, y=yc, x=None if gc == "None" else gc,
#                              template="plotly_dark", color_discrete_sequence=["#8b5cf6"])
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Line" and num_cols:
#                 yc = st.selectbox("Column", num_cols, key="lc_y")
#                 fig = px.line(df.reset_index(), x="index", y=yc,
#                               template="plotly_dark", color_discrete_sequence=["#2dd4bf"])
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig, use_container_width=True)

#         with dtabs[3]:
#             num_cols = df.select_dtypes(include=np.number).columns.tolist()
#             cat_cols = df.select_dtypes(include="object").columns.tolist()
#             if len(num_cols) >= 2:
#                 fig = create_correlation_heatmap(df)
#                 st.plotly_chart(fig, use_container_width=True)
#                 if cat_cols:
#                     tc = st.selectbox("Class column for distribution", cat_cols)
#                     fig2 = create_class_distribution(df[tc], f"{tc} Distribution")
#                     st.plotly_chart(fig2, use_container_width=True)
#             else:
#                 st.info("Need at least 2 numeric columns for correlation.")


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 2 · ML PIPELINE
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_ml:
#     st.markdown('<div class="sec-head">🤖 Machine Learning Pipeline</div>', unsafe_allow_html=True)

#     if st.session_state.df is None:
#         st.markdown('<div class="info-box">👆 Load a dataset in the <strong>📊 Data</strong> tab first.</div>', unsafe_allow_html=True)
#     else:
#         df = st.session_state.df

#         mc1, mc2, mc3 = st.columns(3)
#         with mc1:
#             target_col = st.selectbox("🎯 Target Column", df.columns.tolist())
#             st.session_state.target_column = target_col
#         with mc2:
#             from data.data_loader import DataLoader as _DL
#             task_type = _DL().detect_task_type(df[target_col])
#             st.markdown(f"**Detected Task:** `{task_type}`")
#             model_options = {
#                 "classification": ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "XGBoost", "LightGBM", "Ensemble"],
#                 "regression":     ["Random Forest", "Gradient Boosting", "Ridge Regression", "Lasso Regression", "SVM", "XGBoost", "LightGBM", "Ensemble"],
#             }
#             model_name = st.selectbox("🧠 Algorithm", model_options[task_type])
#         with mc3:
#             test_size = st.slider("Test Split %", 10, 40, 20) / 100
#             cv_folds  = st.slider("CV Folds", 2, 10, 5)

#         if st.button("🚀 Train Model", type="primary", use_container_width=True):
#             with st.spinner("Training…"):
#                 try:
#                     if model_name == "XGBoost":
#                         pipe = XGBoostPipeline(task_type=task_type)
#                     elif model_name == "LightGBM" and LGB_OK and LightGBMPipeline is not None:
#                         pipe = LightGBMPipeline(task_type=task_type)
#                     elif model_name == "Ensemble":
#                         pipe = EnsemblePipeline(task_type=task_type)
#                     else:
#                         pipe = MLPipeline(task_type=task_type, model_name=model_name)

#                     result = pipe.preprocess(df, target_col=target_col)
#                     if result is None:
#                         st.error("❌ Preprocessing returned None. Check your target column.")
#                         st.stop()
#                     X, y = result
#                     if y is None:
#                         st.error("❌ Target column could not be extracted.")
#                         st.stop()
#                     metrics = pipe.train(X, y, test_size=test_size)

#                     st.session_state.ml_pipeline = pipe
#                     st.session_state.ml_metrics  = metrics
#                     st.session_state.ml_results  = {
#                         "model_name": model_name,
#                         "task_type":  task_type,
#                         "feature_importance": pipe.get_feature_importance().to_dict("records")
#                             if hasattr(pipe, "get_feature_importance") else [],
#                     }
#                     st.success(f"✅ **{model_name}** trained successfully!")
#                 except Exception as e:
#                     st.error(f"❌ Training failed: {e}")

#         if st.session_state.ml_metrics:
#             metrics = st.session_state.ml_metrics
#             st.markdown("---")
#             st.markdown('<div class="sec-head">📊 Results</div>', unsafe_allow_html=True)

#             num_m = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
#             cols_m = st.columns(min(len(num_m), 4))
#             for i, (k, v) in enumerate(list(num_m.items())[:4]):
#                 cols_m[i].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

#             fig_dash = create_metrics_dashboard(metrics)
#             st.plotly_chart(fig_dash, use_container_width=True)

#             pipe = st.session_state.ml_pipeline
#             if pipe and pipe.is_fitted:
#                 result_tabs = st.tabs(["📉 Confusion Matrix / Scatter", "📈 Feature Importance", "📋 Report"])

#                 with result_tabs[0]:
#                     task_type = st.session_state.ml_results["task_type"]
#                     if task_type == "classification" and pipe.y_pred is not None:
#                         labels = [str(c) for c in pipe.classes_] if pipe.classes_ is not None else None
#                         fig_cm = create_confusion_matrix(
#                             list(pipe.y_test),  # type: ignore[arg-type]
#                             list(pipe.y_pred),  # type: ignore[arg-type]
#                             labels,
#                         )
#                         st.plotly_chart(fig_cm, use_container_width=True)
#                     elif task_type == "regression" and pipe.y_pred is not None:
#                         fig_avp = create_actual_vs_predicted(
#                             pipe.y_test, pipe.y_pred,
#                             f"{st.session_state.ml_results['model_name']} — Actual vs Predicted"
#                         )
#                         st.plotly_chart(fig_avp, use_container_width=True)

#                 with result_tabs[1]:
#                     if hasattr(pipe, "get_feature_importance"):
#                         fi_df = pipe.get_feature_importance()
#                         if not fi_df.empty:
#                             fig_fi = create_feature_importance_chart(fi_df, top_n=20)
#                             st.plotly_chart(fig_fi, use_container_width=True)
#                             st.dataframe(fi_df.head(20), use_container_width=True)

#                 with result_tabs[2]:
#                     if "classification_report" in metrics:
#                         st.code(metrics["classification_report"], language="text")
#                     else:
#                         for k, v in num_m.items():
#                             st.markdown(f"**{k.replace('_', ' ').title()}:** `{v:.4f}`")


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 3 · DEEP LEARNING
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_dl:
#     st.markdown('<div class="sec-head">🧠 Deep Learning & Computer Vision</div>', unsafe_allow_html=True)

#     from models.dl_module import (
#         _classify_image_tf, _classify_image_torch,
#         detect_edges_opencv, detect_faces_opencv, apply_image_filters,
#     )

#     dl_uploaded = st.file_uploader("📷 Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="dl_up")

#     if dl_uploaded is None:
#         st.markdown('<div class="info-box">👆 Upload any image — try animals, faces, objects, or landscapes.</div>', unsafe_allow_html=True)
#     else:
#         pil_img = Image.open(dl_uploaded)
#         c1, c2 = st.columns([2, 1])
#         with c1:
#             st.image(pil_img, caption="Uploaded Image", use_column_width=True)
#         with c2:
#             w, h = pil_img.size
#             arr_np = np.array(pil_img.convert("RGB"))
#             for lbl, val in [
#                 ("Resolution", f"{w}×{h}"),
#                 ("Mode",       pil_img.mode),
#                 ("Brightness", f"{arr_np.mean():.1f}"),
#                 ("File size",  f"{dl_uploaded.size / 1024:.1f} KB"),
#             ]:
#                 st.markdown(
#                     f'<div class="stat-card" style="margin-bottom:6px">'
#                     f'<div class="stat-val" style="font-size:1rem">{val}</div>'
#                     f'<div class="stat-lbl">{lbl}</div></div>',
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("---")
#         dl_tabs = st.tabs(["🏷️ Classification", "🔥 Grad-CAM", "👁️ Detection", "🎨 Filters"])

#         with dl_tabs[0]:
#             st.subheader("Image Classification — ImageNet 1K")
#             backend = st.radio("Backend", ["TensorFlow/Keras", "PyTorch"], horizontal=True)
#             if backend == "TensorFlow/Keras":
#                 model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50", "VGG16"])
#             else:
#                 model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50"])

#             if st.button("🔍 Classify Image", type="primary", key="cls_btn"):
#                 with st.spinner(f"Running {model_choice}…"):
#                     try:
#                         # Verify TF is importable before calling classify
#                         if backend == "TensorFlow/Keras":
#                             try:
#                                 import tensorflow as _tf_check  # type: ignore[import-untyped]
#                             except ImportError:
#                                 raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
#                             results = _classify_image_tf(pil_img, model_choice)
#                         else:
#                             results = _classify_image_torch(pil_img, model_choice)

#                         import plotly.graph_objects as go
#                         st.markdown(
#                             f'<div class="success-box">🏆 <strong>{results[0]["Label"]}</strong> — {results[0]["Confidence"]}</div>',
#                             unsafe_allow_html=True,
#                         )
#                         st.dataframe(pd.DataFrame(results), use_container_width=True)

#                         labels_list = [r["Label"][:28] for r in results]
#                         scores_list = [r["Score"] for r in results]
#                         colors_list = ["#a78bfa" if i == 0 else "#334155" for i in range(len(scores_list))]
#                         fig = go.Figure(go.Bar(
#                             x=scores_list[::-1], y=labels_list[::-1], orientation="h",
#                             marker=dict(color=colors_list[::-1]),
#                             text=[f"{s * 100:.1f}%" for s in scores_list[::-1]],
#                             textposition="outside",
#                         ))
#                         fig.update_layout(title="Top-5 Confidence Scores", template="plotly_dark",
#                                           paper_bgcolor="rgba(0,0,0,0)", height=280)
#                         st.plotly_chart(fig, use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Classification failed: {e}")
#                         st.info("Make sure TensorFlow or PyTorch is installed.")

#         with dl_tabs[1]:
#             st.subheader("Grad-CAM — Class Activation Heatmap")
#             st.markdown("Highlights **which image regions** the model focused on for its prediction.")

#             _tf_ok, _pt_ok = False, False
#             try:
#                 import tensorflow; _tf_ok = True
#             except ImportError:
#                 pass
#             try:
#                 import torch; _pt_ok = True
#             except ImportError:
#                 pass

#             if not _tf_ok and not _pt_ok:
#                 st.markdown("""
#                 <div class="info-box">
#                 ⚠️ Grad-CAM requires TensorFlow or PyTorch.<br>
#                 Install: <code>pip install tensorflow</code> or <code>pip install torch torchvision</code>
#                 </div>""", unsafe_allow_html=True)
#             else:
#                 backend_opts = (["TensorFlow/Keras"] if _tf_ok else []) + (["PyTorch"] if _pt_ok else [])
#                 gc_backend = st.radio("Backend", backend_opts, horizontal=True, key="gc_back")
#                 gc_model   = st.selectbox(
#                     "Model",
#                     ["MobileNetV2", "ResNet50"] if gc_backend == "PyTorch" else ["MobileNetV2", "ResNet50", "VGG16"],
#                     key="gc_m",
#                 )

#                 if st.button("🔥 Generate Grad-CAM", type="primary"):
#                     with st.spinner("Computing Grad-CAM…"):
#                         try:
#                             import cv2 as _cv
#                             import numpy as _np

#                             orig_224 = _np.array(pil_img.convert("RGB").resize((224, 224)))

#                             if gc_backend == "TensorFlow/Keras":
#                                 # Robust TensorFlow import — handles tf2, tf-cpu, standalone keras
#                                 try:
#                                     import tensorflow as tf  # type: ignore[import-untyped]
#                                 except ImportError:
#                                     st.error("❌ TensorFlow not installed. Run: pip install tensorflow")
#                                     st.stop()
#                                 try:
#                                     from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore[import-untyped]
#                                 except ImportError:
#                                     try:
#                                         from tensorflow.keras.utils import img_to_array  # type: ignore[import-untyped]
#                                     except ImportError:
#                                         from keras.utils import img_to_array  # type: ignore[import-untyped]
#                                 try:
#                                     from models.dl_module import _load_tf_model
#                                 except ImportError:
#                                     from dl_module import _load_tf_model  # type: ignore[import]

#                                 model, preprocess, decode, (ih, iw) = _load_tf_model(gc_model)
#                                 arr = preprocess(_np.expand_dims(img_to_array(
#                                     pil_img.convert("RGB").resize((iw, ih))), 0))

#                                 layer_name = None
#                                 try:
#                                     _conv2d_cls = tf.keras.layers.Conv2D  # type: ignore[attr-defined]
#                                 except AttributeError:
#                                     import keras
#                                     _conv2d_cls = keras.layers.Conv2D  # type: ignore[attr-defined]
#                                 for layer in reversed(model.layers):
#                                     if isinstance(layer, _conv2d_cls):
#                                         layer_name = layer.name
#                                         break

#                                 if not layer_name:
#                                     st.warning("No Conv2D layer found in model.")
#                                     st.stop()

#                                 grad_model = tf.keras.models.Model(
#                                     inputs=model.inputs,
#                                     outputs=[model.get_layer(layer_name).output, model.output],
#                                 )
#                                 with tf.GradientTape() as tape:
#                                     conv_out, preds = grad_model(arr)
#                                     top_cls = int(tf.argmax(preds[0]).numpy())  # type: ignore[union-attr]
#                                     loss = preds[:, top_cls]  # type: ignore[index]
#                                 grads   = tape.gradient(loss, conv_out)
#                                 pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
#                                 heatmap = conv_out[0] @ pooled[..., tf.newaxis]
#                                 heatmap = tf.squeeze(heatmap).numpy()
#                                 top_label = decode(model.predict(arr, verbose=0), top=1)[0][0][1].replace("_", " ").title()

#                             else:  # PyTorch
#                                 import torch
#                                 import torchvision.models as M
#                                 import torchvision.transforms as T
#                                 import json, urllib.request

#                                 pt_map = {"MobileNetV2": M.mobilenet_v2, "ResNet50": M.resnet50}
#                                 pt_model = pt_map.get(gc_model, M.mobilenet_v2)(weights="DEFAULT")
#                                 pt_model.eval()

#                                 last_conv = None
#                                 for name, m in pt_model.named_modules():
#                                     if isinstance(m, torch.nn.Conv2d):
#                                         last_conv = (name, m)

#                                 if last_conv is None:
#                                     st.warning("No Conv2d layer found.")
#                                     st.stop()

#                                 activations, gradients = [], []

#                                 def _fwd_hook(mod, inp, out):
#                                     activations.clear(); activations.append(out.detach())

#                                 def _bwd_hook(mod, grad_in, grad_out):
#                                     gradients.clear(); gradients.append(grad_out[0].detach())

#                                 h1 = last_conv[1].register_forward_hook(_fwd_hook)
#                                 h2 = last_conv[1].register_full_backward_hook(_bwd_hook)

#                                 _transform = T.Compose([
#                                     T.Resize(256), T.CenterCrop(224), T.ToTensor(),
#                                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                 ])
#                                 _tf_raw = _transform(pil_img.convert("RGB"))
#                                 tf_img = _tf_raw.unsqueeze(0)  # torch Tensor — ignore PIL Image complaint  # type: ignore[union-attr]
#                                 tf_img.requires_grad_(True)

#                                 output = pt_model(tf_img)
#                                 top_cls_pt = output.argmax(dim=1).item()
#                                 pt_model.zero_grad()
#                                 output[0, top_cls_pt].backward()
#                                 h1.remove(); h2.remove()

#                                 act     = activations[0].squeeze()
#                                 grad    = gradients[0].squeeze()
#                                 weights = grad.mean(dim=(1, 2))
#                                 heatmap = (weights[:, None, None] * act).sum(dim=0).numpy()
#                                 heatmap = _np.maximum(heatmap, 0)

#                                 try:
#                                     url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
#                                     with urllib.request.urlopen(url, timeout=5) as r:
#                                         class_labels = json.load(r)
#                                     top_label = class_labels[top_cls_pt].replace("_", " ").title()
#                                 except Exception:
#                                     top_label = f"Class {top_cls_pt}"

#                             # ── Render heatmap ──
#                             heatmap = heatmap / (heatmap.max() + 1e-8)
#                             h_res   = _cv.resize(heatmap, (224, 224))
#                             _h_uint8 = _np.array(255 * h_res, dtype=_np.uint8)
#                             h_col   = _cv.cvtColor(
#                                 _cv.applyColorMap(_h_uint8, _cv.COLORMAP_JET),  # type: ignore[call-overload]
#                                 _cv.COLOR_BGR2RGB,
#                             )
#                             overlay = _np.array(orig_224 * 0.6 + h_col * 0.4, dtype=_np.uint8)

#                             gc1, gc2, gc3 = st.columns(3)
#                             gc1.image(orig_224, caption="Original",  use_column_width=True)
#                             gc2.image(h_col,    caption="Heatmap",   use_column_width=True)
#                             gc3.image(overlay,  caption="Overlay",   use_column_width=True)
#                             st.markdown(
#                                 f'<div class="success-box">🏆 Top prediction: <strong>{top_label}</strong> '
#                                 f'— red/yellow regions = highest model attention</div>',
#                                 unsafe_allow_html=True,
#                             )

#                         except Exception as e:
#                             st.error(f"Grad-CAM failed: {e}")

#         with dl_tabs[2]:
#             st.subheader("OpenCV Detection")
#             cv_task = st.selectbox("Task", ["Face Detection", "Edge Detection"])
#             t1 = t2 = None
#             if cv_task == "Edge Detection":
#                 t1 = st.slider("Threshold 1", 10, 200, 50)
#                 t2 = st.slider("Threshold 2", 50, 400, 150)

#             if st.button("▶ Run Detection", type="primary", key="det_btn"):
#                 with st.spinner("Running OpenCV…"):
#                     if cv_task == "Edge Detection":
#                         import cv2 as _cv
#                         _t1: float = float(t1) if t1 is not None else 50.0
#                         _t2: float = float(t2) if t2 is not None else 150.0
#                         gray  = _cv.cvtColor(np.array(pil_img.convert("RGB")), _cv.COLOR_RGB2GRAY)
#                         edges = _cv.Canny(_cv.GaussianBlur(gray, (5, 5), 0), _t1, _t2)
#                         dc1, dc2 = st.columns(2)
#                         dc1.image(pil_img, caption="Original",       use_column_width=True)
#                         dc2.image(edges,   caption="Edges",           use_column_width=True, clamp=True)
#                         st.info(f"Edge pixels: **{np.sum(edges > 0):,}**")
#                     else:
#                         result_img, face_count = detect_faces_opencv(pil_img)
#                         dc1, dc2 = st.columns(2)
#                         dc1.image(pil_img,    caption="Original",    use_column_width=True)
#                         dc2.image(result_img, caption="Detections",  use_column_width=True)
#                         if face_count > 0:
#                             st.markdown(f'<div class="success-box">✅ Detected <strong>{face_count}</strong> face(s).</div>', unsafe_allow_html=True)
#                         else:
#                             st.warning("No faces detected. Try a clear frontal portrait.")

#         with dl_tabs[3]:
#             st.subheader("Image Filters Gallery")
#             if st.button("🎨 Apply All Filters", type="primary", key="flt_btn"):
#                 with st.spinner("Applying filters…"):
#                     filters  = apply_image_filters(pil_img)
#                     cols_f   = st.columns(3)
#                     for i, (name, img) in enumerate(filters.items()):
#                         with cols_f[i % 3]:
#                             st.image(img, caption=name, use_column_width=True, clamp=True)


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 4 · NLP
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_nlp:
#     st.markdown('<div class="sec-head">📝 NLP Suite</div>', unsafe_allow_html=True)

#     try:
#         from models.nlp_module import (  # type: ignore[import]
#             run_sentiment, run_ner, run_text_classification,
#             run_summarization, chat_with_model,
#         )
#     except ImportError:
#         from nlp_module import (  # type: ignore[import]
#             run_sentiment, run_ner, run_text_classification,
#             run_summarization, chat_with_model,
#         )

#     nlp_tabs = st.tabs(["😊 Sentiment", "🏷️ NER", "📂 Classification", "📰 Summarization", "💬 Chatbot"])

#     # ── Sentiment ──
#     with nlp_tabs[0]:
#         st.subheader("Sentiment Analysis — DistilBERT")
#         mode = st.radio("Mode", ["Single", "Batch"], horizontal=True)
#         if mode == "Single":
#             txt = st.text_area("Text to analyze:", height=110,
#                                placeholder="The product quality is amazing and delivery was super fast!")
#             if st.button("🔍 Analyze", type="primary", key="sa_btn"):
#                 if not txt.strip():
#                     st.warning("Enter some text.")
#                 else:
#                     with st.spinner("Loading DistilBERT and analyzing…"):
#                         r = run_sentiment([txt])
#                     if r:
#                         color = "#22c55e" if r[0]["Sentiment"] == "POSITIVE" else "#ef4444"
#                         icon  = "😊" if r[0]["Sentiment"] == "POSITIVE" else "😞"
#                         st.markdown(f"""
#                         <div class="result-box" style="border-left:5px solid {color}">
#                             <div style="font-size:2rem">{icon}</div>
#                             <div style="font-size:1.4rem;font-weight:700;color:{color}">{r[0]["Sentiment"]}</div>
#                             <div style="color:#94a3b8;margin-top:6px">Confidence: <strong style="color:#e2e8f0">{r[0]["Confidence"]}</strong></div>
#                         </div>""", unsafe_allow_html=True)
#         else:
#             batch = st.text_area("One sentence per line:", height=180,
#                                  placeholder="Great product!\nTerrible experience.\nIt was okay.")
#             if st.button("🔍 Analyze All", type="primary", key="sa_batch"):
#                 lines = [ln.strip() for ln in batch.split("\n") if ln.strip()]
#                 if lines:
#                     with st.spinner("Analyzing…"):
#                         results = run_sentiment(lines)
#                     df_s = pd.DataFrame(results)
#                     st.dataframe(df_s, use_container_width=True)
#                     import plotly.express as px
#                     pos = sum(1 for r in results if r["Sentiment"] == "POSITIVE")
#                     fig = px.pie(
#                         values=[pos, len(results) - pos],
#                         names=["Positive", "Negative"],
#                         color_discrete_sequence=["#22c55e", "#ef4444"],
#                         hole=0.4, template="plotly_dark",
#                     )
#                     fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                     st.plotly_chart(fig, use_container_width=True)

#     # ── NER ──
#     with nlp_tabs[1]:
#         st.subheader("Named Entity Recognition")
#         ner_txt = st.text_area("Text for NER:", height=130,
#             value="Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")
#         if st.button("🏷️ Extract Entities", type="primary", key="ner_btn"):
#             with st.spinner("Running NER…"):
#                 ents = run_ner(ner_txt)
#             if ents:
#                 df_ner = pd.DataFrame(ents)
#                 st.dataframe(df_ner, use_container_width=True)
#                 import plotly.express as px
#                 vc = df_ner["Type"].value_counts().reset_index()
#                 vc.columns = ["Type", "count"]
#                 fig = px.bar(vc, x="Type", y="count", template="plotly_dark",
#                              color="Type", color_discrete_sequence=["#a78bfa", "#60a5fa", "#34d399", "#f472b6"])
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
#                 st.plotly_chart(fig, use_container_width=True)
#                 type_icons = {"PER": "🧑", "ORG": "🏢", "LOC": "📍", "MISC": "🔖", "GPE": "🌍"}
#                 for _, row in df_ner.iterrows():
#                     st.markdown(f"- **{row['Entity']}** → {type_icons.get(row['Type'], '')} `{row['Type']}` ({row['Score']})")
#             else:
#                 st.info("No entities found.")

#     # ── Zero-Shot Classification ──
#     with nlp_tabs[2]:
#         st.subheader("Zero-Shot Text Classification")
#         cl_txt = st.text_area("Text to classify:", height=110,
#             value="The new iPhone features an upgraded camera and faster processor.")
#         cl_labels = st.text_input("Candidate labels (comma-separated):",
#             value="technology, sports, politics, business, health, entertainment")
#         if st.button("📂 Classify", type="primary", key="zs_btn"):
#             lbls = [lb.strip() for lb in cl_labels.split(",") if lb.strip()]
#             if cl_txt.strip() and lbls:
#                 with st.spinner("Running zero-shot classification…"):
#                     results = run_text_classification(cl_txt, lbls)
#                 st.markdown(
#                     f'<div class="success-box">🏆 Best: <strong>{results[0]["Label"]}</strong> ({results[0]["Confidence"]})</div>',
#                     unsafe_allow_html=True,
#                 )
#                 import plotly.express as px
#                 df_zs = pd.DataFrame(results)
#                 fig = px.bar(df_zs, x="Label", y="Score", template="plotly_dark",
#                              color="Score", color_continuous_scale="Purples")
#                 fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig, use_container_width=True)

#     # ── Summarization ──
#     with nlp_tabs[3]:
#         st.subheader("Text Summarization — DistilBART")
#         long_txt = st.text_area("Long text to summarize:", height=220,
#             value=(
#                 "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
#                 "the natural intelligence displayed by animals including humans. AI research has been defined "
#                 "as the field of study of intelligent agents, which refers to any system that perceives its "
#                 "environment and takes actions that maximize its chance of achieving its goals. AI applications "
#                 "include advanced web search engines, recommendation systems, understanding human speech, "
#                 "self-driving cars, generative or creative tools, automated decision-making, and competing "
#                 "at the highest level in strategic game systems."
#             ))
#         if st.button("📰 Summarize", type="primary", key="sum_btn"):
#             if len(long_txt.split()) < 30:
#                 st.warning("Need at least 30 words.")
#             else:
#                 with st.spinner("Loading DistilBART and summarizing…"):
#                     summary = run_summarization(long_txt)
#                 st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)
#                 sc1, sc2 = st.columns(2)
#                 sc1.metric("Original Words", len(long_txt.split()))
#                 sc2.metric("Summary Words",  len(summary.split()))

#     # ── Chatbot ──
#     with nlp_tabs[4]:
#         st.subheader("💬 AI Chatbot")
#         if "chat_pairs" not in st.session_state:
#             st.session_state.chat_pairs = []

#         with st.expander("⚙️ Settings"):
#             sys_hint = st.text_input("System hint:", value="You are a helpful AI assistant. Be concise.")
#             if st.button("🗑️ Clear Chat"):
#                 st.session_state.chat_pairs = []
#                 st.rerun()

#         for um, bm in st.session_state.chat_pairs:
#             with st.chat_message("user"):
#                 st.markdown(um)
#             with st.chat_message("assistant"):
#                 st.markdown(bm)

#         user_input = st.chat_input("Ask anything…")
#         if user_input:
#             with st.chat_message("user"):
#                 st.markdown(user_input)
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking…"):
#                     try:
#                         prompt = f"{sys_hint}\n\n{user_input}" if sys_hint else user_input
#                         resp   = chat_with_model(prompt, st.session_state.chat_pairs)
#                     except Exception as e:
#                         resp = f"⚠️ {e}"
#                 st.markdown(resp)
#             st.session_state.chat_pairs.append((user_input, resp))

#         if not st.session_state.chat_pairs:
#             examples = [
#                 "What is machine learning?", "Explain neural networks simply.",
#                 "Top 3 AI programming languages?", "Benefits of deep learning?",
#             ]
#             ecols = st.columns(2)
#             for i, ex in enumerate(examples):
#                 with ecols[i % 2]:
#                     if st.button(ex, key=f"ex_{i}", use_container_width=True):
#                         with st.spinner("Thinking…"):
#                             try:
#                                 resp = chat_with_model(ex, [])
#                             except Exception as e:
#                                 resp = f"⚠️ {e}"
#                         st.session_state.chat_pairs.append((ex, resp))
#                         st.rerun()


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 5 · GENERATIVE AI
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_genai:
#     st.markdown('<div class="sec-head">💡 Generative AI Suite</div>', unsafe_allow_html=True)
#     gen_ai = st.session_state.gen_ai

#     # ── Status banner ──
#     if gen_ai._provider == "smart":
#         st.markdown("""
#         <div class="success-box">
#         ⚡ <strong>Smart AI active</strong> — instant built-in responses, zero downloads, no API key needed.<br>
#         Knows: ML algorithms, deep learning, NLP, Python, data science, and more.
#         For open-ended GPT-4 quality responses, add an API key in the sidebar.
#         </div>""", unsafe_allow_html=True)
#     elif gen_ai.is_available():
#         st.markdown(f'<div class="success-box">✅ <strong>{gen_ai._provider_config["name"]}</strong> connected and ready.</div>', unsafe_allow_html=True)
#     elif not gen_ai.pkg_installed():
#         st.markdown(f"""
#         <div class="info-box">
#         ⚠️ <strong>{gen_ai._provider_config["name"]}</strong> package not installed.<br>
#         Run: <code>{gen_ai.install_cmd()}</code> — then restart Streamlit.
#         </div>""", unsafe_allow_html=True)
#     else:
#         pkg_status = {
#             "🟢 OpenAI":    "✅ installed" if OPENAI_OK    else "❌  pip install openai",
#             "🔵 Google":    "✅ installed" if GOOGLE_OK    else "❌  pip install google-generativeai",
#             "🟣 Anthropic": "✅ installed" if ANTHROPIC_OK else "❌  pip install anthropic",
#         }
#         rows = "<br>".join(f"&nbsp;&nbsp;{k}: <code>{v}</code>" for k, v in pkg_status.items())
#         st.markdown(f"""
#         <div class="info-box">
#         ⚠️ No API key entered — using Smart AI mode.<br>
#         To enable full LLM responses, install a package and add your key:<br><br>{rows}
#         </div>""", unsafe_allow_html=True)

#     gen_tabs = st.tabs(["💬 Chatbot", "❓ Data Q&A", "💻 Code Gen", "🎨 Image Gen", "📄 Report"])

#     # ── Chatbot ──
#     with gen_tabs[0]:
#         st.subheader("Context-Aware Chatbot")
#         if "gen_chat" not in st.session_state:
#             st.session_state.gen_chat    = []
#             st.session_state.gen_history = []

#         for role, msg in st.session_state.gen_chat:
#             with st.chat_message(role):
#                 st.markdown(msg)

#         user_q = st.chat_input("Chat with AI…", key="gen_chat_input")
#         if user_q:
#             st.session_state.gen_chat.append(("user", user_q))
#             st.session_state.gen_history.append({"role": "user", "content": user_q})
#             with st.chat_message("user"):
#                 st.markdown(user_q)
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating…"):
#                     try:
#                         resp = gen_ai.chat(st.session_state.gen_history)
#                     except Exception as e:
#                         resp = f"⚠️ {e}"
#                 st.markdown(resp)
#             st.session_state.gen_chat.append(("assistant", resp))
#             st.session_state.gen_history.append({"role": "assistant", "content": resp})

#         if st.button("🗑️ Clear Conversation", key="clr_gen"):
#             st.session_state.gen_chat    = []
#             st.session_state.gen_history = []
#             st.rerun()

#     # ── Data Q&A ──
#     with gen_tabs[1]:
#         st.subheader("Prompt-Based Data Q&A")
#         st.markdown("Ask questions about your loaded dataset and get AI-generated insights.")
#         gen_option = st.selectbox("Analysis Type", [
#             "General Insights", "Trends Analysis", "Anomaly Detection", "Recommendations", "Custom Question",
#         ])
#         analysis_map = {
#             "General Insights":  "general",
#             "Trends Analysis":   "trends_analysis",
#             "Anomaly Detection": "anomaly_detection",
#             "Recommendations":   "recommendations",
#         }
#         if gen_option != "Custom Question":
#             if st.button("✨ Generate Insights", type="primary", key="gen_ins"):
#                 if st.session_state.data_summary:
#                     with st.spinner("Generating…"):
#                         insights = gen_ai.generate_insights(
#                             st.session_state.data_summary, analysis_map[gen_option]
#                         )
#                     st.markdown(f'<div class="result-box">{insights}</div>', unsafe_allow_html=True)
#                 else:
#                     st.warning("Load data in the 📊 Data tab first.")
#         else:
#             question = st.text_input("Your question:", placeholder="What features are most correlated with the target?")
#             if st.button("🔍 Ask", type="primary", key="gen_qa") and question:
#                 ctx = json.dumps(st.session_state.data_summary, default=str) if st.session_state.data_summary else None
#                 with st.spinner("Thinking…"):
#                     answer = gen_ai.answer_question(question, ctx)
#                 st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)

#     # ── Code Generation ──
#     with gen_tabs[2]:
#         st.subheader("💻 Code Generation")
#         code_lang   = st.selectbox("Language", ["Python", "JavaScript", "SQL", "Bash", "R", "TypeScript"])
#         code_prompt = st.text_area(
#             "Describe what you want to code:", height=110,
#             placeholder="Write a Python function to clean a pandas dataframe by removing duplicates and filling nulls with median values.",
#         )
#         if st.button("⚡ Generate Code", type="primary", key="code_gen"):
#             if code_prompt.strip():
#                 with st.spinner("Generating code…"):
#                     code_result = gen_ai.generate_code(code_prompt, language=code_lang)
#                 st.code(code_result, language=code_lang.lower())
#                 st.download_button(
#                     "⬇️ Download", code_result.encode(),
#                     file_name=f"generated.{code_lang.lower()[:2]}",
#                     mime="text/plain",
#                 )
#             else:
#                 st.warning("Enter a code description.")

#     # ── Image Generation ──
#     with gen_tabs[3]:
#         st.subheader("🎨 Image Generation from Text")
#         img_prompt = st.text_area(
#             "Image description:", height=100,
#             placeholder="A futuristic city at night with neon lights and flying cars, digital art style",
#         )
#         img_style = st.selectbox("Style", ["Photorealistic", "Digital Art", "Oil Painting", "Anime", "Sketch", "Cyberpunk"])
#         img_size  = st.selectbox("Size", ["1024x1024", "512x512", "1792x1024"])

#         if st.button("🎨 Generate Image", type="primary", key="img_gen"):
#             if img_prompt.strip():
#                 full_prompt = f"{img_prompt}, {img_style} style"
#                 with st.spinner("Generating image…"):
#                     result = gen_ai.generate_image(full_prompt, size=img_size)
#                 if result.get("url"):
#                     st.image(result["url"], caption=f"Generated: {img_prompt[:60]}…", use_column_width=True)
#                     st.markdown(f"[🔗 Open full size]({result['url']})")
#                 elif result.get("error"):
#                     st.markdown(f'<div class="info-box">⚠️ {result["error"]}</div>', unsafe_allow_html=True)
#             else:
#                 st.warning("Enter an image description.")

#     # ── Report ──
#     with gen_tabs[4]:
#         st.subheader("📄 Auto-Generated Analysis Report")
#         if st.button("📝 Generate Report", type="primary", key="gen_rep"):
#             payload = {
#                 "ml_metrics":   st.session_state.ml_metrics or {},
#                 "data_summary": st.session_state.data_summary or {},
#                 "model":        st.session_state.ml_results.get("model_name", "N/A") if st.session_state.ml_results else "N/A",
#             }
#             with st.spinner("Writing report…"):
#                 report = gen_ai.generate_report(payload)
#             st.text_area("Report", report, height=420)
#             st.download_button("⬇️ Download Report", report.encode(), file_name="ai_report.txt", mime="text/plain")


# # ──────────────────────────────────────────────────────────────────────────────
# #  TAB 6 · POWER BI
# # ──────────────────────────────────────────────────────────────────────────────
# with tab_pbi:
#     st.markdown('<div class="sec-head">📤 Power BI Export</div>', unsafe_allow_html=True)
#     exporter = st.session_state.powerbi_exp

#     if st.session_state.df is None:
#         st.markdown('<div class="info-box">👆 Load data in the <strong>📊 Data</strong> tab first.</div>', unsafe_allow_html=True)
#     else:
#         df = st.session_state.df
#         ec1, ec2 = st.columns(2)
#         with ec1:
#             include_parquet = st.checkbox("Include Parquet files", value=True)
#             export_name     = st.text_input("Dataset name", value="main_data")
#         with ec2:
#             st.markdown("**Available datasets:**")
#             available = {"Main Data": df}
#             if st.session_state.ml_results and st.session_state.ml_results.get("feature_importance"):
#                 available["Feature Importance"] = pd.DataFrame(st.session_state.ml_results["feature_importance"])
#             for name in available:
#                 st.markdown(f"• {name}")

#         if st.button("📊 Export All for Power BI", type="primary", use_container_width=True):
#             with st.spinner("Exporting…"):
#                 try:
#                     named = {export_name: df}
#                     if "Feature Importance" in available:
#                         named["feature_importance"] = available["Feature Importance"]
#                     paths = exporter.export_all(named, include_parquet=include_parquet)
#                     st.success(f"✅ Exported **{len(paths)}** files to `{OUTPUT_DIR}`")
#                     for p in paths:
#                         st.markdown(f"  • `{p.name}`")
#                 except Exception as e:
#                     st.error(f"❌ {e}")

#         st.divider()
#         st.info(exporter.generate_powerbi_instructions())

#         dl1, dl2 = st.columns(2)
#         with dl1:
#             st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
#                                file_name=f"{export_name}.csv", mime="text/csv", use_container_width=True)
#         with dl2:
#             if st.session_state.ml_results and st.session_state.ml_results.get("feature_importance"):
#                 fi_df = pd.DataFrame(st.session_state.ml_results["feature_importance"])
#                 st.download_button("⬇️ Feature Importance CSV", fi_df.to_csv(index=False).encode(),
#                                    file_name="feature_importance.csv", mime="text/csv", use_container_width=True)


# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="footer">
#     Multi-AI Analytics Platform v2.1 &nbsp;·&nbsp; Clean Edition &nbsp;·&nbsp;
#     ML · DL · NLP · GenAI · PowerBI &nbsp;&nbsp;
#     <span>| KYOTO-Z |</span>
# </div>
# """, unsafe_allow_html=True)









































































"""
app.py  —  Multi-AI Analytics Platform v2.1 (Clean Edition)
Run:  streamlit run app.py

Requirements (install before running):
    pip install streamlit pandas numpy pillow plotly scikit-learn xgboost lightgbm
    pip install opencv-python-headless transformers torch torchvision
    pip install openai google-generativeai anthropic   # optional, for API providers
    pip install spacy && python -m spacy download en_core_web_sm   # optional NER

Place these files in the SAME folder as app.py:
    generative_ai.py
    nlp_module.py
    ml_models.py        (your existing file — unchanged)
    dl_module.py        (your existing file — unchanged)
    data/data_loader.py
    data/powerbi_export.py
    config.py
    utils/helpers.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ── Root path setup ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Core imports ──────────────────────────────────────────────────────────────
try:
    from config import Config, OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

from data.data_loader import DataLoader
from data.powerbi_export import PowerBIExporter
from models.ml_models import MLPipeline, XGBoostPipeline, EnsemblePipeline
from models.generative_ai import GenerativeAI, OPENAI_OK, GOOGLE_OK, ANTHROPIC_OK
from utils.helpers import (
    create_feature_importance_chart, create_metrics_dashboard,
    create_confusion_matrix, create_correlation_heatmap,
    create_class_distribution, create_actual_vs_predicted,
)

try:
    from models.ml_models import LightGBMPipeline
    LGB_OK = True
except Exception:
    LightGBMPipeline = None
    LGB_OK = False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-AI Analytics Platform",
    page_icon=str(ROOT / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 20_000

def valid_export_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value.strip()))

legal_page = st.query_params.get("page")
if legal_page in {"privacy", "terms"}:
    if legal_page == "privacy":
        st.title("Privacy Policy")
        st.write("This app processes files and text you submit to provide analytics. Uploaded data is held in the current Streamlit session and is not stored in a user account by this app. Exports are written to the deployment's local output directory, which may be temporary. If you choose an external AI provider, the prompt or content sent to that provider is subject to that provider's terms and privacy policy. Do not submit sensitive or regulated data unless you have confirmed your organization's requirements.")
        st.write("This project does not implement account-based access controls, persistent database storage, or a retention schedule. Contact the repository owner for privacy questions.")
    else:
        st.title("Terms of Use")
        st.write("Use this project at your own risk for analysis and experimentation. You are responsible for the data you upload, confirming that you have permission to use it, and reviewing generated results before relying on them. The software is provided without warranties and may produce incomplete or incorrect output. Your use of external AI providers is also governed by their terms.")
        st.write("This page is a project-level summary, not jurisdiction-specific legal advice. The repository license governs use and distribution of the software.")
    st.markdown('<a href="/">Return to the app</a>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Global CSS - Aurora Dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root { color-scheme: dark; }
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important; }
.stApp { background:#090a0c !important; background-image:none !important; color:#e5e7eb; }
section[data-testid="stSidebar"] { background:#0d0e10 !important; border-right:1px solid #292b30 !important; }
section[data-testid="stSidebar"] * { color:#d1d5db; }
.hero-wrap,.mod-card,.stat-card,.result-box { background:#111215 !important; border:1px solid #303238 !important; border-radius:6px !important; }
.hero-wrap { padding:2rem; margin-bottom:1.5rem; }
.hero-title { color:#f4f4f5 !important; font-size:clamp(1.8rem,4vw,2.6rem); font-weight:700; line-height:1.2; }
.hero-sub { color:#a1a1aa; font-size:1rem; letter-spacing:0; }
.hero-badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:1rem; }
.badge,.badge.teal,.badge.blue { background:#191a1d !important; border:1px solid #393b40 !important; border-radius:3px !important; color:#d4d4d8 !important; padding:4px 10px; }
.mod-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin:1.2rem 0; }
.mod-card { transition:none !important; }
.mod-card:hover { border-color:#555961 !important; transform:none !important; }
.mod-card .glow { display:none; }
.mod-card .title { color:#f1f1f2; }
.mod-card .desc { color:#a1a1aa; line-height:1.5; }
.stat-row { display:flex; gap:1rem; margin:1rem 0; }
.sec-head { color:#e4e4e7 !important; background:none !important; -webkit-text-fill-color:#e4e4e7 !important; font-size:1.35rem; font-weight:650; margin:1rem 0 .6rem; }
.info-box,.success-box { background:#141518 !important; border-left:3px solid #737780 !important; border-radius:2px !important; color:#d4d4d8 !important; padding:.9rem 1.1rem; }
.stButton > button { background:#25272b !important; color:#f4f4f5 !important; border:1px solid #41444a !important; border-radius:4px !important; font-weight:600 !important; transition:none !important; }
.stButton > button:hover { background:#32353a !important; transform:none !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#101114 !important; border:1px solid #292b30; border-radius:4px !important; padding:4px; }
.stTabs [data-baseweb="tab"],.stTabs [aria-selected="true"] { border-radius:3px !important; }
.stTabs [aria-selected="true"] { background:#292b30 !important; color:#f4f4f5 !important; }
.stTextArea textarea,.stTextInput input,.stSelectbox select { background:#121316 !important; border:1px solid #34363b !important; border-radius:4px !important; color:#e5e7eb !important; }
.stDataFrame { border:1px solid #292b30 !important; border-radius:4px !important; }
[data-testid="stMetricValue"] { color:#e4e4e7 !important; font-weight:700; }
[data-testid="stMetricLabel"] { color:#a1a1aa !important; }
[data-testid="stChatMessage"] { background:#111215 !important; border:1px solid #292b30 !important; border-radius:5px !important; }
.footer { text-align:center; color:#a1a1aa; font-size:.78rem; margin-top:2rem; padding:1rem; border-top:1px solid #292b30; }
.stSpinner > div { border-top-color:#a1a1aa !important; }
hr { border-color:#292b30 !important; }
@media(max-width:700px) { .mod-grid { grid-template-columns:1fr; } .hero-wrap { padding:1.25rem; } .stat-row { flex-wrap:wrap; } .stat-card { min-width:42%; } }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-size:1.1rem;font-weight:700;color:#f4f4f5">
            AI Platform
        </div>
        <div style="font-size:0.72rem;color:#475569;letter-spacing:0.08em;margin-top:2px">
            v2.1 · CLEAN EDITION
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("###  Generative AI")
    provider_choice = st.selectbox(
        "Provider",
        ["smart", "openai", "google", "anthropic"],
        format_func=lambda x: {
            "smart":     " Smart AI (Instant · No API Key)",
            "openai":    " OpenAI GPT-4o",
            "google":    " Google Gemini",
            "anthropic": " Anthropic Claude",
        }[x],
    )
    if provider_choice == "smart":
        st.caption(" Instant responses - no API key, no downloads")
        api_key_input = ""
    else:
            api_key_input = st.text_input("API Key", type="password", placeholder="Paste key here…", max_chars=512)
    st.divider()

    st.markdown("###  System Status")
    for distribution, label in [
        ("torch", "PyTorch"),
        ("scikit-learn", "scikit-learn"),
        ("xgboost", "XGBoost"),
        ("transformers", "Transformers"),
        ("opencv-python-headless", "OpenCV"),
        ("lightgbm", "LightGBM"),
    ]:
        try:
            ver = importlib_metadata.version(distribution)
            st.markdown(f"{label}: `{ver}`")
        except importlib_metadata.PackageNotFoundError:
            st.markdown(f"{label}: unavailable")
    st.divider()

    if st.button(" Reset Session", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:1rem;color:#334155;font-size:0.7rem">
        Multi-AI Analytics Platform<br>
        <span style="color:#6366f1;font-family:'JetBrains Mono',monospace">KYOTO-Z</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "data_loader":    DataLoader(),
        "powerbi_exp":    PowerBIExporter(OUTPUT_DIR),
        "df":             None,
        "data_summary":   None,
        "ml_pipeline":    None,
        "ml_results":     None,
        "ml_metrics":     None,
        "gen_ai":         GenerativeAI(provider="smart"),
        "uploaded_image": None,
        "target_column":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# Always rebuild gen_ai when provider/key change
st.session_state.gen_ai = GenerativeAI(api_key=api_key_input, provider=provider_choice)


# ══════════════════════════════════════════════════════════════════════════════
#  Hero Banner
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title"> Multi-AI Analytics Platform</div>
  <div class="hero-sub">Upload a dataset to inspect its contents, train a model, and export the results. Image and text analysis tools are available in separate tabs.</div>
  <div class="hero-badges">
    <span class="badge">scikit-learn</span>
    <span class="badge">XGBoost</span>
    <span class="badge teal">HuggingFace</span>
    <span class="badge blue">OpenCV</span>
    <span class="badge">PyTorch / TF</span>
    <span class="badge teal">Plotly</span>
    <span class="badge blue">Power BI</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_data, tab_ml, tab_dl, tab_nlp, tab_genai, tab_pbi = st.tabs([
    " Home", " Data", " ML Pipeline",
    " Deep Learning", " NLP", " Generative AI", " Power BI",
])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 0 · HOME
# ──────────────────────────────────────────────────────────────────────────────
with tab_home:
    st.markdown("""
    <div class="mod-grid">
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(99,102,241,0.5)"></div>
        <div class="icon"></div>
        <div class="title">ML Pipeline</div>
        <div class="desc">Train classifiers and regressors, review evaluation results, and inspect feature importance.</div>
      </div>
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(20,184,166,0.5)"></div>
        <div class="icon"></div>
        <div class="title">Data Explorer</div>
        <div class="desc">Upload CSV/Excel/JSON. Auto EDA, correlation heatmaps, distributions, scatter builder.</div>
      </div>
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(139,92,246,0.5)"></div>
        <div class="icon"></div>
        <div class="title">Deep Learning</div>
        <div class="desc">MobileNetV2, ResNet50, VGG16 classification. Grad-CAM, face/edge detection, image filters.</div>
      </div>
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(59,130,246,0.5)"></div>
        <div class="icon"></div>
        <div class="title">NLP Suite</div>
        <div class="desc">Sentiment analysis, NER, zero-shot classification, summarization - all via HuggingFace.</div>
      </div>
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(236,72,153,0.5)"></div>
        <div class="icon"></div>
        <div class="title">Generative AI</div>
        <div class="desc">GPT-4 · Gemini · Claude. Context-aware chatbot, Q&A, code gen, image generation, reports.</div>
      </div>
      <div class="mod-card">
        <div class="glow" style="--gc:rgba(245,158,11,0.5)"></div>
        <div class="icon"></div>
        <div class="title">Power BI Export</div>
        <div class="desc">Export datasets, feature importance tables, and predictions as CSV/Parquet for Power BI.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    st.markdown("Available tools: dataset exploration, model training, image analysis, text analysis, and optional AI providers.")

    st.markdown("""
    <div class="info-box">
     <strong>Quick Start:</strong> Head to <em> Data</em> to upload your dataset,
    then use <em> ML Pipeline</em> to train models, or jump to <em> Deep Learning</em>
    for image analysis. Use <em> Generative AI</em> for automated insights.
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 · DATA
# ──────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="sec-head"> Data Loading & Exploration</div>', unsafe_allow_html=True)

    col_up, col_sum = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, JSON or Image",
            type=["csv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "bmp", "webp"],
        )
        if uploaded_file:
            try:
                if uploaded_file.size > MAX_UPLOAD_BYTES:
                    st.error("Files must be 25 MB or smaller.")
                    st.stop()
                if uploaded_file.type.startswith("image"):
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, caption=uploaded_file.name, width=420)
                    st.session_state.uploaded_image = img
                else:
                    ext = Path(uploaded_file.name).suffix.lower()
                    if ext == ".csv":
                        df = pd.read_csv(uploaded_file)
                    elif ext in [".xlsx", ".xls"]:
                        df = pd.read_excel(uploaded_file)
                    elif ext == ".json":
                        df = pd.read_json(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file)

                    st.session_state.df = df
                    st.session_state.data_summary = st.session_state.data_loader.get_data_summary(df)
                    st.success(f" Loaded **{uploaded_file.name}** - {df.shape[0]:,} rows × {df.shape[1]} cols")
            except Exception:
                st.error("Could not read this file. Check its format and try again.")

    if st.session_state.df is not None:
        df = st.session_state.df
        with col_sum:
            st.markdown(f'<div class="stat-card" style="margin-bottom:8px"><div class="stat-val">{df.shape[0]:,}</div><div class="stat-lbl">Rows</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card" style="margin-bottom:8px"><div class="stat-val">{df.shape[1]}</div><div class="stat-lbl">Columns</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-val">{df.isnull().sum().sum()}</div><div class="stat-lbl">Missing</div></div>', unsafe_allow_html=True)

        dtabs = st.tabs([" Preview", " Statistics", " Charts", " Correlations"])

        with dtabs[0]:
            st.dataframe(df.head(50), width="stretch")

        with dtabs[1]:
            st.dataframe(df.describe(include="all").T, width="stretch")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Data Types**")
                st.dataframe(df.dtypes.rename("Type").reset_index().rename(columns={"index": "Column"}), width="stretch")
            with col_b:
                st.markdown("**Missing Values**")
                miss = df.isnull().sum().reset_index()
                miss.columns = ["Column", "Missing"]
                miss["Pct"] = (miss["Missing"] / len(df) * 100).round(2)
                st.dataframe(miss[miss["Missing"] > 0], width="stretch")

        with dtabs[2]:
            import plotly.express as px
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            chart_type = st.selectbox("Chart Type", ["Histogram", "Bar Chart", "Scatter", "Box Plot", "Line"])

            if chart_type == "Histogram" and num_cols:
                col = st.selectbox("Column", num_cols)
                fig = px.histogram(df, x=col, template="plotly_dark", color_discrete_sequence=["#6366f1"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,18,35,0.6)")
                st.plotly_chart(fig, width="stretch")
            elif chart_type == "Bar Chart" and cat_cols and num_cols:
                xc = st.selectbox("X (categorical)", cat_cols)
                yc = st.selectbox("Y (numeric)", num_cols)
                fig = px.bar(df.groupby(xc)[yc].mean().reset_index(), x=xc, y=yc,
                             template="plotly_dark", color=yc, color_continuous_scale="Viridis")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
            elif chart_type == "Scatter" and len(num_cols) >= 2:
                xc = st.selectbox("X", num_cols, key="sc_x")
                yc = st.selectbox("Y", num_cols, index=1, key="sc_y")
                cc = st.selectbox("Color", ["None"] + cat_cols)
                fig = px.scatter(df, x=xc, y=yc, color=None if cc == "None" else cc,
                                 template="plotly_dark", opacity=0.7)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
            elif chart_type == "Box Plot" and num_cols:
                yc = st.selectbox("Value", num_cols, key="bp_y")
                gc = st.selectbox("Group", ["None"] + cat_cols)
                fig = px.box(df, y=yc, x=None if gc == "None" else gc,
                             template="plotly_dark", color_discrete_sequence=["#8b5cf6"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
            elif chart_type == "Line" and num_cols:
                yc = st.selectbox("Column", num_cols, key="lc_y")
                fig = px.line(df.reset_index(), x="index", y=yc,
                              template="plotly_dark", color_discrete_sequence=["#2dd4bf"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

        with dtabs[3]:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            if len(num_cols) >= 2:
                fig = create_correlation_heatmap(df)
                st.plotly_chart(fig, width="stretch")
                if cat_cols:
                    tc = st.selectbox("Class column for distribution", cat_cols)
                    fig2 = create_class_distribution(df[tc], f"{tc} Distribution")
                    st.plotly_chart(fig2, width="stretch")
            else:
                st.info("Need at least 2 numeric columns for correlation.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2 · ML PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
with tab_ml:
    st.markdown('<div class="sec-head"> Machine Learning Pipeline</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown('<div class="info-box"> Load a dataset in the <strong> Data</strong> tab first.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.df

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            target_col = st.selectbox(" Target Column", df.columns.tolist())
            st.session_state.target_column = target_col
        with mc2:
            from data.data_loader import DataLoader as _DL
            task_type = _DL().detect_task_type(df[target_col])
            st.markdown(f"**Detected Task:** `{task_type}`")
            model_options = {
                "classification": ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "XGBoost", "LightGBM", "Ensemble"],
                "regression":     ["Random Forest", "Gradient Boosting", "Ridge Regression", "Lasso Regression", "SVM", "XGBoost", "LightGBM", "Ensemble"],
            }
            model_name = st.selectbox(" Algorithm", model_options[task_type])
        with mc3:
            test_size = st.slider("Test Split %", 10, 40, 20) / 100
            cv_folds  = st.slider("CV Folds", 2, 10, 5)

        if st.button(" Train Model", type="primary", width="stretch"):
            with st.spinner("Training…"):
                try:
                    if model_name == "XGBoost":
                        pipe = XGBoostPipeline(task_type=task_type)
                    elif model_name == "LightGBM" and LGB_OK and LightGBMPipeline is not None:
                        pipe = LightGBMPipeline(task_type=task_type)
                    elif model_name == "Ensemble":
                        pipe = EnsemblePipeline(task_type=task_type)
                    else:
                        pipe = MLPipeline(task_type=task_type, model_name=model_name)

                    result = pipe.preprocess(df, target_col=target_col)
                    if result is None:
                        st.error(" Preprocessing returned None. Check your target column.")
                        st.stop()
                    X, y = result
                    if y is None:
                        st.error(" Target column could not be extracted.")
                        st.stop()
                    metrics = pipe.train(X, y, test_size=test_size)

                    st.session_state.ml_pipeline = pipe
                    st.session_state.ml_metrics  = metrics
                    st.session_state.ml_results  = {
                        "model_name": model_name,
                        "task_type":  task_type,
                        "feature_importance": pipe.get_feature_importance().to_dict("records")
                            if hasattr(pipe, "get_feature_importance") else [],
                    }
                    st.success(f" **{model_name}** trained successfully!")
                except Exception:
                    st.error("Training failed. Check the selected target column and dataset, then try again.")

        if st.session_state.ml_metrics:
            metrics = st.session_state.ml_metrics
            st.markdown("---")
            st.markdown('<div class="sec-head"> Results</div>', unsafe_allow_html=True)

            num_m = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            cols_m = st.columns(min(len(num_m), 4))
            for i, (k, v) in enumerate(list(num_m.items())[:4]):
                cols_m[i].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

            fig_dash = create_metrics_dashboard(metrics)
            st.plotly_chart(fig_dash, width="stretch")

            pipe = st.session_state.ml_pipeline
            if pipe and pipe.is_fitted:
                result_tabs = st.tabs([" Confusion Matrix / Scatter", " Feature Importance", " Report"])

                with result_tabs[0]:
                    task_type = st.session_state.ml_results["task_type"]
                    if task_type == "classification" and pipe.y_pred is not None:
                        labels = [str(c) for c in pipe.classes_] if pipe.classes_ is not None else None
                        fig_cm = create_confusion_matrix(
                            list(pipe.y_test),  # type: ignore[arg-type]
                            list(pipe.y_pred),  # type: ignore[arg-type]
                            labels,
                        )
                        st.plotly_chart(fig_cm, width="stretch")
                    elif task_type == "regression" and pipe.y_pred is not None:
                        fig_avp = create_actual_vs_predicted(
                            pipe.y_test, pipe.y_pred,
                            f"{st.session_state.ml_results['model_name']} - Actual vs Predicted"
                        )
                        st.plotly_chart(fig_avp, width="stretch")

                with result_tabs[1]:
                    if hasattr(pipe, "get_feature_importance"):
                        fi_df = pipe.get_feature_importance()
                        if not fi_df.empty:
                            fig_fi = create_feature_importance_chart(fi_df, top_n=20)
                            st.plotly_chart(fig_fi, width="stretch")
                            st.dataframe(fi_df.head(20), width="stretch")

                with result_tabs[2]:
                    if "classification_report" in metrics:
                        st.code(metrics["classification_report"], language="text")
                    else:
                        for k, v in num_m.items():
                            st.markdown(f"**{k.replace('_', ' ').title()}:** `{v:.4f}`")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3 · DEEP LEARNING
# ──────────────────────────────────────────────────────────────────────────────
with tab_dl:
    st.markdown('<div class="sec-head"> Deep Learning & Computer Vision</div>', unsafe_allow_html=True)

    from models.dl_module import (
        _classify_image_tf, _classify_image_torch,
        detect_edges_opencv, detect_faces_opencv, apply_image_filters,
    )

    dl_uploaded = st.file_uploader("Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="dl_up")

    if dl_uploaded is None:
        st.markdown('<div class="info-box"> Upload any image - try animals, faces, objects, or landscapes.</div>', unsafe_allow_html=True)
    else:
        if dl_uploaded.size > MAX_UPLOAD_BYTES:
            st.error("Images must be 25 MB or smaller.")
            st.stop()
        try:
            Image.open(dl_uploaded).verify()
            dl_uploaded.seek(0)
            pil_img = Image.open(dl_uploaded).convert("RGB")
        except Exception:
            st.error("Could not read this image. Choose a valid JPG or PNG file.")
            st.stop()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(pil_img, caption="Uploaded Image", width="stretch")
        with c2:
            w, h = pil_img.size
            arr_np = np.array(pil_img.convert("RGB"))
            for lbl, val in [
                ("Resolution", f"{w}×{h}"),
                ("Mode",       pil_img.mode),
                ("Brightness", f"{arr_np.mean():.1f}"),
                ("File size",  f"{dl_uploaded.size / 1024:.1f} KB"),
            ]:
                st.markdown(
                    f'<div class="stat-card" style="margin-bottom:6px">'
                    f'<div class="stat-val" style="font-size:1rem">{val}</div>'
                    f'<div class="stat-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        dl_tabs = st.tabs([" Classification", " Grad-CAM", " Detection", " Filters"])

        with dl_tabs[0]:
            st.subheader("Image Classification - ImageNet 1K")
            backend = st.radio("Backend", ["TensorFlow/Keras", "PyTorch"], horizontal=True)
            if backend == "TensorFlow/Keras":
                model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50", "VGG16"])
            else:
                model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50"])

            if st.button(" Classify Image", type="primary", key="cls_btn"):
                with st.spinner(f"Running {model_choice}…"):
                    try:
                        # Verify TF is importable before calling classify
                        if backend == "TensorFlow/Keras":
                            try:
                                import tensorflow as _tf_check  # type: ignore[import-untyped]
                            except ImportError:
                                raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
                            results = _classify_image_tf(pil_img, model_choice)
                        else:
                            results = _classify_image_torch(pil_img, model_choice)

                        import plotly.graph_objects as go
                        st.markdown(
                            f'<div class="success-box"> <strong>{results[0]["Label"]}</strong> - {results[0]["Confidence"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(pd.DataFrame(results), width="stretch")

                        labels_list = [r["Label"][:28] for r in results]
                        scores_list = [r["Score"] for r in results]
                        colors_list = ["#a78bfa" if i == 0 else "#334155" for i in range(len(scores_list))]
                        fig = go.Figure(go.Bar(
                            x=scores_list[::-1], y=labels_list[::-1], orientation="h",
                            marker=dict(color=colors_list[::-1]),
                            text=[f"{s * 100:.1f}%" for s in scores_list[::-1]],
                            textposition="outside",
                        ))
                        fig.update_layout(title="Top-5 Confidence Scores", template="plotly_dark",
                                          paper_bgcolor="rgba(0,0,0,0)", height=280)
                        st.plotly_chart(fig, width="stretch")
                    except Exception:
                        st.error("Image classification failed. Check the image and try again.")
                        st.info("Make sure TensorFlow or PyTorch is installed.")

        with dl_tabs[1]:
            st.subheader("Grad-CAM - Class Activation Heatmap")
            st.markdown("Highlights **which image regions** the model focused on for its prediction.")

            _tf_ok, _pt_ok = False, False
            try:
                import tensorflow; _tf_ok = True
            except ImportError:
                pass
            try:
                import torch; _pt_ok = True
            except ImportError:
                pass

            if not _tf_ok and not _pt_ok:
                st.markdown("""
                <div class="info-box">
                 Grad-CAM requires TensorFlow or PyTorch.<br>
                Install: <code>pip install tensorflow</code> or <code>pip install torch torchvision</code>
                </div>""", unsafe_allow_html=True)
            else:
                backend_opts = (["TensorFlow/Keras"] if _tf_ok else []) + (["PyTorch"] if _pt_ok else [])
                gc_backend = st.radio("Backend", backend_opts, horizontal=True, key="gc_back")
                gc_model   = st.selectbox(
                    "Model",
                    ["MobileNetV2", "ResNet50"] if gc_backend == "PyTorch" else ["MobileNetV2", "ResNet50", "VGG16"],
                    key="gc_m",
                )

                if st.button(" Generate Grad-CAM", type="primary"):
                    with st.spinner("Computing Grad-CAM…"):
                        try:
                            import cv2 as _cv
                            import numpy as _np

                            orig_224 = _np.array(pil_img.convert("RGB").resize((224, 224)))

                            if gc_backend == "TensorFlow/Keras":
                                # Robust TensorFlow import - handles tf2, tf-cpu, standalone keras
                                try:
                                    import tensorflow as tf  # type: ignore[import-untyped]
                                except ImportError:
                                    st.error(" TensorFlow not installed. Run: pip install tensorflow")
                                    st.stop()
                                try:
                                    from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore[import-untyped]
                                except ImportError:
                                    try:
                                        from tensorflow.keras.utils import img_to_array  # type: ignore[import-untyped]
                                    except ImportError:
                                        from keras.utils import img_to_array  # type: ignore[import-untyped]
                                try:
                                    from models.dl_module import _load_tf_model
                                except ImportError:
                                    from dl_module import _load_tf_model  # type: ignore[import]

                                model, preprocess, decode, (ih, iw) = _load_tf_model(gc_model)
                                arr = preprocess(_np.expand_dims(img_to_array(
                                    pil_img.convert("RGB").resize((iw, ih))), 0))

                                layer_name = None
                                try:
                                    _conv2d_cls = tf.keras.layers.Conv2D  # type: ignore[attr-defined]
                                except AttributeError:
                                    import keras
                                    _conv2d_cls = keras.layers.Conv2D  # type: ignore[attr-defined]
                                for layer in reversed(model.layers):
                                    if isinstance(layer, _conv2d_cls):
                                        layer_name = layer.name
                                        break

                                if not layer_name:
                                    st.warning("No Conv2D layer found in model.")
                                    st.stop()

                                grad_model = tf.keras.models.Model(
                                    inputs=model.inputs,
                                    outputs=[model.get_layer(layer_name).output, model.output],
                                )
                                with tf.GradientTape() as tape:
                                    conv_out, preds = grad_model(arr)
                                    top_cls = int(tf.argmax(preds[0]).numpy())  # type: ignore[union-attr]
                                    loss = preds[:, top_cls]  # type: ignore[index]
                                grads   = tape.gradient(loss, conv_out)
                                pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
                                heatmap = conv_out[0] @ pooled[..., tf.newaxis]
                                heatmap = tf.squeeze(heatmap).numpy()
                                top_label = decode(model.predict(arr, verbose=0), top=1)[0][0][1].replace("_", " ").title()

                            else:  # PyTorch
                                import torch
                                import torchvision.models as M
                                import torchvision.transforms as T
                                import json, urllib.request

                                pt_map = {"MobileNetV2": M.mobilenet_v2, "ResNet50": M.resnet50}
                                pt_model = pt_map.get(gc_model, M.mobilenet_v2)(weights="DEFAULT")
                                pt_model.eval()

                                last_conv = None
                                for name, m in pt_model.named_modules():
                                    if isinstance(m, torch.nn.Conv2d):
                                        last_conv = (name, m)

                                if last_conv is None:
                                    st.warning("No Conv2d layer found.")
                                    st.stop()

                                activations, gradients = [], []

                                def _fwd_hook(mod, inp, out):
                                    activations.clear(); activations.append(out.detach())

                                def _bwd_hook(mod, grad_in, grad_out):
                                    gradients.clear(); gradients.append(grad_out[0].detach())

                                h1 = last_conv[1].register_forward_hook(_fwd_hook)
                                h2 = last_conv[1].register_full_backward_hook(_bwd_hook)

                                _transform = T.Compose([
                                    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
                                _tf_raw = _transform(pil_img.convert("RGB"))
                                tf_img = _tf_raw.unsqueeze(0)  # torch Tensor - ignore PIL Image complaint  # type: ignore[union-attr]
                                tf_img.requires_grad_(True)

                                output = pt_model(tf_img)
                                top_cls_pt = output.argmax(dim=1).item()
                                pt_model.zero_grad()
                                output[0, top_cls_pt].backward()
                                h1.remove(); h2.remove()

                                act     = activations[0].squeeze()
                                grad    = gradients[0].squeeze()
                                weights = grad.mean(dim=(1, 2))
                                heatmap = (weights[:, None, None] * act).sum(dim=0).numpy()
                                heatmap = _np.maximum(heatmap, 0)

                                try:
                                    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                                    with urllib.request.urlopen(url, timeout=5) as r:
                                        class_labels = json.load(r)
                                    top_label = class_labels[top_cls_pt].replace("_", " ").title()
                                except Exception:
                                    top_label = f"Class {top_cls_pt}"

                            # ── Render heatmap ──
                            heatmap = heatmap / (heatmap.max() + 1e-8)
                            h_res   = _cv.resize(heatmap, (224, 224))
                            _h_uint8 = _np.array(255 * h_res, dtype=_np.uint8)
                            h_col   = _cv.cvtColor(
                                _cv.applyColorMap(_h_uint8, _cv.COLORMAP_JET),  # type: ignore[call-overload]
                                _cv.COLOR_BGR2RGB,
                            )
                            overlay = _np.array(orig_224 * 0.6 + h_col * 0.4, dtype=_np.uint8)

                            gc1, gc2, gc3 = st.columns(3)
                            gc1.image(orig_224, caption="Original",  width="stretch")
                            gc2.image(h_col,    caption="Heatmap",   width="stretch")
                            gc3.image(overlay,  caption="Overlay",   width="stretch")
                            st.markdown(
                                f'<div class="success-box"> Top prediction: <strong>{top_label}</strong> '
                                f'- red/yellow regions = highest model attention</div>',
                                unsafe_allow_html=True,
                            )

                        except Exception:
                            st.error("Image analysis failed. Check the selected model and image, then try again.")

        with dl_tabs[2]:
            st.subheader("OpenCV Detection")
            cv_task = st.selectbox("Task", ["Face Detection", "Edge Detection"])
            t1 = t2 = None
            if cv_task == "Edge Detection":
                t1 = st.slider("Threshold 1", 10, 200, 50)
                t2 = st.slider("Threshold 2", 50, 400, 150)

            if st.button("▶ Run Detection", type="primary", key="det_btn"):
                with st.spinner("Running OpenCV…"):
                    if cv_task == "Edge Detection":
                        import cv2 as _cv
                        _t1: float = float(t1) if t1 is not None else 50.0
                        _t2: float = float(t2) if t2 is not None else 150.0
                        gray  = _cv.cvtColor(np.array(pil_img.convert("RGB")), _cv.COLOR_RGB2GRAY)
                        edges = _cv.Canny(_cv.GaussianBlur(gray, (5, 5), 0), _t1, _t2)
                        dc1, dc2 = st.columns(2)
                        dc1.image(pil_img, caption="Original",       width="stretch")
                        dc2.image(edges,   caption="Edges",           width="stretch", clamp=True)
                        st.info(f"Edge pixels: **{np.sum(edges > 0):,}**")
                    else:
                        result_img, face_count = detect_faces_opencv(pil_img)
                        dc1, dc2 = st.columns(2)
                        dc1.image(pil_img,    caption="Original",    width="stretch")
                        dc2.image(result_img, caption="Detections",  width="stretch")
                        if face_count > 0:
                            st.markdown(f'<div class="success-box"> Detected <strong>{face_count}</strong> face(s).</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No faces detected. Try a clear frontal portrait.")

        with dl_tabs[3]:
            st.subheader("Image Filters Gallery")
            if st.button(" Apply All Filters", type="primary", key="flt_btn"):
                with st.spinner("Applying filters…"):
                    filters  = apply_image_filters(pil_img)
                    cols_f   = st.columns(3)
                    for i, (name, img) in enumerate(filters.items()):
                        with cols_f[i % 3]:
                            st.image(img, caption=name, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 4 · NLP
# ──────────────────────────────────────────────────────────────────────────────
with tab_nlp:
    st.markdown('<div class="sec-head"> NLP Suite</div>', unsafe_allow_html=True)

    try:
        from models.nlp_module import (  # type: ignore[import]
            run_sentiment, run_ner, run_text_classification,
            run_summarization, chat_with_model,
        )
    except ImportError:
        from nlp_module import (  # type: ignore[import]
            run_sentiment, run_ner, run_text_classification,
            run_summarization, chat_with_model,
        )

    nlp_tabs = st.tabs([" Sentiment", " NER", " Classification", " Summarization", " Chatbot"])

    # ── Sentiment ──
    with nlp_tabs[0]:
        st.subheader("Sentiment Analysis - DistilBERT")
        mode = st.radio("Mode", ["Single", "Batch"], horizontal=True)
        if mode == "Single":
            txt = st.text_area("Text to analyze:", height=110, max_chars=MAX_TEXT_CHARS,
                               placeholder="The product quality is amazing and delivery was super fast!")
            if st.button(" Analyze", type="primary", key="sa_btn"):
                if not txt.strip():
                    st.warning("Enter some text.")
                else:
                    with st.spinner("Loading DistilBERT and analyzing…"):
                        r = run_sentiment([txt])
                    if r:
                        color = "#22c55e" if r[0]["Sentiment"] == "POSITIVE" else "#ef4444"
                        icon  = "" if r[0]["Sentiment"] == "POSITIVE" else ""
                        st.markdown(f"""
                        <div class="result-box" style="border-left:5px solid {color}">
                            <div style="font-size:2rem">{icon}</div>
                            <div style="font-size:1.4rem;font-weight:700;color:{color}">{r[0]["Sentiment"]}</div>
                            <div style="color:#94a3b8;margin-top:6px">Confidence: <strong style="color:#e2e8f0">{r[0]["Confidence"]}</strong></div>
                        </div>""", unsafe_allow_html=True)
        else:
            batch = st.text_area("One sentence per line:", height=180, max_chars=MAX_TEXT_CHARS,
                                 placeholder="Great product!\nTerrible experience.\nIt was okay.")
            if st.button(" Analyze All", type="primary", key="sa_batch"):
                lines = [ln.strip() for ln in batch.split("\n") if ln.strip()]
                if lines:
                    with st.spinner("Analyzing…"):
                        results = run_sentiment(lines)
                    df_s = pd.DataFrame(results)
                    st.dataframe(df_s, width="stretch")
                    import plotly.express as px
                    pos = sum(1 for r in results if r["Sentiment"] == "POSITIVE")
                    fig = px.pie(
                        values=[pos, len(results) - pos],
                        names=["Positive", "Negative"],
                        color_discrete_sequence=["#22c55e", "#ef4444"],
                        hole=0.4, template="plotly_dark",
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")

    # ── NER ──
    with nlp_tabs[1]:
        st.subheader("Named Entity Recognition")
        ner_txt = st.text_area("Text for NER:", height=130, max_chars=MAX_TEXT_CHARS,
            value="Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")
        if st.button(" Extract Entities", type="primary", key="ner_btn"):
            with st.spinner("Running NER…"):
                ents = run_ner(ner_txt)
            if ents:
                df_ner = pd.DataFrame(ents)
                st.dataframe(df_ner, width="stretch")
                import plotly.express as px
                vc = df_ner["Type"].value_counts().reset_index()
                vc.columns = ["Type", "count"]
                fig = px.bar(vc, x="Type", y="count", template="plotly_dark",
                             color="Type", color_discrete_sequence=["#a78bfa", "#60a5fa", "#34d399", "#f472b6"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig, width="stretch")
                type_icons = {"PER": "", "ORG": "", "LOC": "", "MISC": "", "GPE": ""}
                for _, row in df_ner.iterrows():
                    st.markdown(f"- **{row['Entity']}** → {type_icons.get(row['Type'], '')} `{row['Type']}` ({row['Score']})")
            else:
                st.info("No entities found.")

    # ── Zero-Shot Classification ──
    with nlp_tabs[2]:
        st.subheader("Zero-Shot Text Classification")
        cl_txt = st.text_area("Text to classify:", height=110, max_chars=MAX_TEXT_CHARS,
            value="The new iPhone features an upgraded camera and faster processor.")
        cl_labels = st.text_input("Candidate labels (comma-separated):",
            value="technology, sports, politics, business, health, entertainment")
        if st.button(" Classify", type="primary", key="zs_btn"):
            lbls = [lb.strip() for lb in cl_labels.split(",") if lb.strip()]
            if cl_txt.strip() and lbls:
                with st.spinner("Running zero-shot classification…"):
                    results = run_text_classification(cl_txt, lbls)
                st.markdown(
                    f'<div class="success-box"> Best: <strong>{results[0]["Label"]}</strong> ({results[0]["Confidence"]})</div>',
                    unsafe_allow_html=True,
                )
                import plotly.express as px
                df_zs = pd.DataFrame(results)
                fig = px.bar(df_zs, x="Label", y="Score", template="plotly_dark",
                             color="Score", color_continuous_scale="Purples")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

    # ── Summarization ──
    with nlp_tabs[3]:
        st.subheader("Text Summarization - DistilBART")
        long_txt = st.text_area("Long text to summarize:", height=220, max_chars=MAX_TEXT_CHARS,
            value=(
                "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
                "the natural intelligence displayed by animals including humans. AI research has been defined "
                "as the field of study of intelligent agents, which refers to any system that perceives its "
                "environment and takes actions that maximize its chance of achieving its goals. AI applications "
                "include advanced web search engines, recommendation systems, understanding human speech, "
                "self-driving cars, generative or creative tools, automated decision-making, and competing "
                "at the highest level in strategic game systems."
            ))
        if st.button(" Summarize", type="primary", key="sum_btn"):
            if len(long_txt.split()) < 30:
                st.warning("Need at least 30 words.")
            else:
                with st.spinner("Loading DistilBART and summarizing…"):
                    summary = run_summarization(long_txt)
                st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                sc1.metric("Original Words", len(long_txt.split()))
                sc2.metric("Summary Words",  len(summary.split()))

    # ── Chatbot ──
    with nlp_tabs[4]:
        st.subheader(" AI Chatbot")
        if "chat_pairs" not in st.session_state:
            st.session_state.chat_pairs = []

        with st.expander(" Settings"):
            sys_hint = st.text_input("System hint:", value="You are a helpful AI assistant. Be concise.", max_chars=500)
            if st.button(" Clear Chat"):
                st.session_state.chat_pairs = []
                st.rerun()

        for um, bm in st.session_state.chat_pairs:
            with st.chat_message("user"):
                st.markdown(um)
            with st.chat_message("assistant"):
                st.markdown(bm)

        user_input = st.chat_input("Ask anything…")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        prompt = f"{sys_hint}\n\n{user_input}" if sys_hint else user_input
                        resp   = chat_with_model(prompt, st.session_state.chat_pairs)
                    except Exception:
                        resp = "The chat service is temporarily unavailable. Try again later."
                st.markdown(resp)
            st.session_state.chat_pairs.append((user_input, resp))

        if not st.session_state.chat_pairs:
            examples = [
                "What is machine learning?", "Explain neural networks simply.",
                "Top 3 AI programming languages?", "Benefits of deep learning?",
            ]
            ecols = st.columns(2)
            for i, ex in enumerate(examples):
                with ecols[i % 2]:
                    if st.button(ex, key=f"ex_{i}", width="stretch"):
                        with st.spinner("Thinking…"):
                            try:
                                resp = chat_with_model(ex, [])
                            except Exception:
                                resp = "The chat service is temporarily unavailable. Try again later."
                        st.session_state.chat_pairs.append((ex, resp))
                        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 5 · GENERATIVE AI
# ──────────────────────────────────────────────────────────────────────────────
with tab_genai:
    st.markdown('<div class="sec-head"> Generative AI Suite</div>', unsafe_allow_html=True)
    gen_ai = st.session_state.gen_ai

    # ── Status banner ──
    if gen_ai._provider == "smart":
        st.markdown("""
        <div class="success-box">
         <strong>Smart AI active</strong> - instant built-in responses, zero downloads, no API key needed.<br>
        Knows: ML algorithms, deep learning, NLP, Python, data science, and more.
        For open-ended GPT-4 quality responses, add an API key in the sidebar.
        </div>""", unsafe_allow_html=True)
    elif gen_ai.is_available():
        st.markdown(f'<div class="success-box"> <strong>{gen_ai._provider_config["name"]}</strong> connected and ready.</div>', unsafe_allow_html=True)
    elif not gen_ai.pkg_installed():
        st.markdown(f"""
        <div class="info-box">
         <strong>{gen_ai._provider_config["name"]}</strong> package not installed.<br>
        Run: <code>{gen_ai.install_cmd()}</code> - then restart Streamlit.
        </div>""", unsafe_allow_html=True)
    else:
        pkg_status = {
            " OpenAI":    " installed" if OPENAI_OK    else "  pip install openai",
            " Google":    " installed" if GOOGLE_OK    else "  pip install google-generativeai",
            " Anthropic": " installed" if ANTHROPIC_OK else "  pip install anthropic",
        }
        rows = "<br>".join(f"&nbsp;&nbsp;{k}: <code>{v}</code>" for k, v in pkg_status.items())
        st.markdown(f"""
        <div class="info-box">
         No API key entered - using Smart AI mode.<br>
        To enable full LLM responses, install a package and add your key:<br><br>{rows}
        </div>""", unsafe_allow_html=True)

    gen_tabs = st.tabs([" Chatbot", " Data Q&A", " Code Gen", " Image Gen", " Report"])

    # ── Chatbot ──
    with gen_tabs[0]:
        st.subheader("Context-Aware Chatbot")
        if "gen_chat" not in st.session_state:
            st.session_state.gen_chat    = []
            st.session_state.gen_history = []

        for role, msg in st.session_state.gen_chat:
            with st.chat_message(role):
                st.markdown(msg)

        user_q = st.chat_input("Chat with AI…", max_chars=MAX_TEXT_CHARS, key="gen_chat_input")
        if user_q:
            st.session_state.gen_chat.append(("user", user_q))
            st.session_state.gen_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Generating…"):
                    try:
                        resp = gen_ai.chat(st.session_state.gen_history)
                    except Exception:
                        resp = "The assistant is temporarily unavailable. Try again later."
                st.markdown(resp)
            st.session_state.gen_chat.append(("assistant", resp))
            st.session_state.gen_history.append({"role": "assistant", "content": resp})

        if st.button(" Clear Conversation", key="clr_gen"):
            st.session_state.gen_chat    = []
            st.session_state.gen_history = []
            st.rerun()

    # ── Data Q&A ──
    with gen_tabs[1]:
        st.subheader("Prompt-Based Data Q&A")
        st.markdown("Ask questions about your loaded dataset and get AI-generated insights.")
        gen_option = st.selectbox("Analysis Type", [
            "General Insights", "Trends Analysis", "Anomaly Detection", "Recommendations", "Custom Question",
        ])
        analysis_map = {
            "General Insights":  "general",
            "Trends Analysis":   "trends_analysis",
            "Anomaly Detection": "anomaly_detection",
            "Recommendations":   "recommendations",
        }
        if gen_option != "Custom Question":
            if st.button(" Generate Insights", type="primary", key="gen_ins"):
                if st.session_state.data_summary:
                    with st.spinner("Generating…"):
                        insights = gen_ai.generate_insights(
                            st.session_state.data_summary, analysis_map[gen_option]
                        )
                    st.markdown(f'<div class="result-box">{insights}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Load data in the  Data tab first.")
        else:
            question = st.text_input("Your question:", placeholder="What features are most correlated with the target?")
            if st.button(" Ask", type="primary", key="gen_qa") and question:
                ctx = json.dumps(st.session_state.data_summary, default=str) if st.session_state.data_summary else None
                with st.spinner("Thinking…"):
                    answer = gen_ai.answer_question(question, ctx)
                st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)

    # ── Code Generation ──
    with gen_tabs[2]:
        st.subheader(" Code Generation")
        code_lang   = st.selectbox("Language", ["Python", "JavaScript", "SQL", "Bash", "R", "TypeScript"])
        code_prompt = st.text_area(
            "Describe what you want to code:", height=110,
            max_chars=MAX_TEXT_CHARS,
            placeholder="Write a Python function to clean a pandas dataframe by removing duplicates and filling nulls with median values.",
        )
        if st.button(" Generate Code", type="primary", key="code_gen"):
            if code_prompt.strip():
                with st.spinner("Generating code…"):
                    code_result = gen_ai.generate_code(code_prompt, language=code_lang)
                st.code(code_result, language=code_lang.lower())
                st.download_button(
                    "⬇ Download", code_result.encode(),
                    file_name=f"generated.{code_lang.lower()[:2]}",
                    mime="text/plain",
                )
            else:
                st.warning("Enter a code description.")

    # ── Image Generation ──
    with gen_tabs[3]:
        st.subheader(" Image Generation from Text")
        img_prompt = st.text_area(
            "Image description:", height=100,
            max_chars=MAX_TEXT_CHARS,
            placeholder="A futuristic city at night with neon lights and flying cars, digital art style",
        )
        img_style = st.selectbox("Style", ["Photorealistic", "Digital Art", "Oil Painting", "Anime", "Sketch", "Cyberpunk"])
        img_size  = st.selectbox("Size", ["1024x1024", "512x512", "1792x1024"])

        if st.button(" Generate Image", type="primary", key="img_gen"):
            if img_prompt.strip():
                full_prompt = f"{img_prompt}, {img_style} style"
                with st.spinner("Generating image…"):
                    result = gen_ai.generate_image(full_prompt, size=img_size)
                if result.get("url"):
                    st.image(result["url"], caption=f"Generated: {img_prompt[:60]}…", width="stretch")
                    st.markdown(f"[ Open full size]({result['url']})")
                elif result.get("error"):
                    st.markdown('<div class="info-box">Image generation is unavailable. Check the provider settings and try again.</div>', unsafe_allow_html=True)
            else:
                st.warning("Enter an image description.")

    # ── Report ──
    with gen_tabs[4]:
        st.subheader(" Auto-Generated Analysis Report")
        if st.button(" Generate Report", type="primary", key="gen_rep"):
            payload = {
                "ml_metrics":   st.session_state.ml_metrics or {},
                "data_summary": st.session_state.data_summary or {},
                "model":        st.session_state.ml_results.get("model_name", "N/A") if st.session_state.ml_results else "N/A",
            }
            with st.spinner("Writing report…"):
                report = gen_ai.generate_report(payload)
            st.text_area("Report", report, height=420)
            st.download_button("⬇ Download Report", report.encode(), file_name="ai_report.txt", mime="text/plain")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 6 · POWER BI
# ──────────────────────────────────────────────────────────────────────────────
with tab_pbi:
    st.markdown('<div class="sec-head"> Power BI Export</div>', unsafe_allow_html=True)
    exporter = st.session_state.powerbi_exp

    if st.session_state.df is None:
        st.markdown('<div class="info-box"> Load data in the <strong> Data</strong> tab first.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.df
        ec1, ec2 = st.columns(2)
        with ec1:
            include_parquet = st.checkbox("Include Parquet files", value=True)
            export_name     = st.text_input("Dataset name", value="main_data", max_chars=64)
        with ec2:
            st.markdown("**Available datasets:**")
            available = {"Main Data": df}
            if st.session_state.ml_results and st.session_state.ml_results.get("feature_importance"):
                available["Feature Importance"] = pd.DataFrame(st.session_state.ml_results["feature_importance"])
            for name in available:
                st.markdown(f"• {name}")

        if st.button(" Export All for Power BI", type="primary", width="stretch"):
            export_name = export_name.strip()
            if not valid_export_name(export_name):
                st.error("Use 1 to 64 letters, numbers, underscores, or hyphens. Start with a letter or number.")
                st.stop()
            with st.spinner("Exporting…"):
                try:
                    named = {export_name: df}
                    if "Feature Importance" in available:
                        named["feature_importance"] = available["Feature Importance"]
                    paths = exporter.export_all(named, include_parquet=include_parquet)
                    st.success(f" Exported **{len(paths)}** files to `{OUTPUT_DIR}`")
                    for p in paths:
                        st.markdown(f"  • `{p.name}`")
                except Exception:
                    st.error("Export failed. Check the dataset and try again.")

        st.divider()
        st.info(exporter.generate_powerbi_instructions())

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                               file_name=f"{export_name}.csv", mime="text/csv", width="stretch")
        with dl2:
            if st.session_state.ml_results and st.session_state.ml_results.get("feature_importance"):
                fi_df = pd.DataFrame(st.session_state.ml_results["feature_importance"])
                st.download_button("⬇ Feature Importance CSV", fi_df.to_csv(index=False).encode(),
                                   file_name="feature_importance.csv", mime="text/csv", width="stretch")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Multi-AI Analytics Platform &nbsp;·&nbsp; ML · DL · NLP · GenAI · Power BI
    <div><a href="?page=privacy">Privacy</a> · <a href="?page=terms">Terms</a></div>
</div>
""", unsafe_allow_html=True)
