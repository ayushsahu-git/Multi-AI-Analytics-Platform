# ⚡ Multi-AI Analytics Platform — v2.0 Merged Edition

> ML · Deep Learning · NLP · Generative AI · Power BI — All in one Streamlit app

---

## 🚀 Quick Start

```bash
# 1. Clone or download the project
cd multi_ai_platform_v2

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open → **https://multi-ai-analytics-platform-sl68.onrender.com**

---

## 📁 Project Structure

```
multi_ai_platform_v2/
├── app.py                        ← Main Streamlit app (Aurora Dark UI)
├── config.py                     ← Config & paths
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── ml_models.py              ← MLPipeline, XGBoost, LightGBM, Ensemble
│   ├── dl_module.py              ← Image classification + OpenCV (from project 2.0)
│   ├── nlp_module.py             ← NLP pipelines (from project 2.0)
│   └── generative_ai.py         ← NEW: Chatbot, Q&A, Code Gen, Image Gen, Reports
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py            ← DataLoader class
│   └── powerbi_export.py         ← PowerBIExporter class
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                ← Plotly chart generators
│
└── output/                       ← Auto-created: exported CSVs, Parquets
```

---

## 🎨 UI Design

**Aurora Dark Theme** — trending 2025 palette:
- Deep space background with aurora gradients
- `#a78bfa` Violet · `#60a5fa` Blue · `#2dd4bf` Teal accents
- Font: **Outfit** (UI) + **JetBrains Mono** (code)
- Glassmorphism cards with glow effects
- Smooth hover transitions

---

## 🧩 Modules

| Tab | Module | Source |
|---|---|---|
| 🏠 Home | Overview dashboard | New |
| 📊 Data | Upload + EDA + charts | Project 1 |
| 🤖 ML Pipeline | 9+ models, metrics, feature importance | Project 1 (`ml_models.py`) |
| 🧠 Deep Learning | CNN classification, Grad-CAM, OpenCV | Project 2.0 (`dl_module.py`) |
| 📝 NLP | Sentiment, NER, classification, summarization, chatbot | Project 2.0 (`nlp_module.py`) |
| 💡 Generative AI | Chatbot, Q&A, Code Gen, Image Gen, Reports | **Brand New** |
| 📤 Power BI | CSV/Parquet export | Project 1 |

---

## 💡 Generative AI Features (New)

### 1. Context-Aware Chatbot
Full multi-turn conversation with memory. Supports OpenAI, Gemini, Claude.

### 2. Data Q&A
Ask natural language questions about your loaded dataset. AI reads your data summary and answers.

### 3. Code Generation
Generate production-ready code in Python, JavaScript, SQL, Bash, R, TypeScript.

### 4. Image Generation
Generate images from text prompts using **DALL-E 3** (requires OpenAI key).

### 5. Auto Report
Generate a full markdown analysis report based on your ML results and dataset.

> **No API key?** All features work in offline mode with smart fallback responses.

---

## 🔑 API Keys (Optional)

Add in the sidebar at runtime — never hardcoded.

| Provider | Get Key |
|---|---|
| OpenAI | https://platform.openai.com/api-keys |
| Google Gemini | https://aistudio.google.com/app/apikey |
| Anthropic Claude | https://console.anthropic.com/settings/api-keys |

---

## 📦 Minimal Install (no deep learning)

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm \
            plotly matplotlib seaborn pillow opencv-python-headless \
            openai google-generativeai anthropic
```

---

*Built with Ayush — Multi-AI Analytics Platform v2.0 | KYOTO-Z*
