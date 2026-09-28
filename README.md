# Multi-AI Analytics Platform

Machine learning, deep learning, text analysis, and Power BI exports in a Streamlit app.

---

## Quick Start

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

Open the deployed app: **https://multi-ai-analytics-platform-sl68.onrender.com/**

---

## Project Structure

```
multi_ai_platform_v2/
├── app.py                        ← Main Streamlit app
├── config.py                     ← Config & paths
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── ml_models.py              ← MLPipeline, XGBoost, LightGBM, Ensemble
│   ├── dl_module.py              ← Image classification + OpenCV (from project 2.0)
│   ├── nlp_module.py             ← NLP pipelines (from project 2.0)
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

## UI

The app uses a dark, neutral interface with restrained accent colors.

---

## Modules

| Tab | Features |
|---|---|
| Home | App overview and navigation |
| Data | Upload, summary, statistics, and charts |
| ML Pipeline | Model training and evaluation |
| Deep Learning | Image classification and OpenCV tools |
| NLP | Sentiment, entities, classification, and summarization |
| Power BI | CSV and Parquet exports |

---

## Minimal Install (no deep learning)

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm \
            plotly matplotlib seaborn pillow opencv-python-headless
```

---

*Multi-AI Analytics Platform*
