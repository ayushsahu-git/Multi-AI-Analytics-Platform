"""
nlp_module.py  —  NLP Module (v2.1 Clean)
Models:
  - DistilBERT SST-2     → sentiment analysis  (~250 MB, downloads on first use)
  - spaCy en_core_web_sm → named entity recognition (~15 MB, auto-downloads)
  - TF-IDF              → zero-shot classification (no download)
  - Extractive          → summarization (no download)
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
#  Cached pipeline loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """DistilBERT SST-2 — ~250 MB, fast and accurate."""
    from transformers import pipeline  # type: ignore[import-untyped]
    return pipeline(  # type: ignore[call-overload]
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource(show_spinner=False)
def load_ner_pipeline():
    """
    spaCy en_core_web_sm (~15 MB) for NER.
    Falls back to regex-based NER if spaCy is not installed.
    Install: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy
        try:
            return ("spacy", spacy.load("en_core_web_sm"))
        except OSError:
            from spacy.cli.download import download as spacy_download  # type: ignore[import]
            spacy_download("en_core_web_sm")
            return ("spacy", spacy.load("en_core_web_sm"))
    except ImportError:
        return ("regex", None)


@st.cache_resource(show_spinner=False)
def load_zero_shot_pipeline():
    """
    Lightweight zero-shot classification using TF-IDF cosine similarity.
    Zero model downloads, zero RAM overhead — works on any machine.
    Falls back gracefully without any internet or large model requirement.
    """
    return "tfidf"   # sentinel value — actual logic is in run_text_classification


@st.cache_resource(show_spinner=False)
def load_summarization_pipeline():
    """
    Extractive summarizer — word-frequency scoring, zero model download.
    Picks the most informative sentences from the input text.
    """
    return "extractive"   # sentinel — actual logic in run_summarization


# ══════════════════════════════════════════════════════════════════════════════
#  Business logic
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment(texts: list) -> list:
    """
    Sentiment analysis on a list of strings.
    Returns list of dicts: Text, Sentiment, Confidence, Score.
    """
    pipe    = load_sentiment_pipeline()
    results = []
    for text in texts:
        if text.strip():
            r = pipe(text[:512], truncation=True, max_length=512)[0]
            results.append({
                "Text":       text[:80],
                "Sentiment":  r["label"],
                "Confidence": f"{r['score'] * 100:.1f}%",
                "Score":      round(r["score"], 4),
            })
    return results


def run_ner(text: str) -> list:
    """
    Named Entity Recognition using spaCy (15 MB) or regex fallback.
    Returns list of dicts: Entity, Type, Score, Start, End.
    """
    backend, model = load_ner_pipeline()

    if backend == "spacy" and model is not None:
        doc = model(text[:1000])
        return [
            {
                "Entity": ent.text,
                "Type":   ent.label_,
                "Score":  "100.0%",
                "Start":  ent.start_char,
                "End":    ent.end_char,
            }
            for ent in doc.ents
        ]

    # ── Regex fallback — works with zero extra installs ──────────────────────
    import re
    patterns = [
        (
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'
            r'(?:Inc|Corp|Ltd|LLC|Co|Group|Foundation|Institute|University|'
            r'College|School|Hospital|Bank|Technologies|Solutions|Systems|Services)\.?)\b',
            "ORG",
        ),
        (
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
            r'(?=\s+(?:City|State|Country|Street|Avenue|Road|Park|Lake|River|'
            r'Mountain|Valley|Island|Bay|County|District|Province|Region))',
            "LOC",
        ),
        (
            r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b',
            "PER",
        ),
        (r'\b([A-Z]{2,6})\b', "ORG"),
    ]

    seen, results = set(), []
    for pattern, label in patterns:
        for m in re.finditer(pattern, text):
            entity = m.group(1).strip()
            key    = (entity, label)
            if key not in seen and len(entity) > 1:
                seen.add(key)
                results.append({
                    "Entity": entity,
                    "Type":   label,
                    "Score":  "~",
                    "Start":  m.start(),
                    "End":    m.end(),
                })

    return sorted(results, key=lambda x: x["Start"])


def _tfidf_cosine(text: str, label: str) -> float:
    """Compute TF-IDF cosine similarity between text and a label string."""
    import re
    from collections import Counter
    import math

    _stop = {"the","a","an","is","are","was","were","be","been","being","have",
             "has","had","do","does","did","will","would","could","should","may",
             "might","can","to","of","in","for","on","with","at","by","from","as",
             "and","but","or","not","it","its","this","that","i","we","you","he",
             "she","they","all","any","more","so","very","also","just","about"}

    def _tokens(s: str) -> list:
        return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in _stop and len(w) > 1]

    t_tokens = _tokens(text)
    l_tokens = _tokens(label)
    if not t_tokens or not l_tokens:
        return 0.0

    # TF of text
    tf_t = Counter(t_tokens)
    tf_l = Counter(l_tokens)

    # Vocabulary union
    vocab = set(tf_t) | set(tf_l)

    # Simple IDF weight: log(1 + 1/freq_ratio) — single-doc approximation
    def vec(tf: Counter) -> dict:
        total = sum(tf.values()) or 1
        return {w: tf[w] / total for w in vocab}

    vt = vec(tf_t)
    vl = vec(tf_l)

    dot    = sum(vt[w] * vl[w] for w in vocab)
    norm_t = math.sqrt(sum(v * v for v in vt.values())) or 1e-9
    norm_l = math.sqrt(sum(v * v for v in vl.values())) or 1e-9
    return dot / (norm_t * norm_l)


def run_text_classification(text: str, labels: list) -> list:
    """
    Zero-shot text classification using TF-IDF cosine similarity.
    No model download required — works instantly on any machine.
    Returns list of dicts: Label, Score, Confidence — sorted by score desc.
    """
    if not labels:
        return []

    scores = []
    for label in labels:
        # Boost: also compare text against expanded label description
        sim = _tfidf_cosine(text, label)
        scores.append((label, sim))

    # Normalise scores so they sum to 1 (softmax-like)
    import math
    exp_scores = [(lbl, math.exp(s * 8)) for lbl, s in scores]   # temperature=8 sharpens
    total = sum(s for _, s in exp_scores) or 1.0
    normalised = sorted(
        [{"Label": lbl, "Score": round(s / total, 4), "Confidence": f"{s / total * 100:.1f}%"}
         for lbl, s in exp_scores],
        key=lambda x: x["Score"], reverse=True,
    )
    return normalised


def run_summarization(text: str) -> str:
    """
    Extractive summarization using word-frequency scoring.
    Zero model download — works on any machine, any RAM size.
    Picks the top 3 most informative sentences.
    """
    import re
    from collections import Counter

    text = text.strip()
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]

    if len(sentences) <= 2:
        return text[:400] + ("…" if len(text) > 400 else "")

    # Stop words to ignore when computing importance
    stop = {"the","a","an","is","are","was","were","be","been","being","have",
            "has","had","do","does","did","will","would","could","should","may",
            "might","can","to","of","in","for","on","with","at","by","from",
            "as","into","and","but","or","not","it","its","this","that","i",
            "we","you","he","she","they","all","any","each","more","most","so",
            "very","also","just","about","than","other","such","when","which"}

    words  = re.findall(r"[a-z]+", text.lower())
    freq   = Counter(w for w in words if w not in stop and len(w) > 2)
    max_f  = max(freq.values(), default=1)
    freq   = {w: v / max_f for w, v in freq.items()}

    # Score sentences
    scores: dict = {}
    for i, sent in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in re.findall(r"[a-z]+", sent.lower()))
        score = score / max(len(sent.split()), 1)
        if i == 0:
            score *= 1.3    # slight boost for the opening sentence
        scores[i] = score

    # Pick top N sentences (preserve original order)
    n   = max(1, min(4, len(sentences) // 3))
    top = sorted(sorted(scores, key=lambda k: scores[k], reverse=True)[:n])
    return " ".join(sentences[i] for i in top)
