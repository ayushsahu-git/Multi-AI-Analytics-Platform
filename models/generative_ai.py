"""
generative_ai.py - Generative AI Module
Supports OpenAI GPT, Google Gemini, Anthropic Claude, and Smart AI fallback
"""

import warnings
warnings.filterwarnings("ignore")

OPENAI_OK = False
GOOGLE_OK = False
ANTHROPIC_OK = False

try:
    import openai
    OPENAI_OK = True
except ImportError:
    pass

try:
    import google.generativeai as genai
    GOOGLE_OK = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    pass


def _smart_respond(prompt: str, history: list) -> str:
    """Instant smart AI response without API calls - keyword-based fallback."""
    p = prompt.lower()
    
    if any(w in p for w in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm your AI assistant. How can I help you today?"
    
    if "machine learning" in p or " ml " in p or "machine learning" in p:
        return (
            "**Machine Learning** enables systems to learn from data without explicit programming. "
            "Types: Supervised, Unsupervised, Reinforcement Learning. "
            "Popular libraries: scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow."
        )
    
    if "deep learning" in p or "neural network" in p or "cnn" in p:
        return (
            "**Deep Learning** uses multi-layer neural networks to learn complex patterns. "
            "Best for: images (CNNs), sequences (RNNs/LSTMs), Transformers. "
            "Frameworks: PyTorch, TensorFlow/Keras."
        )
    
    if "xgboost" in p or "gradient boosting" in p:
        return (
            "**XGBoost** builds trees sequentially, each correcting prior errors. "
            "Key parameters: n_estimators, max_depth, learning_rate, subsample. "
            "Extremely fast and accurate for tabular data."
        )
    
    if "lightgbm" in p:
        return (
            "**LightGBM** uses histogram-based gradient boosting for speed. "
            "Great for large datasets. Uses leaf-wise tree growth vs level-wise."
        )
    
    if "overfitting" in p or "underfitting" in p:
        return (
            "**Overfitting** = model memorizes training noise, fails on new data. "
            "Fixes: cross-validation, regularization (L1/L2), dropout, more data, simpler model. "
            "**Underfitting** = model too simple to capture patterns. Fixes: more features, complex model."
        )
    
    if "python" in p:
        return (
            "**Python** dominates AI/ML thanks to: NumPy, Pandas, scikit-learn, "
            "PyTorch, TensorFlow, HuggingFace Transformers. "
            "Use virtual environments (venv/conda) to manage dependencies."
        )
    
    if "nlp" in p or "natural language" in p or "text" in p:
        return (
            "**NLP** (Natural Language Processing) enables machines to understand text. "
            "Key tasks: sentiment analysis, NER, classification, summarization, translation. "
            "Modern approach: HuggingFace Transformers (BERT, GPT, T5), spaCy."
        )
    
    if "data" in p and ("clean" in p or "preprocess" in p):
        return (
            "**Data Preprocessing** steps: 1) Handle missing values (mean/median/mode), "
            "2) Encode categoricals (LabelEncoder, OneHot), 3) Scale numeric features, "
            "4) Remove outliers, 5) Feature engineering."
        )
    
    if "random forest" in p or "rf " in p:
        return (
            "**Random Forest** is an ensemble of decision trees. "
            "Uses bagging and random feature selection. "
            "Key params: n_estimators, max_depth, min_samples_split. "
            "Good for feature importance and handling missing values."
        )
    
    if "classification" in p:
        return (
            "**Classification** predicts categorical labels. "
            "Algorithms: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost. "
            "Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC."
        )
    
    if "regression" in p:
        return (
            "**Regression** predicts continuous values. "
            "Algorithms: Linear Regression, Ridge, Lasso, Random Forest, XGBoost. "
            "Metrics: MSE, RMSE, MAE, R² Score."
        )
    
    if "api" in p or "key" in p or "openai" in p or "gpt" in p:
        return (
            "To use GPT models, set OPENAI_API_KEY environment variable or pass api_key parameter. "
            "Get your key from https://platform.openai.com/api-keys"
        )
    
    if "help" in p or "what can you do" in p:
        return (
            "I can help with: Machine Learning, Deep Learning, NLP, Data Science, "
            "Python programming, XGBoost, scikit-learn, TensorFlow, PyTorch, "
            "model evaluation, and more! Ask me anything."
        )
    
    return (
        f"I understand you're asking about: '{prompt[:50]}...'. "
        "Try asking about: machine learning, neural networks, XGBoost, Python, "
        "NLP, data preprocessing, classification, regression, or specific algorithms!"
    )


class GenerativeAI:
    def __init__(self, api_key: str = "", provider: str = "smart"):
        self.api_key = api_key
        self.provider = provider
        self._provider = provider
        self._provider_config = self._get_provider_config(provider)
        self.client = None
        
        if provider == "openai" and OPENAI_OK and api_key:
            openai.api_key = api_key
            self.client = openai
        elif provider == "google" and GOOGLE_OK and api_key:
            genai.configure(api_key=api_key)
            self.client = genai
        elif provider == "anthropic" and ANTHROPIC_OK and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def _get_provider_config(self, provider: str) -> dict:
        configs = {
            "smart": {"name": "Smart AI", "status": "✅", "desc": "Instant responses - no API key needed"},
            "openai": {"name": "OpenAI GPT-4o", "status": "🟢" if OPENAI_OK else "❌", "desc": "Requires API key"},
            "google": {"name": "Google Gemini", "status": "🔵" if GOOGLE_OK else "❌", "desc": "Requires API key"},
            "anthropic": {"name": "Anthropic Claude", "status": "🟣" if ANTHROPIC_OK else "❌", "desc": "Requires API key"},
        }
        return configs.get(provider, configs["smart"])
    
    def generate(self, prompt: str, history: list = None) -> str:
        """Generate response based on provider."""
        if self.provider == "smart" or not self.client:
            return _smart_respond(prompt, history or [])
        
        try:
            if self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                if history:
                    for h in history:
                        messages.append(h)
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                return response.choices[0].message.content
            
            elif self.provider == "google":
                model = self.client.GenerativeModel("gemini-pro")
                chat = model.start_chat(history=[])
                response = chat.send_message(prompt)
                return response.text
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
        except Exception:
            return "The selected AI provider is temporarily unavailable. The local response mode will be used instead.\n\n" + _smart_respond(prompt, history or [])
        
        return _smart_respond(prompt, history or [])
