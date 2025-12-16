from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.integrate import odeint

from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")  # Force non-GUI backend for FastAPI threads
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64


app = FastAPI()

logger = logging.getLogger("uvicorn.error")
BASE_DIR = Path(__file__).resolve().parent

# Allow your HTML file to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local dev; tighten later for deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

LABELS = ["Negative", "Neutral", "Positive"]

# --- Explanation helper ---
def explain_simulation(label: str) -> str:
    if label == "Negative":
        return (
            "The simulation suggests that negatively framed content spreads quickly at first but loses attention more rapidly over time."
        )
    elif label == "Positive":
        return (
            "The simulation suggests that positively framed content spreads more gradually and maintains engagement for a longer period."
        )
    else:
        return (
            "Neutral content is expected to generate limited diffusion and does not exhibit strong viral dynamics."
        )

# --- Load model once at startup ---
MODEL_DIR = BASE_DIR / "NLP_BERT_Model"  # folder next to server.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}. "
            "Place NLP_BERT_Model/ next to server.py, or update MODEL_DIR."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    logger.info(f"Loaded model from {MODEL_DIR}")
except Exception as e:
    # Fail fast with a clear log message; the server should not silently run without a model.
    logger.exception("Failed to load model/tokenizer")
    raise

def predict(text: str):
    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

# --- SEIR (use your paper params here) ---
N = 2_400_000_000
Follower = 28_451_349
num_users_1st_q = 28_451_348
num_users_4th_q = 28_110_772
aged_50_plus = 180_000_000

Likes_avg = 53566.65
Shares_avg = 1329.975

def get_params(sentiment_label: str):
    comments = 730.7 if sentiment_label == "Positive" else 1200.575
    u = (num_users_1st_q / num_users_4th_q) - 1
    a = (aged_50_plus / N) / 365
    b = (Follower - (Likes_avg + comments + Shares_avg)) / N
    y = (Likes_avg + comments + Shares_avg) / (N + Follower)
    d = Likes_avg / (N - Follower)
    v = 1 - b
    x = (Likes_avg + comments + Shares_avg) / N
    return (u/24, a/24, b/24, y/24, d/24, v/24, x/24)

def seir_rhs(state, t, mu, alpha, beta, gamma, delta, nu, chi):
    S, E, I, R = state
    total = S + E + I + R
    dS = mu - beta*S*I/total - alpha*S - nu*S
    dE = beta*S*I/total - alpha*E - gamma*E - chi*E
    dI = gamma*E - alpha*I - delta*I
    dR = delta*I + nu*S + chi*E - alpha*R
    return [dS, dE, dI, dR]

def simulate_seir_plot(label: str):
    import numpy as np
    import io
    import base64

    # Time axis
    t = np.linspace(0, 336, 300)

    # Dummy curve (replace with your real SEIR I(t) if you want)
    if label == "Positive":
        I = 0.15 * np.exp(-t / 120)
    else:
        I = 0.25 * np.exp(-t / 60)

    # Create figure WITHOUT pyplot
    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    ax.plot(t, I, color="red" if label == "Negative" else "green")
    ax.set_title(f"SEIR Virality Simulation ({label})")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Infected proportion")
    ax.grid(True)

    # Render to PNG buffer
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/predict")
def api_predict(req: PredictRequest):
    pred_idx, probs = predict(req.text)
    label = LABELS[pred_idx]

    img = None
    if label != "Neutral":
        img = simulate_seir_plot(label)

    explanation = explain_simulation(label)

    return {
        "label": label,
        "probs": probs,
        "img": img,
        "explanation": explanation
    }