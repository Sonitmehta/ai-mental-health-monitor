"""
AI Mental Health Monitor — FastAPI Backend
==========================================
Serves multi-page HTML/CSS/JS frontend and exposes ML inference API.
"""

import json, os, re, time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import torch
import numpy as np
import joblib

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import jwt, JWTError

# ─── Internal project imports ───────────────────────────────────────────────
from src.nlp_utils import extract_indicators, SimpleTokenizer
from src.model_utils import TextCNN1D, load_metadata

# ─── Config ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SECRET_KEY = "mhm-secret-key-2024-ai-mental-health"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

# Demo credentials (college project)
DEMO_USERS = {
    "admin@mhm.ai": "mhm2024",
    "student": "mhm2024",
}

# Session history stored in memory (persists while server is up)
_session_history: list[dict] = []

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI Mental Health Monitor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Model loading ───────────────────────────────────────────────────────────
_model: Optional[TextCNN1D]    = None
_tokenizer                     = None
_tfidf                         = None
_meta: Optional[dict]          = None
_risk_classes: list[str]       = ["low", "medium", "high"]
_emotion_classes: list[str]    = ["positive", "neutral", "anxious", "sad"]
_eval_metrics: Optional[dict]  = None
MODEL_LOADED = False

def load_model() -> bool:
    global _model, _tokenizer, _tfidf, _meta, _risk_classes, _emotion_classes, MODEL_LOADED
    try:
        _meta          = load_metadata(MODELS_DIR / "metadata.json")
        _risk_classes  = _meta.get("risk_classes",    ["low", "medium", "high"])
        _emotion_classes = _meta.get("emotion_classes", ["positive", "neutral", "anxious", "sad"])
        _tokenizer     = joblib.load(str(MODELS_DIR / "tokenizer.joblib"))
        _tfidf         = joblib.load(str(MODELS_DIR / "tfidf_vectorizer.joblib"))
        _model         = TextCNN1D(
            vocab_size          = _meta["vocab_size"],
            embed_dim           = 64,
            feature_dim         = _meta["feature_dim"],
            num_risk_classes    = len(_risk_classes),
            num_emotion_classes = len(_emotion_classes),
        )
        _model.load_state_dict(
            torch.load(str(MODELS_DIR / "cnn_model.pt"), map_location="cpu")
        )
        _model.eval()
        MODEL_LOADED = True
        return True
    except Exception as exc:
        print(f"[WARNING] Model load failed: {exc} — running in demo mode")
        MODEL_LOADED = False
        return False

# Load at startup
load_model()

def load_eval_metrics() -> dict:
    global _eval_metrics
    if _eval_metrics is None:
        path = MODELS_DIR / "evaluation_metrics.json"
        if path.exists():
            with open(path) as f:
                _eval_metrics = json.load(f)
        else:
            _eval_metrics = {}
    return _eval_metrics

# ─── JWT helpers ─────────────────────────────────────────────────────────────
def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("mhm_token")
    if not token:
        return None
    return verify_token(token)

def require_auth(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user

# ─── Inference ───────────────────────────────────────────────────────────────
MAX_LEN = 45

HIGH_CRISIS_WORDS = re.compile(
    r'\b(suicid|kill myself|end my life|want to die|no reason to live|'
    r'hopeless|worthless|give up on life|self.harm|cutting myself|'
    r'crying|can\'t take it|breakdown|失望|絶望)\b',
    re.IGNORECASE
)
MEDIUM_WORDS = re.compile(
    r'\b(anxious|anxiety|panic|stressed|overwhelmed|exhausted|'
    r'lonely|isolated|depressed|sad|failure|failed|empty|numb|'
    r'struggling|can\'t sleep|insomnia|irritable|mood swings)\b',
    re.IGNORECASE
)

def run_inference(text: str) -> dict:
    indicators = extract_indicators(text)
    high_count   = len(indicators.get("high_risk", []))
    medium_count = len(indicators.get("medium_risk", []))

    # Demo mode fallback
    if not MODEL_LOADED:
        if high_count >= 2:
            risk, emotion, score = "high",   "sad",     88
            probs_risk    = {"low": 0.05, "medium": 0.15, "high": 0.80}
            probs_emotion = {"positive": 0.02, "neutral": 0.08, "anxious": 0.30, "sad": 0.60}
        elif high_count == 1 or medium_count >= 2:
            risk, emotion, score = "medium", "anxious", 55
            probs_risk    = {"low": 0.15, "medium": 0.65, "high": 0.20}
            probs_emotion = {"positive": 0.05, "neutral": 0.20, "anxious": 0.55, "sad": 0.20}
        else:
            risk, emotion, score = "low",    "positive", 18
            probs_risk    = {"low": 0.75, "medium": 0.20, "high": 0.05}
            probs_emotion = {"positive": 0.70, "neutral": 0.20, "anxious": 0.08, "sad": 0.02}
        return dict(risk=risk, emotion=emotion, risk_score=score,
                    probs_risk=probs_risk, probs_emotion=probs_emotion,
                    indicators=indicators, model_loaded=False)

    # Real inference
    seq = _tokenizer.texts_to_sequences([text])
    padded = np.zeros((1, MAX_LEN), dtype=np.int64)
    t = seq[0][:MAX_LEN]
    padded[0, :len(t)] = t
    x_seq  = torch.tensor(padded, dtype=torch.long)

    feat   = _tfidf.transform([text]).toarray().astype(np.float32)
    x_feat = torch.tensor(feat, dtype=torch.float32)

    with torch.no_grad():
        r_logit, e_logit = _model(x_seq, x_feat)
        r_prob = torch.softmax(r_logit, dim=1).numpy()[0]
        e_prob = torch.softmax(e_logit, dim=1).numpy()[0]

    probs_risk    = {c: float(r_prob[i]) for i, c in enumerate(_risk_classes)}
    probs_emotion = {c: float(e_prob[i]) for i, c in enumerate(_emotion_classes)}

    # Clinical calibration: boost high-risk if crisis words detected
    if high_count >= 1:
        excess = max(0.0, 0.75 - probs_risk.get("high", 0))
        probs_risk["high"]   = max(probs_risk.get("high", 0) + excess, 0.75)
        probs_risk["low"]    = min(probs_risk.get("low",  0), 0.05)
        total = sum(probs_risk.values())
        probs_risk = {k: v / total for k, v in probs_risk.items()}

    risk    = max(probs_risk,    key=probs_risk.get)
    emotion = max(probs_emotion, key=probs_emotion.get)
    score   = int(
        probs_risk.get("low",    0) * 15 +
        probs_risk.get("medium", 0) * 55 +
        probs_risk.get("high",   0) * 92
    )

    return dict(risk=risk, emotion=emotion, risk_score=score,
                probs_risk=probs_risk, probs_emotion=probs_emotion,
                indicators=indicators, model_loaded=True)

# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("mhm_token")
    return response

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    metrics = load_eval_metrics()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"user": user, "metrics": metrics, "model_loaded": MODEL_LOADED}
    )

@app.get("/screening", response_class=HTMLResponse)
async def screening_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="screening.html",
        context={"user": user}
    )

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"user": user}
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    metrics = load_eval_metrics()
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"user": user, "metrics": metrics}
    )

# ─── Auth endpoint ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def api_login(body: LoginRequest):
    stored = DEMO_USERS.get(body.username)
    if not stored or stored != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": body.username})
    response = JSONResponse({"success": True, "token": token, "user": body.username})
    response.set_cookie(
        "mhm_token", token,
        httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax"
    )
    return response

# ─── ML API endpoints ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

@app.post("/api/predict")
async def predict(body: PredictRequest, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not body.text or len(body.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text too short")

    result = run_inference(body.text.strip())

    # Save to session history
    entry = {
        "id":        len(_session_history) + 1,
        "timestamp": datetime.now().isoformat(),
        "text":      body.text[:200],
        "risk":      result["risk"],
        "emotion":   result["emotion"],
        "score":     result["risk_score"],
    }
    _session_history.append(entry)

    return JSONResponse(result)

@app.get("/api/metrics")
async def get_metrics(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse(load_eval_metrics())

@app.get("/api/history")
async def get_history(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({"history": _session_history})

@app.delete("/api/history")
async def clear_history(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    _session_history.clear()
    return JSONResponse({"success": True})

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

