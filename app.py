"""
AI Mental Health Monitor — FastAPI Backend
==========================================
Serves multi-page HTML/CSS/JS frontend, user authentication (Sign In / Sign Up),
user-isolated screening history, and ML inference API with actionable clinical guidance.
"""

import json, os, re, time, hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta

import torch
import numpy as np
import joblib

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError

# ─── Internal project imports ───────────────────────────────────────────────
from src.nlp_utils import extract_indicators, SimpleTokenizer
from src.model_utils import TextCNN1D, load_metadata

# ─── Config & Storage Paths ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "models"
DATA_DIR    = BASE_DIR / "data"
USERS_FILE  = DATA_DIR / "users.json"
HISTORY_FILE= DATA_DIR / "user_history.json"

SECRET_KEY  = "mhm-secret-key-2024-ai-mental-health"
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── User & History Database Helpers ────────────────────────────────────────

def _hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def _load_users() -> dict:
    if not USERS_FILE.exists():
        default_users = {
            "student": {
                "name": "Demo Student",
                "email": "student@mhm.ai",
                "password_hash": _hash_pwd("mhm2024"),
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            "admin@mhm.ai": {
                "name": "Project Lead",
                "email": "admin@mhm.ai",
                "password_hash": _hash_pwd("mhm2024"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        USERS_FILE.write_text(json.dumps(default_users, indent=2), encoding="utf-8")
        return default_users
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def _load_user_history() -> dict:
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(json.dumps({}, indent=2), encoding="utf-8")
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_user_history(hist: dict):
    HISTORY_FILE.write_text(json.dumps(hist, indent=2), encoding="utf-8")

# ─── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI Mental Health Monitor", version="2.5.0")

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
        _meta            = load_metadata(MODELS_DIR / "metadata.json")
        _risk_classes    = _meta.get("risk_classes",    ["low", "medium", "high"])
        _emotion_classes = _meta.get("emotion_classes", ["positive", "neutral", "anxious", "sad"])
        _tokenizer       = joblib.load(str(MODELS_DIR / "tokenizer.joblib"))
        _tfidf           = joblib.load(str(MODELS_DIR / "tfidf_vectorizer.joblib"))
        _model           = TextCNN1D(
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

# Initialize model
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

# ─── JWT Helpers ─────────────────────────────────────────────────────────────
def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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

# ─── Actionable Guidance & Clinical Recommendations ─────────────────────────

def generate_clinical_guidance(risk: str, emotion: str, score: int, indicators: dict) -> dict:
    """
    Generates tailored, actionable steps and verified official helplines
    based on the evaluated risk score and emotional state.
    """
    helplines = [
        {
            "name": "Tele-MANAS (Govt. of India 24/7 Mental Health Helpline)",
            "number": "14416 / 1800-891-4416",
            "tel": "tel:14416",
            "desc": "Toll-free 24/7 tele-counseling in 20+ regional languages."
        },
        {
            "name": "KIRAN National Helpline (Ministry of Social Justice)",
            "number": "1800-599-0019",
            "tel": "tel:18005990019",
            "desc": "Free 24/7 psychological support and distress management."
        },
        {
            "name": "Vandrevala Foundation (Free Counseling & Crisis Support)",
            "number": "+91 9999 666 555",
            "tel": "tel:+919999666555",
            "desc": "24/7 professional mental health counselors on call and WhatsApp."
        },
        {
            "name": "International Crisis Lifeline (US & Canada)",
            "number": "988",
            "tel": "tel:988",
            "desc": "National suicide and mental health crisis hotline."
        }
    ]

    if risk == "high" or score >= 70:
        return {
            "level": "high",
            "title": "Immediate Care & Grounding Guidance",
            "summary": "Your reflection shows signs of intense emotional distress or despair. Please know that whatever you are carrying, you don't have to carry it alone.",
            "action_steps": [
                "🛑 Pause & Breathe: Try the 4-7-8 Breathing technique (Inhale for 4s, Hold for 7s, Exhale slowly for 8s) to calm your nervous system.",
                "🛋️ Move to a safe, comfortable physical space and sip a glass of cold water.",
                "💬 Reach out to someone you trust — a close friend, family member, or colleague. Tell them: 'I am having a tough time today and need someone to listen.'",
                "📞 Connect immediately with a trained counselor using one of the 24/7 free helplines below. They are confidential, kind, and ready to support you right now."
            ],
            "lifestyle_tip": "Do not judge yourself for feeling overwhelmed. Your feelings are valid, and seeking support is a sign of immense strength, not failure.",
            "helplines": helplines,
            "show_emergency_banner": True
        }
    elif risk == "medium" or score >= 40:
        return {
            "level": "medium",
            "title": "Proactive Stress Relief & Mindfulness Plan",
            "summary": "You are experiencing moderate stress, worry, or emotional fatigue. Taking small proactive steps today can help prevent burnout and restore balance.",
            "action_steps": [
                "🧘 5-4-3-2-1 Grounding: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "✍️ Thought Offloading: Write down the top 2 things causing you anxiety on paper, and write one small actionable step you can take for each.",
                "🚶 Physical Reset: Take a 15-minute walk outside without screens, or stretch your shoulders and neck.",
                "😴 Sleep Hygiene: Avoid phone screens 45 minutes before sleep and aim for consistent rest tonight."
            ],
            "lifestyle_tip": "Break large overwhelming tasks into 10-minute micro-tasks. Progress over perfection!",
            "helplines": helplines[:2],
            "show_emergency_banner": False
        }
    else:
        return {
            "level": "low",
            "title": "Wellness Maintenance & Positive Reinforcement",
            "summary": "Your entry reflects healthy emotional equilibrium and positive resilience. Maintaining these habits will support long-term mental clarity.",
            "action_steps": [
                "✨ Gratitude Habit: Take 30 seconds to acknowledge one thing that brought you joy or contentment today.",
                "💧 Hydration & Movement: Keep your body energised with regular water intake and light physical activity.",
                "🤝 Social Connection: Send an encouraging message to a friend or spend meaningful time with loved ones.",
                "🎯 Mindful Reflection: Keep journaling periodically to track your emotional well-being over time."
            ],
            "lifestyle_tip": "Celebrate your peaceful moments. Mental health is not just about avoiding crisis, but cultivating everyday fulfillment!",
            "helplines": helplines[:1],
            "show_emergency_banner": False
        }

# ─── Inference ───────────────────────────────────────────────────────────────
MAX_LEN = 45

def run_inference(text: str) -> dict:
    indicators   = extract_indicators(text)
    high_count   = len(indicators.get("high_risk", []))
    medium_count = len(indicators.get("medium_risk", []))

    if not MODEL_LOADED:
        if high_count >= 2:
            risk, emotion, score = "high", "sad", 88
            probs_risk    = {"low": 0.05, "medium": 0.15, "high": 0.80}
            probs_emotion = {"positive": 0.02, "neutral": 0.08, "anxious": 0.30, "sad": 0.60}
        elif high_count == 1 or medium_count >= 2:
            risk, emotion, score = "medium", "anxious", 55
            probs_risk    = {"low": 0.15, "medium": 0.65, "high": 0.20}
            probs_emotion = {"positive": 0.05, "neutral": 0.20, "anxious": 0.55, "sad": 0.20}
        else:
            risk, emotion, score = "low", "positive", 18
            probs_risk    = {"low": 0.75, "medium": 0.20, "high": 0.05}
            probs_emotion = {"positive": 0.70, "neutral": 0.20, "anxious": 0.08, "sad": 0.02}
    else:
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

    guidance = generate_clinical_guidance(risk, emotion, score, indicators)

    return dict(
        risk=risk,
        emotion=emotion,
        risk_score=score,
        probs_risk=probs_risk,
        probs_emotion=probs_emotion,
        indicators=indicators,
        guidance=guidance,
        model_loaded=MODEL_LOADED
    )

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
    
    # Load user's recent screening status for compact dashboard
    hist_db = _load_user_history()
    user_key = user.get("sub", "student")
    user_records = hist_db.get(user_key, [])
    latest_record = user_records[-1] if user_records else None
    
    metrics = load_eval_metrics()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user": user,
            "metrics": metrics,
            "latest_record": latest_record,
            "total_screenings": len(user_records),
            "model_loaded": MODEL_LOADED
        }
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

# ─── Auth API Endpoints (Sign In & Sign Up) ──────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str

@app.post("/api/login")
async def api_login(body: LoginRequest):
    users = _load_users()
    key = body.username.strip().lower()
    
    # Match by key or by email
    user_record = users.get(key)
    if not user_record:
        for u_id, u_data in users.items():
            if u_data.get("email", "").lower() == key:
                user_record = u_data
                key = u_id
                break
    
    if not user_record or user_record.get("password_hash") != _hash_pwd(body.password):
        # Demo fallback for college demo ease
        if (key in ["student", "admin@mhm.ai"]) and body.password == "mhm2024":
            pass
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")

    user_name = user_record.get("name", key) if user_record else key
    token = create_token({"sub": key, "name": user_name})
    
    response = JSONResponse({"success": True, "token": token, "user": key, "name": user_name})
    response.set_cookie(
        "mhm_token", token,
        httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax"
    )
    return response

@app.post("/api/signup")
async def api_signup(body: SignUpRequest):
    name = body.name.strip()
    email = body.email.strip().lower()
    password = body.password.strip()

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if len(password) < 5:
        raise HTTPException(status_code=400, detail="Password must be at least 5 characters")

    users = _load_users()
    if email in users:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    username = email.split("@")[0]
    # Ensure unique key
    base_key = username
    counter = 1
    while username in users:
        username = f"{base_key}{counter}"
        counter += 1

    users[username] = {
        "name": name,
        "email": email,
        "password_hash": _hash_pwd(password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    _save_users(users)

    token = create_token({"sub": username, "name": name})
    response = JSONResponse({"success": True, "token": token, "user": username, "name": name})
    response.set_cookie(
        "mhm_token", token,
        httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax"
    )
    return response

# ─── ML & History API Endpoints ──────────────────────────────────────────────

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

    # Save to user-isolated persistent history
    user_key = user.get("sub", "student")
    hist_db = _load_user_history()
    if user_key not in hist_db:
        hist_db[user_key] = []

    entry = {
        "id": len(hist_db[user_key]) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": body.text[:250],
        "risk": result["risk"],
        "emotion": result["emotion"],
        "score": result["risk_score"],
        "guidance_summary": result["guidance"]["summary"]
    }
    hist_db[user_key].append(entry)
    _save_user_history(hist_db)

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
    
    user_key = user.get("sub", "student")
    hist_db = _load_user_history()
    user_records = hist_db.get(user_key, [])
    return JSONResponse({"history": user_records})

@app.delete("/api/history")
async def clear_history(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_key = user.get("sub", "student")
    hist_db = _load_user_history()
    hist_db[user_key] = []
    _save_user_history(hist_db)
    return JSONResponse({"success": True})

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
