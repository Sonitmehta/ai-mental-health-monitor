# 🧠 MindScan AI — AI-Powered Mental Health Screening System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-modal clinical mental health monitoring application built with **FastAPI**, **Vanilla HTML5/CSS3/JavaScript (ES6+)**, and **PyTorch**. The system integrates a **Dual-Branch TextCNN1D** with **Feature-Space GAN Augmentation** to classify emotional distress and triage clinical risk levels from natural language self-reflections.

---

## 🌟 Key Features

- ☀️🌙 **Adaptive Light & Dark Themes:** Clean, calming light mode by default with an instant 1-click Sun/Moon toggle in the navbar.
- 💬 **Bilingual & Hinglish NLP Support:** Custom tokenizer and clinical lexicons engineered to understand conversational English, Devanagari Hindi, and mixed **Hinglish** (e.g. *"headache ho rha h"*, *"bohot dard ho rha"*).
- 🩺 **Somatic Distress & Physical Pain Detection:** Recognizes physical manifestations of mental distress (e.g., spinal aches, muscle tension, migraines, exhaustion) as clinical markers.
- 🔄 **Continuous Learning & Active Retraining:** Built-in retraining pipeline (`/api/retrain`) that updates neural network weights and re-synthesizes GAN latent vectors on the fly without server downtime.
- 💡 **Actionable Coping Recommendations:** Provides evidence-based relaxation steps (4-7-8 breathing, 5-4-3-2-1 grounding, thought offloading) for every screening.
- 📞 **Verified 24/7 Helplines:** Integrated official toll-free crisis numbers (Tele-MANAS `14416`, KIRAN `1800-599-0019`, Vandrevala `+91 9999 666 555`, 988) with 1-click direct calling.
- 🔐 **Multi-User Sign-Up & Private History:** Complete user registration with isolated personal history logs and 1-Click Instant Demo Login.
- ⚡ **1-Click Local Launcher:** Includes `run_app.bat` for instant startup on Windows without terminal commands.
- 🚀 **Cloud Deployment Ready:** Configured with `Procfile` for 1-click hosting on Render.com or Railway.

---

## 🏗️ Project Architecture

```
├── app.py                      # FastAPI web controller & API routes
├── Procfile                    # Render / Railway production startup command
├── run_app.bat                 # 1-Click Windows launch script
├── requirements.txt            # Isolated dependency specification
├── LICENSE                     # MIT Open-Source License
├── data/
│   └── sample_dataset.csv      # 188 balanced clinical & Hinglish screening entries
├── models/
│   ├── cnn_model.pt            # Frozen weights for TextCNN1D
│   ├── gan_generator.pt        # Feature-space GAN Generator weights
│   ├── tokenizer.joblib        # Fitted SimpleTokenizer
│   ├── tfidf_vectorizer.joblib # 250-feature TF-IDF Vectorizer
│   └── metadata.json           # Model configuration parameters
├── src/
│   ├── nlp_utils.py            # Tokenizer, Hinglish & Somatic biomarker extractor
│   ├── model_utils.py          # PyTorch TextCNN1D & MLP architecture
│   └── gan.py                  # PyTorch GAN Generator & Discriminator
├── static/
│   ├── css/style.css           # Global adaptive Light & Dark design system
│   └── js/
│       ├── main.js             # Theme toggle & multilingual dictionary
│       ├── screening.js        # Real-time async API call & guidance rendering
│       ├── history.js          # Longitudinal trends & CSV export
│       └── about.js            # GAN loss chart, Viva FAQ & retraining trigger
└── templates/
    ├── base.html               # Sticky navbar, theme toggle & global scaffold
    ├── login.html              # Sign in, Sign up & 1-Click Demo portal
    ├── index.html              # Compact personal wellness dashboard
    ├── screening.html          # Real-time check-in & coping strategy hub
    ├── history.html            # Audit history & trajectory chart
    └── about.html              # Deep learning blueprint, viva Q&A & continuous learning
```

---

## ⚡ Quick Start (Local)

### Method 1: 1-Click Double Click
Simply double-click **`run_app.bat`** in the project folder. It will start the server and open your browser automatically.

### Method 2: Via Terminal
```bash
# Activate virtual environment
.\.venv\Scripts\activate      # Windows PowerShell

# Run server
python app.py
```
Open **`http://127.0.0.1:8000`** in Chrome.

**Demo Credentials:**
* Username: `student` | Password: `mhm2024`

---

## ☁️ 1-Click Cloud Deployment (Render.com)

1. Sign in to **[Render.com](https://render.com)** using your GitHub account.
2. Click **New +** ➔ **Web Service** ➔ Select repository **`ai-mental-health-monitor`**.
3. Verify settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service**. You will receive a live URL: `https://ai-mental-health-monitor.onrender.com`.

---

## 📐 Mathematical Formulation

### 1. Dynamic Risk Score Calculation
$$\text{Score} = P(\text{Low}) \times 15 + P(\text{Medium}) \times 55 + P(\text{High}) \times 92$$

### 2. GAN Minimax Objective
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
