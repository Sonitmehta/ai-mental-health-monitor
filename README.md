# 🧠 MindScan AI — AI-Powered Mental Health Screening System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-modal clinical mental health monitoring application built with **FastAPI**, **Vanilla HTML5/CSS3/JavaScript (ES6+)**, and **PyTorch**. The system integrates a **Dual-Branch TextCNN1D** with **Feature-Space GAN Augmentation** to classify emotional distress and triage clinical risk levels from natural language self-reflections.

---

## 🌟 Key Features

- 🎨 **Modern Dark UI Design:** Glassmorphic cards, teal accents, dynamic CSS animations, responsive typography, and mobile support.
- 🔐 **Authentication & Session Security:** JWT-based login (`/login`) with demo credentials and secure cookie tracking.
- 🌐 **Multilingual Interface:** Instant client-side language switching between **English**, **Devanagari Hindi (हिंदी)**, and **Hinglish**.
- 🧠 **Dual-Branch Neural Architecture:** Fuses 1D multi-scale spatial convolutional features (kernels 2, 3, 4) with 250-dimensional TF-IDF contextual embeddings.
- ⚡ **Feature-Space GAN Augmentation:** Synthesizes realistic continuous representations for the minority high-risk class, improving classifier F1 by **+8.9%** (0.722 → 0.811).
- 🚨 **Safety Clinical Prior:** Overrides raw predictions when urgent crisis biomarkers (e.g., self-harm, severe breakdown) are detected, immediately routing to emergency helplines.
- 📈 **Longitudinal Tracking & CSV Export:** Visualizes risk score progression across sessions with Chart.js time-series curves and enables one-click audit log CSV exports.
- 🚀 **Cloud Deployment Ready:** Includes standard `Procfile` for one-click hosting on Render, Railway, or VPS.

---

## 🏗️ Multi-Page Architecture

```
├── app.py                      # FastAPI web controller & API routes
├── Procfile                    # Render / Railway production startup command
├── requirements.txt            # Isolated dependency specification
├── data/
│   └── sample_dataset.csv      # 176 balanced clinical screening entries
├── models/
│   ├── cnn_model.pt            # Frozen weights for TextCNN1D
│   ├── gan_generator.pt        # Feature-space GAN Generator weights
│   ├── tokenizer.joblib        # Fitted SimpleTokenizer
│   ├── tfidf_vectorizer.joblib # 250-feature TF-IDF Vectorizer
│   └── metadata.json           # Model configuration parameters
├── src/
│   ├── nlp_utils.py            # Tokenizer & regex biomarker extractor
│   ├── model_utils.py          # PyTorch TextCNN1D & MLP architecture
│   └── gan.py                  # PyTorch GAN Generator & Discriminator
├── static/
│   ├── css/style.css           # Global dark theme design system
│   └── js/
│       ├── main.js             # Multilingual dictionary & toast alerts
│       ├── screening.js        # Real-time async API call & Chart.js gauge
│       ├── history.js          # Longitudinal trends & CSV export
│       └── about.js            # GAN convergence chart & Viva accordion
└── templates/
    ├── base.html               # Sticky navbar & global layout scaffold
    ├── login.html              # Secure user sign-in portal
    ├── index.html              # Primary telemetry dashboard
    ├── screening.html          # Real-time self-check & crisis triage
    ├── history.html            # Audit history & trajectory chart
    └── about.html              # Deep learning blueprint & viva defense
```

---

## ⚡ Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Sonitmehta/ai-mental-health-monitor.git
cd ai-mental-health-monitor

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # On Windows PowerShell
# source .venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
uvicorn app:app --reload --port 8000
```
Open your browser and navigate to **`http://localhost:8000`**

### 🔑 Demo Credentials

| Role | Username / Email | Password |
|:---|:---|:---|
| **Student** | `student` | `mhm2024` |
| **Admin** | `admin@mhm.ai` | `mhm2024` |

---

## 📊 Model Performance

| Metric | Baseline | MindScan AI (CNN + GAN) | Improvement |
|:---|:---:|:---:|:---:|
| **Risk Classification Accuracy** | 71.4% | **83.33%** | +11.9% |
| **Emotion Classification Accuracy** | 52.0% | **66.67%** | +14.7% |
| **High-Risk Class F1-Score** | 0.722 | **0.811** | **+8.9%** |

---

## 📐 Mathematical Formulation

### 1. Dynamic Risk Score Calculation
$$\text{Score} = P(\text{Low}) \times 15 + P(\text{Medium}) \times 55 + P(\text{High}) \times 92$$

### 2. GAN Minimax Objective
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

---

## 🎓 Viva Voce & Academic Q&A

1. **Why use FastAPI + HTML/CSS/JS instead of monolithic Streamlit?**
   - FastAPI provides true decoupling between the REST API (`/api/predict`) and frontend clients. It enables asynchronous high-throughput requests, custom UI styling (glassmorphism, CSS animations), and multi-lingual DOM manipulation without server reloads.
2. **Why fuse 1D CNN with TF-IDF?**
   - 1D CNN captures local consecutive n-gram grammatical structures, while TF-IDF captures global clinical word frequencies across the entire vocabulary.
3. **What is the role of the Feature GAN?**
   - In clinical NLP, high-risk distress text is naturally scarce. Instead of generating noisy raw synthetic text, our GAN synthesizes continuous vectors in the TF-IDF feature space, boosting minority class recall without introducing semantic drift.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
