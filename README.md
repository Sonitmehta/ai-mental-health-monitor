# AI-Based Mental Health Monitoring & Risk Screening System

An end-to-end Machine Learning and Deep Learning system designed for mental health risk screening, emotional state prediction, and longitudinal wellbeing monitoring.

> **Important Academic & Medical Notice:** This application is developed as an educational screening and research tool. It is **not a clinical diagnostic instrument** and is not a substitute for professional mental health care.

---

## 🌟 Key Features & Technical Contributions

1. **Multi-Aspect Deep Learning Classification (PyTorch 1D CNN)**:
   - Jointly predicts **Mental Health Risk Level** (`Low`, `Medium`, `High`) and **Emotional State** (`Positive`, `Neutral`, `Anxious`, `Sad`).
   - Uses multi-scale 1D Convolutional Neural Network kernels ($k=2, 3, 4$) combined with dense contextual features.

2. **Generative Adversarial Network (GAN) Feature Augmentation**:
   - Deep Generator and Discriminator architecture trained in feature space to synthesize minority (high-risk / severe distress) samples, addressing class imbalance in clinical datasets.

3. **Dynamic Continuous Risk Scoring (0–100)**:
   - Uses softmax probability distributions instead of hardcoded numbers:
     $$\text{Risk Score} = \text{round}\Big(P(\text{Low}) \times 15 + P(\text{Medium}) \times 55 + P(\text{High}) \times 92\Big)$$

4. **Longitudinal Wellbeing Tracking**:
   - Tracks screening scores over time across multiple sessions with interactive trend charts and CSV export functionality.

5. **Explainable Salience & Keyword Detection**:
   - Extracts salient distress and positive emotional indicators directly from user input.

6. **Interactive Streamlit Web Dashboard**:
   - Modern tabbed interface with real-time inference, longitudinal trends, and an interactive **Model Architecture & Evaluation** tab for viva/project demonstrations.

---

## 📁 Project Structure

```text
AI_Mental_Health_Monitor/
├── app.py                     # Streamlit multi-tab web dashboard
├── train_model.py             # Complete model training & evaluation pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation & viva guide
├── .gitignore
├── data/
│   └── sample_dataset.csv     # Expanded, balanced student mental health dataset (165+ samples)
├── models/                    # Saved weights, metadata, and evaluation metrics
│   ├── cnn_model.pt
│   ├── gan_generator.pt
│   ├── gan_classifier.pt
│   ├── tokenizer.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── metadata.json
│   └── evaluation_metrics.json
└── src/
    ├── __init__.py
    ├── nlp_utils.py           # Preprocessing, tokenizer, and salience indicator extractor
    ├── gan.py                 # PyTorch Generator, Discriminator, and training routines
    └── model_utils.py         # Multi-Branch TextCNN1D and FeatureClassifierMLP
```

---

## 🚀 Setup & Execution Guide

### 1. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

*If script execution is restricted:*
```cmd
.venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Deep Learning & GAN Pipeline
```bash
python train_model.py
```
This executes the full pipeline:
- Preprocesses & tokenizes text sequences and TF-IDF vectors.
- Trains the PyTorch 1D Multi-Branch CNN classifier.
- Trains the GAN on minority high-risk feature vectors.
- Benchmarks baseline vs. GAN-augmented feature classification.
- Saves model weights and `evaluation_metrics.json` into `models/`.

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎓 Academic Viva & Presentation Defense Guide

### 1. Why use a 1D CNN for text?
* **Answer:** 1D Convolutional Neural Networks with multiple kernel sizes ($k=2, 3, 4$) act as parallel n-gram feature detectors, capturing local context and emotional phrase structures efficiently without the high computational overhead of large transformers.

### 2. What is the role of the GAN in this project?
* **Answer:** Mental health datasets typically suffer from severe class imbalance because high-risk/crisis cases are fewer than neutral or positive cases. Our feature-space GAN learns the latent distribution of high-risk TF-IDF feature vectors to synthesize realistic synthetic samples and improve minority class recognition.

### 3. How is the continuous Risk Score computed?
* **Answer:** Rather than using discrete thresholds, the system computes an expected risk score by weighting each class's softmax probability:
  $$\text{Score} = P(\text{Low}) \times 15 + P(\text{Medium}) \times 55 + P(\text{High}) \times 92$$
  This allows continuous nuance (e.g., distinguishing a mild 48/100 from an acute 88/100).

---

## 🔮 Future Enhancements

- Integration of pretrained Sentence-BERT / RoBERTa embeddings.
- Voice emotion & acoustic prosody analysis.
- Clinician portal with HIPAA/GDPR-compliant data encryption.
- Explainable AI with integrated SHAP / LIME gradient attributions.
