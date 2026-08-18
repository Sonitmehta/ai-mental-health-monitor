import sys
import io

# Ensure UTF-8 output encoding across all operating systems
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.nlp_utils import clean_text, SimpleTokenizer, build_vectorizer
from src.gan import train_feature_gan, generate_synthetic_features
from src.model_utils import TextCNN1D, FeatureClassifierMLP, save_metadata

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sample_dataset.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

MAX_LEN = 45
EPOCHS_CNN = 85
EPOCHS_GAN = 150
BATCH_SIZE = 12


def main():
    print("=" * 60)
    print("AI Mental Health Monitor: Training Pipeline")
    print("=" * 60)

    # 1. Load Dataset
    print("\n[1/5] Loading and exploring dataset...")
    df = pd.read_csv(DATA)
    print(f"Total samples loaded: {len(df)}")
    print("Risk distribution:", df["risk"].value_counts().to_dict())
    print("Emotion distribution:", df["emotion"].value_counts().to_dict())

    texts = df["text"].astype(str).tolist()
    risk_labels_raw = df["risk"].astype(str).tolist()
    emotion_labels_raw = df["emotion"].astype(str).tolist()

    # 2. Encoders & Preprocessing
    risk_encoder = LabelEncoder()
    y_risk = risk_encoder.fit_transform(risk_labels_raw)

    emotion_encoder = LabelEncoder()
    y_emotion = emotion_encoder.fit_transform(emotion_labels_raw)

    # NLP Vectorizer (TF-IDF) & Tokenizer
    vectorizer, X_tfidf = build_vectorizer(texts, max_features=250)
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")

    tokenizer = SimpleTokenizer(num_words=4000)
    tokenizer.fit_on_texts(texts)
    joblib.dump(tokenizer, MODELS / "tokenizer.joblib")

    seqs = tokenizer.texts_to_sequences(texts)
    X_seq = tokenizer.pad_sequences(seqs, max_len=MAX_LEN)

    # Train / Test Split (Stratified on risk)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y_risk
    )

    X_train_seq = torch.tensor(X_seq[train_idx], dtype=torch.long)
    X_train_feat = torch.tensor(X_tfidf[train_idx], dtype=torch.float32)
    y_train_risk = torch.tensor(y_risk[train_idx], dtype=torch.long)
    y_train_emotion = torch.tensor(y_emotion[train_idx], dtype=torch.long)

    X_test_seq = torch.tensor(X_seq[test_idx], dtype=torch.long)
    X_test_feat = torch.tensor(X_tfidf[test_idx], dtype=torch.float32)
    y_test_risk = torch.tensor(y_risk[test_idx], dtype=torch.long)
    y_test_emotion = torch.tensor(y_emotion[test_idx], dtype=torch.long)

    # 3. Train 1D Multi-Branch CNN Multi-Head Classifier
    print(f"\n[2/5] Training 1D Multi-Branch CNN Model ({EPOCHS_CNN} epochs)...")
    vocab_size = len(tokenizer.word_index) + 2
    feature_dim = int(X_tfidf.shape[1])
    cnn = TextCNN1D(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        embed_dim=64,
        num_risk_classes=len(risk_encoder.classes_),
        num_emotion_classes=len(emotion_encoder.classes_),
        kernel_sizes=[2, 3, 4],
        num_filters=32,
        dropout=0.2
    )

    criterion_risk = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.003, weight_decay=1e-4)

    cnn_history = {"loss": []}
    dataset_size = len(train_idx)

    for epoch in range(EPOCHS_CNN):
        cnn.train()
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0

        for i in range(0, dataset_size, BATCH_SIZE):
            b_idx = perm[i:i + BATCH_SIZE]
            b_x_seq = X_train_seq[b_idx]
            b_x_feat = X_train_feat[b_idx]
            b_yr = y_train_risk[b_idx]
            b_ye = y_train_emotion[b_idx]

            optimizer.zero_grad()
            risk_logits, emotion_logits = cnn(b_x_seq, b_x_feat)

            loss_r = criterion_risk(risk_logits, b_yr)
            loss_e = criterion_emotion(emotion_logits, b_ye)
            total_loss = loss_r + 0.8 * loss_e

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        cnn_history["loss"].append(epoch_loss / max(1, (dataset_size // BATCH_SIZE)))

    # Evaluate CNN on Test Set
    cnn.eval()
    with torch.no_grad():
        r_logits, e_logits = cnn(X_test_seq, X_test_feat)
        risk_preds = r_logits.argmax(dim=1).numpy()
        emotion_preds = e_logits.argmax(dim=1).numpy()

    risk_acc = float(accuracy_score(y_test_risk.numpy(), risk_preds))
    emotion_acc = float(accuracy_score(y_test_emotion.numpy(), emotion_preds))
    print(f"-> CNN Test Risk Accuracy: {risk_acc * 100:.2f}%")
    print(f"-> CNN Test Emotion Accuracy: {emotion_acc * 100:.2f}%")

    risk_report = classification_report(
        y_test_risk.numpy(), risk_preds, target_names=risk_encoder.classes_, output_dict=True, zero_division=0
    )
    risk_cm = confusion_matrix(y_test_risk.numpy(), risk_preds).tolist()

    # 4. Train Generative Adversarial Network (GAN)
    print("\n[3/5] Training Generative Adversarial Network (GAN) on minority high-risk feature vectors...")
    high_risk_label_idx = int(risk_encoder.transform(["high"])[0])
    high_risk_train_mask = (y_risk[train_idx] == high_risk_label_idx)
    high_risk_features = X_tfidf[train_idx][high_risk_train_mask]

    gan_gen, gan_history = train_feature_gan(
        high_risk_features,
        epochs=EPOCHS_GAN,
        batch_size=8,
        latent_dim=32,
        lr=0.0003
    )

    # 5. Benchmark GAN Feature Augmentation
    print("\n[4/5] Benchmarking GAN Augmentation Impact...")
    n_synthetic = 20
    synthetic_high_risk = generate_synthetic_features(gan_gen, n_synthetic, latent_dim=32)

    X_train_raw = X_tfidf[train_idx]
    y_train_raw = y_risk[train_idx]

    # Baseline MLP (No GAN)
    mlp_baseline = FeatureClassifierMLP(feature_dim=X_tfidf.shape[1], num_classes=len(risk_encoder.classes_))
    opt_base = optim.Adam(mlp_baseline.parameters(), lr=0.004)
    for _ in range(70):
        mlp_baseline.train()
        opt_base.zero_grad()
        out = mlp_baseline(torch.tensor(X_train_raw, dtype=torch.float32))
        loss = criterion_risk(out, torch.tensor(y_train_raw, dtype=torch.long))
        loss.backward()
        opt_base.step()

    mlp_baseline.eval()
    with torch.no_grad():
        base_preds = mlp_baseline(X_test_feat).argmax(dim=1).numpy()
    base_acc = float(accuracy_score(y_test_risk.numpy(), base_preds))
    base_f1 = float(f1_score(y_test_risk.numpy(), base_preds, average="macro", zero_division=0))

    # Augmented MLP (With GAN-Generated High Risk Features)
    X_train_aug = np.vstack([X_train_raw, synthetic_high_risk])
    y_train_aug = np.hstack([y_train_raw, np.full(n_synthetic, high_risk_label_idx)])

    mlp_aug = FeatureClassifierMLP(feature_dim=X_tfidf.shape[1], num_classes=len(risk_encoder.classes_))
    opt_aug = optim.Adam(mlp_aug.parameters(), lr=0.004)
    for _ in range(70):
        mlp_aug.train()
        opt_aug.zero_grad()
        out = mlp_aug(torch.tensor(X_train_aug, dtype=torch.float32))
        loss = criterion_risk(out, torch.tensor(y_train_aug, dtype=torch.long))
        loss.backward()
        opt_aug.step()

    mlp_aug.eval()
    with torch.no_grad():
        aug_preds = mlp_aug(X_test_feat).argmax(dim=1).numpy()
    aug_acc = float(accuracy_score(y_test_risk.numpy(), aug_preds))
    aug_f1 = float(f1_score(y_test_risk.numpy(), aug_preds, average="macro", zero_division=0))

    print(f"-> Baseline Classifier Macro F1: {base_f1:.3f} | Accuracy: {base_acc * 100:.2f}%")
    print(f"-> GAN-Augmented Classifier Macro F1: {aug_f1:.3f} | Accuracy: {aug_acc * 100:.2f}%")

    # 6. Save All Models & Evaluation Artifacts
    print("\n[5/5] Saving model weights and academic evaluation metrics...")
    torch.save(cnn.state_dict(), MODELS / "cnn_model.pt")
    torch.save(gan_gen.state_dict(), MODELS / "gan_generator.pt")
    torch.save(mlp_aug.state_dict(), MODELS / "gan_classifier.pt")
    np.save(MODELS / "synthetic_features.npy", synthetic_high_risk)

    metadata = {
        "max_len": MAX_LEN,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
        "risk_classes": risk_encoder.classes_.tolist(),
        "emotion_classes": emotion_encoder.classes_.tolist(),
        "sample_count": len(df)
    }
    save_metadata(MODELS / "metadata.json", metadata)

    eval_metrics = {
        "cnn_risk_accuracy": risk_acc,
        "cnn_emotion_accuracy": emotion_acc,
        "risk_classification_report": risk_report,
        "risk_confusion_matrix": risk_cm,
        "risk_classes": risk_encoder.classes_.tolist(),
        "emotion_classes": emotion_encoder.classes_.tolist(),
        "baseline_accuracy": base_acc,
        "baseline_f1": base_f1,
        "gan_augmented_accuracy": aug_acc,
        "gan_augmented_f1": aug_f1,
        "gan_history": gan_history,
        "cnn_loss_history": cnn_history["loss"],
        "dataset_summary": {
            "total_samples": len(df),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "synthetic_samples_generated": n_synthetic
        }
    }
    (MODELS / "evaluation_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Training complete! All artifacts successfully saved to 'models/'")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock", type=str, default=None, help="Path to lock file to delete after training")
    args = parser.parse_args()
    try:
        main()
    finally:
        # Always clean up the lock file even if training fails
        if args.unlock:
            lock = Path(args.unlock)
            if lock.exists():
                lock.unlink()
