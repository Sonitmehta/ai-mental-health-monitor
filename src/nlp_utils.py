import re
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "am", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "my", "i", "it", "this", "that", "have",
    "has", "had", "been", "very", "today", "about", "from", "as", "at", "by", "so"
}

# Comprehensive Clinical Distress & Risk Lexicon
HIGH_RISK_INDICATORS = [
    "crying", "cry", "failure", "fail", "failed", "failing", "hopeless", "hopelessness",
    "worthless", "exhausted", "isolated", "nothing brings", "despair", "cannot function",
    "panic attacks", "panic attack", "unbearable", "cannot face", "numb", "empty inside",
    "mental pain", "extreme dread", "giving up", "give up", "uncontrollably", "agony",
    "hyperventilating", "abandoned", "drowning", "impossible", "crippling", "drained of life",
    "breakdown", "darkness", "anguish", "doom", "losing my mind", "nightmare", "desperately",
    "hate myself", "hate my life", "broken inside", "cannot go on", "miserable", "ruined",
    "suffocated", "crying alone", "weeping", "suicidal", "die", "dying"
]

MEDIUM_RISK_INDICATORS = [
    "bad", "felt bad", "feeling bad", "worried", "worry", "worrying", "stress", "stressed",
    "stressful", "anxious", "anxiety", "deadlines", "deadline", "racing", "nervous",
    "overwhelmed", "tense", "uneasy", "panic", "restless", "lonely", "loneliness", "down",
    "lost interest", "discouraged", "fatigued", "fatigue", "disconnected", "skipping meals",
    "falling behind", "irritable", "drained", "left out", "gloomy", "headaches", "headache",
    "overthinking", "weary", "melancholy", "dread", "scared", "fear", "insecure", "struggling",
    "struggle", "confused", "lost", "frustrated", "frustrating", "pressure", "exhausting"
]

POSITIVE_INDICATORS = [
    "good", "great", "productive", "enjoyed", "enjoy", "enjoying", "calm", "relaxed", "relax",
    "excited", "exciting", "confident", "confidence", "pleasant", "peaceful", "content",
    "laughing", "laughter", "accomplished", "grateful", "gratitude", "refreshed", "energized",
    "hopeful", "grounded", "pleased", "cheerful", "motivated", "motivation", "happy", "happiness",
    "proud", "optimistic", "blessed", "enthusiastic", "wonderful", "satisfied", "fine", "better"
]


def clean_text(text: str) -> str:
    """Preprocess text: lowercasing, special character removal, and stopword filtering."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


class SimpleTokenizer:
    """Lightweight and robust tokenizer for sequence-based neural models."""

    def __init__(self, num_words: int = 4000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {"<PAD>": 0, oov_token: 1}
        self.index_word = {0: "<PAD>", 1: oov_token}
        self.word_counts = {}

    def fit_on_texts(self, texts: List[str]):
        counts = {}
        for text in texts:
            cleaned = clean_text(text)
            for word in cleaned.split():
                counts[word] = counts.get(word, 0) + 1
        self.word_counts = counts

        sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        idx = 2
        for word, _ in sorted_words:
            if idx >= self.num_words:
                break
            self.word_index[word] = idx
            self.index_word[idx] = word
            idx += 1

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            cleaned = clean_text(text)
            seq = []
            for word in cleaned.split():
                seq.append(self.word_index.get(word, self.word_index[self.oov_token]))
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences: List[List[int]], max_len: int = 50) -> np.ndarray:
        padded = np.zeros((len(sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            truncated = seq[:max_len]
            padded[i, :len(truncated)] = truncated
        return padded


def build_vectorizer(texts: List[str], max_features: int = 250) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Extract TF-IDF representation used for GAN feature augmentation and feature modeling."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform([clean_text(t) for t in texts])
    return vectorizer, X.toarray()


def extract_indicators(text: str) -> Dict[str, Any]:
    """Explainability & Clinical Lexicon detection utility for prominent emotional and risk cues."""
    t = text.lower()
    high_hits = [w for w in HIGH_RISK_INDICATORS if re.search(r"\b" + re.escape(w) + r"\b", t)]
    med_hits = [w for w in MEDIUM_RISK_INDICATORS if re.search(r"\b" + re.escape(w) + r"\b", t)]
    pos_hits = [w for w in POSITIVE_INDICATORS if re.search(r"\b" + re.escape(w) + r"\b", t)]

    all_hits = list(dict.fromkeys(high_hits + med_hits + pos_hits))
    return {
        "high": high_hits,
        "medium": med_hits,
        "positive": pos_hits,
        "all_indicators": all_hits,
        "high_count": len(high_hits),
        "medium_count": len(med_hits),
        "positive_count": len(pos_hits)
    }
