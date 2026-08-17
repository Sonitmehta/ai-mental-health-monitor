import re
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = {
    "the", "a", "an", "and", "or", "is", "am", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "it", "this", "that",
    "about", "from", "as", "at", "by"
}

# Comprehensive Clinical Distress, Somatic Pain & Hinglish Lexicon
HIGH_RISK_INDICATORS = [
    # Severe Emotional Breakdown & Despair (English)
    "crying", "cry", "failure", "fail", "failed", "failing", "hopeless", "hopelessness",
    "worthless", "exhausted", "isolated", "nothing brings", "despair", "cannot function",
    "panic attacks", "panic attack", "unbearable", "cannot face", "numb", "empty inside",
    "mental pain", "extreme dread", "giving up", "give up", "uncontrollably", "agony",
    "hyperventilating", "abandoned", "drowning", "impossible", "crippling", "drained of life",
    "breakdown", "darkness", "anguish", "doom", "losing my mind", "nightmare", "desperately",
    "hate myself", "hate my life", "broken inside", "cannot go on", "miserable", "ruined",
    "suffocated", "crying alone", "weeping", "suicidal", "die", "dying", "self harm",

    # Severe Distress & Somatic Agony (Hinglish / Hindi)
    "marne ka mann", "mar jana chahta", "mar jana chahti", "himmat haar gaya", "himmat haar gayi",
    "kuch theek nahi hoga", "koi fayda nahi", "rona aa rha", "ro raha hu", "ro rahi hu",
    "jeene ka mann nahi", "sab khatam", "zindagi bekar hai", "bardasht nahi ho raha",
    "unbearable pain", "bohot dard ho rha", "bhot dard"
]

MEDIUM_RISK_INDICATORS = [
    # Mild/Moderate Psychological Strain (English)
    "bad", "felt bad", "feeling bad", "worried", "worry", "worrying", "stress", "stressed",
    "stressful", "anxious", "anxiety", "deadlines", "deadline", "racing", "nervous",
    "overwhelmed", "tense", "uneasy", "panic", "restless", "lonely", "loneliness", "down",
    "lost interest", "discouraged", "fatigued", "fatigue", "disconnected", "skipping meals",
    "falling behind", "irritable", "drained", "left out", "gloomy", "headaches", "headache",
    "overthinking", "weary", "melancholy", "dread", "scared", "fear", "insecure", "struggling",
    "struggle", "confused", "lost", "frustrated", "frustrating", "pressure", "exhausting",
    "insomnia", "sleepless",

    # Somatic Distress & Physical Pain Biomarkers
    "hurting", "hurt", "hurts", "pain", "in so much pain", "aching", "ache", "body ache",
    "spine", "shoulder", "neck pain", "back pain", "chest tightness", "migraine", "severe headache",

    # Mild/Moderate Distress & Somatic Pain (Hinglish / Hindi)
    "dard", "dard ho rha", "dard ho raha hai", "headache ho rha", "sir dard", "sar dard",
    "pareshan", "pareshaan", "pareshani", "bhot kuch", "bohot kuch", "thak gaya", "thak gayi",
    "thakan", "ghabrahat", "bechain", "bechaini", "chinta", "tension", "tension ho rahi",
    "neend nahi aa rahi", "dil ghabra raha", "kuch samajh nahi aa raha", "akelapan",
    "akela feel", "udaas", "udaasi", "darr lag raha", "man udas hai"
]

POSITIVE_INDICATORS = [
    # Positive & Grounded (English)
    "good", "great", "productive", "enjoyed", "enjoy", "enjoying", "calm", "relaxed", "relax",
    "excited", "exciting", "confident", "confidence", "pleasant", "peaceful", "content",
    "laughing", "laughter", "accomplished", "grateful", "gratitude", "refreshed", "energized",
    "hopeful", "grounded", "pleased", "cheerful", "motivated", "motivation", "happy", "happiness",
    "proud", "optimistic", "blessed", "enthusiastic", "wonderful", "satisfied", "fine", "better",

    # Positive (Hinglish / Hindi)
    "accha", "achha lag raha", "bohot accha", "mast", "badhiya", "khush", "khushi",
    "sukoon", "shanti", "maza aaya", "sahi lag raha", "energized feel"
]


def clean_text(text: str) -> str:
    """Preprocess text: lowercasing, special character normalization, and tokenizing."""
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
    """Fit a TF-IDF vectorizer over preprocessed text inputs."""
    cleaned_texts = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    features = vectorizer.fit_transform(cleaned_texts).toarray().astype(np.float32)
    return vectorizer, features


def extract_indicators(text: str) -> Dict[str, Any]:
    """
    Scans natural language input against clinical, somatic pain,
    and Hinglish distress lexicons using word-boundary regex.
    """
    if not isinstance(text, str):
        text = str(text)

    text_lower = text.lower()

    found_high = []
    found_med = []
    found_pos = []

    for ind in HIGH_RISK_INDICATORS:
        pattern = r"\b" + re.escape(ind) + r"\b"
        if re.search(pattern, text_lower):
            found_high.append(ind)

    for ind in MEDIUM_RISK_INDICATORS:
        pattern = r"\b" + re.escape(ind) + r"\b"
        if re.search(pattern, text_lower):
            # Avoid duplicate if part of high risk
            if not any(ind in h for h in found_high):
                found_med.append(ind)

    for ind in POSITIVE_INDICATORS:
        pattern = r"\b" + re.escape(ind) + r"\b"
        if re.search(pattern, text_lower):
            found_pos.append(ind)

    all_found = list(set(found_high + found_med + found_pos))

    return {
        "high_risk": found_high,
        "medium_risk": found_med,
        "positive": found_pos,
        "all_indicators": all_found,
        "high_count": len(found_high),
        "medium_count": len(found_med),
        "positive_count": len(found_pos)
    }
