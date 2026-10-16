"""Tiny intent classifier: text -> (intent, confidence).

Uses scikit-learn: TfidfVectorizer turns sentences into word-count
numbers, LogisticRegression learns which word patterns belong to
which intent. Training on our few hundred generated sentences
takes 1-2 seconds on a laptop CPU.
"""

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from training_data import generate_test_data, generate_training_data

MODEL_FILE = os.path.join(os.path.dirname(__file__), "intent_model.joblib")


def build_classifier():
    """Create the untrained sklearn pipeline."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    classifier = LogisticRegression(max_iter=1000)
    return make_pipeline(vectorizer, classifier)


def train_and_save(path=MODEL_FILE):
    """Train on generated data and save with joblib. Returns the model."""
    rows = generate_training_data()
    texts = [text for text, _intent in rows]
    labels = [intent for _text, intent in rows]
    model = build_classifier()
    model.fit(texts, labels)
    joblib.dump(model, path)
    return model


def load_or_train(path=MODEL_FILE):
    """Load the cached model, or train fresh if missing. Returns the model."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass  # Corrupt cache: fall through and retrain.
    return train_and_save(path)


def predict_intent(model, text):
    """Predict (intent, confidence) for one English sentence.

    Confidence is the top class probability between 0.0 and 1.0.
    Empty text returns ("unknown", 0.0).
    """
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ("unknown", 0.0)
    proba = model.predict_proba([cleaned])[0]
    best_index = int(proba.argmax())
    intent = str(model.classes_[best_index])
    return (intent, float(proba[best_index]))


def evaluate_accuracy(model=None):
    """Score the model on unseen sentences. Returns (accuracy, details).

    Details is a list of (sentence, expected, predicted, confidence).
    """
    if model is None:
        model = load_or_train()
    rows = generate_test_data()
    correct = 0
    details = []
    for sentence, expected in rows:
        predicted, confidence = predict_intent(model, sentence)
        if predicted == expected:
            correct += 1
        details.append((sentence, expected, predicted, confidence))
    accuracy = correct / len(rows) if rows else 0.0
    return (accuracy, details)
