"""Train sentiment analysis models (TF-IDF + ML classifiers)."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from src.preprocess import preprocess_series


DATA_PATH = os.path.join("data", "sample_imdb.csv")
MODEL_DIR = os.path.join("models")
RANDOM_STATE = 42


def load_data(filepath: str = DATA_PATH):
    """Load dataset with columns: text, label."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    return df


def build_pipeline(model_type: str = "logistic_regression"):
    """Build a TF-IDF + classifier pipeline."""
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_type == "naive_bayes":
        clf = MultinomialNB()
    elif model_type == "svm":
        clf = LinearSVC(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", clf),
    ])
    return pipeline


def train(model_type: str = "logistic_regression", test_size: float = 0.2):
    """Train and save the best model."""
    df = load_data()
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    # Preprocess
    processed_texts = preprocess_series(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
    )

    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"sentiment_{model_type}.joblib")
    joblib.dump(pipeline, model_path)

    # Evaluate on test set
    from src.evaluate import evaluate_model
    report = evaluate_model(pipeline, X_test, y_test)

    return pipeline, report, model_path


if __name__ == "__main__":
    pipeline, report, path = train("logistic_regression")
    print(f"Model saved to: {path}")
    print("\nClassification Report:")
    print(report)
