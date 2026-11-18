"""Evaluate sentiment analysis model performance."""

import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained pipeline and return metrics dict."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
    }
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("===== Sentiment Analysis Evaluation =====\n")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def evaluate_saved_model(model_path: str, X_test, y_test):
    """Load a saved model and evaluate."""
    model = joblib.load(model_path)
    return evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    from src.train_model import load_data, preprocess_series
    from sklearn.model_selection import train_test_split

    df = load_data()
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()
    processed = preprocess_series(texts)

    _, X_test, _, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model_path = "models/sentiment_logistic_regression.joblib"
    if not __import__("os").path.exists(model_path):
        print("No saved model found. Train first: python -m src.train_model")
    else:
        m = evaluate_saved_model(model_path, X_test, y_test)
        print_metrics(m)
