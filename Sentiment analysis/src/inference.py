"""Inference logic for Sentiment Analysis.

Loads a trained TF-IDF + ML pipeline and predicts sentiment with confidence.
Falls back to a rule-based predictor if no model weights are available.
"""

import os
import re
import joblib
from typing import List, Dict, Optional

import numpy as np
from src.preprocess import preprocess


MODEL_DIR = os.path.join("models")


class SentimentPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.classes_ = None

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.classes_ = list(self.model.classes_)
                print(f"Loaded model: {model_path}")
            except Exception as exc:
                print(f"Could not load model ({exc}). Using rule-based fallback.")
                self.model = None
        else:
            print("No model path provided. Using rule-based fallback.")

    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text.

        Returns:
            dict with keys: label, confidence, probabilities
        """
        if not text or not text.strip():
            return {"label": "Neutral", "confidence": 0.0, "probabilities": {}}

        if self.model is not None:
            return self._predict_ml(text)
        return self._predict_rule_based(text)

    def _predict_ml(self, text: str) -> Dict:
        cleaned = preprocess(text)
        classes = list(self.model.classes_)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([cleaned])[0]
        else:
            # Linear SVM has no predict_proba; map decision scores to probabilities.
            scores = np.atleast_2d(self.model.decision_function([cleaned]))
            if scores.shape[1] == 1:  # binary classifier
                positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
                proba = np.column_stack([1.0 - positive, positive])[0]
            else:
                shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = (shifted / shifted.sum(axis=1, keepdims=True))[0]
        idx = int(np.argmax(proba))
        label = classes[idx]
        confidence = float(proba[idx])
        probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
        return {"label": label, "confidence": confidence, "probabilities": probabilities}

    def _predict_rule_based(self, text: str) -> Dict:
        """Lightweight rule-based sentiment using keyword lexicon."""
        positive_words = {
            "good", "great", "excellent", "amazing", "love", "loved", "like",
            "liked", "best", "awesome", "fantastic", "wonderful", "brilliant",
            "outstanding", "superb", "perfect", "impressive", "enjoyed", "happy",
            "pleased", "satisfied", "recommend", "highly", "positive"
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "hated", "dislike",
            "disliked", "worst", "poor", "boring", "waste", "disappointing",
            "disappointed", "annoying", "frustrating", "ugly", "stupid", "sad",
            "angry", "negative", "regret", "avoid"
        }

        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        pos_count = len(tokens & positive_words)
        neg_count = len(tokens & negative_words)

        if pos_count > neg_count:
            label = "Positive"
            confidence = min(0.5 + 0.1 * (pos_count - neg_count), 0.95)
        elif neg_count > pos_count:
            label = "Negative"
            confidence = min(0.5 + 0.1 * (neg_count - pos_count), 0.95)
        else:
            label = "Neutral"
            confidence = 0.5

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "Positive": confidence if label == "Positive" else (1 - confidence) / 2,
                "Negative": confidence if label == "Negative" else (1 - confidence) / 2,
                "Neutral": confidence if label == "Neutral" else (1 - confidence) / 2,
            },
        }


def get_predictor(model_path: Optional[str] = None) -> SentimentPredictor:
    """Factory to get a SentimentPredictor instance."""
    return SentimentPredictor(model_path)
