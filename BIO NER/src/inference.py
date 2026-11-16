"""Inference logic for biomedical NER.

Tries to load a fine-tuned BioBERT model. Falls back to base BioBERT, and
finally to a lightweight rule-based predictor so the GUI always works for
demonstration even without trained weights.
"""

import re
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NERPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        rule_based: bool = False,
    ):
        self.rule_based = rule_based
        self.model_path = model_path
        self.model_name = model_name
        self.pipe = None

        if not rule_based:
            try:
                load_path = model_path or model_name
                tokenizer = AutoTokenizer.from_pretrained(load_path)
                model = AutoModelForTokenClassification.from_pretrained(load_path)
                model.eval()
                self.pipe = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=-1,  # CPU
                )
                print(f"Loaded transformer NER model: {load_path}")
            except Exception as exc:
                print(f"Could not load transformer model ({exc}). Falling back to rule-based.")
                self.rule_based = True

        if self.rule_based:
            self._setup_rules()

    # ------------------------------------------------------------------
    # Rule-based fallback (for demos when model weights are unavailable)
    # ------------------------------------------------------------------
    def _setup_rules(self):
        self.patterns = [
            (r"\b(type\s+)?2\s*diabetes\b", "DISEASE"),
            (r"\b(type\s+)?1\s*diabetes\b", "DISEASE"),
            (r"\bdiabetes\b", "DISEASE"),
            (r"\bhypertension\b", "DISEASE"),
            (r"\bhigh blood pressure\b", "DISEASE"),
            (r"\bcancer\b", "DISEASE"),
            (r"\bcarcinoma\b", "DISEASE"),
            (r"\btumor\b|\btumour\b", "DISEASE"),
            (r"\bpneumonia\b", "DISEASE"),
            (r"\bappendicitis\b", "DISEASE"),
            (r"\basthma\b", "DISEASE"),
            (r"\bfever\b", "DISEASE"),
            (r"\bstroke\b", "DISEASE"),
            (r"\bmetformin\b", "DRUG"),
            (r"\binsulin\b", "DRUG"),
            (r"\blisinopril\b", "DRUG"),
            (r"\batorvastatin\b", "DRUG"),
            (r"\balbuterol\b", "DRUG"),
            (r"\bparacetamol\b", "DRUG"),
            (r"\baspirin\b", "DRUG"),
            (r"\bibiuprofen\b|\bibuprofen\b", "DRUG"),
            (r"\bmri\b", "PROCEDURE"),
            (r"\bct scan\b", "PROCEDURE"),
            (r"\bx-ray\b", "PROCEDURE"),
            (r"\bbiopsy\b", "PROCEDURE"),
            (r"\bsurgery\b", "PROCEDURE"),
            (r"\bdialysis\b", "PROCEDURE"),
            (r"\bangiography\b", "PROCEDURE"),
            (r"\bappendectomy\b", "PROCEDURE"),
            (r"\bchemotherapy\b", "PROCEDURE"),
            (r"\bheart\b", "ANATOMY"),
            (r"\bliver\b", "ANATOMY"),
            (r"\bkidney\b", "ANATOMY"),
            (r"\blung\b", "ANATOMY"),
            (r"\bbrain\b", "ANATOMY"),
            (r"\bblood\b", "ANATOMY"),
        ]

    def _rule_based_predict(self, text: str, conf_thresh: float) -> List[Dict]:
        entities = []
        for pattern, label in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "entity": label,
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.95,
                    }
                )
        return self._merge_overlaps(entities, conf_thresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text: str, conf_thresh: float = 0.5) -> List[Dict]:
        if not text or not text.strip():
            return []

        if self.rule_based:
            return self._rule_based_predict(text, conf_thresh)

        results = self.pipe(text)
        entities = []
        for r in results:
            score = float(r.get("score", 0.0))
            if score >= conf_thresh:
                entities.append(
                    {
                        "entity": r["entity_group"],
                        "word": r["word"],
                        "start": int(r["start"]),
                        "end": int(r["end"]),
                        "confidence": round(score, 4),
                    }
                )
        return self._merge_overlaps(entities, conf_thresh)

    @staticmethod
    def _merge_overlaps(entities: List[Dict], conf_thresh: float) -> List[Dict]:
        """Remove overlapping spans, keeping the higher-confidence one."""
        entities = [e for e in entities if e["confidence"] >= conf_thresh]
        entities.sort(key=lambda x: (x["start"], -x["confidence"]))
        merged = []
        for ent in entities:
            overlap = False
            for m in merged:
                if ent["start"] < m["end"] and ent["end"] > m["start"]:
                    overlap = True
                    break
            if not overlap:
                merged.append(ent)
        merged.sort(key=lambda x: x["start"])
        return merged
