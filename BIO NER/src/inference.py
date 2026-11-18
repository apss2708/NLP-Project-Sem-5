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
        self.backend = "rule-based" if rule_based else "uninitialized"
        self.load_error = None

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
                self.backend = "transformer"
                print(f"Loaded transformer NER model: {load_path}")
            except Exception as exc:
                self.load_error = str(exc)
                print(f"Could not load transformer model ({exc}). Falling back to rule-based.")
                self.rule_based = True

        if self.rule_based:
            self.backend = "rule-based"
            self._setup_rules()

    # ------------------------------------------------------------------
    # Rule-based fallback (for demos when model weights are unavailable)
    # ------------------------------------------------------------------
    def _setup_rules(self):
        # Multi-word and specific patterns come first so that overlaps resolve
        # toward the more informative span (e.g. "heart failure" over "heart").
        self.patterns = [
            # Diseases and conditions
            (r"\bmyocardial infarction\b", "DISEASE"),
            (r"\bheart failure\b", "DISEASE"),
            (r"\bchronic kidney disease\b", "DISEASE"),
            (r"\bhigh blood pressure\b", "DISEASE"),
            (r"\b(type\s+)?2\s*diabetes(?:\s+mellitus)?\b", "DISEASE"),
            (r"\b(type\s+)?1\s*diabetes(?:\s+mellitus)?\b", "DISEASE"),
            (r"\bdiabetes mellitus\b", "DISEASE"),
            (r"\bdiabetes\b", "DISEASE"),
            (r"\bhypertension\b", "DISEASE"),
            (r"\bchest pain\b", "DISEASE"),
            (r"\bshortness of breath\b", "DISEASE"),
            (r"\bcancer\b", "DISEASE"),
            (r"\bcarcinoma\b", "DISEASE"),
            (r"\btumou?r\b", "DISEASE"),
            (r"\bmetastasis\b", "DISEASE"),
            (r"\bpneumonia\b", "DISEASE"),
            (r"\bappendicitis\b", "DISEASE"),
            (r"\basthma\b", "DISEASE"),
            (r"\bfever\b", "DISEASE"),
            (r"\bstroke\b", "DISEASE"),
            (r"\binfection\b", "DISEASE"),
            (r"\barrhythmia\b", "DISEASE"),
            (r"\bmigraine\b", "DISEASE"),
            (r"\barthritis\b", "DISEASE"),
            (r"\bdepression\b", "DISEASE"),
            (r"\banemia\b", "DISEASE"),
            (r"\bsepsis\b", "DISEASE"),
            # Medications
            (r"\bmetformin\b", "DRUG"),
            (r"\binsulin\b", "DRUG"),
            (r"\blisinopril\b", "DRUG"),
            (r"\batorvastatin\b", "DRUG"),
            (r"\balbuterol\b", "DRUG"),
            (r"\bparacetamol\b", "DRUG"),
            (r"\baspirin\b", "DRUG"),
            (r"\bibuprofen\b", "DRUG"),
            (r"\bcisplatin\b", "DRUG"),
            (r"\bamoxicillin\b", "DRUG"),
            (r"\bomeprazole\b", "DRUG"),
            (r"\bamlodipine\b", "DRUG"),
            (r"\bwarfarin\b", "DRUG"),
            (r"\bprednisone\b", "DRUG"),
            # Procedures and tests
            (r"\bct scan\b", "PROCEDURE"),
            (r"\bmri scan\b", "PROCEDURE"),
            (r"\bmri\b", "PROCEDURE"),
            (r"\bct\b", "PROCEDURE"),
            (r"\becg\b", "PROCEDURE"),
            (r"\belectrocardiogram\b", "PROCEDURE"),
            (r"\bx-ray\b", "PROCEDURE"),
            (r"\bbiopsy\b", "PROCEDURE"),
            (r"\bsurgery\b", "PROCEDURE"),
            (r"\bdialysis\b", "PROCEDURE"),
            (r"\bangiography\b", "PROCEDURE"),
            (r"\bappendectomy\b", "PROCEDURE"),
            (r"\bchemotherapy\b", "PROCEDURE"),
            (r"\bradiation therapy\b", "PROCEDURE"),
            (r"\bphysical therapy\b", "PROCEDURE"),
            (r"\bultrasound\b", "PROCEDURE"),
            (r"\bblood work\b", "PROCEDURE"),
            (r"\bblood test\b", "PROCEDURE"),
            # Anatomy
            (r"\bheart\b", "ANATOMY"),
            (r"\bliver\b", "ANATOMY"),
            (r"\bkidneys?\b", "ANATOMY"),
            (r"\blungs?\b", "ANATOMY"),
            (r"\bbrain\b", "ANATOMY"),
            (r"\bblood vesse?ls?\b", "ANATOMY"),
            (r"\bpancreas\b", "ANATOMY"),
            (r"\bstomach\b", "ANATOMY"),
            (r"\bspine\b", "ANATOMY"),
            (r"\bartery\b", "ANATOMY"),
            (r"\bvein\b", "ANATOMY"),
            (r"\bthyroid\b", "ANATOMY"),
            (r"\bprostate\b", "ANATOMY"),
            (r"\bbreast\b", "ANATOMY"),
            (r"\bcolon\b", "ANATOMY"),
            (r"\bmyocardium\b", "ANATOMY"),
        ]

    @staticmethod
    def _is_negated(text: str, start: int, end: int) -> bool:
        """Flag simple negation cues in the immediate span context.

        This is a demonstration heuristic, not a clinical assertion classifier.
        It looks at a short token window before and after the entity so phrases
        such as "no acute myocardial infarction" or "dialysis was not required"
        are marked for review.
        """
        cues = {
            "no",
            "not",
            "without",
            "negative",
            "denies",
            "denied",
            "never",
            "neither",
            "nor",
            "absent",
            "absence",
        }
        # Keep the cue search inside the current clause/sentence so a negation
        # earlier in the note cannot flag an unrelated later finding.
        prefix_window = text[max(0, start - 60) : start]
        prefix_window = re.split(r"[.,;:\n]", prefix_window)[-1]
        suffix_window = text[end : end + 45]
        suffix_window = re.split(r"[.;:\n]", suffix_window)[0]
        prefix = re.findall(r"[A-Za-z']+", prefix_window.lower())[-6:]
        suffix = re.findall(r"[A-Za-z']+", suffix_window.lower())[:4]

        if any(token in cues for token in prefix):
            return True

        # A bare "no" after a span usually belongs to the next finding
        # ("heart showed no infarction"), so the suffix requires a tighter
        # postfix construction such as "was not required" or "was negative".
        for index, token in enumerate(suffix):
            if token in {"negative", "absent", "denied"}:
                return True
            if token == "not":
                previous = suffix[index - 1] if index else ""
                following = suffix[index + 1] if index + 1 < len(suffix) else ""
                if previous in {
                    "was",
                    "were",
                    "is",
                    "are",
                    "has",
                    "have",
                    "had",
                    "remains",
                    "remained",
                } or following in {
                    "required",
                    "performed",
                    "indicated",
                    "needed",
                    "present",
                    "detected",
                    "seen",
                }:
                    return True
        return False

    def _rule_based_predict(self, text: str, conf_thresh: float) -> List[Dict]:
        entities = []
        for pattern, label in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Confidence is a transparent specificity heuristic, not a
                # calibrated probability: longer dictionary patterns are more
                # specific, so single tokens score 0.90 and 4+ tokens score 0.96.
                token_count = len(match.group().split())
                confidence = 0.88 + 0.02 * min(4, token_count)
                entities.append(
                    {
                        "entity": label,
                        "word": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": round(confidence, 4),
                        "negated": self._is_negated(text, match.start(), match.end()),
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
                        # Negation is a separate assertion task; the
                        # transformer path does not guess it here.
                        "negated": False,
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
