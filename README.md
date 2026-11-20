# NLP Project Sem 5

**Group 8**
- Ayush Singh — 2401730015
- Krish Jaidka — 2401730188
- Sahil Shrivastava — 2401730016

This repository contains two complete NLP capstone projects, each with a working
Streamlit interface, a reproducible training pipeline, evaluation, documentation
and screenshots.

## 🚀 Live Deployments

| Project | Live URL |
|---------|----------|
| **Project 16 — BioNER Clinical** | <https://nlp-project-bioner.streamlit.app> |
| **Project 1 — Reel Verdict** | <https://nlp-project-sentimentanalysis.streamlit.app> |

Both apps are hosted on Streamlit Community Cloud and run directly from this repository's `main` branch.

---

## Repository structure

```
NLP Project Sem 5/
├── BIO NER/                        # Project 16 — Domain-Specific NER with Transformers
│   ├── app.py                      # Streamlit GUI — BioNER Clinical
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Setup, usage and design notes
│   ├── Project_Report.md           # Full project report
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── .streamlit/config.toml      # Light clinical theme
│   ├── .gitignore                  # Git ignore rules
│   ├── src/
│   │   ├── preprocess.py           # CoNLL loader, splits, NERDataset
│   │   ├── train_ner.py            # BioBERT fine-tuning
│   │   ├── evaluate.py             # Precision / Recall / F1
│   │   └── inference.py            # Transformer pipeline + rule-based fallback
│   ├── data/
│   │   ├── sample_biomedical_ner.conll
│   │   └── README.md
│   ├── screenshots/                # Hero, report, findings, document, mobile
│   └── models/                     # Trained weights (gitignored — retrain after clone)
├── Sentiment analysis/             # Project 1 — Sentiment Analysis
│   ├── app.py                      # Streamlit GUI — Reel Verdict
│   ├── requirements.txt
│   ├── README.md
│   ├── Project_Report.md
│   ├── EDA.ipynb
│   ├── .streamlit/config.toml      # Dark cinematic theme
│   ├── src/
│   ├── data/
│   ├── screenshots/                # Hero, review, batch, mobile
│   └── models/
└── README.md                       # This file
```

### Two distinct interface identities

| Project 16 — BioNER Clinical | Project 1 — Reel Verdict |
|---|---|
| <img src="BIO%20NER/screenshots/hero.png" width="100%"> | <img src="Sentiment%20analysis/screenshots/single_review.png" width="100%"> |

The two projects deliberately use different design systems: a light clinical
paper workbench for biomedical NER, and a near-black cinematic screening room
for sentiment analysis.

---

## Project 16 — Domain-Specific NER with Transformers

**Folder:** `BIO NER/`
**Domain:** Biomedical / Healthcare
**Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`) fine-tuned for token classification
**Entities:** DISEASE, DRUG, PROCEDURE, ANATOMY
**Interface:** **BioNER Clinical** — a clinical-paper workbench with annotated
source, finding ledger, negation-context flags, report metadata and CSV/JSON
exports.

### Quick start

```bash
cd "BIO NER"
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**. Without `models/biobert_ner/`, the app activates
the rule-based demo and reports that fallback transparently.

### Train and evaluate

```bash
python3 -m src.train_ner
python3 -m src.evaluate
```

See [`BIO NER/README.md`](BIO%20NER/README.md) for the full feature list, design
tokens, model behavior and setup instructions.

---

## Project 1 — Sentiment Analysis

**Folder:** `Sentiment analysis/`
**Objective:** Classify a sentence, review or feedback sample as Positive, Negative or Neutral.
**Dataset:** IMDb movie reviews (21-row sample included)
**Model:** TF-IDF + Logistic Regression / Naive Bayes / Linear SVM
**Interface:** **Reel Verdict** — a cinematic near-black and antique-gold
screening room with a verdict card, critic meters, batch ledger and responsive
mobile layout.

### Quick start

```bash
cd "Sentiment analysis"
pip install -r requirements.txt
streamlit run app.py
```

### Train

```bash
python3 -m src.train_model
```

See [`Sentiment analysis/README.md`](Sentiment%20analysis/README.md) for full
details, screenshots and design notes.

---

## Submission checklist

- [x] Complete source code uploaded to GitHub
- [x] `requirements.txt` included for both projects
- [x] `README.md` with setup and execution steps
- [x] Trained model or instructions to train
- [x] Screenshots of the working GUI
- [x] Jupyter notebook showing EDA
- [ ] PPT covering both projects
- [ ] GitHub repository link submitted via Google Form

---

## Contributions

All three group members contributed equally to both projects:

- Dataset collection, preprocessing and EDA
- Model development, training and evaluation
- Streamlit interface design and implementation
- Testing, documentation, screenshots and GitHub deployment

---

**Educational capstone project.** Model predictions are assistive and are not
clinical diagnoses. The biomedical interface is intended for research and
demonstration only, not for patient care.
