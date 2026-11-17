# 🧠 NLP Project Sem 5

**Group 8**  
**Team Members:**
- Ayush Singh – 2401730015
- Krish Jaidka – 2401730188
- Sahil Shrivastava – 2401730016

This repository contains **two complete NLP capstone projects** developed from scratch with working Streamlit GUIs, reproducible training pipelines, evaluation, and documentation.

---

## 📁 Repository Structure

```
NLP Project Sem 5/
├── BIO NER/                        # Project 16 – Domain-Specific NER with Transformers
│   ├── app.py                      # Streamlit GUI (polished website-style UI)
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Project-specific setup & instructions
│   ├── .streamlit/config.toml      # Streamlit theme config
│   ├── .gitignore                  # Git ignore rules
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocess.py           # CoNLL loader, train/val/test split, NERDataset
│   │   ├── train_ner.py            # BioBERT fine-tuning script
│   │   ├── evaluate.py             # Precision / Recall / F1-score evaluation
│   │   └── inference.py            # Transformer prediction + rule-based fallback
│   ├── data/
│   │   ├── sample_biomedical_ner.conll   # Sample labelled biomedical data
│   │   └── README.md               # Dataset format instructions
│   └── models/                     # Trained BioBERT weights (gitignored – retrain after clone)
├── Sentiment analysis/             # Project 1 – Sentiment Analysis (Reel Verdict)
│   ├── app.py                      # Streamlit GUI (cinematic Reel Verdict UI)
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Project-specific setup & instructions
│   ├── .streamlit/config.toml      # Dark cinematic theme config
│   ├── .gitignore                  # Git ignore rules
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Regex cleaning, tokenization, built-in stopwords
│   │   ├── train_model.py          # TF-IDF + ML training
│   │   ├── evaluate.py             # Accuracy / Precision / Recall / F1
│   │   └── inference.py            # ML prediction + rule-based fallback
│   ├── data/
│   │   ├── sample_imdb.csv         # 21 labelled sample reviews
│   │   └── README.md               # Dataset format instructions
│   ├── screenshots/                # Hero, verdict, batch and mobile screenshots
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── Project_Report.md           # Full project report
│   └── models/                     # Trained weights (gitignored – retrain after clone)
└── README.md                       # This file
```

### Two distinct interface identities

| Project 16 — BioNER Studio | Project 1 — Reel Verdict |
|---|---|
| <img src="BIO%20NER/screenshots/results_dashboard.png" width="100%"> | <img src="Sentiment%20analysis/screenshots/single_review.png" width="100%"> |

---

## 🚀 Project 16 – Domain-Specific NER with Transformers (BIO NER folder)

**Domain:** Biomedical / Healthcare  
**Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`) fine-tuned for token classification  
**Entities:** DISEASE, DRUG, PROCEDURE, ANATOMY  
**GUI:** Streamlit with dark biotech theme, entity chips, stats dashboard, highlighted text, CSV download

### Quick start
```bash
cd "BIO NER"
pip install -r requirements.txt
streamlit run app.py
```

Then open: **http://localhost:8501**

### Train the model
```bash
python -m src.train_ner
```

### Evaluate
```bash
python -m src.evaluate
```

See [`BIO NER/README.md`](BIO NER/README.md) for full details.

---

## 📊 Project 1 – Sentiment Analysis (Sentiment analysis folder)

**Objective:** Identify whether a given sentence, review, or feedback is Positive, Negative, or Neutral.  
**Dataset:** IMDb Movie Reviews (21-row sample included)  
**Model:** TF-IDF + Logistic Regression / Naive Bayes / Linear SVM  
**Interface:** Streamlit — **Reel Verdict**, a cinematic near-black + antique-gold screening room with a sentiment scorecard, critic-consensus meters, batch ledger and responsive mobile layout. Deliberately distinct from the BioNER Studio theme.

### Quick start
```bash
cd "Sentiment analysis"
pip install -r requirements.txt
streamlit run app.py
```

### Train the model
```bash
python3 -m src.train_model
```

See [`Sentiment analysis/README.md`](Sentiment%20analysis/README.md) for full details, screenshots and design notes.

---

## ✅ Submission Checklist (per project)

- [x] Complete source code uploaded to GitHub
- [x] `requirements.txt` included
- [x] `README.md` with setup & execution steps
- [x] Trained model or instructions to train
- [x] Screenshots of the working GUI
- [x] Jupyter notebook showing EDA and training (optional but recommended)
- [ ] PPT covering both projects
- [ ] GitHub repository link submitted via Google Form

---

## 👥 Contributions

All three group members contributed equally to both projects:
- Dataset collection & preprocessing
- Model development & evaluation
- Streamlit GUI development
- Testing, documentation & GitHub deployment

---

**Educational capstone project.** Model predictions are assistive, not clinical diagnoses.
