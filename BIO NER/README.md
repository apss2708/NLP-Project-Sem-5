# 🧬 Project 16 – Domain-Specific NER with Transformers

**Group:** 8  
**Team Members:**
- Ayush Singh – 2401730015
- Krish Jaidka – 2401730188
- Sahil Shrivastava – 2401730016

**Domain:** Biomedical / Healthcare  
**Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`) fine-tuned for token classification  
**GUI:** Streamlit web application

---

## 📋 Project Overview

This project builds a domain-specific Named Entity Recognition (NER) system that extracts specialized biomedical entities from clinical and research text, such as diseases, drugs, procedures, and anatomy terms.

### What the system does
- Accepts raw biomedical/clinical text or a `.txt` document upload
- Tokenizes and runs inference with a transformer-based NER model
- Highlights detected entities directly in the text
- Displays extracted entities in a table (Entity, Category, Position, Confidence)

### Supported Entity Labels
| Label | Description | Example |
|-------|-------------|---------|
| DISEASE | Medical conditions, disorders, diseases | diabetes, hypertension |
| DRUG | Medications, pharmaceuticals | metformin, insulin |
| PROCEDURE | Medical procedures, tests, surgeries | MRI, biopsy, dialysis |
| ANATOMY | Body parts, organs, tissues | heart, liver, kidney |

---

## 📁 Repository Structure

```
project16_domain_ner/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── app.py                         # Streamlit GUI application
├── data/
│   ├── README.md                  # Dataset instructions
│   └── sample_biomedical_ner.conll # Sample CoNLL-format training data
├── models/
│   └── biobert_ner/               # Fine-tuned model weights (created after training)
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Data loading & CoNLL parsing
│   ├── train_ner.py               # Model fine-tuning script
│   ├── evaluate.py                # Evaluation metrics (Precision, Recall, F1)
│   └── inference.py               # Prediction logic (transformer + rule-based fallback)
└── screenshots/                   # GUI screenshots (add your own here)
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd project16_domain_ner
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Train the Model

```bash
python -m src.train_ner
```

This will:
1. Load `data/sample_biomedical_ner.conll`
2. Split into train/validation sets
3. Fine-tune BioBERT for token classification
4. Save the model to `models/biobert_ner/`
5. Print a classification report (Precision, Recall, F1-score)

> **Note:** Replace `data/sample_biomedical_ner.conll` with your own labelled biomedical dataset in CoNLL format (one token-label pair per line, blank line between sentences).

---

## 🖥️ How to Run the GUI

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

### GUI Features
1. **Single Text Tab** – Paste biomedical text and click *Extract Entities*
2. **Upload Document Tab** – Upload a `.txt` file and extract entities
3. **Sidebar** – Choose model (Fine-tuned BioBERT / Base BioBERT / Rule-Based Demo) and set confidence threshold
4. **Output** – Highlighted text + entity table with confidence scores

---

## 📊 Evaluation

Run evaluation after training:

```bash
python -m src.evaluate
```

Reports:
- Precision, Recall, F1-score per entity type
- Macro/micro averages
- Confusion-style entity distribution

---

## 📄 Dataset Format (CoNLL)

Each line contains a token and its BIO label separated by a space. Empty lines separate sentences.

Example:
```
The O
patient O
was O
diagnosed B-DISEASE
with O
type O
2 O
diabetes I-DISEASE
mellitus I-DISEASE
. O

The O
patient O
is O
taking B-DRUG
metformin I-DRUG
. O
```

---

## 🧠 Algorithms & Techniques

| Stage | Technique |
|-------|-----------|
| Data Collection | Curated biomedical text (sample included; replace with real corpus) |
| Preprocessing | CoNLL parsing, BIO tagging, train/val split |
| Feature Engineering | BioBERT WordPiece tokenization, subword alignment |
| Model | `AutoModelForTokenClassification` over `dmis-lab/biobert-base-cased-v1.1` |
| Training | AdamW optimizer, linear warmup scheduler, early stopping |
| Evaluation | seqeval-style Precision / Recall / F1-score |
| Inference | Hugging Face `pipeline` with `aggregation_strategy="simple"` |

---

## 📸 GUI Screenshots

Add screenshots of your running Streamlit app to the `screenshots/` folder:
- `screenshots/home.png`
- `screenshots/entities_highlighted.png`
- `screenshots/entity_table.png`

---

## ✅ Submission Checklist

- [ ] Complete source code uploaded to GitHub
- [ ] `requirements.txt` included
- [ ] `README.md` with setup & execution steps
- [ ] Trained model folder (`models/biobert_ner/`) or instructions to train
- [ ] Screenshots of the working GUI
- [ ] Jupyter notebook (optional but recommended) showing EDA and training
- [ ] PPT covering both group projects (Project 16 + Project 1)

---

## 👥 Contributions

All three group members contributed equally:
- Dataset collection & preprocessing
- Model fine-tuning & evaluation
- Streamlit GUI development
- Testing, documentation & GitHub deployment
