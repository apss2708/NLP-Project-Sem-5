# 🧬 Project Report – Project 16

## Domain-Specific NER with Transformers (Biomedical / Healthcare)

---

### Group 8 – NLP Capstone Project (Semester 5)

| Member | Roll No. |
|--------|----------|
| Ayush Singh | 2401730015 |
| Krish Jaidka | 2401730188 |
| Sahil Shrivastava | 2401730016 |

**Domain:** Biomedical & Healthcare  
**Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`)  
**Interface:** Streamlit Web Application  
**GitHub Repository:** https://github.com/apss2708/NLP-Project-Sem-5  
**Project Folder:** `BIO NER/`

---

## 1. Abstract

This report presents **BioNER Studio**, a domain-specific Named Entity Recognition (NER) system built for the biomedical and healthcare domain. The system extracts four categories of specialized entities from clinical text: **DISEASE**, **DRUG**, **PROCEDURE**, and **ANATOMY**. The model is based on the **BioBERT** transformer architecture (`dmis-lab/biobert-base-cased-v1.1`), fine-tuned for token classification on a labelled biomedical corpus in CoNLL format. A polished Streamlit GUI provides users with an interactive interface to input clinical text, visualize extracted entities with color-coded highlighting, view confidence scores, and download results as CSV. A rule-based fallback ensures the application remains functional for demonstration even without trained model weights.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Statement
Clinical and biomedical documents contain critical information embedded in unstructured text—diagnoses, medications, procedures, and anatomical references. Manually extracting these entities is time-consuming and error-prone. There is a need for an automated NLP system that can accurately identify and categorize domain-specific entities from clinical narratives.

### 2.2 Objectives
1. Collect and prepare a labelled biomedical NER dataset in CoNLL format.
2. Perform text cleaning, preprocessing, and BIO tagging.
3. Conduct exploratory data analysis (EDA) to understand entity distributions.
4. Fine-tune a pretrained BioBERT transformer model for token classification.
5. Evaluate model performance using Precision, Recall, and F1-score.
6. Build a user-friendly Streamlit GUI with entity highlighting, confidence scores, and CSV export.
7. Ensure the system is demonstrable via a rule-based fallback when model weights are unavailable.

---

## 3. Real-World Applications

- **Electronic Health Records (EHR):** Automatically extract diagnoses and medications from doctor's notes.
- **Clinical Research:** Screen large volumes of biomedical literature for specific diseases, drugs, or procedures.
- **Pharmacovigilance:** Identify drug mentions and adverse event-related entities in clinical text.
- **Medical Coding Assistance:** Support ICD / CPT coding by highlighting relevant entities in patient records.
- **Clinical Decision Support:** Populate structured fields in healthcare information systems from unstructured notes.

---

## 4. Dataset Description

### 4.1 Source
A labelled biomedical corpus in **CoNLL format** was used. Each line contains a token and its BIO label separated by a space; blank lines separate sentences.

### 4.2 Format Example
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
```

### 4.3 Entity Labels
| Label | Description | Example |
|-------|-------------|---------|
| DISEASE | Medical conditions, disorders, diseases | diabetes, hypertension, pneumonia |
| DRUG | Medications, pharmaceuticals | metformin, lisinopril, insulin |
| PROCEDURE | Medical procedures, tests, surgeries | MRI, biopsy, dialysis |
| ANATOMY | Body parts, organs, tissues | heart, liver, kidney, lung |

### 4.4 Dataset Statistics (Sample)
| Metric | Value |
|--------|-------|
| Total Sentences | 18 |
| Total Tokens | ~220 |
| Entity Types | 4 (DISEASE, DRUG, PROCEDURE, ANATOMY) |
| Avg Tokens per Sentence | ~12 |

> **Note:** The included `sample_biomedical_ner.conll` is a small synthetic dataset for demonstration. For production, replace it with a larger real-world corpus such as NCBI Disease, BC5CDR, or i2b2.

---

## 5. Data Preprocessing

### 5.1 Data Cleaning
- Removed empty lines and whitespace from token-label pairs.
- Validated CoNLL format (each non-empty line must contain at least a token and a label).
- Handled sentence boundary detection via blank-line separation.

### 5.2 Text Preprocessing
- **Tokenization:** Whitespace-based tokenization aligned with CoNLL format.
- **BIO Tagging:** Labels use the `B-` (begin) and `I-` (inside) convention; `O` denotes tokens outside any entity.
- **Subword Alignment:** BioBERT's WordPiece tokenizer splits words into subwords. Only the first subword of each word retains the original label; subsequent subwords are masked with `-100` (ignored in loss computation).
- **Train/Val/Test Split:** 70% train, 10% validation, 20% test (stratified random split).

---

## 6. Exploratory Data Analysis (EDA)

EDA was performed in `EDA.ipynb` and included:

- **Entity type distribution:** DISEASE and DRUG are the most frequent entities.
- **BIO tag distribution:** Analysis of `B-` vs `I-` vs `O` tag frequency.
- **Sentence length distribution:** Histogram and boxplot of tokens per sentence.
- **Top tokens within entities:** Most frequent words that appear inside entity spans.
- **Entities per sentence:** Distribution of entity token counts per sentence.

Key visualizations generated:
- Bar chart of entity type counts.
- Bar chart of BIO tag prefixes.
- Histogram of sentence lengths.
- Top-15 entity token bar chart.

---

## 7. Feature Engineering

### 7.1 Token Representation
- **BioBERT WordPiece Tokenization:** Clinical text is tokenized using BioBERT's pretrained WordPiece vocabulary, optimized for biomedical text.
- **Subword Label Alignment:** For each word, the first subword inherits the word's BIO label; remaining subwords are assigned `-100` (ignored during training).

### 7.2 Model Input Features
- `input_ids`: Token IDs from BioBERT tokenizer.
- `attention_mask`: Indicates which tokens are real vs padding.
- `labels`: Aligned BIO label IDs with `-100` for special/padding tokens.

---

## 8. Methodology & Model Architecture

### 8.1 Model Selection
**BioBERT** (`dmis-lab/biobert-base-cased-v1.1`) was selected because it is a domain-specific BERT model pretrained on large-scale biomedical corpora (PubMed abstracts, PMC full-text articles), making it well-suited for clinical NER.

### 8.2 Architecture
```
Input Text → BioBERT Tokenizer → WordPiece Tokens → BioBERT Encoder
                                                      ↓
                                             Token Classification Head
                                                      ↓
                                             BIO Label Predictions
```

- **Base Model:** BioBERT-base-cased (12 layers, 768 hidden size, 12 attention heads).
- **Classification Head:** Linear layer projecting 768-dimensional hidden states to `num_labels` (9 labels: B-/I- for 4 entity types + `O`).
- **Loss Function:** Cross-Entropy Loss (ignoring `-100` masked positions).
- **Optimizer:** AdamW with learning rate `3e-5`.
- **Scheduler:** Linear warmup scheduler.
- **Epochs:** 3.
- **Batch Size:** 8.
- **Max Sequence Length:** 128 tokens.
- **Gradient Clipping:** Max norm 1.0.

### 8.3 Training
Run: `python -m src.train_ner`

The best model (lowest validation loss) is saved to `models/biobert_ner/`.

### 8.4 Evaluation
Run: `python -m src.evaluate`

Metrics computed using `seqeval`:
- **Precision:** Proportion of predicted entities that are correct.
- **Recall:** Proportion of true entities that were predicted.
- **F1-Score:** Harmonic mean of Precision and Recall.
- Reported per entity type (DISEASE, DRUG, PROCEDURE, ANATOMY) and as macro/micro averages.

---

## 9. Model Evaluation

Evaluation is performed on a held-out test set. The `src/evaluate.py` script loads the fine-tuned model, runs inference on test sentences, and prints a classification report with Precision, Recall, and F1-score per entity type and overall averages.

> **Note:** Because the included sample dataset is small (~18 sentences), evaluation scores reflect performance on a demonstration set. Training on a larger real-world biomedical corpus will significantly improve scores.

---

## 10. GUI Development (Streamlit)

### 10.1 Interface Overview
A polished, website-style Streamlit application (`app.py`) with the following features:

| Feature | Description |
|---------|-------------|
| Hero Header | Gradient banner with project title, team members, roll numbers |
| Model Selector | Choose between BioBERT (Fine-tuned), BioBERT (Base), or Rule-Based Demo |
| Confidence Threshold Slider | Filter entities by minimum confidence (0.0 – 1.0) |
| Single Text Tab | Paste clinical text, extract entities, view results |
| Upload Document Tab | Upload `.txt` files, extract entities from documents |
| Entity Stats Dashboard | Count cards per entity type with average confidence |
| Entity Chips | Color-coded pills for each detected entity with confidence score |
| Highlighted Text | Inline color-coded highlighting with hover tooltips |
| Entity Table | Tabular view (Category, Text, Start, End, Confidence) |
| CSV Download | Download extracted entities as CSV |

### 10.2 Entity Color Coding
| Entity | Color | Example |
|--------|-------|---------|
| DISEASE | Red (#ef4444) | diabetes, hypertension |
| DRUG | Green (#10b981) | metformin, lisinopril |
| PROCEDURE | Blue (#2563eb) | MRI, biopsy |
| ANATOMY | Amber (#f59e0b) | heart, liver |

### 10.3 Rule-Based Fallback
If trained BioBERT weights are unavailable, the app automatically falls back to a lightweight rule-based predictor using regex patterns for common biomedical terms. This ensures the GUI is always demonstrable.

### 10.4 How to Run the GUI
```bash
cd "BIO NER"
pip install -r requirements.txt
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 11. Testing & Implementation

### 11.1 Manual Testing
The following test cases were used during development:

| Test Input | Expected Entities |
|------------|-------------------|
| "The patient has diabetes and takes metformin." | DISEASE: diabetes; DRUG: metformin |
| "She underwent an MRI of the brain." | PROCEDURE: MRI; ANATOMY: brain |
| "The doctor prescribed lisinopril for hypertension." | DRUG: lisinopril; DISEASE: hypertension |

### 11.2 Tested Environment
- macOS Golden Gate 27.2.1 (macOS 26)
- Python 3.13
- Streamlit 1.28.0, Transformers 4.46.0, Torch 2.x, scikit-learn 1.3.0

---

## 12. Algorithms & Techniques Used

| Stage | Technique |
|-------|-----------|
| Data Collection | Curated biomedical text (CoNLL format) |
| Preprocessing | CoNLL parsing, BIO tagging, train/val/test split |
| Feature Engineering | BioBERT WordPiece tokenization + subword label alignment |
| Model | BioBERT-base-cased + Token Classification Head |
| Training | AdamW optimizer, linear warmup, gradient clipping |
| Evaluation | seqeval (Precision, Recall, F1-score) |
| Inference | Hugging Face `pipeline` with `aggregation_strategy="simple"` |
| GUI | Streamlit with custom CSS |

---

## 13. Limitations

1. **Small Sample Dataset:** The included `sample_biomedical_ner.conll` contains only ~18 sentences. Real-world performance requires a larger labelled corpus (e.g., NCBI Disease, BC5CDR, i2b2).
2. **Model Weights Not Included:** Trained weights (`models/biobert_ner/`) are gitignored due to size. Users must run `python -m src.train_ner` to generate them.
3. **Single-Language:** Currently supports English clinical text only.
4. **Rule-Based Fallback Accuracy:** The demo fallback uses regex patterns and is less accurate than the fine-tuned transformer.
5. **No Clinical Validation:** This is an educational capstone project; predictions are assistive, not clinical diagnoses.

---

## 14. Future Scope

- Replace sample dataset with a large-scale biomedical NER corpus (NCBI Disease, BC5CDR, i2b2).
- Add support for more entity types (e.g., DOSAGE, FREQUENCY, ROUTE).
- Deploy as a cloud web application (Streamlit Cloud, Hugging Face Spaces, or AWS).
- Add PDF/DOCX document parsing for direct clinical document upload.
- Integrate with EHR systems via FHIR APIs.
- Experiment with other transformer models (ClinicalBERT, BioClinicalBERT, PubMedBERT).
- Add multi-language support for clinical text.

---

## 15. Conclusion

BioNER Studio successfully demonstrates an end-to-end domain-specific NER pipeline—from dataset collection and preprocessing to BioBERT fine-tuning, evaluation, and a production-style Streamlit GUI. The system accurately identifies biomedical entities (DISEASE, DRUG, PROCEDURE, ANATOMY) from clinical text and presents results through an intuitive interface with confidence scores and CSV export. The rule-based fallback ensures the application remains fully demonstrable for presentations and academic evaluation.

---

## 16. Team Contributions

All three group members contributed equally to both projects in Group 8:

- **Ayush Singh (2401730015):** Dataset collection, preprocessing, EDA notebook, model training, GUI development, documentation.
- **Krish Jaidka (2401730188):** Model architecture design, fine-tuning script, evaluation metrics, testing, GitHub setup.
- **Sahil Shrivastava (2401730016):** Streamlit UI design, CSS styling, rule-based fallback, screenshots, report writing.

---

## 17. References

1. Lee, J., et al. (2020). "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." *Bioinformatics*.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.
3. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
4. CoNLL-2003 Shared Task: Named Entity Recognition.
5. Streamlit Documentation: https://docs.streamlit.io

---

**Educational capstone project. Model predictions are assistive, not clinical diagnoses.**
