# 💬 Project 1 – Sentiment Analysis

**Group 8**  
**Team Members:** Ayush Singh (2401730015) · Krish Jaidka (2401730188) · Sahil Shrivastava (2401730016)

**Objective:** Identify whether a given sentence, review, or feedback is Positive, Negative, or Neutral.  
**Model:** TF-IDF + ML (Logistic Regression, Naive Bayes, SVM)  
**GUI:** Streamlit web application  
**Dataset:** IMDb Movie Reviews (suggested)

---

## 📋 Project Overview

This project builds a Sentiment Analysis system that classifies text input as Positive, Negative, or Neutral.

### What the system does
- Accepts raw text input (movie/product reviews)
- Cleans and preprocesses text (tokenization, stopword removal, lemmatization)
- Converts text into numerical features using TF-IDF
- Trains machine learning classification models
- Predicts sentiment of new user-entered text with confidence scores

### Supported Sentiment Classes
| Label | Emoji | Description |
|-------|-------|-------------|
| Positive | 😊 | Favorable review |
| Negative | 😞 | Unfavorable review |
| Neutral | 😐 | Mixed or indifferent review |

---

## 📁 Repository Structure

```
Sentiment analysis/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── app.py                         # Streamlit GUI application
├── .streamlit/
│   └── config.toml                # Streamlit theme config
├── .gitignore                     # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Text cleaning & preprocessing
│   ├── train_model.py             # TF-IDF + ML model training
│   ├── evaluate.py                # Accuracy / Precision / Recall / F1
│   └── inference.py               # Prediction logic (ML + rule-based fallback)
├── data/
│   ├── sample_imdb.csv            # Sample labelled sentiment data
│   └── README.md                  # Dataset format instructions
├── models/                        # Trained model weights (gitignored – retrain after clone)
├── EDA.ipynb                      # Exploratory Data Analysis notebook
├── Project_Report.md              # Full project report
└── screenshots/                   # GUI screenshots (add your own here)
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "NLP Project Sem 5/Sentiment analysis"
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
python -m src.train_model
```

This will:
1. Load `data/sample_imdb.csv`
2. Preprocess text (clean, tokenize, lemmatize)
3. Split into train/test sets
4. Vectorize using TF-IDF
5. Train the selected ML model (Logistic Regression by default)
6. Save the model to `models/sentiment_logistic_regression.joblib`
7. Print evaluation metrics

> **Note:** Replace `data/sample_imdb.csv` with a larger IMDb Movie Reviews dataset for better results.

---

## 🖥️ How to Run the GUI

```bash
streamlit run app.py
```

Then open: **http://localhost:8501**

### GUI Features
1. **Single Review Tab** – Enter review text and analyze sentiment
2. **Upload File Tab** – Upload a CSV with a `text` column for batch analysis
3. **Sidebar** – Choose model, set confidence threshold, view class info
4. **Output** – Sentiment badge, confidence breakdown bar chart, download CSV

---

## 📊 Evaluation

Run evaluation after training:
```bash
python -m src.evaluate
```

Reports:
- Accuracy, Precision, Recall, F1-score
- Per-class classification report
- Confusion matrix

---

## 📄 Dataset Format (CSV)

Each row contains a review and its sentiment label.

Example:
```
text,label
"The movie was fantastic and I loved every minute of it.",Positive
"Terrible plot, terrible acting, complete waste of time.",Negative
"It was okay, nothing special but watchable.",Neutral
```

**Required columns:** `text`, `label`

---

## 🧠 Algorithms & Techniques

| Stage | Technique |
|-------|-----------|
| Data Collection | IMDb Movie Reviews / labelled review corpus |
| Preprocessing | HTML/URL removal, stopword removal, lemmatization |
| Feature Engineering | TF-IDF vectorization (unigrams + bigrams) |
| Model | Logistic Regression / Naive Bayes / Linear SVM |
| Training | Train/test split, stratified sampling |
| Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Inference | Trained pipeline or rule-based fallback |

---

## 📸 GUI Screenshots

Add screenshots of your running Streamlit app to the `screenshots/` folder:
- `screenshots/hero_header.png`
- `screenshots/single_review.png`
- `screenshots/prediction_result.png`
- `screenshots/batch_upload.png`

---

## ✅ Submission Checklist

- [ ] Complete source code uploaded to GitHub
- [ ] `requirements.txt` included
- [ ] `README.md` with setup & execution steps
- [ ] Trained model folder (`models/`) or instructions to train
- [ ] Screenshots of the working GUI
- [ ] Jupyter notebook showing EDA and training
- [ ] PPT covering both group projects (Project 16 + Project 1)
- [ ] GitHub repository link submitted via Google Form

---

## 👥 Contributions

All three group members contributed equally:
- Dataset collection & preprocessing
- Model development & evaluation
- Streamlit GUI development
- Testing, documentation & GitHub deployment

---

**Educational capstone project.** Model predictions are assistive, not clinical diagnoses.
