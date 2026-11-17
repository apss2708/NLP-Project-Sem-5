"""Text preprocessing for Sentiment Analysis (Project 1)."""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


# Ensure resources are available
download_nltk_resources()

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove HTML tags, URLs, punctuation, digits."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text) # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)          # remove emails
    text = re.sub(r"[^a-zA-Z\s]", " ", text)      # keep only letters
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, remove stopwords, and lemmatize."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocess(text: str, return_tokens: bool = False):
    """Full preprocessing pipeline: clean → tokenize → lemmatize."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    if return_tokens:
        return cleaned, tokens
    return " ".join(tokens)


def preprocess_series(texts, return_tokens: bool = False):
    """Preprocess a pandas Series of texts."""
    if return_tokens:
        cleaned_list, tokens_list = zip(*[preprocess(t, return_tokens=True) for t in texts])
        return list(cleaned_list), list(tokens_list)
    return [preprocess(t) for t in texts]
