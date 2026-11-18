"""Text preprocessing for Sentiment Analysis (Project 1).

No external NLTK downloads required — uses built-in stopwords list and
regex-based tokenization so the app works out of the box.
"""

import re


# Common English stopwords (built-in, no download needed)
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


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


def tokenize(text: str) -> list:
    """Simple whitespace tokenization after cleaning."""
    return clean_text(text).split()


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def preprocess(text: str, return_tokens: bool = False):
    """Full preprocessing pipeline: clean → tokenize → remove stopwords."""
    cleaned = clean_text(text)
    tokens = remove_stopwords(cleaned.split())
    if return_tokens:
        return cleaned, tokens
    return " ".join(tokens)


def preprocess_series(texts, return_tokens: bool = False):
    """Preprocess a list/Series of texts."""
    if return_tokens:
        cleaned_list, tokens_list = zip(*[preprocess(t, return_tokens=True) for t in texts])
        return list(cleaned_list), list(tokens_list)
    return [preprocess(t) for t in texts]
