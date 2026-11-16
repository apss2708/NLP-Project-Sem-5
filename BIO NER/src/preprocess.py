"""Data loading and preprocessing for biomedical NER (CoNLL format)."""

import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def read_conll(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read a CoNLL-formatted NER file.

    Returns:
        sentences: list of token lists
        labels:    list of label lists (BIO tags)
    """
    sentences: List[List[str]] = []
    labels: List[List[str]] = []
    current_tokens: List[str] = []
    current_labels: List[str] = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Empty line = sentence boundary
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_labels.append(parts[-1])

    # Last sentence (if file doesn't end with blank line)
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def build_label_maps(labels: List[List[str]]) -> Tuple[dict, dict]:
    """Create id2label and label2id dictionaries from label lists."""
    unique_labels = sorted({lbl for sent in labels for lbl in sent})
    id2label = {i: lbl for i, lbl in enumerate(unique_labels)}
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    return id2label, label2id


def split_data(
    sentences: List[List[str]],
    labels: List[List[str]],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Split sentences/labels into train, validation, and test sets.
    """
    # First split: train+val vs test
    train_val_sent, test_sent, train_val_lab, test_lab = train_test_split(
        sentences, labels, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val (relative to train_val)
    relative_val = val_size / (1.0 - test_size)
    train_sent, val_sent, train_lab, val_lab = train_test_split(
        train_val_sent, train_val_lab, test_size=relative_val, random_state=random_state
    )

    return {
        "train": (train_sent, train_lab),
        "val": (val_sent, val_lab),
        "test": (test_sent, test_lab),
    }


class NERDataset(Dataset):
    """PyTorch Dataset for NER with subword label alignment."""

    def __init__(self, sentences, labels, tokenizer, label2id, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        self.input_ids = []
        self.attention_masks = []
        self.label_ids = []

        for sent, lab in zip(sentences, labels):
            tokenized = tokenizer(
                sent,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            word_ids = tokenized.word_ids(batch_index=0)

            aligned = []
            for word_idx in word_ids:
                if word_idx is None:
                    aligned.append(-100)  # special tokens ignored in loss
                else:
                    aligned.append(self.label2id.get(lab[word_idx], self.label2id["O"]))

            self.input_ids.append(tokenized["input_ids"].squeeze(0))
            self.attention_masks.append(tokenized["attention_mask"].squeeze(0))
            self.label_ids.append(torch.tensor(aligned, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.label_ids[idx],
        }
