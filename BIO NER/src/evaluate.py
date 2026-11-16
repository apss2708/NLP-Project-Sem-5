"""Evaluate the fine-tuned NER model using Precision, Recall, and F1-score."""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report
from src.preprocess import read_conll, build_label_maps, split_data, NERDataset


MODEL_DIR = os.path.join("models", "biobert_ner")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
DATA_PATH = os.path.join("data", "sample_biomedical_ner.conll")
MAX_LEN = 128
BATCH_SIZE = 8


def evaluate():
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        print("Train the model first: python -m src.train_ner")
        return

    sentences, labels = read_conll(DATA_PATH)
    id2label, label2id = build_label_maps(labels)
    splits = split_data(sentences, labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = NERDataset(
        splits["test"][0], splits["test"][1], tokenizer, label2id
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

            for i in range(input_ids.shape[0]):
                true_sent = []
                pred_sent = []
                for j in range(input_ids.shape[1]):
                    label_id = labels_batch[i][j].item()
                    if label_id == -100:
                        continue
                    true_sent.append(id2label[label_id])
                    pred_sent.append(id2label[predictions[i][j].item()])
                true_labels.append(true_sent)
                pred_labels.append(pred_sent)

    print("\n===== NER Evaluation Report =====\n")
    print(classification_report(true_labels, pred_labels, digits=4))


if __name__ == "__main__":
    evaluate()
