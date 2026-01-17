#!/usr/bin/env python3
"""
Evaluate DeBERTa on dev-test set (317 samples) for error analysis.

Usage:
    python experiments/scripts/evaluate_deberta_devtest.py
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DEV_TEST_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_dev-test.tsv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large"


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

    # Load dev-test
    df = pl.read_csv(DEV_TEST_PATH, separator="\t")
    texts = df["Text"].to_list()
    y_true = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
    print(f"Dev-test: {len(texts)} samples, {sum(y_true)} positive ({100*sum(y_true)/len(y_true):.1f}%)")

    # Inference
    probs = []
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prob = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
            probs.append(prob)
    probs = np.array(probs)

    # Results at threshold 0.50
    y_pred = (probs >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n=== Results @ threshold 0.50 ===")
    print(f"F1: {f1_score(y_true, y_pred):.4f}  Acc: {accuracy_score(y_true, y_pred):.4f}")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # Save predictions with errors marked
    df = df.with_columns([
        pl.Series("prob", probs),
        pl.Series("pred", ["Yes" if p >= 0.5 else "No" for p in probs]),
    ])
    df.write_csv(OUTPUT_DIR / "devtest_predictions.csv")
    print(f"\nSaved: {OUTPUT_DIR / 'devtest_predictions.csv'}")


if __name__ == "__main__":
    main()
