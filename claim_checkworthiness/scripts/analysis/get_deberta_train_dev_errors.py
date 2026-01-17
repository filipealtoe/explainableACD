#!/usr/bin/env python3
"""
Get DeBERTa mispredictions on train and dev sets.

Usage:
    python experiments/scripts/get_deberta_train_dev_errors.py
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large"

THRESHOLD = 0.50


def run_inference(model, tokenizer, texts, device):
    probs = []
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prob = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
            probs.append(prob)
    return np.array(probs)


def get_errors(df, probs, split_name):
    df = df.with_columns([
        pl.Series("prob", probs),
        pl.Series("pred", ["Yes" if p >= THRESHOLD else "No" for p in probs]),
    ])

    fn = df.filter((pl.col("class_label") == "Yes") & (pl.col("pred") == "No")).with_columns(pl.lit("FN").alias("error_type"))
    fp = df.filter((pl.col("class_label") == "No") & (pl.col("pred") == "Yes")).with_columns(pl.lit("FP").alias("error_type"))

    errors = pl.concat([fn, fp]).sort("error_type", "prob", descending=[False, True])

    output_path = OUTPUT_DIR / f"{split_name}_errors.csv"
    errors.write_csv(output_path)
    print(f"{split_name}: {len(fn)} FN + {len(fp)} FP = {len(errors)} errors -> {output_path}")

    return errors


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

    # Dev set
    print("\n=== DEV SET ===")
    dev_df = pl.read_csv(DATA_DIR / "CT24_checkworthy_english_dev.tsv", separator="\t")
    print(f"Samples: {len(dev_df)}")
    dev_probs = run_inference(model, tokenizer, dev_df["Text"].to_list(), device)
    get_errors(dev_df, dev_probs, "dev")

    # Train set (large - ~22k samples)
    print("\n=== TRAIN SET ===")
    train_df = pl.read_csv(DATA_DIR / "CT24_checkworthy_english_train.tsv", separator="\t")
    print(f"Samples: {len(train_df)}")
    train_probs = run_inference(model, tokenizer, train_df["Text"].to_list(), device)
    get_errors(train_df, train_probs, "train")

    print("\nDone!")


if __name__ == "__main__":
    main()
