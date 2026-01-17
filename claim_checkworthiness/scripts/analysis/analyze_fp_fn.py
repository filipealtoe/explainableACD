#!/usr/bin/env python3
"""
Analyze False Positives and False Negatives

Extracts actual text examples of FP/FN for manual analysis.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"


def main():
    print("=" * 100)
    print("FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
    print("=" * 100)

    # Load data
    embed_data = np.load(EMBEDDING_FILE)
    X_train = embed_data["train"].astype(np.float32)
    X_test = embed_data["test"].astype(np.float32)

    df_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    df_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (df_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (df_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    # Train model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    # Get text column
    text_col = "cleaned_text" if "cleaned_text" in df_test.columns else "Text"
    texts = df_test[text_col].to_list()

    # Categorize predictions
    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0]
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]
    tn_idx = np.where((y_pred == 0) & (y_test == 0))[0]

    print(f"\n  Total test samples: {len(y_test)}")
    print(f"  True Positives:  {len(tp_idx)}")
    print(f"  False Positives: {len(fp_idx)}")
    print(f"  False Negatives: {len(fn_idx)}")
    print(f"  True Negatives:  {len(tn_idx)}")

    # =========================================================================
    # FALSE POSITIVES - Predicted Yes, Actually No
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"FALSE POSITIVES ({len(fp_idx)} cases)")
    print("Predicted: CHECKWORTHY | Actual: NOT CHECKWORTHY")
    print("=" * 100)

    # Sort by confidence (highest confidence FPs are most concerning)
    fp_sorted = sorted(fp_idx, key=lambda i: y_proba[i], reverse=True)

    for rank, idx in enumerate(fp_sorted[:15], 1):
        text = texts[idx]
        prob = y_proba[idx]
        print(f"\n  [{rank}] Confidence: {prob:.3f}")
        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  {'-'*80}")

    # =========================================================================
    # FALSE NEGATIVES - Predicted No, Actually Yes
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"FALSE NEGATIVES ({len(fn_idx)} cases)")
    print("Predicted: NOT CHECKWORTHY | Actual: CHECKWORTHY")
    print("=" * 100)

    # Sort by confidence (lowest confidence = most missed)
    fn_sorted = sorted(fn_idx, key=lambda i: y_proba[i])

    for rank, idx in enumerate(fn_sorted[:15], 1):
        text = texts[idx]
        prob = y_proba[idx]
        print(f"\n  [{rank}] Confidence: {prob:.3f}")
        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  {'-'*80}")

    # =========================================================================
    # Pattern Analysis
    # =========================================================================
    print("\n" + "=" * 100)
    print("PATTERN ANALYSIS")
    print("=" * 100)

    # Word count analysis
    fp_lengths = [len(texts[i].split()) for i in fp_idx]
    fn_lengths = [len(texts[i].split()) for i in fn_idx]
    tp_lengths = [len(texts[i].split()) for i in tp_idx]
    tn_lengths = [len(texts[i].split()) for i in tn_idx]

    print(f"\n  Average word count by prediction type:")
    print(f"    True Positives:  {np.mean(tp_lengths):.1f} words")
    print(f"    False Positives: {np.mean(fp_lengths):.1f} words")
    print(f"    False Negatives: {np.mean(fn_lengths):.1f} words")
    print(f"    True Negatives:  {np.mean(tn_lengths):.1f} words")

    # Confidence distribution
    print(f"\n  Confidence distribution:")
    print(f"    True Positives:  mean={np.mean([y_proba[i] for i in tp_idx]):.3f}")
    print(f"    False Positives: mean={np.mean([y_proba[i] for i in fp_idx]):.3f}")
    print(f"    False Negatives: mean={np.mean([y_proba[i] for i in fn_idx]):.3f}")
    print(f"    True Negatives:  mean={np.mean([y_proba[i] for i in tn_idx]):.3f}")

    # Common words in FP vs FN
    print(f"\n  Looking for keyword patterns...")

    fp_texts = " ".join([texts[i].lower() for i in fp_idx])
    fn_texts = " ".join([texts[i].lower() for i in fn_idx])
    tp_texts = " ".join([texts[i].lower() for i in tp_idx])

    keywords = ["trump", "biden", "election", "vote", "covid", "vaccine",
                "percent", "million", "billion", "says", "said", "claim",
                "report", "study", "data", "fact", "true", "false",
                "think", "believe", "opinion", "should", "must", "will"]

    print(f"\n  Keyword frequency (per 1000 words):")
    print(f"  {'Keyword':<12} {'TP':<10} {'FP':<10} {'FN':<10}")
    print(f"  {'-'*42}")

    for kw in keywords:
        tp_freq = 1000 * fp_texts.count(kw) / max(len(fp_texts.split()), 1)
        fp_freq = 1000 * fp_texts.count(kw) / max(len(fp_texts.split()), 1)
        fn_freq = 1000 * fn_texts.count(kw) / max(len(fn_texts.split()), 1)

        if abs(fp_freq - fn_freq) > 2 or fp_freq > 5 or fn_freq > 5:
            print(f"  {kw:<12} {tp_freq:<10.1f} {fp_freq:<10.1f} {fn_freq:<10.1f}")

    # =========================================================================
    # Save to file for detailed review
    # =========================================================================
    output_file = Path(__file__).parent.parent / "results" / "fp_fn_analysis.txt"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        f.write("FALSE POSITIVE / FALSE NEGATIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"FALSE POSITIVES ({len(fp_idx)} cases)\n")
        f.write("Predicted: CHECKWORTHY | Actual: NOT CHECKWORTHY\n")
        f.write("-" * 80 + "\n\n")

        for rank, idx in enumerate(fp_sorted, 1):
            f.write(f"[{rank}] Confidence: {y_proba[idx]:.3f}\n")
            f.write(f"Text: {texts[idx]}\n\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"FALSE NEGATIVES ({len(fn_idx)} cases)\n")
        f.write("Predicted: NOT CHECKWORTHY | Actual: CHECKWORTHY\n")
        f.write("-" * 80 + "\n\n")

        for rank, idx in enumerate(fn_sorted, 1):
            f.write(f"[{rank}] Confidence: {y_proba[idx]:.3f}\n")
            f.write(f"Text: {texts[idx]}\n\n")

    print(f"\n  Full analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
