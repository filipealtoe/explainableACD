#!/usr/bin/env python3
"""
Analyze DeBERTa Mispredictions on CT24 Test Set

Extracts and analyzes False Positives and False Negatives
to understand what types of claims the model struggles with.

Usage:
    python experiments/scripts/analyze_deberta_errors.py
    python experiments/scripts/analyze_deberta_errors.py --threshold 0.5
    python experiments/scripts/analyze_deberta_errors.py --output errors.csv

Output:
    - Console summary of error patterns
    - CSV file with all mispredictions for manual analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, f1_score

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
DEBERTA_RESULTS = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "error_analysis"


def main():
    parser = argparse.ArgumentParser(description="Analyze DeBERTa mispredictions")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Classification threshold (default: 0.50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename")
    args = parser.parse_args()

    print("=" * 80)
    print("DEBERTA ERROR ANALYSIS")
    print("=" * 80)

    # Load test data
    print("\nLoading data...")
    test_df = pl.read_parquet(DATA_DIR / "CT24_test_features.parquet")
    probs = np.load(DEBERTA_RESULTS / "test_probs.npy")

    print(f"Test samples: {len(test_df)}")
    print(f"Predictions loaded: {len(probs)}")

    # Get labels and predictions
    y_true = np.array([1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()])
    y_pred = (probs >= args.threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred)

    print(f"\nThreshold: {args.threshold}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:3d}  FP={fp:3d}")
    print(f"  FN={fn:3d}  TP={tp:3d}")

    # Add predictions to dataframe
    test_df = test_df.with_columns([
        pl.Series("prob", probs),
        pl.Series("y_true", y_true),
        pl.Series("y_pred", y_pred),
    ])

    # Classify prediction types
    def get_pred_type(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        elif row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN"
        elif row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        else:
            return "FN"

    pred_types = [get_pred_type({"y_true": yt, "y_pred": yp})
                  for yt, yp in zip(y_true, y_pred)]
    test_df = test_df.with_columns(pl.Series("pred_type", pred_types))

    # Extract errors
    fp_df = test_df.filter(pl.col("pred_type") == "FP")
    fn_df = test_df.filter(pl.col("pred_type") == "FN")

    # ==========================================================================
    # FALSE NEGATIVES (Missed checkworthy claims)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"FALSE NEGATIVES ({len(fn_df)} cases)")
    print("Claims that ARE checkworthy but model predicted NOT checkworthy")
    print("=" * 80)

    if len(fn_df) > 0:
        # Sort by probability (highest first - closest to correct)
        fn_df = fn_df.sort("prob", descending=True)

        print(f"\n{'#':<4} {'Prob':<8} {'Text':<70}")
        print("-" * 85)

        for i, row in enumerate(fn_df.iter_rows(named=True)):
            text = row["Text"][:67] + "..." if len(row["Text"]) > 70 else row["Text"]
            print(f"{i+1:<4} {row['prob']:<8.3f} {text}")

        # Statistics
        fn_probs = fn_df["prob"].to_numpy()
        print(f"\nFN Probability Statistics:")
        print(f"  Mean: {fn_probs.mean():.3f}")
        print(f"  Median: {np.median(fn_probs):.3f}")
        print(f"  Min: {fn_probs.min():.3f}")
        print(f"  Max: {fn_probs.max():.3f}")

        # How many would be recovered with lower threshold
        for t in [0.45, 0.40, 0.35, 0.30]:
            recovered = (fn_probs >= t).sum()
            print(f"  Would recover {recovered}/{len(fn_df)} with threshold {t}")

    # ==========================================================================
    # FALSE POSITIVES (Incorrectly flagged as checkworthy)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"FALSE POSITIVES ({len(fp_df)} cases)")
    print("Claims that are NOT checkworthy but model predicted checkworthy")
    print("=" * 80)

    if len(fp_df) > 0:
        # Sort by probability (lowest first - closest to correct)
        fp_df = fp_df.sort("prob", descending=False)

        print(f"\n{'#':<4} {'Prob':<8} {'Text':<70}")
        print("-" * 85)

        for i, row in enumerate(fp_df.iter_rows(named=True)):
            text = row["Text"][:67] + "..." if len(row["Text"]) > 70 else row["Text"]
            print(f"{i+1:<4} {row['prob']:<8.3f} {text}")

        # Statistics
        fp_probs = fp_df["prob"].to_numpy()
        print(f"\nFP Probability Statistics:")
        print(f"  Mean: {fp_probs.mean():.3f}")
        print(f"  Median: {np.median(fp_probs):.3f}")
        print(f"  Min: {fp_probs.min():.3f}")
        print(f"  Max: {fp_probs.max():.3f}")

        # How many would be avoided with higher threshold
        for t in [0.55, 0.60, 0.65, 0.70]:
            avoided = (fp_probs < t).sum()
            print(f"  Would avoid {avoided}/{len(fp_df)} with threshold {t}")

    # ==========================================================================
    # TEXT PATTERN ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEXT PATTERN ANALYSIS")
    print("=" * 80)

    def analyze_patterns(df, label):
        if len(df) == 0:
            return

        texts = df["Text"].to_list()

        # Length statistics
        lengths = [len(t.split()) for t in texts]
        print(f"\n{label} - Word count:")
        print(f"  Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}")

        # Question marks
        questions = sum(1 for t in texts if "?" in t)
        print(f"  Contains '?': {questions}/{len(texts)} ({100*questions/len(texts):.1f}%)")

        # Numbers
        has_numbers = sum(1 for t in texts if any(c.isdigit() for c in t))
        print(f"  Contains numbers: {has_numbers}/{len(texts)} ({100*has_numbers/len(texts):.1f}%)")

        # Percentages
        has_percent = sum(1 for t in texts if "%" in t or "percent" in t.lower())
        print(f"  Contains %/percent: {has_percent}/{len(texts)} ({100*has_percent/len(texts):.1f}%)")

        # Quotes
        has_quotes = sum(1 for t in texts if '"' in t or "said" in t.lower() or "says" in t.lower())
        print(f"  Contains quotes/said: {has_quotes}/{len(texts)} ({100*has_quotes/len(texts):.1f}%)")

        # Opinion words
        opinion_words = ["think", "believe", "feel", "opinion", "seems", "appears"]
        has_opinion = sum(1 for t in texts if any(w in t.lower() for w in opinion_words))
        print(f"  Contains opinion words: {has_opinion}/{len(texts)} ({100*has_opinion/len(texts):.1f}%)")

        # First person
        first_person = ["I ", "I'", "my ", "we ", "our "]
        has_first = sum(1 for t in texts if any(w in t for w in first_person))
        print(f"  First person (I/we): {has_first}/{len(texts)} ({100*has_first/len(texts):.1f}%)")

    analyze_patterns(fn_df, "FALSE NEGATIVES")
    analyze_patterns(fp_df, "FALSE POSITIVES")

    # Compare with correct predictions
    tp_df = test_df.filter(pl.col("pred_type") == "TP")
    tn_df = test_df.filter(pl.col("pred_type") == "TN")
    analyze_patterns(tp_df, "TRUE POSITIVES (for comparison)")
    analyze_patterns(tn_df, "TRUE NEGATIVES (for comparison)")

    # ==========================================================================
    # SAVE TO CSV
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = args.output if args.output else f"deberta_errors_thresh{args.threshold}.csv"
    output_path = OUTPUT_DIR / output_file

    # Combine FP and FN for export
    errors_df = pl.concat([fn_df, fp_df])

    # Select relevant columns
    export_cols = ["Sentence_id", "Text", "class_label", "prob", "pred_type"]
    available_cols = [c for c in export_cols if c in errors_df.columns]
    errors_df = errors_df.select(available_cols).sort("pred_type", "prob")

    errors_df.write_csv(output_path)
    print(f"\n{'='*80}")
    print(f"Errors exported to: {output_path}")
    print(f"Total errors: {len(errors_df)} (FN={len(fn_df)}, FP={len(fp_df)})")

    # Also save full predictions for further analysis
    full_output = OUTPUT_DIR / f"deberta_all_predictions_thresh{args.threshold}.csv"
    test_df.select(available_cols + ["y_true", "y_pred"]).write_csv(full_output)
    print(f"Full predictions: {full_output}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nAt threshold {args.threshold}:")
    print(f"  F1: {f1:.4f}")
    print(f"  False Negatives: {len(fn_df)} (missed checkworthy claims)")
    print(f"  False Positives: {len(fp_df)} (incorrectly flagged)")

    if len(fn_df) > 0:
        print(f"\nFN Recovery potential:")
        fn_probs = fn_df["prob"].to_numpy()
        recoverable = (fn_probs >= 0.40).sum()
        print(f"  {recoverable}/{len(fn_df)} FNs have prob >= 0.40 (recoverable with lower threshold)")

    if len(fp_df) > 0:
        print(f"\nFP Avoidance potential:")
        fp_probs = fp_df["prob"].to_numpy()
        avoidable = (fp_probs < 0.60).sum()
        print(f"  {avoidable}/{len(fp_df)} FPs have prob < 0.60 (avoidable with higher threshold)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
