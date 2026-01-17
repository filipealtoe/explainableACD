#!/usr/bin/env python3
"""
Analyze CT24 test errors for a trained classifier run.

Outputs a TSV with misclassified claims, including text, gold/pred labels,
prediction probabilities, and simple text stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.ct24_classifier_core import preprocess_tsv_dataset

DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ct24_classifier_pca_32"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_embeddings" / "pca_32"
CONFIDENCE_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_confidences"


def find_latest_timestamp(results_dir: Path) -> str:
    metrics_files = sorted(results_dir.glob("metrics_summary_*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics_summary_*.json in {results_dir}")
    latest = metrics_files[-1]
    return latest.stem.replace("metrics_summary_", "")


def find_latest_parquet(split: str) -> Path:
    pattern = f"CT24_{split}_with_confidences_*.parquet"
    files = sorted(CONFIDENCE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files for {split} in {CONFIDENCE_DIR}")
    return files[-1]


def load_gold_with_text(parquet_path: Path) -> pd.DataFrame:
    df = pl.read_parquet(parquet_path).select(["sample_id", "text", "label"])
    return df.to_pandas()


def load_predictions(
    pred_path: Path,
    sample_id_col: str = "Sentence_id",
    pred_col: str = "Label",
) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_path, sep="\t")
    pred_df = pred_df.rename(columns={sample_id_col: "sample_id", pred_col: "pred_label"})
    return pred_df


def add_probabilities(
    data_dir: Path,
    model,
    label_encoder,
    split: str,
) -> pd.DataFrame:
    split_path = data_dir / f"{split}.tsv"
    mapping_path = data_dir / f"{split}.mapping.tsv"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split}.tsv in {data_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing {split}.mapping.tsv in {data_dir}")

    X_split, _, _ = preprocess_tsv_dataset(
        str(split_path), label_encoder=label_encoder, fit_encoder=False
    )
    proba = model.predict_proba(X_split)[:, 1]

    mapping_df = pd.read_csv(mapping_path, sep="\t")
    mapping_df["pred_proba_yes"] = proba
    return mapping_df[["sample_id", "pred_proba_yes"]]


def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_len"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().apply(len)
    df["has_number"] = df["text"].astype(str).str.contains(r"\d", regex=True)
    df["has_question"] = df["text"].astype(str).str.contains("?", regex=False)
    df["has_url"] = df["text"].astype(str).str.contains(r"http", regex=True)
    return df


def normalize_pred_labels(
    series: pd.Series, label_encoder: object | None = None
) -> pd.Series:
    """Normalize prediction labels to 'Yes'/'No'."""
    values = series.dropna().astype(str)
    unique_vals = set(values.unique())

    if unique_vals.issubset({"Yes", "No"}):
        return series.astype(str)

    if unique_vals.issubset({"0", "1"}):
        return series.astype(str).map(lambda v: "Yes" if v == "1" else "No")

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        class_strs = {str(c) for c in label_encoder.classes_}
        if class_strs.issubset({"0", "1"}):
            return series.astype(str).map(lambda v: "Yes" if v == "1" else "No")

    return series.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CT24 test errors")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory for the run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory with PCA features",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated splits to analyze (default: test)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp to analyze (default: latest in results dir)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for decision margin (default: from metrics summary)",
    )
    parser.add_argument(
        "--use-predictions-file",
        action="store_true",
        help="Use predictions file if available (only applies to test split)",
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    data_dir = args.data_dir
    timestamp = args.timestamp or find_latest_timestamp(results_dir)

    metrics_path = results_dir / f"metrics_summary_{timestamp}.json"
    model_path = results_dir / f"best_model_{timestamp}.joblib"
    pred_path = results_dir / f"predictions_ct24_{timestamp}.tsv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    threshold = args.threshold
    if threshold is None:
        threshold = metrics.get("threshold_tuned", 0.50)

    model, label_encoder = joblib.load(model_path)

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_list:
        raise ValueError("No splits provided. Use --splits train,dev,test")

    all_errors = []
    for split in split_list:
        gold_parquet = find_latest_parquet(split)
        gold_df = load_gold_with_text(gold_parquet)
        gold_df["sample_id"] = gold_df["sample_id"].astype(str)

        proba_df = add_probabilities(data_dir, model, label_encoder, split)
        proba_df["sample_id"] = proba_df["sample_id"].astype(str)

        merged = gold_df.merge(proba_df, on="sample_id", how="left")

        if split == "test" and args.use_predictions_file and pred_path.exists():
            pred_df = load_predictions(pred_path)
            pred_df["sample_id"] = pred_df["sample_id"].astype(str)
            merged = merged.merge(pred_df, on="sample_id", how="left")
        else:
            pred_int = (merged["pred_proba_yes"] >= threshold).astype(int)
            merged["pred_label"] = pd.Series(pred_int).map(lambda v: "Yes" if v == 1 else "No").values

        merged["gold_label"] = merged["label"]
        merged["pred_label"] = normalize_pred_labels(
            merged["pred_label"], label_encoder=label_encoder
        )
        merged["decision_margin"] = (merged["pred_proba_yes"] - threshold).abs()

        merged["error_type"] = "OK"
        merged.loc[
            (merged["gold_label"] == "Yes") & (merged["pred_label"] == "No"),
            "error_type",
        ] = "FN"
        merged.loc[
            (merged["gold_label"] == "No") & (merged["pred_label"] == "Yes"),
            "error_type",
        ] = "FP"

        merged = compute_text_stats(merged)
        merged["split"] = split

        errors = merged[merged["error_type"] != "OK"].copy()
        errors = errors.sort_values(by="decision_margin", ascending=True)

        output_path = results_dir / f"error_analysis_{split}_{timestamp}.tsv"
        columns = [
            "sample_id",
            "split",
            "gold_label",
            "pred_label",
            "pred_proba_yes",
            "decision_margin",
            "error_type",
            "text_len",
            "word_count",
            "has_number",
            "has_question",
            "has_url",
            "text",
        ]
        errors.to_csv(output_path, sep="\t", index=False, columns=columns)

        fp_count = (errors["error_type"] == "FP").sum()
        fn_count = (errors["error_type"] == "FN").sum()
        print("=" * 70)
        print("CT24 ERROR ANALYSIS")
        print("=" * 70)
        print(f"Timestamp: {timestamp}")
        print(f"Split: {split}")
        print(f"Results dir: {results_dir}")
        print(f"Data dir: {data_dir}")
        print(f"Threshold: {threshold:.2f}")
        print(f"Total errors: {len(errors)}")
        print(f"  False Positives: {fp_count}")
        print(f"  False Negatives: {fn_count}")
        print(f"Output: {output_path}")

        all_errors.append(errors)

    if len(all_errors) > 1:
        combined = pd.concat(all_errors, ignore_index=True)
        combined_path = results_dir / f"error_analysis_{'_'.join(split_list)}_{timestamp}.tsv"
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\nCombined output: {combined_path}")


if __name__ == "__main__":
    main()
