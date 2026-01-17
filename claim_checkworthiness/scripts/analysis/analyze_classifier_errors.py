#!/usr/bin/env python3
"""
Error Analysis for CT24 Checkworthiness Classifier

This script analyzes classification errors to understand:
1. False Positives vs False Negatives patterns
2. Confidence score distributions for errors
3. Feature disagreement (self-reported vs logprob)
4. Text patterns in errors (length, keywords)
5. Comparison with GPT-3.5 base predictions
6. Feature importance for misclassified samples

Usage:
    python analyze_classifier_errors.py
    python analyze_classifier_errors.py --threshold 0.50
    python analyze_classifier_errors.py --split test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ct24_classifier"
DATA_DIR = REPO_ROOT / "data" / "processed" / "CT24_classifier_ready"
CONFIDENCE_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_confidences"
OUTPUT_DIR = RESULTS_DIR / "error_analysis"

FEATURE_COLS = [
    "verifiability_cc", "verifiability_logprob",
    "checkability_cc", "checkability_logprob",
    "harmpot_cc", "harmpot_logprob",
]

# Reverse mapping for loading original data
CONFIDENCE_COLS_ORIGINAL = {
    "verifiability_cc": "verifiability_self_conf",
    "verifiability_logprob": "verifiability_logprob_conf",
    "checkability_cc": "checkability_self_conf",
    "checkability_logprob": "checkability_logprob_conf",
    "harmpot_cc": "harm_self_conf",
    "harmpot_logprob": "harm_logprob_conf",
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_model_and_data(split: str = "test"):
    """Load the trained model and data for a split."""
    # Find latest model
    model_files = sorted(RESULTS_DIR.glob("best_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("No model files found in results directory")
    model_path = model_files[-1]
    print(f"📂 Loading model: {model_path.name}")
    model, label_encoder = joblib.load(model_path)

    # Load TSV data (features)
    tsv_path = DATA_DIR / f"{split}.tsv"
    print(f"📂 Loading features: {tsv_path}")
    df_features = pd.read_csv(tsv_path, sep="\t")

    # Load original parquet (for text and GPT-3.5 predictions)
    parquet_files = sorted(CONFIDENCE_DIR.glob(f"CT24_{split}_*.parquet"))
    if parquet_files:
        parquet_path = parquet_files[-1]
        print(f"📂 Loading original data: {parquet_path.name}")
        df_original = pl.read_parquet(parquet_path).to_pandas()
    else:
        df_original = None
        print("⚠️  Original parquet not found - text analysis will be limited")

    return model, label_encoder, df_features, df_original


def get_predictions(model, df_features, threshold: float = 0.50):
    """Get model predictions and probabilities."""
    X = df_features.iloc[:, :-1].values
    y_true = df_features.iloc[:, -1].values

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_true, y_pred, y_proba


# =============================================================================
# ERROR ANALYSIS FUNCTIONS
# =============================================================================


def analyze_error_types(y_true, y_pred, y_proba, df_original=None):
    """Analyze False Positives and False Negatives."""
    print("\n" + "=" * 70)
    print("1. ERROR TYPE ANALYSIS")
    print("=" * 70)

    # Identify error types
    correct = y_true == y_pred
    fp_mask = (y_pred == 1) & (y_true == 0)  # Predicted Yes, Actual No
    fn_mask = (y_pred == 0) & (y_true == 1)  # Predicted No, Actual Yes
    tp_mask = (y_pred == 1) & (y_true == 0) == False
    tp_mask = (y_pred == 1) & (y_true == 1)  # True Positive
    tn_mask = (y_pred == 0) & (y_true == 0)  # True Negative

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tp = tp_mask.sum()
    n_tn = tn_mask.sum()
    n_errors = n_fp + n_fn

    print(f"\n📊 Error Breakdown:")
    print(f"   True Negatives (correct No):   {n_tn:4d}")
    print(f"   True Positives (correct Yes):  {n_tp:4d}")
    print(f"   False Positives (wrong Yes):   {n_fp:4d}  ← Predicted checkworthy but wasn't")
    print(f"   False Negatives (missed Yes):  {n_fn:4d}  ← Missed checkworthy claims")
    print(f"   Total Errors:                  {n_errors:4d}")

    # Confidence statistics for each group
    print(f"\n📈 Probability Distribution by Group:")
    print(f"   {'Group':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"   {'-'*60}")

    for name, mask in [("True Negatives", tn_mask), ("True Positives", tp_mask),
                       ("False Positives", fp_mask), ("False Negatives", fn_mask)]:
        if mask.sum() > 0:
            probs = y_proba[mask]
            print(f"   {name:<25} {probs.mean():>8.3f} {probs.std():>8.3f} {probs.min():>8.3f} {probs.max():>8.3f}")

    return {
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
        "tp_mask": tp_mask,
        "tn_mask": tn_mask,
        "n_fp": n_fp,
        "n_fn": n_fn,
    }


def analyze_confidence_features(df_features, error_masks):
    """Analyze confidence feature patterns in errors."""
    print("\n" + "=" * 70)
    print("2. CONFIDENCE FEATURE ANALYSIS")
    print("=" * 70)

    fp_mask = error_masks["fp_mask"]
    fn_mask = error_masks["fn_mask"]

    # Feature means for each group
    print(f"\n📊 Mean Feature Values by Group:")
    print(f"   {'Feature':<25} {'Correct':>10} {'FP':>10} {'FN':>10}")
    print(f"   {'-'*55}")

    correct_mask = ~(fp_mask | fn_mask)
    results = {}

    for col in FEATURE_COLS:
        correct_mean = df_features.loc[correct_mask, col].mean()
        fp_mean = df_features.loc[fp_mask, col].mean() if fp_mask.sum() > 0 else np.nan
        fn_mean = df_features.loc[fn_mask, col].mean() if fn_mask.sum() > 0 else np.nan

        print(f"   {col:<25} {correct_mean:>10.2f} {fp_mean:>10.2f} {fn_mean:>10.2f}")

        results[col] = {"correct": correct_mean, "fp": fp_mean, "fn": fn_mean}

    # Disagreement analysis (self-reported vs logprob)
    print(f"\n📊 Self-Reported vs Logprob Disagreement:")

    for prefix in ["checkability", "verifiability", "harmpot"]:
        cc_col = f"{prefix}_cc"
        lp_col = f"{prefix}_logprob"

        if cc_col in df_features.columns and lp_col in df_features.columns:
            disagreement = (df_features[cc_col] - df_features[lp_col]).abs()

            correct_disagree = disagreement[correct_mask].mean()
            fp_disagree = disagreement[fp_mask].mean() if fp_mask.sum() > 0 else np.nan
            fn_disagree = disagreement[fn_mask].mean() if fn_mask.sum() > 0 else np.nan

            print(f"   {prefix:<15} Correct: {correct_disagree:>6.2f}  FP: {fp_disagree:>6.2f}  FN: {fn_disagree:>6.2f}")

    return results


def analyze_text_patterns(df_original, error_masks):
    """Analyze text patterns in errors."""
    print("\n" + "=" * 70)
    print("3. TEXT PATTERN ANALYSIS")
    print("=" * 70)

    if df_original is None:
        print("   ⚠️  Original data not available - skipping text analysis")
        return None

    fp_mask = error_masks["fp_mask"]
    fn_mask = error_masks["fn_mask"]
    correct_mask = ~(fp_mask | fn_mask)

    texts = df_original["text"].values

    # Text length analysis
    lengths = np.array([len(t) for t in texts])

    print(f"\n📏 Text Length Analysis:")
    print(f"   {'Group':<20} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"   {'-'*55}")

    for name, mask in [("Correct", correct_mask), ("False Positives", fp_mask), ("False Negatives", fn_mask)]:
        if mask.sum() > 0:
            lens = lengths[mask]
            print(f"   {name:<20} {lens.mean():>8.1f} {np.median(lens):>8.1f} {lens.min():>8d} {lens.max():>8d}")

    # Word patterns
    print(f"\n📝 Common Words in Errors (excluding stopwords):")

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                 "for", "on", "with", "at", "by", "from", "as", "into", "through",
                 "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
                 "not", "no", "yes", "it", "its", "this", "that", "these", "those",
                 "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
                 "my", "your", "his", "our", "their", "what", "which", "who", "whom"}

    def get_word_freq(mask, top_n=10):
        error_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        words = []
        for text in error_texts:
            words.extend([w.lower().strip(".,!?\"'()[]{}") for w in text.split()])
        words = [w for w in words if w and len(w) > 2 and w not in stopwords]
        return Counter(words).most_common(top_n)

    if fp_mask.sum() > 0:
        print(f"\n   False Positives - Top words:")
        for word, count in get_word_freq(fp_mask):
            print(f"      '{word}': {count}")

    if fn_mask.sum() > 0:
        print(f"\n   False Negatives - Top words:")
        for word, count in get_word_freq(fn_mask):
            print(f"      '{word}': {count}")

    return {"lengths": lengths}


def analyze_gpt35_comparison(df_original, y_pred, error_masks):
    """Compare CatBoost errors with GPT-3.5 predictions."""
    print("\n" + "=" * 70)
    print("4. COMPARISON WITH GPT-3.5 PREDICTIONS")
    print("=" * 70)

    if df_original is None or "model_prediction" not in df_original.columns:
        print("   ⚠️  GPT-3.5 predictions not available - skipping comparison")
        return None

    fp_mask = error_masks["fp_mask"]
    fn_mask = error_masks["fn_mask"]

    # Convert GPT-3.5 predictions to binary
    gpt_pred = (df_original["model_prediction"] == "Yes").astype(int).values
    y_true = (df_original["label"] == "Yes").astype(int).values

    # GPT-3.5 errors
    gpt_fp = (gpt_pred == 1) & (y_true == 0)
    gpt_fn = (gpt_pred == 0) & (y_true == 1)

    print(f"\n📊 Error Comparison:")
    print(f"   {'Metric':<35} {'CatBoost':>10} {'GPT-3.5':>10}")
    print(f"   {'-'*55}")
    print(f"   {'False Positives':<35} {fp_mask.sum():>10} {gpt_fp.sum():>10}")
    print(f"   {'False Negatives':<35} {fn_mask.sum():>10} {gpt_fn.sum():>10}")
    print(f"   {'Total Errors':<35} {(fp_mask | fn_mask).sum():>10} {(gpt_fp | gpt_fn).sum():>10}")

    # Overlap analysis
    shared_fp = (fp_mask & gpt_fp).sum()
    shared_fn = (fn_mask & gpt_fn).sum()
    shared_errors = ((fp_mask | fn_mask) & (gpt_fp | gpt_fn)).sum()

    catboost_only_errors = ((fp_mask | fn_mask) & ~(gpt_fp | gpt_fn)).sum()
    gpt_only_errors = (~(fp_mask | fn_mask) & (gpt_fp | gpt_fn)).sum()

    print(f"\n📊 Error Overlap:")
    print(f"   Shared False Positives:     {shared_fp:>4}")
    print(f"   Shared False Negatives:     {shared_fn:>4}")
    print(f"   Shared Errors (any type):   {shared_errors:>4}")
    print(f"   CatBoost-only errors:       {catboost_only_errors:>4}")
    print(f"   GPT-3.5-only errors:        {gpt_only_errors:>4}")

    # Agreement rate
    agreement = (y_pred == gpt_pred).mean()
    print(f"\n   Overall agreement rate:     {agreement:.1%}")

    return {
        "gpt_fp": gpt_fp,
        "gpt_fn": gpt_fn,
        "shared_fp": shared_fp,
        "shared_fn": shared_fn,
    }


def export_error_samples(df_original, df_features, y_true, y_pred, y_proba, error_masks, output_dir):
    """Export error samples for manual review."""
    print("\n" + "=" * 70)
    print("5. EXPORTING ERROR SAMPLES")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    fp_mask = error_masks["fp_mask"]
    fn_mask = error_masks["fn_mask"]

    if df_original is not None:
        texts = df_original["text"].values
        sample_ids = df_original["sample_id"].values
        gpt_pred = df_original.get("model_prediction", pd.Series(["N/A"] * len(df_original))).values
    else:
        texts = ["N/A"] * len(y_true)
        sample_ids = list(range(len(y_true)))
        gpt_pred = ["N/A"] * len(y_true)

    # Create error dataframe
    error_data = []
    for i in range(len(y_true)):
        if fp_mask[i] or fn_mask[i]:
            error_data.append({
                "sample_id": sample_ids[i],
                "text": texts[i],
                "true_label": "Yes" if y_true[i] == 1 else "No",
                "predicted_label": "Yes" if y_pred[i] == 1 else "No",
                "error_type": "False Positive" if fp_mask[i] else "False Negative",
                "probability": y_proba[i],
                "gpt35_prediction": gpt_pred[i],
                "checkability_cc": df_features.iloc[i]["checkability_cc"],
                "checkability_logprob": df_features.iloc[i]["checkability_logprob"],
                "verifiability_cc": df_features.iloc[i]["verifiability_cc"],
                "verifiability_logprob": df_features.iloc[i]["verifiability_logprob"],
                "harmpot_cc": df_features.iloc[i]["harmpot_cc"],
                "harmpot_logprob": df_features.iloc[i]["harmpot_logprob"],
            })

    df_errors = pd.DataFrame(error_data)

    # Sort by probability (most confident errors first)
    df_errors = df_errors.sort_values("probability", ascending=False)

    # Export
    errors_path = output_dir / "error_samples.tsv"
    df_errors.to_csv(errors_path, sep="\t", index=False)
    print(f"   💾 Saved: {errors_path}")
    print(f"      Total error samples: {len(df_errors)}")

    # Also export separate FP and FN files
    fp_path = output_dir / "false_positives.tsv"
    fn_path = output_dir / "false_negatives.tsv"

    df_errors[df_errors["error_type"] == "False Positive"].to_csv(fp_path, sep="\t", index=False)
    df_errors[df_errors["error_type"] == "False Negative"].to_csv(fn_path, sep="\t", index=False)

    print(f"   💾 Saved: {fp_path} ({(df_errors['error_type'] == 'False Positive').sum()} samples)")
    print(f"   💾 Saved: {fn_path} ({(df_errors['error_type'] == 'False Negative').sum()} samples)")

    return df_errors


def create_visualizations(df_features, y_true, y_pred, y_proba, error_masks, output_dir):
    """Create visualization plots."""
    print("\n" + "=" * 70)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    fp_mask = error_masks["fp_mask"]
    fn_mask = error_masks["fn_mask"]
    correct_mask = ~(fp_mask | fn_mask)

    # Create classification labels
    classification = np.array(["Correct"] * len(y_true))
    classification[fp_mask] = "False Positive"
    classification[fn_mask] = "False Negative"

    # 1. Probability distribution by classification
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [("Correct", "green"), ("False Positive", "red"), ("False Negative", "orange")]:
        mask = classification == label
        if mask.sum() > 0:
            ax.hist(y_proba[mask], bins=20, alpha=0.5, label=f"{label} (n={mask.sum()})", color=color)
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Probability Distribution by Classification Result", fontsize=14)
    ax.legend()
    ax.axvline(0.5, color="gray", linestyle="--", label="Threshold=0.5")

    prob_dist_path = output_dir / "probability_distribution.png"
    plt.savefig(prob_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {prob_dist_path}")

    # 2. Feature comparison boxplot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        data = []
        labels = []
        for label in ["Correct", "False Positive", "False Negative"]:
            mask = classification == label
            if mask.sum() > 0:
                data.append(df_features.loc[mask, col].values)
                labels.append(f"{label}\n(n={mask.sum()})")

        ax.boxplot(data, labels=labels)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Value")

    plt.suptitle("Feature Distributions by Classification Result", fontsize=14, y=1.02)
    plt.tight_layout()

    features_path = output_dir / "feature_distributions.png"
    plt.savefig(features_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {features_path}")

    # 3. Self-reported vs Logprob scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    prefixes = [("checkability", "Checkability"), ("verifiability", "Verifiability"), ("harmpot", "Harm Potential")]

    for ax, (prefix, title) in zip(axes, prefixes):
        cc_col = f"{prefix}_cc"
        lp_col = f"{prefix}_logprob"

        colors = {"Correct": "green", "False Positive": "red", "False Negative": "orange"}
        for label, color in colors.items():
            mask = classification == label
            if mask.sum() > 0:
                ax.scatter(df_features.loc[mask, cc_col], df_features.loc[mask, lp_col],
                          alpha=0.5, label=label, color=color, s=30)

        ax.set_xlabel("Self-Reported Confidence", fontsize=11)
        ax.set_ylabel("Logprob Confidence", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Agreement line")
        ax.legend(fontsize=9)

    plt.suptitle("Self-Reported vs Logprob Confidence by Classification", fontsize=14)
    plt.tight_layout()

    scatter_path = output_dir / "confidence_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {scatter_path}")

    # 4. Confusion matrix style error heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap data for feature means
    feature_means = []
    for col in FEATURE_COLS:
        row = [
            df_features.loc[correct_mask, col].mean(),
            df_features.loc[fp_mask, col].mean() if fp_mask.sum() > 0 else 0,
            df_features.loc[fn_mask, col].mean() if fn_mask.sum() > 0 else 0,
        ]
        feature_means.append(row)

    heatmap_data = np.array(feature_means)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=["Correct", "False Positive", "False Negative"],
                yticklabels=FEATURE_COLS, ax=ax)
    ax.set_title("Mean Feature Values by Classification Result", fontsize=14)

    heatmap_path = output_dir / "feature_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {heatmap_path}")


# =============================================================================
# MAIN
# =============================================================================


def main(split: str = "test", threshold: float = 0.50):
    """Run full error analysis."""
    print("=" * 70)
    print("CT24 CLASSIFIER ERROR ANALYSIS")
    print("=" * 70)
    print(f"Split: {split}")
    print(f"Threshold: {threshold}")

    # Load data
    model, label_encoder, df_features, df_original = load_model_and_data(split)

    # Get predictions
    y_true, y_pred, y_proba = get_predictions(model, df_features, threshold)

    # Run analyses
    error_masks = analyze_error_types(y_true, y_pred, y_proba, df_original)
    analyze_confidence_features(df_features, error_masks)
    analyze_text_patterns(df_original, error_masks)
    gpt_comparison = analyze_gpt35_comparison(df_original, y_pred, error_masks)

    # Export and visualize
    output_dir = OUTPUT_DIR / f"{split}_threshold_{threshold:.2f}"
    df_errors = export_error_samples(df_original, df_features, y_true, y_pred, y_proba, error_masks, output_dir)
    create_visualizations(df_features, y_true, y_pred, y_proba, error_masks, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print(f"\n📊 Key Findings:")
    print(f"   • Total errors: {error_masks['n_fp'] + error_masks['n_fn']}")
    print(f"   • False Positives: {error_masks['n_fp']} (wrongly flagged as checkworthy)")
    print(f"   • False Negatives: {error_masks['n_fn']} (missed checkworthy claims)")

    if gpt_comparison:
        print(f"   • Shared errors with GPT-3.5: {gpt_comparison['shared_fp'] + gpt_comparison['shared_fn']}")

    print(f"\n📝 Review error samples in:")
    print(f"   • {output_dir / 'error_samples.tsv'}")
    print(f"   • {output_dir / 'false_positives.tsv'}")
    print(f"   • {output_dir / 'false_negatives.tsv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze classifier errors")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"],
                       help="Data split to analyze (default: test)")
    parser.add_argument("--threshold", type=float, default=0.50,
                       help="Classification threshold (default: 0.50)")

    args = parser.parse_args()
    main(split=args.split, threshold=args.threshold)
