#!/usr/bin/env python3
"""
Ensemble Experiment: DeBERTa + Gradient Boosting on Text Features

Combines:
1. DeBERTa-v3-large (semantic understanding, F1~0.82)
2. Gradient Boosting on 35 text features (pattern matching, F1~0.67)

Ensemble methods:
- Weighted averaging of probabilities
- Stacking with logistic regression meta-learner
- Voting (majority or soft)

Usage:
    python experiments/scripts/run_ensemble_experiment.py
    python experiments/scripts/run_ensemble_experiment.py --deberta-weight 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# Paths
DEBERTA_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ensemble"

# Features to use for GB model
FEATURE_COLS = [
    # High positive lift
    "has_percentage", "has_dollar", "has_number", "has_precise_number",
    "has_large_scale", "has_source_attribution", "has_specific_year",
    "has_comparative", "has_voted", "has_increase_decrease",
    # High negative lift
    "is_question", "has_first_person_stance", "has_future_modal",
    "has_hedge", "has_vague_quantifier",
    # Interactions
    "has_number_and_time", "has_number_and_comparative", "has_source_and_number",
    # Metadata
    "word_count", "avg_word_length",
]


def load_data(split: str):
    """Load cleaned data and features for a split."""
    # Load cleaned data
    clean_path = CLEAN_DATA_DIR / f"CT24_{split}_clean.parquet"
    if clean_path.exists():
        clean_df = pl.read_parquet(clean_path)
    else:
        clean_tsv = CLEAN_DATA_DIR / f"CT24_{split}_clean.tsv"
        if clean_tsv.exists():
            clean_df = pl.read_csv(clean_tsv, separator="\t")
        else:
            raise FileNotFoundError(f"Cleaned data not found: {clean_path}")

    # Load features
    features_path = FEATURES_DIR / f"CT24_{split}_features.parquet"
    if features_path.exists():
        feat_df = pl.read_parquet(features_path)
        merged = clean_df.join(feat_df, on="Sentence_id", how="left")
    else:
        merged = clean_df

    texts = merged["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in merged["class_label"].to_list()])

    # Extract features
    available_cols = [c for c in FEATURE_COLS if c in merged.columns]
    if available_cols:
        features = merged.select(available_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
    else:
        features = None

    return texts, labels, features, available_cols


def get_deberta_probs(model, tokenizer, texts, device, batch_size=16):
    """Get DeBERTa probabilities for all texts."""
    model.eval()
    probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa inference"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)

    return np.array(probs)


def train_gb_model(X_train, y_train, class_weight=3.0):
    """Train CatBoost classifier on text features."""
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        class_weights={0: 1.0, 1: class_weight},
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    return model


def find_optimal_weights(deberta_probs, gb_probs, labels):
    """Find optimal ensemble weights via grid search."""
    best_f1, best_weight = 0, 0.5

    for w in np.arange(0.5, 1.0, 0.05):
        ensemble_probs = w * deberta_probs + (1 - w) * gb_probs
        for thresh in np.arange(0.35, 0.65, 0.02):
            preds = (ensemble_probs >= thresh).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_weight = w
                best_thresh = thresh

    return best_weight, best_thresh, best_f1


def train_stacking_meta(deberta_probs, gb_probs, labels):
    """Train logistic regression meta-learner for stacking."""
    X_meta = np.column_stack([deberta_probs, gb_probs])
    meta_model = LogisticRegression(class_weight={0: 1, 1: 3}, random_state=42)
    meta_model.fit(X_meta, labels)
    return meta_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deberta-weight", type=float, default=None, help="Weight for DeBERTa (auto-tune if not set)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 70)
    print("ENSEMBLE EXPERIMENT: DeBERTa + Gradient Boosting")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DeBERTa
    print(f"\n🤖 Loading DeBERTa from: {DEBERTA_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_MODEL)
    deberta = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL).to(device)

    # Load data
    print("\n📊 Loading data...")
    train_texts, train_labels, train_features, feat_cols = load_data("train")
    dev_texts, dev_labels, dev_features, _ = load_data("dev")

    print(f"   Train: {len(train_texts)}, Dev: {len(dev_texts)}")
    print(f"   Features: {len(feat_cols)}")

    # Get DeBERTa probabilities
    print("\n🔮 Getting DeBERTa probabilities...")
    train_deberta_probs = get_deberta_probs(deberta, tokenizer, train_texts, device, args.batch_size)
    dev_deberta_probs = get_deberta_probs(deberta, tokenizer, dev_texts, device, args.batch_size)

    # DeBERTa standalone performance
    dev_deberta_preds = (dev_deberta_probs >= 0.5).astype(int)
    deberta_f1 = f1_score(dev_labels, dev_deberta_preds)
    print(f"   DeBERTa alone: F1={deberta_f1:.4f}")

    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    dev_features_scaled = scaler.transform(dev_features)

    # Train GB model
    print("\n🌲 Training Gradient Boosting on text features...")
    gb_model = train_gb_model(train_features_scaled, train_labels)
    train_gb_probs = gb_model.predict_proba(train_features_scaled)[:, 1]
    dev_gb_probs = gb_model.predict_proba(dev_features_scaled)[:, 1]

    # GB standalone performance
    dev_gb_preds = (dev_gb_probs >= 0.5).astype(int)
    gb_f1 = f1_score(dev_labels, dev_gb_preds)
    print(f"   GB alone: F1={gb_f1:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # ENSEMBLE METHOD 1: Weighted Averaging
    # ==========================================================================
    print(f"\n{'='*70}")
    print("ENSEMBLE METHOD 1: Weighted Averaging")
    print("=" * 70)

    if args.deberta_weight is None:
        # Auto-tune weights on dev set
        best_weight, best_thresh, best_f1 = find_optimal_weights(dev_deberta_probs, dev_gb_probs, dev_labels)
        print(f"   Optimal weight: DeBERTa={best_weight:.2f}, GB={1-best_weight:.2f}")
        print(f"   Optimal threshold: {best_thresh:.2f}")
        print(f"   Best F1: {best_f1:.4f}")
    else:
        best_weight = args.deberta_weight
        best_thresh = args.threshold

    # Evaluate weighted ensemble
    ensemble_probs = best_weight * dev_deberta_probs + (1 - best_weight) * dev_gb_probs
    ensemble_preds = (ensemble_probs >= best_thresh).astype(int)
    ensemble_f1 = f1_score(dev_labels, ensemble_preds)
    ensemble_acc = accuracy_score(dev_labels, ensemble_preds)

    print(f"\n   Weighted Ensemble: F1={ensemble_f1:.4f}, Acc={ensemble_acc:.4f}")
    print(f"   Improvement over DeBERTa: {ensemble_f1 - deberta_f1:+.4f}")

    # ==========================================================================
    # ENSEMBLE METHOD 2: Stacking
    # ==========================================================================
    print(f"\n{'='*70}")
    print("ENSEMBLE METHOD 2: Stacking (Logistic Regression Meta-Learner)")
    print("=" * 70)

    # Train meta-learner on train set
    meta_model = train_stacking_meta(train_deberta_probs, train_gb_probs, train_labels)

    # Evaluate on dev
    dev_meta_input = np.column_stack([dev_deberta_probs, dev_gb_probs])
    stacking_probs = meta_model.predict_proba(dev_meta_input)[:, 1]
    stacking_preds = (stacking_probs >= 0.5).astype(int)
    stacking_f1 = f1_score(dev_labels, stacking_preds)
    stacking_acc = accuracy_score(dev_labels, stacking_preds)

    print(f"   Stacking Ensemble: F1={stacking_f1:.4f}, Acc={stacking_acc:.4f}")
    print(f"   Improvement over DeBERTa: {stacking_f1 - deberta_f1:+.4f}")

    # ==========================================================================
    # FINAL COMPARISON
    # ==========================================================================
    print(f"\n{'='*70}")
    print("COMPARISON ON DEV SET")
    print("=" * 70)

    results = [
        ("DeBERTa alone", deberta_f1),
        ("GB alone", gb_f1),
        ("Weighted Ensemble", ensemble_f1),
        ("Stacking Ensemble", stacking_f1),
    ]

    for name, f1 in sorted(results, key=lambda x: -x[1]):
        delta = f1 - deberta_f1
        marker = "👑" if f1 == max(r[1] for r in results) else "  "
        print(f"   {marker} {name}: F1={f1:.4f} ({delta:+.4f})")

    # ==========================================================================
    # TEST SET EVALUATION
    # ==========================================================================
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print("=" * 70)

    for split in ["dev-test", "test"]:
        try:
            test_texts, test_labels, test_features, _ = load_data(split)
            test_features_scaled = scaler.transform(test_features)

            # Get predictions
            test_deberta_probs = get_deberta_probs(deberta, tokenizer, test_texts, device, args.batch_size)
            test_gb_probs = gb_model.predict_proba(test_features_scaled)[:, 1]

            # Weighted ensemble
            test_ensemble_probs = best_weight * test_deberta_probs + (1 - best_weight) * test_gb_probs
            test_ensemble_preds = (test_ensemble_probs >= best_thresh).astype(int)

            # DeBERTa alone
            test_deberta_preds = (test_deberta_probs >= 0.5).astype(int)

            deberta_test_f1 = f1_score(test_labels, test_deberta_preds)
            ensemble_test_f1 = f1_score(test_labels, test_ensemble_preds)

            print(f"\n{split.upper()}:")
            print(f"   DeBERTa: F1={deberta_test_f1:.4f}")
            print(f"   Ensemble: F1={ensemble_test_f1:.4f} ({ensemble_test_f1 - deberta_test_f1:+.4f})")

        except FileNotFoundError:
            print(f"\n{split.upper()}: Skipped (features not found)")

    # Save ensemble config
    import json
    config = {
        "deberta_weight": best_weight,
        "threshold": best_thresh,
        "dev_f1": {
            "deberta": float(deberta_f1),
            "gb": float(gb_f1),
            "weighted_ensemble": float(ensemble_f1),
            "stacking": float(stacking_f1),
        },
        "feature_cols": feat_cols,
    }
    with open(OUTPUT_DIR / "ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Config saved to: {OUTPUT_DIR / 'ensemble_config.json'}")


if __name__ == "__main__":
    main()
