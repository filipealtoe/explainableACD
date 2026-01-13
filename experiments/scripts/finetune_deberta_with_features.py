#!/usr/bin/env python3
"""
DeBERTa + Text Features Fusion

Concatenates DeBERTa's [CLS] embedding with 35 engineered text features.
This combines semantic understanding with high-lift pattern features.

High-lift features from analysis:
- has_percentage: 3.75x lift
- voted_for_against: 3.37x
- has_dollar: 3.34x
- has_million_billion: 3.24x
- is_question: 0.18x (negative indicator)

Architecture:
    Text → DeBERTa → [CLS] (1024-dim)
                         ↓
    Text → Feature Extraction → (35-dim, normalized)
                         ↓
              Concatenate → (1059-dim)
                         ↓
                   MLP Head → Binary Classification

Usage:
    # Full fine-tuning
    python experiments/scripts/finetune_deberta_with_features.py

    # Freeze DeBERTa, only train head (faster)
    python experiments/scripts/finetune_deberta_with_features.py --freeze-encoder

    # Custom epochs
    python experiments/scripts/finetune_deberta_with_features.py --epochs 5 --lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Model, get_linear_schedule_with_warmup

# Paths
BASELINE_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_with_features"

# Top features by predictive lift (from prior analysis)
TOP_FEATURES = [
    # Positive indicators (checkworthy)
    "has_percentage",
    "has_dollar",
    "has_number",
    "has_precise_number",
    "has_large_scale",
    "has_source_attribution",
    "has_specific_year",
    "has_comparative",
    "has_voted",
    "has_increase_decrease",
    # Negative indicators (not checkworthy)
    "is_question",
    "has_first_person_stance",
    "has_future_modal",
    "has_hedge",
    "has_vague_quantifier",
    # Interaction features
    "has_number_and_time",
    "has_number_and_comparative",
    "has_source_and_number",
    # Metadata
    "word_count",
    "avg_word_length",
]


class DeBERTaWithFeatures(nn.Module):
    """DeBERTa encoder with text features concatenated before classification head."""

    def __init__(self, base_model_path: Path, num_features: int, freeze_encoder: bool = False):
        super().__init__()

        # Load pre-trained DeBERTa encoder
        self.deberta = DebertaV2Model.from_pretrained(base_model_path)
        self.hidden_size = self.deberta.config.hidden_size  # 1024 for large

        if freeze_encoder:
            for param in self.deberta.parameters():
                param.requires_grad = False
            print("🔒 DeBERTa encoder frozen")

        # Feature processing
        self.feature_norm = nn.LayerNorm(num_features)

        # Classification head: [CLS] + features → 2 classes
        combined_size = self.hidden_size + num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(combined_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, features):
        # Get [CLS] embedding from DeBERTa
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

        # Normalize features
        features_normed = self.feature_norm(features)  # [batch, num_features]

        # Concatenate and classify
        combined = torch.cat([cls_embedding, features_normed], dim=-1)
        logits = self.classifier(combined)

        return logits


class FusionDataset(Dataset):
    """Dataset that returns text tokens + pre-extracted features."""

    def __init__(self, texts, labels, features, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.features = features  # numpy array [N, num_features]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data_with_features(split: str, tokenizer, scaler=None, feature_cols=None):
    """Load cleaned text data and merge with pre-extracted features."""
    # Load cleaned data (preferred) or fall back to raw
    clean_path = CLEAN_DATA_DIR / f"CT24_{split}_clean.parquet"
    if clean_path.exists():
        clean_df = pl.read_parquet(clean_path)
    else:
        # Try TSV
        clean_tsv = CLEAN_DATA_DIR / f"CT24_{split}_clean.tsv"
        if clean_tsv.exists():
            clean_df = pl.read_csv(clean_tsv, separator="\t")
        else:
            raise FileNotFoundError(f"Cleaned data not found: {clean_path}")

    # Load features
    features_path = FEATURES_DIR / f"CT24_{split}_features.parquet"
    if features_path.exists():
        feat_df = pl.read_parquet(features_path)
    else:
        raise FileNotFoundError(f"Features not found: {features_path}. Run extract_text_features.py first.")

    # Merge on Sentence_id
    merged = clean_df.join(feat_df, on="Sentence_id", how="left")

    texts = merged["Text"].to_list()
    labels = [1 if l == "Yes" else 0 for l in merged["class_label"].to_list()]

    # Extract feature columns
    if feature_cols is None:
        # Use top features that exist in the dataframe
        feature_cols = [f for f in TOP_FEATURES if f in merged.columns]

    features = merged.select(feature_cols).to_numpy().astype(np.float32)

    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    dataset = FusionDataset(texts, labels, features, tokenizer)

    return dataset, scaler, feature_cols


def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, features)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)

            logits = model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)
    all_labels = np.array(all_labels)

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    return f1, acc, all_probs, all_preds, all_labels


def find_optimal_threshold(probs, labels):
    """Find threshold that maximizes F1 on given data."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.30, 0.70, 0.01):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze DeBERTa, only train head")
    parser.add_argument("--class-weight", type=float, default=3.0, help="Weight for positive class")
    args = parser.parse_args()

    print("=" * 70)
    print("DEBERTA + TEXT FEATURES FUSION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\n📦 Loading tokenizer from: {BASELINE_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(BASELINE_MODEL)

    # Load data with features
    print("\n📊 Loading data with features...")
    train_dataset, scaler, feature_cols = load_data_with_features("train", tokenizer)
    dev_dataset, _, _ = load_data_with_features("dev", tokenizer, scaler, feature_cols)

    print(f"   Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")
    print(f"   Features ({len(feature_cols)}): {feature_cols[:5]}...")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # Create model
    print(f"\n🤖 Creating DeBERTa + Features model...")
    model = DeBERTaWithFeatures(
        BASELINE_MODEL,
        num_features=len(feature_cols),
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Class weights for imbalanced data
    class_weights = torch.tensor([1.0, args.class_weight], dtype=torch.float)
    print(f"   Class weights: [1.0, {args.class_weight}]")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline evaluation
    f1, acc, _, _, _ = evaluate(model, dev_loader, device)
    print(f"\n📈 Baseline (before training): F1={f1:.4f}, Acc={acc:.4f}")
    best_f1 = f1

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training for {args.epochs} epochs, lr={args.lr}")
    print("=" * 70)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        f1, acc, probs, _, labels = evaluate(model, dev_loader, device)

        # Find optimal threshold
        opt_thresh, opt_f1 = find_optimal_threshold(probs, labels)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | Dev F1: {f1:.4f} (thresh=0.5), {opt_f1:.4f} (thresh={opt_thresh:.2f})")

        if opt_f1 > best_f1:
            best_f1 = opt_f1
            best_thresh = opt_thresh
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "threshold": best_thresh,
                "f1": best_f1,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"   → New best! Saved.")

    # Final evaluation on dev-test and test
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    best_thresh = checkpoint["threshold"]

    for split in ["dev", "dev-test", "test"]:
        try:
            eval_dataset, _, _ = load_data_with_features(split, tokenizer, scaler, feature_cols)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
            f1, acc, probs, preds, labels = evaluate(model, eval_loader, device, threshold=best_thresh)
            print(f"\n{split.upper()} (threshold={best_thresh:.2f}):")
            print(f"   F1: {f1:.4f}, Accuracy: {acc:.4f}")
            print(classification_report(labels, preds, target_names=["No", "Yes"]))
        except FileNotFoundError:
            print(f"\n{split.upper()}: Skipped (no features file)")

    print(f"\n💾 Model saved to: {OUTPUT_DIR / 'best_model.pt'}")
    print(f"🎯 Best F1: {best_f1:.4f} at threshold {best_thresh:.2f}")


if __name__ == "__main__":
    main()
