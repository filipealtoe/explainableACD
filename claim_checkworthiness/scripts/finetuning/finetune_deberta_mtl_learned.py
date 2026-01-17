#!/usr/bin/env python3
"""
MTL with Learned Task Weights + Feature Selection.

Improvements over simple MTL:
1. Learned weights per auxiliary task (uncertainty weighting)
2. Configurable target features (p_yes, score, or multi-feature)
3. Automatic downweighting of noisy tasks

Usage:
    python experiments/scripts/finetune_deberta_mtl_learned.py
    python experiments/scripts/finetune_deberta_mtl_learned.py --targets score  # use LLM scores instead
    python experiments/scripts/finetune_deberta_mtl_learned.py --targets multi  # use multiple features per head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification

# =============================================================================
# Configuration
# =============================================================================

BASELINE_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"  # Use cleaned data
LLM_FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"  # For dev-test
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_mtl_learned"

# Feature options for auxiliary targets
FEATURE_SETS = {
    "p_yes": {
        "check": ["check_p_yes"],
        "verif": ["verif_p_yes"],
        "harm": ["harm_p_yes"],
    },
    "score": {
        "check": ["check_score"],  # 0-100, will normalize to 0-1
        "verif": ["verif_score"],
        "harm": ["harm_score"],
    },
    "multi": {
        "check": ["check_p_yes", "check_score", "check_margin_p"],
        "verif": ["verif_p_yes", "verif_score", "verif_margin_p"],
        "harm": ["harm_p_yes", "harm_score", "harm_margin_p", "harm_social_fragmentation",
                 "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
    },
}


# =============================================================================
# Model with Learned Task Weights
# =============================================================================

class DeBERTaMTLLearned(nn.Module):
    """MTL with learned uncertainty weights per task."""

    def __init__(self, base_model, target_dims=None):
        super().__init__()
        self.deberta = base_model.deberta
        self.pooler = base_model.pooler  # Keep pooler! Main head was trained with it
        self.dropout = nn.Dropout(0.1)

        # Get hidden size from config (not hardcoded)
        hidden_size = base_model.config.hidden_size

        # Target dimensions (how many features per aux head)
        target_dims = target_dims or {"check": 1, "verif": 1, "harm": 1}

        # Main head (binary classification)
        self.main_head = nn.Linear(hidden_size, 2)

        # Auxiliary heads (can predict multiple features)
        self.check_head = nn.Linear(hidden_size, target_dims["check"])
        self.verif_head = nn.Linear(hidden_size, target_dims["verif"])
        self.harm_head = nn.Linear(hidden_size, target_dims["harm"])

        # Learned log-variance for uncertainty weighting (Kendall et al., 2018)
        # log_var = log(sigma^2), so exp(-log_var) = 1/sigma^2
        # Initialize: main=1.0 weight, aux=0.37 weight (conservative start)
        self.log_var_main = nn.Parameter(torch.tensor(0.0))   # exp(0) = 1.0
        self.log_var_check = nn.Parameter(torch.tensor(1.0))  # exp(-1) ≈ 0.37
        self.log_var_verif = nn.Parameter(torch.tensor(1.0))
        self.log_var_harm = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Pass full hidden states to pooler (it extracts [CLS] internally)
        pooled = self.pooler(outputs.last_hidden_state)
        pooled = self.dropout(pooled)

        main_logits = self.main_head(pooled)
        check_pred = torch.sigmoid(self.check_head(pooled))
        verif_pred = torch.sigmoid(self.verif_head(pooled))
        harm_pred = torch.sigmoid(self.harm_head(pooled))

        return main_logits, check_pred, verif_pred, harm_pred

    def compute_weighted_loss(self, main_loss, check_loss, verif_loss, harm_loss):
        """
        Uncertainty weighting (Kendall et al., 2018):
        - For regression (MSE): L = 1/(2*sigma^2) * MSE + log(sigma)
        - For classification (CE): L = 1/sigma^2 * CE + log(sigma)

        We use log_var = log(sigma^2), so:
        - precision = exp(-log_var) = 1/sigma^2
        - regularizer = 0.5 * log_var (= log(sigma))
        """
        precision_main = torch.exp(-self.log_var_main)
        precision_check = torch.exp(-self.log_var_check)
        precision_verif = torch.exp(-self.log_var_verif)
        precision_harm = torch.exp(-self.log_var_harm)

        # CE loss: precision * loss + 0.5 * log_var
        # MSE loss: 0.5 * precision * loss + 0.5 * log_var (Kendall et al.)
        loss = (
            precision_main * main_loss + 0.5 * self.log_var_main +
            0.5 * precision_check * check_loss + 0.5 * self.log_var_check +
            0.5 * precision_verif * verif_loss + 0.5 * self.log_var_verif +
            0.5 * precision_harm * harm_loss + 0.5 * self.log_var_harm
        )
        return loss

    def get_task_weights(self):
        """Return current effective weights for logging."""
        with torch.no_grad():
            return {
                "main": torch.exp(-self.log_var_main).item(),
                "check": torch.exp(-self.log_var_check).item(),
                "verif": torch.exp(-self.log_var_verif).item(),
                "harm": torch.exp(-self.log_var_harm).item(),
            }


# =============================================================================
# Dataset
# =============================================================================

class MTLDataset(Dataset):
    def __init__(self, texts, labels, sentence_ids, llm_df, tokenizer, feature_set, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        self.sentence_ids = sentence_ids
        self.feature_set = feature_set

        # Handle both 'sentence_id' and 'Sentence_id' column names, or empty DataFrame
        if len(llm_df) > 0:
            id_col = "sentence_id" if "sentence_id" in llm_df.columns else "Sentence_id"
            self.llm_data = {row[id_col]: row for row in llm_df.iter_rows(named=True)}
            matched = sum(1 for sid in sentence_ids if sid in self.llm_data)
            print(f"  LLM feature coverage: {matched}/{len(sentence_ids)} ({100*matched/len(sentence_ids):.1f}%)")
        else:
            self.llm_data = {}
            print(f"  LLM features: N/A (eval only)")

    def __len__(self):
        return len(self.texts)

    def _get_features(self, llm_row, task):
        """Extract and normalize features for a task."""
        features = []
        for feat_name in self.feature_set[task]:
            val = llm_row.get(feat_name, 0.5)

            # Handle None/NaN (both Python and numpy types)
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                val = 0.5
            else:
                # Normalize scores from 0-100 to 0-1
                if "score" in feat_name and "p_" not in feat_name:
                    val = val / 100.0
                # Normalize harm sub-components (0-100 scale)
                elif feat_name in ["harm_social_fragmentation", "harm_spurs_action",
                                   "harm_believability", "harm_exploitativeness"]:
                    val = val / 100.0
                # Clamp all values to [0, 1] for sigmoid targets
                val = max(0.0, min(1.0, float(val)))

            features.append(val)
        return features

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        llm = self.llm_data.get(self.sentence_ids[idx], {})

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "check_target": torch.tensor(self._get_features(llm, "check"), dtype=torch.float),
            "verif_target": torch.tensor(self._get_features(llm, "verif"), dtype=torch.float),
            "harm_target": torch.tensor(self._get_features(llm, "harm"), dtype=torch.float),
        }


def load_data(split, tokenizer, feature_set):
    # Load processed data (parquet) for train/dev/test, raw TSV for dev-test
    processed_path = DATA_DIR / f"CT24_{split}_features.parquet"
    raw_path = RAW_DATA_DIR / f"CT24_checkworthy_english_{split}.tsv"

    if processed_path.exists():
        df = pl.read_parquet(processed_path)
    elif raw_path.exists():
        df = pl.read_csv(raw_path, separator="\t")
    else:
        raise FileNotFoundError(f"No data found for split '{split}'")

    # Try to load LLM features (may not exist for all splits like dev-test)
    llm_path = LLM_FEATURES_DIR / f"{split}_llm_features.parquet"
    if llm_path.exists():
        llm_df = pl.read_parquet(llm_path)
    else:
        print(f"  Warning: No LLM features for {split}, using empty (aux targets=0.5)")
        llm_df = pl.DataFrame()

    # Convert sentence_ids to strings for matching (LLM features use strings)
    sentence_ids = [str(sid) for sid in df["Sentence_id"].to_list()]

    return MTLDataset(
        texts=df["Text"].to_list(),
        labels=[1 if l == "Yes" else 0 for l in df["class_label"].to_list()],
        sentence_ids=sentence_ids,
        llm_df=llm_df,
        tokenizer=tokenizer,
        feature_set=feature_set,
    )


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        check_t = batch["check_target"].to(device)
        verif_t = batch["verif_target"].to(device)
        harm_t = batch["harm_target"].to(device)

        optimizer.zero_grad()

        main_logits, check_p, verif_p, harm_p = model(input_ids, attention_mask)

        # Individual losses
        loss_main = ce_loss_fn(main_logits, labels)
        loss_check = mse_loss_fn(check_p, check_t)
        loss_verif = mse_loss_fn(verif_p, verif_t)
        loss_harm = mse_loss_fn(harm_p, harm_t)

        # Weighted combination (learned weights)
        loss = model.compute_weighted_loss(loss_main, loss_check, loss_verif, loss_harm)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            main_logits, _, _, _ = model(input_ids, attention_mask)
            probs = torch.softmax(main_logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    return f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds), np.array(all_probs)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--targets", type=str, default="p_yes", choices=["p_yes", "score", "multi"],
                        help="Which LLM features to use as auxiliary targets")
    args = parser.parse_args()

    print("=" * 70)
    print("MTL with LEARNED Task Weights")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target features: {args.targets}")

    feature_set = FEATURE_SETS[args.targets]
    target_dims = {k: len(v) for k, v in feature_set.items()}
    print(f"Feature dimensions: {target_dims}")

    # Load baseline
    print(f"\nLoading baseline from: {BASELINE_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(BASELINE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_MODEL)

    # Create model (hidden_size from config automatically)
    model = DeBERTaMTLLearned(base_model, target_dims=target_dims)
    model.main_head.weight.data = base_model.classifier.weight.data.clone()
    model.main_head.bias.data = base_model.classifier.bias.data.clone()
    model.to(device)

    # Data
    print("\nLoading data...")
    train_dataset = load_data("train", tokenizer, feature_set)
    dev_dataset = load_data("dev", tokenizer, feature_set)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                            num_workers=2, pin_memory=True)
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # Baseline eval
    f1, acc, _ = evaluate(model, dev_loader, device)
    print(f"\nBaseline: F1={f1:.4f}, Acc={acc:.4f}")
    best_f1 = f1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n{'='*70}")
    print(f"Training with learned weights...")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        f1, acc, probs = evaluate(model, dev_loader, device)
        weights = model.get_task_weights()

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {loss:.4f} | F1: {f1:.4f}, Acc: {acc:.4f}")
        print(f"  Learned weights: main={weights['main']:.3f}, check={weights['check']:.3f}, "
              f"verif={weights['verif']:.3f}, harm={weights['harm']:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            np.save(OUTPUT_DIR / "dev_probs.npy", probs)
            print(f"  → New best! Saved.")

    # Load best checkpoint for final evaluation
    if (OUTPUT_DIR / "best_model.pt").exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))

    print(f"\n{'='*70}")
    print("FINAL LEARNED WEIGHTS (from best checkpoint)")
    print("=" * 70)
    weights = model.get_task_weights()
    total = sum(weights.values())
    print(f"  main:  {weights['main']:.3f} ({100*weights['main']/total:.1f}%)")
    print(f"  check: {weights['check']:.3f} ({100*weights['check']/total:.1f}%)")
    print(f"  verif: {weights['verif']:.3f} ({100*weights['verif']/total:.1f}%)")
    print(f"  harm:  {weights['harm']:.3f} ({100*weights['harm']/total:.1f}%)")

    # Evaluate on dev-test (separate held-out set for intermediate benchmarking)
    print(f"\n{'='*70}")
    print("EVALUATION ON DEV-TEST (held-out intermediate set)")
    print("=" * 70)
    try:
        devtest_dataset = load_data("dev-test", tokenizer, feature_set)
        devtest_loader = DataLoader(devtest_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
        devtest_f1, devtest_acc, devtest_probs = evaluate(model, devtest_loader, device)
        print(f"Dev-test: F1={devtest_f1:.4f}, Acc={devtest_acc:.4f}")
        np.save(OUTPUT_DIR / "devtest_probs.npy", devtest_probs)
    except Exception as e:
        print(f"Could not evaluate on dev-test: {e}")

    # Evaluate on TEST (final held-out set)
    print(f"\n{'='*70}")
    print("EVALUATION ON TEST (final held-out set)")
    print("=" * 70)
    try:
        test_dataset = load_data("test", tokenizer, feature_set)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
        test_f1, test_acc, test_probs = evaluate(model, test_loader, device)
        print(f"Test: F1={test_f1:.4f}, Acc={test_acc:.4f}")
        print(f"Baseline was: F1=0.8214, Acc=0.9120")
        print(f"Difference: F1={test_f1 - 0.8214:+.4f}")
        np.save(OUTPUT_DIR / "test_probs.npy", test_probs)
    except Exception as e:
        print(f"Could not evaluate on test: {e}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Best Dev F1: {best_f1:.4f}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
