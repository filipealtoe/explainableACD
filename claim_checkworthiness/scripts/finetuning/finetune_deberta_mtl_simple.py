#!/usr/bin/env python3
"""
Simple Multi-Task Learning: End-to-end joint training.

Builds on our fine-tuned DeBERTa baseline (F1=0.82).
Adds 3 auxiliary heads and trains everything jointly.

    Text → Shared Encoder → Multiple Heads → Joint Loss → prediction

Usage:
    python experiments/scripts/finetune_deberta_mtl_simple.py
    python experiments/scripts/finetune_deberta_mtl_simple.py --epochs 3 --aux_weight 0.3
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
from transformers import DebertaV2Tokenizer, AutoConfig
from safetensors.torch import load_file

# =============================================================================
# Configuration
# =============================================================================

BASELINE_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
LLM_FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_mtl_simple"


# =============================================================================
# Model: Simple MTL with joint training
# =============================================================================

class DeBERTaMTLSimple(nn.Module):
    """DeBERTa with main head + 3 auxiliary heads, all trained jointly."""

    def __init__(self, base_model, hidden_size=1024):
        super().__init__()
        self.deberta = base_model.deberta
        self.pooler = base_model.pooler
        self.dropout = nn.Dropout(0.1)

        # Main head (binary classification)
        self.main_head = nn.Linear(hidden_size, 2)

        # Auxiliary heads (regression: predict LLM soft labels)
        self.check_head = nn.Linear(hidden_size, 1)
        self.verif_head = nn.Linear(hidden_size, 1)
        self.harm_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0]  # [CLS]
        pooled = torch.tanh(self.pooler(hidden))
        pooled = self.dropout(pooled)

        # All heads in parallel
        main_logits = self.main_head(pooled)
        check_pred = torch.sigmoid(self.check_head(pooled)).squeeze(-1)
        verif_pred = torch.sigmoid(self.verif_head(pooled)).squeeze(-1)
        harm_pred = torch.sigmoid(self.harm_head(pooled)).squeeze(-1)

        return main_logits, check_pred, verif_pred, harm_pred


# =============================================================================
# Dataset
# =============================================================================

class MTLDataset(Dataset):
    def __init__(self, texts, labels, sentence_ids, llm_df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        self.sentence_ids = sentence_ids

        # LLM features lookup
        self.llm_data = {row["sentence_id"]: row for row in llm_df.iter_rows(named=True)}

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

        llm = self.llm_data.get(self.sentence_ids[idx], {})

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "check_target": torch.tensor(llm.get("check_p_yes", 0.5), dtype=torch.float),
            "verif_target": torch.tensor(llm.get("verif_p_yes", 0.5), dtype=torch.float),
            "harm_target": torch.tensor(llm.get("harm_p_yes", 0.5), dtype=torch.float),
        }


def load_data(split, tokenizer):
    raw_df = pl.read_csv(DATA_DIR / f"CT24_checkworthy_english_{split}.tsv", separator="\t")
    llm_df = pl.read_parquet(LLM_FEATURES_DIR / f"{split}_llm_features.parquet")

    return MTLDataset(
        texts=raw_df["Text"].to_list(),
        labels=[1 if l == "Yes" else 0 for l in raw_df["class_label"].to_list()],
        sentence_ids=raw_df["Sentence_id"].to_list(),
        llm_df=llm_df,
        tokenizer=tokenizer,
    )


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device, aux_weight):
    model.train()
    total_loss, total_main, total_aux = 0, 0, 0

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        check_t = batch["check_target"].to(device)
        verif_t = batch["verif_target"].to(device)
        harm_t = batch["harm_target"].to(device)

        optimizer.zero_grad()

        main_logits, check_p, verif_p, harm_p = model(input_ids, attention_mask)

        # Joint loss
        loss_main = ce_loss(main_logits, labels)
        loss_aux = mse_loss(check_p, check_t) + mse_loss(verif_p, verif_t) + mse_loss(harm_p, harm_t)
        loss = loss_main + aux_weight * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_main += loss_main.item()
        total_aux += loss_aux.item()

    n = len(loader)
    return total_loss / n, total_main / n, total_aux / n


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
    parser.add_argument("--aux_weight", type=float, default=0.3, help="Weight for auxiliary losses")
    args = parser.parse_args()

    print("=" * 70)
    print("SIMPLE MTL: End-to-End Joint Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load baseline
    print(f"\nLoading baseline from: {BASELINE_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(BASELINE_MODEL)

    from transformers import AutoModelForSequenceClassification
    base_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_MODEL)

    # Create MTL model
    model = DeBERTaMTLSimple(base_model, hidden_size=1024)

    # Copy main head weights from baseline
    model.main_head.weight.data = base_model.classifier.weight.data.clone()
    model.main_head.bias.data = base_model.classifier.bias.data.clone()

    model.to(device)
    print("Baseline loaded, auxiliary heads initialized!")

    # Data
    print("\nLoading data...")
    train_dataset = load_data("train", tokenizer)
    dev_dataset = load_data("dev", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # Baseline eval
    f1, acc, _ = evaluate(model, dev_loader, device)
    print(f"\nBaseline: F1={f1:.4f}, Acc={acc:.4f}")
    best_f1 = f1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Joint training
    print(f"\n{'='*70}")
    print(f"Joint training: {args.epochs} epochs, aux_weight={args.aux_weight}")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss, main_loss, aux_loss = train_epoch(model, train_loader, optimizer, device, args.aux_weight)
        f1, acc, probs = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} (main={main_loss:.4f}, aux={aux_loss:.4f}) | F1: {f1:.4f}, Acc: {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            np.save(OUTPUT_DIR / "dev_probs.npy", probs)
            print(f"  → New best! Saved.")

    print(f"\n{'='*70}")
    print(f"DONE. Best F1: {best_f1:.4f}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
