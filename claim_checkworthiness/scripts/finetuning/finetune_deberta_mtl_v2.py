#!/usr/bin/env python3
"""
Multi-Task Learning: Build on our fine-tuned DeBERTa baseline.

Architecture:
    [Pretrained DeBERTa encoder + main head] (already trained, F1=0.82)
            ↓
    + 3 auxiliary regression heads (random init)
    - check_head → predicts check_p_yes
    - verif_head → predicts verif_p_yes
    - harm_head  → predicts harm_p_yes

Training phases:
    Phase 1: Freeze encoder + main head, train aux heads only
    Phase 2: Unfreeze all, joint training with weighted losses
    Phase 3: Polish main head only (optional)

Usage:
    python experiments/scripts/finetune_deberta_mtl_v2.py
    python experiments/scripts/finetune_deberta_mtl_v2.py --phase1_epochs 1 --phase2_epochs 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# =============================================================================
# Configuration
# =============================================================================

BASELINE_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
LLM_FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_mtl_v2"


# =============================================================================
# Model
# =============================================================================

class DeBERTaMTL(DebertaV2PreTrainedModel):
    """DeBERTa with main classification head + 3 auxiliary regression heads."""

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Main head (binary classification - checkworthiness)
        self.main_head = nn.Linear(config.hidden_size, 2)

        # Auxiliary heads (regression - predict LLM soft labels)
        self.check_head = nn.Linear(config.hidden_size, 1)
        self.verif_head = nn.Linear(config.hidden_size, 1)
        self.harm_head = nn.Linear(config.hidden_size, 1)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        check_targets=None,
        verif_targets=None,
        harm_targets=None,
        check_weights=None,
        verif_weights=None,
        harm_weights=None,
        return_aux=False,
    ):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.pooler_activation(self.pooler(hidden))
        pooled = self.dropout(pooled)

        # Main head
        main_logits = self.main_head(pooled)

        # Auxiliary heads
        check_pred = torch.sigmoid(self.check_head(pooled)).squeeze(-1)
        verif_pred = torch.sigmoid(self.verif_head(pooled)).squeeze(-1)
        harm_pred = torch.sigmoid(self.harm_head(pooled)).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(main_logits, labels)

            # Add auxiliary losses if targets provided
            if check_targets is not None:
                aux_loss = self._compute_aux_loss(check_pred, check_targets, check_weights)
                aux_loss += self._compute_aux_loss(verif_pred, verif_targets, verif_weights)
                aux_loss += self._compute_aux_loss(harm_pred, harm_targets, harm_weights)
                loss = loss + 0.3 * aux_loss  # Weight auxiliary loss

        if return_aux:
            return main_logits, check_pred, verif_pred, harm_pred, loss

        return SequenceClassifierOutput(loss=loss, logits=main_logits)

    def _compute_aux_loss(self, pred, target, weight=None):
        """MSE loss with optional sample weighting."""
        if target is None:
            return 0.0
        mse = (pred - target) ** 2
        if weight is not None:
            # Higher entropy = more uncertain = lower weight
            w = 1.0 - weight  # Invert: low entropy → high weight
            mse = mse * w
        return mse.mean()


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class MTLSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    check_target: float
    verif_target: float
    harm_target: float
    check_weight: float
    verif_weight: float
    harm_weight: float


class MTLDataset(Dataset):
    def __init__(self, texts, labels, llm_df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        # Convert LLM features to dict for fast lookup
        self.llm_data = {
            row["sentence_id"]: row
            for row in llm_df.iter_rows(named=True)
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sentence_id = self.sentence_ids[idx] if hasattr(self, 'sentence_ids') else None

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Get LLM features if available
        llm = self.llm_data.get(sentence_id, {})

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "check_targets": torch.tensor(llm.get("check_p_yes", 0.5), dtype=torch.float),
            "verif_targets": torch.tensor(llm.get("verif_p_yes", 0.5), dtype=torch.float),
            "harm_targets": torch.tensor(llm.get("harm_p_yes", 0.5), dtype=torch.float),
            "check_weights": torch.tensor(llm.get("check_entropy_norm", 0.5), dtype=torch.float),
            "verif_weights": torch.tensor(llm.get("verif_entropy_norm", 0.5), dtype=torch.float),
            "harm_weights": torch.tensor(llm.get("harm_entropy_norm", 0.5), dtype=torch.float),
        }


def create_dataset(split: str, tokenizer) -> MTLDataset:
    """Load data and create dataset."""
    # Load raw text data
    if split == "train":
        raw_path = DATA_DIR / "CT24_checkworthy_english_train.tsv"
    elif split == "dev":
        raw_path = DATA_DIR / "CT24_checkworthy_english_dev.tsv"
    else:
        raw_path = DATA_DIR / f"CT24_checkworthy_english_{split}.tsv"

    raw_df = pl.read_csv(raw_path, separator="\t")

    # Load LLM features
    llm_path = LLM_FEATURES_DIR / f"{split}_llm_features.parquet"
    llm_df = pl.read_parquet(llm_path)

    texts = raw_df["Text"].to_list()
    labels = [1 if l == "Yes" else 0 for l in raw_df["class_label"].to_list()]
    sentence_ids = raw_df["Sentence_id"].to_list()

    dataset = MTLDataset(texts, labels, llm_df, tokenizer)
    dataset.sentence_ids = sentence_ids

    return dataset


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, phase: str):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Training ({phase})"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            preds = (probs >= threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, np.array(all_probs)


def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_params(module):
    for param in module.parameters():
        param.requires_grad = True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_epochs", type=int, default=2, help="Epochs for phase 1 (aux heads only)")
    parser.add_argument("--phase2_epochs", type=int, default=3, help="Epochs for phase 2 (joint)")
    parser.add_argument("--phase3_epochs", type=int, default=1, help="Epochs for phase 3 (polish main)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_aux", type=float, default=2e-4, help="LR for aux heads (phase 1)")
    parser.add_argument("--lr_joint", type=float, default=5e-6, help="LR for joint training (phase 2)")
    parser.add_argument("--lr_polish", type=float, default=1e-6, help="LR for polishing (phase 3)")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-TASK LEARNING: Building on DeBERTa Baseline")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load tokenizer from baseline
    print(f"\nLoading baseline model from: {BASELINE_MODEL}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(BASELINE_MODEL)

    # Create MTL model and load baseline weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(BASELINE_MODEL)
    model = DeBERTaMTL(config)

    # Load baseline weights (encoder + main head)
    baseline_state = torch.load(BASELINE_MODEL / "model.safetensors", map_location="cpu", weights_only=False)
    # Handle safetensors format
    if not baseline_state:
        from safetensors.torch import load_file
        baseline_state = load_file(BASELINE_MODEL / "model.safetensors")

    # Map baseline weights to our model
    model_state = model.state_dict()
    for name, param in baseline_state.items():
        # Map classifier to main_head
        if name == "classifier.weight":
            model_state["main_head.weight"] = param
        elif name == "classifier.bias":
            model_state["main_head.bias"] = param
        elif name in model_state:
            model_state[name] = param

    model.load_state_dict(model_state)
    model.to(device)
    print("Baseline weights loaded!")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = create_dataset("train", tokenizer)
    dev_dataset = create_dataset("dev", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # Evaluate baseline
    print("\nBaseline evaluation...")
    f1, acc, _ = evaluate(model, dev_loader, device)
    print(f"Baseline - F1: {f1:.4f}, Acc: {acc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = f1

    # =========================================================================
    # Phase 1: Train auxiliary heads only (encoder + main head frozen)
    # =========================================================================
    if args.phase1_epochs > 0:
        print(f"\n{'='*70}")
        print("PHASE 1: Training auxiliary heads only")
        print("='*70")

        freeze_params(model.deberta)
        freeze_params(model.pooler)
        freeze_params(model.main_head)

        aux_params = list(model.check_head.parameters()) + \
                     list(model.verif_head.parameters()) + \
                     list(model.harm_head.parameters())
        optimizer = torch.optim.AdamW(aux_params, lr=args.lr_aux)

        for epoch in range(args.phase1_epochs):
            loss = train_epoch(model, train_loader, optimizer, device, "Phase 1")
            f1, acc, _ = evaluate(model, dev_loader, device)
            print(f"Epoch {epoch+1}/{args.phase1_epochs} - Loss: {loss:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
                print(f"  → New best F1: {f1:.4f}")

    # =========================================================================
    # Phase 2: Joint training (all unfrozen)
    # =========================================================================
    if args.phase2_epochs > 0:
        print(f"\n{'='*70}")
        print("PHASE 2: Joint training (all parameters)")
        print("='*70")

        unfreeze_params(model.deberta)
        unfreeze_params(model.pooler)
        unfreeze_params(model.main_head)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_joint)

        for epoch in range(args.phase2_epochs):
            loss = train_epoch(model, train_loader, optimizer, device, "Phase 2")
            f1, acc, _ = evaluate(model, dev_loader, device)
            print(f"Epoch {epoch+1}/{args.phase2_epochs} - Loss: {loss:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
                print(f"  → New best F1: {f1:.4f}")

    # =========================================================================
    # Phase 3: Polish main head only
    # =========================================================================
    if args.phase3_epochs > 0:
        print(f"\n{'='*70}")
        print("PHASE 3: Polishing main head")
        print("='*70")

        # Load best model from previous phases
        if (OUTPUT_DIR / "best_model.pt").exists():
            model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))

        freeze_params(model.check_head)
        freeze_params(model.verif_head)
        freeze_params(model.harm_head)
        unfreeze_params(model.deberta)
        unfreeze_params(model.pooler)
        unfreeze_params(model.main_head)

        # Only optimize encoder + main head, no aux loss
        main_params = list(model.deberta.parameters()) + \
                      list(model.pooler.parameters()) + \
                      list(model.main_head.parameters())
        optimizer = torch.optim.AdamW(main_params, lr=args.lr_polish)

        for epoch in range(args.phase3_epochs):
            # Train without aux targets
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc="Phase 3"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            f1, acc, _ = evaluate(model, dev_loader, device)
            print(f"Epoch {epoch+1}/{args.phase3_epochs} - Loss: {total_loss/len(train_loader):.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
                print(f"  → New best F1: {f1:.4f}")

    # =========================================================================
    # Final evaluation
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("='*70")

    # Load best model
    if (OUTPUT_DIR / "best_model.pt").exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))

    f1, acc, probs = evaluate(model, dev_loader, device)
    print(f"Best Dev - F1: {f1:.4f}, Acc: {acc:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
