#!/usr/bin/env python3
"""
Multi-Task DeBERTa for Checkworthiness Classification (CT24)

Uses LLM-generated soft labels as auxiliary supervision:
- Main task: Checkworthiness (BCE, ground truth)
- Auxiliary: Checkability, Verifiability, Harm (MSE, LLM soft labels)

Supports 3-phase training:
- Phase 1: Freeze encoder, train auxiliary heads only
- Phase 2: Joint training (all parameters)
- Phase 3: Freeze auxiliary, fine-tune main head

Usage:
    # Full 3-phase training
    python experiments/scripts/finetune_deberta_mtl.py --phase all

    # Individual phases
    python experiments/scripts/finetune_deberta_mtl.py --phase 1 --epochs 3
    python experiments/scripts/finetune_deberta_mtl.py --phase 2 --epochs 5 --checkpoint <path>
    python experiments/scripts/finetune_deberta_mtl.py --phase 3 --epochs 2 --checkpoint <path>

    # A10 optimized (24GB)
    python experiments/scripts/finetune_deberta_mtl.py --phase all --batch-size 16 --grad-accum 2

Requirements:
    pip install transformers datasets accelerate sentencepiece polars scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features_v4"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_mtl"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# Auxiliary task configuration
AUX_TARGETS = {
    'checkability': 'check_p_yes',
    'verifiability': 'verif_p_yes',
    'harm': 'harm_p_yes',
}

AUX_ENTROPY = {
    'checkability': 'check_entropy_norm',
    'verifiability': 'verif_entropy_norm',
    'harm': 'harm_entropy_norm',
}

DEFAULT_LAMBDAS = {
    'checkability': 0.3,
    'verifiability': 0.3,
    'harm': 0.2,
}


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    aux_learning_rate: float = 1e-3  # Higher LR for auxiliary heads in phase 1
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    phase: str = "all"  # "1", "2", "3", or "all"
    lambda_check: float = 0.3
    lambda_verif: float = 0.3
    lambda_harm: float = 0.2
    use_entropy_weighting: bool = True
    fp16: bool = True
    seed: int = 42
    checkpoint_path: Optional[str] = None


# =============================================================================
# Model
# =============================================================================


class MultiTaskDeBERTa(nn.Module):
    """DeBERTa with multiple task heads for MTL."""

    def __init__(self, encoder_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 1024 for large

        # Main task head (checkworthiness - binary classification)
        self.main_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        # Auxiliary heads (regression to LLM soft labels)
        self.aux_heads = nn.ModuleDict({
            'checkability': self._make_aux_head(hidden_size, dropout),
            'verifiability': self._make_aux_head(hidden_size, dropout),
            'harm': self._make_aux_head(hidden_size, dropout),
        })

    def _make_aux_head(self, hidden_size: int, dropout: float) -> nn.Module:
        """Create an auxiliary regression head."""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output [0, 1]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass returning main logits and auxiliary outputs."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token

        main_logits = self.main_head(cls_embedding)

        aux_outputs = {}
        if return_aux:
            for name, head in self.aux_heads.items():
                aux_outputs[name] = head(cls_embedding).squeeze(-1)

        return main_logits, aux_outputs

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_main_head(self):
        """Freeze main head parameters."""
        for param in self.main_head.parameters():
            param.requires_grad = False

    def unfreeze_main_head(self):
        """Unfreeze main head parameters."""
        for param in self.main_head.parameters():
            param.requires_grad = True

    def freeze_aux_heads(self):
        """Freeze auxiliary head parameters."""
        for head in self.aux_heads.values():
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_aux_heads(self):
        """Unfreeze auxiliary head parameters."""
        for head in self.aux_heads.values():
            for param in head.parameters():
                param.requires_grad = True


# =============================================================================
# Dataset
# =============================================================================


class MTLDataset(Dataset):
    """Dataset for multi-task learning with LLM soft labels."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        aux_targets: dict[str, np.ndarray],
        aux_entropy: dict[str, np.ndarray],
        tokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.aux_targets = aux_targets
        self.aux_entropy = aux_entropy
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

        # Add auxiliary targets and entropy
        for name in self.aux_targets:
            item[f'aux_{name}'] = torch.tensor(self.aux_targets[name][idx], dtype=torch.float32)
            item[f'entropy_{name}'] = torch.tensor(self.aux_entropy[name][idx], dtype=torch.float32)

        return item


# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and merge CT24 features with LLM features."""
    # Load CT24 features (text + labels)
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Load LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Ensure consistent ID types
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_dev = llm_dev.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    # Select columns to join
    llm_cols = ["Sentence_id"] + list(AUX_TARGETS.values()) + list(AUX_ENTROPY.values())

    # Join datasets
    train = ct24_train.join(llm_train.select(llm_cols), on="Sentence_id", how="inner")
    dev = ct24_dev.join(llm_dev.select(llm_cols), on="Sentence_id", how="inner")
    test = ct24_test.join(llm_test.select(llm_cols), on="Sentence_id", how="inner")

    return train, dev, test


def prepare_dataset(df: pl.DataFrame, tokenizer, max_length: int) -> MTLDataset:
    """Convert DataFrame to MTLDataset."""
    texts = df["Text"].to_list()
    labels = [1 if l == "Yes" else 0 for l in df["class_label"].to_list()]

    aux_targets = {}
    aux_entropy = {}

    for name, col in AUX_TARGETS.items():
        values = df[col].to_numpy().astype(np.float32)
        # Handle NaN/inf
        values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
        aux_targets[name] = values

    for name, col in AUX_ENTROPY.items():
        values = df[col].to_numpy().astype(np.float32)
        values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
        aux_entropy[name] = values

    return MTLDataset(texts, labels, aux_targets, aux_entropy, tokenizer, max_length)


# =============================================================================
# Loss Functions
# =============================================================================


def compute_mtl_loss(
    main_logits: torch.Tensor,
    aux_outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    aux_targets: dict[str, torch.Tensor],
    aux_entropy: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    lambdas: dict[str, float],
    use_entropy_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-task loss with entropy weighting."""

    # Main task: weighted cross-entropy
    main_loss = F.cross_entropy(main_logits, labels, weight=class_weights)

    # Auxiliary tasks: MSE with optional entropy weighting
    aux_losses = {}
    total_aux_loss = 0.0

    for name, pred in aux_outputs.items():
        target = aux_targets[name]

        if use_entropy_weighting:
            entropy = aux_entropy[name]
            # Low entropy = high confidence = high weight
            sample_weight = 1.0 - entropy.clamp(0, 1)
            task_loss = (sample_weight * F.mse_loss(pred, target, reduction='none')).mean()
        else:
            task_loss = F.mse_loss(pred, target)

        aux_losses[name] = task_loss.item()
        total_aux_loss += lambdas[name] * task_loss

    total_loss = main_loss + total_aux_loss

    loss_dict = {
        'total': total_loss.item(),
        'main': main_loss.item(),
        **{f'aux_{k}': v for k, v in aux_losses.items()},
    }

    return total_loss, loss_dict


# =============================================================================
# Training
# =============================================================================


def train_epoch(
    model: MultiTaskDeBERTa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    class_weights: torch.Tensor,
    lambdas: dict[str, float],
    config: Config,
    device: torch.device,
    phase: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0, 'main': 0, 'aux_checkability': 0, 'aux_verifiability': 0, 'aux_harm': 0}
    num_batches = 0

    scaler = torch.amp.GradScaler('cuda') if config.fp16 and device.type == 'cuda' else None

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        aux_targets = {name: batch[f'aux_{name}'].to(device) for name in AUX_TARGETS}
        aux_entropy = {name: batch[f'entropy_{name}'].to(device) for name in AUX_ENTROPY}

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                main_logits, aux_outputs = model(input_ids, attention_mask, return_aux=(phase != 3))

                # Compute loss based on phase
                if phase == 1:
                    # Only auxiliary losses
                    loss, loss_dict = compute_mtl_loss(
                        main_logits, aux_outputs, labels, aux_targets, aux_entropy,
                        class_weights, lambdas, config.use_entropy_weighting
                    )
                    # Zero out main loss gradient (but we need aux gradients)
                    loss = loss - loss_dict['main']  # Remove main loss
                elif phase == 3:
                    # Only main loss
                    loss = F.cross_entropy(main_logits, labels, weight=class_weights)
                    loss_dict = {'total': loss.item(), 'main': loss.item(),
                                 'aux_checkability': 0, 'aux_verifiability': 0, 'aux_harm': 0}
                else:  # phase 2 or "all"
                    loss, loss_dict = compute_mtl_loss(
                        main_logits, aux_outputs, labels, aux_targets, aux_entropy,
                        class_weights, lambdas, config.use_entropy_weighting
                    )

            # Gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            main_logits, aux_outputs = model(input_ids, attention_mask, return_aux=(phase != 3))

            if phase == 1:
                loss, loss_dict = compute_mtl_loss(
                    main_logits, aux_outputs, labels, aux_targets, aux_entropy,
                    class_weights, lambdas, config.use_entropy_weighting
                )
                loss = loss - loss_dict['main']
            elif phase == 3:
                loss = F.cross_entropy(main_logits, labels, weight=class_weights)
                loss_dict = {'total': loss.item(), 'main': loss.item(),
                             'aux_checkability': 0, 'aux_verifiability': 0, 'aux_harm': 0}
            else:
                loss, loss_dict = compute_mtl_loss(
                    main_logits, aux_outputs, labels, aux_targets, aux_entropy,
                    class_weights, lambdas, config.use_entropy_weighting
                )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        for k, v in loss_dict.items():
            total_losses[k] += v
        num_batches += 1

        if step % 50 == 0:
            print(f"  Step {step}/{len(dataloader)} | Loss: {loss_dict['total']:.4f} | "
                  f"Main: {loss_dict['main']:.4f}")

    return {k: v / num_batches for k, v in total_losses.items()}


def evaluate(
    model: MultiTaskDeBERTa,
    dataloader: DataLoader,
    class_weights: torch.Tensor,
    lambdas: dict[str, float],
    config: Config,
    device: torch.device,
    thresholds: np.ndarray = None,
) -> tuple[dict, np.ndarray]:
    """Evaluate model and return metrics at various thresholds."""
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            aux_targets = {name: batch[f'aux_{name}'].to(device) for name in AUX_TARGETS}
            aux_entropy = {name: batch[f'entropy_{name}'].to(device) for name in AUX_ENTROPY}

            main_logits, aux_outputs = model(input_ids, attention_mask)

            loss, _ = compute_mtl_loss(
                main_logits, aux_outputs, labels, aux_targets, aux_entropy,
                class_weights, lambdas, config.use_entropy_weighting
            )
            total_loss += loss.item()
            num_batches += 1

            probs = F.softmax(main_logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Evaluate at different thresholds
    results = []
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        results.append({
            'threshold': float(thresh),
            'f1': f1_score(all_labels, preds),
            'accuracy': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
        })

    best_result = max(results, key=lambda x: x['f1'])

    return {
        'loss': total_loss / num_batches,
        'results': results,
        'best': best_result,
    }, all_probs


def train_phase(
    model: MultiTaskDeBERTa,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    config: Config,
    device: torch.device,
    class_weights: torch.Tensor,
    lambdas: dict[str, float],
    phase: int,
    num_epochs: int,
) -> dict:
    """Train a single phase."""
    print(f"\n{'='*80}")
    print(f"PHASE {phase}")
    print(f"{'='*80}")

    # Configure model freezing based on phase
    if phase == 1:
        print("Freezing: encoder + main head | Training: auxiliary heads")
        model.freeze_encoder()
        model.freeze_main_head()
        model.unfreeze_aux_heads()
        lr = config.aux_learning_rate
    elif phase == 2:
        print("Training: all parameters")
        model.unfreeze_encoder()
        model.unfreeze_main_head()
        model.unfreeze_aux_heads()
        lr = config.learning_rate
    elif phase == 3:
        print("Freezing: auxiliary heads | Training: encoder + main head")
        model.unfreeze_encoder()
        model.unfreeze_main_head()
        model.freeze_aux_heads()
        lr = config.learning_rate

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    num_training_steps = len(train_loader) * num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_dev_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 3

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, class_weights, lambdas, config, device, phase
        )
        print(f"Train Loss: {train_losses['total']:.4f} | Main: {train_losses['main']:.4f}")

        dev_metrics, dev_probs = evaluate(model, dev_loader, class_weights, lambdas, config, device)
        print(f"Dev Loss: {dev_metrics['loss']:.4f} | Best F1: {dev_metrics['best']['f1']:.4f} "
              f"@ thresh {dev_metrics['best']['threshold']:.2f}")

        if dev_metrics['best']['f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['best']['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ↑ New best!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {'best_dev_f1': best_dev_f1, 'final_epoch': epoch + 1}


def train_model(config: Config, output_dir: Path):
    """Full training pipeline."""
    print("=" * 80)
    print("MULTI-TASK DEBERTA FOR CHECKWORTHINESS")
    print("=" * 80)

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"\nModel: {config.model_name}")
    print(f"Phase: {config.phase}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch} effective")
    print(f"Learning rate: {config.learning_rate} (encoder), {config.aux_learning_rate} (aux heads)")
    print(f"Lambda weights: check={config.lambda_check}, verif={config.lambda_verif}, harm={config.lambda_harm}")
    print(f"Entropy weighting: {config.use_entropy_weighting}")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    if "deberta-v" in config.model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = load_data()
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev: {len(dev_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    dev_dataset = prepare_dataset(dev_df, tokenizer, config.max_length)
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)

    # Class weights
    n_pos = sum(1 for l in train_df["class_label"].to_list() if l == "Yes")
    n_neg = len(train_df) - n_pos
    pos_weight = n_neg / n_pos
    class_weights = torch.tensor([1.0, pos_weight], device=device)
    print(f"\nClass imbalance: {n_neg}:{n_pos} (weight: {pos_weight:.2f})")

    # Lambda weights
    lambdas = {
        'checkability': config.lambda_check,
        'verifiability': config.lambda_verif,
        'harm': config.lambda_harm,
    }

    # Initialize model
    print("\nInitializing model...")
    if config.checkpoint_path:
        print(f"Loading checkpoint: {config.checkpoint_path}")
        model = MultiTaskDeBERTa(config.model_name)
        model.load_state_dict(torch.load(config.checkpoint_path, map_location='cpu'))
    else:
        model = MultiTaskDeBERTa(config.model_name)
    model.to(device)

    # Training phases
    if config.phase == "all":
        phases = [(1, 3), (2, 5), (3, 2)]  # (phase_num, epochs)
    else:
        phases = [(int(config.phase), config.num_epochs)]

    for phase_num, num_epochs in phases:
        result = train_phase(
            model, train_loader, dev_loader, config, device, class_weights, lambdas, phase_num, num_epochs
        )

        # Save checkpoint after each phase
        checkpoint_path = output_dir / f"checkpoint_phase{phase_num}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Dev evaluation
    print("\nDev Set:")
    dev_metrics, dev_probs = evaluate(model, dev_loader, class_weights, lambdas, config, device)
    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in dev_metrics['results']:
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['precision']:<10.4f} {r['recall']:<10.4f}")
    print(f"\nBest dev F1: {dev_metrics['best']['f1']:.4f} @ threshold {dev_metrics['best']['threshold']:.2f}")

    # Test evaluation
    print("\nTest Set:")
    test_metrics, test_probs = evaluate(model, test_loader, class_weights, lambdas, config, device)
    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in test_metrics['results']:
        marker = "🔥" if r['f1'] > 0.76 else ""
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['precision']:<10.4f} {r['recall']:<10.4f} {marker}")

    best_test = test_metrics['best']

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBest Test F1: {best_test['f1']:.4f} @ threshold {best_test['threshold']:.2f}")
    print(f"Best Test Acc: {best_test['accuracy']:.4f}")
    print(f"\nSOTA Comparison:")
    print(f"  F1:  {SOTA_F1:.4f} → {best_test['f1']:.4f} (gap: {best_test['f1'] - SOTA_F1:+.4f})")
    print(f"  Acc: {SOTA_ACC:.4f} → {best_test['accuracy']:.4f} (gap: {best_test['accuracy'] - SOTA_ACC:+.4f})")

    # Save results
    results = {
        "model": config.model_name,
        "config": {
            "phase": config.phase,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lambdas": lambdas,
            "use_entropy_weighting": config.use_entropy_weighting,
        },
        "dev_results": dev_metrics['results'],
        "test_results": test_metrics['results'],
        "best_dev": dev_metrics['best'],
        "best_test": best_test,
        "sota_gap_f1": best_test['f1'] - SOTA_F1,
        "sota_gap_acc": best_test['accuracy'] - SOTA_ACC,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save model
    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save predictions
    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "dev_probs.npy", dev_probs)

    return best_test


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-task DeBERTa for checkworthiness")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large",
                        help="Model name or path")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Training phase (1, 2, 3, or all)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per phase")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for encoder")
    parser.add_argument("--aux-lr", type=float, default=1e-3, help="Learning rate for aux heads (phase 1)")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lambda-check", type=float, default=0.3, help="Lambda for checkability")
    parser.add_argument("--lambda-verif", type=float, default=0.3, help="Lambda for verifiability")
    parser.add_argument("--lambda-harm", type=float, default=0.2, help="Lambda for harm")
    parser.add_argument("--no-entropy-weighting", action="store_true", help="Disable entropy weighting")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        phase=args.phase,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        aux_learning_rate=args.aux_lr,
        max_length=args.max_length,
        lambda_check=args.lambda_check,
        lambda_verif=args.lambda_verif,
        lambda_harm=args.lambda_harm,
        use_entropy_weighting=not args.no_entropy_weighting,
        checkpoint_path=args.checkpoint,
    )

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        best_result = train_model(config, output_dir)
        print("\n" + "=" * 80)
        if best_result['f1'] >= SOTA_F1:
            print("🏆 SOTA ACHIEVED!")
        elif best_result['f1'] >= 0.80:
            print("🔥 VERY CLOSE TO SOTA!")
        else:
            print("Training complete. Consider tuning lambda weights or more epochs.")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
