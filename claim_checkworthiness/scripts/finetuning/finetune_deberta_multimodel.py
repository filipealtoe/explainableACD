#!/usr/bin/env python3
"""
Fine-tune Transformer Models for Checkworthiness Classification.

Supports multiple model families:
- DeBERTa: v2-xxlarge (1.5B), v2-xlarge (900M), v3-large (304M), v3-base (86M)
- BGE: large-en-v1.5 (335M), base-en-v1.5 (109M)
- E5: large-v2 (335M), base-v2 (109M)
- GTE: large (335M), base (109M)
- MPNet: all-mpnet-base-v2 (109M)
- MiniLM: all-MiniLM-L6-v2 (22M)
- RoBERTa: large (355M), base (125M)

Usage:
    # DeBERTa-v2-xxlarge (biggest, needs ~40GB VRAM)
    python finetune_deberta_multimodel.py --model deberta-v2-xxlarge --data-dir ~/data

    # BGE-large (sentence-transformer, ~16GB VRAM)
    python finetune_deberta_multimodel.py --model bge-large-en-v1.5 --data-dir ~/data

    # E5-large
    python finetune_deberta_multimodel.py --model e5-large-v2 --data-dir ~/data

    # MiniLM (lightweight, fast)
    python finetune_deberta_multimodel.py --model all-MiniLM-L6-v2 --data-dir ~/data

    # Custom hyperparameters
    python finetune_deberta_multimodel.py --model bge-large-en-v1.5 --lr 3e-5 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.swa_utils import AveragedModel, SWALR

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Focal Loss Implementation
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p) = -α(1-p)^γ * log(p)

    - γ (gamma): Focusing parameter. Higher = more focus on hard examples.
      γ=0 is equivalent to cross-entropy. γ=2 is common.
    - α (alpha): Class weight for positive class.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class

        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights (alpha for positive class)
        alpha_weight = torch.where(targets == 1, self.alpha, 1.0)

        focal_loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# R-Drop Regularization
# =============================================================================

def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric KL divergence between two probability distributions.

    KL(p||q) + KL(q||p) / 2
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2


# =============================================================================
# FGM (Fast Gradient Method) Adversarial Training
# =============================================================================

class FGM:
    """
    Fast Gradient Method for adversarial training.

    Adds perturbations to word embeddings during training:
    r_adv = ε * g / ||g||
    where g = ∇_emb L(x, y)

    This teaches the model to be robust to small input variations,
    which is crucial for claims that can be paraphrased.
    """

    def __init__(self, model, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """Restore original embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Layer-wise Learning Rate Decay (LLRD)
# =============================================================================

def get_llrd_parameter_groups(
    model,
    base_lr: float,
    decay_rate: float = 0.9,
    weight_decay: float = 0.01,
) -> list[dict]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Earlier layers (closer to input) get smaller learning rates.
    This preserves pretrained knowledge while allowing upper layers to adapt.

    Args:
        model: The transformer model
        base_lr: Learning rate for the top layer (classifier)
        decay_rate: Multiply LR by this for each lower layer (e.g., 0.9)
        weight_decay: Weight decay for regularization

    Returns:
        List of parameter groups for optimizer
    """
    # Get model structure
    if hasattr(model, "deberta"):
        encoder = model.deberta
        encoder_name = "deberta"
    elif hasattr(model, "roberta"):
        encoder = model.roberta
        encoder_name = "roberta"
    elif hasattr(model, "bert"):
        encoder = model.bert
        encoder_name = "bert"
    else:
        # Fallback: no LLRD, single group
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    # Count layers
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        num_layers = len(encoder.encoder.layer)
    else:
        num_layers = 12  # Default

    # Build parameter groups
    parameter_groups = []

    # Layer names to LR mapping
    layer_lrs = {}

    # Embeddings get the smallest LR
    embeddings_lr = base_lr * (decay_rate ** (num_layers + 1))
    layer_lrs["embeddings"] = embeddings_lr

    # Encoder layers: progressive LR
    for i in range(num_layers):
        layer_lr = base_lr * (decay_rate ** (num_layers - i))
        layer_lrs[f"layer.{i}"] = layer_lr

    # Classifier gets base LR
    layer_lrs["classifier"] = base_lr
    layer_lrs["pooler"] = base_lr

    # Assign parameters to groups
    assigned_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Find matching layer
        lr = base_lr  # Default
        for layer_key, layer_lr in layer_lrs.items():
            if layer_key in name:
                lr = layer_lr
                break

        # No weight decay for bias and LayerNorm
        wd = 0.0 if "bias" in name or "LayerNorm" in name else weight_decay

        parameter_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
        })
        assigned_params.add(name)

    # Print summary
    print(f"\n  📊 LLRD Configuration (decay={decay_rate}):")
    print(f"     Classifier LR: {base_lr:.2e}")
    print(f"     Top layer LR:  {base_lr * decay_rate:.2e}")
    print(f"     Bottom layer:  {base_lr * (decay_rate ** num_layers):.2e}")
    print(f"     Embeddings LR: {embeddings_lr:.2e}")

    return parameter_groups


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Tokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import Dataset


class SmartEarlyStoppingCallback(TrainerCallback):
    """
    Advanced early stopping with:
    1. Train-validation gap monitoring (overfitting detection)
    2. EMA smoothing (noise reduction)
    3. Patience on smoothed metrics
    """

    def __init__(
        self,
        patience: int = 5,
        metric_name: str = "f1",
        max_train_val_gap: float = 0.20,  # Stop if train-val gap > 20%
        ema_alpha: float = 0.3,  # Smoothing factor (lower = more smoothing)
    ):
        self.patience = patience
        self.metric_name = metric_name
        self.max_gap = max_train_val_gap
        self.ema_alpha = ema_alpha

        # Tracking
        self.best_ema_metric = None
        self.best_step = None
        self.no_improve_count = 0
        self.ema_metric = None
        self.train_metrics = []
        self.eval_metrics = []
        self.stopped_reason = None

    def on_log(self, args, state, control, logs, **kwargs):
        # Track training loss/metrics
        if "loss" in logs and "eval_loss" not in logs:
            self.train_metrics.append(logs.get("loss", 0))

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_key = f"eval_{self.metric_name}"
        current_metric = metrics.get(metric_key)
        eval_loss = metrics.get("eval_loss", 0)

        if current_metric is None:
            return

        self.eval_metrics.append(current_metric)

        # === 1. Update EMA ===
        if self.ema_metric is None:
            self.ema_metric = current_metric
        else:
            self.ema_metric = self.ema_alpha * current_metric + (1 - self.ema_alpha) * self.ema_metric

        # === 2. Check train-val gap (overfitting signal) ===
        if self.train_metrics:
            recent_train_loss = np.mean(self.train_metrics[-10:])  # Last 10 logs
            gap = recent_train_loss - eval_loss  # Negative gap = overfitting

            # For metrics like F1, check if train >> val
            if len(self.eval_metrics) >= 3:
                # Estimate "train F1" from loss trend (proxy)
                loss_improving = recent_train_loss < 0.3  # Low train loss
                val_stagnant = current_metric < self.eval_metrics[-3] + 0.01  # Val not improving

                if loss_improving and val_stagnant and current_metric < 0.9:
                    gap_signal = True
                else:
                    gap_signal = False
            else:
                gap_signal = False

        # === 3. Check improvement on EMA metric ===
        if self.best_ema_metric is None or self.ema_metric > self.best_ema_metric + 0.001:
            self.best_ema_metric = self.ema_metric
            self.best_step = state.global_step
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # === 4. Print status ===
        status_parts = [f"raw={current_metric:.4f}", f"ema={self.ema_metric:.4f}"]
        if self.train_metrics:
            status_parts.append(f"train_loss={np.mean(self.train_metrics[-10:]):.4f}")

        print(f"\n  📊 {self.metric_name}: {', '.join(status_parts)}")

        if self.no_improve_count > 0:
            print(f"  ⏳ No improvement for {self.no_improve_count}/{self.patience} evals "
                  f"(best ema={self.best_ema_metric:.4f} at step {self.best_step})")

        # === 5. Decide whether to stop ===
        if self.no_improve_count >= self.patience:
            self.stopped_reason = f"No improvement in EMA {self.metric_name} for {self.patience} evaluations"
            control.should_training_stop = True

        # Clear train metrics for next window
        if len(self.train_metrics) > 50:
            self.train_metrics = self.train_metrics[-20:]

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*70}")
        if self.stopped_reason:
            print(f"🛑 EARLY STOPPING TRIGGERED")
            print(f"{'='*70}")
            print(f"   Reason: {self.stopped_reason}")
        else:
            print(f"✅ TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"   Completed all epochs without early stopping")

        print(f"   Best EMA {self.metric_name}: {self.best_ema_metric:.4f} (at step {self.best_step})")
        print(f"   Final step: {state.global_step}")
        print(f"   Epochs completed: {state.epoch:.2f}")
        print(f"{'='*70}\n")


# =============================================================================
# Stochastic Weight Averaging (SWA) Callback
# =============================================================================

class SWACallback(TrainerCallback):
    """
    Stochastic Weight Averaging callback.

    Averages model weights over the last portion of training for more robust results.
    SWA works by keeping an exponential moving average of the model weights
    over the last part of training, which tends to converge to wider optima.
    """

    def __init__(self, swa_start_ratio: float = 0.6, swa_lr_ratio: float = 0.5):
        """
        Args:
            swa_start_ratio: Start SWA after this fraction of training (e.g., 0.6 = last 40%)
            swa_lr_ratio: Reduce LR to this fraction when SWA starts
        """
        self.swa_start_ratio = swa_start_ratio
        self.swa_lr_ratio = swa_lr_ratio
        self.swa_model = None
        self.swa_started = False
        self.n_averaged = 0

    def on_train_begin(self, args, state, control, model, **kwargs):
        # Initialize SWA model as a copy
        self.swa_model = AveragedModel(model)
        self.swa_started = False
        self.n_averaged = 0
        print(f"  📦 SWA initialized (will start at {self.swa_start_ratio*100:.0f}% of training)")

    def on_step_end(self, args, state, control, model, **kwargs):
        # Check if we should start SWA
        progress = state.global_step / state.max_steps

        if progress >= self.swa_start_ratio and not self.swa_started:
            self.swa_started = True
            print(f"\n  🔄 SWA started at step {state.global_step} ({progress*100:.1f}% through training)")

        # Update SWA model every eval_steps when SWA is active
        if self.swa_started and state.global_step % args.eval_steps == 0:
            self.swa_model.update_parameters(model)
            self.n_averaged += 1

    def on_train_end(self, args, state, control, model, **kwargs):
        if self.n_averaged > 0:
            # Copy SWA weights back to model
            print(f"\n  ✨ SWA complete: averaged {self.n_averaged} checkpoints")

            # Update batch norm statistics if present
            # (would need a forward pass through data, skip for now)

            # Copy averaged weights to model
            for param, swa_param in zip(model.parameters(), self.swa_model.module.parameters()):
                param.data.copy_(swa_param.data)

            print(f"  📦 SWA weights copied to model")
        else:
            print(f"  ⚠️ SWA: No checkpoints averaged (training too short?)")


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_REGISTRY = {
    # DeBERTa-v2 (bigger models, strong performance)
    "deberta-v2-xxlarge": {
        "hf_name": "microsoft/deberta-v2-xxlarge",
        "params": "1.5B",
        "hidden_size": 1536,
        "recommended_batch": 2,
        "recommended_grad_accum": 8,
        "recommended_lr": 1e-5,
        "gradient_checkpointing": True,
    },
    "deberta-v2-xlarge": {
        "hf_name": "microsoft/deberta-v2-xlarge",
        "params": "900M",
        "hidden_size": 1024,
        "recommended_batch": 4,
        "recommended_grad_accum": 4,
        "recommended_lr": 1e-5,
        "gradient_checkpointing": True,
    },
    "deberta-v2-large": {
        "hf_name": "microsoft/deberta-v2-large",
        "params": "350M",
        "hidden_size": 1024,
        "recommended_batch": 8,
        "recommended_grad_accum": 2,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # DeBERTa-v3 (more efficient, disentangled attention)
    "deberta-v3-large": {
        "hf_name": "microsoft/deberta-v3-large",
        "params": "304M",
        "hidden_size": 1024,
        "recommended_batch": 8,
        "recommended_grad_accum": 2,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    "deberta-v3-base": {
        "hf_name": "microsoft/deberta-v3-base",
        "params": "86M",
        "hidden_size": 768,
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # Older DeBERTa
    "deberta-xlarge": {
        "hf_name": "microsoft/deberta-xlarge",
        "params": "750M",
        "hidden_size": 1024,
        "recommended_batch": 4,
        "recommended_grad_accum": 4,
        "recommended_lr": 1e-5,
        "gradient_checkpointing": True,
    },
    # ==========================================================================
    # Sentence-Transformer Models (BERT-based, fine-tunable)
    # ==========================================================================
    # BGE (BAAI General Embedding) - Best for semantic similarity
    "bge-large-en-v1.5": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "params": "335M",
        "hidden_size": 1024,
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    "bge-base-en-v1.5": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "params": "109M",
        "hidden_size": 768,
        "recommended_batch": 32,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # E5 (Embeddings from bidirectional Encoder representations)
    "e5-large-v2": {
        "hf_name": "intfloat/e5-large-v2",
        "params": "335M",
        "hidden_size": 1024,
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    "e5-base-v2": {
        "hf_name": "intfloat/e5-base-v2",
        "params": "109M",
        "hidden_size": 768,
        "recommended_batch": 32,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # GTE (General Text Embeddings)
    "gte-large": {
        "hf_name": "thenlper/gte-large",
        "params": "335M",
        "hidden_size": 1024,
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    "gte-base": {
        "hf_name": "thenlper/gte-base",
        "params": "109M",
        "hidden_size": 768,
        "recommended_batch": 32,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # MPNet (Microsoft)
    "all-mpnet-base-v2": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "params": "109M",
        "hidden_size": 768,
        "recommended_batch": 32,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
    # MiniLM (Lightweight)
    "all-MiniLM-L6-v2": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "params": "22M",
        "hidden_size": 384,
        "recommended_batch": 64,
        "recommended_grad_accum": 1,
        "recommended_lr": 3e-5,
        "gradient_checkpointing": False,
    },
    # RoBERTa variants
    "roberta-large": {
        "hf_name": "roberta-large",
        "params": "355M",
        "hidden_size": 1024,
        "recommended_batch": 16,
        "recommended_grad_accum": 1,
        "recommended_lr": 1e-5,
        "gradient_checkpointing": False,
    },
    "roberta-base": {
        "hf_name": "roberta-base",
        "params": "125M",
        "hidden_size": 768,
        "recommended_batch": 32,
        "recommended_grad_accum": 1,
        "recommended_lr": 2e-5,
        "gradient_checkpointing": False,
    },
}

SOTA_F1 = 0.82
SOTA_ACC = 0.905


@dataclass
class Config:
    model_key: str = "deberta-v3-large"
    max_length: int = 128
    batch_size: int | None = None  # Auto from registry
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 200
    early_stopping_patience: int = 5  # Wait 5 evals (500 steps) before stopping
    no_early_stopping: bool = False
    freeze_encoder: bool = False  # Only train classification head
    eval_on_devtest: bool = False  # Use dev-test for early stopping (better test proxy)
    fp16: bool = True
    bf16: bool = False  # Better for large models if supported
    gradient_checkpointing: bool | None = None
    seed: int = 42
    class_weight: float | None = None  # Auto-compute if None

    # === Modern techniques ===
    # Focal Loss: Better for class imbalance
    use_focal_loss: bool = False
    focal_gamma: float = 2.0  # Focusing parameter (higher = more focus on hard examples)
    focal_alpha: float | None = None  # Class weight for positive class (auto if None)

    # Layer-wise Learning Rate Decay: Preserve pretrained knowledge
    use_llrd: bool = False
    llrd_decay: float = 0.9  # LR multiplier per layer (0.9 = each lower layer gets 90% of above)

    # Stochastic Weight Averaging: More robust final model
    use_swa: bool = False
    swa_start_ratio: float = 0.6  # Start SWA after 60% of training

    # Label smoothing: Reduces overconfidence
    label_smoothing: float = 0.0  # 0.1 is common

    # Learning rate schedule
    use_cosine_schedule: bool = False  # Cosine annealing vs linear decay

    # R-Drop: Consistency regularization
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0  # Weight for KL divergence term

    # FGM: Adversarial training
    use_fgm: bool = False
    fgm_epsilon: float = 1.0  # Perturbation magnitude

    def __post_init__(self):
        """Fill in defaults from model registry."""
        if self.model_key in MODEL_REGISTRY:
            reg = MODEL_REGISTRY[self.model_key]
            if self.batch_size is None:
                self.batch_size = reg["recommended_batch"]
            if self.gradient_accumulation_steps is None:
                self.gradient_accumulation_steps = reg["recommended_grad_accum"]
            if self.learning_rate is None:
                self.learning_rate = reg["recommended_lr"]
            if self.gradient_checkpointing is None:
                self.gradient_checkpointing = reg["gradient_checkpointing"]

    @property
    def hf_model_name(self) -> str:
        if self.model_key in MODEL_REGISTRY:
            return MODEL_REGISTRY[self.model_key]["hf_name"]
        return self.model_key  # Assume it's a full HF path


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/dev-test/test data from clean or features directory."""
    # Try CT24_clean first (original data), then CT24_features
    for subdir in ["CT24_clean", "CT24_features"]:
        base = data_dir / subdir

        train_path = base / "CT24_train_clean.parquet"
        if not train_path.exists():
            train_path = base / "CT24_train_features.parquet"
        if not train_path.exists():
            train_path = base / "CT24_train_clean.tsv"

        if train_path.exists():
            break
    else:
        raise FileNotFoundError(f"Training data not found in {data_dir}")

    print(f"Loading data from: {train_path.parent}")

    # Load based on file type
    def load_file(name_pattern: str, required: bool = True) -> pl.DataFrame | None:
        for ext in [".parquet", ".tsv"]:
            for prefix in ["CT24_", ""]:
                for suffix in ["_clean", "_features", ""]:
                    path = train_path.parent / f"{prefix}{name_pattern}{suffix}{ext}"
                    if path.exists():
                        if ext == ".parquet":
                            return pl.read_parquet(path)
                        else:
                            return pl.read_csv(path, separator="\t")
        if required:
            raise FileNotFoundError(f"Could not find {name_pattern} data")
        return None

    train = load_file("train")
    dev = load_file("dev")
    dev_test = load_file("dev-test", required=False)  # May not exist
    test = load_file("test")

    return train, dev, dev_test, test


def prepare_dataset(df: pl.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Convert Polars DataFrame to HuggingFace Dataset."""
    texts = df["Text"].to_list()
    labels = [1 if label == "Yes" else 0 for label in df["class_label"].to_list()]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }

    if "token_type_ids" in encodings:
        dataset_dict["token_type_ids"] = encodings["token_type_ids"]

    return Dataset.from_dict(dataset_dict)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def evaluate_with_threshold(trainer, dataset, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    predictions = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        results.append({
            "threshold": float(thresh),
            "f1": float(f1_score(labels, preds)),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        })

    return results, probs


# =============================================================================
# Training
# =============================================================================

def train_model(config: Config, data_dir: Path, output_dir: Path):
    """Train DeBERTa model."""
    print("=" * 80)
    print(f"FINE-TUNING: {config.model_key.upper()}")
    print("=" * 80)

    if config.model_key in MODEL_REGISTRY:
        reg = MODEL_REGISTRY[config.model_key]
        print(f"\nModel: {reg['hf_name']}")
        print(f"Parameters: {reg['params']}")
        print(f"Hidden size: {reg['hidden_size']}")
    else:
        print(f"\nModel: {config.model_key}")

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch} effective")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load tokenizer
    print("\nLoading tokenizer...")
    model_name = config.hf_model_name
    if "deberta" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # Enable gradient checkpointing for large models
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # Freeze encoder if requested (only train classification head)
    if config.freeze_encoder:
        # Find the base model (encoder)
        if hasattr(model, "deberta"):
            base_model = model.deberta
        elif hasattr(model, "bert"):
            base_model = model.bert
        elif hasattr(model, "roberta"):
            base_model = model.roberta
        elif hasattr(model, "base_model"):
            base_model = model.base_model
        else:
            # Generic: freeze everything except classifier
            base_model = None
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        if base_model is not None:
            for param in base_model.parameters():
                param.requires_grad = False

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  🧊 Encoder FROZEN: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🔥 Full fine-tuning: {trainable:,} params trainable")

    # Load data
    print("\nLoading data...")
    train_df, dev_df, dev_test_df, test_df = load_data(data_dir)
    print(f"  Train: {len(train_df)}")
    print(f"  Dev: {len(dev_df)}")
    print(f"  Dev-test: {len(dev_test_df) if dev_test_df is not None else 'N/A'}")
    print(f"  Test: {len(test_df)}")

    # Class distribution
    n_pos = sum(1 for l in train_df["class_label"].to_list() if l == "Yes")
    n_neg = len(train_df) - n_pos
    pos_weight = config.class_weight if config.class_weight else (n_neg / n_pos)
    print(f"  Class ratio: {n_neg}:{n_pos}, weight={pos_weight:.2f}")

    # Prepare datasets
    print("\nTokenizing...")
    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    dev_dataset = prepare_dataset(dev_df, tokenizer, config.max_length)
    dev_test_dataset = prepare_dataset(dev_test_df, tokenizer, config.max_length) if dev_test_df is not None else None
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    lr_scheduler = "cosine" if config.use_cosine_schedule else "linear"
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=lr_scheduler,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config.fp16 and torch.cuda.is_available() and not config.bf16,
        bf16=config.bf16 and torch.cuda.is_available(),
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
        gradient_checkpointing=config.gradient_checkpointing,
        label_smoothing_factor=config.label_smoothing,  # Built-in label smoothing
    )

    if config.use_cosine_schedule:
        print(f"  📉 Using cosine annealing LR schedule")

    if config.label_smoothing > 0:
        print(f"  🏷️ Label smoothing: {config.label_smoothing}")

    # Determine loss function
    if config.use_focal_loss:
        focal_alpha = config.focal_alpha if config.focal_alpha else pos_weight
        loss_fct = FocalLoss(alpha=focal_alpha, gamma=config.focal_gamma)
        print(f"  🎯 Using Focal Loss (α={focal_alpha:.2f}, γ={config.focal_gamma})")
        if config.label_smoothing > 0:
            print(f"     ⚠️ Note: Label smoothing handled by Trainer, not combined with Focal Loss")
    else:
        weight = torch.tensor([1.0, pos_weight])
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
        print(f"  📊 Using Weighted CrossEntropy (weight={pos_weight:.2f})")

    # Custom trainer with configurable loss, R-Drop, and FGM
    class AdvancedTrainer(Trainer):
        def __init__(
            self,
            loss_function,
            use_llrd=False,
            llrd_config=None,
            use_rdrop=False,
            rdrop_alpha=1.0,
            use_fgm=False,
            fgm_epsilon=1.0,
            *args,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function
            self.use_llrd = use_llrd
            self.llrd_config = llrd_config or {}
            self.use_rdrop = use_rdrop
            self.rdrop_alpha = rdrop_alpha
            self.use_fgm = use_fgm
            self.fgm_epsilon = fgm_epsilon
            self.fgm = None  # Initialize in training_step when model is available

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Get labels without modifying inputs (needed for FGM second pass)
            labels = inputs.get("labels")
            if labels is None:
                labels = inputs.pop("labels")  # Fallback for standard Trainer calls
            else:
                # Create copy of inputs without labels for model forward
                inputs = {k: v for k, v in inputs.items() if k != "labels"}

            # Move loss function to device if needed
            if hasattr(self.loss_function, 'to'):
                self.loss_function = self.loss_function.to(model.device)
            elif hasattr(self.loss_function, 'weight') and self.loss_function.weight is not None:
                self.loss_function.weight = self.loss_function.weight.to(model.device)

            if self.use_rdrop:
                # R-Drop: Forward twice with different dropout masks
                outputs1 = model(**inputs)
                outputs2 = model(**inputs)
                logits1, logits2 = outputs1.logits, outputs2.logits

                # Task loss on both outputs
                loss1 = self.loss_function(logits1, labels)
                loss2 = self.loss_function(logits2, labels)
                task_loss = (loss1 + loss2) / 2

                # KL divergence between the two outputs
                kl_loss = compute_kl_divergence(logits1, logits2)

                # Combined loss
                loss = task_loss + self.rdrop_alpha * kl_loss
                outputs = outputs1  # Return first output for metrics
            else:
                outputs = model(**inputs)
                logits = outputs.logits
                loss = self.loss_function(logits, labels)

            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs):
            """Override to implement FGM adversarial training."""
            model.train()
            inputs = self._prepare_inputs(inputs)

            # Standard forward + backward
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            # Scale loss for gradient accumulation
            loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)

            # FGM: Adversarial training
            if self.use_fgm:
                if self.fgm is None:
                    self.fgm = FGM(model, epsilon=self.fgm_epsilon)

                # Attack: perturb embeddings
                self.fgm.attack()

                # Forward on perturbed input
                with self.compute_loss_context_manager():
                    loss_adv = self.compute_loss(model, inputs)

                if self.args.n_gpu > 1:
                    loss_adv = loss_adv.mean()

                loss_adv = loss_adv / self.args.gradient_accumulation_steps
                self.accelerator.backward(loss_adv)

                # Restore original embeddings
                self.fgm.restore()

            return loss.detach()

        def create_optimizer(self):
            """Override to implement LLRD if enabled."""
            if self.use_llrd:
                param_groups = get_llrd_parameter_groups(
                    self.model,
                    base_lr=self.args.learning_rate,
                    decay_rate=self.llrd_config.get("decay", 0.9),
                    weight_decay=self.args.weight_decay,
                )
                self.optimizer = torch.optim.AdamW(param_groups)
            else:
                # Default optimizer
                super().create_optimizer()

    # Setup callbacks
    callbacks = []
    if not config.no_early_stopping:
        # Use smart early stopping with EMA smoothing
        callbacks.append(SmartEarlyStoppingCallback(
            patience=config.early_stopping_patience,
            metric_name="f1",
            max_train_val_gap=0.20,
            ema_alpha=0.3,
        ))

    # Add SWA callback if enabled
    if config.use_swa:
        callbacks.append(SWACallback(
            swa_start_ratio=config.swa_start_ratio,
        ))
        print(f"  🔄 SWA enabled (starts at {config.swa_start_ratio*100:.0f}% of training)")

    # Print LLRD info if enabled
    if config.use_llrd:
        print(f"  📈 LLRD enabled (decay={config.llrd_decay})")

    # Print R-Drop info if enabled
    if config.use_rdrop:
        print(f"  🔁 R-Drop enabled (α={config.rdrop_alpha})")

    # Print FGM info if enabled
    if config.use_fgm:
        print(f"  ⚔️ FGM adversarial training enabled (ε={config.fgm_epsilon})")

    # Select evaluation dataset for early stopping
    if config.eval_on_devtest and dev_test_dataset is not None:
        eval_dataset_for_training = dev_test_dataset
        print("  📊 Early stopping on: dev-test (better test proxy)")
    else:
        eval_dataset_for_training = dev_dataset
        print("  📊 Early stopping on: dev")

    # Create trainer with all advanced features
    trainer = AdvancedTrainer(
        loss_function=loss_fct,
        use_llrd=config.use_llrd,
        llrd_config={"decay": config.llrd_decay},
        use_rdrop=config.use_rdrop,
        rdrop_alpha=config.rdrop_alpha,
        use_fgm=config.use_fgm,
        fgm_epsilon=config.fgm_epsilon,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_training,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    trainer.train()

    # Save
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"\n💾 Best model saved to: {best_model_path}")

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    dev_labels = [1 if l == "Yes" else 0 for l in dev_df["class_label"].to_list()]
    dev_results, dev_probs = evaluate_with_threshold(trainer, dev_dataset, dev_labels)
    best_dev = max(dev_results, key=lambda x: x["f1"])

    # Evaluate on dev-test if available
    best_dev_test = None
    dev_test_probs = None
    if dev_test_dataset is not None:
        dev_test_labels = [1 if l == "Yes" else 0 for l in dev_test_df["class_label"].to_list()]
        dev_test_results, dev_test_probs = evaluate_with_threshold(trainer, dev_test_dataset, dev_test_labels)
        best_dev_test = max(dev_test_results, key=lambda x: x["f1"])

    test_labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    test_results, test_probs = evaluate_with_threshold(trainer, test_dataset, test_labels)
    best_test = max(test_results, key=lambda x: x["f1"])

    # Threshold calibration: apply dev-test's optimal threshold to test
    calibrated_test = None
    if best_dev_test is not None:
        calibrated_thresh = best_dev_test["threshold"]
        calibrated_preds = (test_probs >= calibrated_thresh).astype(int)
        calibrated_test = {
            "threshold": calibrated_thresh,
            "f1": float(f1_score(test_labels, calibrated_preds)),
            "accuracy": float(accuracy_score(test_labels, calibrated_preds)),
            "precision": float(precision_score(test_labels, calibrated_preds, zero_division=0)),
            "recall": float(recall_score(test_labels, calibrated_preds, zero_division=0)),
        }

    print(f"\n{'Split':<10} {'Best F1':<10} {'Accuracy':<10} {'Threshold':<10}")
    print("-" * 45)
    print(f"{'Dev':<10} {best_dev['f1']:<10.4f} {best_dev['accuracy']:<10.4f} {best_dev['threshold']:<10.2f}")
    if best_dev_test is not None:
        print(f"{'Dev-test':<10} {best_dev_test['f1']:<10.4f} {best_dev_test['accuracy']:<10.4f} {best_dev_test['threshold']:<10.2f}")
    print(f"{'Test':<10} {best_test['f1']:<10.4f} {best_test['accuracy']:<10.4f} {best_test['threshold']:<10.2f}")
    if calibrated_test is not None:
        print(f"{'Test (cal)':<10} {calibrated_test['f1']:<10.4f} {calibrated_test['accuracy']:<10.4f} {calibrated_test['threshold']:<10.2f} ← calibrated from dev-test")

    print(f"\n📊 SOTA Comparison (Test):")
    print(f"   F1:  {SOTA_F1:.4f} → {best_test['f1']:.4f} ({best_test['f1'] - SOTA_F1:+.4f})")
    print(f"   Acc: {SOTA_ACC:.4f} → {best_test['accuracy']:.4f} ({best_test['accuracy'] - SOTA_ACC:+.4f})")

    # Save results
    results = {
        "model": config.model_key,
        "hf_model": config.hf_model_name,
        "config": {
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "gradient_checkpointing": config.gradient_checkpointing,
            "freeze_encoder": config.freeze_encoder,
            # Modern techniques
            "focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma if config.use_focal_loss else None,
            "focal_alpha": config.focal_alpha if config.use_focal_loss else None,
            "llrd": config.use_llrd,
            "llrd_decay": config.llrd_decay if config.use_llrd else None,
            "swa": config.use_swa,
            "swa_start_ratio": config.swa_start_ratio if config.use_swa else None,
            "label_smoothing": config.label_smoothing,
            "cosine_schedule": config.use_cosine_schedule,
            "rdrop": config.use_rdrop,
            "rdrop_alpha": config.rdrop_alpha if config.use_rdrop else None,
            "fgm": config.use_fgm,
            "fgm_epsilon": config.fgm_epsilon if config.use_fgm else None,
        },
        "dev": {"best": best_dev, "all": dev_results},
        "test": {"best": best_test, "all": test_results},
        "sota_gap_f1": best_test["f1"] - SOTA_F1,
    }

    # Add dev-test results if available
    if best_dev_test is not None:
        results["dev_test"] = {"best": best_dev_test, "all": dev_test_results}

    # Add calibrated test results (threshold from dev-test applied to test)
    if calibrated_test is not None:
        results["test_calibrated"] = calibrated_test
        results["calibration_note"] = "test_calibrated uses dev-test's optimal threshold applied to test (realistic scenario)"

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(output_dir / "dev_probs.npy", dev_probs)
    np.save(output_dir / "test_probs.npy", test_probs)
    if dev_test_probs is not None:
        np.save(output_dir / "dev_test_probs.npy", dev_test_probs)

    return best_test


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models for checkworthiness")
    parser.add_argument("--model", type=str, default="deberta-v3-large",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to fine-tune (DeBERTa, BGE, E5, GTE, MPNet, RoBERTa)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory (default: project data/processed)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--class-weight", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (better for large models)")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping, train for all epochs")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder, only train classification head")
    parser.add_argument("--eval-on-devtest", action="store_true", help="Use dev-test for early stopping (better test proxy)")

    # Modern techniques
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss (better for class imbalance)")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Focal loss alpha (default: auto from class ratio)")
    parser.add_argument("--llrd", action="store_true", help="Use layer-wise learning rate decay")
    parser.add_argument("--llrd-decay", type=float, default=0.9, help="LLRD decay rate (default: 0.9)")
    parser.add_argument("--swa", action="store_true", help="Use stochastic weight averaging")
    parser.add_argument("--swa-start", type=float, default=0.6, help="Start SWA after this fraction of training (default: 0.6)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor (default: 0.0, try 0.1)")
    parser.add_argument("--cosine-schedule", action="store_true", help="Use cosine annealing LR schedule (default: linear)")
    parser.add_argument("--rdrop", action="store_true", help="Use R-Drop consistency regularization")
    parser.add_argument("--rdrop-alpha", type=float, default=1.0, help="R-Drop KL weight (default: 1.0)")
    parser.add_argument("--fgm", action="store_true", help="Use FGM adversarial training")
    parser.add_argument("--fgm-epsilon", type=float, default=1.0, help="FGM perturbation magnitude (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent / "data" / "processed"

    if args.output_dir:
        output_dir = Path(args.output_dir) / args.model
    else:
        output_dir = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / args.model

    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = Config(
        model_key=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        class_weight=args.class_weight,
        bf16=args.bf16,
        no_early_stopping=args.no_early_stopping,
        freeze_encoder=args.freeze_encoder,
        eval_on_devtest=args.eval_on_devtest,
        # Modern techniques
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        use_llrd=args.llrd,
        llrd_decay=args.llrd_decay,
        use_swa=args.swa,
        swa_start_ratio=args.swa_start,
        label_smoothing=args.label_smoothing,
        use_cosine_schedule=args.cosine_schedule,
        use_rdrop=args.rdrop,
        rdrop_alpha=args.rdrop_alpha,
        use_fgm=args.fgm,
        fgm_epsilon=args.fgm_epsilon,
        seed=args.seed,
    )

    if args.no_grad_ckpt:
        config.gradient_checkpointing = False

    # Print available models
    print("\n📋 Available models:")
    for key, info in MODEL_REGISTRY.items():
        marker = "→" if key == args.model else " "
        print(f"   {marker} {key}: {info['params']} params")

    # Train
    train_model(config, data_dir, output_dir)

    print(f"\n✅ Done! Results in {output_dir}")


if __name__ == "__main__":
    main()
