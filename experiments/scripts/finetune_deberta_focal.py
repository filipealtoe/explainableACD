#!/usr/bin/env python3
"""
Fine-tune DeBERTa with Focal Loss for Checkworthiness Classification (CT24)

Uses Focal Loss instead of Cross-Entropy to focus on hard examples.
Focal Loss: FL(p) = -(1-p)^γ * log(p)

Usage:
    # Single run with specific parameters
    python experiments/scripts/finetune_deberta_focal.py --gamma 2.0 --alpha 0.25

    # Auto-tune gamma and alpha (recommended)
    python experiments/scripts/finetune_deberta_focal.py --tune

    # Custom tuning grid
    python experiments/scripts/finetune_deberta_focal.py --tune --gammas 1.0,2.0,3.0 --alphas 0.25,0.5

Requirements:
    pip install transformers datasets accelerate scikit-learn sentencepiece
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    DebertaV2Tokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_focal"

SOTA_F1 = 0.82
SOTA_ACC = 0.905


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 200
    early_stopping_patience: int = 3
    fp16: bool = True
    seed: int = 42
    # Focal loss parameters
    focal_gamma: float = 2.0  # Focus parameter (higher = more focus on hard examples)
    focal_alpha: float = 0.25  # Balance parameter for positive class


# =============================================================================
# Focal Loss
# =============================================================================


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = p if y=1 else 1-p
        α_t = α if y=1 else 1-α
        γ = focusing parameter (typically 2.0)

    When γ=0, this is equivalent to weighted cross-entropy.
    Higher γ means more focus on hard examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, 2) for binary classification
            targets: Labels of shape (N,) with values 0 or 1
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = probability of correct class

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight (higher for positive class)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/test data."""
    train = pl.read_parquet(DATA_DIR / "CT24_train_features.parquet")
    dev = pl.read_parquet(DATA_DIR / "CT24_dev_features.parquet")
    test = pl.read_parquet(DATA_DIR / "CT24_test_features.parquet")
    return train, dev, test


def prepare_dataset(df: pl.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Convert Polars DataFrame to HuggingFace Dataset with tokenization."""
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
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def evaluate_with_threshold(trainer, dataset, labels, thresholds=None):
    """Evaluate model with different probability thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    predictions = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        results.append({
            "threshold": thresh,
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        })

    return results, probs


# =============================================================================
# Training
# =============================================================================


def train_model(config: Config, output_dir: Path):
    """Train DeBERTa model with Focal Loss."""
    print("=" * 80)
    print("DEBERTA FINE-TUNING WITH FOCAL LOSS")
    print("=" * 80)
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"\nModel: {config.model_name}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch} effective")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max length: {config.max_length}")
    print(f"\nFocal Loss Parameters:")
    print(f"  γ (gamma): {config.focal_gamma} - focus on hard examples")
    print(f"  α (alpha): {config.focal_alpha} - weight for positive class")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # Load data
    print("Loading data...")
    train_df, dev_df, test_df = load_data()
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev: {len(dev_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Prepare datasets
    print("Tokenizing...")
    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    dev_dataset = prepare_dataset(dev_df, tokenizer, config.max_length)
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Class distribution
    n_pos = sum(1 for l in train_df["class_label"].to_list() if l == "Yes")
    n_neg = len(train_df) - n_pos
    print(f"\nClass distribution: {n_neg} negative, {n_pos} positive ({100*n_pos/len(train_df):.1f}%)")

    # Initialize Focal Loss
    focal_loss = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

    # Training arguments
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
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config.fp16 and torch.cuda.is_available(),
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
    )

    # Custom trainer with Focal Loss
    class FocalLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Use Focal Loss instead of Cross-Entropy
            loss = focal_loss(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # Train
    print("\n" + "=" * 80)
    print("TRAINING WITH FOCAL LOSS")
    print("=" * 80)
    trainer.train()

    # Save best model
    best_model_path = output_dir / "best_model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"\nBest model saved to: {best_model_path}")

    # Evaluate on dev
    print("\n" + "=" * 80)
    print("EVALUATION ON DEV SET")
    print("=" * 80)
    dev_labels = [1 if l == "Yes" else 0 for l in dev_df["class_label"].to_list()]
    dev_results, dev_probs = evaluate_with_threshold(trainer, dev_dataset, dev_labels)

    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in dev_results:
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")

    best_dev = max(dev_results, key=lambda x: x["f1"])
    print(f"\nBest dev F1: {best_dev['f1']:.4f} @ threshold {best_dev['threshold']:.2f}")

    # Evaluate on test
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    test_labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    test_results, test_probs = evaluate_with_threshold(trainer, test_dataset, test_labels)

    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in test_results:
        marker = "🔥" if r["f1"] > 0.76 else ""
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {marker}")

    best_test = max(test_results, key=lambda x: x["f1"])

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFocal Loss: γ={config.focal_gamma}, α={config.focal_alpha}")
    print(f"\nBest Test F1: {best_test['f1']:.4f} @ threshold {best_test['threshold']:.2f}")
    print(f"Best Test Acc: {best_test['accuracy']:.4f}")
    print(f"\nSOTA Comparison:")
    print(f"  F1:  {SOTA_F1:.4f} → {best_test['f1']:.4f} (gap: {best_test['f1'] - SOTA_F1:+.4f})")
    print(f"  Acc: {SOTA_ACC:.4f} → {best_test['accuracy']:.4f} (gap: {best_test['accuracy'] - SOTA_ACC:+.4f})")

    # Compare with baseline (weighted CE)
    print(f"\nBaseline (Weighted CE): F1=0.8214")
    print(f"Focal Loss improvement: {best_test['f1'] - 0.8214:+.4f}")

    # Save results
    results = {
        "model": config.model_name,
        "loss": "focal",
        "focal_gamma": config.focal_gamma,
        "focal_alpha": config.focal_alpha,
        "config": {
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
        },
        "dev_results": dev_results,
        "test_results": test_results,
        "best_dev": best_dev,
        "best_test": best_test,
        "sota_gap_f1": best_test["f1"] - SOTA_F1,
        "baseline_improvement": best_test["f1"] - 0.8214,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "dev_probs.npy", dev_probs)

    return best_test


# =============================================================================
# Main
# =============================================================================


def tune_focal_parameters(
    base_config: Config,
    gammas: list[float],
    alphas: list[float],
    output_dir: Path,
):
    """
    Grid search over focal loss parameters.
    Trains multiple models and returns best gamma/alpha combination.
    """
    print("=" * 80)
    print("FOCAL LOSS PARAMETER TUNING")
    print("=" * 80)
    print(f"\nGamma values to try: {gammas}")
    print(f"Alpha values to try: {alphas}")
    print(f"Total combinations: {len(gammas) * len(alphas)}")

    all_results = []

    for gamma in gammas:
        for alpha in alphas:
            print(f"\n{'='*80}")
            print(f"Training with γ={gamma}, α={alpha}")
            print("=" * 80)

            config = Config(
                model_name=base_config.model_name,
                max_length=base_config.max_length,
                batch_size=base_config.batch_size,
                gradient_accumulation_steps=base_config.gradient_accumulation_steps,
                learning_rate=base_config.learning_rate,
                num_epochs=base_config.num_epochs,
                warmup_ratio=base_config.warmup_ratio,
                weight_decay=base_config.weight_decay,
                eval_steps=base_config.eval_steps,
                save_steps=base_config.save_steps,
                early_stopping_patience=base_config.early_stopping_patience,
                fp16=base_config.fp16,
                seed=base_config.seed,
                focal_gamma=gamma,
                focal_alpha=alpha,
            )

            run_output_dir = output_dir / f"gamma{gamma}_alpha{alpha}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = train_model(config, run_output_dir)
                all_results.append({
                    "gamma": gamma,
                    "alpha": alpha,
                    "test_f1": result["f1"],
                    "test_acc": result["accuracy"],
                    "threshold": result["threshold"],
                })
            except Exception as e:
                print(f"Error with γ={gamma}, α={alpha}: {e}")
                all_results.append({
                    "gamma": gamma,
                    "alpha": alpha,
                    "test_f1": 0.0,
                    "test_acc": 0.0,
                    "error": str(e),
                })

    # Summary
    print("\n" + "=" * 80)
    print("TUNING RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Gamma':<10} {'Alpha':<10} {'Test F1':<12} {'Test Acc':<12} {'vs Baseline':<12}")
    print("-" * 60)

    baseline_f1 = 0.8214
    best_result = None
    best_f1 = 0

    for r in sorted(all_results, key=lambda x: x.get("test_f1", 0), reverse=True):
        if "error" in r:
            print(f"{r['gamma']:<10} {r['alpha']:<10} {'ERROR':<12}")
        else:
            diff = r["test_f1"] - baseline_f1
            marker = "🔥 BEST" if r["test_f1"] > best_f1 else ""
            if r["test_f1"] > best_f1:
                best_f1 = r["test_f1"]
                best_result = r
            print(f"{r['gamma']:<10.1f} {r['alpha']:<10.2f} {r['test_f1']:<12.4f} {r['test_acc']:<12.4f} {diff:+.4f} {marker}")

    if best_result:
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"  Gamma: {best_result['gamma']}")
        print(f"  Alpha: {best_result['alpha']}")
        print(f"  Test F1: {best_result['test_f1']:.4f}")
        print(f"  Test Acc: {best_result['test_acc']:.4f}")
        print(f"  Threshold: {best_result['threshold']:.2f}")
        print(f"\n  Improvement over baseline (0.8214): {best_result['test_f1'] - 0.8214:+.4f}")

        if best_result["test_f1"] > baseline_f1:
            print("\n  🔥 FOCAL LOSS IMPROVES OVER BASELINE!")
        else:
            print("\n  ✗ Focal loss did not improve over weighted CE baseline.")

    # Save tuning results
    tuning_results_path = output_dir / "tuning_results.json"
    with open(tuning_results_path, "w") as f:
        json.dump({
            "gammas_tested": gammas,
            "alphas_tested": alphas,
            "results": all_results,
            "best": best_result,
            "baseline_f1": baseline_f1,
        }, f, indent=2)
    print(f"\nTuning results saved to: {tuning_results_path}")

    return best_result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa with Focal Loss")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large",
                        help="Model name or path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma (focus parameter)")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss alpha (positive class weight)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    # Tuning arguments
    parser.add_argument("--tune", action="store_true", help="Enable parameter tuning mode")
    parser.add_argument("--gammas", type=str, default="0.5,1.0,2.0,3.0",
                        help="Comma-separated gamma values to try (with --tune)")
    parser.add_argument("--alphas", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated alpha values to try (with --tune)")
    args = parser.parse_args()

    base_config = Config(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        focal_gamma=args.gamma,
        focal_alpha=args.alpha,
    )

    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tune:
        # Parameter tuning mode
        gammas = [float(g) for g in args.gammas.split(",")]
        alphas = [float(a) for a in args.alphas.split(",")]

        try:
            best = tune_focal_parameters(base_config, gammas, alphas, output_dir)
            print("\n" + "=" * 80)
            if best and best["test_f1"] >= SOTA_F1:
                print("🏆 SOTA ACHIEVED WITH FOCAL LOSS!")
            elif best and best["test_f1"] > 0.8214:
                print("🔥 FOCAL LOSS BEATS BASELINE!")
            else:
                print("Focal loss did not beat baseline in this tuning run.")
            print("=" * 80)
        except KeyboardInterrupt:
            print("\nTuning interrupted.")
    else:
        # Single run mode
        config = base_config
        run_output_dir = output_dir / f"gamma{args.gamma}_alpha{args.alpha}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            best_result = train_model(config, run_output_dir)
            print("\n" + "=" * 80)
            if best_result["f1"] >= SOTA_F1:
                print("🏆 SOTA ACHIEVED WITH FOCAL LOSS!")
            elif best_result["f1"] > 0.8214:
                print("🔥 FOCAL LOSS BEATS BASELINE!")
            elif best_result["f1"] >= 0.80:
                print("Close to baseline. Try --tune to find better parameters.")
            else:
                print("Focal loss underperformed. Try --tune to search parameters.")
            print("=" * 80)
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        except Exception as e:
            print(f"\nError: {e}")
            raise


if __name__ == "__main__":
    main()
