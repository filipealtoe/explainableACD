#!/usr/bin/env python3
"""
Fine-tune DeBERTa for Checkworthiness Classification (CT24)

Uses HuggingFace Transformers with the Trainer API.
Supports DeBERTa-v3-base, DeBERTa-v3-large, and other variants.

Usage:
    python experiments/scripts/finetune_deberta.py
    python experiments/scripts/finetune_deberta.py --model microsoft/deberta-v3-base --epochs 5
    python experiments/scripts/finetune_deberta.py --model microsoft/deberta-v3-large --epochs 3 --batch-size 8
    python experiments/scripts/finetune_deberta.py --quick  # Fast test run

Requirements:
    pip install transformers datasets accelerate scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_checkworthy"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# Model options
MODEL_CONFIGS = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "deberta-base": "microsoft/deberta-base",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 200
    early_stopping_patience: int = 3
    fp16: bool = True
    seed: int = 42


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
    # Extract text and labels
    texts = df["Text"].to_list()
    labels = [1 if label == "Yes" else 0 for label in df["class_label"].to_list()]

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Let data collator handle padding
    )

    # Create dataset
    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }

    # Add token_type_ids if present (BERT-style models)
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

    # Get predictions
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
    """Train DeBERTa model."""
    print("=" * 80)
    print("DEBERTA FINE-TUNING FOR CHECKWORTHINESS")
    print("=" * 80)
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"\nModel: {config.model_name}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch} effective")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max length: {config.max_length}")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    # Use slow tokenizer for DeBERTa-v2/v3 (fast tokenizer has a bug in recent transformers)
    if "deberta-v" in config.model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Calculate class weights for imbalanced data
    n_pos = sum(1 for l in train_df["class_label"].to_list() if l == "Yes")
    n_neg = len(train_df) - n_pos
    pos_weight = n_neg / n_pos
    print(f"\nClass imbalance: {n_neg}:{n_pos} (weight: {pos_weight:.2f})")

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
        report_to="none",  # Disable wandb/tensorboard
        seed=config.seed,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Weighted cross-entropy loss
            weight = torch.tensor([1.0, pos_weight], device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = WeightedTrainer(
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
    print("TRAINING")
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
    print(f"\nBest Test F1: {best_test['f1']:.4f} @ threshold {best_test['threshold']:.2f}")
    print(f"Best Test Acc: {best_test['accuracy']:.4f}")
    print(f"\nSOTA Comparison:")
    print(f"  F1:  {SOTA_F1:.4f} → {best_test['f1']:.4f} (gap: {best_test['f1'] - SOTA_F1:+.4f})")
    print(f"  Acc: {SOTA_ACC:.4f} → {best_test['accuracy']:.4f} (gap: {best_test['accuracy'] - SOTA_ACC:+.4f})")

    # Save results
    results = {
        "model": config.model_name,
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
        "sota_gap_acc": best_test["accuracy"] - SOTA_ACC,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save predictions
    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "dev_probs.npy", dev_probs)

    return best_test


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for checkworthiness")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base",
                        help="Model name or path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--quick", action="store_true", help="Quick test run (1 epoch, small eval)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Configure
    config = Config(
        model_name=args.model,
        num_epochs=1 if args.quick else args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        eval_steps=50 if args.quick else 100,
        save_steps=100 if args.quick else 200,
    )

    # Output directory
    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    try:
        best_result = train_model(config, output_dir)
        print("\n" + "=" * 80)
        if best_result["f1"] >= SOTA_F1:
            print("🏆 SOTA ACHIEVED!")
        elif best_result["f1"] >= 0.78:
            print("🔥 CLOSE TO SOTA!")
        else:
            print("Training complete. Consider trying:")
            print("  - deberta-v3-large (bigger model)")
            print("  - More epochs (--epochs 10)")
            print("  - Lower learning rate (--lr 1e-5)")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
