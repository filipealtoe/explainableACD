#!/usr/bin/env python3
"""
Fine-tune DeBERTa on AUGMENTED training data for Checkworthiness Classification.

Uses the augmented dataset (original + hard synthetic samples).
Evaluates on dev, dev-test, and test sets.

Usage:
    python experiments/scripts/finetune_deberta_augmented.py
    python experiments/scripts/finetune_deberta_augmented.py --model microsoft/deberta-v3-large --epochs 3 --batch-size 8
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

# Use augmented training data, cleaned dev, but RAW dev-test/test (never modify final eval!)
AUGMENTED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_augmented"
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "deberta_augmented"

# SOTA benchmarks
SOTA = {
    "dev-test": {"f1": 0.932, "acc": 0.955},
    "test": {"f1": 0.82, "acc": 0.905},
}


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 200
    early_stopping_patience: int = 3
    disable_early_stopping: bool = False
    fp16: bool = True
    seed: int = 42


# =============================================================================
# Data Loading
# =============================================================================


def load_data():
    """Load augmented train, cleaned dev, and RAW dev-test/test (never modify final eval!)."""
    # Load augmented training data
    train_path = AUGMENTED_DATA_DIR / "CT24_train_augmented.parquet"
    if train_path.exists():
        train = pl.read_parquet(train_path)
    else:
        train_tsv = AUGMENTED_DATA_DIR / "CT24_train_augmented.tsv"
        if train_tsv.exists():
            train = pl.read_csv(train_tsv, separator="\t")
        else:
            raise FileNotFoundError(f"Augmented training data not found. Run create_augmented_dataset.py first.")

    # Load CLEANED dev (used for validation during training)
    dev_path = CLEAN_DATA_DIR / "CT24_dev_clean.parquet"
    if dev_path.exists():
        dev = pl.read_parquet(dev_path)
    else:
        dev = pl.read_csv(RAW_DATA_DIR / "CT24_checkworthy_english_dev.tsv", separator="\t")

    # Load RAW dev-test and test (never modify these!)
    devtest = pl.read_csv(RAW_DATA_DIR / "CT24_checkworthy_english_dev-test.tsv", separator="\t")

    # Test set - use gold labels file
    test_path = RAW_DATA_DIR / "CT24_checkworthy_english_test_gold.tsv"
    if test_path.exists():
        test = pl.read_csv(test_path, separator="\t")
    else:
        test = pl.read_csv(RAW_DATA_DIR / "CT24_checkworthy_english_test.tsv", separator="\t")

    return train, dev, devtest, test


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
        results.append({
            "threshold": thresh,
            "f1": f1_score(labels, preds),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
        })

    return results, probs


# =============================================================================
# Training
# =============================================================================


def train_model(config: Config, output_dir: Path):
    print("=" * 80)
    print("DEBERTA FINE-TUNING WITH AUGMENTED DATA")
    print("=" * 80)
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"\nModel: {config.model_name}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch} effective")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Early stopping: {'DISABLED' if config.disable_early_stopping else f'patience={config.early_stopping_patience}'}")

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
    print("\nLoading data...")
    train_df, dev_df, devtest_df, test_df = load_data()

    # Show training data stats
    if "source" in train_df.columns:
        n_original = train_df.filter(pl.col("source") == "original").height
        n_synthetic = train_df.filter(pl.col("source") == "synthetic").height
        print(f"  Train: {len(train_df)} samples (original: {n_original}, synthetic: {n_synthetic})")
    else:
        print(f"  Train: {len(train_df)} samples")

    print(f"  Dev: {len(dev_df)} samples")
    if devtest_df is not None:
        print(f"  Dev-test: {len(devtest_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Prepare datasets
    print("\nTokenizing...")
    train_dataset = prepare_dataset(train_df, tokenizer, config.max_length)
    dev_dataset = prepare_dataset(dev_df, tokenizer, config.max_length)
    devtest_dataset = prepare_dataset(devtest_df, tokenizer, config.max_length) if devtest_df is not None else None
    test_dataset = prepare_dataset(test_df, tokenizer, config.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Class weights
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
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
    )

    # Weighted trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = torch.tensor([1.0, pos_weight], device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[] if config.disable_early_stopping else [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
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

    # Results storage
    all_results = {}

    # Evaluate on DEV
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
    print(f"\n✨ Best dev F1: {best_dev['f1']:.4f} @ threshold {best_dev['threshold']:.2f}")
    all_results["dev"] = {"results": dev_results, "best": best_dev}

    # Evaluate on DEV-TEST
    if devtest_df is not None:
        print("\n" + "=" * 80)
        print("EVALUATION ON DEV-TEST SET")
        print("=" * 80)
        devtest_labels = [1 if l == "Yes" else 0 for l in devtest_df["class_label"].to_list()]
        devtest_results, devtest_probs = evaluate_with_threshold(trainer, devtest_dataset, devtest_labels)

        print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
        print("-" * 55)
        for r in devtest_results:
            marker = "🔥" if r["f1"] >= SOTA["dev-test"]["f1"] else ""
            print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {marker}")

        best_devtest = max(devtest_results, key=lambda x: x["f1"])
        print(f"\n✨ Best dev-test F1: {best_devtest['f1']:.4f} @ threshold {best_devtest['threshold']:.2f}")
        print(f"   SOTA dev-test: F1={SOTA['dev-test']['f1']:.4f}, Acc={SOTA['dev-test']['acc']:.4f}")
        print(f"   Gap: F1 {best_devtest['f1'] - SOTA['dev-test']['f1']:+.4f}, Acc {best_devtest['accuracy'] - SOTA['dev-test']['acc']:+.4f}")
        all_results["dev-test"] = {"results": devtest_results, "best": best_devtest}
        np.save(output_dir / "devtest_probs.npy", devtest_probs)

    # Evaluate on TEST
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    test_labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    test_results, test_probs = evaluate_with_threshold(trainer, test_dataset, test_labels)

    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in test_results:
        marker = "🔥" if r["f1"] >= SOTA["test"]["f1"] else ""
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {marker}")

    best_test = max(test_results, key=lambda x: x["f1"])
    print(f"\n✨ Best test F1: {best_test['f1']:.4f} @ threshold {best_test['threshold']:.2f}")
    print(f"   SOTA test: F1={SOTA['test']['f1']:.4f}, Acc={SOTA['test']['acc']:.4f}")
    print(f"   Gap: F1 {best_test['f1'] - SOTA['test']['f1']:+.4f}, Acc {best_test['accuracy'] - SOTA['test']['acc']:+.4f}")
    all_results["test"] = {"results": test_results, "best": best_test}

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Split':<12} {'Best F1':<10} {'Best Acc':<10} {'SOTA F1':<10} {'Gap':<10}")
    print("-" * 55)
    print(f"{'Dev':<12} {best_dev['f1']:<10.4f} {best_dev['accuracy']:<10.4f} {'-':<10} {'-':<10}")
    if devtest_df is not None:
        gap_devtest = best_devtest['f1'] - SOTA['dev-test']['f1']
        print(f"{'Dev-test':<12} {best_devtest['f1']:<10.4f} {best_devtest['accuracy']:<10.4f} {SOTA['dev-test']['f1']:<10.4f} {gap_devtest:+.4f}")
    gap_test = best_test['f1'] - SOTA['test']['f1']
    print(f"{'Test':<12} {best_test['f1']:<10.4f} {best_test['accuracy']:<10.4f} {SOTA['test']['f1']:<10.4f} {gap_test:+.4f}")

    # Check for SOTA
    if best_test['f1'] >= SOTA['test']['f1']:
        print("\n🏆 TEST SOTA ACHIEVED!")
    if devtest_df is not None and best_devtest['f1'] >= SOTA['dev-test']['f1']:
        print("🏆 DEV-TEST SOTA ACHIEVED!")

    # Save results
    results_dict = {
        "model": config.model_name,
        "config": {
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
        },
        "all_results": all_results,
        "sota": SOTA,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=float)

    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "dev_probs.npy", dev_probs)

    print(f"\nResults saved to: {output_dir}")

    return all_results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa on augmented data")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        disable_early_stopping=args.no_early_stopping,
    )

    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_model(config, output_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
