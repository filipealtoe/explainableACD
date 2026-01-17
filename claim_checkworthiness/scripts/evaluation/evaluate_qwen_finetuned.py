#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen 2.5 7B on CT24 dev-test and test sets.

This model was fine-tuned on Together AI for checkworthiness classification.

Usage:
    python experiments/scripts/evaluate_qwen_finetuned.py --model-path /path/to/model
    python experiments/scripts/evaluate_qwen_finetuned.py --model-path ./qwen_finetuned --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Paths
# =============================================================================

# Try multiple paths for data files
_default_data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
RAW_DATA_DIR = _default_data_dir if _default_data_dir.exists() else Path(".")
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "qwen_finetuned"

# SOTA benchmarks
SOTA = {
    "dev-test": {"f1": 0.932, "acc": 0.955},
    "test": {"f1": 0.82, "acc": 0.905},
}

# Same prompts used during fine-tuning
SYSTEM_PROMPT = """You are a fact-checking assistant. Your task is to determine if statements are checkworthy.

Checkworthy criteria:
- Makes a specific factual claim that can be verified
- Would matter to the public if false
- Contains verifiable information (names, numbers, events)

Not checkworthy:
- Personal opinions or preferences
- Vague statements without specific claims
- Questions, greetings, or casual talk
- Future predictions
- Obvious common knowledge

Always respond with exactly: Yes or No"""

USER_TEMPLATE = """Statement: {text}

Is this statement checkworthy?"""


# =============================================================================
# Data Loading
# =============================================================================

def load_split(split: str) -> tuple[list[str], list[int]]:
    """Load a data split."""
    if split == "dev-test":
        path = RAW_DATA_DIR / "CT24_checkworthy_english_dev-test.tsv"
    elif split == "test":
        path = RAW_DATA_DIR / "CT24_checkworthy_english_test_gold.tsv"
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pl.read_csv(path, separator="\t")
    texts = df["Text"].to_list()
    labels = [1 if l == "Yes" else 0 for l in df["class_label"].to_list()]

    return texts, labels


# =============================================================================
# Model Loading & Inference
# =============================================================================

def load_model(model_path: str, device: str = "cuda"):
    """Load fine-tuned Qwen model."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # CRITICAL: Decoder-only models need left-padding for correct batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Padding side: {tokenizer.padding_side}")

    return model, tokenizer


def format_prompt(text: str, tokenizer) -> str:
    """Format prompt using Qwen chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]

    # Try tokenizer's chat template first
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            pass

    # Fallback: Qwen/ChatML format
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{USER_TEMPLATE.format(text=text)}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    return prompt


def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cuda",
) -> list[str]:
    """Get predictions for a batch of texts."""
    prompts = [format_prompt(text, tokenizer) for text in texts]

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    # Generate - greedy decoding for deterministic output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            top_k=None,  # Disable top_k for greedy
            top_p=None,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part
    predictions = []
    for i, output in enumerate(outputs):
        # With left-padding, the input length is the full padded length
        input_len = inputs["input_ids"].shape[1]

        # Get only the newly generated tokens
        generated_tokens = output[input_len:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Parse Yes/No - check for exact match first, then contains
        gen_lower = generated.lower().strip()
        if gen_lower == "yes" or gen_lower.startswith("yes"):
            predictions.append("Yes")
        elif gen_lower == "no" or gen_lower.startswith("no"):
            predictions.append("No")
        elif "yes" in gen_lower:
            predictions.append("Yes")
        else:
            predictions.append("No")

        # Debug: print first few predictions
        if len(predictions) <= 5:
            print(f"      DEBUG [{len(predictions)}]: raw='{generated}' -> {predictions[-1]}")

    return predictions


def evaluate_split(
    model,
    tokenizer,
    split: str,
    batch_size: int = 4,
    device: str = "cuda",
) -> dict:
    """Evaluate model on a split."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATING ON {split.upper()}")
    print("=" * 70)

    texts, labels = load_split(split)
    print(f"Samples: {len(texts)}")
    print(f"Positive: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")

    # Predict in batches
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting {split}"):
        batch_texts = texts[i:i + batch_size]
        batch_preds = predict_batch(model, tokenizer, batch_texts, device)
        all_preds.extend(batch_preds)

    # Convert to binary
    y_pred = [1 if p == "Yes" else 0 for p in all_preds]
    y_true = labels

    # Prediction distribution
    pred_yes = sum(y_pred)
    pred_no = len(y_pred) - pred_yes
    print(f"\nPrediction distribution: Yes={pred_yes} ({100*pred_yes/len(y_pred):.1f}%), No={pred_no} ({100*pred_no/len(y_pred):.1f}%)")

    # Metrics
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    # Compare to SOTA
    if split in SOTA:
        delta_f1 = f1 - SOTA[split]["f1"]
        delta_acc = acc - SOTA[split]["acc"]
        print(f"\nResults:")
        print(f"  F1:        {f1:.4f} ({delta_f1:+.4f} vs SOTA {SOTA[split]['f1']})")
        print(f"  Accuracy:  {acc:.4f} ({delta_acc:+.4f} vs SOTA {SOTA[split]['acc']})")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
    else:
        print(f"\nResults:")
        print(f"  F1:        {f1:.4f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=['No', 'Yes'])}")

    return {
        "split": split,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "predictions": all_preds,
        "labels": ["Yes" if l == 1 else "No" for l in labels],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen on CT24")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--splits", nargs="+", default=["dev-test", "test"], help="Splits to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU will be very slow!")

    # Load model
    model, tokenizer = load_model(args.model_path, device)

    # Evaluate
    results = {}
    for split in args.splits:
        result = evaluate_split(model, tokenizer, split, args.batch_size, device)
        results[split] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Split':<12} {'F1':<10} {'Acc':<10} {'vs SOTA F1':<12} {'vs SOTA Acc':<12}")
    print("-" * 56)
    for split, res in results.items():
        delta_f1 = res["f1"] - SOTA.get(split, {}).get("f1", 0)
        delta_acc = res["accuracy"] - SOTA.get(split, {}).get("acc", 0)
        print(f"{split:<12} {res['f1']:<10.4f} {res['accuracy']:<10.4f} {delta_f1:+.4f}       {delta_acc:+.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "evaluation_results.json"

    # Convert for JSON serialization
    json_results = {}
    for split, res in results.items():
        json_results[split] = {
            "f1": float(res["f1"]),
            "accuracy": float(res["accuracy"]),
            "precision": float(res["precision"]),
            "recall": float(res["recall"]),
        }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
