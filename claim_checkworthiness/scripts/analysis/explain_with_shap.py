#!/usr/bin/env python3
"""
SHAP Explanations for Checkworthiness Predictions.

Uses SHAP (SHapley Additive exPlanations) to explain which words/tokens
contribute to checkworthiness predictions.

SHAP is based on game theory - it fairly distributes the "credit" for
a prediction among all input features (tokens).

Usage:
    python explain_with_shap.py \
        --model-dir ~/ensemble_results/seed_0/deberta-v3-large \
        --data-dir ~/data \
        --n-samples 50

Requirements:
    pip install shap transformers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    """Load the trained model and tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Find model directory
    if (model_dir / "best_model").exists():
        actual_dir = model_dir / "best_model"
    elif (model_dir / "config.json").exists():
        actual_dir = model_dir
    else:
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                if (subdir / "config.json").exists():
                    actual_dir = subdir
                    break
                if (subdir / "best_model").exists():
                    actual_dir = subdir / "best_model"
                    break

    print(f"   Loading from: {actual_dir}")
    tokenizer = AutoTokenizer.from_pretrained(actual_dir)
    model = AutoModelForSequenceClassification.from_pretrained(actual_dir).to(device)
    model.eval()

    return model, tokenizer


def get_prediction_proba(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    """Get prediction probabilities for texts."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def explain_with_shap(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    n_background: int = 20,
):
    """Generate SHAP explanations for predictions."""
    try:
        import shap
    except ImportError:
        print("❌ SHAP not installed. Run: pip install shap")
        return None

    print("\n🔧 Setting up SHAP explainer...")

    # Create a prediction function for SHAP
    def predict_fn(texts_list):
        """Prediction function for SHAP."""
        if isinstance(texts_list, np.ndarray):
            texts_list = texts_list.tolist()
        if isinstance(texts_list, str):
            texts_list = [texts_list]

        probs = get_prediction_proba(model, tokenizer, texts_list, device)
        return probs

    # Use a subset of texts as background
    background_texts = texts[:n_background]

    # Create SHAP explainer
    print(f"   Using {n_background} background samples")
    explainer = shap.Explainer(predict_fn, tokenizer, output_names=["No", "Yes"])

    print("\n📊 Generating SHAP explanations...")
    shap_values = explainer(texts)

    return shap_values, explainer


def explain_with_integrated_gradients(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
):
    """Use Integrated Gradients via Captum for explanations."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("❌ Captum not installed. Run: pip install captum")
        return None

    print("\n🔧 Setting up Integrated Gradients (Captum)...")

    # For DeBERTa, we use a simpler approach: attribute on input embeddings directly
    # Get the embeddings module
    if hasattr(model, 'deberta'):
        embed_module = model.deberta.embeddings
    elif hasattr(model, 'roberta'):
        embed_module = model.roberta.embeddings
    else:
        print("   ⚠️ Unknown model architecture, skipping IG")
        return None

    def forward_with_embeds(input_embeds, attention_mask):
        """Forward function that takes embeddings directly."""
        # DeBERTa expects inputs_embeds
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits[:, 1]  # Probability of "Yes" (checkworthy)

    ig = IntegratedGradients(forward_with_embeds)

    attributions_all = []
    tokens_all = []

    for text in tqdm(texts, desc="Computing attributions"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get input embeddings using the embedding module
        with torch.no_grad():
            input_embeds = embed_module(input_ids)
            # Baseline: zero embeddings (or padding token embeddings)
            baseline_embeds = torch.zeros_like(input_embeds)

        # Enable gradients for attribution
        input_embeds.requires_grad_(True)
        baseline_embeds.requires_grad_(True)

        # Compute attributions
        attributions = ig.attribute(
            input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            n_steps=25,  # Reduced for speed
        )

        # Sum over embedding dimension to get per-token attribution
        attr_sum = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        attributions_all.append(attr_sum)
        tokens_all.append(tokens)

    return attributions_all, tokens_all


def visualize_attribution(tokens: list[str], attributions: np.ndarray, label: int, pred_prob: float):
    """Create a text visualization of token attributions."""
    # Normalize attributions
    if attributions.max() - attributions.min() > 0:
        attr_normalized = (attributions - attributions.min()) / (attributions.max() - attributions.min())
    else:
        attr_normalized = np.zeros_like(attributions)

    label_str = "Yes (checkworthy)" if label == 1 else "No (not checkworthy)"

    print(f"\n   Label: {label_str} | Predicted prob: {pred_prob:.3f}")
    print("   " + "-" * 60)

    # Color coding in terminal (green = positive, red = negative)
    output = "   "
    for token, attr in zip(tokens, attributions):
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            continue

        # Clean up token
        token_clean = token.replace("▁", " ").replace("Ġ", " ")

        if attr > 0.1:
            output += f"\033[92m{token_clean}\033[0m"  # Green (checkworthy)
        elif attr < -0.1:
            output += f"\033[91m{token_clean}\033[0m"  # Red (not checkworthy)
        else:
            output += token_clean

    print(output)

    # Also show top contributing tokens
    token_attr_pairs = [(t, a) for t, a in zip(tokens, attributions)
                        if t not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]]
    token_attr_pairs.sort(key=lambda x: -abs(x[1]))

    print("\n   Top contributing tokens:")
    for token, attr in token_attr_pairs[:5]:
        direction = "→ checkworthy" if attr > 0 else "→ not checkworthy"
        print(f"      '{token.replace('▁', '').replace('Ġ', '')}': {attr:+.3f} {direction}")


def simple_attention_analysis(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
):
    """Analyze attention patterns as a simpler interpretability method."""
    print("\n🔍 Attention Pattern Analysis")
    print("-" * 50)

    model.eval()

    for i, text in enumerate(texts[:5]):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Get attention from last layer
        attentions = outputs.attentions[-1]  # (batch, heads, seq, seq)

        # Average over heads, get attention to [CLS] token
        cls_attention = attentions[0, :, 0, :].mean(dim=0).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

        print(f"\n   Sample {i+1}: {text}")
        print(f"   Tokens with highest attention from [CLS]:")

        token_attn = list(zip(tokens, cls_attention))
        token_attn.sort(key=lambda x: -x[1])

        for token, attn in token_attn[:8]:
            if token not in ["[CLS]", "[SEP]", "<s>", "</s>"]:
                print(f"      {token:<15} {attn:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--method", choices=["shap", "ig", "attention", "all"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print("\n📂 Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    # Load data
    print("\n📂 Loading data...")
    clean_dir = args.data_dir / "processed" / "CT24_clean"
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        if (clean_dir / name).exists():
            df = pl.read_parquet(clean_dir / name)
            break
    else:
        raw_path = args.data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
        df = pl.read_csv(raw_path, separator="\t")

    texts = df["Text"].to_list()[:args.n_samples]
    labels = [1 if l == "Yes" else 0 for l in df["class_label"].to_list()[:args.n_samples]]

    print(f"   Analyzing {len(texts)} samples")

    # Get predictions
    print("\n📊 Getting predictions...")
    probs = get_prediction_proba(model, tokenizer, texts, device)

    # Method 1: SHAP
    if args.method in ["shap", "all"]:
        print("\n" + "=" * 70)
        print("METHOD 1: SHAP (SHapley Additive exPlanations)")
        print("=" * 70)
        try:
            shap_values, explainer = explain_with_shap(
                model, tokenizer, texts[:10], labels[:10], device
            )
            if shap_values is not None:
                import shap

                # Print text-based SHAP explanations
                print("\n   📝 SHAP Token Attributions (for 'Yes' class):")
                print("   " + "-" * 60)

                for i in range(min(5, len(shap_values))):
                    sv = shap_values[i]
                    label_str = "✓ Yes" if labels[i] == 1 else "✗ No"
                    pred_prob = probs[i, 1]

                    print(f"\n   Sample {i+1} [{label_str}] (pred: {pred_prob:.2%})")
                    print(f"   Text: {texts[i]}")

                    # Get token attributions for "Yes" class (index 1)
                    if hasattr(sv, 'values') and sv.values is not None:
                        values = sv.values[:, 1] if len(sv.values.shape) > 1 else sv.values
                        tokens = sv.data if hasattr(sv, 'data') else []

                        # Get top contributing tokens
                        if len(values) > 0 and len(tokens) > 0:
                            token_attr = list(zip(tokens, values))
                            # Sort by absolute value
                            token_attr_sorted = sorted(token_attr, key=lambda x: -abs(x[1]))

                            print("   Top tokens → checkworthy:")
                            pos_tokens = [(t, v) for t, v in token_attr_sorted if v > 0.01][:5]
                            for token, val in pos_tokens:
                                print(f"      '{token}': {val:+.3f}")

                            print("   Top tokens → NOT checkworthy:")
                            neg_tokens = [(t, v) for t, v in token_attr_sorted if v < -0.01][:5]
                            for token, val in neg_tokens:
                                print(f"      '{token}': {val:+.3f}")

                # Save HTML visualization
                output_html = args.model_dir / "shap_explanations.html"
                print(f"\n   💾 Saving HTML visualization to: {output_html}")
                try:
                    shap.plots.text(shap_values, display=False)
                    # Save using shap's built-in HTML export
                    html_content = shap.plots.text(shap_values[:5], display=False)
                    if html_content:
                        with open(output_html, 'w') as f:
                            f.write(str(html_content))
                        print(f"   ✓ HTML saved to {output_html}")
                except Exception as html_err:
                    print(f"   ⚠️ Could not save HTML: {html_err}")

                print("   ✓ SHAP explanations computed successfully")
        except Exception as e:
            print(f"   ⚠️ SHAP failed: {e}")

    # Method 2: Integrated Gradients
    if args.method in ["ig", "all"]:
        print("\n" + "=" * 70)
        print("METHOD 2: Integrated Gradients (Captum)")
        print("=" * 70)
        try:
            ig_results = explain_with_integrated_gradients(
                model, tokenizer, texts[:5], device
            )
            if ig_results:
                attributions_all, tokens_all = ig_results
                print("\n   Sample explanations:")
                for i in range(min(3, len(texts))):
                    visualize_attribution(
                        tokens_all[i],
                        attributions_all[i],
                        labels[i],
                        probs[i, 1]
                    )
        except Exception as e:
            print(f"   ⚠️ Integrated Gradients failed: {e}")

    # Method 3: Attention Analysis
    if args.method in ["attention", "all"]:
        print("\n" + "=" * 70)
        print("METHOD 3: Attention Pattern Analysis")
        print("=" * 70)
        simple_attention_analysis(model, tokenizer, texts[:5], device)

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETABILITY SUMMARY")
    print("=" * 70)

    print("""
    Available methods for explaining checkworthiness predictions:

    1. SHAP (recommended for papers)
       - Gold standard for feature attribution
       - Shows contribution of each token
       - Game-theoretic foundation

    2. Integrated Gradients (via Captum)
       - Gradient-based attribution
       - Principled baseline comparison
       - Good for transformer models

    3. Attention Analysis
       - Shows what the model "looks at"
       - Fast and simple
       - Less rigorous than SHAP/IG

    For your paper, consider:
    - SHAP visualizations for sample explanations
    - Aggregated token importance across dataset
    - Comparison of attention patterns for Yes vs No samples
    """)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
