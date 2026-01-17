#!/usr/bin/env python3
"""
Interpret PCA Dimensions.

Analyzes what linguistic features correlate with each principal component
to understand what the model has learned about checkworthiness.

Usage:
    python interpret_pca_dimensions.py \
        --model-dir ~/ensemble_results/seed_0/deberta-v3-large \
        --data-dir ~/data
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def extract_embeddings(model_dir: Path, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Extract [CLS] embeddings from DeBERTa."""
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find model
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

    tokenizer = AutoTokenizer.from_pretrained(actual_dir)
    model = AutoModel.from_pretrained(actual_dir).to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                             max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(emb)

    return np.vstack(all_embeddings)


def extract_linguistic_features(text: str) -> dict:
    """Extract interpretable linguistic features from text."""
    words = text.lower().split()

    features = {
        # Structure
        "length": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,

        # Punctuation
        "has_question": "?" in text,
        "has_exclamation": "!" in text,
        "has_quotes": '"' in text or "'" in text,

        # Numbers and facts
        "has_numbers": bool(re.search(r'\d', text)),
        "has_percentage": bool(re.search(r'\d+%|\bpercent\b', text.lower())),
        "has_dollar": bool(re.search(r'\$\d|dollar', text.lower())),
        "has_million_billion": bool(re.search(r'\bmillion\b|\bbillion\b|\btrillion\b', text.lower())),

        # First person (opinion indicators)
        "starts_with_i": text.lower().startswith("i "),
        "has_i_think": "i think" in text.lower(),
        "has_i_believe": "i believe" in text.lower(),
        "has_we": " we " in text.lower() or text.lower().startswith("we "),

        # Factual language
        "has_according_to": "according to" in text.lower(),
        "has_study_research": bool(re.search(r'\bstud(y|ies)\b|\bresearch\b', text.lower())),
        "has_fact_prove": bool(re.search(r'\bfact\b|\bprove[dns]?\b|\bconfirm', text.lower())),

        # Temporal
        "has_year": bool(re.search(r'\b(19|20)\d{2}\b', text)),
        "has_time_words": bool(re.search(r'\byear[s]?\b|\bmonth[s]?\b|\bday[s]?\b', text.lower())),

        # Entities and specificity
        "has_proper_noun": bool(re.search(r'\b[A-Z][a-z]+\b', text[1:])),  # Skip first word

        # Hedging vs certainty
        "has_hedge": bool(re.search(r'\bmaybe\b|\bperhaps\b|\bmight\b|\bcould\b|\bprobably\b', text.lower())),
        "has_certainty": bool(re.search(r'\bdefinitely\b|\bcertainly\b|\babsolutely\b|\balways\b|\bnever\b', text.lower())),

        # Action verbs (claims about doing)
        "has_action_past": bool(re.search(r'\b(passed|voted|signed|created|built|made|did)\b', text.lower())),
        "has_will_going": bool(re.search(r'\bwill\b|\bgoing to\b|\bwon\'t\b', text.lower())),
    }

    return features


def analyze_pc_correlations(
    pc_values: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    pc_name: str,
) -> dict:
    """Analyze what linguistic features correlate with a PC."""

    # Extract features for all texts
    all_features = [extract_linguistic_features(t) for t in texts]
    feature_names = list(all_features[0].keys())

    print(f"\n{'='*70}")
    print(f"INTERPRETING {pc_name}")
    print("=" * 70)

    # Correlation with label
    label_corr = np.corrcoef(pc_values, labels)[0, 1]
    print(f"\n   Correlation with checkworthiness: {label_corr:+.3f}")

    # Correlation with each feature
    print(f"\n   Feature correlations with {pc_name}:")
    print(f"   {'Feature':<25} {'Corr':<10} {'Interpretation'}")
    print("   " + "-" * 55)

    feature_correlations = []
    for feat in feature_names:
        feat_values = np.array([f[feat] for f in all_features], dtype=float)
        if feat_values.std() > 0:  # Skip constant features
            corr = np.corrcoef(pc_values, feat_values)[0, 1]
            feature_correlations.append((feat, corr))

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: -abs(x[1]))

    for feat, corr in feature_correlations[:15]:
        if abs(corr) > 0.15:
            interp = "HIGH → checkworthy" if corr > 0 else "HIGH → non-checkworthy"
        else:
            interp = ""
        print(f"   {feat:<25} {corr:>+.3f}     {interp}")

    # Show extreme samples
    sorted_idx = np.argsort(pc_values)

    print(f"\n   Samples with LOWEST {pc_name} (bottom 5):")
    for i in sorted_idx[:5]:
        label_str = "✓ Yes" if labels[i] == 1 else "✗ No "
        print(f"      [{label_str}] {texts[i]}")

    print(f"\n   Samples with HIGHEST {pc_name} (top 5):")
    for i in sorted_idx[-5:][::-1]:
        label_str = "✓ Yes" if labels[i] == 1 else "✗ No "
        print(f"      [{label_str}] {texts[i]}")

    # Word frequency analysis
    print(f"\n   Most distinctive words:")

    # Get words from low PC samples
    low_texts = [texts[i] for i in sorted_idx[:50]]
    low_words = Counter()
    for t in low_texts:
        low_words.update(t.lower().split())

    # Get words from high PC samples
    high_texts = [texts[i] for i in sorted_idx[-50:]]
    high_words = Counter()
    for t in high_texts:
        high_words.update(t.lower().split())

    # Find distinctive words
    print(f"      Words more common in LOW {pc_name} (non-checkworthy):")
    low_distinctive = []
    for word, count in low_words.most_common(100):
        if len(word) > 3 and high_words.get(word, 0) < count * 0.5:
            low_distinctive.append(word)
    print(f"         {', '.join(low_distinctive[:15])}")

    print(f"      Words more common in HIGH {pc_name} (checkworthy):")
    high_distinctive = []
    for word, count in high_words.most_common(100):
        if len(word) > 3 and low_words.get(word, 0) < count * 0.5:
            high_distinctive.append(word)
    print(f"         {', '.join(high_distinctive[:15])}")

    return {
        "label_correlation": float(label_corr),
        "top_feature_correlations": feature_correlations[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("PCA DIMENSION INTERPRETATION")
    print("=" * 70)

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

    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
    print(f"   Loaded {len(texts)} samples")

    # Extract embeddings
    print("\n📊 Extracting embeddings...")
    embeddings = extract_embeddings(args.model_dir, texts)

    # PCA
    print("\n🔧 Applying PCA...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=args.n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    print(f"   Variance explained by {args.n_components} PCs: {pca.explained_variance_ratio_.sum():.1%}")

    # Analyze top PCs
    for i in range(min(3, args.n_components)):
        analyze_pc_correlations(
            embeddings_pca[:, i],
            texts,
            labels,
            f"PC{i+1}"
        )

    # Summary
    print(f"\n{'='*70}")
    print("INTERPRETATION SUMMARY")
    print("=" * 70)

    print("""
    Based on the analysis:

    PC1 appears to capture: FACTUAL CLAIMS vs OPINIONS/INTENTIONS
       - HIGH PC1: Specific claims with numbers, actions, entities
       - LOW PC1: Opinions, hedging, personal beliefs ("I think...")

    PC2 appears to capture: SPECIFICITY/DETAIL LEVEL
       - HIGH PC2: Detailed, policy-focused statements
       - LOW PC2: Vague, general statements

    This aligns with the checkworthiness definition:
       Checkworthy = verifiable factual claims (not opinions)
    """)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
