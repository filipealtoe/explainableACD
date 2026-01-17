#!/usr/bin/env python3
"""
Embedding Analysis with PCA for Interpretability.

Extracts DeBERTa embeddings, applies PCA, and analyzes:
1. Which principal components correlate with checkworthiness
2. Cluster structure in embedding space
3. What semantic features each PC captures
4. Classification performance with reduced dimensions

Usage:
    python analyze_embeddings_pca.py \
        --model-dir ~/ensemble_results/seed_0/deberta-v3-large \
        --data-dir ~/data \
        --n-components 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# Embedding Extraction
# =============================================================================

def extract_embeddings(
    model_dir: Path,
    texts: list[str],
    batch_size: int = 16,
    layer: str = "last",  # "last", "pooler", or "mean"
) -> np.ndarray:
    """Extract embeddings from DeBERTa model."""
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find model directory
    if (model_dir / "best_model").exists():
        actual_dir = model_dir / "best_model"
    elif (model_dir / "config.json").exists():
        actual_dir = model_dir
    else:
        # Search subdirectories
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and (subdir / "config.json").exists():
                actual_dir = subdir
                break
            if subdir.is_dir() and (subdir / "best_model").exists():
                actual_dir = subdir / "best_model"
                break
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")

    print(f"   Loading model from: {actual_dir}")
    tokenizer = AutoTokenizer.from_pretrained(actual_dir)
    model = AutoModel.from_pretrained(actual_dir).to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if layer == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            elif layer == "mean":
                # Mean of all tokens (excluding padding)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                # [CLS] token from last layer
                emb = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# =============================================================================
# PCA Analysis
# =============================================================================

def analyze_pca_components(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    n_components: int = 16,
) -> dict:
    """Analyze PCA components and their relationship to checkworthiness."""

    print(f"\n📊 PCA Analysis with {n_components} components")
    print("-" * 50)

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Fit PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    print(f"   Original dimensions: {embeddings.shape[1]}")
    print(f"   Reduced dimensions: {n_components}")
    print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Analyze each component's correlation with label
    print(f"\n   Per-component analysis:")
    print(f"   {'PC':<5} {'Var%':<8} {'Corr':<8} {'|Corr|':<8} Interpretation")
    print("   " + "-" * 60)

    component_stats = []
    for i in range(n_components):
        var_ratio = pca.explained_variance_ratio_[i]
        corr = np.corrcoef(embeddings_pca[:, i], labels)[0, 1]

        # Simple interpretation based on correlation
        if abs(corr) > 0.3:
            interp = "STRONG predictor" if corr > 0 else "STRONG negative"
        elif abs(corr) > 0.15:
            interp = "Moderate predictor" if corr > 0 else "Moderate negative"
        else:
            interp = "Weak/noise"

        component_stats.append({
            "pc": i + 1,
            "variance_ratio": float(var_ratio),
            "correlation": float(corr),
            "abs_correlation": float(abs(corr)),
        })

        print(f"   PC{i+1:<3} {var_ratio:>6.2%}   {corr:>+6.3f}   {abs(corr):>6.3f}   {interp}")

    return {
        "pca": pca,
        "embeddings_pca": embeddings_pca,
        "scaler": scaler,
        "component_stats": component_stats,
        "total_variance": float(pca.explained_variance_ratio_.sum()),
    }


def test_classification_with_pca(
    embeddings_pca: np.ndarray,
    labels: np.ndarray,
    n_components_list: list[int] = [2, 4, 8, 16, 32],
) -> dict:
    """Test classification performance with different numbers of PCA components."""

    print(f"\n📈 Classification with Reduced Dimensions")
    print("-" * 50)
    print(f"   {'Components':<12} {'F1':<10} {'Accuracy':<10}")
    print("   " + "-" * 35)

    results = []

    for n in n_components_list:
        if n > embeddings_pca.shape[1]:
            continue

        X = embeddings_pca[:, :n]

        # Simple train/test split (70/30)
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = labels[:split], labels[split:]

        # Logistic regression
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)

        results.append({"n_components": n, "f1": float(f1), "accuracy": float(acc)})
        print(f"   {n:<12} {f1:<10.4f} {acc:<10.4f}")

    # Also test with full embeddings
    return results


def find_interpretable_directions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    pca_result: dict,
) -> dict:
    """Find interpretable directions in embedding space."""

    print(f"\n🔍 Finding Interpretable Directions")
    print("-" * 50)

    embeddings_pca = pca_result["embeddings_pca"]

    # Find most discriminative PC
    correlations = [abs(s["correlation"]) for s in pca_result["component_stats"]]
    best_pc_idx = np.argmax(correlations)
    best_pc = best_pc_idx + 1

    print(f"   Most discriminative: PC{best_pc} (|corr|={correlations[best_pc_idx]:.3f})")

    # Find extreme samples on this PC
    pc_values = embeddings_pca[:, best_pc_idx]
    sorted_indices = np.argsort(pc_values)

    print(f"\n   Samples with LOWEST PC{best_pc} values (likely non-checkworthy):")
    for i in sorted_indices[:5]:
        label_str = "Yes" if labels[i] == 1 else "No"
        print(f"      [{label_str}] {texts[i]}")

    print(f"\n   Samples with HIGHEST PC{best_pc} values (likely checkworthy):")
    for i in sorted_indices[-5:]:
        label_str = "Yes" if labels[i] == 1 else "No"
        print(f"      [{label_str}] {texts[i]}")

    # Analyze what features correlate with top PCs
    return {
        "most_discriminative_pc": best_pc,
        "correlation": correlations[best_pc_idx],
    }


def analyze_cluster_structure(
    embeddings_pca: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Analyze cluster structure in PCA space."""
    from sklearn.cluster import KMeans

    print(f"\n🔮 Cluster Structure Analysis")
    print("-" * 50)

    # Use first 8 PCs for clustering
    X = embeddings_pca[:, :8]

    # Try different numbers of clusters
    for n_clusters in [2, 3, 4]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Check how well clusters align with checkworthiness
        print(f"\n   {n_clusters} clusters:")
        for c in range(n_clusters):
            mask = cluster_labels == c
            n_in_cluster = mask.sum()
            n_positive = labels[mask].sum()
            pct_positive = n_positive / n_in_cluster * 100 if n_in_cluster > 0 else 0
            print(f"      Cluster {c}: {n_in_cluster} samples, {pct_positive:.1f}% checkworthy")

    return {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory with trained DeBERTa model")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--n-components", type=int, default=16,
                        help="Number of PCA components (default: 16)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "dev-test", "test"],
                        help="Data split to analyze")
    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING ANALYSIS WITH PCA")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading {args.split} data...")
    clean_dir = args.data_dir / "processed" / "CT24_clean"

    split_name = args.split.replace("-", "_")
    for name in [f"CT24_{split_name}_clean.parquet", f"CT24_{split_name}.parquet"]:
        if (clean_dir / name).exists():
            df = pl.read_parquet(clean_dir / name)
            break
    else:
        # Try TSV
        for name in [f"CT24_{split_name}_clean.tsv", f"CT24_{split_name}.tsv"]:
            if (clean_dir / name).exists():
                df = pl.read_csv(clean_dir / name, separator="\t")
                break
        else:
            # Raw fallback
            raw_map = {"test": "test_gold", "train": "train", "dev": "dev", "dev_test": "dev-test"}
            raw_name = raw_map.get(split_name, split_name)
            raw_path = args.data_dir / "raw" / "CT24_checkworthy_english" / f"CT24_checkworthy_english_{raw_name}.tsv"
            df = pl.read_csv(raw_path, separator="\t")

    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    print(f"   Loaded {len(texts)} samples")
    print(f"   Class distribution: {labels.sum()} positive, {len(labels) - labels.sum()} negative")

    # Extract embeddings
    print(f"\n📊 Extracting embeddings from model...")
    embeddings = extract_embeddings(args.model_dir, texts)
    print(f"   Embedding shape: {embeddings.shape}")

    # PCA Analysis
    pca_result = analyze_pca_components(embeddings, labels, texts, args.n_components)

    # Test classification with different dimensions
    n_components_list = [2, 4, 8, 16, 32, 64]
    clf_results = test_classification_with_pca(
        pca_result["embeddings_pca"], labels,
        [n for n in n_components_list if n <= args.n_components]
    )

    # Find interpretable directions
    interp_results = find_interpretable_directions(
        embeddings, labels, texts, pca_result
    )

    # Cluster analysis
    cluster_results = analyze_cluster_structure(pca_result["embeddings_pca"], labels)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n   📊 PCA Summary:")
    print(f"      Total variance explained ({args.n_components} PCs): {pca_result['total_variance']:.2%}")

    # Find PCs with strongest correlation
    sorted_pcs = sorted(pca_result["component_stats"], key=lambda x: -x["abs_correlation"])
    print(f"\n   🎯 Most Predictive Principal Components:")
    for pc in sorted_pcs[:5]:
        print(f"      PC{pc['pc']}: corr={pc['correlation']:+.3f}, var={pc['variance_ratio']:.2%}")

    print(f"\n   📈 Classification Performance:")
    for r in clf_results:
        print(f"      {r['n_components']} dims: F1={r['f1']:.4f}")

    # Save results
    if args.output:
        output_data = {
            "n_components": args.n_components,
            "total_variance_explained": pca_result["total_variance"],
            "component_stats": pca_result["component_stats"],
            "classification_results": clf_results,
            "interpretability": interp_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
