#!/usr/bin/env python3
"""
Reduce CT24 embedding features with PCA.

This script:
1. Loads TSV files from generate_embeddings.py (train/dev/test)
2. Fits PCA on train embeddings
3. Applies PCA to dev/test embeddings
4. Saves reduced TSVs with confidence features + reduced embeddings + label

Usage:
    python experiments/scripts/reduce_ct24_embeddings.py --pca 64
    python experiments/scripts/reduce_ct24_embeddings.py --pca 32 \
        --input-dir data/processed/CT24_with_embeddings/full_384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_embeddings" / "full_384"


def load_tsv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t")


def get_embedding_cols(df: pl.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def get_label_col(df: pl.DataFrame, label_col: str | None) -> str:
    if label_col and label_col in df.columns:
        return label_col
    if "label_encoded" in df.columns:
        return "label_encoded"
    return df.columns[-1]


def reduce_split(
    df: pl.DataFrame,
    emb_cols: list[str],
    label_col: str,
    pca: PCA,
    emb_prefix: str,
) -> pl.DataFrame:
    emb_matrix = df.select(emb_cols).to_numpy()
    reduced = pca.transform(emb_matrix)
    reduced_cols = [f"{emb_prefix}{i}" for i in range(reduced.shape[1])]
    reduced_df = pl.DataFrame({col: reduced[:, i] for i, col in enumerate(reduced_cols)})

    non_emb_cols = [col for col in df.columns if col not in emb_cols and col != label_col]
    return pl.concat(
        [df.select(non_emb_cols), reduced_df, df.select(label_col)],
        how="horizontal",
    )


def copy_mapping_files(input_dir: Path, output_dir: Path) -> None:
    for split in ("train", "dev", "test"):
        src = input_dir / f"{split}.mapping.tsv"
        if src.exists():
            dst = output_dir / src.name
            shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce CT24 embedding features using PCA"
    )
    parser.add_argument(
        "--pca",
        type=int,
        required=True,
        help="Number of PCA components (e.g., 32, 64)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with train/dev/test TSVs (default: full_384)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling pca_<dims>)",
    )
    parser.add_argument(
        "--emb-prefix",
        type=str,
        default="emb_",
        help="Embedding column prefix (default: emb_)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name (default: label_encoded or last column)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = input_dir.parent / f"pca_{args.pca}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CT24 EMBEDDING DIMENSION REDUCTION (PCA)")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"PCA components: {args.pca}")

    train_path = input_dir / "train.tsv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train TSV: {train_path}")

    print("\n📂 Loading train split")
    train_df = load_tsv(train_path)
    emb_cols = get_embedding_cols(train_df, args.emb_prefix)
    label_col = get_label_col(train_df, args.label_col)

    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{args.emb_prefix}'")

    if args.pca <= 0 or args.pca >= len(emb_cols):
        raise ValueError(
            f"PCA components must be between 1 and {len(emb_cols) - 1}"
        )

    print(f"   Rows: {len(train_df):,}")
    print(f"   Embedding dims: {len(emb_cols)}")
    print(f"   Label column: {label_col}")

    print("\n🔧 Fitting PCA on train embeddings")
    pca = PCA(n_components=args.pca)
    train_emb = train_df.select(emb_cols).to_numpy()
    pca.fit(train_emb)
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    pca_path = output_dir / "pca_model.joblib"
    joblib.dump(pca, pca_path)
    print(f"   💾 Saved PCA model: {pca_path}")

    for split in ("train", "dev", "test"):
        split_path = input_dir / f"{split}.tsv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split} TSV: {split_path}")

        print(f"\n🔄 Processing {split} split")
        split_df = load_tsv(split_path)
        split_label_col = get_label_col(split_df, args.label_col)
        split_emb_cols = get_embedding_cols(split_df, args.emb_prefix)

        if split_emb_cols != emb_cols:
            raise ValueError(
                f"Embedding columns mismatch in {split}.tsv "
                f"(expected {len(emb_cols)}, got {len(split_emb_cols)})"
            )

        reduced_df = reduce_split(
            df=split_df,
            emb_cols=split_emb_cols,
            label_col=split_label_col,
            pca=pca,
            emb_prefix=args.emb_prefix,
        )

        out_path = output_dir / f"{split}.tsv"
        reduced_df.write_csv(out_path, separator="\t")
        print(f"   💾 Saved: {out_path}")
        print(f"   Shape: {reduced_df.shape}")

    copy_mapping_files(input_dir, output_dir)

    print("\n✅ Done.")
    print(f"   Output directory: {output_dir}")
    print("\n🚀 Next step: Run classifier with embeddings:")
    print(f"   python experiments/scripts/run_ct24_classifier.py \\")
    print(f"       --data-dir {output_dir}")


if __name__ == "__main__":
    main()
