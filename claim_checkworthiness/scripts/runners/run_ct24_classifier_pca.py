#!/usr/bin/env python3
"""
CT24 Checkworthiness Classifier - PCA Embeddings Wrapper

Runs the existing pipeline on PCA-reduced embedding features.
Defaults to data/processed/CT24_with_embeddings/pca_32.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.run_ct24_classifier import run_full_pipeline

DEFAULT_DATA_DIR = (
    REPO_ROOT / "data" / "processed" / "CT24_with_embeddings" / "pca_32"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ct24_classifier_pca_32"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CT24 classifier pipeline using PCA-reduced embeddings"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with train/dev/test TSVs (default: pca_32)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter search (default: 30)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer trials (5) for testing",
    )

    args = parser.parse_args()
    n_trials = 5 if args.quick else args.n_trials

    run_full_pipeline(
        n_trials=n_trials,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        prepare_data=False,
    )


if __name__ == "__main__":
    main()
