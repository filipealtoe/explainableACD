#!/usr/bin/env python3
"""
Train classifiers on DeBERTa CLS embeddings + LLM confidence features.

Pipeline:
1. Load fine-tuned DeBERTa and extract CLS embeddings
2. Load LLM features (p_true, entropy for each dimension)
3. Concatenate: [CLS_embedding (1024) | LLM_features (~20)]
4. Train multiple classifiers: LogReg, RandomForest, CatBoost, MLP
5. Evaluate on dev-test and test

Usage:
    python experiments/scripts/train_embedding_classifiers.py
    python experiments/scripts/train_embedding_classifiers.py --no-cache  # Force re-extraction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV

# Gradient boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =============================================================================
# Paths
# =============================================================================

DEBERTA_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
LLM_FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"
CACHE_DIR = Path(__file__).parent.parent / "results" / "embeddings_cache"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "embedding_classifiers"

# SOTA benchmarks
SOTA = {
    "dev-test": {"f1": 0.932, "acc": 0.955},
    "test": {"f1": 0.82, "acc": 0.905},
}

# LLM features to use (from v4 features)
LLM_FEATURE_COLS = [
    # P(Yes) probabilities for each dimension
    "check_p_yes", "verif_p_yes", "harm_p_yes",
    # Normalized entropy (uncertainty, 0-1 scale)
    "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    # Voting
    "yes_vote_count",
]


# =============================================================================
# MLP Classifier
# =============================================================================

class MLPClassifier(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [512, 128], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    class_weight: float = 3.0,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[MLPClassifier, float]:
    """Train MLP with early stopping."""

    if hidden_dims is None:
        hidden_dims = [512, 128]

    model = MLPClassifier(X_train.shape[1], hidden_dims, dropout=dropout).to(device)

    # Class-weighted BCE loss
    pos_weight = torch.tensor([class_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Prepare data
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = (torch.sigmoid(val_logits) >= 0.5).cpu().numpy().astype(int)
            val_f1 = f1_score(y_val, val_preds)

        scheduler.step(1 - val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_f1


# =============================================================================
# Data Loading
# =============================================================================

def load_split_data(split: str) -> tuple[list[str], np.ndarray, pl.DataFrame]:
    """Load text and labels for a split."""

    if split == "train":
        # Use cleaned training data
        path = CLEAN_DATA_DIR / "CT24_train_clean.parquet"
        if path.exists():
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(CLEAN_DATA_DIR / "CT24_train_clean.tsv", separator="\t")
    elif split == "dev":
        path = CLEAN_DATA_DIR / "CT24_dev_clean.parquet"
        if path.exists():
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(CLEAN_DATA_DIR / "CT24_dev_clean.tsv", separator="\t")
    else:
        # RAW for dev-test and test
        if split == "dev-test":
            df = pl.read_csv(RAW_DATA_DIR / "CT24_checkworthy_english_dev-test.tsv", separator="\t")
        else:
            df = pl.read_csv(RAW_DATA_DIR / "CT24_checkworthy_english_test_gold.tsv", separator="\t")

    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    return texts, labels, df


def load_llm_features(split: str, sentence_ids: list) -> np.ndarray | None:
    """Load LLM features for a split."""

    # Map split names
    split_map = {"dev-test": "dev", "test": "test", "train": "train", "dev": "dev"}
    feat_split = split_map.get(split, split)

    path = LLM_FEATURES_DIR / f"{feat_split}_llm_features.parquet"
    if not path.exists():
        print(f"   ⚠️  LLM features not found: {path}")
        return None

    feat_df = pl.read_parquet(path)

    # Filter to available features
    available_cols = [c for c in LLM_FEATURE_COLS if c in feat_df.columns]
    if not available_cols:
        print(f"   ⚠️  No LLM features available")
        return None

    # Match by sentence_id
    sentence_ids_str = [str(s) for s in sentence_ids]
    feat_df = feat_df.with_columns(pl.col("sentence_id").cast(pl.Utf8))

    # Create lookup
    feat_dict = {row["sentence_id"]: row for row in feat_df.to_dicts()}

    features = []
    for sid in sentence_ids_str:
        if sid in feat_dict:
            row = feat_dict[sid]
            features.append([row.get(c, 0.0) or 0.0 for c in available_cols])
        else:
            features.append([0.0] * len(available_cols))

    return np.array(features, dtype=np.float32)


# =============================================================================
# Embedding Extraction
# =============================================================================

def extract_embeddings(
    texts: list[str],
    model_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Extract CLS embeddings from fine-tuned DeBERTa."""

    from transformers import AutoModel, DebertaV2Tokenizer

    print(f"   Loading model from {model_path}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="   Extracting embeddings"):
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

            # CLS token is first token
            cls_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def get_or_extract_embeddings(
    split: str,
    texts: list[str],
    use_cache: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Get embeddings from cache or extract them."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{split}_cls_embeddings.npy"

    if use_cache and cache_path.exists():
        print(f"   Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    embeddings = extract_embeddings(texts, DEBERTA_MODEL, device=device)

    # Cache
    np.save(cache_path, embeddings)
    print(f"   Cached embeddings to {cache_path}")

    return embeddings


# =============================================================================
# Classifier Training
# =============================================================================

def train_and_evaluate_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
) -> dict:
    """Train multiple classifiers and evaluate."""

    results = {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # ==========================================================================
    # Train MLP with hyperparameter tuning (FIRST - uses GPU)
    # ==========================================================================
    print(f"\n🧠 Training MLP with hyperparameter tuning...")

    mlp_param_grid = {
        "hidden_dims": [[512, 128], [512, 256, 64], [256, 64]],
        "lr": [1e-3, 5e-4],
        "dropout": [0.2, 0.3, 0.4],
        "class_weight": [2.0, 3.0, 4.0],
    }

    best_mlp = None
    best_mlp_f1 = 0
    best_mlp_params = {}

    # Grid search over MLP hyperparameters
    total_configs = (len(mlp_param_grid["hidden_dims"]) *
                     len(mlp_param_grid["lr"]) *
                     len(mlp_param_grid["dropout"]) *
                     len(mlp_param_grid["class_weight"]))
    config_idx = 0

    for hidden_dims in mlp_param_grid["hidden_dims"]:
        for lr in mlp_param_grid["lr"]:
            for dropout in mlp_param_grid["dropout"]:
                for class_weight in mlp_param_grid["class_weight"]:
                    config_idx += 1
                    params = {"hidden_dims": hidden_dims, "lr": lr, "dropout": dropout, "class_weight": class_weight}

                    mlp_candidate, dev_f1 = train_mlp(
                        X_train_scaled, y_train,
                        X_dev_scaled, y_dev,
                        hidden_dims=hidden_dims,
                        epochs=50,  # Fewer epochs for tuning
                        batch_size=64,
                        lr=lr,
                        dropout=dropout,
                        class_weight=class_weight,
                        patience=10,
                        device=device,
                    )

                    if dev_f1 > best_mlp_f1:
                        best_mlp_f1 = dev_f1
                        best_mlp = mlp_candidate
                        best_mlp_params = params
                        print(f"   [{config_idx}/{total_configs}] New best: F1={dev_f1:.4f} | {params}")

    print(f"   Best MLP params: {best_mlp_params}")
    print(f"   Best dev F1: {best_mlp_f1:.4f}")

    mlp = best_mlp
    results["MLP"] = {"eval_sets": {}, "best_dev_f1": best_mlp_f1, "best_params": best_mlp_params}

    mlp.eval()
    for eval_name, (X_eval, y_eval) in eval_sets.items():
        X_eval_scaled = scaler.transform(X_eval)
        X_eval_t = torch.tensor(X_eval_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = mlp(X_eval_t)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)

        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)
        prec = precision_score(y_eval, preds)
        rec = recall_score(y_eval, preds)

        results["MLP"]["eval_sets"][eval_name] = {
            "f1": f1, "acc": acc, "precision": prec, "recall": rec
        }

        if eval_name in SOTA:
            delta_f1 = f1 - SOTA[eval_name]["f1"]
            delta_acc = acc - SOTA[eval_name]["acc"]
            print(f"   {eval_name}: F1={f1:.4f} ({delta_f1:+.4f} vs SOTA), Acc={acc:.4f} ({delta_acc:+.4f} vs SOTA)")
        else:
            print(f"   {eval_name}: F1={f1:.4f}, Acc={acc:.4f}")

    # ==========================================================================
    # Train sklearn classifiers with hyperparameter tuning
    # ==========================================================================

    # Define classifiers with their parameter grids for tuning
    classifiers_with_params = {
        "LogisticRegression": {
            "model": LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42, n_jobs=-1),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10, 15]},
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(class_weight={0: 1, 1: 3}, random_state=42, n_jobs=-1),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10, 15]},
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
        },
        "SVM": {
            "model": SVC(class_weight={0: 1, 1: 3}, probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        },
        "KNN": {
            "model": KNeighborsClassifier(n_jobs=-1),
            "params": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        },
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        classifiers_with_params["XGBoost"] = {
            "model": XGBClassifier(scale_pos_weight=3, random_state=42, n_jobs=-1, verbosity=0),
            "params": {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        }

    # Add CatBoost if available
    if HAS_CATBOOST:
        classifiers_with_params["CatBoost"] = {
            "model": CatBoostClassifier(class_weights={0: 1, 1: 3}, random_seed=42, verbose=False),
            "params": {"iterations": [100, 200], "depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        }

    # Add LightGBM if available
    if HAS_LIGHTGBM:
        classifiers_with_params["LightGBM"] = {
            "model": LGBMClassifier(class_weight={0: 1, 1: 3}, random_state=42, n_jobs=-1, verbose=-1),
            "params": {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        }

    # Train classifiers with hyperparameter tuning on dev set
    for name, config in classifiers_with_params.items():
        print(f"\n🔧 Training {name} with hyperparameter tuning...")

        base_model = config["model"]
        param_grid = config["params"]

        # Grid search with cross-validation on combined train data
        # We use 3-fold CV for speed, scoring by F1
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train_scaled, y_train)

        best_clf = grid_search.best_estimator_
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   CV F1: {grid_search.best_score_:.4f}")

        results[name] = {"eval_sets": {}}

        results[name]["best_params"] = grid_search.best_params_
        results[name]["cv_score"] = grid_search.best_score_

        for eval_name, (X_eval, y_eval) in eval_sets.items():
            X_eval_scaled = scaler.transform(X_eval)
            preds = best_clf.predict(X_eval_scaled)

            f1 = f1_score(y_eval, preds)
            acc = accuracy_score(y_eval, preds)
            prec = precision_score(y_eval, preds)
            rec = recall_score(y_eval, preds)

            results[name]["eval_sets"][eval_name] = {
                "f1": f1, "acc": acc, "precision": prec, "recall": rec
            }

            # Compare to SOTA
            if eval_name in SOTA:
                delta_f1 = f1 - SOTA[eval_name]["f1"]
                delta_acc = acc - SOTA[eval_name]["acc"]
                print(f"   {eval_name}: F1={f1:.4f} ({delta_f1:+.4f} vs SOTA), Acc={acc:.4f} ({delta_acc:+.4f} vs SOTA)")
            else:
                print(f"   {eval_name}: F1={f1:.4f}, Acc={acc:.4f}")

    return results, scaler, mlp


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Force re-extraction of embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embeddings-only", action="store_true", help="Only use CLS embeddings, no LLM features")
    parser.add_argument("--llm-only", action="store_true", help="Only use LLM features, no embeddings")
    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING + LLM FEATURES CLASSIFIER TRAINING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data for all splits
    print("\n📊 Loading data...")
    splits = {}
    for split in ["train", "dev", "dev-test", "test"]:
        texts, labels, df = load_split_data(split)
        sentence_ids = df["Sentence_id"].to_list()
        splits[split] = {"texts": texts, "labels": labels, "sentence_ids": sentence_ids}
        print(f"   {split}: {len(texts)} samples, {sum(labels)} positive ({100*sum(labels)/len(labels):.1f}%)")

    # Extract or load embeddings
    print("\n🔮 Getting CLS embeddings...")
    use_cache = not args.no_cache
    for split in splits:
        if not args.llm_only:
            embeddings = get_or_extract_embeddings(split, splits[split]["texts"], use_cache, device)
            splits[split]["embeddings"] = embeddings
            print(f"   {split}: {embeddings.shape}")
        else:
            splits[split]["embeddings"] = None

    # Load LLM features
    print("\n📈 Loading LLM features...")
    for split in splits:
        if not args.embeddings_only:
            llm_features = load_llm_features(split, splits[split]["sentence_ids"])
            if llm_features is not None:
                print(f"   {split}: {llm_features.shape}")
            splits[split]["llm_features"] = llm_features
        else:
            splits[split]["llm_features"] = None

    # Combine features
    print("\n🔗 Combining features...")
    for split in splits:
        parts = []
        if splits[split]["embeddings"] is not None:
            parts.append(splits[split]["embeddings"])
        if splits[split]["llm_features"] is not None:
            parts.append(splits[split]["llm_features"])

        if parts:
            combined = np.hstack(parts)
            splits[split]["X"] = combined
            print(f"   {split}: {combined.shape}")
        else:
            print(f"   ⚠️  {split}: No features available!")
            return

    # Prepare training/eval sets
    X_train, y_train = splits["train"]["X"], splits["train"]["labels"]
    X_dev, y_dev = splits["dev"]["X"], splits["dev"]["labels"]

    eval_sets = {
        "dev": (X_dev, y_dev),
        "dev-test": (splits["dev-test"]["X"], splits["dev-test"]["labels"]),
        "test": (splits["test"]["X"], splits["test"]["labels"]),
    }

    # Train and evaluate
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIERS")
    print("=" * 70)

    results, scaler, mlp = train_and_evaluate_classifiers(
        X_train, y_train, X_dev, y_dev, eval_sets, device
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nDev-Test Results (SOTA: F1=0.932, Acc=0.955):")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["eval_sets"]["dev-test"]["f1"]):
        metrics = res["eval_sets"]["dev-test"]
        delta = metrics["f1"] - SOTA["dev-test"]["f1"]
        marker = "👑" if metrics["f1"] == max(r["eval_sets"]["dev-test"]["f1"] for r in results.values()) else "  "
        print(f"   {marker} {name}: F1={metrics['f1']:.4f} ({delta:+.4f}), Acc={metrics['acc']:.4f}")

    print("\nTest Results (SOTA: F1=0.82, Acc=0.905):")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["eval_sets"]["test"]["f1"]):
        metrics = res["eval_sets"]["test"]
        delta = metrics["f1"] - SOTA["test"]["f1"]
        marker = "👑" if metrics["f1"] == max(r["eval_sets"]["test"]["f1"] for r in results.values()) else "  "
        print(f"   {marker} {name}: F1={metrics['f1']:.4f} ({delta:+.4f}), Acc={metrics['acc']:.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "classifier_results.json", "w") as f:
        # Convert numpy types
        json_results = {}
        for name, res in results.items():
            json_results[name] = {
                "eval_sets": {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in res["eval_sets"].items()
                }
            }
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to {OUTPUT_DIR / 'classifier_results.json'}")


if __name__ == "__main__":
    main()
