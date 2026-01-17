"""
CheckworthinessPredictor: Ensemble model for claim checkworthiness classification.

Combines:
1. DeBERTa ensemble (seeds 0 + 456) with temperature scaling
2. LLM-based features (checkability, verifiability, harm potential)
3. XGBoost classifier on LLM features
4. Late fusion: 0.6×deberta + 0.4×llm

Architecture:
    claim_text
        ├──→ DeBERTa (seed 0) ──┬──→ temp_scale(T=0.7) ──→ avg ──→ deberta_prob
        └──→ DeBERTa (seed 456) ┘                                       │
                                                                        │
        ├──→ LLM (checkability) ──┐                                     │
        ├──→ LLM (verifiability) ─┼──→ 24 features ──→ XGBoost ──→ llm_prob
        └──→ LLM (harm potential) ┘                                     │
                                                                        │
                    0.6×deberta_prob + 0.4×llm_prob ──→ threshold ≥ 0.5

Usage:
    predictor = CheckworthinessPredictor.from_pretrained(
        deberta_dirs=["path/to/seed_0", "path/to/seed_456"],
        xgboost_path="path/to/xgboost.pkl",
    )

    result = await predictor.predict("Biden claims 10M jobs created")
    print(result.prediction, result.fused_probability)
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_torch = None
_transformers = None
_xgboost = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _transformers = {"AutoModel": AutoModelForSequenceClassification, "AutoTokenizer": AutoTokenizer}
    return _transformers


def _get_xgboost():
    global _xgboost
    if _xgboost is None:
        from xgboost import XGBClassifier
        _xgboost = XGBClassifier
    return _xgboost


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CheckworthinessConfig:
    """Configuration for the checkworthiness predictor."""

    # DeBERTa settings
    deberta_dirs: list[str] = field(default_factory=lambda: [
        "lambda_backup/ubuntu/ensemble_results/seed_0/deberta-v3-large/best_model",
        "lambda_backup/ubuntu/ensemble_results/seed_456/deberta-v3-large/best_model",
    ])
    temperature: float = 0.7  # Temperature scaling for DeBERTa
    device: str = "mps"  # cpu, cuda, mps

    # XGBoost settings
    xgboost_path: str | None = None  # Path to saved XGBoost model
    train_xgboost_on_init: bool = True  # Train XGBoost if no saved model
    llm_features_dir: str = "data/processed/CT24_llm_features_v4"

    # Fusion settings
    deberta_weight: float = 0.6  # Weight for DeBERTa in late fusion
    threshold: float = 0.5  # Classification threshold

    # LLM settings for feature generation
    llm_model: str = "gpt-4o-mini"  # Model for LLM features
    llm_rate_limit_rpm: int = 30  # Rate limit for LLM calls

    @classmethod
    def from_dict(cls, config: dict) -> "CheckworthinessConfig":
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


# LLM feature columns used by XGBoost (from combine_deberta_llm_features.py)
LLM_FEATURE_COLS = [
    # Scores
    "check_score", "verif_score", "harm_score",
    # Entropy
    "check_entropy", "verif_entropy", "harm_entropy",
    # P(Yes)
    "check_p_yes", "verif_p_yes", "harm_p_yes",
    # Margin
    "check_margin_p", "verif_margin_p", "harm_margin_p",
    # Predictions (as 0/1)
    "check_prediction", "verif_prediction", "harm_prediction",
    # Cross-module
    "score_variance", "score_max_diff", "yes_vote_count", "unanimous_yes", "unanimous_no",
    # Harm sub-dimensions
    "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
]


# =============================================================================
# Result Schema
# =============================================================================

@dataclass
class CheckworthinessOutput:
    """Output from the checkworthiness predictor."""

    claim_text: str

    # Individual probabilities
    deberta_prob: float  # Ensemble DeBERTa probability
    llm_prob: float  # XGBoost on LLM features probability
    fused_prob: float  # Late fusion probability

    # Final prediction
    prediction: str  # "Yes" or "No"
    is_checkworthy: bool  # Boolean convenience field

    # Component details (for explainability)
    deberta_probs_by_seed: dict[int, float] = field(default_factory=dict)
    llm_features: dict[str, float] = field(default_factory=dict)

    # Module-level assessments (from LLM)
    checkability_score: float | None = None
    verifiability_score: float | None = None
    harm_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "deberta_prob": self.deberta_prob,
            "llm_prob": self.llm_prob,
            "fused_prob": self.fused_prob,
            "prediction": self.prediction,
            "is_checkworthy": self.is_checkworthy,
            "checkability_score": self.checkability_score,
            "verifiability_score": self.verifiability_score,
            "harm_score": self.harm_score,
        }


# =============================================================================
# DeBERTa Inference
# =============================================================================

class DeBERTaEnsemble:
    """Ensemble of DeBERTa models with temperature scaling."""

    def __init__(self, model_dirs: list[str], temperature: float = 0.7, device: str = "mps"):
        self.model_dirs = model_dirs
        self.temperature = temperature
        self.device = device

        self._models = []
        self._tokenizers = []
        self._loaded = False

    def load(self) -> None:
        """Load all DeBERTa models (lazy loading)."""
        if self._loaded:
            return

        torch = _get_torch()
        transformers = _get_transformers()

        for model_dir in self.model_dirs:
            model_path = Path(model_dir)
            if not model_path.exists():
                logger.warning(f"DeBERTa model not found: {model_path}")
                continue

            logger.info(f"Loading DeBERTa from {model_path}")

            tokenizer = transformers["AutoTokenizer"].from_pretrained(str(model_path))
            model = transformers["AutoModel"].from_pretrained(str(model_path))
            model.to(self.device)
            model.eval()

            self._tokenizers.append(tokenizer)
            self._models.append(model)

        if not self._models:
            raise ValueError("No DeBERTa models loaded!")

        self._loaded = True
        logger.info(f"Loaded {len(self._models)} DeBERTa models")

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        epsilon = 1e-8
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def predict(self, texts: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """
        Predict probabilities for a batch of texts.

        Returns:
            Tuple of (ensemble_probs, probs_by_seed)
        """
        if not self._loaded:
            self.load()

        torch = _get_torch()

        all_probs = []
        probs_by_seed = {}

        for i, (model, tokenizer) in enumerate(zip(self._models, self._tokenizers)):
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(checkworthy)

            # Apply temperature scaling
            probs_scaled = self._apply_temperature(probs)

            all_probs.append(probs_scaled)
            probs_by_seed[i] = probs_scaled

        # Ensemble: average across seeds
        ensemble_probs = np.mean(all_probs, axis=0)

        return ensemble_probs, probs_by_seed

    def predict_single(self, text: str) -> tuple[float, dict[int, float]]:
        """Predict for a single text."""
        probs, probs_by_seed = self.predict([text])
        return float(probs[0]), {k: float(v[0]) for k, v in probs_by_seed.items()}


# =============================================================================
# LLM Feature Extractor
# =============================================================================

class LLMFeatureExtractor:
    """Extract LLM-based features for checkworthiness."""

    def __init__(self, model: str = "gpt-4o-mini", rate_limit_rpm: int = 30):
        self.model = model
        self.rate_limit_rpm = rate_limit_rpm
        self._baseline = None

    def _get_baseline(self):
        """Lazy load the prompting baseline."""
        if self._baseline is None:
            from .prompting_baseline import PromptingBaseline
            from .config import get_model_config

            config = get_model_config(self.model)
            self._baseline = PromptingBaseline(config)
        return self._baseline

    async def extract_features(self, text: str) -> dict[str, float]:
        """
        Extract LLM features for a single claim.

        Returns dict with all 24 LLM feature columns.
        """
        baseline = self._get_baseline()

        # Run the three assessments
        result = baseline.assess(text)

        # Extract features
        features = {
            # Scores (0-100)
            "check_score": result.checkability.confidence,
            "verif_score": result.verifiability.confidence,
            "harm_score": result.harm_potential.confidence,

            # Predictions (0/1)
            "check_prediction": 1 if result.checkability.is_checkable else 0,
            "verif_prediction": 1 if result.verifiability.is_verifiable else 0,
            "harm_prediction": 1 if result.harm_potential.has_harm_potential else 0,

            # Harm sub-dimensions
            "harm_social_fragmentation": result.harm_potential.sub_scores.social_fragmentation if result.harm_potential.sub_scores else 0,
            "harm_spurs_action": result.harm_potential.sub_scores.spurs_action if result.harm_potential.sub_scores else 0,
            "harm_believability": result.harm_potential.sub_scores.believability if result.harm_potential.sub_scores else 0,
            "harm_exploitativeness": result.harm_potential.sub_scores.exploitativeness if result.harm_potential.sub_scores else 0,
        }

        # Compute derived features
        scores = [features["check_score"], features["verif_score"], features["harm_score"]]
        features["score_variance"] = float(np.var(scores))
        features["score_max_diff"] = max(scores) - min(scores)

        predictions = [features["check_prediction"], features["verif_prediction"], features["harm_prediction"]]
        features["yes_vote_count"] = sum(predictions)
        features["unanimous_yes"] = 1 if all(p == 1 for p in predictions) else 0
        features["unanimous_no"] = 1 if all(p == 0 for p in predictions) else 0

        # Logprob-based features (if available)
        # These require logprobs which may not be available for all models
        for module in ["check", "verif", "harm"]:
            features[f"{module}_entropy"] = 0.0  # Default
            features[f"{module}_p_yes"] = features[f"{module}_score"] / 100.0
            features[f"{module}_margin_p"] = abs(features[f"{module}_score"] - 50) / 50.0

        return features

    async def extract_features_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Extract features for multiple texts with rate limiting."""
        import asyncio

        delay = 60.0 / self.rate_limit_rpm
        results = []

        for i, text in enumerate(texts):
            if i > 0:
                await asyncio.sleep(delay)

            try:
                features = await self.extract_features(text)
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features for text {i}: {e}")
                # Return default features
                results.append({col: 0.0 for col in LLM_FEATURE_COLS})

        return results


# =============================================================================
# XGBoost Classifier
# =============================================================================

class LLMXGBoostClassifier:
    """XGBoost classifier trained on LLM features."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None
        self._scaler = None

    def load(self) -> bool:
        """Load saved model. Returns True if successful."""
        if self.model_path and Path(self.model_path).exists():
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
                self._model = saved["model"]
                self._scaler = saved["scaler"]
            logger.info(f"Loaded XGBoost from {self.model_path}")
            return True
        return False

    def train(self, features_dir: str) -> None:
        """Train XGBoost on CT24 training data."""
        import polars as pl
        from sklearn.preprocessing import StandardScaler

        XGBClassifier = _get_xgboost()

        # Load training data
        train_path = Path(features_dir) / "train_llm_features.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Training features not found: {train_path}")

        train_df = pl.read_parquet(train_path)

        # Load labels
        labels_path = Path("data/processed/CT24_clean/CT24_train_clean.parquet")
        if labels_path.exists():
            labels_df = pl.read_parquet(labels_path)
            y_train = np.array([1 if l == "Yes" else 0 for l in labels_df["class_label"].to_list()])
        else:
            raise FileNotFoundError(f"Training labels not found: {labels_path}")

        # Extract features
        available_cols = [c for c in LLM_FEATURE_COLS if c in train_df.columns]
        X_train = train_df.select(available_cols).to_numpy().astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0)

        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)

        # Train XGBoost
        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._model.fit(X_train_scaled, y_train)

        logger.info(f"Trained XGBoost on {len(y_train)} samples")

    def save(self, path: str) -> None:
        """Save the trained model."""
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)
        logger.info(f"Saved XGBoost to {path}")

    def predict_proba(self, features: list[dict[str, float]]) -> np.ndarray:
        """Predict probabilities for a list of feature dicts."""
        if self._model is None:
            raise ValueError("Model not loaded or trained!")

        # Convert to array
        X = np.array([[f.get(col, 0.0) for col in LLM_FEATURE_COLS] for f in features])
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)

        # Scale
        X_scaled = self._scaler.transform(X)

        # Predict
        return self._model.predict_proba(X_scaled)[:, 1]


# =============================================================================
# Main Predictor Class
# =============================================================================

class CheckworthinessPredictor:
    """
    Ensemble predictor combining DeBERTa + LLM features + XGBoost.

    Usage:
        predictor = CheckworthinessPredictor(config)
        predictor.load()  # Load models

        result = await predictor.predict("Biden claims economy is strong")
        print(result.prediction, result.fused_prob)
    """

    def __init__(self, config: CheckworthinessConfig | None = None):
        self.config = config or CheckworthinessConfig()

        # Components (lazy initialized)
        self._deberta = DeBERTaEnsemble(
            model_dirs=self.config.deberta_dirs,
            temperature=self.config.temperature,
            device=self.config.device,
        )
        self._llm_extractor = LLMFeatureExtractor(
            model=self.config.llm_model,
            rate_limit_rpm=self.config.llm_rate_limit_rpm,
        )
        self._xgboost = LLMXGBoostClassifier(model_path=self.config.xgboost_path)

        self._loaded = False

    def load(self) -> None:
        """Load all models."""
        # Load DeBERTa
        self._deberta.load()

        # Load or train XGBoost
        if not self._xgboost.load():
            if self.config.train_xgboost_on_init:
                logger.info("Training XGBoost on CT24 data...")
                self._xgboost.train(self.config.llm_features_dir)

                # Save for next time
                if self.config.xgboost_path:
                    self._xgboost.save(self.config.xgboost_path)
            else:
                raise ValueError("XGBoost model not found and train_xgboost_on_init=False")

        self._loaded = True
        logger.info("CheckworthinessPredictor loaded")

    def _late_fusion(self, deberta_prob: float, llm_prob: float) -> float:
        """Apply late fusion."""
        return self.config.deberta_weight * deberta_prob + (1 - self.config.deberta_weight) * llm_prob

    async def predict(self, text: str) -> CheckworthinessOutput:
        """Predict checkworthiness for a single claim."""
        if not self._loaded:
            self.load()

        # DeBERTa prediction
        deberta_prob, probs_by_seed = self._deberta.predict_single(text)

        # LLM features
        llm_features = await self._llm_extractor.extract_features(text)

        # XGBoost prediction
        llm_probs = self._xgboost.predict_proba([llm_features])
        llm_prob = float(llm_probs[0])

        # Late fusion
        fused_prob = self._late_fusion(deberta_prob, llm_prob)

        # Final prediction
        is_checkworthy = fused_prob >= self.config.threshold
        prediction = "Yes" if is_checkworthy else "No"

        return CheckworthinessOutput(
            claim_text=text,
            deberta_prob=deberta_prob,
            llm_prob=llm_prob,
            fused_prob=fused_prob,
            prediction=prediction,
            is_checkworthy=is_checkworthy,
            deberta_probs_by_seed=probs_by_seed,
            llm_features=llm_features,
            checkability_score=llm_features.get("check_score"),
            verifiability_score=llm_features.get("verif_score"),
            harm_score=llm_features.get("harm_score"),
        )

    async def predict_batch(
        self,
        texts: list[str],
        show_progress: bool = True
    ) -> list[CheckworthinessOutput]:
        """Predict for multiple claims."""
        if not self._loaded:
            self.load()

        results = []

        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing claim {i+1}/{len(texts)}")

            result = await self.predict(text)
            results.append(result)

        return results

    def predict_deberta_only(self, texts: list[str]) -> np.ndarray:
        """Predict using only DeBERTa (faster, no LLM calls)."""
        if not self._loaded:
            self._deberta.load()

        probs, _ = self._deberta.predict(texts)
        return probs

    @classmethod
    def from_config(cls, config_path: str) -> "CheckworthinessPredictor":
        """Load from a config file."""
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        config = CheckworthinessConfig.from_dict(config_dict.get("checkworthiness", {}))
        return cls(config)
