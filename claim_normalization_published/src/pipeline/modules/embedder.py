"""
Embedder Module

Generates sentence embeddings for tweets using sentence-transformers.
Supports batching, caching, and multiple devices (cpu, cuda, mps).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """Configuration for the embedder."""

    enabled: bool = True
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True
    device: str = "mps"  # "cpu", "cuda", "mps"
    cache_embeddings: bool = True
    cache_file: str = "embeddings_cache.npy"
    show_progress: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "EmbedderConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class Embedder:
    """
    Generates sentence embeddings for text.

    Usage:
        embedder = Embedder(config)
        embeddings = embedder.embed(texts)
        # or
        df = embedder.apply(df, text_column="tweet_enriched")
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            logger.info(f"Model loaded on device: {self.config.device}")
            logger.info(f"Embedding dimension: {self._model.get_sentence_embedding_dimension()}")

        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not self.config.enabled:
            logger.warning("Embedder disabled, returning empty array")
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts):,} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=self.config.show_progress,
            convert_to_numpy=True,
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(
            [text],
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )[0]

    def apply(
        self,
        df: pl.DataFrame,
        text_column: str = "tweet_enriched",
        filter_column: str | None = "passes_claim_gate",
        output_dir: Path | None = None,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Apply embedder to dataframe.

        Args:
            df: Input dataframe
            text_column: Column containing text to embed
            filter_column: If provided, only embed rows where this column is True
            output_dir: Directory to cache embeddings

        Returns:
            Tuple of (dataframe with embedding indices, embeddings array)
        """
        if not self.config.enabled:
            logger.info("Embedder disabled")
            return df, np.array([])

        # Determine which rows to embed
        if filter_column and filter_column in df.columns:
            mask = df[filter_column].to_numpy()
            texts_to_embed = [t for t, m in zip(df[text_column].to_list(), mask) if m]
            embed_indices = np.where(mask)[0]
            logger.info(f"Embedding {len(texts_to_embed):,} texts (filtered by {filter_column})")
        else:
            texts_to_embed = df[text_column].to_list()
            embed_indices = np.arange(len(df))
            logger.info(f"Embedding all {len(texts_to_embed):,} texts")

        # Check cache
        cache_path = None
        if self.config.cache_embeddings and output_dir:
            cache_path = Path(output_dir) / self.config.cache_file
            if cache_path.exists():
                logger.info(f"Loading cached embeddings from {cache_path}")
                embeddings = np.load(cache_path)
                if len(embeddings) == len(texts_to_embed):
                    logger.info(f"Loaded {len(embeddings):,} cached embeddings")
                    return df, embeddings
                else:
                    logger.warning("Cache size mismatch, regenerating embeddings")

        # Generate embeddings
        embeddings = self.embed(texts_to_embed)

        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to {cache_path}")

        # Add embedding index to dataframe
        embedding_idx = np.full(len(df), -1, dtype=np.int32)
        embedding_idx[embed_indices] = np.arange(len(texts_to_embed))

        df = df.with_columns(pl.Series(name="embedding_idx", values=embedding_idx))

        return df, embeddings

    def get_stats(self, embeddings: np.ndarray) -> dict:
        """Get statistics about embeddings."""
        if len(embeddings) == 0:
            return {"count": 0}

        norms = np.linalg.norm(embeddings, axis=1)
        return {
            "count": len(embeddings),
            "dimension": embeddings.shape[1],
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        }


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Compute similarity matrix
    return np.dot(normalized, normalized.T)
