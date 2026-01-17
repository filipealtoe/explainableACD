"""
Clusterer Module

Online incremental clustering for tweet embeddings.
Each embedding is compared to existing cluster centroids using cosine similarity.
If similarity exceeds threshold, the tweet joins that cluster and updates the centroid.
Otherwise, a new cluster is created.

Cluster state can be persisted and reloaded for incremental processing.
"""

import gzip
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ClusterState:
    """State of the clustering process."""

    centroids: np.ndarray  # Shape: (n_clusters, embedding_dim)
    sizes: list[int]  # Number of tweets per cluster
    representative_ids: list[list[str]]  # Tweet IDs per cluster (for reference)

    @classmethod
    def empty(cls, embedding_dim: int) -> "ClusterState":
        """Create empty cluster state."""
        return cls(
            centroids=np.empty((0, embedding_dim)),
            sizes=[],
            representative_ids=[],
        )

    @property
    def n_clusters(self) -> int:
        return len(self.sizes)

    def save(self, path: Path) -> None:
        """Save cluster state to compressed pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved cluster state to {path} ({self.n_clusters} clusters)")

    @classmethod
    def load(cls, path: Path) -> "ClusterState":
        """Load cluster state from compressed pickle."""
        with gzip.open(path, "rb") as f:
            state = pickle.load(f)
        logger.info(f"Loaded cluster state from {path} ({state.n_clusters} clusters)")
        return state


@dataclass
class ClustererConfig:
    """Configuration for the clusterer."""

    enabled: bool = True
    algorithm: str = "online_cosine"
    similarity_threshold: float = 0.75
    min_cluster_size: int = 3
    max_clusters: int = 10000
    persist_state: bool = True
    state_file: str = "cluster_state.pkl.gz"
    # Maximum representative tweets to store per cluster
    max_representatives_per_cluster: int = 10

    @classmethod
    def from_dict(cls, config: dict) -> "ClustererConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class OnlineCosineClustering:
    """
    Online incremental clustering using cosine similarity.

    For each new embedding:
    1. Compute cosine similarity to all existing centroids
    2. If max similarity > threshold, assign to that cluster and update centroid
    3. Otherwise, create a new cluster with this embedding as centroid
    """

    def __init__(self, config: ClustererConfig, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self.state = ClusterState.empty(embedding_dim)

    def _cosine_similarity(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embedding and all centroids."""
        if self.state.n_clusters == 0:
            return np.array([])

        # Assume embeddings are already normalized
        # cosine_sim = dot product for normalized vectors
        return np.dot(self.state.centroids, embedding)

    def _update_centroid(self, cluster_idx: int, embedding: np.ndarray) -> None:
        """Update cluster centroid using running average."""
        n = self.state.sizes[cluster_idx]
        # Running average: new_centroid = (old_centroid * n + new_embedding) / (n + 1)
        self.state.centroids[cluster_idx] = (self.state.centroids[cluster_idx] * n + embedding) / (n + 1)
        # Re-normalize
        self.state.centroids[cluster_idx] /= np.linalg.norm(self.state.centroids[cluster_idx])
        self.state.sizes[cluster_idx] += 1

    def _create_cluster(self, embedding: np.ndarray, tweet_id: str) -> int:
        """Create a new cluster with this embedding as centroid."""
        if self.state.n_clusters >= self.config.max_clusters:
            logger.warning(f"Max clusters ({self.config.max_clusters}) reached, assigning to nearest")
            # Fallback: assign to nearest cluster
            similarities = self._cosine_similarity(embedding)
            return int(np.argmax(similarities))

        # Add new centroid
        self.state.centroids = np.vstack([self.state.centroids, embedding.reshape(1, -1)])
        self.state.sizes.append(1)
        self.state.representative_ids.append([tweet_id])

        return self.state.n_clusters - 1

    def assign(self, embedding: np.ndarray, tweet_id: str) -> tuple[int, float]:
        """
        Assign a single embedding to a cluster.

        Returns:
            (cluster_id, similarity_score)
        """
        if self.state.n_clusters == 0:
            # First embedding, create first cluster
            cluster_id = self._create_cluster(embedding, tweet_id)
            return cluster_id, 1.0

        # Compute similarities to all centroids
        similarities = self._cosine_similarity(embedding)
        max_sim = float(np.max(similarities))
        best_cluster = int(np.argmax(similarities))

        if max_sim >= self.config.similarity_threshold:
            # Assign to existing cluster
            self._update_centroid(best_cluster, embedding)
            # Add to representatives (keep limited number)
            if len(self.state.representative_ids[best_cluster]) < self.config.max_representatives_per_cluster:
                self.state.representative_ids[best_cluster].append(tweet_id)
            return best_cluster, max_sim
        else:
            # Create new cluster
            cluster_id = self._create_cluster(embedding, tweet_id)
            return cluster_id, 1.0

    def assign_batch(
        self,
        embeddings: np.ndarray,
        tweet_ids: list[str],
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign multiple embeddings to clusters.

        Returns:
            (cluster_ids, similarity_scores)
        """
        n = len(embeddings)
        cluster_ids = np.zeros(n, dtype=np.int32)
        similarities = np.zeros(n, dtype=np.float32)

        for i, (emb, tid) in enumerate(zip(embeddings, tweet_ids)):
            cluster_ids[i], similarities[i] = self.assign(emb, tid)

            if show_progress and (i + 1) % 5000 == 0:
                logger.info(f"  Clustered {i + 1:,}/{n:,} tweets, {self.state.n_clusters:,} clusters")

        return cluster_ids, similarities


class Clusterer:
    """
    Main clusterer interface.

    Usage:
        clusterer = Clusterer(config, embedding_dim=384)
        df = clusterer.apply(df, embeddings)
    """

    def __init__(self, config: ClustererConfig, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self._algorithm = None

    @property
    def algorithm(self) -> OnlineCosineClustering:
        """Get or create the clustering algorithm."""
        if self._algorithm is None:
            if self.config.algorithm == "online_cosine":
                self._algorithm = OnlineCosineClustering(self.config, self.embedding_dim)
            else:
                raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        return self._algorithm

    def load_state(self, path: Path) -> None:
        """Load cluster state from file."""
        if path.exists():
            self.algorithm.state = ClusterState.load(path)

    def save_state(self, path: Path) -> None:
        """Save cluster state to file."""
        self.algorithm.state.save(path)

    def apply(
        self,
        df: pl.DataFrame,
        embeddings: np.ndarray,
        id_column: str = "id",
        embedding_idx_column: str = "embedding_idx",
        output_dir: Path | None = None,
    ) -> pl.DataFrame:
        """
        Apply clustering to dataframe.

        Args:
            df: Input dataframe
            embeddings: Embedding array (only for tweets that passed claim gate)
            id_column: Column with tweet IDs
            embedding_idx_column: Column mapping rows to embedding indices
            output_dir: Directory to save/load cluster state

        Returns:
            DataFrame with cluster_id and cluster_similarity columns
        """
        if not self.config.enabled:
            logger.info("Clusterer disabled")
            return df.with_columns(
                [
                    pl.lit(-1).alias("cluster_id"),
                    pl.lit(0.0).alias("cluster_similarity"),
                ]
            )

        # Load existing state if available
        if output_dir and self.config.persist_state:
            state_path = Path(output_dir) / self.config.state_file
            if state_path.exists():
                self.load_state(state_path)

        # Get tweet IDs for embeddings
        embedding_indices = df[embedding_idx_column].to_numpy()
        tweet_ids = df[id_column].to_list()

        # Map embedding indices to tweet IDs
        embedded_mask = embedding_indices >= 0
        embedded_tweet_ids = [tweet_ids[i] for i in range(len(df)) if embedded_mask[i]]

        logger.info(f"Clustering {len(embeddings):,} embeddings...")

        # Assign embeddings to clusters
        cluster_ids, similarities = self.algorithm.assign_batch(embeddings, embedded_tweet_ids)

        logger.info(f"Created {self.algorithm.state.n_clusters:,} clusters")

        # Map results back to dataframe
        df_cluster_ids = np.full(len(df), -1, dtype=np.int32)
        df_similarities = np.zeros(len(df), dtype=np.float32)

        embedded_indices = np.where(embedded_mask)[0]
        df_cluster_ids[embedded_indices] = cluster_ids
        df_similarities[embedded_indices] = similarities

        df = df.with_columns(
            [
                pl.Series(name="cluster_id", values=df_cluster_ids),
                pl.Series(name="cluster_similarity", values=df_similarities),
            ]
        )

        # Save state
        if output_dir and self.config.persist_state:
            state_path = Path(output_dir) / self.config.state_file
            self.save_state(state_path)

        return df

    def get_stats(self) -> dict:
        """Get clustering statistics."""
        state = self.algorithm.state
        if state.n_clusters == 0:
            return {"n_clusters": 0}

        sizes = np.array(state.sizes)
        return {
            "n_clusters": state.n_clusters,
            "total_tweets": int(sizes.sum()),
            "avg_cluster_size": float(sizes.mean()),
            "median_cluster_size": float(np.median(sizes)),
            "max_cluster_size": int(sizes.max()),
            "min_cluster_size": int(sizes.min()),
            "clusters_size_1": int((sizes == 1).sum()),
            "clusters_size_gte_min": int((sizes >= self.config.min_cluster_size).sum()),
        }

    def get_cluster_info(self, cluster_id: int) -> dict:
        """Get information about a specific cluster."""
        state = self.algorithm.state
        if cluster_id < 0 or cluster_id >= state.n_clusters:
            return {}

        return {
            "cluster_id": cluster_id,
            "size": state.sizes[cluster_id],
            "representative_ids": state.representative_ids[cluster_id],
            "centroid_norm": float(np.linalg.norm(state.centroids[cluster_id])),
        }
