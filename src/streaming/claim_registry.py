"""
ClaimRegistry: Track cluster → claim mappings with deduplication.

This module maintains a registry of normalized claims, handling:
1. Cluster-to-claim mapping (many clusters can map to one claim)
2. Claim deduplication via embedding similarity
3. LLM normalization via Groq async adapter
4. Statistics tracking for evaluation

Deduplication logic:
- When a new cluster is normalized, compute its claim embedding
- Compare against existing claim embeddings (cosine similarity)
- If similarity > threshold (default 0.85), merge with existing claim
- Otherwise, create a new claim entry

Usage:
    registry = ClaimRegistry(normalizer, embedder, dedup_threshold=0.85)
    claim_id = await registry.get_or_create_claim(cluster_id=5, tweets=["..."])
    registry.update_claim_stats(claim_id, timestamp, count=50, engagement=1000)
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.streaming.schemas import ClaimInfo

if TYPE_CHECKING:
    from src.streaming.llm_normalizer import LLMNormalizerAdapter
    from src.pipeline.modules.embedder import Embedder

logger = logging.getLogger(__name__)

# Lazy import for YAKE keyword extractor
_yake_extractor = None


def _get_yake_extractor(
    language: str = "en",
    max_ngram_size: int = 2,
    deduplication_threshold: float = 0.9,
    num_keywords: int = 10,
):
    """
    Get or create YAKE keyword extractor (lazy initialization).

    YAKE (Yet Another Keyword Extractor) is unsupervised and doesn't need
    a pre-trained corpus, making it ideal for streaming applications.
    """
    global _yake_extractor
    if _yake_extractor is None:
        try:
            import yake
            _yake_extractor = yake.KeywordExtractor(
                lan=language,
                n=max_ngram_size,
                dedupLim=deduplication_threshold,
                top=num_keywords,
                features=None,  # Use default features
            )
        except ImportError:
            logger.warning("YAKE not installed. Install with: pip install yake")
            return None
    return _yake_extractor


def extract_keywords(
    texts: list[str],
    max_keywords: int = 10,
    min_keyword_length: int = 3,
) -> list[str]:
    """
    Extract keywords from a list of texts using YAKE.

    Args:
        texts: List of text strings (e.g., tweets from a cluster)
        max_keywords: Maximum number of keywords to return
        min_keyword_length: Minimum length of keywords

    Returns:
        List of keywords, sorted by relevance (most relevant first)
    """
    extractor = _get_yake_extractor(num_keywords=max_keywords * 2)  # Extract more, then filter
    if extractor is None:
        return []

    # Combine texts into a single document
    combined = " ".join(texts)

    # Clean text: remove URLs, mentions, excess whitespace
    combined = re.sub(r"https?://\S+", "", combined)
    combined = re.sub(r"@\w+", "", combined)
    combined = re.sub(r"#", "", combined)  # Keep hashtag words
    combined = re.sub(r"\s+", " ", combined).strip()

    if not combined or len(combined) < 20:
        return []

    try:
        # YAKE returns list of (keyword, score) tuples, lower score = more relevant
        keywords = extractor.extract_keywords(combined)

        # Filter by length and return just the keywords
        filtered = [
            kw for kw, score in keywords
            if len(kw) >= min_keyword_length
            and kw.lower() not in {"user", "https", "http", "the", "and", "for", "that", "this", "with"}
        ]

        return filtered[:max_keywords]
    except Exception as e:
        logger.debug(f"Keyword extraction failed: {e}")
        return []


@dataclass
class ClaimRegistryConfig:
    """Configuration for the ClaimRegistry."""

    enabled: bool = True
    dedup_threshold: float = 0.85  # Cosine similarity threshold for deduplication
    max_tweets_for_normalization: int = 5
    min_cluster_size: int = 10  # Minimum cluster size to normalize

    @classmethod
    def from_dict(cls, config: dict) -> "ClaimRegistryConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class ClaimRegistry:
    """
    Registry for normalized claims with deduplication.

    Maintains:
    - claims: dict[str, ClaimInfo] - claim_id to claim info
    - cluster_to_claim: dict[int, str] - cluster_id to claim_id mapping
    - claim_embeddings: np.ndarray - embeddings for similarity search
    """

    def __init__(
        self,
        normalizer: "LLMNormalizerAdapter",
        embedder: "Embedder",
        config: ClaimRegistryConfig | None = None,
    ):
        """
        Initialize the claim registry.

        Args:
            normalizer: LLM normalizer adapter for claim normalization
            embedder: Embedder for computing claim embeddings
            config: Configuration object
        """
        self.normalizer = normalizer
        self.embedder = embedder
        self.config = config or ClaimRegistryConfig()

        # Storage
        self.claims: dict[str, ClaimInfo] = {}
        self.cluster_to_claim: dict[int, str] = {}

        # Claim embeddings for similarity search
        self._claim_ids: list[str] = []  # Ordered list of claim IDs
        self._claim_embeddings: np.ndarray | None = None  # Shape: (n_claims, embedding_dim)

        # Statistics
        self._normalizations = 0
        self._deduplications = 0
        self._new_claims = 0

    def _compute_similarity(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """
        Find the most similar existing claim.

        Args:
            embedding: Embedding of the new claim

        Returns:
            Tuple of (claim_id, similarity) if found, (None, 0.0) otherwise
        """
        if self._claim_embeddings is None or len(self._claim_embeddings) == 0:
            return None, 0.0

        # Normalize the query embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # Compute cosine similarity with all existing claims
        # Embeddings should already be normalized, but normalize again to be safe
        norms = np.linalg.norm(self._claim_embeddings, axis=1, keepdims=True)
        normalized_embeddings = self._claim_embeddings / (norms + 1e-10)

        similarities = normalized_embeddings @ embedding

        # Find the most similar
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        return self._claim_ids[max_idx], max_sim

    def _add_claim_embedding(self, claim_id: str, embedding: np.ndarray) -> None:
        """Add a new claim embedding to the index."""
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        self._claim_ids.append(claim_id)

        if self._claim_embeddings is None:
            self._claim_embeddings = embedding.reshape(1, -1)
        else:
            self._claim_embeddings = np.vstack([self._claim_embeddings, embedding])

    async def get_or_create_claim(
        self,
        cluster_id: int,
        tweets: list[str],
        timestamp: datetime | None = None,
    ) -> str:
        """
        Get existing claim for a cluster or create a new one.

        This is the main entry point for claim registration. It:
        1. Checks if cluster is already mapped
        2. Normalizes tweets via Groq LLM
        3. Checks for duplicate claims via embedding similarity
        4. Creates new claim or maps to existing

        Args:
            cluster_id: Cluster identifier
            tweets: List of tweet texts from the cluster
            timestamp: Optional timestamp for first_seen

        Returns:
            claim_id (str)
        """
        # Check if cluster is already mapped
        if cluster_id in self.cluster_to_claim:
            return self.cluster_to_claim[cluster_id]

        if not self.config.enabled:
            # Create placeholder claim without normalization
            claim_id = str(uuid.uuid4())
            self.claims[claim_id] = ClaimInfo(
                claim_id=claim_id,
                claim_text=f"[Cluster {cluster_id}]",
                first_seen=timestamp or datetime.now(),
                cluster_ids=[cluster_id],
            )
            self.cluster_to_claim[cluster_id] = claim_id
            return claim_id

        # Normalize via LLM
        claim_text = await self.normalizer.normalize_claim(cluster_id, tweets)
        self._normalizations += 1

        # Compute embedding for deduplication
        claim_embedding = self.embedder.embed_single(claim_text)

        # Check for duplicate
        similar_claim_id, similarity = self._compute_similarity(claim_embedding)

        if similar_claim_id and similarity >= self.config.dedup_threshold:
            # Deduplicate: map to existing claim
            self._deduplications += 1
            self.cluster_to_claim[cluster_id] = similar_claim_id

            # Update the existing claim
            existing_claim = self.claims[similar_claim_id]
            existing_claim.cluster_ids.append(cluster_id)
            existing_claim.total_clusters = len(existing_claim.cluster_ids)
            if timestamp and (existing_claim.last_seen is None or timestamp > existing_claim.last_seen):
                existing_claim.last_seen = timestamp

            logger.debug(f"Deduplicated cluster {cluster_id} → claim {similar_claim_id[:8]} (sim={similarity:.3f})")
            return similar_claim_id

        # Create new claim
        claim_id = str(uuid.uuid4())
        self._new_claims += 1

        # Extract keywords from cluster tweets
        keywords = extract_keywords(tweets, max_keywords=10)

        self.claims[claim_id] = ClaimInfo(
            claim_id=claim_id,
            claim_text=claim_text,
            first_seen=timestamp or datetime.now(),
            cluster_ids=[cluster_id],
            total_clusters=1,
            keywords=keywords,
        )

        self.cluster_to_claim[cluster_id] = claim_id
        self._add_claim_embedding(claim_id, claim_embedding)

        logger.debug(f"Created new claim {claim_id[:8]} for cluster {cluster_id}: {claim_text[:50]}...")
        if keywords:
            logger.debug(f"  Keywords: {', '.join(keywords[:5])}")
        return claim_id

    def update_claim_stats(
        self,
        claim_id: str,
        timestamp: datetime,
        tweet_count: int = 0,
        engagement: int = 0,
    ) -> None:
        """
        Update statistics for a claim.

        Called during each window to update cumulative stats.

        Args:
            claim_id: Claim to update
            timestamp: Current timestamp
            tweet_count: New tweets in this window
            engagement: New engagement in this window
        """
        if claim_id not in self.claims:
            logger.warning(f"Claim {claim_id} not found in registry")
            return

        claim = self.claims[claim_id]
        claim.total_tweets += tweet_count
        claim.total_engagement += engagement

        if claim.last_seen is None or timestamp > claim.last_seen:
            claim.last_seen = timestamp

    def set_virality_prediction(
        self,
        claim_id: str,
        is_viral: bool,
        confidence: float,
        lead_time_hours: float | None = None,
    ) -> None:
        """Set virality prediction for a claim."""
        if claim_id not in self.claims:
            logger.warning(f"Claim {claim_id} not found in registry")
            return

        claim = self.claims[claim_id]
        claim.is_viral = is_viral
        claim.viral_confidence = confidence
        claim.lead_time_hours = lead_time_hours

    def set_detection_stats(
        self,
        claim_id: str,
        z_score: float,
        kleinberg_state: int,
        trigger_type: str | None = None,
        trigger_cluster_id: int | None = None,
    ) -> None:
        """
        Set detection statistics for a claim (only on first detection).

        These values capture the state at initial anomaly detection
        and should NOT be updated on subsequent observations.

        Args:
            claim_id: Claim to update
            z_score: Composite Z-score at detection
            kleinberg_state: Kleinberg state (0=normal, 1=elevated, 2=burst)
            trigger_type: What triggered detection ("zscore", "kleinberg", "both")
            trigger_cluster_id: Which cluster triggered first detection
        """
        if claim_id not in self.claims:
            logger.warning(f"Claim {claim_id} not found in registry")
            return

        claim = self.claims[claim_id]
        # Only set on first detection (when None)
        if claim.detection_z_score is None:
            claim.detection_z_score = z_score
        if claim.kleinberg_state is None:
            claim.kleinberg_state = kleinberg_state
        if claim.trigger_type is None and trigger_type is not None:
            claim.trigger_type = trigger_type
        if claim.trigger_cluster_id is None and trigger_cluster_id is not None:
            claim.trigger_cluster_id = trigger_cluster_id

    def get_claim(self, claim_id: str) -> ClaimInfo | None:
        """Get claim info by ID."""
        return self.claims.get(claim_id)

    def get_claim_for_cluster(self, cluster_id: int) -> ClaimInfo | None:
        """Get claim info for a cluster."""
        claim_id = self.cluster_to_claim.get(cluster_id)
        if claim_id:
            return self.claims.get(claim_id)
        return None

    def get_all_claims(self) -> list[ClaimInfo]:
        """Get all claims as a list."""
        return list(self.claims.values())

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_claims": len(self.claims),
            "total_clusters_mapped": len(self.cluster_to_claim),
            "normalizations": self._normalizations,
            "deduplications": self._deduplications,
            "new_claims": self._new_claims,
            "dedup_rate": self._deduplications / max(1, self._normalizations),
            "avg_clusters_per_claim": len(self.cluster_to_claim) / max(1, len(self.claims)),
        }

    def save(self, output_dir: Path) -> None:
        """
        Save registry state to disk.

        Saves:
        - claims.json: Claim info
        - cluster_mapping.json: Cluster to claim mapping
        - embeddings.npy: Claim embeddings
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save claims as JSON
        claims_data = {cid: claim.model_dump() for cid, claim in self.claims.items()}
        # Convert datetime to string for JSON
        for claim_data in claims_data.values():
            for key in ["first_seen", "last_seen", "peak_time"]:
                if claim_data.get(key):
                    claim_data[key] = claim_data[key].isoformat() if hasattr(claim_data[key], "isoformat") else str(claim_data[key])

        with open(output_dir / "claims.json", "w") as f:
            json.dump(claims_data, f, indent=2, default=str)

        # Save cluster mapping
        with open(output_dir / "cluster_mapping.json", "w") as f:
            json.dump(self.cluster_to_claim, f, indent=2)

        # Save embeddings
        if self._claim_embeddings is not None:
            np.save(output_dir / "claim_embeddings.npy", self._claim_embeddings)

        # Save claim IDs order
        with open(output_dir / "claim_ids.json", "w") as f:
            json.dump(self._claim_ids, f)

        logger.info(f"Saved registry to {output_dir}")

    @classmethod
    def load(
        cls,
        output_dir: Path,
        normalizer: "LLMNormalizerAdapter",
        embedder: "Embedder",
        config: ClaimRegistryConfig | None = None,
    ) -> "ClaimRegistry":
        """
        Load registry state from disk.

        Args:
            output_dir: Directory containing saved state
            normalizer: LLM normalizer adapter for future normalizations
            embedder: Embedder for future embeddings
            config: Configuration

        Returns:
            Loaded ClaimRegistry
        """
        import json

        output_dir = Path(output_dir)
        registry = cls(normalizer, embedder, config)

        # Load claims
        with open(output_dir / "claims.json") as f:
            claims_data = json.load(f)

        for claim_id, claim_dict in claims_data.items():
            # Parse datetime strings
            for key in ["first_seen", "last_seen", "peak_time"]:
                if claim_dict.get(key):
                    claim_dict[key] = datetime.fromisoformat(claim_dict[key])
            registry.claims[claim_id] = ClaimInfo(**claim_dict)

        # Load cluster mapping
        with open(output_dir / "cluster_mapping.json") as f:
            mapping = json.load(f)
            registry.cluster_to_claim = {int(k): v for k, v in mapping.items()}

        # Load embeddings
        embeddings_path = output_dir / "claim_embeddings.npy"
        if embeddings_path.exists():
            registry._claim_embeddings = np.load(embeddings_path)

        # Load claim IDs
        ids_path = output_dir / "claim_ids.json"
        if ids_path.exists():
            with open(ids_path) as f:
                registry._claim_ids = json.load(f)

        logger.info(f"Loaded registry from {output_dir}: {len(registry.claims)} claims, {len(registry.cluster_to_claim)} mappings")
        return registry
