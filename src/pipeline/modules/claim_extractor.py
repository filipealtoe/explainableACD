"""
Claim Extractor Module

Uses an LLM to extract/summarize the main claim from a cluster of tweets.
Only processes clusters that meet a minimum size threshold.

Supports multiple backends:
- Anthropic (Claude) - default, sync
- Groq (Llama 3.1 8B) - async, faster, free tier available
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


# Rate limiting constants
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


@dataclass
class ClaimExtractorConfig:
    """Configuration for the claim extractor."""

    enabled: bool = False
    model: str = "claude-3-5-haiku-latest"
    max_tweets_per_cluster: int = 5
    min_cluster_size_for_extraction: int = 10
    max_tokens: int = 150
    temperature: float = 0.0
    # Rate limiting
    requests_per_minute: int = 50
    # Caching
    cache_results: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "ClaimExtractorConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


EXTRACTION_PROMPT = """Analyze these tweets about the same topic and extract the main claim or narrative they share.

Tweets:
{tweets}

Respond with ONLY a single concise sentence (max 20 words) that captures the core claim or topic these tweets are discussing. Do not include hashtags, mentions, or any preamble."""


class ClaimExtractor:
    """
    Extracts claim summaries from tweet clusters using an LLM.

    Usage:
        extractor = ClaimExtractor(config)
        claims = extractor.extract_claims(df, text_column="tweet_enriched")
    """

    def __init__(self, config: ClaimExtractorConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy load the Anthropic client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. Set it with: export ANTHROPIC_API_KEY='your-key'"
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")
        return self._client

    @staticmethod
    def is_available() -> bool:
        """Check if the API key is available."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _format_tweets(self, tweets: list[str]) -> str:
        """Format tweets for the prompt."""
        formatted = []
        for i, tweet in enumerate(tweets[: self.config.max_tweets_per_cluster], 1):
            # Truncate very long tweets
            if len(tweet) > 280:
                tweet = tweet[:280] + "..."
            formatted.append(f"{i}. {tweet}")
        return "\n".join(formatted)

    def extract_single(self, tweets: list[str]) -> str:
        """
        Extract claim from a single cluster.

        Args:
            tweets: List of tweet texts from the cluster

        Returns:
            Extracted claim string
        """
        if not tweets:
            return ""

        prompt = EXTRACTION_PROMPT.format(tweets=self._format_tweets(tweets))

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            claim = response.content[0].text.strip()
            return claim
        except Exception as e:
            logger.error(f"Error extracting claim: {e}")
            return f"[Error: {str(e)[:50]}]"

    def extract_claims(
        self,
        df: pl.DataFrame,
        text_column: str = "tweet_enriched",
        cluster_column: str = "cluster_id",
        show_progress: bool = True,
    ) -> dict[int, str]:
        """
        Extract claims for all qualifying clusters.

        Args:
            df: DataFrame with clustered tweets
            text_column: Column containing tweet text
            cluster_column: Column containing cluster IDs
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping cluster_id to extracted claim
        """
        if not self.config.enabled:
            logger.info("Claim extractor disabled")
            return {}

        if not self.is_available():
            logger.warning("ANTHROPIC_API_KEY not set - skipping claim extraction")
            return {}

        # Get cluster sizes
        cluster_sizes = (
            df.filter(pl.col(cluster_column) >= 0)
            .group_by(cluster_column)
            .agg(pl.len().alias("size"))
            .filter(pl.col("size") >= self.config.min_cluster_size_for_extraction)
            .sort("size", descending=True)
        )

        n_clusters = len(cluster_sizes)
        logger.info(
            f"Extracting claims for {n_clusters} clusters (size >= {self.config.min_cluster_size_for_extraction})"
        )

        if n_clusters == 0:
            return {}

        claims = {}
        cluster_ids = cluster_sizes[cluster_column].to_list()

        for i, cluster_id in enumerate(cluster_ids):
            # Get tweets for this cluster
            cluster_tweets = (
                df.filter(pl.col(cluster_column) == cluster_id)
                .select(text_column)
                .head(self.config.max_tweets_per_cluster)[text_column]
                .to_list()
            )

            # Extract claim
            claim = self.extract_single(cluster_tweets)
            claims[cluster_id] = claim

            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"  Extracted {i + 1}/{n_clusters} claims")

        logger.info(f"Extracted {len(claims)} claims")
        return claims

    def apply(
        self,
        df: pl.DataFrame,
        text_column: str = "tweet_enriched",
        cluster_column: str = "cluster_id",
        output_column: str = "cluster_claim",
    ) -> tuple[pl.DataFrame, dict[int, str]]:
        """
        Apply claim extraction and add claims to dataframe.

        Args:
            df: Input DataFrame
            text_column: Column containing tweet text
            cluster_column: Column containing cluster IDs
            output_column: Column to store claims

        Returns:
            Tuple of (DataFrame with claims, claims dict)
        """
        claims = self.extract_claims(df, text_column, cluster_column)

        if not claims:
            df = df.with_columns(pl.lit("").alias(output_column))
            return df, claims

        # Map claims to dataframe
        claim_values = [claims.get(cid, "") for cid in df[cluster_column].to_list()]

        df = df.with_columns(pl.Series(name=output_column, values=claim_values))

        return df, claims

    def get_stats(self, claims: dict[int, str]) -> dict:
        """Get statistics about extracted claims."""
        if not claims:
            return {"n_claims": 0}

        claim_lengths = [len(c) for c in claims.values()]
        return {
            "n_claims": len(claims),
            "avg_claim_length": sum(claim_lengths) / len(claim_lengths),
            "min_claim_length": min(claim_lengths),
            "max_claim_length": max(claim_lengths),
        }


def format_claims_report(claims: dict[int, str], cluster_sizes: dict[int, int]) -> str:
    """Format claims as a readable report."""
    lines = ["=" * 70, "EXTRACTED CLAIMS REPORT", "=" * 70, ""]

    # Sort by cluster size
    sorted_clusters = sorted(
        claims.keys(),
        key=lambda x: cluster_sizes.get(x, 0),
        reverse=True,
    )

    for cluster_id in sorted_clusters:
        claim = claims[cluster_id]
        size = cluster_sizes.get(cluster_id, 0)
        lines.append(f"Cluster {cluster_id} ({size} tweets):")
        lines.append(f"  {claim}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# =============================================================================
# GROQ ASYNC ADAPTER
# =============================================================================


@dataclass
class GroqAsyncAdapterConfig:
    """Configuration for the Groq async adapter."""

    enabled: bool = True
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 256
    temperature: float = 0.3
    max_tweets_per_cluster: int = 5
    rate_limit_rpm: int = 30  # Groq free tier limit
    timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, config: dict) -> "GroqAsyncAdapterConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class GroqAsyncAdapter:
    """
    Async Groq client for claim normalization.

    Uses the Groq SDK with async support for efficient batch processing.
    Includes rate limiting and caching to respect API limits.

    Usage:
        adapter = GroqAsyncAdapter(GroqAsyncAdapterConfig())
        claim = await adapter.normalize_claim(cluster_id=5, tweets=["tweet1", "tweet2"])

        # Or batch processing:
        claims = await adapter.normalize_batch([
            (1, ["tweets..."]),
            (2, ["tweets..."]),
        ])
    """

    def __init__(self, config: GroqAsyncAdapterConfig | None = None):
        """
        Initialize the Groq async adapter.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or GroqAsyncAdapterConfig()
        self._client = None

        # Rate limiting with semaphore
        # Use half the RPM limit to be conservative and avoid hitting limits
        concurrent_limit = max(1, self.config.rate_limit_rpm // 2)
        self._semaphore = asyncio.Semaphore(concurrent_limit)

        # Cache: cluster_id -> claim_text
        self._cache: dict[int, str] = {}

        # Statistics
        self._api_calls = 0
        self._cache_hits = 0
        self._errors = 0
        self._total_tokens = 0

    @property
    def client(self):
        """Lazy load the Groq async client."""
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable not set. "
                    "Get a free key at: https://console.groq.com"
                )
            try:
                from groq import AsyncGroq
            except ImportError:
                raise ImportError(
                    "groq package not installed. Install with: uv add groq"
                )

            self._client = AsyncGroq(api_key=api_key)
            logger.info(f"Initialized Groq async client with model: {self.config.model}")

        return self._client

    @staticmethod
    def is_available() -> bool:
        """Check if the API key is available."""
        return bool(os.environ.get("GROQ_API_KEY"))

    def _format_tweets(self, tweets: list[str]) -> str:
        """Format tweets for the prompt."""
        formatted = []
        for i, tweet in enumerate(tweets[: self.config.max_tweets_per_cluster], 1):
            # Truncate very long tweets
            if len(tweet) > 280:
                tweet = tweet[:280] + "..."
            formatted.append(f"{i}. {tweet}")
        return "\n".join(formatted)

    async def normalize_claim(self, cluster_id: int, tweets: list[str]) -> str:
        """
        Normalize a cluster of tweets into a single claim.

        Uses caching to avoid redundant API calls for the same cluster.

        Args:
            cluster_id: Cluster identifier (used for caching)
            tweets: List of tweet texts from the cluster

        Returns:
            Normalized claim text
        """
        # Check cache first
        if cluster_id in self._cache:
            self._cache_hits += 1
            return self._cache[cluster_id]

        if not tweets:
            return ""

        prompt = EXTRACTION_PROMPT.format(tweets=self._format_tweets(tweets))

        # Rate limit with semaphore
        async with self._semaphore:
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        timeout=self.config.timeout_seconds,
                    )

                    self._api_calls += 1

                    # Track token usage if available
                    if hasattr(response, "usage") and response.usage:
                        self._total_tokens += response.usage.total_tokens

                    claim = response.choices[0].message.content.strip()

                    # Cache the result
                    self._cache[cluster_id] = claim

                    return claim

                except Exception as e:
                    last_error = e
                    self._errors += 1

                    # Check if it's a rate limit error
                    if "rate" in str(e).lower() or "429" in str(e):
                        backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                        logger.warning(f"Rate limit hit, waiting {backoff:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(backoff)
                        continue

                    # For other errors, retry with backoff
                    if attempt < MAX_RETRIES - 1:
                        backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                        logger.warning(f"Error: {e}, retrying in {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                        continue

                    logger.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                    break

            # Return error placeholder
            error_msg = f"[Error: {str(last_error)[:50]}]"
            self._cache[cluster_id] = error_msg
            return error_msg

    async def normalize_batch(
        self,
        clusters: list[tuple[int, list[str]]],
        show_progress: bool = True,
    ) -> dict[int, str]:
        """
        Normalize multiple clusters concurrently with rate limiting.

        Args:
            clusters: List of (cluster_id, tweets) tuples
            show_progress: Whether to log progress

        Returns:
            Dictionary mapping cluster_id to normalized claim
        """
        if not clusters:
            return {}

        logger.info(f"Normalizing {len(clusters)} clusters via Groq...")

        # Create tasks for all clusters
        tasks = [
            self.normalize_claim(cluster_id, tweets)
            for cluster_id, tweets in clusters
        ]

        # Run concurrently (semaphore handles rate limiting)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary
        claims = {}
        for (cluster_id, _), result in zip(clusters, results):
            if isinstance(result, Exception):
                logger.error(f"Cluster {cluster_id} failed: {result}")
                claims[cluster_id] = f"[Error: {str(result)[:50]}]"
            else:
                claims[cluster_id] = result

        if show_progress:
            logger.info(f"Normalized {len(claims)} claims ({self._cache_hits} cache hits, {self._errors} errors)")

        return claims

    def clear_cache(self) -> None:
        """Clear the claim cache."""
        self._cache.clear()
        logger.info("Cleared claim cache")

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "errors": self._errors,
            "total_tokens": self._total_tokens,
            "model": self.config.model,
            "rate_limit_rpm": self.config.rate_limit_rpm,
        }

    def get_cached_claims(self) -> dict[int, str]:
        """Get all cached claims."""
        return dict(self._cache)
