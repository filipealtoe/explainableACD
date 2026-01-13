"""
ClaimGate: Linguistic pre-filter for claim-like tweets.

This module filters tweets before expensive processing (embedding, clustering)
to reduce noise and focus on tweets that could potentially contain verifiable claims.

Filters OUT:
- Too short (<5 words): reactions, fragments ("lol", "wow")
- Too long (>100 words): spam, threads, copypasta
- No verb: not a statement (just nouns/hashtags)
- Questions: not claims ("What's happening?")
- Pure reactions: RT-only, emoji-only, @mention-only

Expected pass rate: 50-70% (varies by dataset)

Reference: This preprocessing is inspired by checkworthiness literature,
which shows that claims have distinct linguistic patterns:
- Claims are declarative statements (have verbs)
- Claims are complete thoughts (minimum length)
- Claims are not questions or reactions
"""

import logging
import re
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ClaimGateConfig:
    """Configuration for the ClaimGate filter."""

    enabled: bool = True
    min_words: int = 5
    max_words: int = 100
    filter_questions: bool = True
    filter_reactions: bool = True
    require_verb: bool = True
    filter_retweets: bool = True  # Filter "RT @..." tweets

    @classmethod
    def from_dict(cls, config: dict) -> "ClaimGateConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class ClaimGate:
    """
    Linguistic pre-filter for claim-like tweets.

    Usage:
        gate = ClaimGate(ClaimGateConfig())
        df = gate.apply(df, text_column="tweet")
        # df now has "passes_claim_gate" and "claim_gate_reason" columns
    """

    # Common English verbs that indicate a statement/claim
    VERB_PATTERN = re.compile(
        r"\b(is|are|was|were|be|been|being|"
        r"has|have|had|having|"
        r"do|does|did|done|doing|"
        r"will|would|shall|should|"
        r"can|could|may|might|must|"
        r"said|says|say|saying|"
        r"claims?|claimed|claiming|"
        r"announced?|announcing|"
        r"reported?|reporting|"
        r"confirmed?|confirming|"
        r"denied?|denying|"
        r"stated?|stating|"
        r"declared?|declaring|"
        r"revealed?|revealing|"
        r"showed?|shown|showing|"
        r"found|finds?|finding|"
        r"won|wins?|winning|"
        r"lost|loses?|losing|"
        r"voted?|voting|"
        r"killed?|killing|"
        r"arrested?|arresting|"
        r"charged?|charging|"
        r"elected?|electing)\b",
        re.IGNORECASE,
    )

    # Patterns that indicate questions
    QUESTION_PATTERN = re.compile(
        r"^(who|what|when|where|why|how|which|whose|whom)\s|"
        r"\?\s*$|"
        r"^(is|are|was|were|do|does|did|can|could|will|would|should|have|has|had)\s+(you|we|they|it|he|she)\b",
        re.IGNORECASE,
    )

    # Patterns for pure reactions (little informational content)
    REACTION_PATTERN = re.compile(
        r"^(lol|lmao|omg|wow|wtf|smh|ikr|tbh|imo|imho|fr|ngl|idk|rn|af|bruh|yikes|oof|mood|same|facts?|true|this|exactly|periodt?|deadass|cap|no cap|slay|fire|lit|goat|based|cringe|cope|seethe|ratio)\b|"
        r"^[^\w\s]*$|"  # Only punctuation/emoji
        r"^@\w+\s*$",  # Only @mention
        re.IGNORECASE,
    )

    # Retweet pattern
    RT_PATTERN = re.compile(r"^RT\s+@", re.IGNORECASE)

    def __init__(self, config: ClaimGateConfig | None = None):
        self.config = config or ClaimGateConfig()

    def _clean_for_analysis(self, text: str) -> str:
        """Clean text for linguistic analysis (remove URLs, extra spaces)."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove @mentions (but keep the text after for word count)
        text = re.sub(r"@\w+", "", text)
        # Remove hashtag symbols (keep the word)
        text = re.sub(r"#", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _count_words(self, text: str) -> int:
        """Count words in cleaned text."""
        cleaned = self._clean_for_analysis(text)
        if not cleaned:
            return 0
        return len(cleaned.split())

    def _check_tweet(self, text: str) -> tuple[bool, str]:
        """
        Check if a tweet passes the claim gate.

        Returns:
            Tuple of (passes: bool, reason: str)
            reason is empty string if passes, otherwise explains why it failed
        """
        if not isinstance(text, str) or not text.strip():
            return False, "empty"

        # Check for retweet
        if self.config.filter_retweets and self.RT_PATTERN.search(text):
            return False, "retweet"

        # Clean text for analysis
        cleaned = self._clean_for_analysis(text)
        if not cleaned:
            return False, "empty_after_clean"

        # Check word count
        word_count = len(cleaned.split())
        if word_count < self.config.min_words:
            return False, f"too_short_{word_count}_words"
        if word_count > self.config.max_words:
            return False, f"too_long_{word_count}_words"

        # Check for questions
        if self.config.filter_questions and self.QUESTION_PATTERN.search(cleaned):
            return False, "question"

        # Check for pure reactions
        if self.config.filter_reactions and self.REACTION_PATTERN.match(cleaned):
            return False, "reaction"

        # Check for verb (indicates a statement)
        if self.config.require_verb and not self.VERB_PATTERN.search(cleaned):
            return False, "no_verb"

        return True, ""

    def apply(self, df: pl.DataFrame, text_column: str = "tweet") -> pl.DataFrame:
        """
        Apply ClaimGate filter to dataframe.

        Adds two columns:
        - passes_claim_gate: bool - whether the tweet passes
        - claim_gate_reason: str - reason for failure (empty if passes)

        Args:
            df: Input dataframe
            text_column: Name of column containing tweet text

        Returns:
            DataFrame with added filter columns
        """
        if not self.config.enabled:
            logger.info("ClaimGate disabled, passing all tweets")
            return df.with_columns(
                pl.lit(True).alias("passes_claim_gate"),
                pl.lit("").alias("claim_gate_reason"),
            )

        logger.info(f"Applying ClaimGate to {len(df):,} tweets...")

        # Apply filter to each tweet
        results = [self._check_tweet(text) for text in df[text_column].to_list()]
        passes = [r[0] for r in results]
        reasons = [r[1] for r in results]

        df = df.with_columns(
            pl.Series(name="passes_claim_gate", values=passes),
            pl.Series(name="claim_gate_reason", values=reasons),
        )

        # Log statistics
        pass_count = sum(passes)
        pass_rate = pass_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"ClaimGate: {pass_count:,}/{len(df):,} tweets passed ({pass_rate:.1f}%)")

        # Log failure reasons
        if reasons:
            reason_counts = {}
            for r in reasons:
                if r:
                    key = r.split("_")[0]  # Group by prefix
                    reason_counts[key] = reason_counts.get(key, 0) + 1
            if reason_counts:
                logger.info(f"ClaimGate failures: {reason_counts}")

        return df

    def get_stats(self, df: pl.DataFrame) -> dict:
        """Get statistics about ClaimGate filtering."""
        if "passes_claim_gate" not in df.columns:
            return {"error": "ClaimGate not applied"}

        total = len(df)
        passed = df["passes_claim_gate"].sum()
        failed = total - passed

        # Count reasons
        reason_counts = (
            df.filter(~pl.col("passes_claim_gate"))
            .group_by("claim_gate_reason")
            .len()
            .to_dict()
        )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "failure_reasons": dict(zip(reason_counts.get("claim_gate_reason", []), reason_counts.get("len", []))),
        }
