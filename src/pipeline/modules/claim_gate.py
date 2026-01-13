"""
Claim Gate Module

Filters tweets to identify claim-like content vs reactions/noise.
Tweets that pass the gate are candidates for embedding and clustering.
Tweets that fail are assigned cluster_id = -1.

Filters applied:
1. Minimum/maximum word count
2. Minimum character count
3. Not a question (optional)
4. Not a reaction phrase (optional)
5. Contains at least one verb (optional, requires spaCy)
"""

import logging
import re
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ClaimGateConfig:
    """Configuration for the claim gate."""

    enabled: bool = True
    min_word_count: int = 5
    max_word_count: int = 100
    min_char_count: int = 20
    filter_questions: bool = True
    filter_reactions: bool = True
    require_verb: bool = True
    reaction_patterns: list = field(
        default_factory=lambda: [
            r"^(wow|omg|lol|lmao|haha|damn|whoa|yikes|oof)$",
            r"^(yes|no|nope|yep|yeah|nah)$",
            r"^(great|awesome|amazing|cool|nice)!*$",
        ]
    )
    filtered_cluster_id: int = -1

    @classmethod
    def from_dict(cls, config: dict) -> "ClaimGateConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class ClaimGate:
    """
    Filters tweets to identify claim-like content.

    Usage:
        gate = ClaimGate(config)
        df = gate.apply(df, text_column="tweet_enriched")
    """

    def __init__(self, config: ClaimGateConfig):
        self.config = config
        self._nlp = None
        self._reaction_regex = None

        if config.reaction_patterns:
            combined = "|".join(f"({p})" for p in config.reaction_patterns)
            self._reaction_regex = re.compile(combined, re.IGNORECASE)

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.config.require_verb:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("Loaded spaCy model for verb detection")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                logger.warning("Disabling verb check")
                self.config.require_verb = False
        return self._nlp

    def _has_verb(self, text: str) -> bool:
        """Check if text contains at least one verb."""
        if not self.nlp:
            return True
        doc = self.nlp(text)
        return any(token.pos_ == "VERB" for token in doc)

    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text = text.strip()
        return text.endswith("?") or text.lower().startswith(("who ", "what ", "where ", "when ", "why ", "how "))

    def _is_reaction(self, text: str) -> bool:
        """Check if text matches reaction patterns."""
        if not self._reaction_regex:
            return False
        # Check the whole text and individual words
        text_clean = text.strip().lower()
        if self._reaction_regex.match(text_clean):
            return True
        # Also check if it's very short and matches
        words = text_clean.split()
        if len(words) <= 3:
            for word in words:
                if self._reaction_regex.match(word):
                    return True
        return False

    def _get_word_count(self, text: str) -> int:
        """Get word count."""
        return len(text.split())

    def check_single(self, text: str) -> tuple[bool, str]:
        """
        Check if a single text passes the claim gate.

        Returns:
            (passes: bool, reason: str)
        """
        if text is None or not isinstance(text, str):
            return False, "null_or_invalid"

        text = text.strip()

        # Character count
        if len(text) < self.config.min_char_count:
            return False, "too_short_chars"

        # Word count
        word_count = self._get_word_count(text)
        if word_count < self.config.min_word_count:
            return False, "too_few_words"
        if word_count > self.config.max_word_count:
            return False, "too_many_words"

        # Question filter
        if self.config.filter_questions and self._is_question(text):
            return False, "is_question"

        # Reaction filter
        if self.config.filter_reactions and self._is_reaction(text):
            return False, "is_reaction"

        # Verb check (slowest, do last)
        if self.config.require_verb and not self._has_verb(text):
            return False, "no_verb"

        return True, "passed"

    def apply(
        self,
        df: pl.DataFrame,
        text_column: str = "tweet_enriched",
        output_column: str = "passes_claim_gate",
        reason_column: str = "claim_gate_reason",
    ) -> pl.DataFrame:
        """
        Apply claim gate to dataframe.

        Args:
            df: Input dataframe
            text_column: Column containing text to check
            output_column: Column to store pass/fail boolean
            reason_column: Column to store rejection reason

        Returns:
            DataFrame with added columns
        """
        if not self.config.enabled:
            logger.info("Claim gate disabled, all tweets pass")
            return df.with_columns(
                [
                    pl.lit(True).alias(output_column),
                    pl.lit("gate_disabled").alias(reason_column),
                ]
            )

        logger.info(f"Applying claim gate to {len(df):,} tweets...")

        texts = df[text_column].to_list()
        results = []
        reasons = []

        # Process with progress
        for i, text in enumerate(texts):
            passes, reason = self.check_single(text)
            results.append(passes)
            reasons.append(reason)

            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i + 1:,}/{len(texts):,} tweets")

        # Add columns
        df = df.with_columns(
            [
                pl.Series(name=output_column, values=results),
                pl.Series(name=reason_column, values=reasons),
            ]
        )

        # Log statistics
        passed = sum(results)
        failed = len(results) - passed
        logger.info(f"Claim gate results: {passed:,} passed ({100 * passed / len(results):.1f}%), {failed:,} filtered")

        # Reason breakdown
        reason_counts = df.group_by(reason_column).len().sort("len", descending=True)
        logger.info("Filter reasons:")
        for row in reason_counts.to_dicts():
            logger.info(f"  {row[reason_column]}: {row['len']:,}")

        return df

    def get_stats(self, df: pl.DataFrame, output_column: str = "passes_claim_gate") -> dict:
        """Get statistics about claim gate results."""
        total = len(df)
        passed = df.filter(pl.col(output_column)).shape[0]
        return {
            "total": total,
            "passed": passed,
            "filtered": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
        }
