"""Pydantic schemas for Checkworthiness module outputs."""

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


# =============================================================================
# Logprob-based Confidence Calculation
# =============================================================================

# Token variants for ternary fields (handles tokenization differences)
# From Anthropic's "Language Models (Mostly) Know What They Know"
# Note: Output is now string ("true"/"false"/"uncertain") not boolean
TRUE_VARIANTS = ["True", " True", "true", " true", '"true"', '"True"', ' "true"', ' "True"']
FALSE_VARIANTS = ["False", " False", "false", " false", '"false"', '"False"', ' "false"', ' "False"']
UNCERTAIN_VARIANTS = ["Uncertain", " Uncertain", "uncertain", " uncertain", '"uncertain"', '"Uncertain"', ' "uncertain"', ' "Uncertain"']


@dataclass
class LogprobConfidence:
    """Logprob-based confidence scores for a ternary field (true/false/uncertain).

    Uses the Anthropic formula from "Language Models (Mostly) Know What They Know":
    P(true) = sum(P(True-like)) / (sum(P(True-like)) + sum(P(False-like)) + sum(P(Uncertain-like)))

    This normalizes the probability over all ternary token variants.
    """

    p_true: float  # P(true) normalized across ternary options (0-100 scale)
    p_false: float  # P(false) normalized across ternary options (0-100 scale)
    p_uncertain: float  # P(uncertain) normalized across ternary options (0-100 scale)
    self_reported_confidence: float  # Original model output (0-100 scale)
    raw_logprobs: dict | None = None  # Raw logprob data for debugging

    @property
    def boolean_confidence(self) -> float:
        """Backwards compatible: P(true) / (P(true) + P(false)) ignoring uncertain."""
        if self.p_true + self.p_false > 0:
            return (self.p_true / (self.p_true + self.p_false)) * 100
        return 50.0  # Fallback

    @classmethod
    def from_logprobs(
        cls,
        top_logprobs: dict[str, float],
        self_reported: float,
    ) -> "LogprobConfidence":
        """Calculate logprob-based confidence from raw logprob data.

        Uses the Anthropic formula extended for ternary:
        P(x) = sum(P(X-like)) / (sum(P(True-like)) + sum(P(False-like)) + sum(P(Uncertain-like)))

        Args:
            top_logprobs: Dict mapping token strings to their logprobs at ternary position
            self_reported: The confidence value reported by the model (0-100)

        Returns:
            LogprobConfidence with calculated values
        """
        # Sum probabilities for each category
        p_true_sum = sum(math.exp(top_logprobs.get(t, -100)) for t in TRUE_VARIANTS)
        p_false_sum = sum(math.exp(top_logprobs.get(t, -100)) for t in FALSE_VARIANTS)
        p_uncertain_sum = sum(math.exp(top_logprobs.get(t, -100)) for t in UNCERTAIN_VARIANTS)

        total = p_true_sum + p_false_sum + p_uncertain_sum
        if total > 0:
            p_true = (p_true_sum / total) * 100
            p_false = (p_false_sum / total) * 100
            p_uncertain = (p_uncertain_sum / total) * 100
        else:
            # Fallback if no ternary tokens found
            p_true = 33.3
            p_false = 33.3
            p_uncertain = 33.3

        return cls(
            p_true=p_true,
            p_false=p_false,
            p_uncertain=p_uncertain,
            self_reported_confidence=self_reported,
            raw_logprobs=top_logprobs,
        )

    @classmethod
    def unavailable(cls, self_reported: float) -> "LogprobConfidence":
        """Create a placeholder when logprobs are not available."""
        return cls(
            p_true=self_reported,
            p_false=100 - self_reported,
            p_uncertain=0.0,
            self_reported_confidence=self_reported,
            raw_logprobs=None,
        )


@dataclass
class LogprobData:
    """Raw logprob data extracted from API response."""

    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[dict[str, float]] = field(default_factory=list)

    def find_ternary_logprobs(self, field_name: str) -> dict[str, float]:
        """Find the top_logprobs at the position after a ternary field name.

        Uses character-position mapping since field names may be split across tokens.

        Args:
            field_name: The JSON field name to search for (e.g., "is_checkable")

        Returns:
            Dict of token -> logprob at the ternary value position
        """
        if not self.tokens:
            return {}

        # Reconstruct full text and find field position
        full_text = "".join(self.tokens)

        # Look for the field name in the full text
        field_pos = full_text.find(f'"{field_name}"')
        if field_pos == -1:
            # Try without quotes (some JSON formats)
            field_pos = full_text.find(field_name)
        if field_pos == -1:
            return {}

        # Map character position to token index
        char_pos = 0
        field_start_token = -1
        for i, token in enumerate(self.tokens):
            if char_pos <= field_pos < char_pos + len(token):
                field_start_token = i
                break
            char_pos += len(token)

        if field_start_token == -1:
            return {}

        # Search for ternary token after the field name
        # Look in the next 10 tokens for a true/false/uncertain value
        for j in range(field_start_token, min(field_start_token + 10, len(self.tokens))):
            token_stripped = self.tokens[j].lower().strip().strip('"')
            if token_stripped in ("true", "false", "uncertain"):
                if j < len(self.top_logprobs):
                    return self.top_logprobs[j]

        return {}

    def get_reasoning_logprobs(self) -> list[float]:
        """Get logprobs for the reasoning section of the response.

        Returns logprobs for tokens in the "reasoning" field value.
        """
        in_reasoning = False
        reasoning_probs = []

        for i, token in enumerate(self.tokens):
            # Detect start of reasoning field
            if '"reasoning"' in token or "'reasoning'" in token:
                in_reasoning = True
                continue

            # Detect end of reasoning (next field or closing brace)
            if in_reasoning:
                if any(f'"{f}"' in token for f in ["confidence", "is_", "social_", "spurs_", "believability", "exploitativeness"]):
                    break
                if i < len(self.logprobs):
                    reasoning_probs.append(self.logprobs[i])

        return reasoning_probs

    @classmethod
    def from_api_response(cls, logprobs_content: list | None) -> "LogprobData | None":
        """Extract LogprobData from OpenAI API response.

        Args:
            logprobs_content: The response.choices[0].logprobs.content list

        Returns:
            LogprobData or None if logprobs not available
        """
        if not logprobs_content:
            return None

        tokens = []
        logprobs = []
        top_logprobs = []

        for item in logprobs_content:
            tokens.append(item.token if hasattr(item, "token") else item.get("token", ""))
            logprobs.append(item.logprob if hasattr(item, "logprob") else item.get("logprob", 0.0))

            # Extract top_logprobs dict
            top_lp = item.top_logprobs if hasattr(item, "top_logprobs") else item.get("top_logprobs", [])
            if top_lp:
                top_dict = {}
                for lp in top_lp:
                    token = lp.token if hasattr(lp, "token") else lp.get("token", "")
                    prob = lp.logprob if hasattr(lp, "logprob") else lp.get("logprob", 0.0)
                    top_dict[token] = prob
                top_logprobs.append(top_dict)
            else:
                top_logprobs.append({})

        return cls(tokens=tokens, logprobs=logprobs, top_logprobs=top_logprobs)


# =============================================================================
# Module Output Schemas
# =============================================================================

class CheckabilityOutput(BaseModel):
    """Output from the Checkability module.

    A claim is checkable if it is not an opinion, not a prediction,
    and does not rely on vague or subjective generalities.
    """

    reasoning: str = Field(description="Step-by-step reasoning explaining why the claim is or is not checkable")
    confidence: float = Field(ge=0.0, le=100.0, description="Final confidence (logprob-derived if available, else self-reported)")

    # Separate confidence sources for analysis
    self_confidence: float | None = Field(default=None, description="Self-reported confidence from model's JSON output (0-100)")
    logprob_confidence: float | None = Field(default=None, description="Logprob-derived confidence (p_true * 100), None if logprobs unavailable")

    # Ternary probability distribution from logprobs (0-1 scale)
    p_true: float | None = Field(default=None, description="P(true) from logprobs, normalized over ternary options (0-1 scale)")
    p_false: float | None = Field(default=None, description="P(false) from logprobs, normalized over ternary options (0-1 scale)")
    p_uncertain: float | None = Field(default=None, description="P(uncertain) from logprobs, normalized over ternary options (0-1 scale)")
    entropy: float | None = Field(default=None, description="Shannon entropy of ternary distribution (0=certain, 1.585=uniform)")

    # Data quality flags (for filtering/analysis)
    json_parse_failed: bool = Field(default=False, description="True if JSON parsing failed and fallback values were used")
    logprobs_missing: bool = Field(default=False, description="True if logprobs were unavailable (confidence from self-report or fallback)")


class VerifiabilityOutput(BaseModel):
    """Output from the Verifiability module.

    A claim is verifiable if there is sufficient publicly available data
    and reputable sources that allow the claim to be checked empirically.
    """

    reasoning: str = Field(description="Step-by-step reasoning explaining the verifiability assessment")
    confidence: float = Field(
        ge=0.0, le=100.0, description="Final confidence (logprob-derived if available, else self-reported)"
    )

    # Separate confidence sources for analysis
    self_confidence: float | None = Field(default=None, description="Self-reported confidence from model's JSON output (0-100)")
    logprob_confidence: float | None = Field(default=None, description="Logprob-derived confidence (p_true * 100), None if logprobs unavailable")

    # Ternary probability distribution from logprobs (0-1 scale)
    p_true: float | None = Field(default=None, description="P(true) from logprobs, normalized over ternary options (0-1 scale)")
    p_false: float | None = Field(default=None, description="P(false) from logprobs, normalized over ternary options (0-1 scale)")
    p_uncertain: float | None = Field(default=None, description="P(uncertain) from logprobs, normalized over ternary options (0-1 scale)")
    entropy: float | None = Field(default=None, description="Shannon entropy of ternary distribution (0=certain, 1.585=uniform)")

    # Data quality flags (for filtering/analysis)
    json_parse_failed: bool = Field(default=False, description="True if JSON parsing failed and fallback values were used")
    logprobs_missing: bool = Field(default=False, description="True if logprobs were unavailable (confidence from self-report or fallback)")


class HarmSubScores(BaseModel):
    """Sub-scores for the four harm potential criteria."""

    social_fragmentation: float = Field(
        ge=0.0, le=100.0, description="Confidence that claim contributes to social fragmentation"
    )
    spurs_action: float = Field(ge=0.0, le=100.0, description="Confidence that claim could spur harmful action")
    believability: float = Field(
        ge=0.0, le=100.0, description="Confidence that claim is believable in a misleading way"
    )
    exploitativeness: float = Field(ge=0.0, le=100.0, description="Confidence that claim exploits vulnerabilities")


class HarmPotentialOutput(BaseModel):
    """Output from the Harm Potential module.

    A claim has high harm potential if it contributes to social fragmentation,
    spurs harmful action, is highly believable in a misleading way,
    and/or exploits human or group vulnerabilities.
    """

    reasoning: str = Field(description="Step-by-step reasoning explaining the harm potential assessment")
    confidence: float = Field(ge=0.0, le=100.0, description="Final confidence (logprob-derived if available, else self-reported)")
    sub_scores: HarmSubScores = Field(description="Individual scores for each harm criterion")

    # Separate confidence sources for analysis
    self_confidence: float | None = Field(default=None, description="Self-reported confidence from model's JSON output (0-100)")
    logprob_confidence: float | None = Field(default=None, description="Logprob-derived confidence (p_true * 100), None if logprobs unavailable")

    # Ternary probability distribution from logprobs (0-1 scale)
    p_true: float | None = Field(default=None, description="P(true) from logprobs, normalized over ternary options (0-1 scale)")
    p_false: float | None = Field(default=None, description="P(false) from logprobs, normalized over ternary options (0-1 scale)")
    p_uncertain: float | None = Field(default=None, description="P(uncertain) from logprobs, normalized over ternary options (0-1 scale)")
    entropy: float | None = Field(default=None, description="Shannon entropy of ternary distribution (0=certain, 1.585=uniform)")

    # Data quality flags (for filtering/analysis)
    json_parse_failed: bool = Field(default=False, description="True if JSON parsing failed and fallback values were used")
    logprobs_missing: bool = Field(default=False, description="True if logprobs were unavailable (confidence from self-report or fallback)")


class CheckworthinessResult(BaseModel):
    """Combined result from all three modules plus final classification."""

    claim_text: str = Field(description="The original claim text")

    checkability: CheckabilityOutput = Field(description="Checkability assessment")
    verifiability: VerifiabilityOutput = Field(description="Verifiability assessment")
    harm_potential: HarmPotentialOutput = Field(description="Harm potential assessment")

    average_confidence: float = Field(ge=0.0, le=100.0, description="Average of the three main confidence scores")
    prediction: str = Field(description="Final prediction: 'Yes' if checkworthy, 'No' otherwise")

    @classmethod
    def from_modules(
        cls,
        claim_text: str,
        checkability: CheckabilityOutput,
        verifiability: VerifiabilityOutput,
        harm_potential: HarmPotentialOutput,
        threshold: float = 50.0,
    ) -> "CheckworthinessResult":
        """Create result from module outputs using simple average formula."""
        avg = (checkability.confidence + verifiability.confidence + harm_potential.confidence) / 3.0

        prediction = "Yes" if avg > threshold else "No"

        return cls(
            claim_text=claim_text,
            checkability=checkability,
            verifiability=verifiability,
            harm_potential=harm_potential,
            average_confidence=avg,
            prediction=prediction,
        )
