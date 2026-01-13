"""DSPy modules for Checkworthiness assessment."""

from concurrent.futures import ThreadPoolExecutor

import dspy

from .schemas import (
    CheckabilityOutput,
    CheckworthinessResult,
    HarmPotentialOutput,
    VerifiabilityOutput,
)


class CheckabilitySignature(dspy.Signature):
    """Assess whether a claim is checkable.

    A claim is considered "checkable" if it:
    - Is NOT an opinion (subjective belief or preference)
    - Is NOT a prediction about future events
    - Does NOT rely on vague or subjective generalities

    First, analyze the claim step by step. Then provide your confidence score.
    """

    claim: str = dspy.InputField(desc="The claim text to assess for checkability")
    output: CheckabilityOutput = dspy.OutputField(desc="Reasoning followed by confidence score (0-100%)")


class VerifiabilitySignature(dspy.Signature):
    """Assess whether a claim is verifiable.

    A claim is considered "verifiable" if there is sufficient publicly available
    data and reputable sources that allow the claim to be checked empirically.

    You are assessing verifiability, NOT whether the claim is true or false.

    First, analyze what evidence would be needed and its availability. Then provide your confidence score.
    """

    claim: str = dspy.InputField(desc="The claim text to assess for verifiability")
    output: VerifiabilityOutput = dspy.OutputField(desc="Reasoning followed by confidence score (0-100%)")


class HarmPotentialSignature(dspy.Signature):
    """Assess the potential harm of a claim.

    A claim has "high harm potential" if it:
    - SOCIAL FRAGMENTATION: Fits polarizing narratives, undermines trust in institutions
    - SPURS ACTION: Includes calls to action or could lead to direct harm
    - BELIEVABILITY: Likely to be believed due to lack of accessible refuting information
    - EXPLOITATIVENESS: Exploits vulnerabilities (fear, complexity, targeting vulnerable groups)

    You are assessing harm POTENTIAL, not whether the claim is true or false.

    First, analyze each criterion step by step. Then provide your confidence scores.
    """

    claim: str = dspy.InputField(desc="The claim text to assess for harm potential")
    output: HarmPotentialOutput = dspy.OutputField(
        desc="Reasoning followed by overall and sub-category confidence scores (0-100%)"
    )


class CheckabilityModule(dspy.Module):
    """DSPy module for checkability assessment."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.TypedPredictor(CheckabilitySignature)

    def forward(self, claim: str) -> CheckabilityOutput:
        result = self.predictor(claim=claim)
        return result.output


class VerifiabilityModule(dspy.Module):
    """DSPy module for verifiability assessment."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.TypedPredictor(VerifiabilitySignature)

    def forward(self, claim: str) -> VerifiabilityOutput:
        result = self.predictor(claim=claim)
        return result.output


class HarmPotentialModule(dspy.Module):
    """DSPy module for harm potential assessment."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.TypedPredictor(HarmPotentialSignature)

    def forward(self, claim: str) -> HarmPotentialOutput:
        result = self.predictor(claim=claim)
        return result.output


class CheckworthinessPipeline(dspy.Module):
    """Complete pipeline combining all three modules.

    Uses simple average of confidence scores with 50% threshold
    to determine final Yes/No prediction.
    """

    def __init__(self, threshold: float = 50.0) -> None:
        super().__init__()
        self.checkability = CheckabilityModule()
        self.verifiability = VerifiabilityModule()
        self.harm_potential = HarmPotentialModule()
        self.threshold = threshold

    def forward(self, claim: str) -> CheckworthinessResult:
        # Run all three modules in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_check = executor.submit(self.checkability, claim=claim)
            future_verif = executor.submit(self.verifiability, claim=claim)
            future_harm = executor.submit(self.harm_potential, claim=claim)

            check_out = future_check.result()
            verif_out = future_verif.result()
            harm_out = future_harm.result()

        # Combine results using simple average formula
        result = CheckworthinessResult.from_modules(
            claim_text=claim,
            checkability=check_out,
            verifiability=verif_out,
            harm_potential=harm_out,
            threshold=self.threshold,
        )

        return result


def create_metric(threshold: float = 50.0):
    """Create a metric function for GEPA optimization.

    The metric returns 1.0 if the prediction matches the ground truth label,
    0.0 otherwise.
    """

    def checkworthiness_metric(example, prediction, trace=None) -> float:
        """Metric for GEPA: accuracy on Yes/No prediction."""
        # Get ground truth from example
        ground_truth = example.label  # "Yes" or "No"

        # Get prediction
        if hasattr(prediction, "prediction"):
            pred_label = prediction.prediction
        else:
            # If running individual modules, compute from outputs
            avg = (
                prediction.checkability.confidence
                + prediction.verifiability.confidence
                + prediction.harm_potential.confidence
            ) / 3.0
            pred_label = "Yes" if avg > threshold else "No"

        # Return 1.0 if correct, 0.0 if wrong
        return 1.0 if pred_label == ground_truth else 0.0

    return checkworthiness_metric
