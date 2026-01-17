"""Run baseline checkworthiness experiment (without GEPA optimization).

This script:
1. Loads the CT24 checkworthy dataset
2. Runs the checkworthiness pipeline on a subset
3. Tracks token usage and costs
4. Evaluates accuracy using simple average formula
5. Saves results

Usage:
    python experiments/scripts/run_checkworthiness_baseline.py --model gpt-4.1 --samples 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import dspy
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.checkworthiness.config import (
    MODELS,
    ExperimentConfig,
    ExperimentStats,
    TokenUsage,
)
from src.checkworthiness.modules import CheckworthinessPipeline


def load_dataset(filepath: Path) -> list[dict]:
    """Load CT24 dataset from TSV file."""
    data = []
    with open(filepath, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                data.append(
                    {
                        "sentence_id": parts[0],
                        "text": parts[1],
                        "label": parts[2],
                    }
                )
    return data


def setup_dspy(config: ExperimentConfig) -> dspy.LM:
    """Configure DSPy with the specified model and return the LM for token tracking."""
    model_config = config.get_model_config()

    # Build LM kwargs
    lm_kwargs = {
        "model": f"openai/{model_config.model_name}",
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
        "api_key": model_config.get_api_key(),
    }

    # Add api_base for non-OpenAI providers
    if model_config.api_base:
        lm_kwargs["api_base"] = model_config.api_base

    # Create LM
    lm = dspy.LM(**lm_kwargs)

    # Configure DSPy
    if config.use_baml_adapter:
        try:
            from dspy.adapters.baml_adapter import BAMLAdapter

            dspy.configure(lm=lm, adapter=BAMLAdapter())
            if config.verbose:
                print("Using BAML adapter")
        except ImportError:
            print("Warning: BAML adapter not available, using default adapter")
            dspy.configure(lm=lm)
    else:
        dspy.configure(lm=lm)

    return lm


def extract_token_usage(lm: dspy.LM) -> TokenUsage:
    """Extract token usage from the last LM call."""
    try:
        # DSPy stores history of calls
        if hasattr(lm, "history") and lm.history:
            last_call = lm.history[-1]
            usage = last_call.get("usage", {})
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
    except Exception:
        pass
    return TokenUsage()


def run_baseline_experiment(
    config: ExperimentConfig,
    num_samples: int | None = None,
) -> dict:
    """Run baseline experiment without GEPA optimization."""

    # Load environment variables
    load_dotenv()

    # Print configuration
    if config.verbose:
        config.print_config()

    # Setup DSPy
    lm = setup_dspy(config)
    model_config = config.get_model_config()

    # Load dataset
    data_path = Path(config.data_dir)
    dev_data = load_dataset(data_path / "CT24_checkworthy_english_dev.tsv")

    # Limit samples if specified
    if num_samples:
        dev_data = dev_data[:num_samples]

    print(f"Running on {len(dev_data)} samples from dev set\n")

    # Create pipeline
    pipeline = CheckworthinessPipeline(threshold=config.threshold)

    # Track statistics
    stats = ExperimentStats()
    results = []
    correct = 0
    errors = []

    for item in tqdm(dev_data, desc="Processing claims"):
        try:
            # Run pipeline
            result = pipeline(claim=item["text"])

            # Extract token usage (approximate - DSPy may batch calls)
            usage = extract_token_usage(lm)
            stats.add_usage(usage, model_config)

            # Check accuracy
            is_correct = result.prediction == item["label"]
            if is_correct:
                correct += 1

            results.append(
                {
                    "sentence_id": item["sentence_id"],
                    "claim": item["text"],
                    "ground_truth": item["label"],
                    "prediction": result.prediction,
                    "average_confidence": result.average_confidence,
                    "checkability_confidence": result.checkability.confidence,
                    "checkability_reasoning": result.checkability.reasoning,
                    "verifiability_confidence": result.verifiability.confidence,
                    "verifiability_reasoning": result.verifiability.reasoning,
                    "harm_potential_confidence": result.harm_potential.confidence,
                    "harm_potential_reasoning": result.harm_potential.reasoning,
                    "harm_sub_scores": {
                        "social_fragmentation": result.harm_potential.sub_scores.social_fragmentation,
                        "spurs_action": result.harm_potential.sub_scores.spurs_action,
                        "believability": result.harm_potential.sub_scores.believability,
                        "exploitativeness": result.harm_potential.sub_scores.exploitativeness,
                    },
                    "is_correct": is_correct,
                }
            )

            if config.verbose and len(results) == 1:
                # Print first result as example
                print("\n" + "-" * 60)
                print("EXAMPLE OUTPUT (first claim)")
                print("-" * 60)
                print(f"Claim: {item['text'][:100]}...")
                print(f"Checkability: {result.checkability.confidence}%")
                print(f"  Reasoning: {result.checkability.reasoning[:100]}...")
                print(f"Verifiability: {result.verifiability.confidence}%")
                print(f"Harm Potential: {result.harm_potential.confidence}%")
                print(f"Average: {result.average_confidence:.1f}%")
                print(f"Prediction: {result.prediction} (Ground Truth: {item['label']})")
                print("-" * 60 + "\n")

        except Exception as e:
            errors.append(
                {
                    "sentence_id": item["sentence_id"],
                    "claim": item["text"],
                    "error": str(e),
                }
            )
            if config.verbose:
                print(f"\nError processing claim {item['sentence_id']}: {e}")

    # Calculate metrics
    accuracy = correct / len(dev_data) if dev_data else 0.0

    # Count Yes/No distribution
    yes_count = sum(1 for r in results if r["prediction"] == "Yes")
    no_count = sum(1 for r in results if r["prediction"] == "No")
    gt_yes = sum(1 for item in dev_data if item["label"] == "Yes")
    gt_no = sum(1 for item in dev_data if item["label"] == "No")

    summary = {
        "experiment": "baseline",
        "model": config.model_name,
        "model_config": {
            "provider": model_config.provider.value,
            "api_base": model_config.api_base,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "is_thinking_model": model_config.is_thinking_model,
        },
        "threshold": config.threshold,
        "use_baml_adapter": config.use_baml_adapter,
        "num_samples": len(dev_data),
        "accuracy": accuracy,
        "correct": correct,
        "errors": len(errors),
        "predictions": {"Yes": yes_count, "No": no_count},
        "ground_truth": {"Yes": gt_yes, "No": gt_no},
        "token_usage": stats.summary(),
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "summary": summary,
        "results": results,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Run checkworthiness baseline experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        choices=list(MODELS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Threshold for Yes/No classification (default: 50.0)",
    )
    parser.add_argument(
        "--no-baml",
        action="store_true",
        help="Disable BAML adapter",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        model_name=args.model,
        threshold=args.threshold,
        use_baml_adapter=not args.no_baml,
        verbose=not args.quiet,
    )

    # Run experiment
    output = run_baseline_experiment(config, num_samples=args.samples)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    summary = output["summary"]
    print(f"Model: {summary['model']}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Correct: {summary['correct']}/{summary['num_samples']}")
    print(f"Errors: {summary['errors']}")
    print(f"Predictions: Yes={summary['predictions']['Yes']}, No={summary['predictions']['No']}")
    print(f"Ground Truth: Yes={summary['ground_truth']['Yes']}, No={summary['ground_truth']['No']}")
    print("\nTOKEN USAGE:")
    token_usage = summary["token_usage"]
    print(f"  Total Calls: {token_usage['total_calls']}")
    print(f"  Prompt Tokens: {token_usage['total_prompt_tokens']:,}")
    print(f"  Completion Tokens: {token_usage['total_completion_tokens']:,}")
    print(f"  Reasoning Tokens: {token_usage['total_reasoning_tokens']:,}")
    print(f"  Total Tokens: {token_usage['total_tokens']:,}")
    print(f"  Estimated Cost: ${token_usage['total_cost_usd']:.4f}")
    print("=" * 60)

    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"baseline_{args.model}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
