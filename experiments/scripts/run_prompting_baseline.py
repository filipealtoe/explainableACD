"""Run prompting baseline experiment (direct API calls, no DSPy).

This script:
1. Loads the CT24 checkworthy dataset
2. Runs the prompting baseline pipeline on a subset
3. Tracks token usage and costs
4. Evaluates accuracy using simple average formula
5. Saves results
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.checkworthiness.config import (
    MODELS,
    ExperimentConfig,
    ExperimentStats,
)
from src.checkworthiness.prompting_baseline import PromptingBaseline
from src.checkworthiness.prompting_baseline import load_prompts


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


def run_prompting_baseline_experiment(
    config: ExperimentConfig,
    num_samples: int | None = None,
) -> dict:
    """Run prompting baseline experiment without DSPy."""
    load_dotenv()

    if config.verbose:
        config.print_config()
        prompts = load_prompts()
        check_max = int(prompts["checkability"].get("max_tokens", config.get_model_config().max_tokens))
        verif_max = int(prompts["verifiability"].get("max_tokens", config.get_model_config().max_tokens))
        harm_max = int(prompts["harm_potential"].get("max_tokens", config.get_model_config().max_tokens))
        print("\n[PROMPT MAX TOKENS]")
        print(f"  checkability: {check_max}")
        print(f"  verifiability: {verif_max}")
        print(f"  harm_potential: {harm_max}")
        print()

    model_config = config.get_model_config()

    data_path = Path(config.data_dir)
    dev_data = load_dataset(data_path / "CT24_checkworthy_english_dev.tsv")

    if num_samples:
        dev_data = dev_data[:num_samples]

    print(f"Running on {len(dev_data)} samples from dev set\n")

    pipeline = PromptingBaseline(model_config, threshold=config.threshold)

    stats = ExperimentStats()
    results = []
    correct = 0
    errors = []

    for item in tqdm(dev_data, desc="Processing claims"):
        try:
            result, usage = pipeline(item["text"])

            stats.add_usage(usage, model_config)

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
        except Exception as exc:
            errors.append(
                {
                    "sentence_id": item["sentence_id"],
                    "claim": item["text"],
                    "error": str(exc),
                }
            )
            if config.verbose:
                print(f"\nError processing claim {item['sentence_id']}: {exc}")

    accuracy = correct / len(dev_data) if dev_data else 0.0

    yes_count = sum(1 for r in results if r["prediction"] == "Yes")
    no_count = sum(1 for r in results if r["prediction"] == "No")
    gt_yes = sum(1 for item in dev_data if item["label"] == "Yes")
    gt_no = sum(1 for item in dev_data if item["label"] == "No")

    summary = {
        "experiment": "prompting_baseline",
        "model": config.model_name,
        "model_config": {
            "provider": model_config.provider.value,
            "api_base": model_config.api_base,
            "temperature": config.temperature,
            "max_tokens": model_config.max_tokens,
            "is_thinking_model": model_config.is_thinking_model,
        },
        "threshold": config.threshold,
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


def print_summary(summary: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
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


def save_results(output: dict, output_path: Path) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompting baseline experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        choices=list(MODELS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models with available API keys",
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
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for a single model run (default: auto-generated)",
    )

    args = parser.parse_args()

    if args.all and args.output:
        print("Warning: --output ignored when --all is set.")

    model_names = list(MODELS.keys()) if args.all else [args.model]

    for model_name in model_names:
        model_config = MODELS[model_name]
        if not model_config.get_api_key():
            print(f"Skipping {model_name}: missing {model_config.api_key_env}")
            continue

        config = ExperimentConfig(
            model_name=model_name,
            threshold=args.threshold,
            use_baml_adapter=False,
            verbose=not args.quiet,
        )

        output = run_prompting_baseline_experiment(config, num_samples=args.samples)
        summary = output["summary"]
        print_summary(summary)

        results_dir = Path(config.results_dir)
        if args.output and not args.all:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"prompting_baseline_{model_name}_{timestamp}.json"

        save_results(output, output_path)
        print(f"\nResults saved to: {output_path}\n")


if __name__ == "__main__":
    main()
