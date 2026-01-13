#!/usr/bin/env python3
"""
GPT batch analysis for CT24 error cases.

Reads error_analysis_<split>_<timestamp>.tsv files, batches rows,
and asks GPT-5.2 to analyze patterns and suggest improvements.

Outputs:
  - gpt_error_analysis_<split>_<timestamp>.jsonl (one JSON per batch)
  - gpt_error_summary_<split>_<timestamp>.json  (aggregated counts)
  - gpt_error_failures_<split>_<timestamp>.jsonl (raw responses if parsing fails)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.checkworthiness.config import MODELS

DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ct24_classifier_pca_32"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_embeddings" / "pca_32"


def resolve_model(model_name: str) -> tuple[str, bool, int]:
    if model_name in MODELS:
        cfg = MODELS[model_name]
        return cfg.model_name, cfg.uses_max_completion_tokens, cfg.max_tokens
    for cfg in MODELS.values():
        if cfg.model_name == model_name:
            return cfg.model_name, cfg.uses_max_completion_tokens, cfg.max_tokens
    uses_max_completion_tokens = model_name.startswith("gpt-5")
    return model_name, uses_max_completion_tokens, 1024


def chunk_list(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def build_prompt(
    split: str,
    batch_id: str,
    threshold: float,
    items: list[dict[str, Any]],
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert error analyst for claim checkworthiness detection. "
        "Your job is to find recurring patterns in misclassified claims and suggest "
        "features or prompt changes that could reduce these errors. "
        "Return ONLY valid JSON. No markdown, no commentary."
    )

    tag_options = [
        "opinion_or_value_judgment",
        "prediction_or_future",
        "vague_or_ambiguous",
        "numeric_or_statistical",
        "multi_claim",
        "requires_external_context",
        "private_or_unverifiable",
        "named_entity_disambiguation",
        "temporal_or_time_sensitive",
        "negation_or_polarity",
        "causal_or_attribution",
        "comparison_or_superlative",
        "question_or_command",
        "sarcasm_or_irony",
        "quote_or_reported_speech",
    ]

    payload = {
        "split": split,
        "batch_id": batch_id,
        "threshold": round(threshold, 2),
        "items": items,
        "tag_options": tag_options,
    }

    user_prompt = (
        "Analyze the following misclassified claims.\n"
        "Rules:\n"
        "- Use only the provided data.\n"
        "- Choose tags from tag_options (you may add at most 2 new tags if necessary).\n"
        "- Keep strings concise.\n"
        "- Output JSON with the exact schema below.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "split": string,\n'
        '  "batch_id": string,\n'
        '  "n_items": integer,\n'
        '  "recurring_patterns": [string],\n'
        '  "feature_gaps": [string],\n'
        '  "suggested_features": [string],\n'
        '  "suggested_prompt_changes": [string],\n'
        '  "labeling_ambiguities": [string],\n'
        '  "items": [\n'
        "    {\n"
        '      "sample_id": string,\n'
        '      "error_type": "FP" or "FN",\n'
        '      "tags": [string],\n'
        '      "reason": string,\n'
        '      "fix_hint": string\n'
        "    }\n"
        "  ],\n"
        '  "summary": string\n'
        "}\n\n"
        f"Data:\n{json.dumps(payload, ensure_ascii=True)}"
    )

    return system_prompt, user_prompt


def call_gpt(
    client: OpenAI,
    model_name: str,
    uses_max_completion_tokens: bool,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    api_params: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    if uses_max_completion_tokens:
        api_params["max_completion_tokens"] = max_tokens
    else:
        api_params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**api_params)
    return response.choices[0].message.content or ""


def summarize_batches(batch_jsons: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "batches": len(batch_jsons),
        "tag_counts": Counter(),
        "recurring_patterns": Counter(),
        "feature_gaps": Counter(),
        "suggested_features": Counter(),
        "suggested_prompt_changes": Counter(),
        "labeling_ambiguities": Counter(),
    }

    for batch in batch_jsons:
        for item in batch.get("items", []):
            for tag in item.get("tags", []):
                summary["tag_counts"][tag] += 1
        for key in [
            "recurring_patterns",
            "feature_gaps",
            "suggested_features",
            "suggested_prompt_changes",
            "labeling_ambiguities",
        ]:
            for value in batch.get(key, []):
                summary[key][value] += 1

    for key in [
        "tag_counts",
        "recurring_patterns",
        "feature_gaps",
        "suggested_features",
        "suggested_prompt_changes",
        "labeling_ambiguities",
    ]:
        summary[key] = dict(summary[key].most_common())

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT batch analysis for CT24 errors")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory containing error_analysis files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory used to generate error analysis files",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp to analyze (default: latest in results dir)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help="Comma-separated splits to analyze",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for GPT analysis (default: 12)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows per split (default: no limit)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="Model name (default: gpt-5.2-2025-12-11)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between batches to avoid rate limits",
    )

    args = parser.parse_args()
    results_dir = args.results_dir

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    model_name, uses_max_completion_tokens, max_tokens = resolve_model(args.model)
    client = OpenAI(api_key=api_key)

    if args.timestamp:
        timestamp = args.timestamp
    else:
        metrics_files = sorted(results_dir.glob("metrics_summary_*.json"))
        if not metrics_files:
            raise FileNotFoundError(f"No metrics_summary_*.json in {results_dir}")
        timestamp = metrics_files[-1].stem.replace("metrics_summary_", "")

    metrics_path = results_dir / f"metrics_summary_{timestamp}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    threshold = metrics.get("threshold_tuned", 0.50)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits provided. Use --splits train,dev,test")

    missing = [
        split
        for split in splits
        if not (results_dir / f"error_analysis_{split}_{timestamp}.tsv").exists()
    ]
    if missing:
        analyze_script = REPO_ROOT / "experiments" / "scripts" / "analyze_ct24_errors.py"
        cmd = [
            sys.executable,
            str(analyze_script),
            "--results-dir",
            str(results_dir),
            "--data-dir",
            str(args.data_dir),
            "--splits",
            ",".join(missing),
            "--timestamp",
            timestamp,
        ]
        print(f"⚠️  Missing error files for {missing}. Generating them now...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate error analysis files:\n{result.stderr}"
            )
        if result.stdout:
            print(result.stdout)

    for split in splits:
        errors_path = results_dir / f"error_analysis_{split}_{timestamp}.tsv"
        if not errors_path.exists():
            raise FileNotFoundError(f"Missing error analysis file: {errors_path}")

        df = pd.read_csv(errors_path, sep="\t")
        if args.max_rows is not None:
            df = df.head(args.max_rows)

        items = []
        for _, row in df.iterrows():
            items.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "error_type": row["error_type"],
                    "gold_label": row["gold_label"],
                    "pred_label": row["pred_label"],
                    "pred_proba_yes": float(row["pred_proba_yes"]),
                    "decision_margin": float(row["decision_margin"]),
                    "text": str(row["text"]),
                }
            )

        batches = chunk_list(items, args.batch_size)
        output_jsonl = results_dir / f"gpt_error_analysis_{split}_{timestamp}.jsonl"
        output_summary = results_dir / f"gpt_error_summary_{split}_{timestamp}.json"
        output_failures = results_dir / f"gpt_error_failures_{split}_{timestamp}.jsonl"

        parsed_batches: list[dict[str, Any]] = []

        with open(output_jsonl, "w") as out_f, open(output_failures, "w") as fail_f:
            for i, batch in enumerate(batches):
                batch_id = f"{split}_batch_{i:03d}"
                system_prompt, user_prompt = build_prompt(
                    split=split,
                    batch_id=batch_id,
                    threshold=threshold,
                    items=batch,
                )

                try:
                    response_text = call_gpt(
                        client=client,
                        model_name=model_name,
                        uses_max_completion_tokens=uses_max_completion_tokens,
                        max_tokens=max_tokens,
                        temperature=args.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                except Exception as exc:
                    fail_f.write(
                        json.dumps(
                            {
                                "split": split,
                                "batch_id": batch_id,
                                "error": str(exc),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    continue

                parsed = parse_json(response_text)
                if parsed is None:
                    fail_f.write(
                        json.dumps(
                            {
                                "split": split,
                                "batch_id": batch_id,
                                "error": "json_parse_failed",
                                "raw_response": response_text,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                else:
                    parsed_batches.append(parsed)
                    out_f.write(json.dumps(parsed, ensure_ascii=True) + "\n")

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        summary = summarize_batches(parsed_batches)
        summary.update(
            {
                "split": split,
                "timestamp": timestamp,
                "model": model_name,
                "batch_size": args.batch_size,
                "threshold": round(threshold, 2),
                "total_items": len(items),
            }
        )

        with open(output_summary, "w") as f:
            json.dump(summary, f, indent=2)

        print("=" * 70)
        print("GPT ERROR ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Split: {split}")
        print(f"Model: {model_name}")
        print(f"Batches: {len(batches)}")
        print(f"Output: {output_jsonl}")
        print(f"Summary: {output_summary}")
        print(f"Failures: {output_failures}")


if __name__ == "__main__":
    main()
