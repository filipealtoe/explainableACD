#!/usr/bin/env python3
"""
Simplest Fine-Tuning Script for Checkworthiness Classification

Converts CT24 binary labels to chat format and fine-tunes on Together AI.

Usage:
    # Step 1: Prepare data
    python experiments/scripts/finetune_together.py --prepare

    # Step 2: Start fine-tuning
    python experiments/scripts/finetune_together.py --train

    # Step 3: Check status
    python experiments/scripts/finetune_together.py --status JOB_ID

    # Step 4: Test the model
    python experiments/scripts/finetune_together.py --test MODEL_NAME
"""

import argparse
import json
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CT24_DIR = DATA_DIR / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent / "results"

# Together AI model to fine-tune
# Check available models: https://docs.together.ai/docs/fine-tuning-models
# Qwen 2.5 7B - good balance of quality vs cost for classification
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Alternative models (if 7B isn't available or you want to try others):
# BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"     # Better quality, 2x cost
# BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"     # Even better, 4x cost
# BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"  # Llama alternative

# =============================================================================
# Qwen 2.5 Optimized Prompts
# - Qwen excels at instruction following and structured output
# - Keep system prompt clear and role-focused
# - Qwen is resilient to prompt diversity, but benefits from explicit criteria
# =============================================================================

SYSTEM_PROMPT = """You are a fact-checking assistant. Your task is to determine if statements are checkworthy.

Checkworthy criteria:
- Makes a specific factual claim that can be verified
- Would matter to the public if false
- Contains verifiable information (names, numbers, events)

Not checkworthy:
- Personal opinions or preferences
- Vague statements without specific claims
- Questions, greetings, or casual talk
- Future predictions
- Obvious common knowledge

Always respond with exactly: Yes or No"""

USER_TEMPLATE = """Statement: {text}

Is this statement checkworthy?"""


# =============================================================================
# Step 1: Prepare Data (Convert to JSONL)
# =============================================================================

def prepare_data(subset_size: int | None = None):
    """Convert CT24 train data to Together AI fine-tuning format.

    Args:
        subset_size: If set, only use this many samples (for cheap testing).
    """
    print("=" * 60)
    print("STEP 1: Preparing fine-tuning data")
    print("=" * 60)

    # Load training data
    train = pl.read_parquet(CT24_DIR / "CT24_train_features.parquet")
    print(f"Loaded {len(train)} training samples")

    # Optional: use subset for initial testing
    if subset_size and subset_size < len(train):
        # Stratified sample to keep class balance
        train = train.sample(n=subset_size, shuffle=True, seed=42)
        print(f"Using subset: {subset_size} samples (for cost-effective testing)")

    # Check class distribution
    pos = (train["class_label"] == "Yes").sum()
    neg = (train["class_label"] == "No").sum()
    print(f"  Yes: {pos} ({100*pos/len(train):.1f}%)")
    print(f"  No:  {neg} ({100*neg/len(train):.1f}%)")

    # Determine text column
    text_col = "cleaned_text" if "cleaned_text" in train.columns else "Text"

    # Convert to JSONL format
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ct24_finetune.jsonl"

    with open(output_path, "w") as f:
        for row in train.iter_rows(named=True):
            # Each sample is a simple conversation
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=row[text_col])},
                    {"role": "assistant", "content": row["class_label"]}  # "Yes" or "No"
                ]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show example
    print("\n" + "-" * 60)
    print("Example entry:")
    print("-" * 60)
    with open(output_path) as f:
        example = json.loads(f.readline())
    print(json.dumps(example, indent=2))

    return output_path


# =============================================================================
# Step 2: Upload & Train
# =============================================================================

def start_training(data_path: Path):
    """Upload data and start fine-tuning job on Together AI."""
    print("=" * 60)
    print("STEP 2: Starting fine-tuning on Together AI")
    print("=" * 60)

    import os
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set in environment")
        return

    client = Together(api_key=api_key)

    # Upload file
    print(f"\nUploading {data_path.name}...")
    file_response = client.files.upload(file=str(data_path))
    file_id = file_response.id
    print(f"  File ID: {file_id}")

    # Start fine-tuning
    print(f"\nStarting fine-tuning job...")
    print(f"  Base model: {BASE_MODEL}")

    job = client.fine_tuning.create(
        training_file=file_id,
        model=BASE_MODEL,
        n_epochs=3,
        batch_size="max",
        learning_rate=1e-5,
        lora=True,  # LoRA is faster and cheaper
        suffix="checkworthy",  # Your model will be named with this suffix
    )

    print(f"\n✓ Job started!")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"\nCheck status with:")
    print(f"  python experiments/scripts/finetune_together.py --status {job.id}")

    return job.id


# =============================================================================
# Step 3: Check Status
# =============================================================================

def check_status(job_id: str):
    """Check the status of a fine-tuning job."""
    import os
    from together import Together

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    job = client.fine_tuning.retrieve(job_id)

    print("=" * 60)
    print(f"Job Status: {job.status}")
    print("=" * 60)
    print(f"  Job ID:     {job.id}")
    print(f"  Model:      {job.model}")
    print(f"  Status:     {job.status}")

    if hasattr(job, "output_name") and job.output_name:
        print(f"  Output:     {job.output_name}")
        print(f"\nYour fine-tuned model is ready!")
        print(f"Test with:")
        print(f"  python experiments/scripts/finetune_together.py --test {job.output_name}")

    if job.status == "running":
        print(f"\n  Still training... check again later.")

    return job


# =============================================================================
# Step 4: Test the Model
# =============================================================================

def test_model(model_name: str):
    """Test the fine-tuned model on a few examples."""
    import os
    import time
    from together import Together
    from together.error import ServiceUnavailableError

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    # Test examples
    test_claims = [
        "COVID-19 vaccines contain microchips.",
        "The weather today is sunny.",
        "Election fraud occurred in multiple states.",
        "I love pizza.",
        "5G towers spread coronavirus.",
    ]

    print("=" * 60)
    print(f"Testing model: {model_name}")
    print("=" * 60)

    for claim in test_claims:
        # Retry logic for cold start / 503 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(text=claim)},
                    ],
                    max_tokens=5,
                    temperature=0,
                )
                answer = response.choices[0].message.content.strip()
                print(f"\n  Claim: {claim}")
                print(f"  → {answer}")
                break
            except ServiceUnavailableError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                    print(f"\n  ⏳ Model warming up... retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n  ❌ Model unavailable after {max_retries} attempts.")
                    print(f"     Error: {e}")
                    print(f"     Try again later or check Together AI dashboard for deployment status.")
                    return


# =============================================================================
# Step 5: Evaluate on Test Set
# =============================================================================

def evaluate_model(model_name: str):
    """Evaluate the fine-tuned model on CT24 test set."""
    import os
    from together import Together
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    # Load test data
    test = pl.read_parquet(CT24_DIR / "CT24_test_features.parquet")
    text_col = "cleaned_text" if "cleaned_text" in test.columns else "Text"

    print("=" * 60)
    print(f"Evaluating: {model_name}")
    print(f"Test set: {len(test)} samples")
    print("=" * 60)

    predictions = []
    actuals = []

    for i, row in enumerate(test.iter_rows(named=True)):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=row[text_col])},
            ],
            max_tokens=5,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()

        # Parse response
        pred = "Yes" if "yes" in answer.lower() else "No"
        predictions.append(pred)
        actuals.append(row["class_label"])

        if (i + 1) % 50 == 0:
            y_true = [1 if a == "Yes" else 0 for a in actuals]
            y_pred = [1 if p == "Yes" else 0 for p in predictions]
            interim_f1 = f1_score(y_true, y_pred)
            print(f"  Progress: {i+1}/{len(test)} | Interim F1: {interim_f1:.4f}")

    # Final metrics
    y_true = [1 if a == "Yes" else 0 for a in actuals]
    y_pred = [1 if p == "Yes" else 0 for p in predictions]

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"\n  SOTA F1:   0.82")
    print(f"  Gap:       {f1_score(y_true, y_pred) - 0.82:+.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["No", "Yes"]))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral on CT24")
    parser.add_argument("--prepare", action="store_true", help="Prepare JSONL data")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use subset of training data (e.g., 1000 for cheap test)")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning")
    parser.add_argument("--status", type=str, help="Check job status")
    parser.add_argument("--test", type=str, help="Test model on examples")
    parser.add_argument("--evaluate", type=str, help="Evaluate on full test set")
    args = parser.parse_args()

    if args.prepare:
        prepare_data(subset_size=args.subset)
    elif args.train:
        data_path = OUTPUT_DIR / "ct24_finetune.jsonl"
        if not data_path.exists():
            print("Data not found. Run --prepare first.")
            return
        start_training(data_path)
    elif args.status:
        check_status(args.status)
    elif args.test:
        test_model(args.test)
    elif args.evaluate:
        evaluate_model(args.evaluate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
