# Retrieval-Augmented Few-Shot Claim Normalization

This repository contains the code for reproducing the claim normalization experiments submitted to **CheckThat! 2025 Task 2** and the accompanying **IJCAI 2026** paper.

## Overview

Our approach uses **retrieval-augmented few-shot prompting** with a two-stage verification pipeline:

1. **Post-to-post similarity**: Find training posts similar to the query
2. **Claim-to-post verification**: Verify retrieved claims are relevant to the query content
3. **LLM normalization**: Generate normalized claim using few-shot examples from the same topic cluster

Key features:
- Topic-aware few-shot example selection via K-means clustering
- Contrastive learning with good/bad examples
- Hybrid retrieval (embedding + BM25 + entity overlap)
- Support for both API-based and local GPU inference

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# For fine-tuning support
pip install -e ".[finetune]"

# Download NLTK data (for METEOR evaluation)
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Data Setup

Download the CheckThat! 2025 Task 2 dataset from the [official repository](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task2) and place files in the `data/` directory:

```
data/
├── train-eng.csv    # Training set (11,374 samples)
├── dev-eng.csv      # Development set (1,171 samples)
└── test-eng.csv     # Test set
```

Each CSV file should have two columns:
- `post`: The original social media post
- `normalized claim`: The ground-truth normalized claim

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for API-based inference
TOGETHER_API_KEY=your_together_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional

# For local GPU inference, no API keys needed
```

## Directory Structure

```
claim_norm/
├── scripts/
│   ├── run_claim_normalization_ct25.py   # Main inference script
│   ├── tune_fewshot_hyperparams.py       # Hyperparameter tuning
│   └── finetune_claim_normalizer.py      # QLoRA fine-tuning
├── src/
│   ├── checkworthiness/
│   │   └── config.py                     # Model configurations
│   └── pipeline/
│       └── modules/
│           ├── embedder.py               # Sentence embeddings
│           └── clusterer.py              # Topic clustering
├── prompts/
│   └── claim_normalization.yaml          # System prompt
├── data/                                 # Place dataset here
├── pyproject.toml
└── README.md
```

---

## Scripts

### 1. Main Inference: `run_claim_normalization_ct25.py`

Run few-shot claim normalization with retrieval-augmented prompting.

#### Basic Usage

```bash
# Run on dev set with default settings (mixtral-8x7b, 3 examples)
python scripts/run_claim_normalization_ct25.py --split dev

# Run on test set with specific model
python scripts/run_claim_normalization_ct25.py --model mistral-7b-v0.3 --split test

# Zero-shot baseline (no examples)
python scripts/run_claim_normalization_ct25.py --zero-shot --split dev

# Single claim inference (no dataset needed)
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --claim "Biden won the 2020 election with 81 million votes"
```

#### Advanced Usage

```bash
# Topic-clustered few-shot selection (10 clusters)
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split dev \
    --topic-clusters 10 \
    --num-examples 5

# With contrastive examples (good + bad examples)
python scripts/run_claim_normalization_ct25.py \
    --model mixtral-8x7b \
    --split dev \
    --contrastive-file path/to/contrastive.jsonl \
    --contrastive-examples 2

# Retrieval-only mode (bypass LLM for high-similarity matches)
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split dev \
    --retrieval-threshold 0.85 \
    --claim-verify-threshold 0.5

# Local GPU inference (no API needed)
python scripts/run_claim_normalization_ct25.py \
    --local \
    --local-model mistralai/Mistral-7B-Instruct-v0.3 \
    --split dev
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `mixtral-8x7b` | Model name (see `config.py` for options) |
| `--split` | str | `dev` | Dataset split: `dev`, `test`, or `train` |
| `--claim` | str | None | Single claim to normalize (skips dataset, prints result) |
| `--limit` | int | None | Limit number of samples (for testing) |
| `--parallel` | int | 10 | Number of parallel API requests |
| `--rate-limit` | float | 600.0 | Max requests per minute |
| `--timeout` | float | 30.0 | Request timeout in seconds |
| `--no-resume` | flag | False | Start fresh, ignore existing results |
| `--no-save` | flag | False | Don't save results to file |
| `--compare-models` | flag | False | Run comparison across all models |

**Few-shot options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-examples` | int | 3 | Number of few-shot examples |
| `--zero-shot` | flag | False | Run without examples |
| `--topic-clusters` | int | 0 | Number of K-means clusters (0=disabled) |
| `--filtered-examples` | flag | False | Use quality-filtered examples |

**Retrieval options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--retrieval-threshold` | float | 0.85 | Post similarity threshold for retrieval fallback |
| `--claim-verify-threshold` | float | 0.5 | Claim-post verification threshold |
| `--retrieval-mode` | str | `embedding` | Mode: `embedding`, `bm25`, or `hybrid` |

**Contrastive learning:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contrastive-file` | str | None | Path to contrastive examples JSONL |
| `--contrastive-examples` | int | 1 | Number of contrastive pairs per query |
| `--contrastive-mode` | str | `similar` | Selection mode: `similar` or `random` |
| `--auto-contrastive` | flag | False | Automatically run two-stage contrastive learning |

The `--auto-contrastive` flag automates the two-stage contrastive learning process:
1. Checks if baseline results exist at `results/{model}_{split}.jsonl`
2. If not, runs a baseline pass first (without contrastive examples)
3. Then runs the contrastive pass using the baseline results

Example:
```bash
# Single command runs both stages automatically
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split dev \
    --auto-contrastive \
    --contrastive-examples 2
```

**Full pipeline mode (streaming pipeline data):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--full-pipeline` | flag | False | Load data from streaming pipeline output instead of CT25 benchmark |
| `--pipeline-dir` | str | None | Path to pipeline output directory (required if `--full-pipeline`) |
| `--top-k` | int | 5 | Number of representative tweets per cluster for normalization |

Use `--full-pipeline` to normalize claims from streaming pipeline output (parquet files) instead of the CT25 benchmark CSVs. The script will:
1. Load `claims.parquet` and `tweets.parquet` from `--pipeline-dir`
2. Concatenate top-K tweets per cluster as input
3. Auto-discover contrastive file (`{model}_train.jsonl`) if available
4. Update `claims.parquet` with normalized `claim_text`

Example:
```bash
python scripts/run_claim_normalization_ct25.py \
    --model deepseek-v3.1 \
    --full-pipeline \
    --pipeline-dir data/pipeline_output \
    --topic-clusters 0 \
    --retrieval-threshold 0.85 \
    --claim-verify-threshold 0.65 \
    --num-examples 5 \
    --contrastive-examples 2
```

> **Note:** Do not use `--auto-contrastive` with `--full-pipeline`. The contrastive file is auto-discovered in pipeline mode.

**Local GPU inference:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--local` | flag | False | Use local GPU instead of API |
| `--local-model` | str | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace model ID |
| `--batch-size` | int | 8 | Batch size for local inference |

**Ensemble:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ensemble` | int | 1 | Number of ensemble generations |
| `--ensemble-temperature` | float | 0.7 | Temperature for ensemble diversity |

---

### 2. Hyperparameter Tuning: `tune_fewshot_hyperparams.py`

Tune hyperparameters using grid search or binary search (coordinate descent).

#### Basic Usage

```bash
# Grid search with default ranges
python scripts/tune_fewshot_hyperparams.py --model mixtral-8x7b

# Binary search (faster, fewer evaluations)
python scripts/tune_fewshot_hyperparams.py --model mixtral-8x7b --search binary

# Custom search ranges
python scripts/tune_fewshot_hyperparams.py \
    --model deepseek-chat \
    --topic-clusters "0,10,20" \
    --retrieval-threshold "0.8,0.85,0.9" \
    --num-positive "3,4,5" \
    --num-negative "1,2"
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `deepseek-chat` | Model to tune |
| `--split` | str | `dev` | Dataset split for evaluation |
| `--limit` | int | None | Limit samples per evaluation |
| `--search` | str | `grid` | Search strategy: `grid` or `binary` |
| `--max-evals` | int | None | Maximum evaluations (grid search) |
| `--max-iterations` | int | 3 | Iterations for binary search |
| `--output` | str | None | Output file for results |

**Search ranges (comma-separated values):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--topic-clusters` | str | `0,10,20,30` | K-means cluster counts to try |
| `--retrieval-threshold` | str | `0.7,0.8,0.85,0.9,1.0` | Retrieval thresholds |
| `--num-positive` | str | `2,3,4,5` | Positive example counts |
| `--num-negative` | str | `0,1,2,3` | Negative example counts |

**Contrastive tuning:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contrastive-file` | str | None | Path to contrastive examples |

**Parallelization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--parallel` | int | 10 | Parallel requests |
| `--rate-limit` | float | 600.0 | Rate limit (req/min) |
| `--timeout` | float | 30.0 | Request timeout |

---

### 3. Fine-Tuning: `finetune_claim_normalizer.py`

QLoRA fine-tuning of language models for claim normalization.

#### Basic Usage

```bash
# Fine-tune Mistral-7B with auto GPU detection
python scripts/finetune_claim_normalizer.py --model mistral-7b

# Specify GPU preset
python scripts/finetune_claim_normalizer.py --model mistral-7b --gpu a10

# Custom training parameters
python scripts/finetune_claim_normalizer.py \
    --model qwen-7b \
    --epochs 5 \
    --lr 1e-4 \
    --batch-size 4
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `mistral-7b` | Model: `mistral-7b`, `qwen-7b`, `llama-8b`, `qwen-14b`, `mixtral-8x7b`, `qwen-72b`, `llama-70b` |
| `--gpu` | str | auto | GPU preset: `rtx5090`, `gh200`, `a100`, `a10`, `rtx4090` |
| `--data-dir` | str | `data` | Directory containing train/dev CSVs |
| `--output-dir` | str | `checkpoints` | Output directory for checkpoints |
| `--epochs` | int | 3 | Training epochs |
| `--batch-size` | int | auto | Batch size (auto-detected per GPU) |
| `--gradient-accumulation` | int | auto | Gradient accumulation steps |
| `--lr` | float | 2e-4 | Learning rate |
| `--max-length` | int | 2048 | Maximum sequence length |
| `--resume` | str | None | Path to checkpoint to resume from |

---

## Configuration

### Model Configuration (`src/checkworthiness/config.py`)

Supported models and their API endpoints:

| Model | Provider | API Key Env Var |
|-------|----------|-----------------|
| `gpt-4o`, `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `mixtral-8x7b`, `mistral-7b-v0.3` | Together AI | `TOGETHER_API_KEY` |
| `deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` |
| `qwen-2.5-72b`, `llama-3.1-70b` | Together AI | `TOGETHER_API_KEY` |

### Prompts (`prompts/claim_normalization.yaml`)

The system prompt defines:
- What constitutes a normalized claim
- Rules for claim extraction and normalization
- JSON output schema

---

## Evaluation

Results are automatically evaluated using **METEOR score** (the official CheckThat! metric). Results are saved as JSONL files:

```
results/
├── mixtral-8x7b_dev.jsonl
├── mistral-7b-v0.3_test.jsonl
└── contrastive_mixtral-8x7b_dev.jsonl
```

Each line contains:
```json
{
  "idx": 0,
  "post": "Original post text...",
  "ground_truth": "Ground truth normalized claim",
  "prediction": "Model's normalized claim",
  "meteor": 0.85,
  "retrieval_hit": false,
  "latency_ms": 1250,
  "cost_usd": 0.0002
}
```

---

## Reproducing Paper Results

### Best Configuration (METEOR = 0.4679)

```bash
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split test \
    --num-examples 5 \
    --topic-clusters 10 \
    --retrieval-threshold 0.85 \
    --claim-verify-threshold 0.5
```

### Ablation: Pure LLM (no retrieval)

```bash
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split test \
    --retrieval-threshold 1.0
```

### Ablation: Pure Retrieval (no LLM)

```bash
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split test \
    --retrieval-threshold 0.0
```

### Zero-shot Baseline

```bash
python scripts/run_claim_normalization_ct25.py \
    --model mistral-7b-v0.3 \
    --split test \
    --zero-shot
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{author2026claim,
  title={Retrieval-Augmented Few-Shot Claim Normalization},
  author={Author Names},
  booktitle={Proceedings of IJCAI 2026},
  year={2026}
}
```

---

## License

MIT License
