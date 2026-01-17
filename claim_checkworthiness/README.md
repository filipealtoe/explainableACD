# Claim Checkworthiness Assessment

Explainable claim checkworthiness assessment for the IJCAI 2026 paper. This package implements a three-module pipeline:

1. **Checkability** - Is this a factual assertion that can be fact-checked?
2. **Verifiability** - Can this claim be verified with publicly available data?
3. **Harm Potential** - What is the potential harm if this claim spreads unchecked?

## Installation

```bash
# Basic installation
pip install -e .

# With fine-tuning support (GPU)
pip install -e ".[finetune]"

# With visualization tools
pip install -e ".[visualization]"
```

## Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
XAI_API_KEY=xai-...
MOONSHOT_API_KEY=sk-...
TOGETHER_API_KEY=...
```

## Directory Structure

```
claim_checkworthiness/
├── src/checkworthiness/     # Core modules
│   ├── config.py            # Model configurations (34 models)
│   ├── schemas.py           # Pydantic output schemas
│   ├── modules.py           # DSPy modules
│   ├── prompting_baseline.py # Direct API baseline
│   ├── metrics.py           # Evaluation metrics (F1, F2, ECE)
│   ├── statistical_tests.py # Statistical analysis (McNemar, bootstrap)
│   └── predictor.py         # Ensemble predictor (DeBERTa + LLM)
├── scripts/                 # Experiment scripts (120+)
│   ├── runners/             # Core experiment runners
│   ├── finetuning/          # DeBERTa fine-tuning
│   ├── evaluation/          # Benchmark evaluation
│   ├── ensemble/            # Ensemble methods
│   ├── feature_engineering/ # Feature generation
│   ├── analysis/            # Error/result analysis
│   ├── hyperparameter_tuning/ # Grid/greedy search
│   ├── data_processing/     # Dataset preparation
│   └── utilities/           # Testing/debugging
├── prompts/                 # Prompt templates (v4 binary Yes/No)
├── baml_src/                # BAML type definitions
├── data/                    # Dataset storage
└── results/                 # Experiment outputs
```

## Quick Start

```python
from claim_checkworthiness.src.checkworthiness.config import MODELS
from claim_checkworthiness.src.checkworthiness.prompting_baseline import PromptingBaseline

# Initialize baseline with a model config
baseline = PromptingBaseline(model_config=MODELS["gpt-4o-mini"])

# Assess a claim
result = baseline.assess_claim("COVID-19 vaccines cause autism")

# Access confidence scores (0-100 scale)
print(f"Checkability: {result.checkability.confidence:.1f}%")
print(f"  - P(checkable): {result.checkability.p_true:.2f}")
print(f"  - Reasoning: {result.checkability.reasoning}")

print(f"Verifiability: {result.verifiability.confidence:.1f}%")
print(f"  - P(verifiable): {result.verifiability.p_true:.2f}")

print(f"Harm Potential: {result.harm_potential.confidence:.1f}%")
print(f"  - P(harmful): {result.harm_potential.p_true:.2f}")
print(f"  - Sub-scores:")
print(f"    - Social fragmentation: {result.harm_potential.sub_scores.social_fragmentation:.1f}%")
print(f"    - Spurs action: {result.harm_potential.sub_scores.spurs_action:.1f}%")
print(f"    - Believability: {result.harm_potential.sub_scores.believability:.1f}%")
print(f"    - Exploitativeness: {result.harm_potential.sub_scores.exploitativeness:.1f}%")
```

---

## Scripts

### 1. Prompting Baseline: `runners/run_prompting_baseline.py`

Run LLM-based checkworthiness assessment with logprob confidence extraction.

#### Basic Usage

```bash
# Run on 50 samples with default model (gpt-4o-mini)
python scripts/runners/run_prompting_baseline.py

# Run with specific model
python scripts/runners/run_prompting_baseline.py --model deepseek-v3.2 --samples 100

# Run all models with available API keys
python scripts/runners/run_prompting_baseline.py --all --samples 50

# Adjust classification threshold
python scripts/runners/run_prompting_baseline.py --model gpt-4o --threshold 60.0
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `gpt-4o-mini` | Model to use (34 options: gpt-4o, deepseek-v3.2, grok-4.1, kimi-k2, llama-3.3-70b, etc.) |
| `--all` | flag | - | Run all models with available API keys |
| `--samples` | int | `50` | Number of samples to evaluate |
| `--threshold` | float | `50.0` | Threshold for Yes/No classification |
| `--quiet` | flag | - | Reduce output verbosity |
| `--output` | str | auto | Output file path (default: auto-generated with timestamp) |

---

### 2. DeBERTa Fine-tuning: `finetuning/finetune_deberta.py`

Fine-tune DeBERTa-v3 for checkworthiness classification.

#### Basic Usage

```bash
# Default fine-tuning (3 epochs, batch size 16)
python scripts/finetuning/finetune_deberta.py

# Quick test run (1 epoch)
python scripts/finetuning/finetune_deberta.py --quick

# Custom hyperparameters
python scripts/finetuning/finetune_deberta.py \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5 \
    --output-dir checkpoints/deberta-custom
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `microsoft/deberta-v3-base` | Model name or path |
| `--epochs` | int | `3` | Number of training epochs |
| `--batch-size` | int | `16` | Batch size per device |
| `--grad-accum` | int | `1` | Gradient accumulation steps |
| `--lr` | float | `2e-5` | Learning rate |
| `--max-length` | int | `256` | Max sequence length |
| `--quick` | flag | - | Quick test (1 epoch, small eval) |
| `--output-dir` | str | `checkpoints/` | Output directory for checkpoints |

---

### 3. LLM Checkworthiness: `runners/run_llm_checkworthiness.py`

Generate LLM features (confidences, logprobs) for downstream classifiers.

#### Basic Usage

```bash
# Generate features for all splits
python scripts/runners/run_llm_checkworthiness.py --model gpt-4o-mini

# Generate for specific split
python scripts/runners/run_llm_checkworthiness.py --model deepseek-v3.2 --split dev

# Binary Yes/No mode (v4 prompts)
python scripts/runners/run_llm_checkworthiness_v4.py --model gpt-4o
```

---

### 4. Benchmark Evaluation: `evaluation/evaluate_deberta_*.py`

Evaluate fine-tuned models on CheckThat! benchmarks.

#### Basic Usage

```bash
# Evaluate on CT22 benchmark
python scripts/evaluation/evaluate_deberta_ct22.py --checkpoint checkpoints/best-model

# Evaluate on CT23 benchmark
python scripts/evaluation/evaluate_deberta_ct23.py --checkpoint checkpoints/best-model

# Evaluate on ClaimBuster dataset
python scripts/evaluation/evaluate_deberta_claimbuster.py --checkpoint checkpoints/best-model

# Compare multiple models
python scripts/evaluation/compare_models_ct24.py --models gpt-4o,deepseek-v3.2,grok-4.1
```

---

### 5. Ensemble Methods: `ensemble/ensemble_predictions.py`

Combine multiple model predictions.

#### Basic Usage

```bash
# Ensemble DeBERTa with different seeds
python scripts/ensemble/ensemble_deberta_seeds.py

# Mixed architecture ensemble
python scripts/ensemble/ensemble_mixed_architectures.py

# Find minimum viable ensemble
python scripts/ensemble/find_minimum_ensemble.py
```

---

### 6. Hyperparameter Tuning: `hyperparameter_tuning/*.py`

Grid search and optimization scripts.

#### Basic Usage

```bash
# Fast grid search
python scripts/hyperparameter_tuning/fast_grid_search.py

# Feature ablation study
python scripts/hyperparameter_tuning/feature_ablation_study.py

# LLM feature grid search
python scripts/hyperparameter_tuning/llm_feature_grid_search.py
```

---

## Output Schemas

### CheckworthinessResult

```python
class CheckworthinessResult:
    claim_text: str                    # Original claim
    checkability: CheckabilityOutput   # Module 1 output
    verifiability: VerifiabilityOutput # Module 2 output
    harm_potential: HarmPotentialOutput # Module 3 output
```

### Module Outputs (shared structure)

Each module output contains:

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | str | Step-by-step explanation |
| `confidence` | float (0-100) | Final confidence score |
| `p_true` | float (0-1) | Logprob P(true) |
| `p_false` | float (0-1) | Logprob P(false) |
| `p_uncertain` | float (0-1) | Logprob P(uncertain) |
| `entropy` | float | Shannon entropy of distribution |
| `logprobs_missing` | bool | True if logprobs unavailable |

### HarmPotentialOutput (additional fields)

| Field | Type | Description |
|-------|------|-------------|
| `sub_scores.social_fragmentation` | float (0-100) | Contributes to division |
| `sub_scores.spurs_action` | float (0-100) | Could cause harmful action |
| `sub_scores.believability` | float (0-100) | Misleadingly believable |
| `sub_scores.exploitativeness` | float (0-100) | Exploits vulnerabilities |

---

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4.1, gpt-5.2, gpt-4.1-mini |
| **DeepSeek** | deepseek-v3.2, deepseek-v3, deepseek-v3.1 |
| **xAI** | grok-4.1 |
| **Moonshot** | kimi-k2 |
| **Together AI** | llama-3.3-70b, llama-4-scout, mistral-small-24b, qwen-2.5-72b, etc. |

---

## Citation

```bibtex
@inproceedings{pinto2026explainable,
  title={Explainable Automatic Claim Detection},
  author={Pinto, Sérgio},
  booktitle={IJCAI 2026},
  year={2026}
}
```
