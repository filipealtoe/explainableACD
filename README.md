# Quick Guide

## How to Use This Repo

1. **Run experiments**: Execute `python experiments/scripts/run_streaming_bertopic.py` - it automatically logs everything to MLflow

2. **View experiment history**: Run `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns` in a separate terminal and browse `http://localhost:5000` to see all runs, parameters, and results

3. **Explore results interactively**: Run `marimo edit notebooks/explore_topics.py` to play with visualizations - it reads from `experiments/results/output/`

4. **Document findings**: Open `research_notes/` in Obsidian and create daily notes linking concepts with `[[wiki-links]]`

5. **Key insight**: Nothing connects directly - experiment scripts write files, MLflow UI and Marimo notebooks just read those files

## Installation

```bash
uv sync
uv sync --extra dev
```

## Code Quality

```bash
ruff check .
mypy src/
```
# Check-worthiness Classifier

## LLMs tested directly for Check-worthiness (all have training cutoff dates prior to CheckThat! 2024 Task 1 dataset publishing date)

DEV vs TEST COMPARISON
qwen-2.5-72b: Dev F1 0.836 → Test F1 0.689 (Δ -0.147)
mistral-small-24b: Dev F1 0.822 → Test F1 0.677 (Δ -0.145)
llama-3.1-70b: Dev F1 0.813 → Test F1 0.635 (Δ -0.178)
llama-3.3-70b: Dev F1 0.798 → Test F1 0.633 (Δ -0.165)
gpt-4o-mini: Dev F1 0.790 → Test F1 0.607 (Δ -0.183)
gpt-3.5-turbo: Dev F1 0.780 → Test F1 0.603 (Δ -0.177)
mixtral-8x7b: Dev F1 0.837 → Test F1 0.585 (Δ -0.252)
RANKING (by Test F1):

qwen-2.5-72b — 0.689 ✅ Best generalization
mistral-small-24b — 0.677 ✅
llama-3.1-70b — 0.635
llama-3.3-70b — 0.633
gpt-4o-mini — 0.607
gpt-3.5-turbo — 0.603
mixtral-8x7b — 0.585 ❌ Worst generalization

Δ: Difference between ChechThat1 2024 Lab 1 F1 leader model and the obtained F1 with LLM prompting.

## SOTA History - CT24 Check-worthiness Classification

**Session Date:** January 2026
**Dataset:** CT24 Checkworthy English (ClaimBuster Task)
**Test Set:** 341 samples
**Metric:** F1-score (positive class = "Yes" / checkworthy)

---

## Results Summary

| Rank | Method | F1 | Threshold | Key Config |
|------|--------|-----|-----------|------------|
| 1 | DeBERTa Ensemble + LLM Fusion | **0.8362** | 0.50 | T=0.3, W=0.5 |
| 2 | DeBERTa 3-Seed Ensemble | 0.8343 | 0.55 | T=0.5 |
| 3 | DeBERTa Single Model (best) | 0.8242 | 0.50 | seed=0 |

---

## Method 1: DeBERTa Ensemble + LLM Fusion (BEST)

**F1 = 0.8362**

### Architecture
```
DeBERTa seed_0  ─┐
DeBERTa seed_42 ─┼─► Temperature Scaling (T=0.3) ─► Mean ─┐
DeBERTa seed_456─┘                                        │
                                                          ├─► 0.5×DeBERTa + 0.5×LLM ─► threshold@0.50
LLM Features (24) ─► XGBoost ────────────────────────────┘
```

### DeBERTa Training Configuration

**Base Model:** `microsoft/deberta-v3-large` (435M parameters)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Seeds | 0, 42, 456 | Diversity for ensemble |
| Max Sequence Length | 128 tokens | Sufficient for tweet-length claims |
| Epochs | 3 | Early stopping on dev F1 |
| Batch Size | 16 | Memory-efficient on single GPU |
| Learning Rate | 2e-5 | Standard for transformer fine-tuning |
| Weight Decay | 0.01 | L2 regularization |
| Warmup Ratio | 0.1 | 10% of steps for LR warmup |
| Optimizer | AdamW | Decoupled weight decay |
| LR Scheduler | Linear with warmup | Gradual decay after warmup |
| Loss Function | Focal Loss (γ=2, α=0.25) | Handles class imbalance, focuses on hard examples |
| Gradient Clipping | 1.0 | Prevents exploding gradients |

**Training Infrastructure:**
- GPU: NVIDIA A10 (Lambda Labs)
- Training Time: ~2-3 hours per seed
- Framework: HuggingFace Transformers + PyTorch

### Ensemble Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Temperature (T) | 0.3 | Sharpens predictions (models were underconfident) |
| Aggregation | Mean | Average of temperature-scaled probabilities |
| Seeds Used | 0, 42, 456 | Minimum needed to achieve 0.8343 |

**Temperature Scaling Formula:**
```
logits = log(p / (1-p))
scaled_logits = logits / T
final_prob = sigmoid(scaled_logits)
```
When T < 1: sharpens (increases confidence)
When T > 1: softens (decreases confidence)

### LLM Features v4

**Model Used:** qwen-2.5-72b (via Together AI)

| Feature Group | Features | Description |
|---------------|----------|-------------|
| scores | check_score, verif_score, harm_score | 0-100 confidence scores |
| entropy | check_entropy, verif_entropy, harm_entropy | Uncertainty in predictions |
| p_yes | check_p_yes, verif_p_yes, harm_p_yes | Probability of "Yes" from logprobs |
| margin_p | check_margin_p, verif_margin_p, harm_margin_p | p_yes - p_no margin |
| predictions | check_prediction, verif_prediction, harm_prediction | Binary predictions |
| cross_basic | score_variance, score_max_diff, yes_vote_count, unanimous_yes, unanimous_no | Cross-module agreement |
| harm_subdims | social_fragmentation, spurs_action, believability, exploitativeness | Harm sub-dimensions |

**XGBoost Classifier (for LLM features):**
- n_estimators: 100
- max_depth: 4
- learning_rate: 0.1
- scale_pos_weight: 3 (class imbalance)

### Late Fusion

```python
final_prob = 0.5 × deberta_ensemble_prob + 0.5 × llm_xgboost_prob
prediction = "Yes" if final_prob >= 0.50 else "No"
```

---

## Method 2: DeBERTa 3-Seed Ensemble

**F1 = 0.8343**

Same DeBERTa training as above, without LLM fusion.

| Parameter | Value |
|-----------|-------|
| Seeds | 0, 42, 456 |
| Temperature | 0.5 |
| Threshold | 0.55 |
| Aggregation | Mean of temperature-scaled probs |

---

## Method 3: Single DeBERTa

**F1 = 0.8242** (seed=0)

Single model, no ensemble, threshold calibrated on dev set.

---

## Reproducibility Commands

```bash
# Train DeBERTa models
python finetune_deberta_multimodel.py --seed 0 --output-dir ~/ensemble_results/seed_0
python finetune_deberta_multimodel.py --seed 42 --output-dir ~/ensemble_results/seed_42
python finetune_deberta_multimodel.py --seed 456 --output-dir ~/ensemble_results/seed_456

# Ensemble only (F1=0.8343)
python ensemble_deberta_seeds.py --ensemble-dir ~/ensemble_results --data-dir ~/data --seeds 0 42 456

# Ensemble + LLM fusion (F1=0.8362)
python combine_deberta_llm_features.py --ensemble-dir ~/ensemble_results --data-dir ~/data --seeds 0 42 456 --temperature 0.3
```

