# Code Review Request: Temperature Experiment

## Context

You are reviewing a temperature sanity check experiment for an IJCAI 2026 paper on checkworthiness classification. The experiment tests whether T=0.0 (deterministic) is safe to use across 5 LLM models, or if any model catastrophically fails compared to T=0.7.

## Files to Review

**Primary file:**
- `experiments/scripts/temperature_experiment.py` (~2300 lines)

**Supporting file:**
- `src/checkworthiness/prompting_baseline.py` (the API calling logic)

## What the Experiment Does

1. Loads 20 stratified samples from CT24 checkworthiness dataset (10 Yes, 10 No)
2. For each of 5 models (GPT-4.1, GPT-4.1-mini, DeepSeek Chat, DeepSeek Reasoner, Grok):
   - Runs a **drift probe** before predictions (sends "x" 10 times, collects logprobs)
   - Runs predictions at T=0.0 and T=0.7
   - Runs a **drift probe** after predictions
   - Compares pre/post logprobs using permutation test to detect model drift
3. For DeepSeek Reasoner specifically, captures internal reasoning (`reasoning_content`) separately from final answer
4. Computes alignment between internal reasoning and stated reasoning using embeddings
5. Saves everything to CSV and Parquet

## Areas to Scrutinize

### 1. Statistical Validity

- Is N=20 samples actually enough to detect "catastrophic" (>20%) differences?
- Is the power analysis calculation correct?
- Are the confidence intervals (Wilson score) appropriate for this sample size?
- Is the permutation test for drift detection implemented correctly?
- Should we use a different test statistic than mean absolute difference?

### 2. Logprob Handling

- Are we correctly extracting logprobs from different API responses?
- Is the imputation strategy (using min logprob for missing tokens) sound?
- For reasoning models, are we handling `logprobs.reasoning_content` vs `logprobs.content` correctly?
- Is the drift statistic S meaningful? Should we use KL divergence instead?

### 3. API Call Logic

- Are we handling rate limits correctly?
- Is the error handling robust enough?
- For thinking models (DeepSeek Reasoner, Grok), we skip assistant prefill - is this correct?
- Are we correctly handling the 5-tuple return from `prompting_baseline.py`?

### 4. Reasoning Analysis

- Is the cosine similarity between internal and stated reasoning meaningful?
- Are we using the right embedding model (`text-embedding-3-large`)?
- Is the alignment score computed correctly?
- Does the calibration comparison between internal/final logprobs make sense?

### 5. Data Integrity

- Are we saving all relevant data to both CSV and Parquet?
- Is anything being silently dropped or truncated?
- Are timestamps and sample IDs consistent across files?
- Could there be data leakage between train/test?

### 6. Edge Cases

- What happens if a model returns no logprobs?
- What happens if all API calls fail for one model?
- What happens if reasoning_content is empty?
- What happens if the permutation test has all identical samples?

### 7. Reproducibility

- Is the random seed used consistently everywhere?
- Are there any sources of non-determinism we're not controlling?
- Would running this twice produce the same results?

### 8. Scientific Claims

- Can we actually claim "model stability" from 10 pre + 10 post samples?
- Is the catastrophic threshold of 20% justified?
- Are we over-interpreting alignment scores?
- Is there p-hacking risk from multiple comparisons?

## Specific Questions

1. **Bug hunt**: Are there any obvious bugs that would produce incorrect results?

2. **Silent failures**: Are there places where errors are caught but might hide real problems?

3. **Metric correctness**: Are ECE, Brier score, F2 score calculated correctly?

4. **Off-by-one errors**: Are all loops, indices, and slices correct?

5. **Type mismatches**: Are we comparing strings to strings, floats to floats consistently?

6. **JSON serialization**: Are we correctly serializing/deserializing logprobs?

## Output Format

Please provide:

1. **Critical Issues**: Things that would produce wrong results or crash
2. **Methodological Concerns**: Statistical or scientific problems
3. **Code Quality Issues**: Maintainability, clarity, potential future bugs
4. **Suggestions**: Improvements that would strengthen the experiment

For each issue, provide:
- File and line number (approximate is fine)
- Description of the problem
- Suggested fix or investigation

## Don't Worry About

- Code style / formatting (we use ruff)
- Documentation completeness
- Performance optimization (this is a small experiment)
- Adding new features

Focus on **correctness** and **scientific validity**.
