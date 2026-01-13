# Research Discipline for the Age of AI

A practical checklist of what not to do, how to think, and how this applies to ML, DL, and LLMs.

## When to Use

Activate this skill when:
- Designing ML/LLM experiments
- Reviewing your own or others' research methodology
- Writing experiment sections for papers
- Evaluating claims from papers or benchmarks
- Making decisions about model selection, metrics, or baselines

## Core Principle

> Your job is not to impress — it's to reduce uncertainty.

---

## 1. First Principles (Before Touching Data or Models)

### Core Questions You MUST Answer Explicitly

| Question | Why It Matters |
|----------|----------------|
| What is the claim? | Descriptive? Predictive? Causal? Mechanistic? |
| What would falsify it? | If nothing can falsify it, it's not science |
| Why should this work beyond this dataset/prompt/benchmark? | Generalization is the goal, not leaderboard rank |

**Red flag:** "We explore whether…" without a falsifiable hypothesis.

---

## 2. Design Errors (Research-Agnostic)

### Common Failures

| Failure | Description |
|---------|-------------|
| **Degrees of freedom abuse** | Many choices, few justifications |
| **Post-hoc rationalization** | Story written after seeing results |
| **Convenience framing** | Methods chosen because easy or trendy |

### Required Discipline

**Pre-commit to:**
- Dataset inclusion/exclusion criteria
- Metrics (primary and secondary)
- Comparators / baselines
- Analysis plan

**Write why-not arguments:** Explicitly document why alternatives were rejected.

**Red flag:** Decisions justified with "standard practice" or "commonly used."

---

## 3. Data & Sampling Mistakes

### Typical Errors

| Error | Example |
|-------|---------|
| **Silent filtering** | Dropping "outliers" without reporting |
| **Post-hoc subgroup analysis** | "Interestingly, performance was higher for..." |
| **Leakage** | Train/test overlap, semantic contamination |
| **Benchmark reification** | Treating benchmarks as ground truth |

### Proper Thinking

- Ask: "What population does this data actually represent?"
- Separate measurement noise from model error
- **Assume your dataset is biased unless proven otherwise**

**Red flag:** No section titled "Data limitations & failure modes."

---

## 4. Statistics & Evaluation Traps

### Classic Failures

| Trap | Problem |
|------|---------|
| Single-run reporting | Variance unknown |
| No uncertainty estimates | Precision unknown |
| Metric over-optimization | Goodhart's Law |
| Multiple testing | False discovery inflation |

### Required Standards

| Instead of... | Use... |
|---------------|--------|
| Point estimates | Variance across runs |
| p-values | Confidence intervals |
| Cherry-picked wins | Failure case analysis |

**Red flag:** "State-of-the-art" claims without robustness checks.

---

## 5. Interpretation Errors (The Most Dangerous)

### What Researchers Routinely Conflate

| Observation | ≠ | Conclusion |
|-------------|---|------------|
| Correlation | → | Causation |
| Performance | → | Understanding |
| Fluency | → | Intelligence |
| Explanation | → | Mechanism |

### Mandatory Humility

- Separate **what** the model does from **why** it might do it
- Treat explanations as hypotheses, not facts
- Use interventional language only when you've done interventions

**Red flag:** Mechanistic language ("the model reasons by...") without interventions.

---

## 6. Deep Learning–Specific Pitfalls

### Common DL Mistakes

| Mistake | What Actually Happened |
|---------|------------------------|
| "Generalization" | Overfitting masked by large test set |
| "Baseline comparison" | Untuned or unfair baselines |
| "Architecture innovation" | Dataset artifacts driving gains |
| "Novel method wins" | Simpler model would have won with proper tuning |

### Good Practice

- [ ] Stress-test with domain shifts
- [ ] Compare against strong, **tuned** baselines
- [ ] Report when simpler models win
- [ ] Test if gains survive data perturbations

**Red flag:** Gains disappear with minor data perturbations.

---

## 7. LLM-Specific Failure Modes (Critical)

### Prompt Engineering Traps

| Trap | Description |
|------|-------------|
| **Prompt overfitting** | Tuning prompts on test data |
| **Hidden changes** | Unreported system prompt modifications |
| **Single-prompt conclusions** | No prompt sensitivity analysis |
| **Narrative claims** | "Elicits reasoning" without controls |

### Evaluation Pitfalls

| Pitfall | Problem |
|---------|---------|
| **Benchmark contamination** | Test data in training corpus |
| **Style bias** | Human evals favor fluency over accuracy |
| **No seed/temperature sensitivity** | Results not reproducible |
| **Small delta worship** | 0.5% improvement treated as breakthrough |

### Interpretation Fallacies

| Observation | ≠ | Claim |
|-------------|---|-------|
| Chain-of-thought | → | Reasoning |
| Emergence | → | Understanding |
| Tool use | → | Planning |
| Correct output | → | Correct process |

**Red flag:** Claims about cognition without counterfactual tests.

---

## 8. Decision Justification Protocol

For every major choice, answer:

```markdown
## Decision: [What was chosen]

**Alternatives considered:**
1. [Option A]
2. [Option B]

**Why alternatives were rejected:**
- Option A: [Specific reason with evidence]
- Option B: [Specific reason with evidence]

**Assumptions introduced:**
- [Assumption 1]
- [Assumption 2]

**How this choice could fail:**
- [Failure mode 1]
- [Failure mode 2]
```

**If you can't answer these → the choice is unjustified.**

---

## 9. Using AI Tools Responsibly in Research

### Safe Usage

| Do | Why |
|-----|-----|
| Ask for counterexamples | Challenge your design |
| Request failure modes | Find blind spots |
| Generate null hypotheses | Consider alternatives |
| Treat outputs as drafts to attack | Not validation |

### Unsafe Usage

| Don't | Why |
|-------|-----|
| Let models choose metrics | Degrees of freedom abuse |
| Let models select samples | Introduces bias |
| Accept fluent explanations | Fluency ≠ correctness |
| Use generated text as evidence | Hallucination risk |

---

## 10. Gold-Standard Mindset

| Prioritize | Over |
|------------|------|
| Slower thinking | Faster output |
| Clarity | Cleverness |
| Negative results | Inflated claims |
| Understanding | Leaderboard rank |
| Reproducibility | Novelty |
| Honest limitations | Overstated contributions |

---

## Pre-Experiment Checklist

Before running any experiment:

- [ ] Hypothesis is falsifiable
- [ ] Success/failure criteria defined in advance
- [ ] Dataset limitations documented
- [ ] Baselines are strong and tuned
- [ ] Metrics justified (not just "commonly used")
- [ ] Analysis plan pre-registered (even informally)
- [ ] Leakage checks performed
- [ ] Failure modes anticipated

## Pre-Submission Checklist

Before claiming results:

- [ ] Multiple runs with variance reported
- [ ] Confidence intervals provided
- [ ] Ablations show which components matter
- [ ] Failure cases analyzed (not hidden)
- [ ] Claims match evidence (no overclaiming)
- [ ] Limitations section is honest
- [ ] Reproducibility details complete
- [ ] Why-not arguments documented

---

## Red Flags Summary

Immediately question any research that:

| Red Flag | What It Signals |
|----------|-----------------|
| "We explore whether..." | No falsifiable hypothesis |
| "Standard practice" | Unjustified choice |
| No data limitations section | Unexamined assumptions |
| "State-of-the-art" without robustness | Fragile results |
| Mechanistic claims without interventions | Interpretation error |
| Single-run results | Unknown variance |
| Prompt details hidden | Unreproducible |
| Small deltas celebrated | Noise vs signal confusion |
| "The model understands/reasons" | Anthropomorphization |

---

## Quick Reference

**Before any experiment:**
> "What would convince me this is wrong?"

**After any result:**
> "What else could explain this?"

**Before any claim:**
> "Would I bet money on this generalizing?"
