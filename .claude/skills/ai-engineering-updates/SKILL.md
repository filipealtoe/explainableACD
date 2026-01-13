# AI Engineering Communication Protocol

Rigorous, data-driven communication for AI engineering tasks. This skill governs bidirectional communication—how tasks are requested and how results are reported.

## When to Use

Activate this skill when:
- Reporting results from ML experiments or prompt optimization
- Requesting AI engineering tasks (model tuning, evaluation, prompt design)
- Discussing trade-offs between approaches
- Making decisions about model deployment or rollback

---

## The AHIRT Framework

Every AI engineering communication should address:

| Component | Question | Bad Example | Good Example |
|-----------|----------|-------------|--------------|
| **A**ssumptions | What conditions are we working with? | "The model isn't working" | "Assuming latency < 200ms is acceptable" |
| **H**ypothesis | What do we believe will happen? | "This might help" | "Adding few-shot examples will improve accuracy" |
| **I**ntervention | What specific action was/will be taken? | "Made some changes" | "Added 5 domain-specific examples to the prompt" |
| **R**esults | What are the quantified outcomes? | "It's better now" | "F1: 0.72 → 0.83 (+15.3%)" |
| **T**rade-offs | What are the costs and benefits? | (omitted) | "Inference time +20ms, token cost +12%" |

---

## Bad vs Good Updates

### Bad Update (Activity Description)

> "I tried some of the suggestions we had last week, and the results look a lot better."

**Problems:**
- Vague adjectives ("better") hide actual performance
- No hypothesis → impossible to interpret results
- No baseline → 1% improvement or 50%?
- Not reproducible or actionable

### Good Update (Experiment Report)

> "**Hypothesis:** Hybrid search (lexical + semantic) will outperform pure semantic search.
>
> **Intervention:** Implemented BM25 + dense retrieval fusion with RRF.
>
> **Results:**
> | Metric | Baseline | Hybrid | Delta |
> |--------|----------|--------|-------|
> | Recall@5 | 73% | 85% | +16.4% |
> | Recall@10 | 80% | 93% | +16.3% |
>
> **Trade-offs:** +5ms latency, minimal code change (rollout-ready)."

---

## Communication Rules

### Rule 1: Quantify or Don't Mention

| Don't Say | Say Instead |
|-----------|-------------|
| "Much better" | "+23% accuracy" |
| "Faster" | "Latency: 450ms → 180ms" |
| "More accurate" | "F1: 0.72 → 0.83" |
| "Cheaper" | "Cost: $0.12 → $0.04/query" |

**Adjectives are a signal you're hiding something.** If you can't quantify, state why: "Qualitative improvement in coherence (no automated metric available, human eval pending)."

### Rule 2: State Hypotheses Explicitly

Before any experiment:
```
Hypothesis: [Specific intervention] will [improve/reduce] [metric] by [expected magnitude]
Rationale: [Why we believe this]
```

After any experiment:
```
Result: Hypothesis [confirmed/rejected/partially confirmed]
Evidence: [Quantified outcomes]
```

### Rule 3: Always Include Baselines

Results without baselines are meaningless:
- ❌ "We achieved 85% accuracy"
- ✅ "Accuracy improved from 73% (baseline) to 85% (+16.4%)"

### Rule 4: Surface Trade-offs Proactively

Every improvement has costs. Always report:
- **Latency**: Did response time change?
- **Cost**: Token usage, API costs, compute costs
- **Complexity**: Lines of code, maintenance burden
- **Quality elsewhere**: Did other metrics degrade?

### Rule 5: Make It Actionable

End updates with clear next steps:
- "Ready to deploy—approval needed"
- "Promising direction—will test with larger dataset"
- "Dead end—recommend abandoning this approach"
- "Inconclusive—need 500 more samples for statistical significance"

---

## Task Request Format

When requesting AI engineering work, provide:

```markdown
## Task: [Clear objective]

**Hypothesis:** [What we expect to learn/achieve]

**Constraints:**
- Latency budget: [X ms]
- Cost budget: [$ per query]
- Quality floor: [Metric > threshold]

**Evaluation:**
- Primary metric: [What defines success]
- Secondary metrics: [What else to track]
- Test set: [Where to evaluate]

**Baseline:** [Current performance to beat]
```

---

## Update Templates

### Experiment Report

```markdown
## Experiment: [Name]

**Hypothesis:** [What we tested]

**Intervention:** [What we changed]

**Results:**
| Metric | Baseline | New | Delta |
|--------|----------|-----|-------|
| [M1]   | [X]      | [Y] | [±Z%] |

**Trade-offs:** [Costs incurred]

**Conclusion:** [Confirmed/Rejected/Partial]

**Next Steps:** [Action items]
```

### Progress Update

```markdown
## Status: [In Progress / Blocked / Complete]

**Completed:**
- [Intervention 1]: [Result with numbers]
- [Intervention 2]: [Result with numbers]

**In Progress:**
- [What's being tested now]

**Blockers:** [If any, be specific]

**Key Learning:** [One sentence insight]
```

### Decision Request

```markdown
## Decision Needed: [Topic]

**Context:** [Brief background]

**Options:**
| Option | Benefit | Cost | Risk |
|--------|---------|------|------|
| A      | [+X%]   | [Y]  | [Z]  |
| B      | [+X%]   | [Y]  | [Z]  |

**Recommendation:** [Option X] because [quantified reasoning]

**Reversibility:** [Easy/Hard to undo]
```

---

## Anti-Patterns

### The Activity Report
> "I spent the day working on the model."

**Problem:** Describes effort, not outcomes. Time spent ≠ progress.

### The Vague Positive
> "The new approach is showing promise."

**Problem:** "Promise" is subjective. Quantify or specify what was observed.

### The Missing Baseline
> "We achieved 92% accuracy!"

**Problem:** Is that good? Without baseline, impossible to evaluate.

### The Hidden Regression
> "Accuracy improved to 85%."

**Problem:** What about latency, cost, other metrics? Surface all changes.

### The Unactionable Update
> "Results were mixed."

**Problem:** What's the next step? Always end with clear direction.

---

## Quick Reference

**Before speaking, check:**
- [ ] Did I state my hypothesis?
- [ ] Did I quantify results?
- [ ] Did I include baselines?
- [ ] Did I surface trade-offs?
- [ ] Did I make it actionable?

**Red flag words to avoid:**
- "Better", "worse", "improved", "faster", "slower"
- "Promising", "concerning", "interesting"
- "Some", "a lot", "a few", "many"
- "Seems like", "appears to", "might be"

**Replace with:** Numbers, percentages, deltas, confidence intervals.
