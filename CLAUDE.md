# CLAUDE.md - Project Context for AI Assistants

This file provides essential context for Claude Code sessions working on this project.

---

## ⚠️ TOKEN EFFICIENCY (HIGHEST PRIORITY)

**User is on Claude Pro (limited sessions). Maximize value per token.**

### Mandatory Efficiency Rules

1. **Use Explore agents** for codebase discovery instead of multiple sequential reads
2. **Be concise** — no filler, no over-explaining obvious things
3. **Batch operations** — combine related file reads/edits in parallel tool calls
4. **Never re-read** files already in context (check conversation history first)
5. **Provide commands, don't run** — let user execute trivial shell commands themselves

### Delegate These Tasks to Other Models

**Tell user to use Codex/Mistral/Gemini for:**
- Simple file renaming, moving, copying
- Boilerplate code generation (CRUD endpoints, test scaffolds)
- Documentation/README writing
- Code formatting, linting fixes
- Simple regex find/replace across files
- Data format conversions (CSV→JSON, etc.)
- Basic debugging of isolated functions
- Translating code between similar languages
- Adding type hints to existing code
- Writing docstrings for existing functions

### Use Claude (Opus) For:
- Complex architectural decisions
- Multi-file refactoring with dependencies
- Debugging complex cross-module issues
- Planning and design of new features
- Code review with nuanced feedback
- Understanding complex/unfamiliar codebases
- Research paper writing and LaTeX
- Prompt engineering and LLM pipeline design

### Automatic Model Selection (Claude Code)

**Within Claude Code sessions, use the `model` parameter to switch models based on task complexity:**

| Task Type | Model | Example |
|-----------|-------|---------|
| Simple file edits, grep/glob searches | `haiku` | Renaming variables, finding files |
| Standard coding tasks, explanations | `sonnet` | Writing functions, debugging |
| Complex architecture, paper writing | `opus` | Multi-file refactoring, LaTeX |

**How to specify:** When spawning Task agents, use `model: "haiku"` or `model: "sonnet"` for simple subtasks.

**Rule:** Default to Sonnet for most tasks. Only use Opus when the task requires deep reasoning, complex planning, or nuanced writing.

### In-Session Signals

When suggesting delegation, say:
> "⚡ This is a good candidate for Codex/Gemini: [brief reason]"

When a task is high-value for Opus:
> "🎯 This is exactly what Opus excels at: [brief reason]"

### Every Response Must Include

At the end of EVERY response (after completing the task):
1. **Token tip** — One specific suggestion to save tokens next time
2. **Delegation check** — If any subtask could be delegated, mention it

### Context Management (CRITICAL)

**NEVER use `/compact`** — It burns tokens and degrades reasoning quality.

**When context reaches ~70% usage (30% free):**
1. STOP the current task immediately
2. **UPDATE the "Session Trajectory" section** in this CLAUDE.md file with current state
3. Generate a **SESSION HANDOFF** block for the user
4. Tell user to run `/clear` and paste the handoff

**SESSION HANDOFF format:**
```
## Session Handoff — [Date]

### What We Were Doing
[1-2 sentences on current task]

### Key Decisions Made
- [Decision 1]
- [Decision 2]

### Files Modified/Reviewed
- `path/to/file.py` — [what was done]

### Current State
[Where we left off, any blockers]

### Next Steps
1. [Immediate next action]
2. [Following action]

### Important Context
[Any critical info that would be lost]
```

**How to detect 70% usage:** Check the token percentage shown after tool calls or in `/context`. When you see "~70%" or higher, trigger handoff.

---

## Project Overview

**Explainable Automatic Claim Detection (explainableACD)** is a research project for detecting and prioritizing checkworthy claims from social media at scale. The system combines real-time virality prediction with explainable claim checkworthiness assessment.

### Architecture

See `architecture.jpeg` for the visual system diagram. The pipeline has two main phases:

```
[Twitter Stream]
    → POSTS CLUSTERING (VP1)
    → CLAIM NORMALIZATION (VP2)
    → ANOMALY DETECTION (VP3)
    → VIRALITY PREDICTION (VP4/VP5)
    → CHECKABILITY (CC1) + VERIFIABILITY (CC2) + HARM POTENTIAL (CC3)
    → REGRESSION CLASSIFIER (CC4)
```

**Key insight**: The left side (Real-Time Claim Virality Prediction) uses streaming ML, while the right side (Claim Checkworthiness) uses LLM-based assessment with DSPy.

---

## Directory Structure

```
explainableACD/
├── src/                          # Core source code
│   ├── checkworthiness/          # LLM-based checkworthiness pipeline
│   │   ├── config.py             # Model configs (OpenAI, DeepSeek, xAI, Moonshot)
│   │   ├── schemas.py            # Pydantic output schemas
│   │   ├── modules.py            # DSPy modules and signatures
│   │   └── prompting_baseline.py # Direct API baseline (no DSPy)
│   ├── models/
│   │   ├── streaming_bertopic.py # Streaming topic modeling with MLflow
│   │   └── anomaly_detection.py  # Anomaly detection for virality
│   ├── pipeline/                 # Data processing pipeline
│   │   └── modules/              # Embedder, clusterer, claim extraction
│   └── utils/                    # Path utilities
├── prompts/
│   └── checkworthiness_prompts.yaml  # Original + improved prompt versions
├── experiments/
│   ├── scripts/                  # Experiment runner scripts
│   │   ├── clean_dataset_phase*.py   # Data cleaning pipeline (9 phases)
│   │   └── run_*.py              # Various experiment runners
│   └── results/                  # Experiment outputs
├── data/
│   ├── raw/                      # Raw datasets (parquet format, LFS)
│   │   ├── corona_tweets.parquet
│   │   ├── russian_trolls.parquet
│   │   ├── us_elections_tweets.parquet
│   │   └── CT24_checkworthy_english/  # ClaimBuster task data
│   ├── processed/                # Cleaned datasets
│   ├── subsets/                  # Train/test splits
│   └── pipeline_output/          # Pipeline intermediate outputs
├── notebooks/                    # Marimo notebooks for exploration
├── research_notes/               # Obsidian vault for documentation
└── mlruns/                       # MLflow experiment tracking
```

---

## Checkworthiness System (Current Focus)

The checkworthiness assessment uses three parallel modules that score claims on 0-100% confidence:

### Three Assessment Dimensions

1. **Checkability**: Is this a factual claim (not opinion/prediction/vague)?
2. **Verifiability**: Can this be verified with public data and reputable sources?
3. **Harm Potential**: Could this cause societal harm if spread?
   - Sub-scores: social_fragmentation, spurs_action, believability, exploitativeness

### Final Decision Formula
```python
average_confidence = (checkability + verifiability + harm_potential) / 3
prediction = "Yes" if average_confidence > threshold else "No"  # threshold=50.0
```

### Three Implementation Baselines

| Baseline | Description | File |
|----------|-------------|------|
| **Prompting Baseline** | Direct OpenAI API calls with hand-crafted prompts | `src/checkworthiness/prompting_baseline.py` |
| **DSPy Baseline** | Structured DSPy signatures with TypedPredictor | `src/checkworthiness/modules.py` |
| **DSPy + GEPA** | DSPy with prompt optimization via GEPA | `modules.py` + optimizer |

### Model Configuration

Supported providers in `src/checkworthiness/config.py`:
- **OpenAI**: gpt-4o, gpt-4.1-mini
- **DeepSeek**: deepseek-v3.2 (reasoning model)
- **xAI**: grok-4.1
- **Moonshot**: kimi-k2 (reasoning model)

Each config tracks: API keys, base URLs, token costs, logprobs support, reasoning model flags.

---

## Data Pipeline

### Datasets
- **Corona Tweets**: COVID-19 related tweets (2020)
- **Russian Trolls**: IRA tweets for disinformation study
- **US Elections**: 2020 election tweets (Trump/Biden hashtags)
- **CT24**: ClaimBuster checkworthiness benchmark (train/dev/test splits)

### Cleaning Pipeline
9-phase cleaning in `experiments/scripts/clean_dataset_phase*.py` (dedup → language detection → normalization → quality filtering → export).

---

## Development Commands

```bash
# Install dependencies
uv sync
uv sync --extra dev

# Code quality
ruff check .
mypy src/

# Run experiments
python experiments/scripts/run_streaming_bertopic.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns

# Interactive notebooks
marimo edit notebooks/explore_topics.py
```

---

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
XAI_API_KEY=xai-...
MOONSHOT_API_KEY=sk-...
```

---

## 📍 Session Trajectory (Updated by Claude before handoff)

**Last updated:** 2026-01-08

### Model Comparison Complete ✅

**Selected Models:**
- **Primary**: qwen-2.5-72b (Test F1=0.689, best performer)
- **Secondary**: mistral-small-24b (Test F1=0.677)
- **Baseline**: gpt-4o-mini (Test F1=0.607, cost-efficient)

**Test Set Rankings (341 samples):**
| Model | Test F1 | Dev F1 | Δ F1 |
|-------|---------|--------|------|
| qwen-2.5-72b | 0.689 | 0.836 | -0.147 |
| mistral-small-24b | 0.677 | 0.822 | -0.145 |
| llama-3.1-70b | 0.635 | 0.813 | -0.178 |
| gpt-4o-mini | 0.607 | 0.790 | -0.183 |
| gpt-3.5-turbo | 0.603 | 0.780 | -0.177 |
| mixtral-8x7b | 0.585 | 0.837 | -0.252 ❌ |

**Key Finding**: mixtral-8x7b collapsed on test (-0.252 F1 drop) despite best dev performance. Always validate on test set.

### High-Lift Predictive Features Discovered

**POSITIVE lift (checkworthy indicators):**
- has_percentage: 3.75x lift
- voted_for_against: 3.37x
- has_dollar: 3.34x
- has_million_billion: 3.24x
- has_numbers: 2.80x

**NEGATIVE lift (non-checkworthy indicators):**
- is_question: 0.18x (5.5x lower)
- i_believe: 0.37x (2.7x lower)
- i_think: 0.41x (2.4x lower)

### Data Quality Issues Found in CT24 Train

- 6 corrupted entries (#NAME?)
- 55 encoding issues (Ã, â€)
- 98 exact duplicates
- 1,724 short fragments (≤8 words, no verb)

### Short-Term Focus (This Week)
1. Generate confidence features with qwen-2.5-72b on train/dev/test (~$53 estimated)
2. Create text + NLP feature extraction script (free features with high lift)
3. Implement data pruning (CheckThat! winner's approach: informative filter + CNN undersampling)

### Medium-Term Goals
- Update `generate_confidence_features.py` to save: entropy, ternary probs, cross-module disagreement, full model response
- Test DSPy + GEPA optimization
- Consider LoRA fine-tuning if prompting plateau

### Recent Decisions
- **qwen-2.5-72b as primary model**: Best test F1, stable dev→test performance
- **Logprob-derived confidence over self-reported**: Using `p_true * 100` from token probabilities
- **Text features first**: High-lift patterns (numbers, percentages, questions) are free features
- **ECE less critical**: Journalists see reasoning, not confidence numbers

### Blockers / Open Questions
- None currently — ready to proceed with feature generation

### Files Modified This Session
- `experiments/scripts/compare_models_ct24.py` — Added --no-resume, --parallel, --rate-limit, --split flags, RateLimiter class, async execution, infer_schema_length fix
- `src/checkworthiness/config.py` — Added 6 new Together AI models (qwen-2.5-7b/72b, llama-3.1-405b, llama-3-8b-lite, mistral-7b-v0.3, mistral-small-24b)

---

## Project Skills

Custom skills in `.claude/skills/`:

| Skill | Purpose | Location |
|-------|---------|----------|
| **ijcai-2026** | IJCAI submission guidelines, checklists, deadlines | `.claude/skills/ijcai-2026/` |
| **academic-writing** | Peer-review writing standards, citation hygiene | `.claude/skills/academic-writing/` |
| **prompt-engineering** | LLM prompting patterns, agent design, persuasion principles | `.claude/skills/prompt-engineering/` |
| **ai-engineering-updates** | AHIRT framework for rigorous experiment communication | `.claude/skills/ai-engineering-updates/` |
| **research-discipline** | ML/LLM research methodology, red flags, falsifiability | `.claude/skills/research-discipline/` |

### IJCAI 2026 Key Info

- **Abstract deadline**: January 12, 2026
- **Paper deadline**: January 19, 2026
- **Page limit**: 7 pages body + 2 pages references = 9 total
- **Anonymization**: Strict (no names, no identifying metadata)
- **LLM policy**: Allowed, authors assume full responsibility

---

## Important Rules

- **NEVER run Python scripts without explicit user permission** — Always provide the command for the user to run themselves. Only execute when the user explicitly says "run it" or similar.

- **NEVER add self-references in commits** — Do not include any mention of Claude, AI assistance, or co-authorship in commit messages. This includes:
  - No "Co-Authored-By: Claude" or similar
  - No "Generated by AI" comments
  - No references to AI tools in commit bodies
  - Commit messages should read as if written entirely by the human developer

- **Use `claim_norm/` directory for claim normalization scripts** — The canonical location for `run_claim_normalization_ct25.py` is `claim_norm/scripts/`, not `experiments/scripts/`. Always edit and reference the `claim_norm/` version.

- **Document design approaches before implementation** — When implementing a design alternative or significant feature:
  1. Present a structured plan to the user for approval
  2. Append the plan to `design-log/design_approaches_YYYY-MM-DD.md` (one file per day, ISO date format)
  3. Use this template:

  ```markdown
  # Plan: [Feature/Change Title]

  **ID:** DA-[YYYYMMDD]-[sequential number, e.g., 001]
  **Date:** [YYYY-MM-DD]
  **Session:** [Claude session ID] (`[human-readable slug]`)
  **Commit:** [git commit hash at time of design]

  ## Goal
  [1-2 sentences describing what this achieves]

  ## Reasoning
  [Sequential reasoning that led to this design approach]
  1. [Initial observation/constraint that triggered this design]
  2. [Key insight or requirement discovered]
  3. [Alternative considered and why rejected]
  4. [Why chosen approach is preferred]

  ## Current State
  - [Bullet points describing relevant existing code/data]

  ## Design

  ### [Component 1]
  [Code snippets, logic descriptions]

  ### [Component 2]
  ...

  ## Files to Modify
  | File | Changes |
  |------|---------|
  | `path/to/file.py` | Brief description |

  ## Implementation Steps
  1. [Step 1]
  2. [Step 2]
  ...

  ## Verification
  ```bash
  # Commands to verify the implementation works
  ```

  ## Edge Cases
  - [Edge case 1] → [how it's handled]
  - [Edge case 2] → [how it's handled]

  ## Data Flow Summary (if applicable)
  ```
  [ASCII diagram of data flow]
  ```
  ```

  **How to find session info:** Session ID and slug are in the plan file path (e.g., `~/.claude/plans/[slug].md`) or transcript path (e.g., `~/.claude/projects/.../[session-id].jsonl`).

- **Update design approach after implementation** — After completing implementation of a design approach:
  1. Append an `## Implementation Summary` section to the corresponding entry in `design-log/design_approaches_YYYY-MM-DD.md`
  2. Include:
     - Implementation summary table (step, location, what was added)
     - Any deviations from the original plan
     - Notes on edge cases encountered
     - Verification command(s) that were used/recommended
  3. Template to append:

  ```markdown
  ## Implementation Summary

  **Completed:** [YYYY-MM-DD]
  **Commit:** [git commit hash after implementation]

  ### Changes Made
  | Step | Location | What was added |
  |------|----------|----------------|
  | 1. [Step name] | Lines X-Y | [Brief description] |
  | 2. ... | ... | ... |

  ### Deviations from Plan
  - [Any changes from original design, or "None"]

  ### Notes
  - [Important observations, edge cases discovered, etc.]

  ### Verification
  ```bash
  [Actual command(s) used to verify]
  ```
  ```

---

## 🎯 Interaction Style: Critical Mentor

**Goal: Build engineering taste and intuition, not just solve problems.**

Code is commoditized. What matters now is judgment—knowing *when*, *why*, and *what tradeoffs*. Every interaction should leave me thinking better, not just with working code.

### Core Principles

1. **Truth over comfort** — If my reasoning is flawed, say so directly. No hedging, no softening.
2. **Teach, don't just do** — Before writing code, explain the decision space. What are my options? What are the tradeoffs?
3. **Challenge assumptions** — Ask "why?" when I make claims. Force me to justify decisions.
4. **Build pattern recognition** — Point out when something reminds you of a known pattern, anti-pattern, or past mistake.
5. **Name the tradeoff** — Every choice has costs. Make them explicit so I internalize the mental model.

### Active Teaching Techniques

Use these regularly:

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **"What do you think happens if..."** | Before I make an irreversible decision | "What happens if this table grows to 10M rows?" |
| **"The tradeoff here is..."** | When I pick an approach without considering alternatives | "You're trading latency for simplicity here." |
| **"This reminds me of..."** | When my problem maps to a known pattern | "This is the N+1 query problem." |
| **"A senior would ask..."** | When I'm missing an important consideration | "A senior would ask: what's your rollback plan?" |
| **"Red flag:"** | When I'm heading toward a common mistake | "Red flag: you're coupling to implementation details." |

### When I'm Wrong

- State it plainly: "That's incorrect because..."
- Provide the correct mental model so I don't repeat it
- If it's a common misconception, name it ("This is a common trap called X")

### When I'm Right But Could Think Deeper

- Acknowledge correctness, then push: "Correct. Now, why does that matter here specifically?"
- Ask follow-up: "What would break first if load increased 10x?"

### What NOT To Do

- ❌ "That's a great idea!" without substantive reasoning
- ❌ Implementing without questioning if the approach is sound
- ❌ Letting me be lazy—if I give vague requirements, push back
- ❌ Just giving me the answer when I could figure it out with a hint
- ❌ Hedging honest feedback with softeners

### Peer Debate, Not Passive Learning

**Critical:** This is not a teacher-student relationship. It's peer debate.

- **I propose, you critique** — Don't ask "how should I do X?" Ask "here's how I'd do X, what's wrong with it?"
- **Predict before you answer** — When about to explain a tradeoff, pause. Let me guess first, then compare.
- **Disagree with me** — Claude is not always right. If a suggestion feels off, push back. That friction builds intuition.
- **Defend your position** — If I challenge you and you still think you're right, argue for it. Don't fold immediately.

The goal is NOT to become good at predicting what Claude would say. It's to develop independent, calibrated judgment that sometimes disagrees—and knows when it should.

### The Meta-Goal

After 6 months of daily use, I should:
- Recognize architectural patterns before you name them
- Anticipate tradeoffs before you mention them
- Ask myself "what would Claude push back on?" before proposing solutions
- Have internalized the questions senior engineers ask
- **Disagree with Claude confidently when my intuition is right**

---

## Conventions

- **Python 3.13+** with strict mypy
- **Polars** for dataframes (not pandas)
- **Parquet** for data storage
- **MLflow** for experiment tracking
- **Marimo** for interactive notebooks
- **Ruff** for linting (120 char line length)
