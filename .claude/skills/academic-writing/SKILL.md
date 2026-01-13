# Academic Writing Standards Skill

Expert knowledge of academic writing standards for peer-reviewed papers.

## When to Use

Activate when:
- Reviewing or editing academic manuscripts
- Writing methodology, results, or discussion sections
- Checking citation integrity
- Improving clarity and scientific rigor

## Core Principles

### Clarity Over Complexity
- Prioritize clarity and precision over elaborate phrasing
- Favor active voice and simple sentence structures
- One idea per sentence
- One topic per paragraph

### Eliminate Weak Language

**Remove overused qualifiers:**
- ❌ "clearly", "obviously", "novel", "significant", "very", "quite"
- ✅ Let the evidence speak for itself

**Avoid hedging stacks:**
- ❌ "It might possibly be suggested that..."
- ✅ "We hypothesize that..." or "The results suggest..."

**Be precise:**
- ❌ "The approach outperforms existing methods significantly"
- ✅ "The approach achieves 15% higher F1 score than the baseline"

### Redundancy Elimination

| Redundant | Better |
|-----------|--------|
| past history | history |
| future plans | plans |
| completely eliminate | eliminate |
| basic fundamentals | fundamentals |
| end result | result |
| consensus of opinion | consensus |

## Citation Standards

### Formatting (Pandoc/LaTeX)
```latex
% In-text citation
According to Smith et al.~\cite{smith2024}...
Previous work~\cite{jones2023,doe2024} has shown...

% Pandoc markdown
According to @smith2024...
Previous work [@jones2023; @doe2024] has shown...
```

### Citation Hygiene
- [ ] Every claim has a citation OR is your contribution
- [ ] No placeholder citations like "[citation needed]"
- [ ] Self-citations use third person for blind review
- [ ] All cited works appear in references
- [ ] All references are cited at least once

### BibTeX Quality
- Use DBLP for CS papers (authoritative, no hallucinations)
- Include: author, title, booktitle/journal, year, pages
- Use consistent key format: `authorYYYYkeyword`

## Structure Guidelines

### Abstract (150-250 words)
1. Problem/motivation (1-2 sentences)
2. Gap/limitation of existing work (1 sentence)
3. Your approach (2-3 sentences)
4. Key results with numbers (1-2 sentences)
5. Impact/significance (1 sentence)

### Introduction Pattern
1. Broad context
2. Specific problem
3. Why it matters
4. What's missing (gap)
5. Your contribution
6. Paper outline (optional)

### Related Work
- Organize thematically, not chronologically
- Compare, don't just describe
- End each subsection with how your work differs

### Methodology
- Reproducibility is paramount
- Include: model architecture, hyperparameters, training details
- Justify design choices
- Reference code/data availability

### Results
- Lead with key findings
- Use tables for numbers, figures for trends
- Statistical significance where appropriate
- Compare fairly (same train/test splits, metrics)

### Limitations
- Be honest but not self-deprecating
- Frame as "future work" where appropriate
- Shows intellectual maturity

## Formatting Rules

### Text
- One sentence per line in source (easier diffs)
- Max 80 characters per line
- British OR American English (be consistent)
- No trailing whitespace

### Numbers
- Spell out one through nine
- Use numerals for 10+
- Always use numerals with units: "5 GB", "3 epochs"
- Use commas in large numbers: 1,000,000

### Figures and Tables
- Every figure/table must be referenced in text
- Captions should be self-contained
- Readable in black & white
- Vector graphics when possible (PDF, not PNG)

## Review Checklist

Before submission:
- [ ] Abstract standalone and compelling
- [ ] Contributions clearly stated
- [ ] All claims supported by evidence or citations
- [ ] Methodology reproducible
- [ ] Results match claims
- [ ] Limitations acknowledged
- [ ] No first-person in blind submissions (we → the authors)
- [ ] Consistent terminology throughout
- [ ] All acronyms defined on first use
