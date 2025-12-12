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
