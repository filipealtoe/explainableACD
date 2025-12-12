from __future__ import annotations

from pathlib import Path
from typing import Iterable


def get_repo_root(markers: Iterable[str] = ("pyproject.toml", ".git")) -> Path:
    start = Path(__file__).resolve().parent
    for parent in [start, *start.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Could not locate repository root")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else get_repo_root() / p
