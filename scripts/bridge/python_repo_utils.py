from __future__ import annotations

from pathlib import Path


def resolve_python_repo_root(current_file: Path) -> Path:
    """Return the repository root that contains the Python model code."""
    # Walk up the directory tree to find the progress_model package
    for parent in current_file.resolve().parents:
        candidate = parent / "progress_model" / "config.py"
        if candidate.exists():
            return parent
    raise RuntimeError("progress_model/config.py not found; the Python model is missing")
