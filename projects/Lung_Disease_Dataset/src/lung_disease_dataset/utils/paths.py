# utils/paths.py
from __future__ import annotations

from pathlib import Path
import importlib.resources as ir
import sys

"""
Reliable path utilities for ML and VS Code workflows.

Features:
- project_root(): find repository root by searching for pyproject.toml or .git
- resource_path(): stable way to load data/models/configs regardless of CWD
- safe for VS Code, Jupyter, Hydra, CLI scripts, Docker, pip installs

This file is the only place in the project that handles path logic.
Everything else should import functions from here.
"""

_MARKERS = ["pyproject.toml", ".git"]


# -----------------------------------------------------------
# Project root discovery
# -----------------------------------------------------------

def project_root() -> Path:
    """
    Returns the top-level project directory by searching upward for
    pyproject.toml or .git. Works regardless of where the interpreter runs.
    """
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        for marker in _MARKERS:
            if (p / marker).exists():
                return p
    return here.parent   # Fallback (should rarely be needed)


# -----------------------------------------------------------
# Entry script directory (when running "python scripts/train.py")
# -----------------------------------------------------------

def entry_script_dir() -> Path | None:
    """
    Returns the directory of the entry script (sys.argv[0]).
    Useful when you want paths relative to where the user executed a script.
    """
    try:
        entry = Path(sys.argv[0]).resolve()
        return entry.parent if entry.exists() else None
    except Exception:
        return None


# -----------------------------------------------------------
# Main resource loader
# -----------------------------------------------------------

def resource_path(*parts: str) -> Path:
    """
    Resolve paths in a stable way.

    Priority:
    1. Special case: models/ always anchored to entry script (scripts/models)
    2. Project root (development mode)
    3. importlib.resources (installed package)
    """
    # Special case: models/
    if parts and parts[0] == "models":
        entry = entry_script_dir()
        if entry is not None:
            return entry / "models" / Path(*parts[1:])
        # Fallback for notebooks / REPL
        return project_root() / "models" / Path(*parts[1:])

    # Normal case: use project root
    root = project_root()
    direct_path = root.joinpath(*parts)

    # If path exists in development, return it
    if direct_path.exists():
        return direct_path

    # Installed package fallback (for pip-installed packages)
    try:
        # Use __package__ if available, otherwise infer from __name__
        pkg_name = __package__.split(
            ".")[0] if __package__ else __name__.split(".")[0]
        pkg = ir.files(pkg_name).joinpath(*parts)
        with ir.as_file(pkg) as p:
            return Path(p)
    except Exception:
        # Best-effort: return project root path even if it doesn't exist yet
        return direct_path
