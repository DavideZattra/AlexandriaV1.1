"""Filesystem anchors for the project.

All data directories are resolved relative to the project root (computed from
this file's location), NOT the current working directory. This means scripts
and entrypoints work no matter where they are launched from.
"""

from pathlib import Path

# alexandria/paths.py -> parent is alexandria/, parent.parent is the repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Persisted data directories (gitignored)
CHROMA_DIR = PROJECT_ROOT / "alexandria_db"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "alexandria_knowledge_base"
EVAL_RESULTS_DIR = PROJECT_ROOT / "Docs" / "Evaluation_Results"
