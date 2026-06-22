"""Shared helpers for the Infino examples.

Kept tiny and dependency-light so each example stays focused on Infino,
not boilerplate. Lives at the examples root and is imported by notebooks
across every category (e.g. ``rag/``), which add this root to ``sys.path``.
"""

import warnings

# Keep notebook output clean: the examples show no progress bars.
warnings.filterwarnings("ignore", message="IProgress not found.*")

__all__ = ["embedding", "chunking", "loaders", "sql", "llm"]
