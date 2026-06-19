"""Shared helpers for the Infino RAG examples.

Kept tiny and dependency-light so each example stays focused on Infino,
not boilerplate. Imported by every notebook in this directory.
"""

import warnings

# Keep notebook output clean: the examples show no progress bars.
warnings.filterwarnings("ignore", message="IProgress not found.*")

__all__ = ["embedding", "chunking", "loaders", "sql", "llm"]
