"""Fast search on object storage — SQL, full-text, and vector search.

The public API is a thin re-export of the compiled `infino._infino`
extension; this package adds typing artifacts and module metadata.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from infino._infino import (
    Connection,
    GcReport,
    IndexSpec,
    MutationStats,
    OptimizeOptions,
    Table,
    connect,
)

try:
    __version__ = version("infino")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"

__all__ = [
    "connect",
    "Connection",
    "Table",
    "IndexSpec",
    "MutationStats",
    "GcReport",
    "OptimizeOptions",
]
