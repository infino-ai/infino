from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pyarrow import RecordBatch, Schema, Table as ArrowTable

Metric: TypeAlias = Literal["cosine", "l2sq", "l2", "negdot", "dot"]
BoolMode: TypeAlias = Literal["or", "and"]
Bm25Stats: TypeAlias = Literal["per_superfile", "global"]
ColdFetchMode: TypeAlias = Literal[
    "hybrid_with_prefetch",
    "range_only",
    "lazy_foreground_with_background_fill",
]

# Inputs `append` / `update` coerce to Arrow under the table's declared
# schema. A pandas `DataFrame` is also accepted at runtime but is omitted
# here deliberately: typing it would couple these stubs to pandas' optional
# type information. For a statically-typed path, convert with
# `pyarrow.Table.from_pandas(df)`.
RowData: TypeAlias = RecordBatch | ArrowTable | Sequence[Mapping[str, Any]]

def connect(
    uri: str,
    *,
    storage_options: Mapping[str, str] | None = ...,
    cache_dir: str | None = ...,
    cache_budget_bytes: int | None = ...,
    connection_memory_budget_bytes: int | None = ...,
    cold_fetch_mode: ColdFetchMode | None = ...,
    validate: bool | None = ...,
) -> Connection: ...

class InfinoError(Exception):
    """Base class for infino's errors. Catch it to handle any infino failure."""

class ConnectionMemoryBudgetError(InfinoError):
    """Raised when an ingest or query would exceed the connection's memory budget
    (set via ``connect(connection_memory_budget_bytes=...)``). Recoverable: catch
    it and back off, e.g. narrow the query, split the ingest, or raise the budget."""

class Connection:
    def create_database(self) -> None: ...
    def create_table(self, name: str, schema: Schema, indexes: IndexSpec) -> Table: ...
    def open_table(self, name: str) -> Table: ...
    def drop_table(self, name: str, purge: bool = ...) -> None: ...
    def list_tables(self) -> list[str]: ...
    def query_sql(self, sql: str) -> ArrowTable: ...

class IndexSpec:
    def __init__(self) -> None: ...
    def fts(self, column: str, analyzer: str | None = None) -> IndexSpec: ...
    # `dim` must be in [16, 4096]; out-of-range raises at `create_table`.
    def vector(self, column: str, dim: int, n_cent: int, metric: Metric) -> IndexSpec: ...

class Table:
    def append(self, data: RowData) -> None: ...
    def bm25_search(
        self,
        column: str,
        query: str,
        k: int,
        mode: BoolMode | None = ...,
        projection: Sequence[str] | None = ...,
        stats: Bm25Stats | None = ...,
    ) -> ArrowTable: ...
    def vector_search(
        self,
        column: str,
        query: Sequence[float],
        k: int,
        nprobe: int | None = ...,
        rerank_mult: int | None = ...,
        filter_column: str | None = ...,
        filter_query: str | None = ...,
        filter_mode: BoolMode | None = ...,
        projection: Sequence[str] | None = ...,
    ) -> ArrowTable: ...
    def token_match(
        self,
        column: str,
        query: str,
        mode: BoolMode | None = ...,
        projection: Sequence[str] | None = ...,
    ) -> ArrowTable: ...
    def exact_match(
        self,
        column: str,
        value: str,
        projection: Sequence[str] | None = ...,
    ) -> ArrowTable: ...
    def count(
        self,
        column: str,
        query: str,
        mode: BoolMode | None = ...,
    ) -> int: ...
    def hybrid_search(
        self,
        text_column: str,
        text_query: str,
        vector_column: str,
        vector_query: Sequence[float],
        k: int,
        mode: BoolMode | None = ...,
        nprobe: int | None = ...,
        rerank_mult: int | None = ...,
        projection: Sequence[str] | None = ...,
    ) -> ArrowTable: ...
    def delete(self, predicate: str) -> MutationStats: ...
    def update(self, predicate: str, new_rows: RowData) -> MutationStats: ...
    def optimize(self, settings: OptimizeOptions | None = ...) -> None: ...
    def gc(self, grace_secs: float) -> GcReport: ...
    def schema(self) -> Schema: ...

class MutationStats:
    @property
    def matched(self) -> int: ...
    @property
    def n_tombstoned(self) -> int: ...
    @property
    def n_not_found(self) -> int: ...
    def __repr__(self) -> str: ...

class GcReport:
    @property
    def bytes_freed(self) -> int: ...
    @property
    def objects_deleted(self) -> int: ...
    @property
    def objects_skipped_live(self) -> int: ...
    @property
    def objects_skipped_too_new(self) -> int: ...
    @property
    def delete_errors(self) -> int: ...
    def __repr__(self) -> str: ...

class OptimizeOptions:
    def __init__(
        self,
        *,
        max_memory_mb: int | None = ...,
        min_fill_percent: int | None = ...,
        target_superfile_size_mb: int | None = ...,
        stale_seal_timeout_ms: int | None = ...,
    ) -> None: ...
