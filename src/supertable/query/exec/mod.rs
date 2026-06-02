//! DataFusion `ExecutionPlan` nodes + table-valued functions that
//! expose infino's retrieval kernels through the SQL surface.
//!
//! Each search TVF registers via `register_udtf` and lowers to a
//! custom `ExecutionPlan` that calls the matching async kernel on
//! [`SupertableReader`](crate::supertable::handle::SupertableReader)
//! inside `execute()`, resolving each
//! [`SuperfileHit`](crate::supertable::query::SuperfileHit) to the
//! supertable's `_id` + projected scalar columns via
//! [`SuperfileReader::take_by_local_doc_ids`](crate::superfile::SuperfileReader::take_by_local_doc_ids).

pub mod common;
pub mod fts_exec;
pub mod hybrid_exec;
pub mod vector_exec;
