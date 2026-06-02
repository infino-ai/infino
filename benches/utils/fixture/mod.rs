//! Process-wide shared bench state (build once per `cargo bench` process).
//!
//! Not Infino product terminology — just "expensive setup reused by every
//! criterion group" (one object-storage ingest + one search consumer).

pub mod supertable;
