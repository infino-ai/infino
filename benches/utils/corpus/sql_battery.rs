// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Schema-derived SQL query battery for real corpora.
//!
//! Real datasets carry their own column names, so a fixed query text
//! cannot serve them. Instead of vendoring an external benchmark's
//! queries, this module derives a small battery from the corpus's OWN
//! Arrow schema at load time: every query is built from columns the
//! dataset actually has, quoted verbatim, against the fixed table name
//! Infino's `query_sql` registers the corpus under. Value literals are
//! avoided entirely — filters compare against scalar subqueries
//! (`> (SELECT AVG(..))`), so the battery is valid for any data
//! distribution without knowing one row of it.

use arrow_schema::DataType;

use crate::harness::{SqlCorpusSpec, SqlQuery};

/// The fixed table name Infino's `query_sql` registers the corpus under
/// (`src/supertable/query/provider.rs`).
const TABLE: &str = "supertable";

/// Rows returned by the grouped top-k query — small enough that result
/// transfer never competes with the aggregation being measured.
const GROUP_TOP_LIMIT: usize = 10;

/// Whether a column is orderable — MIN/MAX are valid. Dates qualify.
fn supports_min_max(dt: &DataType) -> bool {
    supports_avg(dt) || matches!(dt, DataType::Date32)
}

/// Whether a column can be averaged — `AVG(date)` is rejected by the
/// engine, so [`supports_min_max`] admits `Date32` and this does not.
fn supports_avg(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Quote an identifier for SQL, doubling any embedded `"` — an Arrow
/// schema can legally carry quote characters in a column name.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The schema-derived battery for `spec`.
///
/// Shapes, each emitted only when the schema carries a fitting column:
/// full-table count, min/max aggregate over the first numeric column,
/// selective count above that column's mean, grouped top-K over the
/// first string column (preferring one that is not FTS-indexed, so the
/// group keys are low-fanout labels rather than document text), and a
/// distinct count over the same column. Deterministic for a given
/// schema: same corpus, same battery, comparable across runs.
/// The schema-derived battery for `spec`.
///
/// Shapes, each emitted only when the schema carries a fitting column:
/// full-table count; min/max over the first orderable column (dates
/// qualify); a selective count above the first AVERAGEABLE column's
/// mean (dates are excluded there — `AVG(date)` is rejected by the
/// engine, so a date-first schema still gets min/max but skips the
/// mean filter unless a true numeric exists); grouped top-K over the
/// first string column (preferring one that is not FTS-indexed, so the
/// group keys are low-fanout labels rather than document text); and a
/// distinct count over the same column. Deterministic for a given
/// schema: same corpus, same battery, comparable across runs.
///
/// Built fresh per call and leaked into `'static` (the [`SqlQuery`]
/// contract): bench-only code that runs once per corpus load, so the
/// leak is a handful of strings per process — and no process-global
/// cache means a second corpus in the same process can never be served
/// the first corpus's battery.
pub fn battery_for(spec: &SqlCorpusSpec) -> &'static [SqlQuery] {
    let mut queries: Vec<(&'static str, String)> = Vec::new();
    queries.push(("count_star", format!("SELECT COUNT(*) FROM {TABLE}")));

    let min_max_col = spec
        .schema
        .fields()
        .iter()
        .find(|f| supports_min_max(f.data_type()))
        .map(|f| quote_ident(f.name()));
    if let Some(col) = &min_max_col {
        queries.push((
            "numeric_min_max",
            format!("SELECT MIN({col}), MAX({col}) FROM {TABLE}"),
        ));
    }
    let avg_col = spec
        .schema
        .fields()
        .iter()
        .find(|f| supports_avg(f.data_type()))
        .map(|f| quote_ident(f.name()));
    if let Some(col) = &avg_col {
        queries.push((
            "count_above_mean",
            format!(
                "SELECT COUNT(*) FROM {TABLE} WHERE {col} > \
                 (SELECT AVG({col}) FROM {TABLE})"
            ),
        ));
    }

    let string_col = spec
        .schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8))
        .map(|f| f.name().clone())
        .min_by_key(|name| spec.fts_columns.contains(name))
        .map(|name| quote_ident(&name));
    if let Some(col) = &string_col {
        queries.push((
            "group_top_k",
            format!(
                "SELECT {col}, COUNT(*) AS n FROM {TABLE} \
                 GROUP BY {col} ORDER BY n DESC LIMIT {GROUP_TOP_LIMIT}"
            ),
        ));
        queries.push((
            "distinct_count",
            format!("SELECT COUNT(DISTINCT {col}) FROM {TABLE}"),
        ));
    }

    let battery: Vec<SqlQuery> = queries
        .into_iter()
        .map(|(name, sql)| SqlQuery {
            name,
            sql: Box::leak(sql.into_boxed_str()),
        })
        .collect();
    Box::leak(battery.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{Field, Schema};

    use super::*;

    /// The battery adapts to the schema it is handed: every query names
    /// only columns the schema carries, quoted, and the shapes that need
    /// a numeric or string column appear exactly when one exists.
    #[test]
    fn battery_derives_from_schema_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("EventDate", DataType::Date32, false),
            Field::new("Title", DataType::Utf8, true),
            Field::new("body", DataType::Utf8, true),
        ]));
        let spec = SqlCorpusSpec {
            schema,
            fts_columns: vec!["body".into()],
            vector: None,
        };
        let battery = battery_for(&spec);
        let names: Vec<&str> = battery.iter().map(|q| q.name).collect();
        assert!(names.contains(&"count_star"));
        assert!(names.contains(&"numeric_min_max"));
        assert!(names.contains(&"group_top_k"));
        // Date32 is orderable but NOT averageable: this schema's only
        // orderable column is the date, so min/max targets it and the
        // mean filter is absent — never `AVG(date)`.
        let min_max = battery
            .iter()
            .find(|q| q.name == "numeric_min_max")
            .unwrap();
        assert!(min_max.sql.contains("\"EventDate\""));
        assert!(!names.contains(&"count_above_mean"));
        // The grouped query prefers the non-FTS string column.
        let group = battery.iter().find(|q| q.name == "group_top_k").unwrap();
        assert!(group.sql.contains("\"Title\""));
        // Every query targets the registered table, no external names.
        for q in battery {
            assert!(q.sql.contains(TABLE), "{} must query {TABLE}", q.name);
        }
    }

    /// A true numeric column gets the mean filter, an embedded quote in
    /// a column name is doubled, and two schemas in one process each get
    /// their own battery (no process-global cache to go stale).
    #[test]
    fn battery_handles_numerics_quoting_and_multiple_schemas() {
        let spec = SqlCorpusSpec {
            schema: Arc::new(Schema::new(vec![
                Field::new("watch\"count", DataType::Int64, true),
                Field::new("label", DataType::Utf8, true),
            ])),
            fts_columns: Vec::new(),
            vector: None,
        };
        let battery = battery_for(&spec);
        let mean = battery
            .iter()
            .find(|q| q.name == "count_above_mean")
            .expect("Int64 column must produce the mean filter");
        assert!(mean.sql.contains("\"watch\"\"count\""), "{}", mean.sql);

        let other = SqlCorpusSpec {
            schema: Arc::new(Schema::new(vec![Field::new(
                "score",
                DataType::Float32,
                true,
            )])),
            fts_columns: Vec::new(),
            vector: None,
        };
        let other_battery = battery_for(&other);
        let other_min_max = other_battery
            .iter()
            .find(|q| q.name == "numeric_min_max")
            .expect("second schema derives its own battery");
        assert!(
            other_min_max.sql.contains("\"score\""),
            "stale battery served"
        );
    }
}
