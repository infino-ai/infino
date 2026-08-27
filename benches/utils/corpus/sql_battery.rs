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

use std::sync::OnceLock;

use arrow_schema::DataType;

use crate::harness::{SqlCorpusSpec, SqlQuery};

/// The fixed table name Infino's `query_sql` registers the corpus under
/// (`src/supertable/query/provider.rs`).
const TABLE: &str = "supertable";

/// Rows returned by the grouped top-k query — small enough that result
/// transfer never competes with the aggregation being measured.
const GROUP_TOP_LIMIT: usize = 10;

/// Generated query text, built once from the first corpus spec this
/// process benches (one bench process loads one corpus). Held in a
/// static so [`SqlQuery`]'s `&'static str` fields can borrow from it.
static TEXT: OnceLock<Vec<(&'static str, String)>> = OnceLock::new();
/// The battery borrowing from [`TEXT`].
static BATTERY: OnceLock<Vec<SqlQuery>> = OnceLock::new();

/// Whether a column is a plain orderable numeric the aggregates can use.
fn is_numeric(dt: &DataType) -> bool {
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
            | DataType::Date32
    )
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
pub fn battery_for(spec: &SqlCorpusSpec) -> &'static [SqlQuery] {
    let text = TEXT.get_or_init(|| {
        let mut queries: Vec<(&'static str, String)> = Vec::new();
        queries.push(("count_star", format!("SELECT COUNT(*) FROM {TABLE}")));

        let numeric = spec
            .schema
            .fields()
            .iter()
            .find(|f| is_numeric(f.data_type()))
            .map(|f| f.name().clone());
        if let Some(col) = &numeric {
            queries.push((
                "numeric_min_max",
                format!("SELECT MIN(\"{col}\"), MAX(\"{col}\") FROM {TABLE}"),
            ));
            queries.push((
                "count_above_mean",
                format!(
                    "SELECT COUNT(*) FROM {TABLE} WHERE \"{col}\" > \
                     (SELECT AVG(\"{col}\") FROM {TABLE})"
                ),
            ));
        }

        let string_col = spec
            .schema
            .fields()
            .iter()
            .filter(|f| matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8))
            .map(|f| f.name().clone())
            .min_by_key(|name| spec.fts_columns.contains(name));
        if let Some(col) = &string_col {
            queries.push((
                "group_top_k",
                format!(
                    "SELECT \"{col}\", COUNT(*) AS n FROM {TABLE} \
                     GROUP BY \"{col}\" ORDER BY n DESC LIMIT {GROUP_TOP_LIMIT}"
                ),
            ));
            queries.push((
                "distinct_count",
                format!("SELECT COUNT(DISTINCT \"{col}\") FROM {TABLE}"),
            ));
        }
        queries
    });
    BATTERY.get_or_init(|| {
        text.iter()
            .map(|(name, sql)| SqlQuery {
                name,
                sql: sql.as_str(),
            })
            .collect()
    })
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
        // The grouped query prefers the non-FTS string column.
        let group = battery.iter().find(|q| q.name == "group_top_k").unwrap();
        assert!(group.sql.contains("\"Title\""));
        // Every query targets the registered table, no external names.
        for q in battery {
            assert!(q.sql.contains(TABLE), "{} must query {TABLE}", q.name);
        }
    }
}
