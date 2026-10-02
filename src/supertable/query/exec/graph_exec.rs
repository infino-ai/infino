// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Graph walks as DataFusion table-valued functions.
//!
//! `graph_walk(seeds, hops, k)` and `graph_rank(seeds, hops, k)` register
//! via `register_udtf` over an edge table's resident adjacency index (see
//! `supertable::query::graph`) and return the nodes a walk from the seeds
//! reaches, so a statement joins them to the tables whose rows they name:
//!
//! ```sql
//! SELECT t.*, g.hop
//! FROM graph_walk(['issues/42'], 2, 200) AS g
//! JOIN chunks AS t ON t.key = g.key
//! ```
//!
//! Seeds are node keys — one string literal, or an array literal of them —
//! as the edge table's key column spells them; `hops` bounds the walk and
//! `k` the rows, like every search function's `k`. `graph_walk` lists the
//! nodes nearest first; `graph_rank` orders them by personalized PageRank
//! from the seeds, so a hub's many leaves do not outrank what the seeds
//! share.
//!
//! The walk runs when the function is called, at plan time: it is
//! sub-millisecond warm and tens of milliseconds on the first touch, when
//! the index hydrates, and its result is a few hundred rows at most. So a
//! batch served through `MemTable` is the whole provider — there is no scan
//! to push a predicate into and no kernel to run per partition, which is
//! what the search functions' custom execution plans exist for.
//!
//! Output: `key Utf8`, `id Int64` (the edge table's `src` / `dst` value),
//! `hop UInt32` (fewest edges from a seed; the seeds are 0) and
//! `score Float64` (the ranking's score; 0 from a walk).

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{
    Array, Float64Array, Int64Array, ListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider},
    datasource::MemTable,
    error::{DataFusionError, Result as DfResult},
    logical_expr::Expr,
    scalar::ScalarValue,
};

use crate::supertable::{
    handle::{SupertableReader, WeakReader},
    query::{
        exec::common::{arg_to_string, arg_to_usize},
        graph::GraphHit,
    },
};

/// SQL name of the breadth-first walk.
pub(crate) const GRAPH_WALK_UDTF: &str = "graph_walk";
/// SQL name of the ranked walk.
pub(crate) const GRAPH_RANK_UDTF: &str = "graph_rank";
/// Argument count for `graph_walk(seeds, hops, k)` / `graph_rank(seeds, hops, k)`.
const GRAPH_ARG_COUNT: usize = 3;
/// The output columns, in order.
pub(crate) const GRAPH_KEY_COLUMN: &str = "key";
pub(crate) const GRAPH_ID_COLUMN: &str = "id";
pub(crate) const GRAPH_HOP_COLUMN: &str = "hop";
pub(crate) const GRAPH_SCORE_COLUMN: &str = "score";

/// Which traversal a function runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Traversal {
    /// Nearest first.
    Walk,
    /// Highest personalized-PageRank score first.
    Rank,
}

impl Traversal {
    /// The SQL name the traversal is registered under.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Traversal::Walk => GRAPH_WALK_UDTF,
            Traversal::Rank => GRAPH_RANK_UDTF,
        }
    }
}

/// The output schema of both functions.
pub(crate) fn graph_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(GRAPH_KEY_COLUMN, DataType::Utf8, false),
        Field::new(GRAPH_ID_COLUMN, DataType::Int64, false),
        Field::new(GRAPH_HOP_COLUMN, DataType::UInt32, false),
        Field::new(GRAPH_SCORE_COLUMN, DataType::Float64, false),
    ]))
}

/// `TableFunctionImpl` for `graph_walk` / `graph_rank`, bound to the edge
/// table's pinned reader. `call_with_args` parses the arguments, runs the
/// traversal and hands back its rows as a `MemTable`.
#[derive(Debug)]
pub(crate) struct GraphFunc {
    reader: WeakReader,
    traversal: Traversal,
}

impl GraphFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, traversal: Traversal) -> Self {
        Self {
            reader: WeakReader::from_reader(&reader),
            traversal,
        }
    }
}

impl TableFunctionImpl for GraphFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let name = self.traversal.name();
        let args = args.exprs();
        if args.len() != GRAPH_ARG_COUNT {
            return Err(DataFusionError::Plan(format!(
                "{name} expects {GRAPH_ARG_COUNT} arguments (seeds, hops, k), got {}",
                args.len()
            )));
        }
        let seeds = arg_to_strings(&args[0], &format!("{name} seeds"))?;
        let hops = arg_to_usize(&args[1], &format!("{name} hops"))?;
        let hops = u32::try_from(hops)
            .map_err(|_| DataFusionError::Plan(format!("{name} hops must fit in 32 bits")))?;
        let k = arg_to_usize(&args[2], &format!("{name} k"))?;
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "{name}: supertable consumer dropped before execution"
            ))
        })?;
        let keys: Vec<&str> = seeds.iter().map(String::as_str).collect();
        let hits = match self.traversal {
            Traversal::Walk => reader.graph_walk_keys(&keys, hops, k),
            Traversal::Rank => reader.graph_rank_keys(&keys, hops, k),
        }
        .map_err(|e| DataFusionError::Execution(format!("{name}: {e}")))?;
        let table = MemTable::try_new(graph_schema(), vec![vec![hits_batch(&hits)?]])?;
        Ok(Arc::new(table))
    }
}

/// The traversal's rows as one batch of the output schema.
fn hits_batch(hits: &[GraphHit]) -> DfResult<RecordBatch> {
    RecordBatch::try_new(
        graph_schema(),
        vec![
            Arc::new(StringArray::from(
                hits.iter().map(|hit| hit.key.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                hits.iter().map(|hit| hit.hop).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                hits.iter().map(|hit| hit.score).collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|e| DataFusionError::Execution(e.to_string()))
}

/// Extract one string literal or an array literal of strings (`['a', 'b']`,
/// which the planner const-folds to a `List` scalar, or leaves as
/// `make_array(...)` when an element is not a plain literal).
pub(crate) fn arg_to_strings(expr: &Expr, what: &str) -> DfResult<Vec<String>> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(_)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(_)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(_)), _) => Ok(vec![arg_to_string(expr, what)?]),
        Expr::Literal(ScalarValue::List(list), _) => list_literal_to_strings(list, what),
        Expr::ScalarFunction(sf) if sf.func.name() == "make_array" => {
            sf.args.iter().map(|arg| arg_to_string(arg, what)).collect()
        }
        other => Err(DataFusionError::Plan(format!(
            "{what} must be a string literal or an array literal of strings, got {other:?}"
        ))),
    }
}

/// The strings of a single-row `List` scalar (`['a', 'b']`).
fn list_literal_to_strings(list: &ListArray, what: &str) -> DfResult<Vec<String>> {
    if list.len() != 1 {
        return Err(DataFusionError::Plan(format!(
            "{what} list literal must have exactly one row, got {}",
            list.len()
        )));
    }
    let values = cast(&list.value(0), &DataType::Utf8).map_err(|e| {
        DataFusionError::Plan(format!("{what}: cannot read the elements as strings: {e}"))
    })?;
    let strings = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Plan(format!("{what}: cast did not yield Utf8")))?;
    if strings.null_count() > 0 {
        return Err(DataFusionError::Plan(format!(
            "{what} contains null elements"
        )));
    }
    Ok(strings.iter().flatten().map(str::to_string).collect())
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, LargeStringArray, StringArray};
    use datafusion::{
        functions_nested::make_array::MakeArray, logical_expr::ScalarUDF, prelude::lit,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::{
        catalog::{Connection, IndexSpec, connect},
        config::{CompactionSettings, OptimizeOptions},
    };

    /// A chain `n1 - n2 - n3 - n4` of nodes keyed `n<id>`, every edge both
    /// ways, in the table `edges` of `db`, indexed; and a table `things`
    /// naming each node.
    fn seed(db: &Connection) {
        let edge_schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("key", DataType::LargeUtf8, false),
        ]));
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for (s, d) in [(1i64, 2i64), (2, 3), (3, 4)] {
            src.extend([s, d]);
            dst.extend([d, s]);
        }
        let keys: Vec<String> = src.iter().map(|s| format!("n{s}")).collect();
        let edges = RecordBatch::try_new(
            edge_schema.clone(),
            vec![
                Arc::new(Int64Array::from(src)),
                Arc::new(Int64Array::from(dst)),
                Arc::new(LargeStringArray::from(keys)),
            ],
        )
        .expect("edges");
        let table = db
            .create_table("edges", edge_schema, IndexSpec::new())
            .expect("create edges");
        table.append(&edges).expect("append edges");
        table
            .optimize(
                &OptimizeOptions::compact(CompactionSettings::default())
                    .with_adjacency("src", "dst", "key"),
            )
            .expect("optimize");
        things(db);
    }

    fn things(db: &Connection) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["n1", "n2", "n3", "n4"])),
                Arc::new(StringArray::from(vec!["one", "two", "three", "four"])),
            ],
        )
        .expect("things");
        db.create_table("things", schema, IndexSpec::new())
            .expect("create things")
            .append(&batch)
            .expect("append things");
    }

    fn column_strings(batches: &[RecordBatch], column: usize) -> Vec<String> {
        batches
            .iter()
            .flat_map(|batch| {
                let values = cast(batch.column(column), &DataType::Utf8).expect("cast");
                values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("utf8")
                    .iter()
                    .flatten()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The functions are relations: a walk's rows join to the table whose
    /// rows they name, hop by hop; a ranking from two seeds puts the node
    /// they share first; `k` bounds the rows.
    #[test]
    fn a_statement_walks_the_graph_and_joins_its_rows() {
        let dir = TempDir::new().expect("tempdir");
        let db = connect(format!("file://{}", dir.path().display())).expect("connect");
        seed(&db);

        let rows = db
            .query_sql(
                "SELECT t.name, g.hop FROM graph_walk('edges', 'n1', 2, 10) AS g \
                 JOIN things AS t ON t.key = g.key ORDER BY g.hop",
            )
            .expect("walk");
        assert_eq!(column_strings(&rows, 0), vec!["one", "two", "three"]);
        assert_eq!(column_strings(&rows, 1), vec!["0", "1", "2"]);

        let rows = db
            .query_sql(
                "SELECT key FROM graph_rank('edges', ['n1', 'n3'], 1, 10) ORDER BY score DESC",
            )
            .expect("rank");
        let ranked = column_strings(&rows, 0);
        let position = |key: &str| ranked.iter().position(|k| k == key).expect("ranked");
        assert!(
            position("n2") < position("n4"),
            "the node both seeds reach outranks the one only n3 reaches: {ranked:?}"
        );

        let rows = db
            .query_sql("SELECT key FROM graph_walk('edges', ['n1'], 3, 2)")
            .expect("bounded");
        assert_eq!(
            column_strings(&rows, 0),
            vec!["n1", "n2"],
            "k bounds the rows"
        );

        assert!(
            db.query_sql("SELECT key FROM graph_walk('edges', 'n1', 2)")
                .is_err(),
            "k is required"
        );
        assert!(
            db.query_sql("SELECT key FROM graph_walk('things', 'n1', 2, 10)")
                .is_err(),
            "a table without an adjacency index is refused"
        );
    }

    /// A graph attached from another catalog serves the functions without a
    /// table argument, and its table is not a table of the connection it is
    /// attached to: not listed, not scannable by name.
    #[test]
    fn an_attached_graph_is_walked_without_naming_its_table() {
        let dir = TempDir::new().expect("tempdir");
        let graph = connect(format!("file://{}", dir.path().display())).expect("connect");
        seed(&graph);
        let customer = connect("memory://").expect("connect");
        things(&customer);
        customer.attach_graph(graph.clone(), "edges");

        let rows = customer
            .query_sql(
                "SELECT t.name FROM graph_walk('n1', 1, 10) AS g \
                 JOIN things AS t ON t.key = g.key ORDER BY g.hop",
            )
            .expect("walk through the attached graph");
        assert_eq!(column_strings(&rows, 0), vec!["one", "two"]);
        assert_eq!(
            customer.list_tables().expect("tables"),
            vec!["things".to_string()],
            "the edge table is the graph catalog's alone"
        );
        assert!(
            customer.query_sql("SELECT count(*) FROM edges").is_err(),
            "not reachable by name"
        );
    }

    /// One string, an array literal and an unfolded `make_array` all read
    /// as seeds; anything else is refused with the argument named.
    #[test]
    fn seeds_read_from_a_string_or_an_array() {
        assert_eq!(arg_to_strings(&lit("a"), "seeds").expect("one"), vec!["a"]);
        let list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::from("a"), ScalarValue::from("b")],
            &DataType::Utf8,
        ));
        assert_eq!(
            arg_to_strings(&Expr::Literal(list, None), "seeds").expect("list"),
            vec!["a", "b"]
        );
        let make_array = ScalarUDF::from(MakeArray::new());
        let unfolded = make_array.call(vec![lit("x"), lit("y")]);
        assert_eq!(
            arg_to_strings(&unfolded, "seeds").expect("make_array"),
            vec!["x", "y"]
        );
        let refused = arg_to_strings(&lit(3), "seeds").expect_err("not a string");
        assert!(refused.to_string().contains("seeds"), "{refused}");
    }
}
