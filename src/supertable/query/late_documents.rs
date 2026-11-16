// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`LateDocuments`]: read a document column after a plan's limit, for the
//! rows the limit keeps, rather than for every row the scan reads.
//!
//! The provider plans a document column as a rebuild over a scan of every
//! column a document can draw on ([`DocumentExec`]), so a filtered `LIMIT 10`
//! rebuilds a document for each row of each batch it scans, then keeps ten.
//! Where nothing between the limit and the rebuild reads the document, this
//! rule reads it in two phases instead:
//!
//! ```text
//!   GlobalLimitExec fetch=10                LateDocumentExec rebuild=[source]
//!     FilterExec name = 'kernel'              GlobalLimitExec fetch=10
//!       DocumentExec rebuild=[source]           FilterExec name = 'kernel'
//!         scan: name, osid, _id,                  ProjectionExec source ← NULL,
//!               source#skeleton,                                 _id#row ← _id
//!               source#residual,                    scan: name, osid, _id
//!               every leaf
//! ```
//!
//! Phase 1 scans only what the rest of the plan reads, plus `_id`, with a
//! null placeholder where the document goes. Phase 2 ([`LateDocumentExec`])
//! takes the rows that come out of the limit, finds each at the superfile
//! and row the scan read it from by its `_id` ([`LateRows`]), and rebuilds
//! the documents from those rows alone.
//!
//! The rule fires on a node that limits its output (`fetch`, at most
//! [`MAX_FETCH`]), the top-most such node first, when every node from it
//! down to the rebuild is one that passes rows through without reading the
//! document:
//!
//! - a projection that passes the document on as a bare column (renamed or
//!   repeated) and reads it in no other expression;
//! - a filter, a sort (a top-k or a full sort), an order-preserving merge
//!   or a hash repartition whose expressions read no document;
//! - a limit, a gather of partitions, a round-robin repartition, or
//!   DataFusion's cooperative wrapper.
//!
//! Each of these gives the same rows in the same order whatever the
//! document holds, so moving the rebuild above them is exact. Anything else
//! between them (an aggregate, a join, a window, a node of ours) keeps the
//! rebuild where the provider put it; a limit lower in the plan may still
//! take phase 2. A plan with no limit keeps it too: every row it scans
//! reaches the result, and a scan reads each page once where phase 2 would
//! seek row by row. So does a scan that already stops at the limit itself
//! (no `WHERE`), which rebuilds no more rows than the limit keeps.
//!
//! The rule runs after DataFusion's rules, once limits and top-ks have their
//! final places, and before ours that wrap gathers and scans
//! (`StageLimitGathers`, `ReadAheadOff`), which then see phase 1's plain
//! nodes.

use std::sync::Arc;

use datafusion::{
    common::ScalarValue,
    config::ConfigOptions,
    error::Result as DfResult,
    physical_expr::{
        Partitioning, PhysicalExpr,
        expressions::{Column, Literal},
        projection::ProjectionExpr,
        utils::collect_columns,
    },
    physical_optimizer::{PhysicalOptimizerRule, projection_pushdown::ProjectionPushdown},
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties,
        coalesce_partitions::CoalescePartitionsExec,
        coop::CooperativeExec,
        filter::{FilterExec, FilterExecBuilder},
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use crate::supertable::{
    query::exec::document_exec::{DocumentExec, LateDocumentExec, LateRows, Output},
    schema::DocumentColumn,
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "LateDocuments";

/// Largest limit the rule reads documents late for. Phase 2 reads its rows
/// one by one, which pays against a scan's whole pages only while they are
/// few; the same bound as `RowFilterUnderTopK`'s, not measured for this rule.
const MAX_FETCH: usize = 10_000;

/// Moves a document column's rebuild above the limit that keeps its rows,
/// where nothing between them reads it. See the module docs.
#[derive(Debug, Default)]
pub(crate) struct LateDocuments;

impl PhysicalOptimizerRule for LateDocuments {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        rewrite(plan, config)
    }

    fn name(&self) -> &str {
        RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// `plan` with each limited chain over a document rebuild read in two
/// phases, the top-most limit of a chain first.
fn rewrite(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    if plan.fetch().is_some_and(|fetch| fetch <= MAX_FETCH)
        && let Some(threaded) = thread(&plan, config)?
    {
        let documents: Vec<(usize, DocumentColumn)> = threaded
            .documents
            .into_iter()
            .enumerate()
            .filter_map(|(at, document)| Some((at, document?)))
            .collect();
        if !documents.is_empty() {
            return Ok(Arc::new(LateDocumentExec::try_new(
                threaded.plan,
                &plan,
                documents,
                threaded.late,
            )?));
        }
    }
    let children: Vec<Arc<dyn ExecutionPlan>> = plan.children().into_iter().cloned().collect();
    if children.is_empty() {
        return Ok(plan);
    }
    let rewritten = children
        .iter()
        .map(|child| rewrite(Arc::clone(child), config))
        .collect::<DfResult<Vec<_>>>()?;
    match children
        .iter()
        .zip(&rewritten)
        .all(|(old, new)| Arc::ptr_eq(old, new))
    {
        true => Ok(plan),
        false => plan.with_new_children(rewritten),
    }
}

/// A node of a chain rewritten for phase 1: its columns, then each row's
/// `_id` as one more.
struct Threaded {
    plan: Arc<dyn ExecutionPlan>,
    /// The document each column of the original node holds, if any (a
    /// placeholder in `plan`).
    documents: Vec<Option<DocumentColumn>>,
    late: Arc<LateRows>,
}

impl Threaded {
    /// Whether `expr`, over this node's columns, reads a document.
    fn reads_document(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        collect_columns(expr).iter().any(|column| {
            self.documents
                .get(column.index())
                .is_some_and(Option::is_some)
        })
    }

    /// The position of the `_id` column in `plan`.
    fn row_id(&self) -> usize {
        self.plan.schema().fields().len() - 1
    }

    /// The name phase 1 gives the `_id` it carries up.
    fn row_id_name(&self) -> String {
        row_id_name(&self.late)
    }
}

/// The name phase 1 gives the `_id` it carries up: not a column a table
/// can have, so it never meets one the plan already passes on.
fn row_id_name(late: &LateRows) -> String {
    format!("{}#row", late.row_id())
}

/// `plan` rewritten for phase 1, when it is a chain of pass-through nodes
/// down to a rebuild that can read late; `None` otherwise.
fn thread(plan: &Arc<dyn ExecutionPlan>, config: &ConfigOptions) -> DfResult<Option<Threaded>> {
    if let Some(rebuild) = plan.downcast_ref::<DocumentExec>() {
        return phase_one(rebuild, config);
    }
    let [input] = plan.children()[..] else {
        return Ok(None);
    };
    let Some(child) = thread(input, config)? else {
        return Ok(None);
    };
    // A node that needs its input in an order still gets it: phase 1 keeps
    // every column the order is on.
    if let Some(ordering) = input.output_ordering()
        && !child
            .plan
            .equivalence_properties()
            .ordering_satisfy(ordering.clone())?
    {
        return Ok(None);
    }
    let passes_on = |node: Arc<dyn ExecutionPlan>, child: Threaded| Threaded {
        plan: node,
        documents: child.documents,
        late: child.late,
    };

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let mut exprs: Vec<ProjectionExpr> = Vec::with_capacity(projection.expr().len() + 1);
        let mut documents = Vec::with_capacity(projection.expr().len());
        for expr in projection.expr() {
            let passed = expr
                .expr
                .downcast_ref::<Column>()
                .and_then(|column| child.documents.get(column.index()).cloned().flatten());
            match passed {
                Some(document) => documents.push(Some(document)),
                None if child.reads_document(&expr.expr) => return Ok(None),
                None => documents.push(None),
            }
            exprs.push(expr.clone());
        }
        let name = child.row_id_name();
        exprs.push(ProjectionExpr {
            expr: Arc::new(Column::new(&name, child.row_id())),
            alias: name,
        });
        let node = Arc::new(ProjectionExec::try_new(exprs, Arc::clone(&child.plan))?);
        return Ok(Some(Threaded {
            plan: node,
            documents,
            late: child.late,
        }));
    }

    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        if child.reads_document(filter.predicate()) {
            return Ok(None);
        }
        let row_id = child.row_id();
        let (projection, documents) = match filter.projection() {
            Some(kept) => (
                Some(kept.iter().copied().chain([row_id]).collect::<Vec<_>>()),
                kept.iter().map(|&at| child.documents[at].clone()).collect(),
            ),
            None => (None, child.documents.clone()),
        };
        let node = FilterExecBuilder::from(filter)
            .with_input(Arc::clone(&child.plan))
            .apply_projection(None)?
            .apply_projection(projection)?
            .build()?;
        return Ok(Some(Threaded {
            plan: Arc::new(node),
            documents,
            late: child.late,
        }));
    }

    let reads = if let Some(sort) = plan.downcast_ref::<SortExec>() {
        sort.expr()
            .iter()
            .any(|key| child.reads_document(&key.expr))
    } else if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        merge
            .expr()
            .iter()
            .any(|key| child.reads_document(&key.expr))
    } else if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
        match repartition.partitioning() {
            Partitioning::RoundRobinBatch(_) => false,
            Partitioning::Hash(exprs, _) => exprs.iter().any(|expr| child.reads_document(expr)),
            _ => return Ok(None),
        }
    } else if plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<CooperativeExec>()
    {
        false
    } else {
        return Ok(None);
    };
    if reads {
        return Ok(None);
    }
    let node = Arc::clone(plan).with_new_children(vec![Arc::clone(&child.plan)])?;
    Ok(Some(passes_on(node, child)))
}

/// Phase 1 in place of `rebuild`: its scan narrowed to the columns passed
/// through and `_id`, a null where each document goes, and `_id` last.
/// `None` when the rebuild cannot read late, when its scan already stops at
/// the limit, or when the scan cannot be narrowed.
fn phase_one(rebuild: &DocumentExec, config: &ConfigOptions) -> DfResult<Option<Threaded>> {
    let Some(late) = rebuild.late() else {
        return Ok(None);
    };
    let input = rebuild.input();
    if input.fetch().is_some() {
        return Ok(None);
    }
    let input_schema = input.schema();
    let Ok(row_id) = input_schema.index_of(late.row_id()) else {
        return Ok(None);
    };

    // The scan's columns phase 1 keeps, each once, in the order first used.
    let mut kept: Vec<usize> = Vec::new();
    let mut keep = |at: usize| match kept.iter().position(|&k| k == at) {
        Some(position) => position,
        None => {
            kept.push(at);
            kept.len() - 1
        }
    };
    let mut positions: Vec<Option<usize>> = Vec::new();
    for output in rebuild.outputs() {
        positions.push(match output {
            Output::Input(at) => Some(keep(*at)),
            Output::Document(_) => None,
        });
    }
    let row_id_at = keep(row_id);
    let narrowing = ProjectionExec::try_new(
        kept.iter().map(|&at| {
            let name = input_schema.field(at).name().clone();
            (
                Arc::new(Column::new(&name, at)) as Arc<dyn PhysicalExpr>,
                name,
            )
        }),
        Arc::clone(input),
    )?;
    let narrowed = ProjectionPushdown::new().optimize(Arc::new(narrowing), config)?;
    // The narrowing pays only when it reaches the scan: a scan still
    // reading a document's stored columns would read them for every row,
    // and phase 2 again.
    let stored: Vec<&str> = rebuild
        .outputs()
        .iter()
        .filter_map(|output| match output {
            Output::Document(binding) => Some(binding.document()),
            Output::Input(_) => None,
        })
        .flat_map(|document| [document.skeleton.as_str(), document.residual.as_str()])
        .collect();
    if reads_any(&narrowed, &stored) {
        return Ok(None);
    }

    let schema = rebuild.schema();
    let mut exprs: Vec<ProjectionExpr> = Vec::with_capacity(schema.fields().len() + 1);
    let mut documents = Vec::with_capacity(schema.fields().len());
    for ((field, output), position) in schema.fields().iter().zip(rebuild.outputs()).zip(positions)
    {
        let expr: Arc<dyn PhysicalExpr> = match (output, position) {
            (Output::Document(binding), _) => {
                documents.push(Some(binding.document().clone()));
                Arc::new(Literal::new(ScalarValue::try_from(field.data_type())?))
            }
            (Output::Input(_), Some(position)) => {
                documents.push(None);
                Arc::new(Column::new(field.name(), position))
            }
            (Output::Input(_), None) => unreachable!("a passed-through column is kept"),
        };
        exprs.push(ProjectionExpr {
            expr,
            alias: field.name().clone(),
        });
    }
    // The scan's `_id`, renamed here to the name phase 1 carries it under.
    exprs.push(ProjectionExpr {
        expr: Arc::new(Column::new(late.row_id(), row_id_at)),
        alias: row_id_name(late),
    });
    let placeholder = ProjectionExec::try_new(exprs, narrowed)?;
    Ok(Some(Threaded {
        plan: Arc::new(placeholder),
        documents,
        late: Arc::clone(late),
    }))
}

/// Whether a leaf under `plan` reads a column named in `names`.
fn reads_any(plan: &Arc<dyn ExecutionPlan>, names: &[&str]) -> bool {
    let children = plan.children();
    if children.is_empty() {
        let schema = plan.schema();
        return names.iter().any(|name| schema.index_of(name).is_ok());
    }
    children.into_iter().any(|child| reads_any(child, names))
}
