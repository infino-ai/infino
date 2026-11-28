// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A column's table-wide value counts, carried from one snapshot of a table
//! to the next.
//!
//! The counts of a table are the sum of its superfiles' own. Summed afresh
//! on every snapshot they cost time in proportion to the table, paid by the
//! first statement after each commit. An append only adds superfiles and a
//! compaction swaps a few for one, so the sum the last snapshot reached is
//! kept, and a new snapshot adds the superfiles that came and takes off the
//! ones that went. The counts are exact, so taking a superfile's counts off
//! leaves exactly the sum of the rest.

use std::{
    collections::HashMap,
    mem,
    sync::{Arc, Mutex},
};

use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use uuid::Uuid;

use crate::supertable::{
    manifest::{SuperfileEntry, list::ScalarValueCounts},
    query::skip::file_lacks_column,
    schema::FieldId,
};

/// The most distinct values a column's table-wide counts hold. Past it the
/// statistics hold no exact answer for the table, and the grouped count runs
/// instead, giving the same rows. It bounds what the carry keeps for a column
/// and what one answer from the statistics returns. It is its own setting:
/// what one superfile keeps is bounded by bytes, by
/// [`VALUE_TABLES_BUDGET_BYTES`](crate::supertable::manifest::list::VALUE_TABLES_BUDGET_BYTES).
pub(crate) const MAX_TABLE_VALUE_COUNTS: usize = 1024;

/// The carried sums of one table, one per column asked for. Shared by every
/// reader of the table, whatever snapshot each pins: a reader on an older
/// snapshot takes the newer superfiles back off.
#[derive(Debug, Default)]
pub(crate) struct ValueCountCarry {
    columns: Mutex<HashMap<FieldId, CarriedColumn>>,
}

/// One column's sum and the superfiles it covers.
#[derive(Debug)]
struct CarriedColumn {
    /// The type the values were counted in. A column the table now reads as
    /// another type starts over.
    data_type: DataType,
    /// The superfiles the sum covers, each with what it added to the sum,
    /// so it can be taken off again when the superfile leaves. The counts
    /// are kept rather than the superfile's entry: an entry reads its
    /// statistics from its part's summaries section, which covers every
    /// superfile of the part as that commit wrote it, so holding the entry
    /// first seen for each superfile would keep one superseded section alive
    /// per commit the table took between reads.
    files: HashMap<Uuid, FileCounts>,
    sum: Sum,
}

#[derive(Debug)]
enum Sum {
    /// Each value's count across the superfiles, at most
    /// [`MAX_TABLE_VALUE_COUNTS`] values.
    Counts(HashMap<ScalarValue, u64>),
    /// No exact answer: the values ran past the limit, or a superfile holding
    /// values recorded no counts. More superfiles keep it so; one leaving
    /// starts the sum over, since nothing can be taken off an unknown sum.
    Unknown,
}

impl Sum {
    /// Adds a superfile's `counts`; an unknown sum stays so.
    fn add(&mut self, counts: &FileCounts) {
        let Sum::Counts(sum) = self else {
            return;
        };
        let known = match counts {
            FileCounts::NoValues => true,
            FileCounts::Unrecorded => false,
            FileCounts::Values(counts) => {
                counts.entries().iter().all(|(value, count)| {
                    let total = sum.entry(value.clone()).or_default();
                    match total.checked_add(*count) {
                        Some(added) => {
                            *total = added;
                            true
                        }
                        None => false,
                    }
                }) && sum.len() <= MAX_TABLE_VALUE_COUNTS
            }
        };
        if !known {
            *self = Sum::Unknown;
        }
    }
}

/// What one superfile adds to a column's sum.
#[derive(Debug, Clone)]
enum FileCounts {
    Values(ScalarValueCounts),
    /// The column holds only nulls there: nothing to count.
    NoValues,
    /// The superfile holds values but recorded no counts of them.
    Unrecorded,
}

/// What `entry` adds to column `column`'s sum.
fn file_counts(entry: &SuperfileEntry, column: FieldId) -> FileCounts {
    let Some(stats) = entry.scalar_stats.column(column) else {
        // A file without the column holds it null in every row.
        return match file_lacks_column(entry, column) {
            true => FileCounts::NoValues,
            false => FileCounts::Unrecorded,
        };
    };
    match stats.value_counts() {
        Some(counts) => FileCounts::Values(counts),
        None if stats.all_null() => FileCounts::NoValues,
        None => FileCounts::Unrecorded,
    }
}

/// `total`, a column's counts over a set of superfiles, without the counts
/// `files` among them recorded: the counts of the rest of the set. `None`
/// when one of `files` holds values it recorded no counts of, or recorded
/// more of a value than `total` holds, which a sum over a set holding it
/// never does. A value whose count reaches zero is left out.
pub(crate) fn counts_without(
    total: &ScalarValueCounts,
    files: &[Arc<SuperfileEntry>],
    column: FieldId,
) -> Option<HashMap<ScalarValue, u64>> {
    let mut left: HashMap<ScalarValue, u64> = total.entries().iter().cloned().collect();
    for entry in files {
        match file_counts(entry, column) {
            FileCounts::NoValues => {}
            FileCounts::Unrecorded => return None,
            FileCounts::Values(counts) => {
                left = take_off(left, counts.entries().iter().map(|(v, n)| (v, *n)))?;
            }
        }
    }
    Some(left)
}

/// `counts` with each of `less` taken off, a value reaching zero left out;
/// `None` when one takes off more than `counts` holds.
pub(crate) fn take_off<'a>(
    mut counts: HashMap<ScalarValue, u64>,
    less: impl Iterator<Item = (&'a ScalarValue, u64)>,
) -> Option<HashMap<ScalarValue, u64>> {
    for (value, count) in less {
        let held = counts.get_mut(value)?;
        *held = held.checked_sub(count)?;
        if *held == 0 {
            counts.remove(value);
        }
    }
    Some(counts)
}

/// Column `column`'s counts over `files`, added up. `None` when one of them
/// holds values it recorded no counts of, or they hold more than
/// [`MAX_TABLE_VALUE_COUNTS`] distinct values together.
pub(crate) fn counts_of(
    files: &[Arc<SuperfileEntry>],
    column: FieldId,
) -> Option<HashMap<ScalarValue, u64>> {
    let mut sum = Sum::Counts(HashMap::new());
    for entry in files {
        sum.add(&file_counts(entry, column));
    }
    match sum {
        Sum::Counts(counts) => Some(counts),
        Sum::Unknown => None,
    }
}

impl ValueCountCarry {
    /// Column `column`'s exact value counts over `files`, read as
    /// `data_type`: the carried sum brought to `files` by adding the
    /// superfiles it lacks and taking off the ones `files` lacks. A
    /// superfile whose column holds only nulls adds nothing, so a column
    /// with no value at all has empty counts. `None` when there is no exact
    /// answer: more than [`MAX_TABLE_VALUE_COUNTS`] values, or a superfile
    /// holding values that recorded no counts.
    pub(crate) fn sum(
        &self,
        column: FieldId,
        data_type: &DataType,
        files: &[Arc<SuperfileEntry>],
    ) -> Option<ScalarValueCounts> {
        let mut columns = self
            .columns
            .lock()
            .expect("value-count carry lock poisoned");
        let carried = columns
            .entry(column)
            .or_insert_with(|| CarriedColumn::empty(data_type));
        if carried.data_type != *data_type {
            *carried = CarriedColumn::empty(data_type);
        }
        let current: HashMap<Uuid, &Arc<SuperfileEntry>> = files
            .iter()
            .map(|entry| (entry.superfile_id, entry))
            .collect();
        let gone: Vec<Uuid> = carried
            .files
            .keys()
            .filter(|suid| !current.contains_key(suid))
            .copied()
            .collect();
        if !gone.is_empty() && !carried.take_off(&gone) {
            *carried = CarriedColumn::empty(data_type);
        }
        for (suid, entry) in current {
            if !carried.files.contains_key(&suid) {
                carried.add(entry, column);
            }
        }
        carried.counts()
    }
}

impl CarriedColumn {
    /// A sum over no superfiles.
    fn empty(data_type: &DataType) -> Self {
        Self {
            data_type: data_type.clone(),
            files: HashMap::new(),
            sum: Sum::Counts(HashMap::new()),
        }
    }

    /// Adds `entry`'s counts of column `column`.
    fn add(&mut self, entry: &SuperfileEntry, column: FieldId) {
        // An unknown sum is never taken from, only started over: keep the
        // superfile's id, not its decoded counts.
        let counts = match self.sum {
            Sum::Unknown => FileCounts::Unrecorded,
            Sum::Counts(_) => file_counts(entry, column),
        };
        self.sum.add(&counts);
        self.files.insert(entry.superfile_id, counts);
    }

    /// Takes the counts of the superfiles `gone` off the sum. `false` when
    /// they can't be: the sum is unknown, or holds less than a superfile
    /// recorded, which a carried sum never does. The sum then starts over.
    fn take_off(&mut self, gone: &[Uuid]) -> bool {
        let Sum::Counts(sum) = &mut self.sum else {
            return false;
        };
        for suid in gone {
            let Some(counts) = self.files.remove(suid) else {
                continue;
            };
            let counts = match counts {
                FileCounts::NoValues => continue,
                FileCounts::Unrecorded => return false,
                FileCounts::Values(counts) => counts,
            };
            let less = counts.entries().iter().map(|(v, n)| (v, *n));
            let Some(left) = take_off(mem::take(sum), less) else {
                return false;
            };
            *sum = left;
        }
        true
    }

    /// The sum as counts, ordered by value; `None` when it is unknown.
    fn counts(&self) -> Option<ScalarValueCounts> {
        let Sum::Counts(sum) = &self.sum else {
            return None;
        };
        if sum.is_empty() {
            return Some(ScalarValueCounts::empty());
        }
        ScalarValueCounts::from_entries(
            sum.iter()
                .map(|(value, count)| (value.clone(), *count))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use arrow_array::{ArrayRef, Int64Array};
    use bytes::Bytes;

    use super::*;
    use crate::{
        superfile::vector::layout::VectorLayout,
        supertable::manifest::{
            ScalarStats, ScalarStatsAgg, SuperfileUri,
            list::FtsSummaryAgg,
            part::PartId,
            summaries::{SummariesSection, rows_from_entries},
        },
        test_helpers::fid,
    };

    /// A superfile whose column `n` holds `values`, its statistics decoded.
    fn superfile(values: Vec<i64>) -> Arc<SuperfileEntry> {
        let column: ArrayRef = Arc::new(Int64Array::from(values));
        let mut stats = HashMap::new();
        stats.insert(
            fid("n"),
            ScalarStatsAgg::from_column(&column).expect("i64 is orderable"),
        );
        Arc::new(SuperfileEntry {
            physical_schema: None,
            stem: None,
            birth_version: 0,
            superfile_id: Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: stats.into(),
            fts_summary: HashMap::<FieldId, FtsSummaryAgg>::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    /// `entries` as one commit hands them to readers: each reading its row
    /// of one summaries section written over all of them, as a part's
    /// entries do. Returns the section too, so a test can watch it.
    fn sectioned(
        entries: &[Arc<SuperfileEntry>],
    ) -> (Vec<Arc<SuperfileEntry>>, Arc<SummariesSection>) {
        let ids: Vec<Uuid> = entries.iter().map(|entry| entry.superfile_id).collect();
        let bytes = SummariesSection::encode(PartId::new_v4(), &ids, rows_from_entries(entries))
            .expect("encode");
        let section = Arc::new(SummariesSection::parse(Bytes::from(bytes)).expect("parse"));
        let read = entries
            .iter()
            .enumerate()
            .map(|(row, entry)| {
                Arc::new(SuperfileEntry {
                    scalar_stats: ScalarStats::Section {
                        section: Arc::clone(&section),
                        row: row as u32,
                    },
                    ..(**entry).clone()
                })
            })
            .collect();
        (read, section)
    }

    fn counts(carry: &ValueCountCarry, files: &[Arc<SuperfileEntry>]) -> Vec<(i64, u64)> {
        let counts = carry
            .sum(fid("n"), &DataType::Int64, files)
            .expect("exact counts");
        let mut out: Vec<(i64, u64)> = counts
            .entries()
            .iter()
            .map(|(value, count)| match value {
                ScalarValue::Int64(Some(v)) => (*v, *count),
                other => panic!("unexpected value {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Each commit rewrites the latest part's summaries section over all of
    /// its superfiles, and hands out fresh entries reading the new one. The
    /// carry must not hold the entries it first saw: each would keep the
    /// section of its own commit alive, so a table read between commits
    /// would hold one superseded section per commit, each as large as the
    /// part was then.
    #[test]
    fn a_carried_sum_keeps_no_superseded_section_alive() {
        let first = superfile(vec![1, 2, 2]);
        let second = superfile(vec![2, 3]);
        let carry = ValueCountCarry::default();

        let (at_first_commit, first_section) = sectioned(&[Arc::clone(&first)]);
        assert_eq!(counts(&carry, &at_first_commit), vec![(1, 1), (2, 2)]);
        let superseded: Weak<SummariesSection> = Arc::downgrade(&first_section);
        drop((at_first_commit, first_section));

        let (at_second_commit, _second_section) =
            sectioned(&[Arc::clone(&first), Arc::clone(&second)]);
        assert_eq!(
            counts(&carry, &at_second_commit),
            vec![(1, 1), (2, 3), (3, 1)]
        );
        assert!(
            superseded.upgrade().is_none(),
            "the carry keeps the first commit's summaries section alive"
        );
    }

    /// Once the sum is unknown, a superfile that comes is recorded by its id
    /// alone: its counts are never read again, so they are not decoded and
    /// kept.
    #[test]
    fn a_superfile_added_to_an_unknown_sum_keeps_no_counts() {
        let past_the_limit = superfile((0..=MAX_TABLE_VALUE_COUNTS as i64).collect());
        let later = superfile(vec![1, 2]);
        let carry = ValueCountCarry::default();

        assert!(
            carry
                .sum(fid("n"), &DataType::Int64, &[Arc::clone(&past_the_limit)])
                .is_none()
        );
        assert!(
            carry
                .sum(
                    fid("n"),
                    &DataType::Int64,
                    &[Arc::clone(&past_the_limit), Arc::clone(&later)]
                )
                .is_none()
        );
        let columns = carry.columns.lock().expect("lock");
        let kept = &columns[&fid("n")].files[&later.superfile_id];
        assert!(
            matches!(kept, FileCounts::Unrecorded),
            "kept {kept:?} for a superfile added to an unknown sum"
        );
    }

    /// A superfile that leaves is taken off with the counts it added, kept
    /// without its entry.
    #[test]
    fn a_superfile_that_leaves_is_taken_off_the_carried_sum() {
        let first = superfile(vec![1, 2, 2]);
        let second = superfile(vec![2, 3]);
        let carry = ValueCountCarry::default();

        let (both, _section) = sectioned(&[Arc::clone(&first), Arc::clone(&second)]);
        assert_eq!(counts(&carry, &both), vec![(1, 1), (2, 3), (3, 1)]);
        let (after_compaction, _section) = sectioned(&[Arc::clone(&second)]);
        assert_eq!(counts(&carry, &after_compaction), vec![(2, 1), (3, 1)]);
    }
}
