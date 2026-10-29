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
    sync::{Arc, Mutex},
};

use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use uuid::Uuid;

use crate::supertable::{
    manifest::{
        SuperfileEntry,
        list::{MAX_EXACT_VALUE_COUNTS, ScalarValueCounts},
    },
    schema::FieldId,
};

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
    /// The superfiles the sum covers.
    files: HashMap<Uuid, Arc<SuperfileEntry>>,
    sum: Sum,
}

#[derive(Debug)]
enum Sum {
    /// Each value's count across the superfiles, at most
    /// [`MAX_EXACT_VALUE_COUNTS`] values.
    Counts(HashMap<ScalarValue, u64>),
    /// No exact answer: the values ran past the cap, or a superfile holding
    /// values recorded no counts. More superfiles keep it so; one leaving
    /// starts the sum over, since nothing can be taken off an unknown sum.
    Unknown,
}

/// What one superfile adds to a column's sum.
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
        return FileCounts::Unrecorded;
    };
    match stats.value_counts() {
        Some(counts) => FileCounts::Values(counts),
        None if stats.all_null() => FileCounts::NoValues,
        None => FileCounts::Unrecorded,
    }
}

impl ValueCountCarry {
    /// Column `column`'s exact value counts over `files`, read as
    /// `data_type`: the carried sum brought to `files` by adding the
    /// superfiles it lacks and taking off the ones `files` lacks. A
    /// superfile whose column holds only nulls adds nothing, so a column
    /// with no value at all has empty counts. `None` when there is no exact
    /// answer: more than [`MAX_EXACT_VALUE_COUNTS`] values, or a superfile
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
        if !gone.is_empty() && !carried.take_off(&gone, column) {
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

    /// Adds `entry`'s counts.
    fn add(&mut self, entry: &Arc<SuperfileEntry>, column: FieldId) {
        self.files.insert(entry.superfile_id, Arc::clone(entry));
        let Sum::Counts(sum) = &mut self.sum else {
            return;
        };
        let known = match file_counts(entry, column) {
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
                }) && sum.len() <= MAX_EXACT_VALUE_COUNTS
            }
        };
        if !known {
            self.sum = Sum::Unknown;
        }
    }

    /// Takes the counts of the superfiles `gone` off the sum. `false` when
    /// they can't be: the sum is unknown, or holds less than a superfile
    /// recorded, which a carried sum never does. The sum then starts over.
    fn take_off(&mut self, gone: &[Uuid], column: FieldId) -> bool {
        let Sum::Counts(sum) = &mut self.sum else {
            return false;
        };
        for suid in gone {
            let Some(entry) = self.files.remove(suid) else {
                continue;
            };
            let counts = match file_counts(&entry, column) {
                FileCounts::NoValues => continue,
                FileCounts::Unrecorded => return false,
                FileCounts::Values(counts) => counts,
            };
            for (value, count) in counts.entries() {
                let Some(total) = sum.get_mut(value) else {
                    return false;
                };
                let Some(left) = total.checked_sub(*count) else {
                    return false;
                };
                if left == 0 {
                    sum.remove(value);
                } else {
                    *total = left;
                }
            }
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
