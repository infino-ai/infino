// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::{
    Supertable,
    config::{DEFAULT_GC_SAFETY_GAP, OptimizeOptions},
    supertable::error::OptimizeError,
};

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.compact(&opts.compaction)?;
        self.gc(DEFAULT_GC_SAFETY_GAP)?;
        Ok(())
    }
}
