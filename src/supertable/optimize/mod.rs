// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::{
    Supertable,
    config::{DEFAULT_GC_SAFETY_GAP, OptimizeOptions},
    supertable::error::{GcError, OptimizeError},
};

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.compact(&opts.compaction)?;
        match self.gc(DEFAULT_GC_SAFETY_GAP) {
            Ok(_) | Err(GcError::NoStorage) => {}
            Err(e) => return Err(OptimizeError::Gc(e)),
        }
        Ok(())
    }
}
