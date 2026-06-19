// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::Supertable;
use crate::config::OptimizeOptions;
use crate::supertable::error::OptimizeError;

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.compact(&opts.compaction).map_err(OptimizeError::from)
    }
}
