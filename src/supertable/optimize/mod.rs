// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::{
    Supertable,
    config::OptimizeOptions,
    supertable::error::OptimizeError,
};

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        if let Some(hidden) = self.inner().vector_index_table.as_ref() {
            hidden
                .drain()
                .map_err(|e| OptimizeError::Build(e.to_string()))?;
            hidden
                .compact(&hidden.inner().options.compaction)
                .map_err(OptimizeError::from)?;
        }
        self.compact(&opts.compaction).map_err(OptimizeError::from)
    }
}
