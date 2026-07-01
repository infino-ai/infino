// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::{
    Supertable,
    config::OptimizeOptions,
    supertable::error::OptimizeError,
};

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.drain_hidden_vector_cells_sync()
            .map_err(|e| OptimizeError::Build(e.to_string()))?;
        self.compact(&opts.compaction).map_err(OptimizeError::from)
    }
}
