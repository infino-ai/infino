// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Manual table maintenance entry point.
//!
//! [`Supertable::optimize`] (and [`crate::Supertable::compact`]) are **never**
//! invoked automatically on commit or in background threads. Operators call
//! them explicitly when they want size-based user-table merge and/or hidden
//! hot-region overlap consolidation (see [`crate::supertable::opann::maintenance`]).

use crate::{Supertable, config::OptimizeOptions, supertable::error::OptimizeError};

impl Supertable {
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.compact(&opts.compaction).map_err(OptimizeError::from)
    }
}
