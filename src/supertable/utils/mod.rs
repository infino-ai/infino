// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable-internal helpers with no clear single-concern home.
//!
//! Discipline for what belongs here: cross-cutting helpers used by
//! the supertable layer but with no domain coupling to manifest,
//! query, writer, or reader_cache specifically. Anything with a
//! clear concern gets its own module at `supertable::` instead.

pub mod idgen;
pub mod vector_split;
