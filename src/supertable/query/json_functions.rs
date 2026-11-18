// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! JSON accessors for text columns.
//!
//! A column that holds a JSON document as text is read field by field with
//! `json_get_str(col, 'key')` and its typed siblings, or with the `->` and
//! `->>` operators. The functions come from the DataFusion JSON extension;
//! this module only registers them on a session.

use datafusion::{error::Result as DfResult, prelude::SessionContext};

/// Register the JSON functions and the `->` / `->>` operator rewrite on `ctx`.
pub(crate) fn register_json_functions(ctx: &SessionContext) -> DfResult<()> {
    // The registry trait takes `&mut`, and a context clone shares its state,
    // so registering on the clone registers on `ctx`.
    let mut registry = ctx.clone();
    datafusion_functions_json::register_all(&mut registry)
}
