// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! JSON accessors for text columns.
//!
//! A column that holds a JSON document as text is read field by field:
//! `json_get_str(col, 'key')` and its typed siblings take the column and a
//! path of keys or array positions, and read as null where the path is
//! missing or the text is not JSON. The functions come from the DataFusion
//! JSON extension; this module chooses which of them a session offers.
//!
//! The list is spelled out rather than taken whole from the extension, so a
//! version bump cannot add a function to the SQL surface on its own.
//!
//! The `->` and `->>` operators are left out on purpose. In the dialect SQL
//! is parsed with they bind looser than a comparison, so `col->>'k' = 'v'`
//! reads as `col->>('k' = 'v')` and fails at planning with a type error that
//! says nothing about precedence. `json_get_str(col, 'k') = 'v'` has no such
//! trap.

use datafusion::prelude::SessionContext;
use datafusion_functions_json::udfs::{
    json_contains_udf, json_get_array_udf, json_get_bool_udf, json_get_float_udf, json_get_int_udf,
    json_get_json_udf, json_get_str_udf, json_length_udf, json_object_keys_udf,
};

/// Register the JSON functions a session offers on `ctx`.
pub(crate) fn register_json_functions(ctx: &SessionContext) {
    for udf in [
        json_get_str_udf(),
        json_get_int_udf(),
        json_get_float_udf(),
        json_get_bool_udf(),
        json_get_json_udf(),
        json_get_array_udf(),
        json_contains_udf(),
        json_length_udf(),
        json_object_keys_udf(),
    ] {
        ctx.register_udf(udf.as_ref().clone());
    }
}
