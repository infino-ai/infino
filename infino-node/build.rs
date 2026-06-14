// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! napi-rs build setup.
//!
//! `napi_build::setup()` emits the platform link flags a Node native
//! addon needs — including `-undefined dynamic_lookup` on macOS, so the
//! `_napi_*` symbols resolve at load time against the Node binary that
//! loads the `.node` (the addon must NOT link a Node library directly,
//! the same shape as the Python extension's `dynamic_lookup`).

fn main() {
    napi_build::setup();
}
