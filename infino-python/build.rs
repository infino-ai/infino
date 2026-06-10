// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Make the `extension-module` cdylib link on macOS.
//!
//! A Python extension module must NOT link `libpython`: the `_Py*`
//! symbols are resolved at runtime by the interpreter that loads the
//! `.so`. On macOS that requires `-undefined dynamic_lookup` on the
//! cdylib link. pyo3's `extension-module` feature normally arranges
//! this, but it isn't reliably applied across the maturin / pyo3
//! versions here, so emit it explicitly. `rustc-cdylib-link-arg`
//! affects only the final cdylib link (not the dependency rlibs), and
//! a build script runs regardless of the invocation directory — unlike
//! a `.cargo/config.toml`, which cargo reads from the cwd.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
