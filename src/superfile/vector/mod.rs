// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector subsystem — IVF + 1-bit RaBitQ + full-precision rerank.
//!
//! Layered as: pure-math primitives (`distance`, `quant`,
//! `rotation`, `kmeans`) underneath the `VectorBuilder` /
//! `VectorReader` pair that produces and consumes the multi-column
//! vector blob.
//!
//! See `docs/architecture/superfile.md` for the per-column
//! subsection layout and the IVF + RaBitQ + rerank query pipeline.

pub mod builder;
pub mod distance;
pub mod kmeans;
pub mod quant;
pub mod reader;
pub mod rerank_codec;
pub(crate) mod reservoir;
pub mod rotation;
pub mod simd_dispatch;
pub(crate) mod spill;
pub mod sq8_simd;
