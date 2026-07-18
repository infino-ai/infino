// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Runtime SIMD dispatch gates for the vector + bloom kernels.
//!
//! Sibling to `distance.rs` because multiple kernels across the
//! codebase want to query the same per-feature gates — `distance::dot`
//! / `distance::l2_sq` (AVX-512F or AVX2), `quant::estimate_dot_rotated`
//! (AVX-512 VPOPCNTDQ), `supertable::manifest::bloom::contains`
//! (AVX-512F + DQ for `vpternlogq` / `kortestz`), and the Sq8
//! cross-product kernel (AVX-512 VPMOVZXBD or AVX2 VPMOVZXBD).
//!
//! Each gate is a `OnceLock<bool>` cached on first call. The cost
//! per call after the first is one relaxed atomic load (~1 ns)
//! and an inlined `&*` deref — negligible next to the kernel work
//! it gates. Initialization consults `diagnostics.disable_avx512` (or
//! `diagnostics.disable_avx2`) first (the config toggles for A/B perf /
//! regression isolation), then runs the appropriate
//! `is_x86_feature_detected!` chain.
//!
//! The gates are sticky once cached, so changing the config after the
//! first call has **no effect**.

use std::sync::OnceLock;

/// True iff this binary should use AVX-512 fast-path kernels.
/// Checks the CPUID baseline that *every* AVX-512 kernel in the
/// codebase relies on: F (foundation), BW (byte/word), DQ
/// (doubleword/quadword), VL (vector length).
///
/// Per-instruction extensions (VPOPCNTDQ) live in their own
/// gates ([`has_vpopcntdq`]) because a kernel that uses only
/// those needs them in addition to F — and there's a small
/// but real population of AVX-512F-only hosts (Knights Landing —
/// not in our fleet but cheap to be correct about) that lack the
/// extensions.
///
/// Set `diagnostics.disable_avx512` to force the AVX2 / wide path on
/// hosts that *do* support AVX-512 — for A/B perf comparison or
/// regression isolation without rebuilding. Consults the config
/// exactly once on the first call.
// Every dispatch site that calls this is itself x86-gated, so on other
// targets the function is unused in the library build (it stays defined
// because the `cfg(test)` gate tests reference it on all targets).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
pub(crate) fn avx512_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if avx512_disabled_by_config() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// True iff the host supports AVX-512 VPOPCNTDQ (per-element 64-bit
/// popcount). Required by `quant::estimate_dot_rotated`'s AVX-512
/// rewrite — its "masked add of query lanes keyed by code bits"
/// path uses `_mm512_mask_add_ps` whose mask comes from a code-byte
/// load, but the throughput-equivalent `popcount` formulation in
/// some shapes also benefits.
///
/// Also implies [`avx512_enabled`] (we never enable a specialized
/// kernel on a host without the foundation), so callers should
/// check this gate alone.
#[cfg(test)]
#[inline]
pub(crate) fn has_vpopcntdq() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !avx512_enabled() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// True iff this binary should use AVX2 fast-path kernels in the
/// "wide" tier. Checks `is_x86_feature_detected!("avx2")` at
/// runtime; near-universally true on production x86_64 hosts (Intel
/// Haswell+ / AMD Excavator+) but not assumed by the build target.
///
/// Sits between [`avx512_enabled`] (the fastest tier — 512-bit) and
/// the portable scalar-widen fallback. Hosts that have AVX-512
/// always also have AVX2, but [`avx512_enabled`] gets checked first
/// at every dispatch site, so the AVX2 gate is only consulted when
/// AVX-512 is off (either no AVX-512 silicon, or
/// `diagnostics.disable_avx512`).
///
/// Set `diagnostics.disable_avx2` to force the portable scalar-widen
/// path on hosts that *do* support AVX2 — for A/B perf comparison
/// or pinning the universal fallback path under test without
/// rebuilding. Consults the config exactly once on the first call.
#[inline]
pub fn avx2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if avx2_disabled_by_config() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Whether `diagnostics.disable_avx512` forces the AVX2 / wide path.
/// Pulled into its own helper so the gates above read a single named
/// source.
// Reached only through the x86-gated `avx512_enabled`, so it is unused
// in a non-x86 library build (its sibling `avx2_disabled_by_config` is
// reached on all targets via the `pub` `avx2_enabled`, hence no allowance).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn avx512_disabled_by_config() -> bool {
    crate::config::global().diagnostics.disable_avx512
}

/// Whether `diagnostics.disable_avx2` forces the scalar-widen path.
#[inline]
fn avx2_disabled_by_config() -> bool {
    crate::config::global().diagnostics.disable_avx2
}

#[cfg(test)]
mod tests {
    /// Per-feature gates must imply the AVX-512 foundation gate —
    /// otherwise a host that lacks F but reports an extension
    /// (impossible in practice, but cheap to be defensive about)
    /// would bypass the foundation check and we'd run an
    /// extension-only kernel that loads `_mm512_*` intrinsics on
    /// an unsupported host. The implication direction in the code
    /// (every per-feature gate returns early when `avx512_enabled()`
    /// is false) is what this test pins by inspection at the
    /// type-system level. Runtime check below confirms the
    /// implication actually holds on whatever host this CI builds
    /// on.
    #[test]
    fn per_feature_gates_imply_avx512_foundation() {
        use super::*;
        if has_vpopcntdq() {
            assert!(
                avx512_enabled(),
                "has_vpopcntdq() returned true but avx512_enabled() is false"
            );
        }
    }
}
