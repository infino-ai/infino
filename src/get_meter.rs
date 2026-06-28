// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Optional GET-phase tags for bench / diagnostic wave attribution.
//!
//! A **wave** is one parallel batch of range GETs (`try_join_all`). Bench
//! storage wrappers record the current phase when each GET completes.

use std::sync::atomic::{AtomicU8, Ordering};

/// No phase tag — ordinary storage traffic.
pub const GET_PHASE_NONE: u8 = 0;
/// OPANN routing-tree page loads (one wave per tree level).
pub const GET_PHASE_OPANN: u8 = 1;
/// Vector subsection open-time speculation GETs (`vec_open_ranges`).
pub const GET_PHASE_VEC_OPEN: u8 = 2;
/// Direct leaf / cluster probe GETs after OPANN descent.
pub const GET_PHASE_LEAF_FETCH: u8 = 3;

static CURRENT: AtomicU8 = AtomicU8::new(GET_PHASE_NONE);

/// Set the phase tag applied to subsequent storage GETs until reset.
pub fn set_get_phase(phase: u8) {
    CURRENT.store(phase, Ordering::Relaxed);
}

/// Current GET phase tag.
pub fn get_phase() -> u8 {
    CURRENT.load(Ordering::Relaxed)
}

/// Human-readable label for a phase id (bench / diag output).
pub fn phase_label(phase: u8) -> &'static str {
    match phase {
        GET_PHASE_OPANN => "opann",
        GET_PHASE_VEC_OPEN => "vec_open",
        GET_PHASE_LEAF_FETCH => "leaf_fetch",
        _ => "other",
    }
}

/// Restores the previous phase on drop — safe for nested probe scopes.
pub struct GetPhaseGuard(u8);

impl GetPhaseGuard {
    pub fn new(phase: u8) -> Self {
        let prev = get_phase();
        set_get_phase(phase);
        Self(prev)
    }
}

impl Drop for GetPhaseGuard {
    fn drop(&mut self) {
        set_get_phase(self.0);
    }
}
