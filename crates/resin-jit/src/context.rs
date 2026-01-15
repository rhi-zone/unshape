//! JIT compilation context with helpers.

#[cfg(feature = "cranelift")]
use cranelift::ir::{FuncRef, Type, types};
#[cfg(feature = "cranelift")]
use std::collections::HashMap;

use crate::traits::SimdWidth;

/// Context for JIT compilation.
///
/// Provides helpers for emitting IR and managing external function references.
#[cfg(feature = "cranelift")]
pub struct JitContext {
    /// External function references by name.
    pub(crate) externals: HashMap<String, FuncRef>,
    /// Current SIMD width (if SIMD compilation).
    pub(crate) simd_width: Option<SimdWidth>,
    /// Index for generating unique stateful processor IDs.
    pub(crate) stateful_idx: usize,
}

#[cfg(feature = "cranelift")]
impl JitContext {
    /// Creates a new JIT context.
    pub fn new() -> Self {
        Self {
            externals: HashMap::new(),
            simd_width: None,
            stateful_idx: 0,
        }
    }

    /// Creates a context for SIMD compilation.
    pub fn with_simd(width: SimdWidth) -> Self {
        Self {
            externals: HashMap::new(),
            simd_width: Some(width),
            stateful_idx: 0,
        }
    }

    /// Gets an external function reference by name.
    ///
    /// Returns `None` if the function hasn't been registered.
    pub fn get_external(&self, name: &str) -> Option<FuncRef> {
        self.externals.get(name).copied()
    }

    /// Returns the Cranelift type for the current SIMD width.
    ///
    /// Returns `F32` if not in SIMD mode.
    pub fn value_type(&self) -> Type {
        match self.simd_width {
            Some(SimdWidth::X4) => types::F32X4,
            Some(SimdWidth::X8) => types::F32X8,
            None => types::F32,
        }
    }

    /// Returns the Cranelift SIMD type for the given width.
    pub fn simd_type(&self, width: SimdWidth) -> Type {
        match width {
            SimdWidth::X4 => types::F32X4,
            SimdWidth::X8 => types::F32X8,
        }
    }

    /// Returns whether we're compiling in SIMD mode.
    pub fn is_simd(&self) -> bool {
        self.simd_width.is_some()
    }

    /// Allocates the next stateful processor index.
    pub fn next_stateful_idx(&mut self) -> usize {
        let idx = self.stateful_idx;
        self.stateful_idx += 1;
        idx
    }
}

#[cfg(feature = "cranelift")]
impl Default for JitContext {
    fn default() -> Self {
        Self::new()
    }
}
