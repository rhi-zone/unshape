//! Compute backend abstraction for resin.
//!
//! This crate provides the infrastructure for heterogeneous execution —
//! running graph nodes on different backends (CPU, GPU, SIMD, etc.)
//! based on capabilities and policies.
//!
//! # Core Types
//!
//! - [`ComputeBackend`] - Trait for execution backends
//! - [`BackendRegistry`] - Collection of available backends
//! - [`ExecutionPolicy`] - How to choose backends
//! - [`CpuBackend`] - Default CPU backend (always available)
//!
//! # Example
//!
//! ```
//! use rhizome_resin_backend::{BackendRegistry, CpuBackend, ExecutionPolicy};
//! use std::sync::Arc;
//!
//! // Create registry with CPU backend
//! let mut registry = BackendRegistry::new();
//! registry.register(Arc::new(CpuBackend));
//!
//! // GPU backends can be registered when available
//! // if let Ok(gpu) = GpuBackend::new() {
//! //     registry.register(Arc::new(gpu));
//! // }
//!
//! // Policy determines backend selection
//! let policy = ExecutionPolicy::Auto;
//! ```
//!
//! See `docs/design/compute-backends.md` for full design documentation.

mod backend;
mod cpu;
mod error;
mod policy;
mod registry;

pub use backend::{BackendCapabilities, BackendKind, ComputeBackend, Cost, WorkloadHint};
pub use cpu::CpuBackend;
pub use error::BackendError;
pub use policy::ExecutionPolicy;
pub use registry::BackendRegistry;

// Re-export core types for convenience
pub use rhizome_resin_core::{DataLocation, DynNode, EvalContext, GraphValue, Value};
