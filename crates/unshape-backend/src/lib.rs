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
//! - [`BackendNodeExecutor`] - Node executor that routes through backends
//! - [`backend_evaluator`] - Convenience function for common setup
//!
//! # Quick Start
//!
//! ```
//! use unshape_backend::{backend_evaluator, ExecutionPolicy};
//!
//! // Create evaluator with CPU backend and auto policy
//! let mut evaluator = backend_evaluator(ExecutionPolicy::Auto);
//!
//! // Use with graph evaluation:
//! // let result = evaluator.evaluate(&graph, &[output], &ctx)?;
//! ```
//!
//! # Advanced Usage
//!
//! For custom backend registration (e.g., GPU):
//!
//! ```
//! use unshape_backend::{BackendRegistry, BackendNodeExecutor, Scheduler, ExecutionPolicy, LazyEvaluator};
//!
//! let registry = BackendRegistry::with_cpu();
//! // registry.register(Arc::new(gpu_backend));  // Add GPU if available
//! let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
//! let evaluator = LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler));
//! ```
//!
//! See `docs/design/compute-backends.md` for full design documentation.

mod backend;
mod cpu;
mod error;
mod policy;
mod registry;
mod scheduler;

pub use backend::{BackendCapabilities, BackendKind, ComputeBackend, Cost, WorkloadHint};
pub use cpu::CpuBackend;
pub use error::BackendError;
pub use policy::ExecutionPolicy;
pub use registry::BackendRegistry;
pub use scheduler::{BackendEvalResult, BackendNodeExecutor, Scheduler};

// Re-export core types for convenience
pub use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphValue, LazyEvaluator, Value,
};

/// Creates a backend-aware evaluator with CPU backend and the given policy.
///
/// This is a convenience function for the common case. For more control,
/// build the components manually:
///
/// ```
/// use unshape_backend::{BackendRegistry, BackendNodeExecutor, Scheduler, ExecutionPolicy};
/// use unshape_core::LazyEvaluator;
///
/// let registry = BackendRegistry::with_cpu();
/// let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
/// let evaluator = LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler));
/// ```
pub fn backend_evaluator(policy: ExecutionPolicy) -> LazyEvaluator<BackendNodeExecutor> {
    let registry = BackendRegistry::with_cpu();
    let scheduler = Scheduler::new(registry, policy);
    LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler))
}
