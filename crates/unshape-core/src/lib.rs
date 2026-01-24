//! Core types and traits for the unshape node graph system.
//!
//! This crate provides the foundational types for node graph execution:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//! - [`EvalContext`] - Evaluation context (time, cancellation, quality hints)
//! - [`Evaluator`] - Trait for evaluation strategies
//! - [`LazyEvaluator`] - Lazy evaluator with caching
//!
//! # Quick Example
//!
//! ```
//! use unshape_core::{Graph, EvalContext};
//!
//! // Create a graph and add nodes
//! let mut graph = Graph::new();
//!
//! // Execute with default context
//! // let result = graph.execute(output_node).unwrap();
//!
//! // Or with custom context for animation, cancellation, etc.
//! let ctx = EvalContext::new()
//!     .with_time(1.0, 60, 1.0/60.0)
//!     .with_seed(42);
//! ```
//!
//! # Evaluation Strategies
//!
//! Two evaluation strategies are available:
//!
//! - **Eager** ([`Graph::execute`]): Computes all nodes in topological order
//! - **Lazy** ([`LazyEvaluator`]): Only computes nodes needed for requested outputs,
//!   with memoization/caching
//!
//! See the [`eval`] module for details on caching, cancellation, and progress reporting.
//!
//! For geometry attribute traits, see `unshape-geometry`.

mod error;
mod eval;
mod graph;
mod node;
pub mod optimize;
mod value;

pub use error::{GraphError, TypeError};
pub use eval::{
    CacheEntry, CacheKey, CachePolicy, CancellationMode, CancellationToken, DefaultNodeExecutor,
    ErrorHandling, EvalCache, EvalContext, EvalProgress, EvalResult, Evaluator, FeedbackState,
    KeepAllPolicy, LazyEvaluator, NodeExecutor,
};
pub use glam;
pub use graph::{Graph, NodeId, Wire};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use unshape_macros::DynNode as DynNodeDerive;
pub use value::{DataLocation, GraphValue, Value, ValueType};
