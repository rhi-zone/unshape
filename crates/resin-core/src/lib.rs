//! Core types and traits for the resin node graph system.
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
//! For geometry attribute traits, see `resin-geometry`.

mod error;
mod eval;
mod graph;
mod node;
mod value;

pub use error::{GraphError, TypeError};
pub use eval::{
    CacheEntry, CacheKey, CachePolicy, CancellationMode, CancellationToken, ErrorHandling,
    EvalCache, EvalContext, EvalProgress, EvalResult, Evaluator, FeedbackState, KeepAllPolicy,
    LazyEvaluator,
};
pub use glam;
pub use graph::{Graph, NodeId, Wire};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use rhizome_resin_macros::DynNode as DynNodeDerive;
pub use value::{Value, ValueType};
