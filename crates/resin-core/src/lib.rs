//! Core types and traits for resin.
//!
//! This crate provides the foundational types for the resin ecosystem:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//! - [`EvalContext`] - Evaluation context (time, resolution, etc.)

mod context;
mod error;
mod graph;
mod node;
mod value;

pub use context::EvalContext;
pub use error::{GraphError, TypeError};
pub use glam;
pub use graph::{Edge, Graph, NodeId};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use value::{Value, ValueType};
