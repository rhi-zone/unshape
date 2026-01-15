//! Core types and traits for the resin node graph system.
//!
//! This crate provides the foundational types for node graph execution:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//!
//! For geometry attribute traits, see `resin-geometry`.

mod error;
mod graph;
mod node;
mod value;

pub use error::{GraphError, TypeError};
pub use glam;
pub use graph::{Graph, NodeId, Wire};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use rhizome_resin_macros::DynNode as DynNodeDerive;
pub use value::{Value, ValueType};
