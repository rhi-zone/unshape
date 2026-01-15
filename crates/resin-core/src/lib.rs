//! Core types and traits for resin.
//!
//! This crate provides the foundational types for the resin ecosystem:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//! - Attribute traits ([`HasPositions`], [`HasNormals`], etc.)

mod attributes;
mod error;
mod graph;
mod node;
mod value;

pub use attributes::{
    FullGeometry, Geometry, HasColors, HasIndices, HasNormals, HasPositions, HasUVs,
};
pub use error::{GraphError, TypeError};
pub use glam;
pub use graph::{Graph, NodeId, Wire};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use rhizome_resin_macros::DynNode as DynNodeDerive;
pub use value::{Value, ValueType};
