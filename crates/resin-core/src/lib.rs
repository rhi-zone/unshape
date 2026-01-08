//! Core types and traits for resin.
//!
//! This crate provides the foundational types for the resin ecosystem:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//! - [`EvalContext`] - Evaluation context (time, resolution, etc.)
//! - Attribute traits ([`HasPositions`], [`HasNormals`], etc.)

mod attributes;
mod context;
mod error;
pub mod field;
mod graph;
mod node;
pub mod noise;
mod value;

pub use attributes::{
    FullGeometry, Geometry, HasColors, HasIndices, HasNormals, HasPositions, HasUVs,
};
pub use field::{
    Add, Constant, Coordinates, Fbm2D, Fbm3D, Field, FnField, Gradient2D, Map, Mix, Mul,
    Perlin2D, Perlin3D, Radial2D, Scale, Simplex2D, Simplex3D, Translate, from_fn,
};
pub use context::EvalContext;
pub use error::{GraphError, TypeError};
pub use glam;
pub use graph::{Edge, Graph, NodeId};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use resin_macros::DynNode as DynNodeDerive;
pub use noise::{
    fbm2, fbm3, fbm_perlin2, fbm_perlin3, fbm_simplex2, fbm_simplex3,
    perlin2, perlin2v, perlin3, perlin3v,
    simplex2, simplex2v, simplex3, simplex3v,
};
pub use value::{Value, ValueType};
