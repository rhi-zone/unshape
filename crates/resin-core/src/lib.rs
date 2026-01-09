//! Core types and traits for resin.
//!
//! This crate provides the foundational types for the resin ecosystem:
//!
//! - [`Graph`] - Node graph container and execution engine
//! - [`DynNode`] - Trait for dynamic node execution
//! - [`Value`] - Runtime value type for graph data
//! - [`EvalContext`] - Evaluation context (time, resolution, etc.)
//! - Attribute traits ([`HasPositions`], [`HasNormals`], etc.)
//! - [`expr::Expr`] - Expression language for field evaluation

mod attributes;
mod context;
mod error;
pub mod expr;
pub mod field;
mod graph;
pub mod image_field;
mod node;
mod value;

pub use attributes::{
    FullGeometry, Geometry, HasColors, HasIndices, HasNormals, HasPositions, HasUVs,
};
pub use context::EvalContext;
pub use error::{GraphError, TypeError};
pub use field::{
    // Combinators
    Add,
    // Domain modifiers
    Bend,
    // Patterns
    Brick,
    Checkerboard,
    // Basic fields
    Constant,
    Coordinates,
    // Warping
    Displacement,
    // SDF primitives
    DistanceBox,
    DistanceCircle,
    DistanceLine,
    DistancePoint,
    Dots,
    // Noise
    Fbm2D,
    Fbm3D,
    // Trait
    Field,
    FnField,
    // Gradients
    Gradient2D,
    Map,
    // Metaballs
    Metaball,
    MetaballSdf2D,
    MetaballSdf3D,
    Metaballs2D,
    Metaballs3D,
    Mirror,
    Mix,
    Mul,
    Perlin2D,
    Perlin3D,
    Radial2D,
    Repeat,
    Repeat3D,
    Rotate2D,
    Scale,
    // SDF operations
    SdfAnnular,
    SdfIntersection,
    SdfRound,
    SdfSmoothIntersection,
    SdfSmoothSubtraction,
    SdfSmoothUnion,
    SdfSubtraction,
    SdfUnion,
    Simplex2D,
    Simplex3D,
    SmoothDots,
    SmoothStripes,
    Stripes,
    Translate,
    Twist,
    Voronoi,
    VoronoiId,
    Warp,
    from_fn,
};
pub use glam;
pub use graph::{Edge, Graph, NodeId};
pub use image_field::{FilterMode, ImageField, ImageFieldError, WrapMode};
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use resin_macros::DynNode as DynNodeDerive;
pub use value::{Value, ValueType};
