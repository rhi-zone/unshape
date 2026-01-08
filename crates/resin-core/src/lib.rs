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
mod node;
pub mod noise;
pub mod particle;
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
pub use node::{BoxedNode, DynNode, PortDescriptor};
pub use noise::{
    fbm_perlin2, fbm_perlin3, fbm_simplex2, fbm_simplex3, fbm2, fbm3, perlin2, perlin2v, perlin3,
    perlin3v, simplex2, simplex2v, simplex3, simplex3v,
};
pub use particle::{
    // Forces
    Attractor,
    // Emitters
    ConeEmitter,
    CurlNoise,
    Drag,
    // Traits
    Emitter,
    Force,
    Gravity,
    // Core types
    Particle,
    ParticleRng,
    ParticleSystem,
    PointEmitter,
    SphereEmitter,
    Turbulence,
    Vortex,
    Wind,
};
pub use resin_macros::DynNode as DynNodeDerive;
pub use value::{Value, ValueType};
