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
pub mod lsystem;
mod node;
pub mod particle;
pub mod scatter;
pub mod space_colonization;
pub mod spline;
pub mod spring;
pub mod surface;
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
pub use lsystem::{
    LSystem, Rule, TurtleConfig, TurtleSegment2D, TurtleSegment3D, TurtleState2D, TurtleState3D,
    interpret_turtle_2d, interpret_turtle_3d, presets as lsystem_presets, segments_to_paths_2d,
};
pub use node::{BoxedNode, DynNode, PortDescriptor};
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
pub use scatter::{
    Instance, ScatterConfig, jitter_positions, randomize_rotation, randomize_scale, scatter_circle,
    scatter_grid, scatter_grid_2d, scatter_line, scatter_poisson_2d, scatter_random,
    scatter_random_with_config, scatter_sphere,
};
pub use space_colonization::{
    BranchEdge, BranchNode, SpaceColonization, SpaceColonizationConfig, generate_lightning,
    generate_tree,
};
pub use spline::{
    BSpline, BezierSpline, CatmullRom, CubicBezier, Interpolatable, Nurbs, WeightedPoint,
    cubic_bezier, nurbs_arc, nurbs_circle, nurbs_circle_2d, nurbs_ellipse, quadratic_bezier,
    smooth_through_points,
};
pub use spring::{
    DistanceConstraint, Particle as SpringParticle, Spring, SpringConfig, SpringSystem,
    VerletParticle, create_cloth, create_rope, create_soft_sphere, solve_distance_constraint,
};
pub use surface::{
    NurbsSurface, SurfacePoint, TessellatedSurface, nurbs_bilinear_patch, nurbs_cone,
    nurbs_cylinder, nurbs_sphere, nurbs_torus,
};
pub use value::{Value, ValueType};
