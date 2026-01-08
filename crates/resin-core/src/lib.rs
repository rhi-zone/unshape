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
pub mod color;
mod context;
pub mod easing;
mod error;
pub mod expr;
pub mod field;
mod graph;
mod node;
pub mod noise;
pub mod particle;
pub mod spline;
pub mod surface;
mod value;

pub use attributes::{
    FullGeometry, Geometry, HasColors, HasIndices, HasNormals, HasPositions, HasUVs,
};
pub use color::{
    BlendMode, ColorStop, Gradient, Hsl, Hsv, LinearRgb, Rgba, blend, blend_with_alpha,
    presets as color_presets,
};
pub use context::EvalContext;
pub use easing::{
    Easing, back_in, back_in_out, back_out, bounce_in, bounce_in_out, bounce_out, circ_in,
    circ_in_out, circ_out, cubic_in, cubic_in_out, cubic_out, ease_between, ease_value, elastic_in,
    elastic_in_out, elastic_out, expo_in, expo_in_out, expo_out, linear, quad_in, quad_in_out,
    quad_out, quart_in, quart_in_out, quart_out, quint_in, quint_in_out, quint_out, sine_in,
    sine_in_out, sine_out, smootherstep, smoothstep, stepped,
};
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
pub use spline::{
    BSpline, BezierSpline, CatmullRom, CubicBezier, Interpolatable, Nurbs, WeightedPoint,
    cubic_bezier, nurbs_arc, nurbs_circle, nurbs_circle_2d, nurbs_ellipse, quadratic_bezier,
    smooth_through_points,
};
pub use surface::{
    NurbsSurface, SurfacePoint, TessellatedSurface, nurbs_bilinear_patch, nurbs_cone,
    nurbs_cylinder, nurbs_sphere, nurbs_torus,
};
pub use value::{Value, ValueType};
