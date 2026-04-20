//! Field trait for lazy evaluation.
//!
//! A `Field<I, O>` represents a function that can be sampled at any point.
//! Fields are lazy - they describe computation, not data. Evaluation happens
//! on demand when you call `sample()`.
//!
//! # Examples
//!
//! ```
//! use unshape_field::{Field, EvalContext, Perlin2D};
//! use glam::Vec2;
//!
//! // Create a noise field
//! let noise = Perlin2D::new().scale(4.0);
//!
//! // Sample at a point
//! let ctx = EvalContext::new();
//! let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
//! ```

mod erosion;
mod fbm;
mod metaball;
mod network;
mod noise;
mod pattern;
mod sdf;
mod spectral;
mod terrain;

pub use erosion::*;
pub use fbm::*;
pub use metaball::*;
pub use network::*;
pub use noise::*;
pub use pattern::*;
pub use sdf::*;
pub use spectral::*;
pub use terrain::*;
pub use unshape_field_ops::{
    Abs, Clamp, Constant, Coordinates, EvalContext, Field, FnField, Map, Negate, Pow, Remap, Scale,
    Smoothstep, Step, Translate, Zip, Zip3, add, div, from_fn, lerp, mix, mul, sub, zip, zip3,
};

/// Registers all field operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of field ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<HydraulicErosion>("resin::HydraulicErosion");
    registry.register_type::<ThermalErosion>("resin::ThermalErosion");
    registry.register_type::<RoadNetwork>("resin::RoadNetwork");
    registry.register_type::<RiverNetwork>("resin::RiverNetwork");
}

#[cfg(test)]
mod tests;

/// Invariant tests for field properties.
///
/// These tests verify mathematical and statistical properties that should hold
/// for all field implementations. Run with:
///
/// ```sh
/// cargo test -p unshape-field --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests;
