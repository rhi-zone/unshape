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

mod combinators;
mod context;
mod erosion;
mod fbm;
mod metaball;
mod network;
mod noise;
mod pattern;
mod primitives;
mod sdf;
mod spectral;
mod terrain;

pub use combinators::{Map, Scale, Translate, Zip, Zip3, add, div, lerp, mix, mul, sub, zip, zip3};
pub use context::EvalContext;
pub use erosion::*;
pub use fbm::*;
pub use metaball::*;
pub use network::*;
pub use noise::*;
pub use pattern::*;
pub use primitives::{Constant, Coordinates, FnField, from_fn};
pub use sdf::*;
pub use spectral::*;
pub use terrain::*;

use std::marker::PhantomData;

/// A field that can be sampled at any point.
///
/// Fields are the core abstraction for lazy, spatial computation.
/// They represent functions from input coordinates to output values.
pub trait Field<I, O> {
    /// Samples the field at a given input coordinate.
    fn sample(&self, input: I, ctx: &EvalContext) -> O;

    /// Transforms the output of this field.
    fn map<O2, F>(self, f: F) -> Map<Self, F, O>
    where
        Self: Sized,
        F: Fn(O) -> O2,
    {
        Map {
            field: self,
            f,
            _phantom: PhantomData,
        }
    }

    /// Scales the input coordinates.
    fn scale(self, factor: f32) -> Scale<Self>
    where
        Self: Sized,
    {
        Scale {
            field: self,
            factor,
        }
    }

    /// Translates the input coordinates.
    fn translate(self, offset: I) -> Translate<Self, I>
    where
        Self: Sized,
        I: Clone,
    {
        Translate {
            field: self,
            offset,
        }
    }

    /// Zips this field with another, yielding a tuple of their outputs.
    ///
    /// This is a fundamental combinator - addition, multiplication, mixing
    /// can all be expressed as `zip().map(...)`. See also the `add()`, `mul()`,
    /// `mix()` helper functions.
    ///
    /// # Example
    /// ```
    /// use unshape_field::{Field, EvalContext, Constant, Zip};
    ///
    /// let a = Constant::new(1.0_f32);
    /// let b = Constant::new(2.0_f32);
    /// let zipped = Zip::new(a, b);
    ///
    /// let ctx = EvalContext::new();
    /// let (va, vb): (f32, f32) = Field::<f32, _>::sample(&zipped, 0.0, &ctx);
    /// assert_eq!(va, 1.0);
    /// assert_eq!(vb, 2.0);
    /// ```
    fn zip<F2, O2>(self, other: F2) -> Zip<Self, F2>
    where
        Self: Sized,
        F2: Field<I, O2>,
    {
        Zip { a: self, b: other }
    }

    /// Zips this field with two others, yielding a triple of their outputs.
    ///
    /// Useful for operations like lerp: `a.zip3(b, t).map(|(a, b, t)| a * (1.0 - t) + b * t)`
    fn zip3<F2, O2, F3, O3>(self, second: F2, third: F3) -> Zip3<Self, F2, F3>
    where
        Self: Sized,
        F2: Field<I, O2>,
        F3: Field<I, O3>,
    {
        Zip3 {
            a: self,
            b: second,
            c: third,
        }
    }
}

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
