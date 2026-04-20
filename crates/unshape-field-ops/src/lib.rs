//! Field trait, combinators, and primitives for procedural graphics.
//!
//! This crate provides the core [`Field<I, O>`] trait and [`EvalContext`] for lazy
//! spatial evaluation, plus combinator types (`Map`, `Zip`, `Zip3`, `Scale`, `Translate`,
//! `Remap`, `Clamp`, `Smoothstep`, `Step`, `Abs`, `Negate`, `Pow`) and basic primitives
//! (`Constant`, `Coordinates`, `FnField`).

mod combinators;
mod context;
mod output_ops;
mod primitives;

pub use combinators::{Map, Scale, Translate, Zip, Zip3, add, div, lerp, mix, mul, sub, zip, zip3};
pub use context::EvalContext;
pub use output_ops::{Abs, Clamp, Negate, Pow, Remap, Smoothstep, Step};
pub use primitives::{Constant, Coordinates, FnField, from_fn};

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
    /// use unshape_field_ops::{Field, EvalContext, Constant, Zip};
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

    /// Remaps the field output from `[in_min, in_max]` to `[out_min, out_max]`,
    /// clamping to the output range. Only available for `f32`-output fields.
    fn remap(self, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> Remap<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Remap {
            field: self,
            in_min,
            in_max,
            out_min,
            out_max,
        }
    }

    /// Clamps the field output to `[min, max]`.
    /// Only available for `f32`-output fields.
    fn clamp(self, min: f32, max: f32) -> Clamp<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Clamp {
            field: self,
            min,
            max,
        }
    }

    /// Applies smoothstep mapping from `[edge0, edge1]` to `[0, 1]`.
    /// Only available for `f32`-output fields.
    fn smoothstep(self, edge0: f32, edge1: f32) -> Smoothstep<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Smoothstep {
            field: self,
            edge0,
            edge1,
        }
    }

    /// Returns 0.0 if field output < threshold, else 1.0.
    /// Only available for `f32`-output fields.
    fn step(self, threshold: f32) -> Step<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Step {
            field: self,
            threshold,
        }
    }

    /// Returns the absolute value of the field output.
    /// Only available for `f32`-output fields.
    fn abs(self) -> Abs<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Abs { field: self }
    }

    /// Negates the field output.
    /// Only available for `f32`-output fields.
    fn negate(self) -> Negate<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Negate { field: self }
    }

    /// Raises the field output to a power.
    /// Only available for `f32`-output fields.
    fn pow(self, exponent: f32) -> Pow<Self>
    where
        Self: Sized + Field<I, f32>,
    {
        Pow {
            field: self,
            exponent,
        }
    }
}
