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

mod context;
mod erosion;
mod fbm;
mod metaball;
mod network;
mod noise;
mod pattern;
mod sdf;
mod spectral;
mod terrain;

use glam::{Vec2, Vec3};
use std::marker::PhantomData;
use unshape_easing::Lerp;

pub use context::EvalContext;
pub use erosion::*;
pub use fbm::*;
pub use metaball::*;
pub use network::*;
pub use noise::*;
pub use pattern::*;
pub use sdf::*;
pub use spectral::*;
pub use terrain::*;

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

// Combinators

/// Maps the output of a field.
pub struct Map<F, M, O> {
    /// The inner field to transform.
    pub field: F,
    /// The mapping function.
    pub f: M,
    /// Phantom data for the original output type.
    pub _phantom: PhantomData<O>,
}

impl<I, O, O2, F, M> Field<I, O2> for Map<F, M, O>
where
    F: Field<I, O>,
    M: Fn(O) -> O2,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> O2 {
        (self.f)(self.field.sample(input, ctx))
    }
}

/// Zips two fields, evaluating both at the same input.
///
/// This is a fundamental primitive - binary operations like addition and
/// multiplication can be expressed as `Zip + Map`. See the `add()` and `mul()`
/// helper functions.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, Zip, Map};
/// use std::marker::PhantomData;
///
/// let a = Constant::new(3.0_f32);
/// let b = Constant::new(4.0_f32);
///
/// // Manual addition via zip + map
/// let zipped = Zip::new(a, b);
/// let sum = Map { field: zipped, f: |(x, y): (f32, f32)| x + y, _phantom: PhantomData };
///
/// let ctx = EvalContext::new();
/// assert_eq!(Field::<f32, f32>::sample(&sum, 0.0, &ctx), 7.0);
/// ```
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Zip<A, B> {
    /// Creates a new zip combinator.
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I, A, B, OA, OB> Field<I, (OA, OB)> for Zip<A, B>
where
    I: Clone,
    A: Field<I, OA>,
    B: Field<I, OB>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> (OA, OB) {
        let va = self.a.sample(input.clone(), ctx);
        let vb = self.b.sample(input, ctx);
        (va, vb)
    }
}

/// Zips three fields, evaluating all at the same input.
///
/// This is a fundamental primitive - ternary operations like lerp/mix
/// can be expressed as `Zip3 + Map`. See the `lerp()` and `mix()` helper
/// functions.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, Zip3, Map};
/// use std::marker::PhantomData;
///
/// let a = Constant::new(0.0_f32);
/// let b = Constant::new(10.0_f32);
/// let t = Constant::new(0.5_f32);
///
/// // Manual lerp via zip3 + map
/// let zipped = Zip3::new(a, b, t);
/// let lerp = Map { field: zipped, f: |(a, b, t): (f32, f32, f32)| a * (1.0 - t) + b * t, _phantom: PhantomData };
///
/// let ctx = EvalContext::new();
/// assert_eq!(Field::<f32, f32>::sample(&lerp, 0.0, &ctx), 5.0);
/// ```
pub struct Zip3<A, B, C> {
    a: A,
    b: B,
    c: C,
}

impl<A, B, C> Zip3<A, B, C> {
    /// Creates a new zip3 combinator.
    pub fn new(a: A, b: B, c: C) -> Self {
        Self { a, b, c }
    }
}

impl<I, A, B, C, OA, OB, OC> Field<I, (OA, OB, OC)> for Zip3<A, B, C>
where
    I: Clone,
    A: Field<I, OA>,
    B: Field<I, OB>,
    C: Field<I, OC>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> (OA, OB, OC) {
        let va = self.a.sample(input.clone(), ctx);
        let vb = self.b.sample(input.clone(), ctx);
        let vc = self.c.sample(input, ctx);
        (va, vb, vc)
    }
}

/// Scales the input coordinates of a field.
pub struct Scale<F> {
    field: F,
    factor: f32,
}

impl<O, F> Field<Vec2, O> for Scale<F>
where
    F: Field<Vec2, O>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

impl<O, F> Field<Vec3, O> for Scale<F>
where
    F: Field<Vec3, O>,
{
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

impl<O, F> Field<f32, O> for Scale<F>
where
    F: Field<f32, O>,
{
    fn sample(&self, input: f32, ctx: &EvalContext) -> O {
        self.field.sample(input * self.factor, ctx)
    }
}

/// Translates the input coordinates of a field.
pub struct Translate<F, I> {
    field: F,
    offset: I,
}

impl<O, F> Field<Vec2, O> for Translate<F, Vec2>
where
    F: Field<Vec2, O>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

impl<O, F> Field<Vec3, O> for Translate<F, Vec3>
where
    F: Field<Vec3, O>,
{
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

impl<O, F> Field<f32, O> for Translate<F, f32>
where
    F: Field<f32, O>,
{
    fn sample(&self, input: f32, ctx: &EvalContext) -> O {
        self.field.sample(input - self.offset, ctx)
    }
}

// ============================================================================
// Ergonomic Helper Functions (Layer 2)
// ============================================================================
//
// These functions provide convenient APIs that expand to Zip/Map compositions.
// They're not primitives - just sugar over the true primitives.

/// Zips two fields together.
///
/// Standalone function version of `Field::zip()`.
pub fn zip<A, B>(a: A, b: B) -> Zip<A, B> {
    Zip::new(a, b)
}

/// Zips three fields together.
///
/// Standalone function version of `Field::zip3()`.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> Zip3<A, B, C> {
    Zip3::new(a, b, c)
}

/// Linearly interpolates between two fields.
///
/// This is an ergonomic helper that expands to `Zip3 + Map`.
/// Works with any type that implements `Lerp` (f32, Vec2, Vec3, Rgba, etc.).
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, lerp};
/// use glam::Vec2;
///
/// let a = Constant::new(0.0_f32);
/// let b = Constant::new(10.0_f32);
/// let t = Constant::new(0.25_f32);
///
/// let result = lerp::<Vec2, _, _, _, _>(a, b, t);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 2.5);
/// ```
pub fn lerp<I, O, A, B, T>(
    a: A,
    b: B,
    t: T,
) -> Map<Zip3<A, B, T>, impl Fn((O, O, f32)) -> O, (O, O, f32)>
where
    I: Clone,
    O: Lerp,
    A: Field<I, O>,
    B: Field<I, O>,
    T: Field<I, f32>,
{
    Zip3::new(a, b, t).map(|(a, b, t)| a.lerp_to(&b, t))
}

/// Adds two fields together (component-wise for color types).
///
/// This is an ergonomic helper that expands to `Zip + Map`.
/// Works with any type that implements `Add`.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, add};
/// use glam::Vec2;
///
/// let a = Constant::new(3.0_f32);
/// let b = Constant::new(4.0_f32);
///
/// let result = add::<Vec2, _, _, _>(a, b);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 7.0);
/// ```
pub fn add<I, A, B, O>(a: A, b: B) -> Map<Zip<A, B>, impl Fn((O, O)) -> O, (O, O)>
where
    I: Clone,
    O: std::ops::Add<Output = O>,
    A: Field<I, O>,
    B: Field<I, O>,
{
    Zip::new(a, b).map(|(a, b)| a + b)
}

/// Multiplies two fields together (component-wise for color types).
///
/// This is an ergonomic helper that expands to `Zip + Map`.
/// Works with any type that implements `Mul`.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, mul};
/// use glam::Vec2;
///
/// let a = Constant::new(3.0_f32);
/// let b = Constant::new(4.0_f32);
///
/// let result = mul::<Vec2, _, _, _>(a, b);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 12.0);
/// ```
pub fn mul<I, A, B, O>(a: A, b: B) -> Map<Zip<A, B>, impl Fn((O, O)) -> O, (O, O)>
where
    I: Clone,
    O: std::ops::Mul<Output = O>,
    A: Field<I, O>,
    B: Field<I, O>,
{
    Zip::new(a, b).map(|(a, b)| a * b)
}

/// Subtracts two fields (a - b).
///
/// This is an ergonomic helper that expands to `Zip + Map`.
/// Works with any type that implements `Sub`.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, sub};
/// use glam::Vec2;
///
/// let a = Constant::new(7.0_f32);
/// let b = Constant::new(3.0_f32);
///
/// let result = sub::<Vec2, _, _, _>(a, b);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 4.0);
/// ```
pub fn sub<I, A, B, O>(a: A, b: B) -> Map<Zip<A, B>, impl Fn((O, O)) -> O, (O, O)>
where
    I: Clone,
    O: std::ops::Sub<Output = O>,
    A: Field<I, O>,
    B: Field<I, O>,
{
    Zip::new(a, b).map(|(a, b)| a - b)
}

/// Divides two fields (a / b).
///
/// This is an ergonomic helper that expands to `Zip + Map`.
/// Works with any type that implements `Div`.
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, div};
/// use glam::Vec2;
///
/// let a = Constant::new(12.0_f32);
/// let b = Constant::new(3.0_f32);
///
/// let result = div::<Vec2, _, _, _>(a, b);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 4.0);
/// ```
pub fn div<I, A, B, O>(a: A, b: B) -> Map<Zip<A, B>, impl Fn((O, O)) -> O, (O, O)>
where
    I: Clone,
    O: std::ops::Div<Output = O>,
    A: Field<I, O>,
    B: Field<I, O>,
{
    Zip::new(a, b).map(|(a, b)| a / b)
}

/// Mixes two fields using a blend factor.
///
/// This is an alias for `lerp` - both perform linear interpolation.
/// Works with any type that implements `Lerp` (f32, Vec2, Vec3, Rgba, etc.).
///
/// # Example
/// ```
/// use unshape_field::{Field, EvalContext, Constant, mix};
/// use glam::Vec2;
///
/// let a = Constant::new(0.0_f32);
/// let b = Constant::new(10.0_f32);
/// let t = Constant::new(0.5_f32);
///
/// let result = mix::<Vec2, _, _, _, _>(a, b, t);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 5.0);
/// ```
pub fn mix<I, O, A, B, T>(
    a: A,
    b: B,
    t: T,
) -> Map<Zip3<A, B, T>, impl Fn((O, O, f32)) -> O, (O, O, f32)>
where
    I: Clone,
    O: Lerp,
    A: Field<I, O>,
    B: Field<I, O>,
    T: Field<I, f32>,
{
    lerp(a, b, t)
}

// Basic field implementations

/// A constant field that always returns the same value.
#[derive(Debug, Clone, Copy)]
pub struct Constant<O> {
    /// The constant value to return.
    pub value: O,
}

impl<O> Constant<O> {
    /// Create a new constant field.
    pub fn new(value: O) -> Self {
        Self { value }
    }
}

impl<I, O: Clone> Field<I, O> for Constant<O> {
    fn sample(&self, _input: I, _ctx: &EvalContext) -> O {
        self.value.clone()
    }
}

/// A field that returns the input coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct Coordinates;

impl Field<Vec2, Vec2> for Coordinates {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        input
    }
}

impl Field<Vec3, Vec3> for Coordinates {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> Vec3 {
        input
    }
}

impl Field<f32, f32> for Coordinates {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        input
    }
}

// ============================================================================
// Function adapter
// ============================================================================

/// Function adapter - wraps a closure as a field.
pub struct FnField<I, O, F> {
    f: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F: Fn(I, &EvalContext) -> O> FnField<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Fn(I, &EvalContext) -> O> Field<I, O> for FnField<I, O, F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> O {
        (self.f)(input, ctx)
    }
}

/// Creates a field from a closure.
pub fn from_fn<I, O, F: Fn(I, &EvalContext) -> O>(f: F) -> FnField<I, O, F> {
    FnField::new(f)
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
mod tests {
    use super::*;

    #[test]
    fn test_constant_field() {
        let field = Constant::new(42.0f32);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 42.0);
        assert_eq!(field.sample(Vec2::new(100.0, 100.0), &ctx), 42.0);
    }

    #[test]
    fn test_coordinates_field() {
        let field = Coordinates;
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::new(1.0, 2.0), &ctx), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_scale_combinator() {
        let field = <Coordinates as Field<Vec2, Vec2>>::scale(Coordinates, 2.0);
        let ctx = EvalContext::new();

        // Scaling input by 2 means we query at 2x the position
        let result: Vec2 = field.sample(Vec2::new(1.0, 1.0), &ctx);
        assert_eq!(result, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_translate_combinator() {
        let field = <Coordinates as Field<Vec2, Vec2>>::translate(Coordinates, Vec2::new(1.0, 1.0));
        let ctx = EvalContext::new();

        // Translating subtracts from input before sampling
        let result: Vec2 = field.sample(Vec2::new(2.0, 2.0), &ctx);
        assert_eq!(result, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn test_map_combinator() {
        let field = <Constant<f32> as Field<Vec2, f32>>::map(Constant::new(5.0f32), |x| x * 2.0);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 10.0);
    }

    #[test]
    fn test_add_combinator() {
        let a = Constant::new(3.0f32);
        let b = Constant::new(4.0f32);
        let field = add::<Vec2, _, _, _>(a, b);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 7.0);
    }

    #[test]
    fn test_zip_combinator() {
        let a = Constant::new(3.0f32);
        let b = Constant::new(4.0f32);
        let zipped = Zip::new(a, b);
        let ctx = EvalContext::new();

        let (va, vb): (f32, f32) = Field::<Vec2, (f32, f32)>::sample(&zipped, Vec2::ZERO, &ctx);
        assert_eq!(va, 3.0);
        assert_eq!(vb, 4.0);
    }

    #[test]
    fn test_zip_with_map_equals_add() {
        let a = Constant::new(3.0f32);
        let b = Constant::new(4.0f32);
        let zipped = Zip::new(a, b);
        let sum = Map {
            field: zipped,
            f: |(x, y): (f32, f32)| x + y,
            _phantom: PhantomData::<(f32, f32)>,
        };
        let ctx = EvalContext::new();

        assert_eq!(Field::<Vec2, f32>::sample(&sum, Vec2::ZERO, &ctx), 7.0);
    }

    #[test]
    fn test_zip3_combinator() {
        let a = Constant::new(1.0f32);
        let b = Constant::new(2.0f32);
        let c = Constant::new(3.0f32);
        let zipped = Zip3::new(a, b, c);
        let ctx = EvalContext::new();

        let (va, vb, vc): (f32, f32, f32) =
            Field::<Vec2, (f32, f32, f32)>::sample(&zipped, Vec2::ZERO, &ctx);
        assert_eq!(va, 1.0);
        assert_eq!(vb, 2.0);
        assert_eq!(vc, 3.0);
    }

    #[test]
    fn test_zip3_lerp() {
        let a = Constant::new(0.0f32);
        let b = Constant::new(10.0f32);
        let t = Constant::new(0.5f32);
        let zipped = Zip3::new(a, b, t);
        let result = Map {
            field: zipped,
            f: |(a, b, t): (f32, f32, f32)| a * (1.0 - t) + b * t,
            _phantom: PhantomData::<(f32, f32, f32)>,
        };
        let ctx = EvalContext::new();

        assert_eq!(Field::<Vec2, f32>::sample(&result, Vec2::ZERO, &ctx), 5.0);
    }

    #[test]
    fn test_lerp_helper() {
        let a = Constant::new(0.0f32);
        let b = Constant::new(10.0f32);
        let t = Constant::new(0.25f32);
        let result = super::lerp::<Vec2, _, _, _, _>(a, b, t);
        let ctx = EvalContext::new();

        assert_eq!(result.sample(Vec2::ZERO, &ctx), 2.5);
    }

    #[test]
    fn test_lerp_vec2_output() {
        // Generic lerp works with any Lerp-implementing type
        let a = Constant::new(Vec2::new(0.0, 0.0));
        let b = Constant::new(Vec2::new(10.0, 20.0));
        let t = Constant::new(0.5f32);
        let result = super::lerp::<Vec2, _, _, _, _>(a, b, t);
        let ctx = EvalContext::new();

        let v = result.sample(Vec2::ZERO, &ctx);
        assert_eq!(v, Vec2::new(5.0, 10.0));
    }

    #[test]
    fn test_add_vec2_output() {
        // Generic add works with any Add-implementing type
        let a = Constant::new(Vec2::new(1.0, 2.0));
        let b = Constant::new(Vec2::new(3.0, 4.0));
        let result = super::add::<Vec2, _, _, _>(a, b);
        let ctx = EvalContext::new();

        let v = result.sample(Vec2::ZERO, &ctx);
        assert_eq!(v, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_zip_standalone_fn() {
        let a = Constant::new(5.0f32);
        let b = Constant::new(7.0f32);
        let zipped = super::zip(a, b);
        let ctx = EvalContext::new();

        let (va, vb): (f32, f32) = zipped.sample(0.0f32, &ctx);
        assert_eq!(va, 5.0);
        assert_eq!(vb, 7.0);
    }

    #[test]
    fn test_zip3_standalone_fn() {
        let a = Constant::new(1.0f32);
        let b = Constant::new(2.0f32);
        let c = Constant::new(3.0f32);
        let zipped = super::zip3(a, b, c);
        let ctx = EvalContext::new();

        let (va, vb, vc): (f32, f32, f32) = zipped.sample(0.0f32, &ctx);
        assert_eq!(va, 1.0);
        assert_eq!(vb, 2.0);
        assert_eq!(vc, 3.0);
    }

    #[test]
    fn test_perlin_field() {
        let field = Perlin2D::new().scale(4.0);
        let ctx = EvalContext::new();

        let v1 = field.sample(Vec2::new(0.0, 0.0), &ctx);
        let v2 = field.sample(Vec2::new(1.0, 1.0), &ctx);

        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_white_noise_1d() {
        let field = WhiteNoise1D::with_seed(42);
        let ctx = EvalContext::new();

        // Values should be in [0, 1]
        for i in 0..100 {
            let v = field.sample(i as f32 * 0.1, &ctx);
            assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
        }

        // Same input = same output (deterministic)
        let v1 = field.sample(0.5, &ctx);
        let v2 = field.sample(0.5, &ctx);
        assert_eq!(v1, v2);

        // Different seeds = different output
        let field2 = WhiteNoise1D::with_seed(123);
        let v3 = field2.sample(0.5, &ctx);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_white_noise_2d() {
        let field = WhiteNoise2D::with_seed(42);
        let ctx = EvalContext::new();

        // Values should be in [0, 1]
        for y in 0..10 {
            for x in 0..10 {
                let v = field.sample(Vec2::new(x as f32 * 0.1, y as f32 * 0.1), &ctx);
                assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
            }
        }

        // Deterministic
        let v1 = field.sample(Vec2::new(0.5, 0.5), &ctx);
        let v2 = field.sample(Vec2::new(0.5, 0.5), &ctx);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_white_noise_3d() {
        let field = WhiteNoise3D::with_seed(42);
        let ctx = EvalContext::new();

        // Values should be in [0, 1]
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let v = field.sample(
                        Vec3::new(x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1),
                        &ctx,
                    );
                    assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
                }
            }
        }

        // Useful for temporal dithering: same (x,y,t) = same value
        let v1 = field.sample(Vec3::new(0.5, 0.5, 0.0), &ctx);
        let v2 = field.sample(Vec3::new(0.5, 0.5, 0.0), &ctx);
        assert_eq!(v1, v2);

        // Different time = different value (for animation)
        let v3 = field.sample(Vec3::new(0.5, 0.5, 1.0), &ctx);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_checkerboard() {
        let field = Checkerboard::new();
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        assert_eq!(field.sample(Vec2::new(1.5, 0.5), &ctx), 0.0);
    }

    #[test]
    fn test_distance_circle() {
        let field = DistanceCircle::new(Vec2::ZERO, 1.0);
        let ctx = EvalContext::new();

        assert!(field.sample(Vec2::ZERO, &ctx) < 0.0);
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        assert!(field.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_metaballs() {
        let balls = vec![Metaball::new_2d(Vec2::ZERO, 1.0)];
        let field = Metaballs2D::new(balls);
        let ctx = EvalContext::new();

        let center_val = field.sample(Vec2::ZERO, &ctx);
        assert!(center_val > 10.0);

        let radius_val = field.sample(Vec2::new(1.0, 0.0), &ctx);
        assert!((radius_val - 1.0).abs() < 0.01);
    }

    // Terrain generation tests

    #[test]
    fn test_terrain_basic() {
        let terrain = Terrain2D::new();
        let ctx = EvalContext::new();

        // Sample at various points
        let v1 = terrain.sample(Vec2::new(0.0, 0.0), &ctx);
        let v2 = terrain.sample(Vec2::new(1.0, 1.0), &ctx);
        let v3 = terrain.sample(Vec2::new(10.0, 10.0), &ctx);

        // All values should be in [0, 1] range
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
        assert!((0.0..=1.0).contains(&v3));

        // Values should vary (not all the same)
        assert!((v1 - v2).abs() > 0.001 || (v2 - v3).abs() > 0.001);
    }

    #[test]
    fn test_terrain_presets() {
        let ctx = EvalContext::new();
        let point = Vec2::new(0.5, 0.5);

        // Test all presets produce valid values
        let hills = Terrain2D::rolling_hills().sample(point, &ctx);
        let mountains = Terrain2D::mountains().sample(point, &ctx);
        let plains = Terrain2D::plains().sample(point, &ctx);
        let canyons = Terrain2D::canyons().sample(point, &ctx);

        assert!((0.0..=1.0).contains(&hills));
        assert!((0.0..=1.0).contains(&mountains));
        assert!((0.0..=1.0).contains(&plains));
        assert!((0.0..=1.0).contains(&canyons));
    }

    #[test]
    fn test_terrain_deterministic() {
        let ctx = EvalContext::new();
        let terrain1 = Terrain2D {
            seed: 42,
            ..Default::default()
        };
        let terrain2 = Terrain2D {
            seed: 42,
            ..Default::default()
        };

        let point = Vec2::new(0.5, 0.5);
        assert_eq!(terrain1.sample(point, &ctx), terrain2.sample(point, &ctx));
    }

    #[test]
    fn test_ridged_terrain() {
        let terrain = RidgedTerrain2D::new();
        let ctx = EvalContext::new();

        let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_billowy_terrain() {
        let terrain = BillowyTerrain2D::new();
        let ctx = EvalContext::new();

        let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_island_terrain() {
        let terrain = IslandTerrain2D {
            radius: 1.0,
            ..Default::default()
        };
        let ctx = EvalContext::new();

        // Center should have terrain
        let center = terrain.sample(Vec2::ZERO, &ctx);
        assert!(center >= 0.0);

        // Far from center should be zero (outside island)
        let far = terrain.sample(Vec2::new(10.0, 10.0), &ctx);
        assert_eq!(far, 0.0);
    }

    #[test]
    fn test_terraced_terrain() {
        let base = Terrain2D::new();
        let terraced = TerracedTerrain2D::new(base, 5);
        let ctx = EvalContext::new();

        let v = terraced.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_terrain_struct_literal() {
        let terrain = Terrain2D {
            seed: 123,
            octaves: 4,
            lacunarity: 2.5,
            persistence: 0.4,
            scale: 2.0,
            exponent: 1.5,
        };

        assert_eq!(terrain.seed, 123);
        assert_eq!(terrain.octaves, 4);
        assert!((terrain.lacunarity - 2.5).abs() < 0.001);
        assert!((terrain.persistence - 0.4).abs() < 0.001);
        assert!((terrain.scale - 2.0).abs() < 0.001);
        assert!((terrain.exponent - 1.5).abs() < 0.001);
    }

    // ========================================================================
    // Heightmap tests
    // ========================================================================

    #[test]
    fn test_heightmap_new() {
        let hm = Heightmap::new(10, 20);
        assert_eq!(hm.width, 10);
        assert_eq!(hm.height, 20);
        assert_eq!(hm.data.len(), 200);
        assert!(hm.data.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn test_heightmap_get_set() {
        let mut hm = Heightmap::new(10, 10);
        hm.set(5, 5, 1.0);
        assert_eq!(hm.get(5, 5), 1.0);
        assert_eq!(hm.get(0, 0), 0.0);

        // Out of bounds returns 0
        assert_eq!(hm.get(100, 100), 0.0);
    }

    #[test]
    fn test_heightmap_sample() {
        let mut hm = Heightmap::new(4, 4);
        hm.set(1, 1, 0.0);
        hm.set(2, 1, 1.0);
        hm.set(1, 2, 0.0);
        hm.set(2, 2, 1.0);

        // Sample at integer position
        assert!((hm.sample(1.0, 1.0) - 0.0).abs() < 0.001);
        assert!((hm.sample(2.0, 1.0) - 1.0).abs() < 0.001);

        // Sample at midpoint - should interpolate
        let mid = hm.sample(1.5, 1.5);
        assert!((mid - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_heightmap_gradient() {
        let mut hm = Heightmap::new(10, 10);
        // Create a slope: height increases with x
        for y in 0..10 {
            for x in 0..10 {
                hm.set(x, y, x as f32);
            }
        }

        let grad = hm.gradient(5.0, 5.0);
        // Gradient should point in +x direction
        assert!(grad.x > 0.5);
        assert!(grad.y.abs() < 0.1);
    }

    #[test]
    fn test_heightmap_bounds() {
        let mut hm = Heightmap::new(5, 5);
        hm.set(0, 0, -10.0);
        hm.set(2, 2, 20.0);

        let (min, max) = hm.bounds();
        assert_eq!(min, -10.0);
        assert_eq!(max, 20.0);
    }

    #[test]
    fn test_heightmap_normalize() {
        let mut hm = Heightmap::new(5, 5);
        hm.set(0, 0, -10.0);
        hm.set(2, 2, 20.0);
        hm.set(4, 4, 5.0);

        hm.normalize();

        let (min, max) = hm.bounds();
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);

        // Check specific values
        assert!((hm.get(0, 0) - 0.0).abs() < 0.001); // -10 -> 0
        assert!((hm.get(2, 2) - 1.0).abs() < 0.001); // 20 -> 1
        assert!((hm.get(4, 4) - 0.5).abs() < 0.001); // 5 -> 0.5
    }

    #[test]
    fn test_heightmap_from_field() {
        let field = Constant::new(0.5f32);
        let hm = Heightmap::from_field(&field, 10, 10, 1.0);

        assert_eq!(hm.width, 10);
        assert_eq!(hm.height, 10);
        assert!(hm.data.iter().all(|&h| (h - 0.5).abs() < 0.001));
    }

    // ========================================================================
    // Hydraulic erosion tests
    // ========================================================================

    #[test]
    fn test_hydraulic_erosion_config_default() {
        let config = HydraulicErosionConfig::default();
        assert_eq!(config.iterations, 10000);
        assert_eq!(config.max_lifetime, 64);
        assert_eq!(config.brush_radius, 3);
    }

    #[test]
    fn test_hydraulic_erosion_modifies_terrain() {
        // Create a simple cone terrain
        let mut hm = Heightmap::new(32, 32);
        let cx = 16.0;
        let cy = 16.0;

        for y in 0..32 {
            for x in 0..32 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                hm.set(x, y, (16.0 - dist).max(0.0));
            }
        }

        let original_sum: f32 = hm.data.iter().sum();

        let config = HydraulicErosionConfig {
            iterations: 1000,
            max_lifetime: 30,
            brush_radius: 2,
            ..Default::default()
        };

        hydraulic_erosion(&mut hm, &config, 12345);

        // Terrain should be modified (not identical to original)
        let new_sum: f32 = hm.data.iter().sum();
        assert!(new_sum != original_sum, "terrain should be modified");

        // Most material should remain (within bounds)
        assert!(new_sum > 0.0, "terrain should have material");

        // Heights should still be reasonable
        assert!(hm.data.iter().all(|&h| h >= -5.0 && h < 25.0));
    }

    #[test]
    fn test_hydraulic_erosion_deterministic() {
        let create_terrain = || {
            let mut hm = Heightmap::new(16, 16);
            for y in 0..16 {
                for x in 0..16 {
                    hm.set(x, y, ((x + y) as f32 / 30.0).sin() * 5.0);
                }
            }
            hm
        };

        let config = HydraulicErosionConfig {
            iterations: 500,
            ..Default::default()
        };

        let mut hm1 = create_terrain();
        let mut hm2 = create_terrain();

        hydraulic_erosion(&mut hm1, &config, 42);
        hydraulic_erosion(&mut hm2, &config, 42);

        // Same seed should produce same result
        assert_eq!(hm1.data, hm2.data);

        // Different seed should produce different result
        let mut hm3 = create_terrain();
        hydraulic_erosion(&mut hm3, &config, 999);
        assert_ne!(hm1.data, hm3.data);
    }

    // ========================================================================
    // Thermal erosion tests
    // ========================================================================

    #[test]
    fn test_thermal_erosion_config_default() {
        let config = ThermalErosionConfig::default();
        assert_eq!(config.iterations, 50);
        assert!((config.talus_angle - 0.8).abs() < 0.001);
        assert!((config.transfer_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_thermal_erosion_smooths_steep_slopes() {
        // Create terrain with a sharp spike
        let mut hm = Heightmap::new(16, 16);
        hm.set(8, 8, 10.0); // Sharp spike

        // Calculate initial max slope
        let initial_spike = hm.get(8, 8);

        let config = ThermalErosionConfig {
            iterations: 100,
            talus_angle: 0.5,
            transfer_rate: 0.5,
        };

        thermal_erosion(&mut hm, &config);

        // Spike should be reduced
        assert!(hm.get(8, 8) < initial_spike);

        // Material should have spread to neighbors
        assert!(hm.get(7, 8) > 0.0);
        assert!(hm.get(9, 8) > 0.0);
        assert!(hm.get(8, 7) > 0.0);
        assert!(hm.get(8, 9) > 0.0);
    }

    #[test]
    fn test_thermal_erosion_preserves_mass() {
        let mut hm = Heightmap::new(16, 16);
        // Create terrain where mass is away from edges
        for y in 4..12 {
            for x in 4..12 {
                hm.set(x, y, 5.0 + ((x + y) as f32 * 0.1));
            }
        }

        let original_sum: f32 = hm.data.iter().sum();

        let config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &config);

        let new_sum: f32 = hm.data.iter().sum();

        // Mass should be conserved (within floating point tolerance)
        assert!((new_sum - original_sum).abs() < 0.001);
    }

    #[test]
    fn test_thermal_erosion_flat_terrain_unchanged() {
        let mut hm = Heightmap::new(16, 16);
        // Flat terrain
        for y in 0..16 {
            for x in 0..16 {
                hm.set(x, y, 5.0);
            }
        }

        let original_data = hm.data.clone();

        let config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &config);

        // Flat terrain should remain unchanged
        assert_eq!(hm.data, original_data);
    }

    #[test]
    fn test_combined_erosion() {
        // Create initial terrain from noise
        let terrain = Terrain2D {
            seed: 42,
            ..Default::default()
        };
        let mut hm = Heightmap::from_field(&terrain, 32, 32, 0.1);
        hm.normalize();

        // Scale up for erosion
        for h in &mut hm.data {
            *h *= 10.0;
        }

        // Apply both erosion types
        let hydro_config = HydraulicErosionConfig {
            iterations: 500,
            ..Default::default()
        };
        hydraulic_erosion(&mut hm, &hydro_config, 12345);

        let thermal_config = ThermalErosionConfig::default();
        thermal_erosion(&mut hm, &thermal_config);

        // Result should be valid terrain
        assert!(hm.data.iter().all(|&h| h.is_finite()));
        let (min, max) = hm.bounds();
        assert!(min < max);
    }

    // ====== Network Tests ======

    #[test]
    fn test_network_node() {
        let node = NetworkNode {
            position: Vec2::new(10.0, 20.0),
            height: 5.0,
            importance: 2.0,
        };

        assert_eq!(node.position, Vec2::new(10.0, 20.0));
        assert_eq!(node.height, 5.0);
        assert_eq!(node.importance, 2.0);
    }

    #[test]
    fn test_network_edge() {
        let edge = NetworkEdge {
            start: 0,
            end: 1,
            weight: 10.0,
            path: vec![Vec2::new(5.0, 5.0)],
            edge_type: 1.0,
        };

        assert_eq!(edge.start, 0);
        assert_eq!(edge.end, 1);
        assert_eq!(edge.weight, 10.0);
        assert_eq!(edge.path.len(), 1);
    }

    #[test]
    fn test_network_creation() {
        let mut network = Network::new();
        let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
        let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));
        let n2 = network.add_node(NetworkNode::new(Vec2::new(5.0, 10.0)));

        network.add_edge(n0, n1);
        network.add_edge(n1, n2);
        network.add_edge(n2, n0);

        assert_eq!(network.nodes.len(), 3);
        assert_eq!(network.edges.len(), 3);
    }

    #[test]
    fn test_network_neighbors() {
        let mut network = Network::new();
        let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
        let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));
        let n2 = network.add_node(NetworkNode::new(Vec2::new(0.0, 10.0)));

        network.add_edge(n0, n1);
        network.add_edge(n0, n2);

        let neighbors = network.neighbors(n0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n1));
        assert!(neighbors.contains(&n2));
    }

    #[test]
    fn test_network_edge_path() {
        let mut network = Network::new();
        let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
        let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));

        network.add_edge(n0, n1).path = vec![Vec2::new(3.0, 1.0), Vec2::new(7.0, 1.0)];

        let full_path = network.edge_path(&network.edges[0]);
        assert_eq!(full_path.len(), 4); // start + 2 intermediate + end
        assert_eq!(full_path[0], Vec2::ZERO);
        assert_eq!(full_path[3], Vec2::new(10.0, 0.0));
    }

    #[test]
    fn test_road_network_config_default() {
        let config = RoadNetworkConfig::default();
        assert!(config.num_nodes > 0);
        assert!(config.use_mst);
    }

    #[test]
    fn test_generate_road_network() {
        let config = RoadNetworkConfig {
            num_nodes: 8,
            bounds: (0.0, 0.0, 50.0, 50.0),
            ..Default::default()
        };

        let network = generate_road_network(&config, 42);

        assert_eq!(network.nodes.len(), 8);
        // MST should have n-1 edges plus some extras
        assert!(network.edges.len() >= 7);

        // All nodes should be within bounds
        for node in &network.nodes {
            assert!(node.position.x >= 0.0 && node.position.x <= 50.0);
            assert!(node.position.y >= 0.0 && node.position.y <= 50.0);
        }
    }

    #[test]
    fn test_road_network_connected() {
        let config = RoadNetworkConfig {
            num_nodes: 5,
            use_mst: true,
            extra_connectivity: 0.0,
            ..Default::default()
        };

        let network = generate_road_network(&config, 123);

        // MST with 5 nodes should have exactly 4 edges
        assert_eq!(network.edges.len(), 4);
    }

    #[test]
    fn test_river_network_config_default() {
        let config = RiverNetworkConfig::default();
        assert!(config.num_sources > 0);
        assert!(config.max_steps > 0);
    }

    #[test]
    fn test_generate_river_network() {
        // Create a sloped terrain
        let mut hm = Heightmap::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                // Height decreases from top-left to bottom-right
                let h = 1.0 - (x + y) as f32 / 62.0;
                hm.set(x, y, h);
            }
        }

        let config = RiverNetworkConfig {
            num_sources: 2,
            source_min_height: 0.7,
            ..Default::default()
        };

        let network = generate_river_network(&hm, &config, 42);

        // Should have at least some nodes and edges
        assert!(network.nodes.len() >= 1);
    }

    #[test]
    fn test_flow_accumulation() {
        // Create a simple bowl-shaped terrain
        let mut hm = Heightmap::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let dx = x as f32 - 7.5;
                let dy = y as f32 - 7.5;
                let h = (dx * dx + dy * dy).sqrt() / 10.0;
                hm.set(x, y, h);
            }
        }

        let flow = compute_flow_accumulation(&hm);

        // Center should have highest flow (everything drains there)
        let center_flow = flow.get(7, 7);
        let edge_flow = flow.get(0, 0);
        assert!(center_flow > edge_flow);
    }

    #[test]
    fn test_network_sample_edges() {
        let mut network = Network::new();
        let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
        let n1 = network.add_node(NetworkNode::new(Vec2::new(20.0, 0.0)));

        network.add_edge(n0, n1);

        let sampled = network.sample_edges(5.0);
        assert_eq!(sampled.len(), 1);

        // Should have multiple points along the edge
        assert!(sampled[0].len() >= 4);
    }

    #[test]
    fn test_road_network_deterministic() {
        let config = RoadNetworkConfig::default();

        let network1 = generate_road_network(&config, 12345);
        let network2 = generate_road_network(&config, 12345);

        // Same seed should produce same network
        assert_eq!(network1.nodes.len(), network2.nodes.len());
        assert_eq!(network1.edges.len(), network2.edges.len());

        for (n1, n2) in network1.nodes.iter().zip(network2.nodes.iter()) {
            assert_eq!(n1.position, n2.position);
        }
    }

    // 2D SDF primitive tests

    #[test]
    fn test_distance_rounded_box() {
        let ctx = EvalContext::new();
        let sdf = DistanceRoundedBox::new(Vec2::ZERO, Vec2::new(1.0, 0.5), 0.1);

        // Center should be inside (negative)
        assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

        // Far away should be outside (positive)
        assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);

        // At edge should be close to zero
        let edge_val = sdf.sample(Vec2::new(1.0, 0.0), &ctx).abs();
        assert!(edge_val < 0.2);
    }

    #[test]
    fn test_distance_ellipse() {
        let ctx = EvalContext::new();
        let sdf = DistanceEllipse::new(Vec2::ZERO, Vec2::new(2.0, 1.0));

        // Center should be inside (negative)
        assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

        // On the ellipse boundary should be close to zero
        let on_x = sdf.sample(Vec2::new(2.0, 0.0), &ctx);
        let on_y = sdf.sample(Vec2::new(0.0, 1.0), &ctx);
        assert!(on_x.abs() < 0.1);
        assert!(on_y.abs() < 0.1);

        // Outside should be positive
        assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_distance_capsule() {
        let ctx = EvalContext::new();
        let sdf = DistanceCapsule::new(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), 0.5);

        // Center should be inside
        assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

        // At ends should be at edge
        let end_val = sdf.sample(Vec2::new(1.5, 0.0), &ctx).abs();
        assert!(end_val < 0.1);

        // Outside should be positive
        assert!(sdf.sample(Vec2::new(3.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_distance_triangle() {
        let ctx = EvalContext::new();
        let sdf = DistanceTriangle::new(
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
        );

        // Center should be inside (negative)
        assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

        // Far outside should be positive
        assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_distance_regular_polygon() {
        let ctx = EvalContext::new();

        // Hexagon
        let hex = DistanceRegularPolygon::new(Vec2::ZERO, 1.0, 6);

        // Center should be inside
        assert!(hex.sample(Vec2::ZERO, &ctx) < 0.0);

        // Outside should be positive
        assert!(hex.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);

        // Square
        let square = DistanceRegularPolygon::new(Vec2::ZERO, 1.0, 4);
        assert!(square.sample(Vec2::ZERO, &ctx) < 0.0);
    }

    #[test]
    fn test_distance_arc() {
        let ctx = EvalContext::new();

        // Pie shape (filled arc)
        let pie = DistanceArc::pie(Vec2::ZERO, 1.0, std::f32::consts::FRAC_PI_4);

        // Center of pie should be inside
        assert!(pie.sample(Vec2::new(0.3, 0.0), &ctx) < 0.0);

        // Outside the angle should be positive
        assert!(pie.sample(Vec2::new(0.0, 0.5), &ctx) > 0.0);

        // Far outside radius should be positive
        assert!(pie.sample(Vec2::new(3.0, 0.0), &ctx) > 0.0);
    }
}

/// Invariant tests for field properties.
///
/// These tests verify mathematical and statistical properties that should hold
/// for all field implementations. Run with:
///
/// ```sh
/// cargo test -p unshape-field --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use glam::{Vec2, Vec3};

    const SAMPLE_COUNT: usize = 10_000;

    // ========================================================================
    // Noise field range tests
    // ========================================================================

    /// All 2D noise fields should produce values in [0, 1].
    #[test]
    fn test_noise_2d_range() {
        let ctx = EvalContext::new();
        let fields: Vec<(&str, Box<dyn Field<Vec2, f32>>)> = vec![
            ("Perlin2D", Box::new(Perlin2D::new())),
            ("Simplex2D", Box::new(Simplex2D::new())),
            ("Value2D", Box::new(Value2D::new())),
            ("Worley2D", Box::new(Worley2D::new())),
            ("WorleyF2_2D", Box::new(WorleyF2_2D::new())),
            ("WorleyEdge2D", Box::new(WorleyEdge2D::new())),
            ("WhiteNoise2D", Box::new(WhiteNoise2D::new())),
            ("PinkNoise2D", Box::new(PinkNoise2D::new())),
            ("BrownNoise2D", Box::new(BrownNoise2D::new())),
        ];

        for (name, field) in fields {
            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for i in 0..SAMPLE_COUNT {
                let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let v = field.sample(Vec2::new(x, y), &ctx);
                min = min.min(v);
                max = max.max(v);
            }

            assert!(
                min >= -0.01 && max <= 1.01,
                "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
            );
        }
    }

    /// All 3D noise fields should produce values in [0, 1].
    #[test]
    fn test_noise_3d_range() {
        let ctx = EvalContext::new();
        let fields: Vec<(&str, Box<dyn Field<Vec3, f32>>)> = vec![
            ("Perlin3D", Box::new(Perlin3D::new())),
            ("Simplex3D", Box::new(Simplex3D::new())),
            ("Value3D", Box::new(Value3D::new())),
            ("Worley3D", Box::new(Worley3D::new())),
        ];

        for (name, field) in fields {
            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for i in 0..SAMPLE_COUNT {
                let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let z = ((i * 13) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let v = field.sample(Vec3::new(x, y, z), &ctx);
                min = min.min(v);
                max = max.max(v);
            }

            assert!(
                min >= -0.01 && max <= 1.01,
                "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
            );
        }
    }

    /// 1D noise fields should produce values in [0, 1].
    #[test]
    fn test_noise_1d_range() {
        let ctx = EvalContext::new();
        let fields: Vec<(&str, Box<dyn Field<f32, f32>>)> = vec![
            ("Perlin1D", Box::new(Perlin1D::new())),
            ("Simplex1D", Box::new(Simplex1D::new())),
            ("Value1D", Box::new(Value1D::new())),
            ("Worley1D", Box::new(Worley1D::new())),
            ("WhiteNoise1D", Box::new(WhiteNoise1D::new())),
            ("PinkNoise1D", Box::new(PinkNoise1D::new())),
            ("BrownNoise1D", Box::new(BrownNoise1D::new())),
            ("VioletNoise1D", Box::new(VioletNoise1D::new())),
            ("GreyNoise1D", Box::new(GreyNoise1D::new())),
            ("VelvetNoise1D", Box::new(VelvetNoise1D::new())),
        ];

        for (name, field) in fields {
            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for i in 0..SAMPLE_COUNT {
                let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
                let v = field.sample(x, &ctx);
                min = min.min(v);
                max = max.max(v);
            }

            assert!(
                min >= -0.01 && max <= 1.01,
                "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
            );
        }
    }

    // ========================================================================
    // Determinism tests
    // ========================================================================

    /// Noise fields with the same seed should produce identical output.
    #[test]
    fn test_noise_determinism() {
        let ctx = EvalContext::new();
        let seeds = [0, 42, 12345, -999];
        let positions = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.5, -2.3),
            Vec2::new(100.0, 100.0),
            Vec2::new(-50.0, 25.0),
        ];

        for seed in seeds {
            for pos in positions {
                // Create two instances with same seed
                let a = Perlin2D::with_seed(seed);
                let b = Perlin2D::with_seed(seed);

                let va = a.sample(pos, &ctx);
                let vb = b.sample(pos, &ctx);

                assert_eq!(
                    va, vb,
                    "Same seed should produce identical output: seed={seed}, pos={pos:?}"
                );
            }
        }
    }

    /// Different seeds should produce different output (with high probability).
    #[test]
    fn test_different_seeds_differ() {
        let ctx = EvalContext::new();
        let pos = Vec2::new(5.5, -3.2);

        let v0 = Perlin2D::with_seed(0).sample(pos, &ctx);
        let v1 = Perlin2D::with_seed(1).sample(pos, &ctx);
        let v2 = Perlin2D::with_seed(42).sample(pos, &ctx);

        // At least 2 of 3 should be different
        let different = (v0 != v1) as u32 + (v1 != v2) as u32 + (v0 != v2) as u32;
        assert!(
            different >= 2,
            "Different seeds should produce different outputs"
        );
    }

    // ========================================================================
    // FBM property tests
    // ========================================================================

    /// FBM should still produce values in [0, 1] after octave composition.
    #[test]
    fn test_fbm_range() {
        let ctx = EvalContext::new();

        for octaves in [1, 2, 4, 8] {
            let fbm = Fbm2D::new(Perlin2D::new()).octaves(octaves);

            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for i in 0..SAMPLE_COUNT {
                let x = (i as f32 / SAMPLE_COUNT as f32) * 50.0 - 25.0;
                let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 50.0 - 25.0;
                let v = fbm.sample(Vec2::new(x, y), &ctx);
                min = min.min(v);
                max = max.max(v);
            }

            assert!(
                min >= -0.01 && max <= 1.01,
                "FBM({octaves} octaves): values out of range, got [{min:.3}, {max:.3}]"
            );
        }
    }

    // ========================================================================
    // Terrain property tests
    // ========================================================================

    /// Terrain fields should produce values in [0, 1].
    #[test]
    fn test_terrain_range() {
        let ctx = EvalContext::new();
        let terrains: Vec<(&str, Box<dyn Field<Vec2, f32>>)> = vec![
            ("Terrain2D::default", Box::new(Terrain2D::default())),
            ("Terrain2D::mountains", Box::new(Terrain2D::mountains())),
            ("Terrain2D::plains", Box::new(Terrain2D::plains())),
            ("Terrain2D::canyons", Box::new(Terrain2D::canyons())),
            ("RidgedTerrain2D", Box::new(RidgedTerrain2D::default())),
            ("BillowyTerrain2D", Box::new(BillowyTerrain2D::default())),
        ];

        for (name, terrain) in terrains {
            let mut min = f32::MAX;
            let mut max = f32::MIN;

            for i in 0..SAMPLE_COUNT {
                let x = (i as f32 / SAMPLE_COUNT as f32) * 20.0 - 10.0;
                let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 20.0 - 10.0;
                let v = terrain.sample(Vec2::new(x, y), &ctx);
                min = min.min(v);
                max = max.max(v);
            }

            assert!(
                min >= -0.01 && max <= 1.01,
                "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
            );
        }
    }

    // ========================================================================
    // Combinator property tests
    // ========================================================================

    /// Scale combinator should scale input coordinates (zoom effect).
    #[test]
    fn test_scale_combinator() {
        let ctx = EvalContext::new();
        // Use checkerboard which has visible scale effect
        let field = Checkerboard::with_scale(1.0);

        // At scale 2.0, pattern should appear twice as large (sample at 1.0 = unscaled at 2.0)
        let scaled = Scale {
            field: field.clone(),
            factor: 2.0,
        };

        // Sample scaled at (0.25, 0.25) = unscaled at (0.5, 0.5)
        let v_scaled = scaled.sample(Vec2::new(0.25, 0.25), &ctx);
        let v_unscaled = field.sample(Vec2::new(0.5, 0.5), &ctx);

        assert_eq!(
            v_scaled, v_unscaled,
            "Scale should multiply input coordinates"
        );
    }

    /// Add helper should sum outputs correctly.
    #[test]
    fn test_add_combinator() {
        let ctx = EvalContext::new();
        let a = Constant::<f32>::new(0.3);
        let b = Constant::<f32>::new(0.4);

        let sum = add::<Vec2, _, _, _>(a, b);
        let v = sum.sample(Vec2::ZERO, &ctx);

        assert!((v - 0.7).abs() < 0.001, "add: expected 0.7, got {v}");
    }

    /// Translate combinator should shift input coordinates.
    #[test]
    fn test_translate_combinator() {
        let ctx = EvalContext::new();
        // Use a field that depends on position
        let field = Checkerboard::with_scale(1.0);

        // Translate shifts input by subtracting offset
        // So translated.sample(p) = field.sample(p - offset)
        let translated = Translate {
            field: field.clone(),
            offset: Vec2::new(1.0, 1.0),
        };

        let v1 = translated.sample(Vec2::new(1.5, 1.5), &ctx);
        let v2 = field.sample(Vec2::new(0.5, 0.5), &ctx);

        assert_eq!(v1, v2, "Translate should shift input coordinates");
    }

    // ========================================================================
    // Statistical distribution tests
    // ========================================================================

    /// White noise should have approximately uniform distribution.
    #[test]
    fn test_white_noise_distribution() {
        let ctx = EvalContext::new();
        let noise = WhiteNoise2D::new();

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for i in 0..SAMPLE_COUNT {
            let x = i as f32 * 0.1;
            let y = (i * 7) as f32 * 0.1;
            let v = noise.sample(Vec2::new(x, y), &ctx);
            sum += v;
            sum_sq += v * v;
        }

        let mean = sum / SAMPLE_COUNT as f32;
        let variance = sum_sq / SAMPLE_COUNT as f32 - mean * mean;

        // Uniform [0,1] has mean 0.5 and variance 1/12 ≈ 0.0833
        assert!(
            (mean - 0.5).abs() < 0.05,
            "White noise mean should be ~0.5, got {mean}"
        );
        assert!(
            (variance - 0.0833).abs() < 0.02,
            "White noise variance should be ~0.0833, got {variance}"
        );
    }

    /// Velvet noise should be mostly neutral (0.5) with occasional impulses.
    #[test]
    fn test_velvet_noise_sparsity() {
        let ctx = EvalContext::new();
        let velvet = VelvetNoise1D::new().density(0.1);

        let mut neutral_count = 0;

        for i in 0..SAMPLE_COUNT {
            let x = i as f32 * 0.01;
            let v = velvet.sample(x, &ctx);
            if (v - 0.5).abs() < 0.01 {
                neutral_count += 1;
            }
        }

        let neutral_ratio = neutral_count as f32 / SAMPLE_COUNT as f32;

        // With 10% density, ~90% should be neutral
        assert!(
            neutral_ratio > 0.85,
            "Velvet noise should be mostly neutral, got {:.1}% neutral",
            neutral_ratio * 100.0
        );
    }
}
