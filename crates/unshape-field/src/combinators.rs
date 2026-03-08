//! Field combinators: Map, Zip, Zip3, Scale, Translate, and ergonomic helpers.
//!
//! These types compose existing fields into new ones without storing data —
//! computation is deferred until `sample()` is called.

use std::marker::PhantomData;

use glam::{Vec2, Vec3};
use unshape_easing::Lerp;

use crate::{EvalContext, Field};

// ============================================================================
// Core combinator structs
// ============================================================================

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
    pub(crate) a: A,
    pub(crate) b: B,
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
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
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

// ============================================================================
// Input-space transforms
// ============================================================================

/// Scales the input coordinates of a field.
pub struct Scale<F> {
    pub(crate) field: F,
    pub(crate) factor: f32,
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
    pub(crate) field: F,
    pub(crate) offset: I,
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
#[allow(clippy::type_complexity)]
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
#[allow(clippy::type_complexity)]
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
#[allow(clippy::type_complexity)]
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
#[allow(clippy::type_complexity)]
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
#[allow(clippy::type_complexity)]
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
#[allow(clippy::type_complexity)]
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
