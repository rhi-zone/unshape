//! Field trait for lazy evaluation.
//!
//! A `Field<I, O>` represents a function that can be sampled at any point.
//! Fields are lazy - they describe computation, not data. Evaluation happens
//! on demand when you call `sample()`.
//!
//! # Examples
//!
//! ```
//! use rhizome_resin_field::{Field, EvalContext, Perlin2D};
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

use glam::{Vec2, Vec3};
use std::marker::PhantomData;

pub use context::EvalContext;

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

    /// Adds this field to another.
    fn add<F2>(self, other: F2) -> Add<Self, F2>
    where
        Self: Sized,
        F2: Field<I, O>,
    {
        Add { a: self, b: other }
    }

    /// Multiplies this field by another.
    fn mul<F2>(self, other: F2) -> Mul<Self, F2>
    where
        Self: Sized,
        F2: Field<I, O>,
    {
        Mul { a: self, b: other }
    }

    /// Mixes this field with another using a blend factor field.
    fn mix<F2, FB>(self, other: F2, blend: FB) -> Mix<Self, F2, FB>
    where
        Self: Sized,
        F2: Field<I, O>,
        FB: Field<I, f32>,
    {
        Mix {
            a: self,
            b: other,
            blend,
        }
    }

    /// Zips this field with another, yielding a tuple of their outputs.
    ///
    /// This is a fundamental combinator - `Add`, `Mul`, `Mix` can all be
    /// expressed as `zip().map(...)`.
    ///
    /// # Example
    /// ```
    /// use rhizome_resin_field::{Field, EvalContext, Constant, Zip};
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
/// This is a fundamental primitive - binary operations like `Add`, `Mul`
/// can be expressed as `Zip + Map`.
///
/// # Example
/// ```
/// use rhizome_resin_field::{Field, EvalContext, Constant, Zip, Map};
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
/// This is a fundamental primitive - ternary operations like `Mix`/lerp
/// can be expressed as `Zip3 + Map`.
///
/// # Example
/// ```
/// use rhizome_resin_field::{Field, EvalContext, Constant, Zip3, Map};
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

/// Adds two fields together.
pub struct Add<A, B> {
    a: A,
    b: B,
}

impl<I, A, B> Field<I, f32> for Add<A, B>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec2> for Add<A, B>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec3> for Add<A, B>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        self.a.sample(input.clone(), ctx) + self.b.sample(input, ctx)
    }
}

/// Multiplies two fields together.
pub struct Mul<A, B> {
    a: A,
    b: B,
}

impl<I, A, B> Field<I, f32> for Mul<A, B>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec2> for Mul<A, B>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

impl<I, A, B> Field<I, Vec3> for Mul<A, B>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        self.a.sample(input.clone(), ctx) * self.b.sample(input, ctx)
    }
}

/// Mixes two fields using a blend factor.
pub struct Mix<A, B, Blend> {
    a: A,
    b: B,
    blend: Blend,
}

impl<I, A, B, Blend> Field<I, f32> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a * (1.0 - t) + b * t
    }
}

impl<I, A, B, Blend> Field<I, Vec2> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, Vec2>,
    B: Field<I, Vec2>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec2 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.lerp(b, t)
    }
}

impl<I, A, B, Blend> Field<I, Vec3> for Mix<A, B, Blend>
where
    I: Clone,
    A: Field<I, Vec3>,
    B: Field<I, Vec3>,
    Blend: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> Vec3 {
        let t = self.blend.sample(input.clone(), ctx);
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.lerp(b, t)
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
///
/// # Example
/// ```
/// use rhizome_resin_field::{Field, EvalContext, Constant, lerp};
/// use glam::Vec2;
///
/// let a = Constant::new(0.0_f32);
/// let b = Constant::new(10.0_f32);
/// let t = Constant::new(0.25_f32);
///
/// let result = lerp::<Vec2, _, _, _>(a, b, t);
///
/// let ctx = EvalContext::new();
/// assert_eq!(result.sample(Vec2::ZERO, &ctx), 2.5);
/// ```
pub fn lerp<I, A, B, T>(
    a: A,
    b: B,
    t: T,
) -> Map<Zip3<A, B, T>, impl Fn((f32, f32, f32)) -> f32, (f32, f32, f32)>
where
    I: Clone,
    A: Field<I, f32>,
    B: Field<I, f32>,
    T: Field<I, f32>,
{
    Zip3::new(a, b, t).map(|(a, b, t)| a * (1.0 - t) + b * t)
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

/// Perlin noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin2D {
    /// Random seed.
    pub seed: i32,
}

impl Perlin2D {
    /// Create a new Perlin noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Perlin2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Perlin2D::with_seed(self.seed).sample(input.x, input.y)
    }
}

/// Perlin noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin3D {
    /// Random seed.
    pub seed: i32,
}

impl Perlin3D {
    /// Create a new Perlin noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Perlin3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise3D;
        rhizome_resin_noise::Perlin3D::with_seed(self.seed).sample(input.x, input.y, input.z)
    }
}

/// Simplex noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex2D {
    /// Random seed.
    pub seed: i32,
}

impl Simplex2D {
    /// Create a new Simplex noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Simplex2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Simplex2D::with_seed(self.seed).sample(input.x, input.y)
    }
}

/// Simplex noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex3D {
    /// Random seed.
    pub seed: i32,
}

impl Simplex3D {
    /// Create a new Simplex noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Simplex3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise3D;
        rhizome_resin_noise::Simplex3D::with_seed(self.seed).sample(input.x, input.y, input.z)
    }
}

/// White noise field (1D).
///
/// Returns uniformly distributed random values in [0, 1] with no spatial correlation.
/// Useful for scanline-based dithering effects.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhiteNoise1D {
    /// Random seed.
    pub seed: u32,
}

impl WhiteNoise1D {
    /// Create a new white noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: u32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for WhiteNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        hash_to_float_1d(input, self.seed)
    }
}

/// White noise field (2D).
///
/// Returns uniformly distributed random values in [0, 1] with no spatial correlation.
/// Simple dithering threshold - produces grainy results compared to blue noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhiteNoise2D {
    /// Random seed.
    pub seed: u32,
}

impl WhiteNoise2D {
    /// Create a new white noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: u32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for WhiteNoise2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        hash_to_float_2d(input, self.seed)
    }
}

/// White noise field (3D).
///
/// Returns uniformly distributed random values in [0, 1] with no spatial correlation.
/// Useful for temporally stable dithering in animations (z = time).
#[derive(Debug, Clone, Copy, Default)]
pub struct WhiteNoise3D {
    /// Random seed.
    pub seed: u32,
}

impl WhiteNoise3D {
    /// Create a new white noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: u32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for WhiteNoise3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        hash_to_float_3d(input, self.seed)
    }
}

// Hash functions for white noise - based on xxHash-style mixing
fn hash_to_float_1d(x: f32, seed: u32) -> f32 {
    let mut h = seed;
    h ^= x.to_bits();
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    (h & 0x00ffffff) as f32 / 0x01000000 as f32
}

fn hash_to_float_2d(p: Vec2, seed: u32) -> f32 {
    let mut h = seed;
    h ^= p.x.to_bits();
    h = h.wrapping_mul(0x85ebca6b);
    h ^= p.y.to_bits();
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 16;
    (h & 0x00ffffff) as f32 / 0x01000000 as f32
}

fn hash_to_float_3d(p: Vec3, seed: u32) -> f32 {
    let mut h = seed;
    h ^= p.x.to_bits();
    h = h.wrapping_mul(0x85ebca6b);
    h ^= p.y.to_bits();
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= p.z.to_bits();
    h = h.wrapping_mul(0x9e3779b9);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 16;
    (h & 0x00ffffff) as f32 / 0x01000000 as f32
}

/// Perlin noise field (1D).
///
/// Gradient noise for audio modulation, 1D patterns, etc.
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin1D {
    /// Random seed.
    pub seed: i32,
}

impl Perlin1D {
    /// Create a new Perlin noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for Perlin1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Perlin1D::with_seed(self.seed).sample(input)
    }
}

/// Simplex noise field (1D).
///
/// In 1D, equivalent to Perlin noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex1D {
    /// Random seed.
    pub seed: i32,
}

impl Simplex1D {
    /// Create a new Simplex noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for Simplex1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Simplex1D::with_seed(self.seed).sample(input)
    }
}

/// Value noise field (1D).
///
/// Simpler than Perlin: random values at integers, interpolated.
/// Faster but more grid-aligned artifacts.
#[derive(Debug, Clone, Copy, Default)]
pub struct Value1D {
    /// Random seed.
    pub seed: i32,
}

impl Value1D {
    /// Create a new Value noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for Value1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Value1D::with_seed(self.seed).sample(input)
    }
}

/// Value noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Value2D {
    /// Random seed.
    pub seed: i32,
}

impl Value2D {
    /// Create a new Value noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Value2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Value2D::with_seed(self.seed).sample(input.x, input.y)
    }
}

/// Value noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Value3D {
    /// Random seed.
    pub seed: i32,
}

impl Value3D {
    /// Create a new Value noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Value3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise3D;
        rhizome_resin_noise::Value3D::with_seed(self.seed).sample(input.x, input.y, input.z)
    }
}

/// Worley noise field (1D).
///
/// Distance to nearest random point on a line. Creates sawtooth patterns
/// with valleys at random intervals. Useful for random event timing,
/// tension/release in audio, non-uniform spacing.
#[derive(Debug, Clone, Copy, Default)]
pub struct Worley1D {
    /// Random seed.
    pub seed: i32,
}

impl Worley1D {
    /// Create a new Worley noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for Worley1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Worley1D::with_seed(self.seed).sample(input)
    }
}

/// Worley (cellular) noise field (2D).
///
/// Distance to nearest feature point. Creates organic cell patterns.
#[derive(Debug, Clone, Copy, Default)]
pub struct Worley2D {
    /// Random seed.
    pub seed: i32,
}

impl Worley2D {
    /// Create a new Worley noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Worley2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Worley2D::with_seed(self.seed).sample(input.x, input.y)
    }
}

/// Worley (cellular) noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Worley3D {
    /// Random seed.
    pub seed: i32,
}

impl Worley3D {
    /// Create a new Worley noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Worley3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise3D;
        rhizome_resin_noise::Worley3D::with_seed(self.seed).sample(input.x, input.y, input.z)
    }
}

/// Worley noise returning F2 (second nearest distance).
///
/// More complex cell patterns with visible boundaries.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorleyF2_2D {
    /// Random seed.
    pub seed: i32,
}

impl WorleyF2_2D {
    /// Create a new Worley F2 noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for WorleyF2_2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::{Noise2D, WorleyReturn};
        rhizome_resin_noise::Worley2D::with_seed(self.seed)
            .return_type(WorleyReturn::F2)
            .sample(input.x, input.y)
    }
}

/// Worley edge detection (F2 - F1).
///
/// Highlights cell boundaries. Good for cracked earth, scales, etc.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorleyEdge2D {
    /// Random seed.
    pub seed: i32,
}

impl WorleyEdge2D {
    /// Create a new Worley edge noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for WorleyEdge2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::{Noise2D, WorleyReturn};
        rhizome_resin_noise::Worley2D::with_seed(self.seed)
            .return_type(WorleyReturn::Edge)
            .sample(input.x, input.y)
    }
}

/// Pink noise field (1D).
///
/// Equal energy per octave (1/f spectrum). Natural-sounding variation.
/// Useful for audio modulation and organic parameter drift.
#[derive(Debug, Clone, Copy)]
pub struct PinkNoise1D {
    /// Random seed (applied via coordinate offset).
    pub seed: i32,
    /// Number of octaves (more = smoother, default 8).
    pub octaves: u32,
}

impl Default for PinkNoise1D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 8,
        }
    }
}

impl PinkNoise1D {
    /// Create a new pink noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, octaves: 8 }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }
}

impl Field<f32, f32> for PinkNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Pink1D::with_seed(self.seed)
            .octaves(self.octaves)
            .sample(input)
    }
}

/// Pink noise field (2D).
///
/// Equal energy per octave. Creates natural-looking textures.
#[derive(Debug, Clone, Copy)]
pub struct PinkNoise2D {
    /// Random seed.
    pub seed: i32,
    /// Number of octaves.
    pub octaves: u32,
}

impl Default for PinkNoise2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 8,
        }
    }
}

impl PinkNoise2D {
    /// Create a new pink noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, octaves: 8 }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }
}

impl Field<Vec2, f32> for PinkNoise2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Pink2D::with_seed(self.seed)
            .octaves(self.octaves)
            .sample(input.x, input.y)
    }
}

/// Brown (Brownian/Red) noise field (1D).
///
/// Strong low-frequency bias (1/f² spectrum). Random walk character.
/// Creates slow drifts - good for parameter modulation, deep rumble.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrownNoise1D {
    /// Random seed.
    pub seed: i32,
}

impl BrownNoise1D {
    /// Create a new brown noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for BrownNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Brown1D::with_seed(self.seed).sample(input)
    }
}

/// Brown noise field (2D).
///
/// Very smooth, low-frequency noise. Good for terrain bases.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrownNoise2D {
    /// Random seed.
    pub seed: i32,
}

impl BrownNoise2D {
    /// Create a new brown noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for BrownNoise2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        rhizome_resin_noise::Brown2D::with_seed(self.seed).sample(input.x, input.y)
    }
}

/// Violet noise field (1D).
///
/// Very high frequency (f² spectrum). Differentiated white noise.
/// Primarily useful for audio dithering of high-frequency content.
#[derive(Debug, Clone, Copy, Default)]
pub struct VioletNoise1D {
    /// Random seed.
    pub seed: i32,
}

impl VioletNoise1D {
    /// Create a new violet noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for VioletNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Violet1D::with_seed(self.seed).sample(input)
    }
}

/// Grey noise field (1D).
///
/// Psychoacoustically flat - sounds equally loud at all frequencies to human
/// ears, unlike white noise which sounds "bright". Useful for audio testing,
/// tinnitus masking, and perceptually neutral randomness.
///
/// Note: This is an approximation. True grey noise requires equal-loudness
/// contour weighting (ISO 226).
#[derive(Debug, Clone, Copy, Default)]
pub struct GreyNoise1D {
    /// Random seed.
    pub seed: i32,
}

impl GreyNoise1D {
    /// Create a new grey noise field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<f32, f32> for GreyNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Grey1D::with_seed(self.seed).sample(input)
    }
}

/// Velvet noise field (1D).
///
/// Sparse impulse noise - most samples are neutral (0.5), with occasional
/// impulses toward 0 or 1. Used in audio for:
/// - Efficient convolution reverb (sparse = fast)
/// - Decorrelation
/// - Click/impulse textures
#[derive(Debug, Clone, Copy)]
pub struct VelvetNoise1D {
    /// Random seed.
    pub seed: i32,
    /// Probability of impulse (0.0 to 1.0, typically 0.01-0.2).
    pub density: f32,
}

impl Default for VelvetNoise1D {
    fn default() -> Self {
        Self {
            seed: 0,
            density: 0.1,
        }
    }
}

impl VelvetNoise1D {
    /// Create a new velvet noise field with default 10% density.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, density: 0.1 }
    }

    /// Set the impulse density (probability of non-neutral value).
    pub fn density(mut self, density: f32) -> Self {
        self.density = density.clamp(0.0, 1.0);
        self
    }
}

impl Field<f32, f32> for VelvetNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise1D;
        rhizome_resin_noise::Velvet1D::with_seed(self.seed)
            .density(self.density)
            .sample(input)
    }
}

/// Fractal Brownian Motion field (2D).
#[derive(Debug, Clone, Copy)]
pub struct Fbm2D<F> {
    /// Base noise field.
    pub base: F,
    /// Number of octaves (layers).
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub gain: f32,
}

impl<F> Fbm2D<F> {
    /// Create a new FBM field from a base noise.
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Set the lacunarity (frequency multiplier).
    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Set the gain (amplitude multiplier).
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

impl<F: Field<Vec2, f32>> Field<Vec2, f32> for Fbm2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * self.base.sample(input * frequency, ctx);
            max_value += amplitude;
            amplitude *= self.gain;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

/// Fractal Brownian Motion field (3D).
#[derive(Debug, Clone, Copy)]
pub struct Fbm3D<F> {
    /// Base noise field.
    pub base: F,
    /// Number of octaves (layers).
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub gain: f32,
}

impl<F> Fbm3D<F> {
    /// Create a new FBM field from a base noise.
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    /// Set the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Set the lacunarity (frequency multiplier).
    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Set the gain (amplitude multiplier).
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

impl<F: Field<Vec3, f32>> Field<Vec3, f32> for Fbm3D<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * self.base.sample(input * frequency, ctx);
            max_value += amplitude;
            amplitude *= self.gain;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Terrain generation
// ============================================================================

/// Terrain heightfield generator.
///
/// Combines multiple noise octaves with configurable parameters for
/// realistic terrain generation.
#[derive(Debug, Clone, Copy)]
pub struct Terrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Amplitude multiplier between octaves.
    pub persistence: f32,
    /// Overall scale of the terrain features.
    pub scale: f32,
    /// Height exponent for terrain shaping (>1 = steeper peaks).
    pub exponent: f32,
}

impl Default for Terrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
            scale: 1.0,
            exponent: 1.0,
        }
    }
}

impl Terrain2D {
    /// Creates a new terrain generator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a preset for gentle rolling hills.
    pub fn rolling_hills() -> Self {
        Self {
            seed: 0,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.4,
            scale: 0.5,
            exponent: 0.8,
        }
    }

    /// Creates a preset for mountainous terrain.
    pub fn mountains() -> Self {
        Self {
            seed: 0,
            octaves: 8,
            lacunarity: 2.1,
            persistence: 0.55,
            scale: 1.5,
            exponent: 1.5,
        }
    }

    /// Creates a preset for flat plains with minor variation.
    pub fn plains() -> Self {
        Self {
            seed: 0,
            octaves: 3,
            lacunarity: 2.0,
            persistence: 0.3,
            scale: 0.3,
            exponent: 0.5,
        }
    }

    /// Creates a preset for canyon-like terrain.
    pub fn canyons() -> Self {
        Self {
            seed: 0,
            octaves: 5,
            lacunarity: 2.5,
            persistence: 0.6,
            scale: 1.0,
            exponent: 2.0,
        }
    }
}

impl Field<Vec2, f32> for Terrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let noise = rhizome_resin_noise::Simplex2D::with_seed(self.seed);
        let p = input * self.scale;

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += amplitude * noise.sample_signed(p.x * frequency, p.y * frequency);
            max_value += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        // Normalize to 0-1 range
        let normalized = (value / max_value + 1.0) * 0.5;

        // Apply exponent for terrain shaping
        normalized.powf(self.exponent).clamp(0.0, 1.0)
    }
}

/// Ridged noise terrain for sharp mountain ridges.
#[derive(Debug, Clone, Copy)]
pub struct RidgedTerrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Overall scale.
    pub scale: f32,
    /// Ridge sharpness (higher = sharper).
    pub sharpness: f32,
}

impl Default for RidgedTerrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 6,
            lacunarity: 2.0,
            scale: 1.0,
            sharpness: 2.0,
        }
    }
}

impl RidgedTerrain2D {
    /// Creates a new ridged terrain generator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for RidgedTerrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let noise_fn = rhizome_resin_noise::Simplex2D::with_seed(self.seed);
        let p = input * self.scale;

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut weight = 1.0;

        for _ in 0..self.octaves {
            let noise = noise_fn.sample_signed(p.x * frequency, p.y * frequency);

            // Ridged noise: abs(noise) with inversion for ridges
            let ridge = 1.0 - noise.abs();
            let ridge = ridge.powf(self.sharpness);

            value += ridge * amplitude * weight;

            // Reduce weight for successive octaves based on current value
            weight = ridge.clamp(0.0, 1.0);

            amplitude *= 0.5;
            frequency *= self.lacunarity;
        }

        value.clamp(0.0, 1.0)
    }
}

/// Billowy terrain for rounded, cloud-like hills.
#[derive(Debug, Clone, Copy)]
pub struct BillowyTerrain2D {
    /// Random seed.
    pub seed: i32,
    /// Number of noise octaves.
    pub octaves: u32,
    /// Frequency multiplier between octaves.
    pub lacunarity: f32,
    /// Overall scale.
    pub scale: f32,
}

impl Default for BillowyTerrain2D {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 5,
            lacunarity: 2.0,
            scale: 1.0,
        }
    }
}

impl BillowyTerrain2D {
    /// Creates a new billowy terrain generator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for BillowyTerrain2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let noise_fn = rhizome_resin_noise::Simplex2D::with_seed(self.seed);
        let p = input * self.scale;

        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            let noise = noise_fn.sample_signed(p.x * frequency, p.y * frequency);

            // Billowy: abs(noise) gives rounded peaks
            value += noise.abs() * amplitude;
            max_value += amplitude;

            amplitude *= 0.5;
            frequency *= self.lacunarity;
        }

        (value / max_value).clamp(0.0, 1.0)
    }
}

/// Island terrain generator with falloff at edges.
#[derive(Debug, Clone, Copy)]
pub struct IslandTerrain2D {
    /// Base terrain noise.
    pub terrain: Terrain2D,
    /// Falloff radius (island size).
    pub radius: f32,
    /// Center of the island.
    pub center: Vec2,
    /// Falloff sharpness.
    pub falloff: f32,
}

impl Default for IslandTerrain2D {
    fn default() -> Self {
        Self {
            terrain: Terrain2D::default(),
            radius: 1.0,
            center: Vec2::ZERO,
            falloff: 2.0,
        }
    }
}

impl IslandTerrain2D {
    /// Creates a new island terrain generator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for IslandTerrain2D {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let dist = (input - self.center).length() / self.radius;

        // Smooth falloff using smoothstep-like curve
        let falloff = if dist >= 1.0 {
            0.0
        } else {
            let t = 1.0 - dist;
            let s = t.powf(self.falloff);
            s
        };

        let terrain_height = self.terrain.sample(input, ctx);

        (terrain_height * falloff).clamp(0.0, 1.0)
    }
}

/// Terraced terrain for stepped plateaus.
#[derive(Debug, Clone, Copy)]
pub struct TerracedTerrain2D<F> {
    /// Base terrain field.
    pub base: F,
    /// Number of terrace levels.
    pub levels: u32,
    /// Terrace sharpness (0=smooth, 1=sharp).
    pub sharpness: f32,
}

impl<F> TerracedTerrain2D<F> {
    /// Creates a new terraced terrain.
    pub fn new(base: F, levels: u32) -> Self {
        Self {
            base,
            levels,
            sharpness: 0.8,
        }
    }

    /// Sets the terrace sharpness.
    pub fn with_sharpness(mut self, sharpness: f32) -> Self {
        self.sharpness = sharpness.clamp(0.0, 1.0);
        self
    }
}

impl<F: Field<Vec2, f32>> Field<Vec2, f32> for TerracedTerrain2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let base_value = self.base.sample(input, ctx);

        // Quantize to terrace levels
        let scaled = base_value * self.levels as f32;
        let terrace = scaled.floor();
        let frac = scaled - terrace;

        // Smooth the transition between terraces
        let smoothed_frac = if self.sharpness >= 1.0 {
            0.0
        } else {
            // Smoothstep-like transition
            let t = frac / (1.0 - self.sharpness);
            if t >= 1.0 {
                1.0
            } else {
                t * t * (3.0 - 2.0 * t)
            }
        };

        ((terrace + smoothed_frac * (1.0 - self.sharpness)) / self.levels as f32).clamp(0.0, 1.0)
    }
}

/// Gradient field - returns gradient based on coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Gradient2D {
    /// Start value.
    pub start: f32,
    /// End value.
    pub end: f32,
    /// Gradient direction.
    pub direction: Vec2,
}

impl Gradient2D {
    /// Create a horizontal gradient (left to right).
    pub fn horizontal() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::X,
        }
    }

    /// Create a vertical gradient (bottom to top).
    pub fn vertical() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::Y,
        }
    }

    /// Create a radial gradient.
    pub fn radial(center: Vec2) -> Radial2D {
        Radial2D {
            center,
            radius: 1.0,
        }
    }
}

impl Field<Vec2, f32> for Gradient2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction);
        (self.start + t * (self.end - self.start)).clamp(0.0, 1.0)
    }
}

/// Radial gradient field.
#[derive(Debug, Clone, Copy)]
pub struct Radial2D {
    /// Center point.
    pub center: Vec2,
    /// Radius of the gradient.
    pub radius: f32,
}

impl Radial2D {
    /// Create a new radial gradient.
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Field<Vec2, f32> for Radial2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let d = (input - self.center).length();
        (1.0 - d / self.radius).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Texture patterns
// ============================================================================

/// Checkerboard pattern.
#[derive(Debug, Clone, Copy)]
pub struct Checkerboard {
    /// Pattern scale.
    pub scale: f32,
}

impl Default for Checkerboard {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl Checkerboard {
    /// Create a new checkerboard pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific scale.
    pub fn with_scale(scale: f32) -> Self {
        Self { scale }
    }
}

impl Field<Vec2, f32> for Checkerboard {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        if (x + y) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

impl Field<Vec3, f32> for Checkerboard {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        let z = p.z.floor() as i32;
        if (x + y + z) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

/// Stripe pattern (horizontal by default).
#[derive(Debug, Clone, Copy)]
pub struct Stripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for Stripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl Stripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    /// Set the direction (will be normalized).
    pub fn with_direction(mut self, direction: Vec2) -> Self {
        self.direction = direction.normalize();
        self
    }
}

impl Field<Vec2, f32> for Stripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency;
        if (t.floor() as i32) % 2 == 0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Smooth stripe pattern (sine wave).
#[derive(Debug, Clone, Copy)]
pub struct SmoothStripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for SmoothStripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl SmoothStripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }
}

impl Field<Vec2, f32> for SmoothStripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency * std::f32::consts::TAU;
        (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// Brick pattern.
#[derive(Debug, Clone, Copy)]
pub struct Brick {
    /// Brick size.
    pub scale: Vec2,
    /// Mortar width.
    pub mortar: f32,
    /// Row offset (0.5 for standard brick).
    pub offset: f32,
}

impl Default for Brick {
    fn default() -> Self {
        Self {
            scale: Vec2::new(2.0, 1.0),
            mortar: 0.05,
            offset: 0.5,
        }
    }
}

impl Brick {
    /// Create a new brick pattern.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for Brick {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let row = p.y.floor() as i32;
        let x = if row % 2 == 0 { p.x } else { p.x + self.offset };

        let fx = x.fract();
        let fy = p.y.fract();

        // Check if in mortar
        if fx < self.mortar || fx > 1.0 - self.mortar || fy < self.mortar || fy > 1.0 - self.mortar
        {
            0.0
        } else {
            1.0
        }
    }
}

/// Polka dots pattern.
#[derive(Debug, Clone, Copy)]
pub struct Dots {
    /// Grid scale.
    pub scale: f32,
    /// Dot radius.
    pub radius: f32,
}

impl Default for Dots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.3,
        }
    }
}

impl Dots {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for Dots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        if d < self.radius { 1.0 } else { 0.0 }
    }
}

/// Smooth dots (falloff from center).
#[derive(Debug, Clone, Copy)]
pub struct SmoothDots {
    pub scale: f32,
    pub radius: f32,
}

impl Default for SmoothDots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.4,
        }
    }
}

impl SmoothDots {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for SmoothDots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        (1.0 - d / self.radius).clamp(0.0, 1.0)
    }
}

/// Voronoi/Cellular noise - returns distance to nearest cell center.
#[derive(Debug, Clone, Copy)]
pub struct Voronoi {
    pub scale: f32,
    pub seed: i32,
}

impl Default for Voronoi {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl Voronoi {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash function for cell randomization.
    pub fn hash(x: i32, y: i32, seed: i32) -> Vec2 {
        // Simple hash based on prime multipliers
        let n = (x.wrapping_mul(374761393))
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed.wrapping_mul(1013904223));
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        let fx = ((n & 0xFFFF) as f32) / 65535.0;
        let fy = (((n >> 16) & 0xFFFF) as f32) / 65535.0;
        Vec2::new(fx, fy)
    }
}

impl Field<Vec2, f32> for Voronoi {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;

        // Check 3x3 neighborhood
        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Self::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                min_dist = min_dist.min(dist);
            }
        }

        // Normalize roughly to 0-1 (max distance in a cell is ~sqrt(2))
        (min_dist / 1.5).clamp(0.0, 1.0)
    }
}

/// Voronoi that returns cell ID (for coloring).
#[derive(Debug, Clone, Copy)]
pub struct VoronoiId {
    pub scale: f32,
    pub seed: i32,
}

impl Default for VoronoiId {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl VoronoiId {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for VoronoiId {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;
        let mut closest_cell = cell;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Voronoi::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                if dist < min_dist {
                    min_dist = dist;
                    closest_cell = neighbor;
                }
            }
        }

        // Return a pseudo-random value based on cell coordinates
        let h = Voronoi::hash(
            closest_cell.x as i32,
            closest_cell.y as i32,
            self.seed + 12345,
        );
        h.x
    }
}

/// Domain warping - distorts input coordinates using another field.
pub struct Warp<F, D> {
    pub field: F,
    pub displacement: D,
    pub amount: f32,
}

impl<F, D> Warp<F, D> {
    pub fn new(field: F, displacement: D, amount: f32) -> Self {
        Self {
            field,
            displacement,
            amount,
        }
    }
}

impl<F, D> Field<Vec2, f32> for Warp<F, D>
where
    F: Field<Vec2, f32>,
    D: Field<Vec2, Vec2>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let offset = self.displacement.sample(input, ctx);
        let warped = input + offset * self.amount;
        self.field.sample(warped, ctx)
    }
}

/// Creates displacement from two scalar fields (for x and y).
pub struct Displacement<X, Y> {
    pub x: X,
    pub y: Y,
}

impl<X, Y> Displacement<X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        Self { x, y }
    }
}

impl<X, Y> Field<Vec2, Vec2> for Displacement<X, Y>
where
    X: Field<Vec2, f32>,
    Y: Field<Vec2, f32>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> Vec2 {
        Vec2::new(self.x.sample(input, ctx), self.y.sample(input, ctx))
    }
}

/// Distance field - returns distance from a point.
#[derive(Debug, Clone, Copy)]
pub struct DistancePoint {
    pub point: Vec2,
}

impl DistancePoint {
    pub fn new(point: Vec2) -> Self {
        Self { point }
    }
}

impl Field<Vec2, f32> for DistancePoint {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.point).length()
    }
}

/// Distance field - returns distance from a line segment.
#[derive(Debug, Clone, Copy)]
pub struct DistanceLine {
    pub a: Vec2,
    pub b: Vec2,
}

impl DistanceLine {
    pub fn new(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }
}

impl Field<Vec2, f32> for DistanceLine {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let pa = input - self.a;
        let ba = self.b - self.a;
        let t = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
        (pa - ba * t).length()
    }
}

/// Distance field - returns distance from a circle.
#[derive(Debug, Clone, Copy)]
pub struct DistanceCircle {
    pub center: Vec2,
    pub radius: f32,
}

impl DistanceCircle {
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Field<Vec2, f32> for DistanceCircle {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.center).length() - self.radius
    }
}

/// Distance field - returns distance from a box.
#[derive(Debug, Clone, Copy)]
pub struct DistanceBox {
    pub center: Vec2,
    pub half_size: Vec2,
}

impl DistanceBox {
    pub fn new(center: Vec2, half_size: Vec2) -> Self {
        Self { center, half_size }
    }
}

impl Field<Vec2, f32> for DistanceBox {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = (input - self.center).abs();
        let d = p - self.half_size;
        d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
    }
}

/// Distance field - returns distance from a rounded box.
#[derive(Debug, Clone, Copy)]
pub struct DistanceRoundedBox {
    /// Center of the box.
    pub center: Vec2,
    /// Half-size of the box (before rounding).
    pub half_size: Vec2,
    /// Corner radius.
    pub radius: f32,
}

impl DistanceRoundedBox {
    /// Creates a new rounded box SDF.
    pub fn new(center: Vec2, half_size: Vec2, radius: f32) -> Self {
        Self {
            center,
            half_size,
            radius,
        }
    }
}

impl Field<Vec2, f32> for DistanceRoundedBox {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = (input - self.center).abs();
        let q = p - self.half_size + Vec2::splat(self.radius);
        q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - self.radius
    }
}

/// Distance field - returns distance from an ellipse.
#[derive(Debug, Clone, Copy)]
pub struct DistanceEllipse {
    /// Center of the ellipse.
    pub center: Vec2,
    /// Radii (half-width, half-height).
    pub radii: Vec2,
}

impl DistanceEllipse {
    /// Creates a new ellipse SDF.
    pub fn new(center: Vec2, radii: Vec2) -> Self {
        Self { center, radii }
    }
}

impl Field<Vec2, f32> for DistanceEllipse {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Approximate ellipse SDF using normalized space
        let p = (input - self.center).abs();
        let ab = self.radii;

        if ab.x == ab.y {
            // Circle case
            return p.length() - ab.x;
        }

        // Iterative Newton-Raphson for accurate ellipse SDF
        let mut t = 0.25 * std::f32::consts::PI;
        for _ in 0..4 {
            let c = t.cos();
            let s = t.sin();
            let e = Vec2::new(ab.x * c, ab.y * s);
            let d = p - e;
            let g = Vec2::new(-ab.x * s, ab.y * c);
            let dt = d.dot(g) / g.dot(g);
            t = (t + dt).clamp(0.0, std::f32::consts::FRAC_PI_2);
        }

        let closest = Vec2::new(ab.x * t.cos(), ab.y * t.sin());
        let dist = (p - closest).length();

        // Determine sign (inside or outside)
        let normalized = p / ab;
        if normalized.length_squared() < 1.0 {
            -dist
        } else {
            dist
        }
    }
}

/// Distance field - returns distance from a capsule (stadium shape).
#[derive(Debug, Clone, Copy)]
pub struct DistanceCapsule {
    /// First endpoint of the capsule centerline.
    pub a: Vec2,
    /// Second endpoint of the capsule centerline.
    pub b: Vec2,
    /// Capsule radius.
    pub radius: f32,
}

impl DistanceCapsule {
    /// Creates a new capsule SDF.
    pub fn new(a: Vec2, b: Vec2, radius: f32) -> Self {
        Self { a, b, radius }
    }
}

impl Field<Vec2, f32> for DistanceCapsule {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let pa = input - self.a;
        let ba = self.b - self.a;
        let t = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
        (pa - ba * t).length() - self.radius
    }
}

/// Distance field - returns distance from a triangle.
#[derive(Debug, Clone, Copy)]
pub struct DistanceTriangle {
    /// First vertex.
    pub a: Vec2,
    /// Second vertex.
    pub b: Vec2,
    /// Third vertex.
    pub c: Vec2,
}

impl DistanceTriangle {
    /// Creates a new triangle SDF.
    pub fn new(a: Vec2, b: Vec2, c: Vec2) -> Self {
        Self { a, b, c }
    }
}

impl Field<Vec2, f32> for DistanceTriangle {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input;

        // Edge vectors and point-to-vertex vectors
        let e0 = self.b - self.a;
        let e1 = self.c - self.b;
        let e2 = self.a - self.c;
        let v0 = p - self.a;
        let v1 = p - self.b;
        let v2 = p - self.c;

        // Perpendicular vectors
        let pq0 = v0 - e0 * (v0.dot(e0) / e0.dot(e0)).clamp(0.0, 1.0);
        let pq1 = v1 - e1 * (v1.dot(e1) / e1.dot(e1)).clamp(0.0, 1.0);
        let pq2 = v2 - e2 * (v2.dot(e2) / e2.dot(e2)).clamp(0.0, 1.0);

        // Signed distance
        let s = (e0.x * e2.y - e0.y * e2.x).signum();
        let d = (pq0.dot(pq0).min(pq1.dot(pq1)).min(pq2.dot(pq2))).sqrt();

        // Determine if inside or outside
        let c0 = s * (v0.x * e0.y - v0.y * e0.x);
        let c1 = s * (v1.x * e1.y - v1.y * e1.x);
        let c2 = s * (v2.x * e2.y - v2.y * e2.x);

        if c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0 {
            -d
        } else {
            d
        }
    }
}

/// Distance field - returns distance from a regular polygon.
#[derive(Debug, Clone, Copy)]
pub struct DistanceRegularPolygon {
    /// Center of the polygon.
    pub center: Vec2,
    /// Circumradius (distance from center to vertex).
    pub radius: f32,
    /// Number of sides (3 = triangle, 4 = square, 6 = hexagon, etc.).
    pub sides: u32,
}

impl DistanceRegularPolygon {
    /// Creates a new regular polygon SDF.
    pub fn new(center: Vec2, radius: f32, sides: u32) -> Self {
        Self {
            center,
            radius,
            sides: sides.max(3),
        }
    }
}

impl Field<Vec2, f32> for DistanceRegularPolygon {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use std::f32::consts::PI;

        let p = input - self.center;
        let n = self.sides as f32;

        // Angle between vertices
        let an = PI / n;

        // Convert to polar, reduce to one sector
        let angle = p.y.atan2(p.x);
        let sector_angle = ((angle + PI) / (2.0 * an)).floor() * 2.0 * an - PI + an;

        // Rotate point into canonical sector
        let cs = sector_angle.cos();
        let sn = sector_angle.sin();
        let q = Vec2::new(p.x * cs + p.y * sn, -p.x * sn + p.y * cs).abs();

        // Distance to edge
        let edge_dist = q.x - self.radius * an.cos();
        let corner_dist = (q - Vec2::new(self.radius * an.cos(), self.radius * an.sin())).length();

        if q.y > self.radius * an.sin() {
            corner_dist
        } else {
            edge_dist
        }
    }
}

/// Distance field - returns distance from a pie/arc shape.
#[derive(Debug, Clone, Copy)]
pub struct DistanceArc {
    /// Center of the arc.
    pub center: Vec2,
    /// Radius of the arc.
    pub radius: f32,
    /// Half-angle of the arc in radians.
    pub half_angle: f32,
    /// Thickness of the arc (0 for just the arc curve).
    pub thickness: f32,
}

impl DistanceArc {
    /// Creates a new arc/pie SDF.
    pub fn new(center: Vec2, radius: f32, half_angle: f32, thickness: f32) -> Self {
        Self {
            center,
            radius,
            half_angle,
            thickness,
        }
    }

    /// Creates a pie shape (filled arc).
    pub fn pie(center: Vec2, radius: f32, half_angle: f32) -> Self {
        Self {
            center,
            radius,
            half_angle,
            thickness: radius, // Fill to center
        }
    }
}

impl Field<Vec2, f32> for DistanceArc {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input - self.center;

        // Use symmetry around x-axis
        let p = Vec2::new(p.x, p.y.abs());
        let r = p.length();

        // Angle of point from positive x-axis
        let angle = p.y.atan2(p.x);

        // For a pie/sector shape (thickness >= radius means filled to center):
        // - Inside: angle <= half_angle AND r <= radius
        // - Distance is to the nearest boundary

        if angle <= self.half_angle {
            // Inside angular range
            if self.thickness >= self.radius {
                // Pie mode: distance to arc (outer boundary)
                r - self.radius
            } else {
                // Arc mode: distance to thick arc boundary
                let radial_dist = (r - self.radius).abs();
                radial_dist - self.thickness
            }
        } else {
            // Outside angular range - distance to edge line
            // Edge direction from center
            let edge_dir = Vec2::new(self.half_angle.cos(), self.half_angle.sin());

            // Project p onto edge direction (line from origin along edge_dir)
            let proj_length = p.dot(edge_dir).max(0.0).min(self.radius);
            let proj_point = edge_dir * proj_length;
            let dist_to_edge = (p - proj_point).length();

            if self.thickness >= self.radius {
                // Pie mode: just distance to edge line
                dist_to_edge
            } else {
                dist_to_edge - self.thickness
            }
        }
    }
}

// ============================================================================
// SDF operations
// ============================================================================

/// Union of two SDFs (min).
pub struct SdfUnion<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfUnion<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.min(b)
    }
}

/// Intersection of two SDFs (max).
pub struct SdfIntersection<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfIntersection<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(b)
    }
}

/// Subtraction of SDF B from A.
pub struct SdfSubtraction<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfSubtraction<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(-b)
    }
}

/// Smooth union of two SDFs using polynomial smooth min.
pub struct SdfSmoothUnion<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothUnion<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_min(a, b, self.k)
    }
}

/// Smooth intersection of two SDFs.
pub struct SdfSmoothIntersection<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothIntersection<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, b, self.k)
    }
}

/// Smooth subtraction of SDF B from A.
pub struct SdfSmoothSubtraction<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothSubtraction<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, -b, self.k)
    }
}

/// Polynomial smooth minimum.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth maximum.
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// Rounds/expands an SDF by a radius.
pub struct SdfRound<F> {
    pub field: F,
    pub radius: f32,
}

impl<F> SdfRound<F> {
    pub fn new(field: F, radius: f32) -> Self {
        Self { field, radius }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfRound<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx) - self.radius
    }
}

/// Annular (ring/shell) version of an SDF.
pub struct SdfAnnular<F> {
    pub field: F,
    pub thickness: f32,
}

impl<F> SdfAnnular<F> {
    pub fn new(field: F, thickness: f32) -> Self {
        Self { field, thickness }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfAnnular<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).abs() - self.thickness
    }
}

// ============================================================================
// Domain modifiers
// ============================================================================

/// Twists space around the Y axis (for 3D fields).
pub struct Twist<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Twist<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Twist<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.y * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let twisted = Vec3::new(
            input.x * cos - input.z * sin,
            input.y,
            input.x * sin + input.z * cos,
        );
        self.field.sample(twisted, ctx)
    }
}

/// Bends space around the Y axis (for 3D fields).
pub struct Bend<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Bend<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Bend<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.x * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let bent = Vec3::new(
            cos * input.x - sin * input.y,
            sin * input.x + cos * input.y,
            input.z,
        );
        self.field.sample(bent, ctx)
    }
}

/// Repeats space infinitely.
pub struct Repeat<F> {
    pub field: F,
    pub period: Vec2,
}

impl<F> Repeat<F> {
    pub fn new(field: F, period: Vec2) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Repeat<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let repeated = Vec2::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Repeats space infinitely (3D).
pub struct Repeat3D<F> {
    pub field: F,
    pub period: Vec3,
}

impl<F> Repeat3D<F> {
    pub fn new(field: F, period: Vec3) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Repeat3D<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let repeated = Vec3::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
            ((input.z % self.period.z) + self.period.z) % self.period.z - self.period.z * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Rotates 2D input coordinates.
pub struct Rotate2D<F> {
    pub field: F,
    pub angle: f32,
}

impl<F> Rotate2D<F> {
    pub fn new(field: F, angle: f32) -> Self {
        Self { field, angle }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Rotate2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let cos = self.angle.cos();
        let sin = self.angle.sin();
        let rotated = Vec2::new(input.x * cos - input.y * sin, input.x * sin + input.y * cos);
        self.field.sample(rotated, ctx)
    }
}

/// Mirrors space across an axis.
pub struct Mirror<F> {
    pub field: F,
    pub axis: Vec2,
}

impl<F> Mirror<F> {
    pub fn new(field: F, axis: Vec2) -> Self {
        Self {
            field,
            axis: axis.normalize(),
        }
    }

    pub fn x(field: F) -> Self {
        Self::new(field, Vec2::X)
    }

    pub fn y(field: F) -> Self {
        Self::new(field, Vec2::Y)
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Mirror<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let d = input.dot(self.axis);
        let mirrored = if d < 0.0 {
            input - 2.0 * d * self.axis
        } else {
            input
        };
        self.field.sample(mirrored, ctx)
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

// ============================================================================
// Metaballs
// ============================================================================

/// A single metaball (blob) center for use in metaball fields.
#[derive(Debug, Clone, Copy)]
pub struct Metaball {
    /// Center position.
    pub center: Vec3,
    /// Radius of influence.
    pub radius: f32,
    /// Strength (default 1.0).
    pub strength: f32,
}

impl Metaball {
    /// Creates a new metaball at the given position with the given radius.
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            strength: 1.0,
        }
    }

    /// Creates a 2D metaball (z=0).
    pub fn new_2d(center: Vec2, radius: f32) -> Self {
        Self::new(center.extend(0.0), radius)
    }
}

/// 2D metaball field - computes the sum of influences at each point.
///
/// The field value is the sum of `strength * f(distance)` for each ball,
/// where f is the falloff function. Values above 1.0 are typically "inside"
/// the merged surface.
///
/// # Example
///
/// ```
/// use glam::Vec2;
/// use rhizome_resin_field::{Field, EvalContext, Metaball, Metaballs2D};
///
/// let balls = vec![
///     Metaball::new_2d(Vec2::new(0.0, 0.0), 1.0),
///     Metaball::new_2d(Vec2::new(1.5, 0.0), 1.0),
/// ];
///
/// let field = Metaballs2D::new(balls);
/// let ctx = EvalContext::new();
///
/// // Sample the field - values > 1.0 are "inside"
/// let value = field.sample(Vec2::new(0.75, 0.0), &ctx);
/// ```
#[derive(Debug, Clone)]
pub struct Metaballs2D {
    balls: Vec<Metaball>,
    threshold: f32,
}

impl Metaballs2D {
    /// Creates a new 2D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec2, f32> for Metaballs2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let ball_pos = Vec2::new(ball.center.x, ball.center.y);
            let dist_sq = (input - ball_pos).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                // Very close to center - return large value
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// 3D metaball field.
///
/// Similar to Metaballs2D but operates in 3D space.
#[derive(Debug, Clone)]
pub struct Metaballs3D {
    balls: Vec<Metaball>,
    threshold: f32,
}

impl Metaballs3D {
    /// Creates a new 3D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec3, f32> for Metaballs3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let dist_sq = (input - ball.center).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// Converts a 2D metaball field to an SDF-like representation.
///
/// Returns negative values inside (where field > threshold),
/// positive values outside. This makes it compatible with SDF
/// operations like smooth union.
#[derive(Debug, Clone)]
pub struct MetaballSdf2D {
    field: Metaballs2D,
}

impl MetaballSdf2D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs2D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs2D::new(balls))
    }
}

impl Field<Vec2, f32> for MetaballSdf2D {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        // Convert to SDF-like: negative inside, positive outside
        // When value > threshold, we're inside, so return negative
        self.field.threshold - value
    }
}

/// Converts a 3D metaball field to an SDF-like representation.
#[derive(Debug, Clone)]
pub struct MetaballSdf3D {
    field: Metaballs3D,
}

impl MetaballSdf3D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs3D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs3D::new(balls))
    }
}

impl Field<Vec3, f32> for MetaballSdf3D {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        self.field.threshold - value
    }
}

// ============================================================================
// Terrain erosion simulation
// ============================================================================

/// A heightmap for terrain erosion simulation.
///
/// Stores height values as a 2D grid.
#[derive(Clone)]
pub struct Heightmap {
    /// Height values (row-major).
    pub data: Vec<f32>,
    /// Width of the heightmap.
    pub width: usize,
    /// Height of the heightmap.
    pub height: usize,
}

impl Heightmap {
    /// Creates a new heightmap filled with zeros.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Creates a heightmap from a Field.
    pub fn from_field<F: Field<Vec2, f32>>(
        field: &F,
        width: usize,
        height: usize,
        scale: f32,
    ) -> Self {
        let ctx = EvalContext::new();
        let mut data = vec![0.0; width * height];

        for y in 0..height {
            for x in 0..width {
                let pos = Vec2::new(x as f32 * scale, y as f32 * scale);
                data[y * width + x] = field.sample(pos, &ctx);
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Gets the height at grid coordinates.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Sets the height at grid coordinates.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Gets the height with bilinear interpolation.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let h00 = self.get_clamped(x0, y0);
        let h10 = self.get_clamped(x0 + 1, y0);
        let h01 = self.get_clamped(x0, y0 + 1);
        let h11 = self.get_clamped(x0 + 1, y0 + 1);

        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;

        h0 * (1.0 - fy) + h1 * fy
    }

    /// Gets height with boundary clamping.
    fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Computes the gradient at a position.
    pub fn gradient(&self, x: f32, y: f32) -> Vec2 {
        let h = 0.5;
        let dx = (self.sample(x + h, y) - self.sample(x - h, y)) / (2.0 * h);
        let dy = (self.sample(x, y + h) - self.sample(x, y - h)) / (2.0 * h);
        Vec2::new(dx, dy)
    }

    /// Adds sediment at a position (with bilinear distribution).
    fn add_sediment(&mut self, x: f32, y: f32, amount: f32) {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        // Bilinear weights
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        self.add_clamped(x0, y0, amount * w00);
        self.add_clamped(x0 + 1, y0, amount * w10);
        self.add_clamped(x0, y0 + 1, amount * w01);
        self.add_clamped(x0 + 1, y0 + 1, amount * w11);
    }

    /// Adds to height with boundary clamping.
    fn add_clamped(&mut self, x: i32, y: i32, amount: f32) {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            let idx = y as usize * self.width + x as usize;
            self.data[idx] += amount;
        }
    }

    /// Gets the minimum and maximum heights.
    pub fn bounds(&self) -> (f32, f32) {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Normalizes heights to 0-1 range.
    pub fn normalize(&mut self) {
        let (min, max) = self.bounds();
        let range = max - min;
        if range > 0.0001 {
            for h in &mut self.data {
                *h = (*h - min) / range;
            }
        }
    }
}

/// Hydraulic erosion simulation operation.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Heightmap))]
pub struct HydraulicErosion {
    /// Number of water droplets to simulate.
    pub iterations: usize,
    /// Maximum steps per droplet lifetime.
    pub max_lifetime: usize,
    /// Initial water volume per droplet.
    pub initial_water: f32,
    /// Initial speed of droplets.
    pub initial_speed: f32,
    /// Inertia factor (0 = follow gradient exactly, 1 = maintain direction).
    pub inertia: f32,
    /// Minimum slope for erosion to occur.
    pub min_slope: f32,
    /// Sediment capacity multiplier.
    pub capacity_factor: f32,
    /// Rate of sediment pickup.
    pub erosion_rate: f32,
    /// Rate of sediment deposition.
    pub deposition_rate: f32,
    /// Water evaporation rate per step.
    pub evaporation_rate: f32,
    /// Gravity strength.
    pub gravity: f32,
    /// Erosion brush radius.
    pub brush_radius: usize,
    /// Random seed for simulation.
    pub seed: u64,
}

impl Default for HydraulicErosion {
    fn default() -> Self {
        Self {
            iterations: 10000,
            max_lifetime: 64,
            initial_water: 1.0,
            initial_speed: 1.0,
            inertia: 0.3,
            min_slope: 0.01,
            capacity_factor: 4.0,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            evaporation_rate: 0.01,
            gravity: 4.0,
            brush_radius: 3,
            seed: 0,
        }
    }
}

impl HydraulicErosion {
    /// Applies this erosion operation to a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Heightmap {
        let mut result = heightmap.clone();
        hydraulic_erosion(&mut result, self, self.seed);
        result
    }
}

/// Backwards-compatible type alias.
pub type HydraulicErosionConfig = HydraulicErosion;

/// Simulates hydraulic erosion on a heightmap.
///
/// Water droplets flow downhill, picking up sediment and depositing it
/// as they slow down, creating realistic river valleys and gullies.
pub fn hydraulic_erosion(heightmap: &mut Heightmap, config: &HydraulicErosion, seed: u64) {
    let mut rng = SimpleRng::new(seed);
    let width = heightmap.width;
    let height = heightmap.height;

    // Pre-compute erosion brush weights
    let brush = compute_erosion_brush(config.brush_radius);

    for _ in 0..config.iterations {
        // Random starting position
        let mut x = rng.next_f32() * (width - 1) as f32;
        let mut y = rng.next_f32() * (height - 1) as f32;

        // Initial direction (random)
        let angle = rng.next_f32() * std::f32::consts::TAU;
        let mut dir = Vec2::new(angle.cos(), angle.sin());

        let mut speed = config.initial_speed;
        let mut water = config.initial_water;
        let mut sediment = 0.0;

        for _ in 0..config.max_lifetime {
            let xi = x as usize;
            let yi = y as usize;

            // Get gradient at current position
            let gradient = heightmap.gradient(x, y);
            let gradient_len = gradient.length();

            // Update direction with inertia
            if gradient_len > 0.0001 {
                dir = dir * config.inertia - gradient.normalize() * (1.0 - config.inertia);
                let dir_len = dir.length();
                if dir_len > 0.0001 {
                    dir /= dir_len;
                }
            }

            // Move to next position
            let new_x = x + dir.x;
            let new_y = y + dir.y;

            // Check boundaries
            if new_x < 0.0
                || new_x >= (width - 1) as f32
                || new_y < 0.0
                || new_y >= (height - 1) as f32
            {
                break;
            }

            // Calculate height difference
            let old_height = heightmap.sample(x, y);
            let new_height = heightmap.sample(new_x, new_y);
            let height_diff = new_height - old_height;

            // Calculate sediment capacity
            let slope = (-height_diff).max(config.min_slope);
            let capacity = slope * speed * water * config.capacity_factor;

            if sediment > capacity || height_diff > 0.0 {
                // Deposit sediment
                let deposit_amount = if height_diff > 0.0 {
                    // Moving uphill - deposit all sediment
                    height_diff.min(sediment)
                } else {
                    // Deposit excess sediment
                    (sediment - capacity) * config.deposition_rate
                };

                sediment -= deposit_amount;
                heightmap.add_sediment(x, y, deposit_amount);
            } else {
                // Erode terrain
                let erode_amount = ((capacity - sediment) * config.erosion_rate).min(-height_diff);

                // Apply erosion with brush
                apply_erosion_brush(heightmap, xi, yi, erode_amount, &brush);
                sediment += erode_amount;
            }

            // Update physics
            speed = (speed * speed + height_diff.abs() * config.gravity)
                .sqrt()
                .max(0.1);
            water *= 1.0 - config.evaporation_rate;

            // Stop if water dried up
            if water < 0.01 {
                break;
            }

            x = new_x;
            y = new_y;
        }
    }
}

/// Brush weights for erosion.
struct ErosionBrush {
    offsets: Vec<(i32, i32)>,
    weights: Vec<f32>,
}

fn compute_erosion_brush(radius: usize) -> ErosionBrush {
    let mut offsets = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    let r = radius as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist <= radius as f32 {
                let weight = 1.0 - dist / (radius as f32 + 1.0);
                offsets.push((dx, dy));
                weights.push(weight);
                total_weight += weight;
            }
        }
    }

    // Normalize weights
    if total_weight > 0.0 {
        for w in &mut weights {
            *w /= total_weight;
        }
    }

    ErosionBrush { offsets, weights }
}

fn apply_erosion_brush(
    heightmap: &mut Heightmap,
    cx: usize,
    cy: usize,
    amount: f32,
    brush: &ErosionBrush,
) {
    for (i, (dx, dy)) in brush.offsets.iter().enumerate() {
        let x = cx as i32 + dx;
        let y = cy as i32 + dy;
        if x >= 0 && x < heightmap.width as i32 && y >= 0 && y < heightmap.height as i32 {
            let idx = y as usize * heightmap.width + x as usize;
            heightmap.data[idx] -= amount * brush.weights[i];
        }
    }
}

/// Thermal erosion simulation operation.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Heightmap))]
pub struct ThermalErosion {
    /// Number of iterations.
    pub iterations: usize,
    /// Maximum slope angle (as tangent) before material slides.
    pub talus_angle: f32,
    /// Rate of material transfer per iteration.
    pub transfer_rate: f32,
}

impl Default for ThermalErosion {
    fn default() -> Self {
        Self {
            iterations: 50,
            talus_angle: 0.8, // ~40 degrees
            transfer_rate: 0.5,
        }
    }
}

impl ThermalErosion {
    /// Applies this erosion operation to a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Heightmap {
        let mut result = heightmap.clone();
        thermal_erosion(&mut result, self);
        result
    }
}

/// Backwards-compatible type alias.
pub type ThermalErosionConfig = ThermalErosion;

/// Simulates thermal erosion (talus slopes).
///
/// Material slides down when slopes exceed the talus angle,
/// creating realistic scree slopes and smoothing sharp features.
pub fn thermal_erosion(heightmap: &mut Heightmap, config: &ThermalErosion) {
    let width = heightmap.width;
    let height = heightmap.height;

    // Neighbor offsets
    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    // Distance for each neighbor (diagonal = sqrt(2))
    let distances: [f32; 8] = [
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
    ];

    let mut deltas = vec![0.0; width * height];

    for _ in 0..config.iterations {
        // Clear deltas
        deltas.fill(0.0);

        // Calculate transfers
        for y in 0..height {
            for x in 0..width {
                let h = heightmap.get(x, y);
                let mut total_diff: f32 = 0.0;
                let mut max_diff: f32 = 0.0;

                // Find total height difference to lower neighbors
                for (i, (dx, dy)) in neighbors.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nh = heightmap.get(nx as usize, ny as usize);
                        let slope = (h - nh) / distances[i];

                        if slope > config.talus_angle {
                            let diff = slope - config.talus_angle;
                            total_diff += diff;
                            max_diff = max_diff.max(diff);
                        }
                    }
                }

                if total_diff > 0.0 {
                    // Distribute material to lower neighbors
                    let transfer_amount = max_diff * config.transfer_rate * 0.5;

                    for (i, (dx, dy)) in neighbors.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nh = heightmap.get(nx as usize, ny as usize);
                            let slope = (h - nh) / distances[i];

                            if slope > config.talus_angle {
                                let diff = slope - config.talus_angle;
                                let ratio = diff / total_diff;
                                let amount = transfer_amount * ratio;

                                deltas[y * width + x] -= amount;
                                deltas[ny as usize * width + nx as usize] += amount;
                            }
                        }
                    }
                }
            }
        }

        // Apply deltas
        for (i, delta) in deltas.iter().enumerate() {
            heightmap.data[i] += delta;
        }
    }
}

/// Simple random number generator for erosion simulation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

// ============================================================================
// Road/River Networks
// ============================================================================

/// A node in a network (intersection, city, water source).
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Position in 2D space.
    pub position: Vec2,
    /// Optional height for terrain integration.
    pub height: f32,
    /// Node importance (affects connectivity priority).
    pub importance: f32,
}

impl NetworkNode {
    /// Creates a new network node.
    pub fn new(position: Vec2) -> Self {
        Self {
            position,
            height: 0.0,
            importance: 1.0,
        }
    }
}

/// An edge connecting two nodes with a path.
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    /// Index of start node.
    pub start: usize,
    /// Index of end node.
    pub end: usize,
    /// Intermediate path points.
    pub path: Vec<Vec2>,
    /// Edge weight (distance, cost, etc.).
    pub weight: f32,
    /// Edge type (road width, river flow, etc.).
    pub edge_type: f32,
}

impl NetworkEdge {
    /// Creates a new edge.
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            path: Vec::new(),
            weight: 0.0,
            edge_type: 1.0,
        }
    }
}

/// A network of connected nodes and edges.
#[derive(Debug, Clone)]
pub struct Network {
    /// Nodes in the network.
    pub nodes: Vec<NetworkNode>,
    /// Edges connecting nodes.
    pub edges: Vec<NetworkEdge>,
}

impl Network {
    /// Creates an empty network.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a node and returns its index.
    pub fn add_node(&mut self, node: NetworkNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(&mut self, start: usize, end: usize) -> &mut NetworkEdge {
        let edge = NetworkEdge::new(start, end);
        self.edges.push(edge);
        self.edges.last_mut().unwrap()
    }

    /// Returns all edges connected to a node.
    pub fn edges_for_node(&self, node_idx: usize) -> Vec<&NetworkEdge> {
        self.edges
            .iter()
            .filter(|e| e.start == node_idx || e.end == node_idx)
            .collect()
    }

    /// Returns neighboring node indices.
    pub fn neighbors(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.start == node_idx {
                    Some(e.end)
                } else if e.end == node_idx {
                    Some(e.start)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the full path for an edge including endpoints.
    pub fn edge_path(&self, edge: &NetworkEdge) -> Vec<Vec2> {
        let start_pos = self.nodes[edge.start].position;
        let end_pos = self.nodes[edge.end].position;

        let mut full_path = vec![start_pos];
        full_path.extend(edge.path.iter().cloned());
        full_path.push(end_pos);
        full_path
    }

    /// Samples all edge paths at regular intervals.
    pub fn sample_edges(&self, segment_length: f32) -> Vec<Vec<Vec2>> {
        self.edges
            .iter()
            .map(|edge| {
                let path = self.edge_path(edge);
                sample_path(&path, segment_length)
            })
            .collect()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// Samples a path at regular intervals.
fn sample_path(path: &[Vec2], segment_length: f32) -> Vec<Vec2> {
    if path.len() < 2 {
        return path.to_vec();
    }

    let mut result = vec![path[0]];
    let mut accumulated = 0.0;

    for window in path.windows(2) {
        let start = window[0];
        let end = window[1];
        let seg_len = (end - start).length();
        let dir = (end - start).normalize_or_zero();

        let mut pos = 0.0;
        while pos < seg_len {
            let remaining = segment_length - accumulated;
            if pos + remaining <= seg_len {
                pos += remaining;
                result.push(start + dir * pos);
                accumulated = 0.0;
            } else {
                accumulated += seg_len - pos;
                break;
            }
        }
    }

    if result.last() != path.last() {
        result.push(*path.last().unwrap());
    }

    result
}

/// Road network generation operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Network))]
pub struct RoadNetwork {
    /// Number of cities/intersections.
    pub num_nodes: usize,
    /// Area bounds (min_x, min_y, max_x, max_y).
    pub bounds: (f32, f32, f32, f32),
    /// Whether to generate minimum spanning tree first.
    pub use_mst: bool,
    /// Extra connections beyond MST (0.0 = none, 1.0 = full).
    pub extra_connectivity: f32,
    /// Number of path relaxation iterations.
    pub relaxation_iterations: usize,
    /// Path curvature amount.
    pub curvature: f32,
    /// Random seed for generation.
    pub seed: u64,
}

impl Default for RoadNetwork {
    fn default() -> Self {
        Self {
            num_nodes: 10,
            bounds: (0.0, 0.0, 100.0, 100.0),
            use_mst: true,
            extra_connectivity: 0.2,
            relaxation_iterations: 3,
            curvature: 0.3,
            seed: 0,
        }
    }
}

impl RoadNetwork {
    /// Applies this operation to generate a road network.
    pub fn apply(&self) -> Network {
        generate_road_network(self, self.seed)
    }
}

/// Backwards-compatible type alias.
pub type RoadNetworkConfig = RoadNetwork;

/// Generates a road network using random node placement and MST.
pub fn generate_road_network(config: &RoadNetwork, seed: u64) -> Network {
    let mut rng = SimpleRng::new(seed);
    let mut network = Network::new();

    // Generate random nodes
    let (min_x, min_y, max_x, max_y) = config.bounds;
    for _ in 0..config.num_nodes {
        let x = min_x + rng.next_f32() * (max_x - min_x);
        let y = min_y + rng.next_f32() * (max_y - min_y);
        network.add_node(NetworkNode::new(Vec2::new(x, y)));
    }

    if network.nodes.len() < 2 {
        return network;
    }

    if config.use_mst {
        // Build MST using Prim's algorithm
        build_mst(&mut network);
    }

    // Add extra connections
    if config.extra_connectivity > 0.0 {
        add_extra_connections(&mut network, config.extra_connectivity, &mut rng);
    }

    // Relax paths for natural curves
    for edge in &mut network.edges {
        relax_edge_path(
            &network.nodes,
            edge,
            config.relaxation_iterations,
            config.curvature,
        );
    }

    network
}

/// Builds a minimum spanning tree connecting all nodes.
fn build_mst(network: &mut Network) {
    let n = network.nodes.len();
    if n < 2 {
        return;
    }

    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f32::MAX; n];
    let mut min_edge = vec![0usize; n];

    // Start from node 0
    in_tree[0] = true;

    // Initialize distances from node 0
    for i in 1..n {
        let d = (network.nodes[i].position - network.nodes[0].position).length();
        min_dist[i] = d;
        min_edge[i] = 0;
    }

    // Add n-1 edges
    for _ in 0..n - 1 {
        // Find closest node not in tree
        let mut best_dist = f32::MAX;
        let mut best_node = 0;

        for i in 0..n {
            if !in_tree[i] && min_dist[i] < best_dist {
                best_dist = min_dist[i];
                best_node = i;
            }
        }

        if best_dist == f32::MAX {
            break; // No more reachable nodes
        }

        // Add edge to tree
        in_tree[best_node] = true;
        network.add_edge(min_edge[best_node], best_node).weight = best_dist;

        // Update distances
        for i in 0..n {
            if !in_tree[i] {
                let d = (network.nodes[i].position - network.nodes[best_node].position).length();
                if d < min_dist[i] {
                    min_dist[i] = d;
                    min_edge[i] = best_node;
                }
            }
        }
    }
}

/// Adds extra connections beyond the MST.
fn add_extra_connections(network: &mut Network, connectivity: f32, rng: &mut SimpleRng) {
    let n = network.nodes.len();
    let existing_edges: std::collections::HashSet<(usize, usize)> = network
        .edges
        .iter()
        .flat_map(|e| [(e.start, e.end), (e.end, e.start)])
        .collect();

    // Calculate how many extra edges to add
    let max_extra = n * (n - 1) / 2 - network.edges.len();
    let num_extra = (max_extra as f32 * connectivity) as usize;

    // Collect possible edges sorted by distance
    let mut possible: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if !existing_edges.contains(&(i, j)) {
                let d = (network.nodes[j].position - network.nodes[i].position).length();
                possible.push((i, j, d));
            }
        }
    }

    // Sort by distance
    possible.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Add edges with probability favoring shorter ones
    let mut added = 0;
    for (i, j, d) in possible {
        if added >= num_extra {
            break;
        }
        // Higher probability for shorter edges
        let prob = 1.0 / (1.0 + d * 0.1);
        if rng.next_f32() < prob {
            network.add_edge(i, j).weight = d;
            added += 1;
        }
    }
}

/// Relaxes an edge path for natural curves.
fn relax_edge_path(
    nodes: &[NetworkNode],
    edge: &mut NetworkEdge,
    iterations: usize,
    curvature: f32,
) {
    if iterations == 0 || curvature <= 0.0 {
        return;
    }

    let start = nodes[edge.start].position;
    let end = nodes[edge.end].position;
    let dist = (end - start).length();

    // Add intermediate points
    let num_points = (dist / 10.0).max(2.0) as usize;
    edge.path = (1..num_points)
        .map(|i| {
            let t = i as f32 / num_points as f32;
            start.lerp(end, t)
        })
        .collect();

    // Perturb and relax
    let perp = Vec2::new(-(end - start).y, (end - start).x).normalize_or_zero();
    for (i, p) in edge.path.iter_mut().enumerate() {
        let t = (i + 1) as f32 / (num_points) as f32;
        let wave = (t * std::f32::consts::PI).sin() * curvature * dist * 0.1;
        *p += perp * wave;
    }

    // Smooth the path
    for _ in 0..iterations {
        let path_clone = edge.path.clone();
        for i in 0..edge.path.len() {
            let prev = if i == 0 { start } else { path_clone[i - 1] };
            let next = if i == edge.path.len() - 1 {
                end
            } else {
                path_clone[i + 1]
            };
            edge.path[i] = (prev + next) * 0.5;
        }
    }
}

/// River network generation operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Network))]
pub struct RiverNetwork {
    /// Number of river sources.
    pub num_sources: usize,
    /// Minimum height for sources.
    pub source_min_height: f32,
    /// Number of steps per river.
    pub max_steps: usize,
    /// Step size for gradient descent.
    pub step_size: f32,
    /// Whether to merge rivers that meet.
    pub merge_rivers: bool,
    /// Random seed for generation.
    pub seed: u64,
}

impl Default for RiverNetwork {
    fn default() -> Self {
        Self {
            num_sources: 3,
            source_min_height: 0.7,
            max_steps: 200,
            step_size: 1.0,
            merge_rivers: true,
            seed: 0,
        }
    }
}

impl RiverNetwork {
    /// Applies this operation to generate a river network from a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Network {
        generate_river_network(heightmap, self, self.seed)
    }
}

/// Backwards-compatible type alias.
pub type RiverNetworkConfig = RiverNetwork;

/// Generates a river network by following terrain gradients.
pub fn generate_river_network(heightmap: &Heightmap, config: &RiverNetwork, seed: u64) -> Network {
    let mut rng = SimpleRng::new(seed);
    let mut network = Network::new();

    // Find source points (high elevation)
    let mut sources: Vec<Vec2> = Vec::new();
    let (width, height) = (heightmap.width, heightmap.height);

    for _ in 0..config.num_sources * 10 {
        if sources.len() >= config.num_sources {
            break;
        }

        let x = rng.next_f32() * (width - 2) as f32 + 1.0;
        let y = rng.next_f32() * (height - 2) as f32 + 1.0;
        let h = heightmap.sample(x, y);

        // Accept high points not too close to existing sources
        if h >= config.source_min_height {
            let too_close = sources
                .iter()
                .any(|s| (*s - Vec2::new(x, y)).length() < 10.0);
            if !too_close {
                sources.push(Vec2::new(x, y));
            }
        }
    }

    // Trace each river
    let mut all_paths: Vec<Vec<Vec2>> = Vec::new();

    for source in &sources {
        let path = trace_river_path(heightmap, *source, config);
        if path.len() >= 2 {
            all_paths.push(path);
        }
    }

    // Build network from paths
    if config.merge_rivers {
        build_merged_river_network(&mut network, &all_paths);
    } else {
        for path in all_paths {
            if path.len() >= 2 {
                let start_idx = network.add_node(NetworkNode::new(path[0]));
                let end_idx = network.add_node(NetworkNode::new(*path.last().unwrap()));
                network.add_edge(start_idx, end_idx).path = path[1..path.len() - 1].to_vec();
            }
        }
    }

    network
}

/// Traces a river path following the steepest descent.
fn trace_river_path(heightmap: &Heightmap, start: Vec2, config: &RiverNetworkConfig) -> Vec<Vec2> {
    let mut path = vec![start];
    let mut pos = start;
    let (width, height) = (heightmap.width as f32, heightmap.height as f32);

    for _ in 0..config.max_steps {
        // Check bounds
        if pos.x <= 1.0 || pos.x >= width - 2.0 || pos.y <= 1.0 || pos.y >= height - 2.0 {
            break;
        }

        // Find steepest descent
        let current_h = heightmap.sample(pos.x, pos.y);
        let mut best_dir = Vec2::ZERO;
        let mut best_drop = 0.0;

        // Sample in 8 directions
        for angle_idx in 0..8 {
            let angle = angle_idx as f32 * std::f32::consts::TAU / 8.0;
            let dir = Vec2::new(angle.cos(), angle.sin());
            let sample_pos = pos + dir * config.step_size;

            if sample_pos.x > 0.0
                && sample_pos.x < width - 1.0
                && sample_pos.y > 0.0
                && sample_pos.y < height - 1.0
            {
                let sample_h = heightmap.sample(sample_pos.x, sample_pos.y);
                let drop = current_h - sample_h;
                if drop > best_drop {
                    best_drop = drop;
                    best_dir = dir;
                }
            }
        }

        // Stop if no downhill direction
        if best_drop <= 0.0 {
            break;
        }

        // Move in best direction
        pos += best_dir * config.step_size;
        path.push(pos);

        // Stop if reached low ground
        let h = heightmap.sample(pos.x, pos.y);
        if h < 0.05 {
            break;
        }
    }

    path
}

/// Builds a river network that merges rivers at intersection points.
fn build_merged_river_network(network: &mut Network, paths: &[Vec<Vec2>]) {
    let merge_threshold = 3.0;
    let mut node_positions: Vec<Vec2> = Vec::new();

    // Find or create node at position
    let find_or_create_node = |network: &mut Network, pos: Vec2, positions: &mut Vec<Vec2>| {
        for (i, &p) in positions.iter().enumerate() {
            if (p - pos).length() < merge_threshold {
                return i;
            }
        }
        let idx = network.add_node(NetworkNode::new(pos));
        positions.push(pos);
        idx
    };

    for path in paths {
        if path.len() < 2 {
            continue;
        }

        let mut prev_node = find_or_create_node(network, path[0], &mut node_positions);
        let mut segment_path: Vec<Vec2> = Vec::new();

        for &point in &path[1..] {
            // Check if this point is near an existing node
            let mut near_node = None;
            for (i, &p) in node_positions.iter().enumerate() {
                if (p - point).length() < merge_threshold && i != prev_node {
                    near_node = Some(i);
                    break;
                }
            }

            if let Some(node_idx) = near_node {
                // Create edge to existing node
                if prev_node != node_idx {
                    network.add_edge(prev_node, node_idx).path = segment_path.clone();
                }
                prev_node = node_idx;
                segment_path.clear();
            } else {
                segment_path.push(point);
            }
        }

        // Final segment
        let end_pos = *path.last().unwrap();
        let end_node = find_or_create_node(network, end_pos, &mut node_positions);
        if prev_node != end_node {
            network.add_edge(prev_node, end_node).path = segment_path;
        }
    }
}

/// Computes accumulated flow for each point in a heightmap.
/// Higher values indicate more upstream area draining through that point.
pub fn compute_flow_accumulation(heightmap: &Heightmap) -> Heightmap {
    let (width, height) = (heightmap.width, heightmap.height);
    let mut flow = Heightmap::new(width, height);

    // Initialize with 1 (each cell contributes its own area)
    for i in 0..flow.data.len() {
        flow.data[i] = 1.0;
    }

    // Sort cells by height (highest first)
    let mut cells: Vec<(usize, usize, f32)> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            cells.push((x, y, heightmap.get(x, y)));
        }
    }
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Process cells from high to low
    for (x, y, _) in cells {
        let h = heightmap.get(x, y);
        let cell_flow = flow.get(x, y);

        // Find lowest neighbor
        let mut best_drop = 0.0;
        let mut best_x = x;
        let mut best_y = y;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nh = heightmap.get(nx as usize, ny as usize);
                    let drop = h - nh;
                    if drop > best_drop {
                        best_drop = drop;
                        best_x = nx as usize;
                        best_y = ny as usize;
                    }
                }
            }
        }

        // Transfer flow to lowest neighbor
        if best_drop > 0.0 && (best_x != x || best_y != y) {
            let current = flow.get(best_x, best_y);
            flow.set(best_x, best_y, current + cell_flow);
        }
    }

    flow
}

/// Registers all field operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of field ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
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
        let field = <Constant<f32> as Field<Vec2, f32>>::add(a, b);
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
        let result = super::lerp::<Vec2, _, _, _>(a, b, t);
        let ctx = EvalContext::new();

        assert_eq!(result.sample(Vec2::ZERO, &ctx), 2.5);
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
/// cargo test -p rhizome-resin-field --features invariant-tests
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

    /// Add combinator should sum outputs correctly.
    #[test]
    fn test_add_combinator() {
        let ctx = EvalContext::new();
        let a = Constant::<f32>::new(0.3);
        let b = Constant::<f32>::new(0.4);

        let sum = Add { a, b };
        let v = sum.sample(Vec2::ZERO, &ctx);

        assert!((v - 0.7).abs() < 0.001, "Add: expected 0.7, got {v}");
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
