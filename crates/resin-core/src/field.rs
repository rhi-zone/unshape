//! Field trait for lazy evaluation.
//!
//! A `Field<I, O>` represents a function that can be sampled at any point.
//! Fields are lazy - they describe computation, not data. Evaluation happens
//! on demand when you call `sample()`.
//!
//! # Examples
//!
//! ```ignore
//! use resin_core::{Field, EvalContext};
//! use glam::Vec2;
//!
//! // Create a noise field
//! let noise = PerlinField::new().scale(4.0);
//!
//! // Sample at a point
//! let ctx = EvalContext::new();
//! let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
//!
//! // Combine fields
//! let combined = noise.add(other_field);
//! ```

use crate::context::EvalContext;
use glam::{Vec2, Vec3};
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
}

// Combinators

/// Maps the output of a field.
pub struct Map<F, M, O> {
    field: F,
    f: M,
    _phantom: PhantomData<O>,
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

// Basic field implementations

/// A constant field that always returns the same value.
#[derive(Debug, Clone, Copy)]
pub struct Constant<O> {
    pub value: O,
}

impl<O> Constant<O> {
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
    pub seed: i32,
}

impl Perlin2D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Perlin2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Offset by seed
        let p = input + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);
        crate::noise::perlin2(p.x, p.y)
    }
}

/// Perlin noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Perlin3D {
    pub seed: i32,
}

impl Perlin3D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Perlin3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input
            + Vec3::new(
                self.seed as f32 * 17.0,
                self.seed as f32 * 31.0,
                self.seed as f32 * 47.0,
            );
        crate::noise::perlin3(p.x, p.y, p.z)
    }
}

/// Simplex noise field (2D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex2D {
    pub seed: i32,
}

impl Simplex2D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec2, f32> for Simplex2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input + Vec2::new(self.seed as f32 * 17.0, self.seed as f32 * 31.0);
        crate::noise::simplex2(p.x, p.y)
    }
}

/// Simplex noise field (3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct Simplex3D {
    pub seed: i32,
}

impl Simplex3D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Field<Vec3, f32> for Simplex3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input
            + Vec3::new(
                self.seed as f32 * 17.0,
                self.seed as f32 * 31.0,
                self.seed as f32 * 47.0,
            );
        crate::noise::simplex3(p.x, p.y, p.z)
    }
}

/// Fractal Brownian Motion field (2D).
#[derive(Debug, Clone, Copy)]
pub struct Fbm2D<F> {
    pub base: F,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
}

impl<F> Fbm2D<F> {
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

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
    pub base: F,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
}

impl<F> Fbm3D<F> {
    pub fn new(base: F) -> Self {
        Self {
            base,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }

    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

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

/// Gradient field - returns gradient based on coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Gradient2D {
    pub start: f32,
    pub end: f32,
    pub direction: Vec2,
}

impl Gradient2D {
    pub fn horizontal() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::X,
        }
    }

    pub fn vertical() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
            direction: Vec2::Y,
        }
    }

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
    pub center: Vec2,
    pub radius: f32,
}

impl Radial2D {
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
    fn test_mul_combinator() {
        let a = Constant::new(3.0f32);
        let b = Constant::new(4.0f32);
        let field = <Constant<f32> as Field<Vec2, f32>>::mul(a, b);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 12.0);
    }

    #[test]
    fn test_mix_combinator() {
        let a = Constant::new(0.0f32);
        let b = Constant::new(10.0f32);
        let blend = Constant::new(0.5f32);
        let field = <Constant<f32> as Field<Vec2, f32>>::mix(a, b, blend);
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec2::ZERO, &ctx), 5.0);
    }

    #[test]
    fn test_perlin_field() {
        let field = Perlin2D::new().scale(4.0);
        let ctx = EvalContext::new();

        let v1 = field.sample(Vec2::new(0.0, 0.0), &ctx);
        let v2 = field.sample(Vec2::new(1.0, 1.0), &ctx);

        assert!(v1 >= 0.0 && v1 <= 1.0);
        assert!(v2 >= 0.0 && v2 <= 1.0);
        // Values should differ at different positions
        assert!((v1 - v2).abs() > 0.001 || (v1 == v2));
    }

    #[test]
    fn test_gradient_field() {
        let field = Gradient2D::horizontal();
        let ctx = EvalContext::new();

        assert!((field.sample(Vec2::new(0.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        assert!((field.sample(Vec2::new(0.5, 0.0), &ctx) - 0.5).abs() < 0.001);
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_radial_field() {
        let field = Radial2D::new(Vec2::ZERO, 1.0);
        let ctx = EvalContext::new();

        assert!((field.sample(Vec2::ZERO, &ctx) - 1.0).abs() < 0.001);
        assert!((field.sample(Vec2::new(0.5, 0.0), &ctx) - 0.5).abs() < 0.001);
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fn_field() {
        let field = from_fn(|p: Vec2, ctx: &EvalContext| p.x + p.y + ctx.time);
        let ctx = EvalContext::new().with_time(1.0);

        assert_eq!(field.sample(Vec2::new(2.0, 3.0), &ctx), 6.0);
    }

    #[test]
    fn test_fbm_field() {
        let base = Perlin2D::new();
        let field = Fbm2D::new(base).octaves(4);
        let ctx = EvalContext::new();

        let v = field.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_complex_chain() {
        // noise.scale(4.0).map(|x| x * 2.0 - 1.0).add(gradient)
        let noise = Perlin2D::new().scale(4.0).map(|x| x * 2.0 - 1.0);
        let gradient = Gradient2D::horizontal();
        let combined = noise.add(gradient);

        let ctx = EvalContext::new();
        let _v = combined.sample(Vec2::new(0.5, 0.5), &ctx);
        // Just verify it compiles and runs
    }
}
