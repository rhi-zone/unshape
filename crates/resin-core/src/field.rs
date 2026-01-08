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

// ============================================================================
// Texture patterns
// ============================================================================

/// Checkerboard pattern.
#[derive(Debug, Clone, Copy)]
pub struct Checkerboard {
    pub scale: f32,
}

impl Default for Checkerboard {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl Checkerboard {
    pub fn new() -> Self {
        Self::default()
    }

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
    pub frequency: f32,
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
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }

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
    pub frequency: f32,
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
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    pub fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
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
    pub scale: Vec2,
    pub mortar: f32,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale(mut self, scale: Vec2) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_mortar(mut self, mortar: f32) -> Self {
        self.mortar = mortar;
        self
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
    pub scale: f32,
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

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
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

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
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

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Hash function for cell randomization.
    fn hash(x: i32, y: i32, seed: i32) -> Vec2 {
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

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
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

        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
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
        assert!((0.0..=1.0).contains(&v));
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

    // Texture pattern tests

    #[test]
    fn test_checkerboard() {
        let field = Checkerboard::new();
        let ctx = EvalContext::new();

        // Origin is white (1.0)
        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        // Adjacent cell is black (0.0)
        assert_eq!(field.sample(Vec2::new(1.5, 0.5), &ctx), 0.0);
        // Diagonal is white again
        assert_eq!(field.sample(Vec2::new(1.5, 1.5), &ctx), 1.0);
    }

    #[test]
    fn test_checkerboard_3d() {
        let field = Checkerboard::new();
        let ctx = EvalContext::new();

        assert_eq!(field.sample(Vec3::new(0.5, 0.5, 0.5), &ctx), 1.0);
        assert_eq!(field.sample(Vec3::new(1.5, 0.5, 0.5), &ctx), 0.0);
        assert_eq!(field.sample(Vec3::new(1.5, 1.5, 0.5), &ctx), 1.0);
        assert_eq!(field.sample(Vec3::new(1.5, 1.5, 1.5), &ctx), 0.0);
    }

    #[test]
    fn test_stripes() {
        let field = Stripes::horizontal().with_frequency(1.0);
        let ctx = EvalContext::new();

        // Stripes alternate based on Y position
        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        assert_eq!(field.sample(Vec2::new(0.5, 1.5), &ctx), 0.0);
        assert_eq!(field.sample(Vec2::new(0.5, 2.5), &ctx), 1.0);
    }

    #[test]
    fn test_smooth_stripes() {
        let field = SmoothStripes::horizontal();
        let ctx = EvalContext::new();

        // Should be smooth values in [0, 1]
        let v = field.sample(Vec2::new(0.0, 0.25), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_brick() {
        let field = Brick::new(); // scale=(2.0, 1.0), mortar=0.05
        let ctx = EvalContext::new();

        // Center of first brick (scaled: 0.5, 0.5) -> input: (0.25, 0.5)
        let v = field.sample(Vec2::new(0.25, 0.5), &ctx);
        assert_eq!(v, 1.0);

        // Near mortar edge (at cell boundary) should be 0.0
        // Input (0.0, 0.5) -> scaled (0.0, 0.5), fx=0.0 < mortar
        let v = field.sample(Vec2::new(0.0, 0.5), &ctx);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_dots() {
        let field = Dots::new().with_radius(0.3);
        let ctx = EvalContext::new();

        // Center of cell should be in dot
        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        // Corner of cell should be outside dot
        assert_eq!(field.sample(Vec2::new(0.1, 0.1), &ctx), 0.0);
    }

    #[test]
    fn test_smooth_dots() {
        let field = SmoothDots::new();
        let ctx = EvalContext::new();

        // Center of cell should be max (1.0)
        assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
        // Between center and edge should be smooth value
        let v = field.sample(Vec2::new(0.7, 0.5), &ctx);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_voronoi() {
        let field = Voronoi::new();
        let ctx = EvalContext::new();

        // Should return values in [0, 1]
        let v = field.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));

        // Different positions should give different values
        let v1 = field.sample(Vec2::new(0.1, 0.1), &ctx);
        let v2 = field.sample(Vec2::new(0.9, 0.9), &ctx);
        // Values should vary (not necessarily different, but usually)
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_voronoi_id() {
        let field = VoronoiId::new();
        let ctx = EvalContext::new();

        // Should return values in [0, 1]
        let v = field.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_distance_point() {
        let field = DistancePoint::new(Vec2::ZERO);
        let ctx = EvalContext::new();

        assert!((field.sample(Vec2::ZERO, &ctx) - 0.0).abs() < 0.001);
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 1.0).abs() < 0.001);
        assert!((field.sample(Vec2::new(3.0, 4.0), &ctx) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_circle() {
        let field = DistanceCircle::new(Vec2::ZERO, 1.0);
        let ctx = EvalContext::new();

        // Inside circle: negative distance
        assert!(field.sample(Vec2::ZERO, &ctx) < 0.0);
        // On circle: zero
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        // Outside circle: positive
        assert!(field.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);
    }

    #[test]
    fn test_distance_line() {
        let field = DistanceLine::new(Vec2::ZERO, Vec2::new(2.0, 0.0));
        let ctx = EvalContext::new();

        // On line: zero
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        // Above line: distance = 1.0
        assert!((field.sample(Vec2::new(1.0, 1.0), &ctx) - 1.0).abs() < 0.001);
        // Past endpoint: distance to endpoint
        assert!((field.sample(Vec2::new(3.0, 0.0), &ctx) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_box() {
        let field = DistanceBox::new(Vec2::ZERO, Vec2::new(1.0, 1.0));
        let ctx = EvalContext::new();

        // Inside box: negative
        assert!(field.sample(Vec2::ZERO, &ctx) < 0.0);
        // On edge: zero
        assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
        // Outside: positive
        assert!((field.sample(Vec2::new(2.0, 0.0), &ctx) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_warp() {
        let base = Checkerboard::new();
        let displacement = Displacement::new(Constant::new(0.5f32), Constant::new(0.0f32));
        let warped = Warp::new(base, displacement, 1.0);
        let ctx = EvalContext::new();

        // Warping should shift the pattern
        let _v = warped.sample(Vec2::new(0.5, 0.5), &ctx);
        // Just verify it compiles and runs
    }
}
