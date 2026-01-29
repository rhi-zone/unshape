use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

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
