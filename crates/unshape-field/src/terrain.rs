use glam::Vec2;

use crate::{EvalContext, Field};

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
        use unshape_noise::Noise2D;
        let noise = unshape_noise::Simplex2D::with_seed(self.seed);
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
        use unshape_noise::Noise2D;
        let noise_fn = unshape_noise::Simplex2D::with_seed(self.seed);
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
        use unshape_noise::Noise2D;
        let noise_fn = unshape_noise::Simplex2D::with_seed(self.seed);
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
