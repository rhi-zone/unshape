use glam::Vec2;

use crate::{EvalContext, Field};

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
        use unshape_noise::Noise1D;
        unshape_noise::Pink1D::with_seed(self.seed)
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
        use unshape_noise::Noise2D;
        unshape_noise::Pink2D::with_seed(self.seed)
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
        use unshape_noise::Noise1D;
        unshape_noise::Brown1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise2D;
        unshape_noise::Brown2D::with_seed(self.seed).sample(input.x, input.y)
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
        use unshape_noise::Noise1D;
        unshape_noise::Violet1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise1D;
        unshape_noise::Grey1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise1D;
        unshape_noise::Velvet1D::with_seed(self.seed)
            .density(self.density)
            .sample(input)
    }
}
