use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

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
        use unshape_noise::Noise1D;
        unshape_noise::Perlin1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise2D;
        unshape_noise::Perlin2D::with_seed(self.seed).sample(input.x, input.y)
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
        use unshape_noise::Noise3D;
        unshape_noise::Perlin3D::with_seed(self.seed).sample(input.x, input.y, input.z)
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
        use unshape_noise::Noise1D;
        unshape_noise::Simplex1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise2D;
        unshape_noise::Simplex2D::with_seed(self.seed).sample(input.x, input.y)
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
        use unshape_noise::Noise3D;
        unshape_noise::Simplex3D::with_seed(self.seed).sample(input.x, input.y, input.z)
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
pub(crate) fn hash_to_float_1d(x: f32, seed: u32) -> f32 {
    let mut h = seed;
    h ^= x.to_bits();
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    (h & 0x00ffffff) as f32 / 0x01000000 as f32
}

pub(crate) fn hash_to_float_2d(p: Vec2, seed: u32) -> f32 {
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

pub(crate) fn hash_to_float_3d(p: Vec3, seed: u32) -> f32 {
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
        use unshape_noise::Noise1D;
        unshape_noise::Value1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise2D;
        unshape_noise::Value2D::with_seed(self.seed).sample(input.x, input.y)
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
        use unshape_noise::Noise3D;
        unshape_noise::Value3D::with_seed(self.seed).sample(input.x, input.y, input.z)
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
        use unshape_noise::Noise1D;
        unshape_noise::Worley1D::with_seed(self.seed).sample(input)
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
        use unshape_noise::Noise2D;
        unshape_noise::Worley2D::with_seed(self.seed).sample(input.x, input.y)
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
        use unshape_noise::Noise3D;
        unshape_noise::Worley3D::with_seed(self.seed).sample(input.x, input.y, input.z)
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
        use unshape_noise::{Noise2D, WorleyReturn};
        unshape_noise::Worley2D::with_seed(self.seed)
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
        use unshape_noise::{Noise2D, WorleyReturn};
        unshape_noise::Worley2D::with_seed(self.seed)
            .return_type(WorleyReturn::Edge)
            .sample(input.x, input.y)
    }
}
