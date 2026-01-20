//! Noise functions for procedural generation.
//!
//! Provides classic noise algorithms used across all domains:
//! textures, mesh displacement, audio modulation, etc.
//!
//! # Struct-based API
//!
//! All noise types are structs that can be configured and sampled:
//!
//! ```
//! use rhizome_resin_noise::{Perlin2D, Noise2D};
//!
//! let noise = Perlin2D::new();
//! let value = noise.sample(1.5, 2.5);
//!
//! // With custom seed
//! let seeded = Perlin2D::with_seed(42);
//! ```
//!
//! # Composing Noise
//!
//! Use [`Fbm`] for fractal Brownian motion:
//!
//! ```
//! use rhizome_resin_noise::{Perlin2D, Fbm, Noise2D};
//!
//! let fbm = Fbm::new(Perlin2D::new())
//!     .octaves(4)
//!     .lacunarity(2.0)
//!     .persistence(0.5);
//!
//! let value = fbm.sample(1.0, 2.0);
//! ```

use glam::{Vec2, Vec3};

// =============================================================================
// Noise Traits
// =============================================================================

/// Trait for 1D noise functions.
pub trait Noise1D {
    /// Sample the noise at position x.
    fn sample(&self, x: f32) -> f32;

    /// Sample the noise at position x, returning value in [-1, 1] range.
    fn sample_signed(&self, x: f32) -> f32 {
        self.sample(x) * 2.0 - 1.0
    }
}

/// Trait for 2D noise functions.
pub trait Noise2D {
    /// Sample the noise at position (x, y).
    fn sample(&self, x: f32, y: f32) -> f32;

    /// Sample the noise at position p.
    fn sample_vec(&self, p: Vec2) -> f32 {
        self.sample(p.x, p.y)
    }

    /// Sample the noise, returning value in [-1, 1] range.
    fn sample_signed(&self, x: f32, y: f32) -> f32 {
        self.sample(x, y) * 2.0 - 1.0
    }
}

/// Trait for 3D noise functions.
pub trait Noise3D {
    /// Sample the noise at position (x, y, z).
    fn sample(&self, x: f32, y: f32, z: f32) -> f32;

    /// Sample the noise at position p.
    fn sample_vec(&self, p: Vec3) -> f32 {
        self.sample(p.x, p.y, p.z)
    }

    /// Sample the noise, returning value in [-1, 1] range.
    fn sample_signed(&self, x: f32, y: f32, z: f32) -> f32 {
        self.sample(x, y, z) * 2.0 - 1.0
    }
}

// =============================================================================
// Internal: Permutation table and helpers
// =============================================================================

/// Permutation table for noise functions.
/// Classic permutation from Ken Perlin's reference implementation.
const PERM: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

#[inline]
fn perm(x: i32, seed: i32) -> u8 {
    PERM[((x.wrapping_add(seed)) & 255) as usize]
}

#[inline]
fn grad1(hash: u8, x: f32) -> f32 {
    if hash & 1 != 0 { -x } else { x }
}

#[inline]
fn grad2(hash: u8, x: f32, y: f32) -> f32 {
    let h = hash & 7;
    let u = if h < 4 { x } else { y };
    let v = if h < 4 { y } else { x };
    (if h & 1 != 0 { -u } else { u }) + (if h & 2 != 0 { -2.0 * v } else { 2.0 * v })
}

#[inline]
fn grad3(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 != 0 { -u } else { u }) + (if h & 2 != 0 { -v } else { v })
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// Simplex noise helpers
const F2: f32 = 0.5 * (1.732_050_8 - 1.0); // (sqrt(3) - 1) / 2
const G2: f32 = (3.0 - 1.732_050_8) / 6.0; // (3 - sqrt(3)) / 6
const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

// =============================================================================
// Perlin Noise
// =============================================================================

/// 1D Perlin (gradient) noise.
///
/// Classic gradient noise with smooth interpolation.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Perlin1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Perlin1D {
    /// Creates a new Perlin noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Perlin noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Perlin1D {
    fn sample(&self, x: f32) -> f32 {
        let xi = x.floor() as i32;
        let xf = x - x.floor();
        let u = fade(xf);

        let a = perm(xi, self.seed);
        let b = perm(xi + 1, self.seed);

        (lerp(grad1(a, xf), grad1(b, xf - 1.0), u) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// 2D Perlin (gradient) noise.
///
/// Classic gradient noise with smooth interpolation.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Perlin2D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Perlin2D {
    /// Creates a new Perlin noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Perlin noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise2D for Perlin2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        let xf = x - x.floor();
        let yf = y - y.floor();

        let u = fade(xf);
        let v = fade(yf);

        let aa = perm(perm(xi, self.seed) as i32 + yi, self.seed);
        let ab = perm(perm(xi, self.seed) as i32 + yi + 1, self.seed);
        let ba = perm(perm(xi + 1, self.seed) as i32 + yi, self.seed);
        let bb = perm(perm(xi + 1, self.seed) as i32 + yi + 1, self.seed);

        let x1 = lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u);
        let x2 = lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u);

        (lerp(x1, x2, v) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// 3D Perlin (gradient) noise.
///
/// Classic gradient noise with smooth interpolation.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Perlin3D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Perlin3D {
    /// Creates a new Perlin noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Perlin noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise3D for Perlin3D {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let s = self.seed;
        let aaa = perm(perm(perm(xi, s) as i32 + yi, s) as i32 + zi, s);
        let aba = perm(perm(perm(xi, s) as i32 + yi + 1, s) as i32 + zi, s);
        let aab = perm(perm(perm(xi, s) as i32 + yi, s) as i32 + zi + 1, s);
        let abb = perm(perm(perm(xi, s) as i32 + yi + 1, s) as i32 + zi + 1, s);
        let baa = perm(perm(perm(xi + 1, s) as i32 + yi, s) as i32 + zi, s);
        let bba = perm(perm(perm(xi + 1, s) as i32 + yi + 1, s) as i32 + zi, s);
        let bab = perm(perm(perm(xi + 1, s) as i32 + yi, s) as i32 + zi + 1, s);
        let bbb = perm(perm(perm(xi + 1, s) as i32 + yi + 1, s) as i32 + zi + 1, s);

        let x1 = lerp(grad3(aaa, xf, yf, zf), grad3(baa, xf - 1.0, yf, zf), u);
        let x2 = lerp(
            grad3(aba, xf, yf - 1.0, zf),
            grad3(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = lerp(x1, x2, v);

        let x1 = lerp(
            grad3(aab, xf, yf, zf - 1.0),
            grad3(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x2 = lerp(
            grad3(abb, xf, yf - 1.0, zf - 1.0),
            grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = lerp(x1, x2, v);

        (lerp(y1, y2, w) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Simplex Noise
// =============================================================================

/// 1D Simplex noise.
///
/// In 1D, simplex noise is equivalent to Perlin noise.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simplex1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Simplex1D {
    /// Creates a new Simplex noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Simplex noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Simplex1D {
    fn sample(&self, x: f32) -> f32 {
        Perlin1D { seed: self.seed }.sample(x)
    }
}

/// 2D Simplex noise.
///
/// More efficient than Perlin noise with fewer directional artifacts.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simplex2D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Simplex2D {
    /// Creates a new Simplex noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Simplex noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise2D for Simplex2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let s = (x + y) * F2;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;

        let t = (i + j) as f32 * G2;
        let x0 = x - (i as f32 - t);
        let y0 = y - (j as f32 - t);

        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f32 + G2;
        let y1 = y0 - j1 as f32 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let seed = self.seed;
        let gi0 = perm(perm(i, seed) as i32 + j, seed);
        let gi1 = perm(perm(i + i1, seed) as i32 + j + j1, seed);
        let gi2 = perm(perm(i + 1, seed) as i32 + j + 1, seed);

        let mut n0 = 0.0;
        let mut t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 >= 0.0 {
            t0 *= t0;
            n0 = t0 * t0 * grad2(gi0, x0, y0);
        }

        let mut n1 = 0.0;
        let mut t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 >= 0.0 {
            t1 *= t1;
            n1 = t1 * t1 * grad2(gi1, x1, y1);
        }

        let mut n2 = 0.0;
        let mut t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 >= 0.0 {
            t2 *= t2;
            n2 = t2 * t2 * grad2(gi2, x2, y2);
        }

        // Scale to [0, 1]
        ((70.0 * (n0 + n1 + n2)) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// 3D Simplex noise.
///
/// More efficient than Perlin noise with fewer directional artifacts.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simplex3D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Simplex3D {
    /// Creates a new Simplex noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Simplex noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise3D for Simplex3D {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let s = (x + y + z) * F3;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;
        let k = (z + s).floor() as i32;

        let t = (i + j + k) as f32 * G3;
        let x0 = x - (i as f32 - t);
        let y0 = y - (j as f32 - t);
        let z0 = z - (k as f32 - t);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };

        let x1 = x0 - i1 as f32 + G3;
        let y1 = y0 - j1 as f32 + G3;
        let z1 = z0 - k1 as f32 + G3;
        let x2 = x0 - i2 as f32 + 2.0 * G3;
        let y2 = y0 - j2 as f32 + 2.0 * G3;
        let z2 = z0 - k2 as f32 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let seed = self.seed;
        let gi0 = perm(perm(perm(i, seed) as i32 + j, seed) as i32 + k, seed);
        let gi1 = perm(
            perm(perm(i + i1, seed) as i32 + j + j1, seed) as i32 + k + k1,
            seed,
        );
        let gi2 = perm(
            perm(perm(i + i2, seed) as i32 + j + j2, seed) as i32 + k + k2,
            seed,
        );
        let gi3 = perm(
            perm(perm(i + 1, seed) as i32 + j + 1, seed) as i32 + k + 1,
            seed,
        );

        let mut n0 = 0.0;
        let mut t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t0 >= 0.0 {
            t0 *= t0;
            n0 = t0 * t0 * grad3(gi0, x0, y0, z0);
        }

        let mut n1 = 0.0;
        let mut t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t1 >= 0.0 {
            t1 *= t1;
            n1 = t1 * t1 * grad3(gi1, x1, y1, z1);
        }

        let mut n2 = 0.0;
        let mut t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t2 >= 0.0 {
            t2 *= t2;
            n2 = t2 * t2 * grad3(gi2, x2, y2, z2);
        }

        let mut n3 = 0.0;
        let mut t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t3 >= 0.0 {
            t3 *= t3;
            n3 = t3 * t3 * grad3(gi3, x3, y3, z3);
        }

        // Scale to [0, 1]
        ((32.0 * (n0 + n1 + n2 + n3)) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Value Noise
// =============================================================================

/// 1D Value noise.
///
/// Random values at integer points, smoothly interpolated.
/// Simpler and faster than Perlin, but with more visible grid artifacts.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Value1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Value1D {
    /// Creates a new Value noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Value noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Value1D {
    fn sample(&self, x: f32) -> f32 {
        let xi = x.floor() as i32;
        let xf = x - x.floor();
        let u = fade(xf);

        let a = perm(xi, self.seed) as f32 / 255.0;
        let b = perm(xi + 1, self.seed) as f32 / 255.0;

        lerp(a, b, u)
    }
}

/// 2D Value noise.
///
/// Random values at grid points, smoothly interpolated.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Value2D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Value2D {
    /// Creates a new Value noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Value noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise2D for Value2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        let xf = x - x.floor();
        let yf = y - y.floor();

        let u = fade(xf);
        let v = fade(yf);

        let s = self.seed;
        let aa = perm(perm(xi, s) as i32 + yi, s) as f32 / 255.0;
        let ab = perm(perm(xi, s) as i32 + yi + 1, s) as f32 / 255.0;
        let ba = perm(perm(xi + 1, s) as i32 + yi, s) as f32 / 255.0;
        let bb = perm(perm(xi + 1, s) as i32 + yi + 1, s) as f32 / 255.0;

        let x1 = lerp(aa, ba, u);
        let x2 = lerp(ab, bb, u);

        lerp(x1, x2, v)
    }
}

/// 3D Value noise.
///
/// Random values at grid points, smoothly interpolated.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Value3D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Value3D {
    /// Creates a new Value noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Value noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise3D for Value3D {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let s = self.seed;
        let aaa = perm(perm(perm(xi, s) as i32 + yi, s) as i32 + zi, s) as f32 / 255.0;
        let aba = perm(perm(perm(xi, s) as i32 + yi + 1, s) as i32 + zi, s) as f32 / 255.0;
        let aab = perm(perm(perm(xi, s) as i32 + yi, s) as i32 + zi + 1, s) as f32 / 255.0;
        let abb = perm(perm(perm(xi, s) as i32 + yi + 1, s) as i32 + zi + 1, s) as f32 / 255.0;
        let baa = perm(perm(perm(xi + 1, s) as i32 + yi, s) as i32 + zi, s) as f32 / 255.0;
        let bba = perm(perm(perm(xi + 1, s) as i32 + yi + 1, s) as i32 + zi, s) as f32 / 255.0;
        let bab = perm(perm(perm(xi + 1, s) as i32 + yi, s) as i32 + zi + 1, s) as f32 / 255.0;
        let bbb = perm(perm(perm(xi + 1, s) as i32 + yi + 1, s) as i32 + zi + 1, s) as f32 / 255.0;

        let x1 = lerp(aaa, baa, u);
        let x2 = lerp(aba, bba, u);
        let y1 = lerp(x1, x2, v);

        let x1 = lerp(aab, bab, u);
        let x2 = lerp(abb, bbb, u);
        let y2 = lerp(x1, x2, v);

        lerp(y1, y2, w)
    }
}

// =============================================================================
// Worley (Cellular) Noise
// =============================================================================

/// Distance function for Worley noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DistanceFunction {
    /// Euclidean distance (default).
    #[default]
    Euclidean,
    /// Manhattan distance.
    Manhattan,
    /// Chebyshev distance.
    Chebyshev,
}

/// What to return from Worley noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WorleyReturn {
    /// Distance to nearest feature point (F1).
    #[default]
    F1,
    /// Distance to second nearest feature point (F2).
    F2,
    /// F2 - F1 (cell edges).
    Edge,
}

/// 1D Worley (cellular) noise.
///
/// Distance to nearest random point on a line.
/// Creates sawtooth-like patterns with valleys at random intervals.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Worley1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Worley1D {
    /// Creates a new Worley noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Worley noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Worley1D {
    fn sample(&self, x: f32) -> f32 {
        let xi = x.floor() as i32;

        let mut min_dist = f32::MAX;

        // Check 3 neighboring cells
        for d in -1..=1 {
            let cx = xi + d;
            let h = perm(cx, self.seed);
            let px = cx as f32 + (h as f32 / 255.0);
            let dist = (x - px).abs();
            min_dist = min_dist.min(dist);
        }

        min_dist.clamp(0.0, 1.0)
    }
}

/// 2D Worley (cellular) noise.
///
/// Returns the distance to the nearest feature point.
/// Creates organic, cell-like patterns.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Worley2D {
    /// Random seed for the noise.
    pub seed: i32,
    /// Distance function to use.
    pub distance: DistanceFunction,
    /// What value to return.
    pub return_type: WorleyReturn,
}

impl Worley2D {
    /// Creates a new Worley noise with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Worley noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Sets the distance function.
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }

    /// Sets the return type (F1, F2, or Edge).
    pub fn return_type(mut self, return_type: WorleyReturn) -> Self {
        self.return_type = return_type;
        self
    }

    fn compute_distance(&self, dx: f32, dy: f32) -> f32 {
        match self.distance {
            DistanceFunction::Euclidean => (dx * dx + dy * dy).sqrt(),
            DistanceFunction::Manhattan => dx.abs() + dy.abs(),
            DistanceFunction::Chebyshev => dx.abs().max(dy.abs()),
        }
    }
}

impl Noise2D for Worley2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        let mut min_dist1 = f32::MAX;
        let mut min_dist2 = f32::MAX;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;

                let h = perm(perm(cx, self.seed) as i32 + cy, self.seed);
                let px = cx as f32 + (h as f32 / 255.0);
                let h2 = perm(h as i32 + 1, self.seed);
                let py = cy as f32 + (h2 as f32 / 255.0);

                let dist = self.compute_distance(x - px, y - py);
                if dist < min_dist1 {
                    min_dist2 = min_dist1;
                    min_dist1 = dist;
                } else if dist < min_dist2 {
                    min_dist2 = dist;
                }
            }
        }

        let result = match self.return_type {
            WorleyReturn::F1 => min_dist1 / 1.5,
            WorleyReturn::F2 => min_dist2 / 1.5,
            WorleyReturn::Edge => (min_dist2 - min_dist1) * 2.0,
        };

        result.clamp(0.0, 1.0)
    }
}

/// 3D Worley (cellular) noise.
///
/// Returns the distance to the nearest feature point.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Worley3D {
    /// Random seed for the noise.
    pub seed: i32,
    /// Distance function to use.
    pub distance: DistanceFunction,
    /// What value to return.
    pub return_type: WorleyReturn,
}

impl Worley3D {
    /// Creates a new Worley noise with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Worley noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Sets the distance function.
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }

    /// Sets the return type (F1, F2, or Edge).
    pub fn return_type(mut self, return_type: WorleyReturn) -> Self {
        self.return_type = return_type;
        self
    }

    fn compute_distance(&self, dx: f32, dy: f32, dz: f32) -> f32 {
        match self.distance {
            DistanceFunction::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            DistanceFunction::Manhattan => dx.abs() + dy.abs() + dz.abs(),
            DistanceFunction::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()),
        }
    }
}

impl Noise3D for Worley3D {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let mut min_dist1 = f32::MAX;
        let mut min_dist2 = f32::MAX;

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cx = xi + dx;
                    let cy = yi + dy;
                    let cz = zi + dz;

                    let s = self.seed;
                    let h = perm(perm(perm(cx, s) as i32 + cy, s) as i32 + cz, s);
                    let px = cx as f32 + (h as f32 / 255.0);
                    let h2 = perm(h as i32 + 1, s);
                    let py = cy as f32 + (h2 as f32 / 255.0);
                    let h3 = perm(h2 as i32 + 1, s);
                    let pz = cz as f32 + (h3 as f32 / 255.0);

                    let dist = self.compute_distance(x - px, y - py, z - pz);
                    if dist < min_dist1 {
                        min_dist2 = min_dist1;
                        min_dist1 = dist;
                    } else if dist < min_dist2 {
                        min_dist2 = dist;
                    }
                }
            }
        }

        let result = match self.return_type {
            WorleyReturn::F1 => min_dist1 / 1.8,
            WorleyReturn::F2 => min_dist2 / 1.8,
            WorleyReturn::Edge => (min_dist2 - min_dist1) * 2.0,
        };

        result.clamp(0.0, 1.0)
    }
}

// =============================================================================
// Fractal Brownian Motion (fBm)
// =============================================================================

/// Fractal Brownian Motion - layers multiple octaves of noise.
///
/// Can wrap any noise type to add fractal detail.
///
/// # Example
///
/// ```
/// use rhizome_resin_noise::{Fbm, Perlin2D, Noise2D};
///
/// let fbm = Fbm::new(Perlin2D::new())
///     .octaves(4)
///     .lacunarity(2.0)
///     .persistence(0.5);
///
/// let value = fbm.sample(1.0, 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fbm<N> {
    /// Base noise function.
    pub noise: N,
    /// Number of noise layers.
    pub octaves: u32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub persistence: f32,
}

impl<N> Fbm<N> {
    /// Creates a new fBm with default parameters (4 octaves, 2.0 lacunarity, 0.5 persistence).
    pub fn new(noise: N) -> Self {
        Self {
            noise,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }

    /// Sets the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Sets the lacunarity (frequency multiplier per octave).
    pub fn lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }

    /// Sets the persistence (amplitude multiplier per octave).
    pub fn persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence;
        self
    }
}

impl<N: Noise1D> Noise1D for Fbm<N> {
    fn sample(&self, x: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += self.noise.sample(x * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        value / max_value
    }
}

impl<N: Noise2D> Noise2D for Fbm<N> {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += self.noise.sample(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        value / max_value
    }
}

impl<N: Noise3D> Noise3D for Fbm<N> {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..self.octaves {
            value += self
                .noise
                .sample(x * frequency, y * frequency, z * frequency)
                * amplitude;
            max_value += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        value / max_value
    }
}

// =============================================================================
// Colored Noise
// =============================================================================

/// 1D Pink noise.
///
/// Pink noise has equal energy per octave (1/f spectrum).
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pink1D {
    /// Random seed for the noise.
    pub seed: i32,
    /// Number of octaves.
    pub octaves: u32,
}

impl Pink1D {
    /// Creates a new Pink noise with default settings.
    pub fn new() -> Self {
        Self {
            seed: 0,
            octaves: 8,
        }
    }

    /// Creates a new Pink noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, octaves: 8 }
    }

    /// Sets the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }
}

impl Default for Pink1D {
    fn default() -> Self {
        Self::new()
    }
}

impl Noise1D for Pink1D {
    fn sample(&self, x: f32) -> f32 {
        let value_noise = Value1D::with_seed(self.seed);
        let mut sum = 0.0;
        let mut max = 0.0;

        for i in 0..self.octaves {
            let freq = 1.0 / (1 << i) as f32;
            sum += value_noise.sample(x * freq);
            max += 1.0;
        }

        sum / max
    }
}

/// 2D Pink noise.
///
/// Pink noise has equal energy per octave (1/f spectrum).
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pink2D {
    /// Random seed for the noise.
    pub seed: i32,
    /// Number of octaves.
    pub octaves: u32,
}

impl Pink2D {
    /// Creates a new Pink noise with default settings.
    pub fn new() -> Self {
        Self {
            seed: 0,
            octaves: 8,
        }
    }

    /// Creates a new Pink noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, octaves: 8 }
    }

    /// Sets the number of octaves.
    pub fn octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }
}

impl Default for Pink2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Noise2D for Pink2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let value_noise = Value2D::with_seed(self.seed);
        let mut sum = 0.0;
        let mut max = 0.0;

        for i in 0..self.octaves {
            let freq = 1.0 / (1 << i) as f32;
            sum += value_noise.sample(x * freq, y * freq);
            max += 1.0;
        }

        sum / max
    }
}

/// 1D Brown (Brownian/Red) noise.
///
/// Brown noise has a 1/f² spectrum - strong low-frequency bias.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Brown1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Brown1D {
    /// Creates a new Brown noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Brown noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Brown1D {
    fn sample(&self, x: f32) -> f32 {
        let value_noise = Value1D::with_seed(self.seed);
        let base = value_noise.sample(x * 0.1);
        let detail = value_noise.sample(x * 0.2) * 0.5;
        ((base + detail) / 1.5).clamp(0.0, 1.0)
    }
}

/// 2D Brown (Brownian/Red) noise.
///
/// Brown noise has a 1/f² spectrum - strong low-frequency bias.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Brown2D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Brown2D {
    /// Creates a new Brown noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Brown noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise2D for Brown2D {
    fn sample(&self, x: f32, y: f32) -> f32 {
        let value_noise = Value2D::with_seed(self.seed);
        let base = value_noise.sample(x * 0.1, y * 0.1);
        let detail = value_noise.sample(x * 0.2, y * 0.2) * 0.5;
        ((base + detail) / 1.5).clamp(0.0, 1.0)
    }
}

/// 1D Violet noise.
///
/// Violet noise has an f² spectrum - high frequency emphasis.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Violet1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Violet1D {
    /// Creates a new Violet noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Violet noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Violet1D {
    fn sample(&self, x: f32) -> f32 {
        // Second difference: d²w/dx² ≈ w[n+1] - 2*w[n] + w[n-1]
        let xi = x.floor() as i32;
        let t = x - x.floor();

        let w0 = perm(xi - 1, self.seed) as f32 / 255.0;
        let w1 = perm(xi, self.seed) as f32 / 255.0;
        let w2 = perm(xi + 1, self.seed) as f32 / 255.0;
        let w3 = perm(xi + 2, self.seed) as f32 / 255.0;

        let d1 = w2 - 2.0 * w1 + w0;
        let d2 = w3 - 2.0 * w2 + w1;

        let v = d1 + t * (d2 - d1);

        (v * 0.25 + 0.5).clamp(0.0, 1.0)
    }
}

/// 1D Grey noise (approximation).
///
/// Grey noise is psychoacoustically flat - sounds equally loud at all frequencies.
/// Returns values in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Grey1D {
    /// Random seed for the noise.
    pub seed: i32,
}

impl Grey1D {
    /// Creates a new Grey noise with default seed (0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Grey noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed }
    }
}

impl Noise1D for Grey1D {
    fn sample(&self, x: f32) -> f32 {
        let p = Pink1D::with_seed(self.seed).sample(x);
        let w = perm(x.floor() as i32, self.seed) as f32 / 255.0;
        let b = Brown1D::with_seed(self.seed).sample(x);
        (p * 0.5 + w * 0.3 + b * 0.2).clamp(0.0, 1.0)
    }
}

/// 1D Velvet noise.
///
/// Sparse impulse noise - most samples are neutral (0.5), with occasional
/// impulses toward 0 or 1. Used in audio for efficient convolution reverb.
/// Returns values in [0, 1] where 0.5 is neutral.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Velvet1D {
    /// Random seed for the noise.
    pub seed: i32,
    /// Probability of non-neutral value (0.0 to 1.0, typically 0.01-0.2).
    pub density: f32,
}

impl Velvet1D {
    /// Creates a new Velvet noise with default settings.
    pub fn new() -> Self {
        Self {
            seed: 0,
            density: 0.1,
        }
    }

    /// Creates a new Velvet noise with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self { seed, density: 0.1 }
    }

    /// Sets the impulse density.
    pub fn density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }
}

impl Default for Velvet1D {
    fn default() -> Self {
        Self::new()
    }
}

impl Noise1D for Velvet1D {
    fn sample(&self, x: f32) -> f32 {
        let xi = x.floor() as i32;
        let h = perm(xi, self.seed);
        let threshold = (self.density * 255.0) as u8;

        if h < threshold {
            let polarity = perm(xi.wrapping_add(127), self.seed);
            if polarity < 128 { 0.0 } else { 1.0 }
        } else {
            0.5
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin2d_range() {
        let noise = Perlin2D::new();
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = noise.sample(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "perlin2d({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_simplex2d_range() {
        let noise = Simplex2D::new();
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = noise.sample(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "simplex2d({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_value2d_range() {
        let noise = Value2D::new();
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = noise.sample(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "value2d({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_worley2d_range() {
        let noise = Worley2D::new();
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = noise.sample(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "worley2d({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_fbm_range() {
        let fbm = Fbm::new(Perlin2D::new()).octaves(4);
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = fbm.sample(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "fbm({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_noise_deterministic() {
        let perlin = Perlin2D::new();
        let v1 = perlin.sample(3.14, 2.71);
        let v2 = perlin.sample(3.14, 2.71);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_seed_changes_output() {
        let n1 = Perlin2D::with_seed(0);
        let n2 = Perlin2D::with_seed(42);
        let v1 = n1.sample(1.5, 2.5);
        let v2 = n2.sample(1.5, 2.5);
        assert_ne!(v1, v2, "Different seeds should produce different output");
    }

    #[test]
    fn test_worley_return_types() {
        let f1 = Worley2D::new().return_type(WorleyReturn::F1);
        let f2 = Worley2D::new().return_type(WorleyReturn::F2);
        let edge = Worley2D::new().return_type(WorleyReturn::Edge);

        let x = 1.5;
        let y = 2.5;
        let v1 = f1.sample(x, y);
        let v2 = f2.sample(x, y);
        let ve = edge.sample(x, y);

        // F2 should be >= F1
        assert!(v2 >= v1 * 0.9, "F2 should be >= F1");
        // Edge should be valid range
        assert!((0.0..=1.0).contains(&ve));
    }

    #[test]
    fn test_colored_noise_range() {
        let pink = Pink1D::new();
        let brown = Brown1D::new();
        let violet = Violet1D::new();
        let grey = Grey1D::new();
        let velvet = Velvet1D::new();

        for i in 0..100 {
            let x = i as f32 * 0.1;
            assert!((0.0..=1.0).contains(&pink.sample(x)));
            assert!((0.0..=1.0).contains(&brown.sample(x)));
            assert!((0.0..=1.0).contains(&violet.sample(x)));
            assert!((0.0..=1.0).contains(&grey.sample(x)));
            assert!((0.0..=1.0).contains(&velvet.sample(x)));
        }
    }

    #[test]
    fn test_velvet_mostly_neutral() {
        let velvet = Velvet1D::new().density(0.05);
        let mut neutral_count = 0;
        for i in 0..1000 {
            let v = velvet.sample(i as f32);
            if (v - 0.5).abs() < 0.01 {
                neutral_count += 1;
            }
        }
        assert!(
            neutral_count > 900,
            "Velvet with 5% density should be mostly neutral, got {} neutral out of 1000",
            neutral_count
        );
    }
}

/// Statistical invariant tests - run with `cargo test --features invariant-tests`
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    const SAMPLES: usize = 10000;

    fn mean(values: &[f32]) -> f32 {
        values.iter().sum::<f32>() / values.len() as f32
    }

    fn variance(values: &[f32], mean: f32) -> f32 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
    }

    fn autocorrelation(values: &[f32], lag: usize) -> f32 {
        let m = mean(values);
        let var = variance(values, m);
        if var < 1e-10 {
            return 0.0;
        }
        let n = values.len() - lag;
        let sum: f32 = (0..n)
            .map(|i| (values[i] - m) * (values[i + lag] - m))
            .sum();
        sum / (n as f32 * var)
    }

    #[test]
    fn test_perlin_noise_distribution() {
        let noise = Perlin1D::new();
        let values: Vec<f32> = (0..SAMPLES).map(|i| noise.sample(i as f32 * 0.1)).collect();

        let m = mean(&values);
        assert!(
            (m - 0.5).abs() < 0.1,
            "Perlin noise mean should be ~0.5, got {}",
            m
        );
    }

    #[test]
    fn test_value_noise_distribution() {
        let noise = Value1D::new();
        let values: Vec<f32> = (0..SAMPLES).map(|i| noise.sample(i as f32 * 0.1)).collect();

        let m = mean(&values);
        assert!(
            (m - 0.5).abs() < 0.1,
            "Value noise mean should be ~0.5, got {}",
            m
        );
    }

    #[test]
    fn test_perlin_noise_has_autocorrelation() {
        let noise = Perlin1D::new();
        let values: Vec<f32> = (0..SAMPLES)
            .map(|i| noise.sample(i as f32 * 0.05))
            .collect();

        let ac1 = autocorrelation(&values, 1);
        assert!(
            ac1 > 0.5,
            "Perlin noise should have high autocorrelation at lag 1, got {}",
            ac1
        );
    }

    #[test]
    fn test_brown_noise_very_high_autocorrelation() {
        let noise = Brown1D::new();
        let values: Vec<f32> = (0..SAMPLES).map(|i| noise.sample(i as f32 * 0.1)).collect();

        let ac1 = autocorrelation(&values, 1);
        assert!(
            ac1 > 0.8,
            "Brown noise should have very high autocorrelation, got {}",
            ac1
        );
    }

    #[test]
    fn test_noise_deterministic() {
        let perlin = Perlin2D::new();
        let simplex = Simplex2D::new();
        let value = Value2D::new();
        let worley = Worley2D::new();

        for i in 0..100 {
            let x = i as f32 * 0.37;
            assert_eq!(perlin.sample(x, x * 1.5), perlin.sample(x, x * 1.5));
            assert_eq!(simplex.sample(x, x), simplex.sample(x, x));
            assert_eq!(value.sample(x, x), value.sample(x, x));
            assert_eq!(worley.sample(x, x), worley.sample(x, x));
        }
    }

    #[test]
    fn test_worley_has_zeros() {
        let noise = Worley2D::new();
        let mut found_near_zero = false;
        for i in 0..SAMPLES {
            let v = noise.sample(i as f32 * 0.1, i as f32 * 0.07);
            if v < 0.05 {
                found_near_zero = true;
                break;
            }
        }
        assert!(
            found_near_zero,
            "Worley noise should have values near 0 at feature points"
        );
    }

    #[test]
    fn test_worley_f2_greater_than_f1() {
        let f1_noise = Worley2D::new().return_type(WorleyReturn::F1);
        let f2_noise = Worley2D::new().return_type(WorleyReturn::F2);

        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.07;
            let f1 = f1_noise.sample(x, y);
            let f2 = f2_noise.sample(x, y);
            assert!(
                f2 >= f1 * 0.9,
                "F2 should be >= F1, got f1={}, f2={}",
                f1,
                f2
            );
        }
    }
}
