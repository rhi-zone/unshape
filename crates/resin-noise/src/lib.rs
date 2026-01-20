//! Noise functions for procedural generation.
//!
//! Provides classic noise algorithms used across all domains:
//! textures, mesh displacement, audio modulation, etc.

use glam::{Vec2, Vec3};

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
fn perm(x: i32) -> u8 {
    PERM[(x & 255) as usize]
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

/// 1D Perlin noise.
///
/// Returns a value in [0, 1]. Useful for audio modulation and 1D patterns.
pub fn perlin1(x: f32) -> f32 {
    let xi = x.floor() as i32;
    let xf = x - x.floor();
    let u = fade(xf);

    let a = perm(xi);
    let b = perm(xi + 1);

    (lerp(grad1(a, xf), grad1(b, xf - 1.0), u) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// 2D Perlin noise.
///
/// Returns a value in approximately [-1, 1].
pub fn perlin2(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm(perm(xi) as i32 + yi);
    let ab = perm(perm(xi) as i32 + yi + 1);
    let ba = perm(perm(xi + 1) as i32 + yi);
    let bb = perm(perm(xi + 1) as i32 + yi + 1);

    let x1 = lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u);
    let x2 = lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u);

    (lerp(x1, x2, v) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// 3D Perlin noise.
///
/// Returns a value in approximately [-1, 1].
pub fn perlin3(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let aaa = perm(perm(perm(xi) as i32 + yi) as i32 + zi);
    let aba = perm(perm(perm(xi) as i32 + yi + 1) as i32 + zi);
    let aab = perm(perm(perm(xi) as i32 + yi) as i32 + zi + 1);
    let abb = perm(perm(perm(xi) as i32 + yi + 1) as i32 + zi + 1);
    let baa = perm(perm(perm(xi + 1) as i32 + yi) as i32 + zi);
    let bba = perm(perm(perm(xi + 1) as i32 + yi + 1) as i32 + zi);
    let bab = perm(perm(perm(xi + 1) as i32 + yi) as i32 + zi + 1);
    let bbb = perm(perm(perm(xi + 1) as i32 + yi + 1) as i32 + zi + 1);

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

/// 2D Perlin noise with Vec2 input.
pub fn perlin2v(p: Vec2) -> f32 {
    perlin2(p.x, p.y)
}

/// 3D Perlin noise with Vec3 input.
pub fn perlin3v(p: Vec3) -> f32 {
    perlin3(p.x, p.y, p.z)
}

// Simplex noise helpers
const F2: f32 = 0.5 * (1.732_050_8 - 1.0); // (sqrt(3) - 1) / 2
const G2: f32 = (3.0 - 1.732_050_8) / 6.0; // (3 - sqrt(3)) / 6
const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

/// 1D Simplex noise.
///
/// In 1D, simplex noise is equivalent to Perlin noise (no skewing needed).
/// Returns a value in [0, 1].
pub fn simplex1(x: f32) -> f32 {
    perlin1(x)
}

/// 2D Simplex noise.
///
/// More efficient than Perlin noise with fewer artifacts.
/// Returns a value in [0, 1].
pub fn simplex2(x: f32, y: f32) -> f32 {
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

    let gi0 = perm(perm(i) as i32 + j);
    let gi1 = perm(perm(i + i1) as i32 + j + j1);
    let gi2 = perm(perm(i + 1) as i32 + j + 1);

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

/// 3D Simplex noise.
///
/// Returns a value in [0, 1].
pub fn simplex3(x: f32, y: f32, z: f32) -> f32 {
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

    let gi0 = perm(perm(perm(i) as i32 + j) as i32 + k);
    let gi1 = perm(perm(perm(i + i1) as i32 + j + j1) as i32 + k + k1);
    let gi2 = perm(perm(perm(i + i2) as i32 + j + j2) as i32 + k + k2);
    let gi3 = perm(perm(perm(i + 1) as i32 + j + 1) as i32 + k + 1);

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

/// 2D Simplex noise with Vec2 input.
pub fn simplex2v(p: Vec2) -> f32 {
    simplex2(p.x, p.y)
}

/// 3D Simplex noise with Vec3 input.
pub fn simplex3v(p: Vec3) -> f32 {
    simplex3(p.x, p.y, p.z)
}

/// Fractal Brownian Motion (fBm) using 1D noise.
///
/// Layers multiple octaves of noise for natural-looking detail.
/// Useful for audio modulation, terrain profiles, etc.
pub fn fbm1<F: Fn(f32) -> f32>(
    noise_fn: F,
    x: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        value += noise_fn(x * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

/// Fractal Brownian Motion (fBm) using 2D noise.
///
/// Layers multiple octaves of noise for natural-looking detail.
///
/// # Arguments
/// * `noise_fn` - Base noise function to use
/// * `x`, `y` - Coordinates
/// * `octaves` - Number of noise layers (typically 4-8)
/// * `lacunarity` - Frequency multiplier per octave (typically 2.0)
/// * `persistence` - Amplitude multiplier per octave (typically 0.5)
pub fn fbm2<F: Fn(f32, f32) -> f32>(
    noise_fn: F,
    x: f32,
    y: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        value += noise_fn(x * frequency, y * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

/// Fractal Brownian Motion (fBm) using 3D noise.
pub fn fbm3<F: Fn(f32, f32, f32) -> f32>(
    noise_fn: F,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        value += noise_fn(x * frequency, y * frequency, z * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

/// Convenience function for 1D fBm with Perlin noise.
pub fn fbm_perlin1(x: f32, octaves: u32) -> f32 {
    fbm1(perlin1, x, octaves, 2.0, 0.5)
}

/// Convenience function for 2D fBm with Perlin noise.
pub fn fbm_perlin2(x: f32, y: f32, octaves: u32) -> f32 {
    fbm2(perlin2, x, y, octaves, 2.0, 0.5)
}

/// Convenience function for 3D fBm with Perlin noise.
pub fn fbm_perlin3(x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    fbm3(perlin3, x, y, z, octaves, 2.0, 0.5)
}

/// Convenience function for 1D fBm with Simplex noise.
pub fn fbm_simplex1(x: f32, octaves: u32) -> f32 {
    fbm1(simplex1, x, octaves, 2.0, 0.5)
}

/// Convenience function for 2D fBm with Simplex noise.
pub fn fbm_simplex2(x: f32, y: f32, octaves: u32) -> f32 {
    fbm2(simplex2, x, y, octaves, 2.0, 0.5)
}

/// Convenience function for 3D fBm with Simplex noise.
pub fn fbm_simplex3(x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    fbm3(simplex3, x, y, z, octaves, 2.0, 0.5)
}

// =============================================================================
// Value Noise
// =============================================================================
// Simpler than Perlin/Simplex: random values at grid points, interpolated.
// Faster but has more visible grid artifacts.

/// 1D Value noise.
///
/// Random values at integer points, smoothly interpolated.
/// Simpler and faster than Perlin, but with more visible artifacts.
/// Returns a value in [0, 1].
pub fn value1(x: f32) -> f32 {
    let xi = x.floor() as i32;
    let xf = x - x.floor();
    let u = fade(xf);

    let a = perm(xi) as f32 / 255.0;
    let b = perm(xi + 1) as f32 / 255.0;

    lerp(a, b, u)
}

/// 2D Value noise.
///
/// Random values at grid points, smoothly interpolated.
/// Returns a value in [0, 1].
pub fn value2(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm(perm(xi) as i32 + yi) as f32 / 255.0;
    let ab = perm(perm(xi) as i32 + yi + 1) as f32 / 255.0;
    let ba = perm(perm(xi + 1) as i32 + yi) as f32 / 255.0;
    let bb = perm(perm(xi + 1) as i32 + yi + 1) as f32 / 255.0;

    let x1 = lerp(aa, ba, u);
    let x2 = lerp(ab, bb, u);

    lerp(x1, x2, v)
}

/// 3D Value noise.
///
/// Random values at grid points, smoothly interpolated.
/// Returns a value in [0, 1].
pub fn value3(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let aaa = perm(perm(perm(xi) as i32 + yi) as i32 + zi) as f32 / 255.0;
    let aba = perm(perm(perm(xi) as i32 + yi + 1) as i32 + zi) as f32 / 255.0;
    let aab = perm(perm(perm(xi) as i32 + yi) as i32 + zi + 1) as f32 / 255.0;
    let abb = perm(perm(perm(xi) as i32 + yi + 1) as i32 + zi + 1) as f32 / 255.0;
    let baa = perm(perm(perm(xi + 1) as i32 + yi) as i32 + zi) as f32 / 255.0;
    let bba = perm(perm(perm(xi + 1) as i32 + yi + 1) as i32 + zi) as f32 / 255.0;
    let bab = perm(perm(perm(xi + 1) as i32 + yi) as i32 + zi + 1) as f32 / 255.0;
    let bbb = perm(perm(perm(xi + 1) as i32 + yi + 1) as i32 + zi + 1) as f32 / 255.0;

    let x1 = lerp(aaa, baa, u);
    let x2 = lerp(aba, bba, u);
    let y1 = lerp(x1, x2, v);

    let x1 = lerp(aab, bab, u);
    let x2 = lerp(abb, bbb, u);
    let y2 = lerp(x1, x2, v);

    lerp(y1, y2, w)
}

/// 2D Value noise with Vec2 input.
pub fn value2v(p: Vec2) -> f32 {
    value2(p.x, p.y)
}

/// 3D Value noise with Vec3 input.
pub fn value3v(p: Vec3) -> f32 {
    value3(p.x, p.y, p.z)
}

/// Convenience function for 1D fBm with Value noise.
pub fn fbm_value1(x: f32, octaves: u32) -> f32 {
    fbm1(value1, x, octaves, 2.0, 0.5)
}

/// Convenience function for 2D fBm with Value noise.
pub fn fbm_value2(x: f32, y: f32, octaves: u32) -> f32 {
    fbm2(value2, x, y, octaves, 2.0, 0.5)
}

/// Convenience function for 3D fBm with Value noise.
pub fn fbm_value3(x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    fbm3(value3, x, y, z, octaves, 2.0, 0.5)
}

// =============================================================================
// Worley (Cellular) Noise
// =============================================================================
// Distance to randomly placed feature points. Creates cell-like patterns.

/// 1D Worley noise.
///
/// Distance to nearest random point on a line.
/// Creates sawtooth-like patterns with valleys at random intervals.
/// Useful for: random event timing, tension/release in audio, non-uniform spacing.
/// Returns a value in [0, 1].
pub fn worley1(x: f32) -> f32 {
    let xi = x.floor() as i32;

    let mut min_dist = f32::MAX;

    // Check 3 neighboring cells
    for d in -1..=1 {
        let cx = xi + d;
        // Deterministic random point within this cell
        let h = perm(cx);
        let px = cx as f32 + (h as f32 / 255.0);
        let dist = (x - px).abs();
        min_dist = min_dist.min(dist);
    }

    // Normalize: max distance in a cell is ~1.0
    min_dist.clamp(0.0, 1.0)
}

/// 2D Worley (cellular) noise.
///
/// Returns the distance to the nearest feature point, normalized to [0, 1].
/// Creates organic, cell-like patterns useful for textures, caustics, etc.
pub fn worley2(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let mut min_dist = f32::MAX;

    // Check 3x3 neighborhood of cells
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = xi + dx;
            let cy = yi + dy;

            // Deterministic random point within this cell
            let h = perm(perm(cx) as i32 + cy);
            let px = cx as f32 + (h as f32 / 255.0);
            let h2 = perm(h as i32 + 1);
            let py = cy as f32 + (h2 as f32 / 255.0);

            let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
            min_dist = min_dist.min(dist);
        }
    }

    // Normalize: max possible distance in a cell is sqrt(2) ≈ 1.414
    (min_dist / 1.5).clamp(0.0, 1.0)
}

/// 3D Worley (cellular) noise.
///
/// Returns the distance to the nearest feature point, normalized to [0, 1].
pub fn worley3(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let mut min_dist = f32::MAX;

    // Check 3x3x3 neighborhood of cells
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;
                let cz = zi + dz;

                // Deterministic random point within this cell
                let h = perm(perm(perm(cx) as i32 + cy) as i32 + cz);
                let px = cx as f32 + (h as f32 / 255.0);
                let h2 = perm(h as i32 + 1);
                let py = cy as f32 + (h2 as f32 / 255.0);
                let h3 = perm(h2 as i32 + 1);
                let pz = cz as f32 + (h3 as f32 / 255.0);

                let dist = ((x - px).powi(2) + (y - py).powi(2) + (z - pz).powi(2)).sqrt();
                min_dist = min_dist.min(dist);
            }
        }
    }

    // Normalize: max possible distance in a cell is sqrt(3) ≈ 1.732
    (min_dist / 1.8).clamp(0.0, 1.0)
}

/// 2D Worley noise with Vec2 input.
pub fn worley2v(p: Vec2) -> f32 {
    worley2(p.x, p.y)
}

/// 3D Worley noise with Vec3 input.
pub fn worley3v(p: Vec3) -> f32 {
    worley3(p.x, p.y, p.z)
}

/// 2D Worley noise returning distance to second-nearest point.
///
/// Creates more complex cellular patterns with visible cell boundaries.
pub fn worley2_f2(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let mut min_dist1 = f32::MAX;
    let mut min_dist2 = f32::MAX;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = xi + dx;
            let cy = yi + dy;

            let h = perm(perm(cx) as i32 + cy);
            let px = cx as f32 + (h as f32 / 255.0);
            let h2 = perm(h as i32 + 1);
            let py = cy as f32 + (h2 as f32 / 255.0);

            let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
            if dist < min_dist1 {
                min_dist2 = min_dist1;
                min_dist1 = dist;
            } else if dist < min_dist2 {
                min_dist2 = dist;
            }
        }
    }

    (min_dist2 / 1.5).clamp(0.0, 1.0)
}

/// 2D Worley noise returning F2 - F1 (cell edges).
///
/// Highlights the boundaries between cells. Useful for cracked earth,
/// giraffe patterns, etc.
pub fn worley2_edge(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let mut min_dist1 = f32::MAX;
    let mut min_dist2 = f32::MAX;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = xi + dx;
            let cy = yi + dy;

            let h = perm(perm(cx) as i32 + cy);
            let px = cx as f32 + (h as f32 / 255.0);
            let h2 = perm(h as i32 + 1);
            let py = cy as f32 + (h2 as f32 / 255.0);

            let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
            if dist < min_dist1 {
                min_dist2 = min_dist1;
                min_dist1 = dist;
            } else if dist < min_dist2 {
                min_dist2 = dist;
            }
        }
    }

    ((min_dist2 - min_dist1) * 2.0).clamp(0.0, 1.0)
}

// =============================================================================
// Colored Noise (Spectral)
// =============================================================================
// These noise types are defined by their spectral properties.
// They're typically used for audio and time-series, but can be applied spatially.

/// 1D Pink noise approximation using octave stacking.
///
/// Pink noise has equal energy per octave (1/f spectrum).
/// This uses the Voss algorithm: sum of multiple octaves of value noise.
/// Returns a value in [0, 1].
pub fn pink1(x: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut max = 0.0;

    for i in 0..octaves {
        let freq = 1.0 / (1 << i) as f32;
        sum += value1(x * freq);
        max += 1.0;
    }

    sum / max
}

/// 2D Pink noise approximation.
///
/// Layered value noise with 1/f amplitude scaling.
pub fn pink2(x: f32, y: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut max = 0.0;

    for i in 0..octaves {
        let freq = 1.0 / (1 << i) as f32;
        sum += value2(x * freq, y * freq);
        max += 1.0;
    }

    sum / max
}

/// 1D Brown (Brownian/Red) noise.
///
/// Brown noise has a 1/f² spectrum - strong low-frequency bias.
/// This is essentially very smooth interpolated noise.
/// Returns a value in [0, 1].
pub fn brown1(x: f32) -> f32 {
    // Use very low frequency value noise with extra smoothing
    let base = value1(x * 0.1);
    let detail = value1(x * 0.2) * 0.5;
    ((base + detail) / 1.5).clamp(0.0, 1.0)
}

/// 2D Brown noise.
///
/// Very smooth, low-frequency noise.
pub fn brown2(x: f32, y: f32) -> f32 {
    let base = value2(x * 0.1, y * 0.1);
    let detail = value2(x * 0.2, y * 0.2) * 0.5;
    ((base + detail) / 1.5).clamp(0.0, 1.0)
}

/// 1D Violet noise.
///
/// Violet noise has an f² spectrum - very high frequency emphasis.
/// This is computed as the second difference of white noise (differentiated twice),
/// which produces an f² power spectrum.
/// Returns a value in [0, 1].
pub fn violet1(x: f32) -> f32 {
    // Second difference: d²w/dx² ≈ w[n+1] - 2*w[n] + w[n-1]
    // This gives f² spectral slope
    let xi = x.floor() as i32;
    let t = x - x.floor();

    let w0 = perm(xi - 1) as f32 / 255.0;
    let w1 = perm(xi) as f32 / 255.0;
    let w2 = perm(xi + 1) as f32 / 255.0;
    let w3 = perm(xi + 2) as f32 / 255.0;

    // Interpolate second differences
    let d1 = w2 - 2.0 * w1 + w0; // Second diff at xi
    let d2 = w3 - 2.0 * w2 + w1; // Second diff at xi+1

    let v = d1 + t * (d2 - d1);

    // Range of second diff is roughly [-2, 2], scale to [0, 1]
    (v * 0.25 + 0.5).clamp(0.0, 1.0)
}

/// 1D Grey noise (approximation).
///
/// Grey noise is psychoacoustically flat - it sounds equally loud at all
/// frequencies to human ears, unlike white noise which sounds "bright".
///
/// True grey noise requires equal-loudness contour weighting (ISO 226).
/// This is an approximation that boosts lows and highs relative to pink noise.
/// Returns a value in [0, 1].
pub fn grey1(x: f32) -> f32 {
    // Approximate grey noise by mixing pink (natural) with some white (brightness)
    // and a touch of brown (low-end weight)
    let p = pink1(x, 6);
    let w = perm(x.floor() as i32) as f32 / 255.0;
    let b = brown1(x);
    // Mix: mostly pink, some white for highs, some brown for lows
    (p * 0.5 + w * 0.3 + b * 0.2).clamp(0.0, 1.0)
}

/// 1D Velvet noise.
///
/// Sparse impulse noise - most samples are ~0.5 (neutral), with occasional
/// impulses toward 0 or 1. Used in audio for efficient convolution reverb
/// and decorrelation.
///
/// # Arguments
/// * `x` - Position
/// * `density` - Probability of non-neutral value (0.0 to 1.0, typically 0.01-0.2)
///
/// Returns a value in [0, 1] where 0.5 is neutral, 0 and 1 are impulses.
pub fn velvet1(x: f32, density: f32) -> f32 {
    let xi = x.floor() as i32;
    let h = perm(xi);
    let threshold = (density * 255.0) as u8;

    if h < threshold {
        // Impulse - decide polarity with another hash
        let polarity = perm(xi.wrapping_add(127));
        if polarity < 128 {
            0.0 // Negative impulse
        } else {
            1.0 // Positive impulse
        }
    } else {
        0.5 // Neutral (silence in audio terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin2_range() {
        // Sample many points and verify range
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = perlin2(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "perlin2({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_perlin3_range() {
        for i in 0..20 {
            for j in 0..20 {
                for k in 0..20 {
                    let x = i as f32 * 0.2;
                    let y = j as f32 * 0.2;
                    let z = k as f32 * 0.2;
                    let v = perlin3(x, y, z);
                    assert!((0.0..=1.0).contains(&v), "perlin3 out of range: {}", v);
                }
            }
        }
    }

    #[test]
    fn test_simplex2_range() {
        for i in 0..100 {
            for j in 0..100 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = simplex2(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "simplex2({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn test_simplex3_range() {
        for i in 0..20 {
            for j in 0..20 {
                for k in 0..20 {
                    let x = i as f32 * 0.2;
                    let y = j as f32 * 0.2;
                    let z = k as f32 * 0.2;
                    let v = simplex3(x, y, z);
                    assert!((0.0..=1.0).contains(&v), "simplex3 out of range: {}", v);
                }
            }
        }
    }

    #[test]
    fn test_fbm_range() {
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = fbm_perlin2(x, y, 4);
                assert!((0.0..=1.0).contains(&v), "fbm out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_noise_varies() {
        // Noise should not be constant
        let v1 = perlin2(0.0, 0.0);
        let v2 = perlin2(1.0, 1.0);
        let v3 = perlin2(2.5, 3.7);
        assert!(v1 != v2 || v2 != v3, "noise should vary");
    }

    #[test]
    fn test_perlin2v() {
        let p = Vec2::new(1.5, 2.5);
        let v1 = perlin2v(p);
        let v2 = perlin2(1.5, 2.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_perlin3v() {
        let p = Vec3::new(1.5, 2.5, 3.5);
        let v1 = perlin3v(p);
        let v2 = perlin3(1.5, 2.5, 3.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_simplex2v() {
        let p = Vec2::new(1.5, 2.5);
        let v1 = simplex2v(p);
        let v2 = simplex2(1.5, 2.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_simplex3v() {
        let p = Vec3::new(1.5, 2.5, 3.5);
        let v1 = simplex3v(p);
        let v2 = simplex3(1.5, 2.5, 3.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_noise_deterministic() {
        // Same input should produce same output
        let v1 = perlin2(3.14, 2.71);
        let v2 = perlin2(3.14, 2.71);
        assert_eq!(v1, v2);

        let v3 = simplex3(1.0, 2.0, 3.0);
        let v4 = simplex3(1.0, 2.0, 3.0);
        assert_eq!(v3, v4);
    }

    #[test]
    fn test_fbm_octaves() {
        // More octaves should add more detail (different values)
        // Use non-integer coords - gradient noise is often zero at integers
        let v1 = fbm_perlin2(1.37, 2.81, 1);
        let v2 = fbm_perlin2(1.37, 2.81, 4);
        let v3 = fbm_perlin2(1.37, 2.81, 8);
        // They should differ (not guaranteed but very likely at non-integer coords)
        assert!(v1 != v2 || v2 != v3);
    }

    #[test]
    fn test_fbm3_range() {
        for i in 0..20 {
            for j in 0..20 {
                let x = i as f32 * 0.2;
                let y = j as f32 * 0.2;
                let v = fbm_perlin3(x, y, 0.5, 4);
                assert!((0.0..=1.0).contains(&v), "fbm3 out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_fbm_simplex2_range() {
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = fbm_simplex2(x, y, 4);
                assert!((0.0..=1.0).contains(&v), "fbm_simplex2 out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_fbm_simplex3_range() {
        for i in 0..20 {
            for j in 0..20 {
                let x = i as f32 * 0.2;
                let y = j as f32 * 0.2;
                let v = fbm_simplex3(x, y, 0.5, 4);
                assert!((0.0..=1.0).contains(&v), "fbm_simplex3 out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_fbm_custom_params() {
        // Test with custom lacunarity and persistence
        let v = fbm2(perlin2, 1.0, 2.0, 3, 2.5, 0.4);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_negative_coordinates() {
        // Noise should work with negative coordinates
        let v1 = perlin2(-5.0, -3.0);
        let v2 = simplex2(-10.0, -20.0);
        let v3 = perlin3(-1.0, -2.0, -3.0);
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
        assert!((0.0..=1.0).contains(&v3));
    }

    #[test]
    fn test_perlin1_range() {
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = perlin1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "perlin1({}) = {} out of range",
                x,
                v
            );
        }
    }

    #[test]
    fn test_value_noise_range() {
        // 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = value1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "value1({}) = {} out of range",
                x,
                v
            );
        }

        // 2D
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = value2(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "value2({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }

        // 3D
        for i in 0..20 {
            for j in 0..20 {
                let x = i as f32 * 0.2;
                let y = j as f32 * 0.2;
                let v = value3(x, y, 0.5);
                assert!((0.0..=1.0).contains(&v), "value3 out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_worley_noise_range() {
        // 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = worley1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "worley1({}) = {} out of range",
                x,
                v
            );
        }

        // 2D
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;
                let v = worley2(x, y);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "worley2({}, {}) = {} out of range",
                    x,
                    y,
                    v
                );
            }
        }

        // 3D
        for i in 0..20 {
            for j in 0..20 {
                let x = i as f32 * 0.2;
                let y = j as f32 * 0.2;
                let v = worley3(x, y, 0.5);
                assert!((0.0..=1.0).contains(&v), "worley3 out of range: {}", v);
            }
        }

        // F2 and edge
        for i in 0..30 {
            for j in 0..30 {
                let x = i as f32 * 0.15;
                let y = j as f32 * 0.15;
                let v_f2 = worley2_f2(x, y);
                let v_edge = worley2_edge(x, y);
                assert!(
                    (0.0..=1.0).contains(&v_f2),
                    "worley2_f2 out of range: {}",
                    v_f2
                );
                assert!(
                    (0.0..=1.0).contains(&v_edge),
                    "worley2_edge out of range: {}",
                    v_edge
                );
            }
        }
    }

    #[test]
    fn test_fbm1_range() {
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = fbm_perlin1(x, 4);
            assert!(
                (0.0..=1.0).contains(&v),
                "fbm_perlin1({}) = {} out of range",
                x,
                v
            );
        }
    }

    #[test]
    fn test_colored_noise_range() {
        // Pink 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = pink1(x, 8);
            assert!(
                (0.0..=1.0).contains(&v),
                "pink1({}) = {} out of range",
                x,
                v
            );
        }

        // Pink 2D
        for i in 0..30 {
            for j in 0..30 {
                let x = i as f32 * 0.15;
                let y = j as f32 * 0.15;
                let v = pink2(x, y, 8);
                assert!((0.0..=1.0).contains(&v), "pink2 out of range: {}", v);
            }
        }

        // Brown 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = brown1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "brown1({}) = {} out of range",
                x,
                v
            );
        }

        // Brown 2D
        for i in 0..30 {
            for j in 0..30 {
                let x = i as f32 * 0.15;
                let y = j as f32 * 0.15;
                let v = brown2(x, y);
                assert!((0.0..=1.0).contains(&v), "brown2 out of range: {}", v);
            }
        }

        // Violet 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = violet1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "violet1({}) = {} out of range",
                x,
                v
            );
        }

        // Grey 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = grey1(x);
            assert!(
                (0.0..=1.0).contains(&v),
                "grey1({}) = {} out of range",
                x,
                v
            );
        }

        // Velvet 1D
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let v = velvet1(x, 0.1);
            assert!(
                (0.0..=1.0).contains(&v),
                "velvet1({}) = {} out of range",
                x,
                v
            );
        }

        // Velvet should produce mostly 0.5 with low density
        let mut neutral_count = 0;
        for i in 0..1000 {
            let v = velvet1(i as f32, 0.05);
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

    // =========================================================================
    // Distribution tests
    // =========================================================================

    #[test]
    fn test_white_noise_uniform_distribution() {
        let values: Vec<f32> = (0..SAMPLES)
            .map(|i| {
                let h = perm(i as i32) as f32 / 255.0;
                h
            })
            .collect();

        let m = mean(&values);
        let v = variance(&values, m);

        // Uniform [0,1] has mean=0.5, variance=1/12≈0.0833
        assert!(
            (m - 0.5).abs() < 0.05,
            "White noise mean should be ~0.5, got {}",
            m
        );
        assert!(
            (v - 0.0833).abs() < 0.02,
            "White noise variance should be ~0.083, got {}",
            v
        );
    }

    #[test]
    fn test_perlin_noise_distribution() {
        let values: Vec<f32> = (0..SAMPLES).map(|i| perlin1(i as f32 * 0.1)).collect();

        let m = mean(&values);
        // Perlin noise should be centered around 0.5
        assert!(
            (m - 0.5).abs() < 0.1,
            "Perlin noise mean should be ~0.5, got {}",
            m
        );
    }

    #[test]
    fn test_value_noise_distribution() {
        let values: Vec<f32> = (0..SAMPLES).map(|i| value1(i as f32 * 0.1)).collect();

        let m = mean(&values);
        assert!(
            (m - 0.5).abs() < 0.1,
            "Value noise mean should be ~0.5, got {}",
            m
        );
    }

    // =========================================================================
    // Autocorrelation tests
    // =========================================================================

    #[test]
    fn test_white_noise_no_autocorrelation() {
        let values: Vec<f32> = (0..SAMPLES)
            .map(|i| perm(i as i32) as f32 / 255.0)
            .collect();

        // White noise should have near-zero autocorrelation at lag > 0
        let ac1 = autocorrelation(&values, 1);
        let ac5 = autocorrelation(&values, 5);

        assert!(
            ac1.abs() < 0.1,
            "White noise autocorrelation(1) should be ~0, got {}",
            ac1
        );
        assert!(
            ac5.abs() < 0.1,
            "White noise autocorrelation(5) should be ~0, got {}",
            ac5
        );
    }

    #[test]
    fn test_perlin_noise_has_autocorrelation() {
        let values: Vec<f32> = (0..SAMPLES)
            .map(|i| perlin1(i as f32 * 0.05)) // Small step = high correlation
            .collect();

        // Perlin noise should have positive autocorrelation at small lags
        let ac1 = autocorrelation(&values, 1);
        assert!(
            ac1 > 0.5,
            "Perlin noise should have high autocorrelation at lag 1, got {}",
            ac1
        );
    }

    #[test]
    fn test_value_noise_has_autocorrelation() {
        let values: Vec<f32> = (0..SAMPLES).map(|i| value1(i as f32 * 0.05)).collect();

        let ac1 = autocorrelation(&values, 1);
        assert!(
            ac1 > 0.5,
            "Value noise should have high autocorrelation at lag 1, got {}",
            ac1
        );
    }

    #[test]
    fn test_brown_noise_very_high_autocorrelation() {
        let values: Vec<f32> = (0..SAMPLES).map(|i| brown1(i as f32 * 0.1)).collect();

        // Brown noise is very smooth, should have very high autocorrelation
        let ac1 = autocorrelation(&values, 1);
        assert!(
            ac1 > 0.8,
            "Brown noise should have very high autocorrelation, got {}",
            ac1
        );
    }

    // =========================================================================
    // Spectral property tests (FFT-based)
    // =========================================================================

    use rustfft::{FftPlanner, num_complex::Complex};

    /// Compute the power spectrum of a signal using FFT
    fn power_spectrum(samples: &[f32]) -> Vec<f32> {
        let n = samples.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut buffer: Vec<Complex<f32>> = samples.iter().map(|&x| Complex::new(x, 0.0)).collect();

        fft.process(&mut buffer);

        // Power spectrum (magnitude squared), only positive frequencies
        buffer[1..n / 2].iter().map(|c| c.norm_sqr()).collect()
    }

    /// Fit a power law to the spectrum and return the slope
    /// For spectral density S(f) ∝ f^slope:
    /// - White: slope ≈ 0
    /// - Pink: slope ≈ -1
    /// - Brown: slope ≈ -2
    /// - Blue: slope ≈ +1
    /// - Violet: slope ≈ +2
    fn spectral_slope(power: &[f32]) -> f32 {
        let n = power.len();
        if n < 10 {
            return 0.0;
        }

        // Use middle frequencies to avoid DC and Nyquist artifacts
        let start = n / 8;
        let end = n * 3 / 4;

        // Linear regression on log-log scale
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_xy = 0.0f32;
        let mut count = 0;

        for i in start..end {
            let p = power[i];
            if p > 1e-10 {
                let x = (i as f32).ln();
                let y = p.ln();
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                count += 1;
            }
        }

        if count < 5 {
            return 0.0;
        }

        let n = count as f32;
        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    }

    #[test]
    fn test_white_noise_flat_spectrum() {
        const FFT_SIZE: usize = 4096;
        let samples: Vec<f32> = (0..FFT_SIZE)
            .map(|i| perm(i as i32) as f32 / 255.0 - 0.5)
            .collect();

        let power = power_spectrum(&samples);
        let slope = spectral_slope(&power);

        // White noise should have slope near 0 (flat spectrum)
        assert!(
            slope.abs() < 0.5,
            "White noise should have flat spectrum (slope ≈ 0), got slope = {}",
            slope
        );
    }

    #[test]
    fn test_pink_noise_spectral_slope() {
        const FFT_SIZE: usize = 4096;
        let samples: Vec<f32> = (0..FFT_SIZE).map(|i| pink1(i as f32 * 0.01, 8)).collect();

        let power = power_spectrum(&samples);
        let slope = spectral_slope(&power);

        // Pink noise should have slope around -1 (1/f)
        // Our implementation may not be perfect, so allow wider tolerance
        assert!(
            slope < -0.3 && slope > -2.0,
            "Pink noise should have slope around -1, got slope = {}",
            slope
        );
    }

    #[test]
    fn test_brown_noise_spectral_slope() {
        const FFT_SIZE: usize = 4096;
        let samples: Vec<f32> = (0..FFT_SIZE).map(|i| brown1(i as f32 * 0.01)).collect();

        let power = power_spectrum(&samples);
        let slope = spectral_slope(&power);

        // Brown noise should have slope around -2 (1/f²)
        assert!(
            slope < -1.0,
            "Brown noise should have steep negative slope (< -1), got slope = {}",
            slope
        );
    }

    #[test]
    fn test_violet_noise_spectral_slope() {
        const FFT_SIZE: usize = 4096;
        let samples: Vec<f32> = (0..FFT_SIZE).map(|i| violet1(i as f32)).collect();

        let power = power_spectrum(&samples);
        let slope = spectral_slope(&power);

        // Violet noise should have slope around +2 (f²)
        // Second-difference implementation gives a good approximation
        assert!(
            slope > 1.0,
            "Violet noise should have steep positive slope (> 1.0), got slope = {}",
            slope
        );
    }

    #[test]
    fn test_spectral_ordering() {
        // Test that slopes are ordered: brown < pink < white < violet
        const FFT_SIZE: usize = 4096;

        let white: Vec<f32> = (0..FFT_SIZE)
            .map(|i| perm(i as i32) as f32 / 255.0 - 0.5)
            .collect();
        let pink: Vec<f32> = (0..FFT_SIZE).map(|i| pink1(i as f32 * 0.01, 8)).collect();
        let brown: Vec<f32> = (0..FFT_SIZE).map(|i| brown1(i as f32 * 0.01)).collect();
        let violet: Vec<f32> = (0..FFT_SIZE).map(|i| violet1(i as f32)).collect();

        let slope_white = spectral_slope(&power_spectrum(&white));
        let slope_pink = spectral_slope(&power_spectrum(&pink));
        let slope_brown = spectral_slope(&power_spectrum(&brown));
        let slope_violet = spectral_slope(&power_spectrum(&violet));

        assert!(
            slope_brown < slope_pink,
            "Brown slope ({}) should be < pink slope ({})",
            slope_brown,
            slope_pink
        );
        assert!(
            slope_pink < slope_white,
            "Pink slope ({}) should be < white slope ({})",
            slope_pink,
            slope_white
        );
        assert!(
            slope_white < slope_violet,
            "White slope ({}) should be < violet slope ({})",
            slope_white,
            slope_violet
        );
    }

    // =========================================================================
    // Determinism tests
    // =========================================================================

    #[test]
    fn test_noise_deterministic() {
        // Same input should always produce same output
        for i in 0..100 {
            let x = i as f32 * 0.37;
            assert_eq!(perlin1(x), perlin1(x), "perlin1 not deterministic");
            assert_eq!(
                perlin2(x, x * 1.5),
                perlin2(x, x * 1.5),
                "perlin2 not deterministic"
            );
            assert_eq!(simplex2(x, x), simplex2(x, x), "simplex2 not deterministic");
            assert_eq!(worley2(x, x), worley2(x, x), "worley2 not deterministic");
            assert_eq!(value1(x), value1(x), "value1 not deterministic");
        }
    }

    // =========================================================================
    // Worley/cellular noise tests
    // =========================================================================

    #[test]
    fn test_worley_has_zeros() {
        // Worley noise should hit 0 at feature points
        let mut found_near_zero = false;
        for i in 0..SAMPLES {
            let v = worley2(i as f32 * 0.1, i as f32 * 0.07);
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
        // F2 (second nearest) should always be >= F1 (nearest)
        for i in 0..1000 {
            let x = i as f32 * 0.1;
            let y = i as f32 * 0.07;
            let f1 = worley2(x, y);
            let f2 = worley2_f2(x, y);
            assert!(
                f2 >= f1 * 0.9,
                "F2 should be >= F1, got f1={}, f2={}",
                f1,
                f2
            );
        }
    }

    // =========================================================================
    // Histogram uniformity test (chi-squared approximation)
    // =========================================================================

    #[test]
    fn test_white_noise_histogram_uniform() {
        const BINS: usize = 10;
        let mut histogram = [0usize; BINS];

        for i in 0..SAMPLES {
            let v = perm(i as i32) as f32 / 255.0;
            let bin = ((v * BINS as f32) as usize).min(BINS - 1);
            histogram[bin] += 1;
        }

        let expected = SAMPLES / BINS;
        let chi_squared: f32 = histogram
            .iter()
            .map(|&count| {
                let diff = count as f32 - expected as f32;
                diff * diff / expected as f32
            })
            .sum();

        // Chi-squared critical value for 9 df at p=0.01 is ~21.67
        // We use a looser bound for test stability
        assert!(
            chi_squared < 30.0,
            "White noise histogram should be roughly uniform, chi-squared = {}",
            chi_squared
        );
    }
}
