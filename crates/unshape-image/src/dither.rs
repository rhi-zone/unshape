use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_color::Rgba;
use unshape_field::{EvalContext, Field};

use crate::ImageField;

/// Quantize a value to discrete levels.
///
/// This is the fundamental primitive for dithering - it rounds a continuous
/// value to the nearest level in a discrete set.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Quantize {
    /// Number of discrete levels (2-256).
    pub levels: u32,
}

impl Quantize {
    /// Creates a new quantizer with the given number of levels.
    pub fn new(levels: u32) -> Self {
        Self {
            levels: levels.clamp(2, 256),
        }
    }

    /// Quantizes a single value to the nearest level.
    #[inline]
    pub fn apply(&self, value: f32) -> f32 {
        let factor = (self.levels - 1) as f32;
        ((value * factor).round() / factor).clamp(0.0, 1.0)
    }

    /// Returns the spread (step size between levels).
    #[inline]
    pub fn spread(&self) -> f32 {
        1.0 / self.levels as f32
    }
}

// -----------------------------------------------------------------------------
// Threshold Fields - Field<Vec2, f32> implementations
// -----------------------------------------------------------------------------

/// Bayer ordered dithering pattern as a field.
///
/// Produces a repeating threshold pattern based on a Bayer matrix.
/// When combined with quantization, creates characteristic crosshatch dithering.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BayerField {
    /// Matrix size (2, 4, or 8).
    pub size: u32,
}

impl BayerField {
    /// Creates a 2x2 Bayer field.
    pub fn bayer2x2() -> Self {
        Self { size: 2 }
    }

    /// Creates a 4x4 Bayer field (default).
    pub fn bayer4x4() -> Self {
        Self { size: 4 }
    }

    /// Creates an 8x8 Bayer field.
    pub fn bayer8x8() -> Self {
        Self { size: 8 }
    }
}

impl Field<Vec2, f32> for BayerField {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Convert UV to pixel coordinates (assume tiling)
        let x = (input.x.abs() * 1000.0) as usize;
        let y = (input.y.abs() * 1000.0) as usize;

        match self.size {
            2 => BAYER_2X2[y % 2][x % 2],
            4 => BAYER_4X4[y % 4][x % 4],
            _ => BAYER_8X8[y % 8][x % 8],
        }
    }
}

/// Blue noise threshold field from a texture.
///
/// Blue noise has optimal spectral properties for dithering - it minimizes
/// low-frequency content while maintaining uniform energy distribution.
#[derive(Clone)]
pub struct BlueNoise2D {
    /// The blue noise texture (grayscale values 0-1).
    pub texture: ImageField,
}

impl BlueNoise2D {
    /// Creates a blue noise field from an existing texture.
    pub fn from_texture(texture: ImageField) -> Self {
        Self { texture }
    }

    /// Generates a new blue noise field of the given size.
    pub fn generate(size: u32) -> Self {
        Self {
            texture: generate_blue_noise_2d(size),
        }
    }
}

impl Field<Vec2, f32> for BlueNoise2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let (w, h) = self.texture.dimensions();
        // Tile the texture
        let x = ((input.x.abs() * w as f32) as u32) % w;
        let y = ((input.y.abs() * h as f32) as u32) % h;
        self.texture.get_pixel(x, y)[0]
    }
}

/// 1D blue noise field.
///
/// Well-distributed noise without clumping. Optimal for audio dithering and sampling.
#[derive(Debug, Clone)]
pub struct BlueNoise1D {
    /// The blue noise samples (values 0-1).
    pub data: Vec<f32>,
}

impl BlueNoise1D {
    /// Creates a blue noise field from existing data.
    pub fn from_data(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Generates a new blue noise field of the given size.
    pub fn generate(size: u32) -> Self {
        Self {
            data: generate_blue_noise_1d(size),
        }
    }
}

impl Field<f32, f32> for BlueNoise1D {
    fn sample(&self, input: f32, _ctx: &EvalContext) -> f32 {
        if self.data.is_empty() {
            return 0.5;
        }
        let size = self.data.len();
        let idx = ((input.abs() * size as f32) as usize) % size;
        self.data[idx]
    }
}

/// 3D blue noise field.
///
/// Well-distributed noise in 3D. Useful for temporally stable animation dithering.
///
/// **Note**: Generation is expensive (O(n³)). Pre-generate and reuse.
#[derive(Debug, Clone)]
pub struct BlueNoise3D {
    /// The blue noise samples (flattened x + y*size + z*size*size).
    pub data: Vec<f32>,
    /// Size of each dimension.
    pub size: u32,
}

impl BlueNoise3D {
    /// Creates a blue noise field from existing data.
    pub fn from_data(data: Vec<f32>, size: u32) -> Self {
        Self { data, size }
    }

    /// Generates a new blue noise field of the given size.
    ///
    /// **Warning**: This is expensive! O(n³) complexity. Size is clamped to 4..=32.
    pub fn generate(size: u32) -> Self {
        let size = size.max(4).min(32);
        Self {
            data: generate_blue_noise_3d(size),
            size,
        }
    }
}

impl Field<Vec3, f32> for BlueNoise3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        if self.data.is_empty() || self.size == 0 {
            return 0.5;
        }
        let size = self.size as usize;
        let x = ((input.x.abs() * size as f32) as usize) % size;
        let y = ((input.y.abs() * size as f32) as usize) % size;
        let z = ((input.z.abs() * size as f32) as usize) % size;
        let idx = x + y * size + z * size * size;
        self.data.get(idx).copied().unwrap_or(0.5)
    }
}

// -----------------------------------------------------------------------------
// QuantizeWithThreshold - composed field operation
// -----------------------------------------------------------------------------

/// Quantizes a color field using a threshold field for dithering.
///
/// This is the core dithering composition: for each position, it samples
/// both the input color and threshold, then quantizes the adjusted value.
///
/// The formula is: `quantize(color + (threshold - 0.5) * spread, levels)`
#[derive(Clone)]
pub struct QuantizeWithThreshold<F, T> {
    /// The input color field.
    pub input: F,
    /// The threshold field (values 0-1).
    pub threshold: T,
    /// Number of quantization levels.
    pub levels: u32,
}

impl<F, T> QuantizeWithThreshold<F, T> {
    /// Creates a new quantize-with-threshold field.
    pub fn new(input: F, threshold: T, levels: u32) -> Self {
        Self {
            input,
            threshold,
            levels: levels.clamp(2, 256),
        }
    }
}

impl<F, T> Field<Vec2, Rgba> for QuantizeWithThreshold<F, T>
where
    F: Field<Vec2, Rgba>,
    T: Field<Vec2, f32>,
{
    fn sample(&self, pos: Vec2, ctx: &EvalContext) -> Rgba {
        let color = self.input.sample(pos, ctx);
        let thresh = self.threshold.sample(pos, ctx);
        let quantize = Quantize::new(self.levels);
        let offset = (thresh - 0.5) * quantize.spread();

        Rgba::new(
            quantize.apply(color.r + offset),
            quantize.apply(color.g + offset),
            quantize.apply(color.b + offset),
            color.a,
        )
    }
}

// -----------------------------------------------------------------------------
// Error Diffusion - sequential operations (not fields)
// -----------------------------------------------------------------------------

/// Error diffusion kernel for dithering.
///
/// Each kernel defines how quantization error is distributed to neighboring pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DiffusionKernel {
    /// Floyd-Steinberg - classic, high quality.
    #[default]
    FloydSteinberg,
    /// Atkinson - lighter, preserves detail (Mac classic look).
    Atkinson,
    /// Sierra - smooth gradients.
    Sierra,
    /// Sierra Two-Row - faster variant.
    SierraTwoRow,
    /// Sierra Lite - fastest variant.
    SierraLite,
    /// Jarvis-Judice-Ninke - very smooth, large kernel.
    JarvisJudiceNinke,
    /// Stucki - sharper than JJN.
    Stucki,
    /// Burkes - simplified Stucki.
    Burkes,
}

/// 2x2 Bayer threshold matrix (normalized to 0-1).
const BAYER_2X2: [[f32; 2]; 2] = [[0.0 / 4.0, 2.0 / 4.0], [3.0 / 4.0, 1.0 / 4.0]];

/// 4x4 Bayer threshold matrix (normalized to 0-1).
const BAYER_4X4: [[f32; 4]; 4] = [
    [0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0],
    [12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0],
    [3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0],
    [15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0],
];

/// 8x8 Bayer threshold matrix (normalized to 0-1).
const BAYER_8X8: [[f32; 8]; 8] = [
    [
        0.0 / 64.0,
        32.0 / 64.0,
        8.0 / 64.0,
        40.0 / 64.0,
        2.0 / 64.0,
        34.0 / 64.0,
        10.0 / 64.0,
        42.0 / 64.0,
    ],
    [
        48.0 / 64.0,
        16.0 / 64.0,
        56.0 / 64.0,
        24.0 / 64.0,
        50.0 / 64.0,
        18.0 / 64.0,
        58.0 / 64.0,
        26.0 / 64.0,
    ],
    [
        12.0 / 64.0,
        44.0 / 64.0,
        4.0 / 64.0,
        36.0 / 64.0,
        14.0 / 64.0,
        46.0 / 64.0,
        6.0 / 64.0,
        38.0 / 64.0,
    ],
    [
        60.0 / 64.0,
        28.0 / 64.0,
        52.0 / 64.0,
        20.0 / 64.0,
        62.0 / 64.0,
        30.0 / 64.0,
        54.0 / 64.0,
        22.0 / 64.0,
    ],
    [
        3.0 / 64.0,
        35.0 / 64.0,
        11.0 / 64.0,
        43.0 / 64.0,
        1.0 / 64.0,
        33.0 / 64.0,
        9.0 / 64.0,
        41.0 / 64.0,
    ],
    [
        51.0 / 64.0,
        19.0 / 64.0,
        59.0 / 64.0,
        27.0 / 64.0,
        49.0 / 64.0,
        17.0 / 64.0,
        57.0 / 64.0,
        25.0 / 64.0,
    ],
    [
        15.0 / 64.0,
        47.0 / 64.0,
        7.0 / 64.0,
        39.0 / 64.0,
        13.0 / 64.0,
        45.0 / 64.0,
        5.0 / 64.0,
        37.0 / 64.0,
    ],
    [
        63.0 / 64.0,
        31.0 / 64.0,
        55.0 / 64.0,
        23.0 / 64.0,
        61.0 / 64.0,
        29.0 / 64.0,
        53.0 / 64.0,
        21.0 / 64.0,
    ],
];

/// Error diffusion kernel entry: (dx, dy, weight).
type DiffusionEntry = (i32, i32, f32);

impl DiffusionKernel {
    /// Returns the diffusion coefficients for this kernel.
    fn coefficients(&self) -> &'static [DiffusionEntry] {
        match self {
            Self::FloydSteinberg => &[
                (1, 0, 7.0 / 16.0),
                (-1, 1, 3.0 / 16.0),
                (0, 1, 5.0 / 16.0),
                (1, 1, 1.0 / 16.0),
            ],
            Self::Atkinson => &[
                (1, 0, 1.0 / 8.0),
                (2, 0, 1.0 / 8.0),
                (-1, 1, 1.0 / 8.0),
                (0, 1, 1.0 / 8.0),
                (1, 1, 1.0 / 8.0),
                (0, 2, 1.0 / 8.0),
            ],
            Self::Sierra => &[
                (1, 0, 5.0 / 32.0),
                (2, 0, 3.0 / 32.0),
                (-2, 1, 2.0 / 32.0),
                (-1, 1, 4.0 / 32.0),
                (0, 1, 5.0 / 32.0),
                (1, 1, 4.0 / 32.0),
                (2, 1, 2.0 / 32.0),
                (-1, 2, 2.0 / 32.0),
                (0, 2, 3.0 / 32.0),
                (1, 2, 2.0 / 32.0),
            ],
            Self::SierraTwoRow => &[
                (1, 0, 4.0 / 16.0),
                (2, 0, 3.0 / 16.0),
                (-2, 1, 1.0 / 16.0),
                (-1, 1, 2.0 / 16.0),
                (0, 1, 3.0 / 16.0),
                (1, 1, 2.0 / 16.0),
                (2, 1, 1.0 / 16.0),
            ],
            Self::SierraLite => &[(1, 0, 2.0 / 4.0), (-1, 1, 1.0 / 4.0), (0, 1, 1.0 / 4.0)],
            Self::JarvisJudiceNinke => &[
                (1, 0, 7.0 / 48.0),
                (2, 0, 5.0 / 48.0),
                (-2, 1, 3.0 / 48.0),
                (-1, 1, 5.0 / 48.0),
                (0, 1, 7.0 / 48.0),
                (1, 1, 5.0 / 48.0),
                (2, 1, 3.0 / 48.0),
                (-2, 2, 1.0 / 48.0),
                (-1, 2, 3.0 / 48.0),
                (0, 2, 5.0 / 48.0),
                (1, 2, 3.0 / 48.0),
                (2, 2, 1.0 / 48.0),
            ],
            Self::Stucki => &[
                (1, 0, 8.0 / 42.0),
                (2, 0, 4.0 / 42.0),
                (-2, 1, 2.0 / 42.0),
                (-1, 1, 4.0 / 42.0),
                (0, 1, 8.0 / 42.0),
                (1, 1, 4.0 / 42.0),
                (2, 1, 2.0 / 42.0),
                (-2, 2, 1.0 / 42.0),
                (-1, 2, 2.0 / 42.0),
                (0, 2, 4.0 / 42.0),
                (1, 2, 2.0 / 42.0),
                (2, 2, 1.0 / 42.0),
            ],
            Self::Burkes => &[
                (1, 0, 8.0 / 32.0),
                (2, 0, 4.0 / 32.0),
                (-2, 1, 2.0 / 32.0),
                (-1, 1, 4.0 / 32.0),
                (0, 1, 8.0 / 32.0),
                (1, 1, 4.0 / 32.0),
                (2, 1, 2.0 / 32.0),
            ],
        }
    }
}

/// Error diffusion dithering operation.
///
/// This is a sequential operation (not a field) because each pixel's output
/// depends on previously processed pixels.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ErrorDiffuse {
    /// The diffusion kernel to use.
    pub kernel: DiffusionKernel,
    /// Number of quantization levels.
    pub levels: u32,
}

impl ErrorDiffuse {
    /// Creates a new error diffusion operation.
    pub fn new(kernel: DiffusionKernel, levels: u32) -> Self {
        Self {
            kernel,
            levels: levels.clamp(2, 256),
        }
    }

    /// Floyd-Steinberg error diffusion.
    pub fn floyd_steinberg(levels: u32) -> Self {
        Self::new(DiffusionKernel::FloydSteinberg, levels)
    }

    /// Atkinson error diffusion.
    pub fn atkinson(levels: u32) -> Self {
        Self::new(DiffusionKernel::Atkinson, levels)
    }

    /// Applies error diffusion to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        error_diffuse_impl(image, self.kernel, self.levels)
    }
}

/// Internal implementation of error diffusion.
fn error_diffuse_impl(image: &ImageField, kernel: DiffusionKernel, levels: u32) -> ImageField {
    let (width, height) = image.dimensions();
    let quantize = Quantize::new(levels);
    let coeffs = kernel.coefficients();

    let mut buffer: Vec<[f32; 3]> = Vec::with_capacity((width * height) as usize);
    let mut alphas: Vec<f32> = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let p = image.get_pixel(x, y);
            buffer.push([p[0], p[1], p[2]]);
            alphas.push(p[3]);
        }
    }

    let mut output = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let old_pixel = buffer[idx];

            let new_pixel = [
                quantize.apply(old_pixel[0]),
                quantize.apply(old_pixel[1]),
                quantize.apply(old_pixel[2]),
            ];

            let error = [
                old_pixel[0] - new_pixel[0],
                old_pixel[1] - new_pixel[1],
                old_pixel[2] - new_pixel[2],
            ];

            for &(dx, dy, weight) in coeffs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny < height as i32 {
                    let nidx = (ny as u32 * width + nx as u32) as usize;
                    buffer[nidx][0] += error[0] * weight;
                    buffer[nidx][1] += error[1] * weight;
                    buffer[nidx][2] += error[2] * weight;
                }
            }

            output.push([new_pixel[0], new_pixel[1], new_pixel[2], alphas[idx]]);
        }
    }

    ImageField::from_raw(output, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

// -----------------------------------------------------------------------------
// Curve-based Diffusion (Riemersma)
// -----------------------------------------------------------------------------

/// Traversal curve for curve-based dithering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TraversalCurve {
    /// Hilbert space-filling curve.
    #[default]
    Hilbert,
}

/// Curve-based error diffusion (Riemersma dithering).
///
/// Uses a space-filling curve instead of scanline order, eliminating
/// directional artifacts common in traditional error diffusion.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurveDiffuse {
    /// The traversal curve to use.
    pub curve: TraversalCurve,
    /// Size of the error history buffer.
    pub history_size: usize,
    /// Decay ratio for error weights (0-1, smaller = faster decay).
    pub decay: f32,
    /// Number of quantization levels.
    pub levels: u32,
}

impl Default for CurveDiffuse {
    fn default() -> Self {
        Self {
            curve: TraversalCurve::Hilbert,
            history_size: 16,
            decay: 1.0 / 8.0,
            levels: 2,
        }
    }
}

impl CurveDiffuse {
    /// Creates a new curve diffusion operation (Riemersma dithering).
    pub fn new(levels: u32) -> Self {
        Self {
            levels: levels.clamp(2, 256),
            ..Default::default()
        }
    }

    /// Sets the history size.
    pub fn with_history_size(mut self, size: usize) -> Self {
        self.history_size = size.max(1);
        self
    }

    /// Sets the decay ratio.
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(0.001, 1.0);
        self
    }

    /// Applies curve-based diffusion to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        curve_diffuse_impl(image, self)
    }
}

/// Internal implementation of curve-based diffusion.
fn curve_diffuse_impl(image: &ImageField, config: &CurveDiffuse) -> ImageField {
    let (width, height) = image.dimensions();
    let quantize = Quantize::new(config.levels);

    let mut data: Vec<[f32; 4]> = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(image.get_pixel(x, y));
        }
    }

    // Precompute weights with exponential falloff
    let weights: Vec<f32> = (0..config.history_size)
        .map(|i| {
            config
                .decay
                .powf(i as f32 / (config.history_size - 1).max(1) as f32)
        })
        .collect();
    let weight_sum: f32 = weights.iter().sum();

    // Error history buffer (ring buffer)
    let mut error_history_r: Vec<f32> = vec![0.0; config.history_size];
    let mut error_history_g: Vec<f32> = vec![0.0; config.history_size];
    let mut error_history_b: Vec<f32> = vec![0.0; config.history_size];
    let mut history_idx = 0usize;

    // Generate curve path
    let curve_order = (width.max(height) as f32).log2().ceil() as u32;
    let curve_size = 1u32 << curve_order;

    let total_points = curve_size * curve_size;
    for d in 0..total_points {
        let (hx, hy) = hilbert_d2xy(curve_order, d);

        if hx >= width || hy >= height {
            continue;
        }

        let idx = (hy * width + hx) as usize;
        let pixel = data[idx];

        // Calculate weighted error sum from history
        let mut error_sum_r = 0.0f32;
        let mut error_sum_g = 0.0f32;
        let mut error_sum_b = 0.0f32;

        for i in 0..config.history_size {
            let hist_i = (history_idx + config.history_size - 1 - i) % config.history_size;
            error_sum_r += error_history_r[hist_i] * weights[i];
            error_sum_g += error_history_g[hist_i] * weights[i];
            error_sum_b += error_history_b[hist_i] * weights[i];
        }

        // Apply error and quantize
        let adjusted_r = pixel[0] + error_sum_r / weight_sum;
        let adjusted_g = pixel[1] + error_sum_g / weight_sum;
        let adjusted_b = pixel[2] + error_sum_b / weight_sum;

        let quantized_r = quantize.apply(adjusted_r);
        let quantized_g = quantize.apply(adjusted_g);
        let quantized_b = quantize.apply(adjusted_b);

        data[idx] = [quantized_r, quantized_g, quantized_b, pixel[3]];

        // Store new errors in history
        error_history_r[history_idx] = pixel[0] - quantized_r;
        error_history_g[history_idx] = pixel[1] - quantized_g;
        error_history_b[history_idx] = pixel[2] - quantized_b;
        history_idx = (history_idx + 1) % config.history_size;
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Convert Hilbert curve index to (x, y) coordinates.
pub(crate) fn hilbert_d2xy(order: u32, d: u32) -> (u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    let mut d = d;
    let mut s = 1u32;

    while s < (1 << order) {
        let rx = (d / 2) & 1;
        let ry = (d ^ rx) & 1;

        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }

        x += s * rx;
        y += s * ry;
        d /= 4;
        s *= 2;
    }

    (x, y)
}

// -----------------------------------------------------------------------------
// Werness Dithering (Obra Dinn style)
// -----------------------------------------------------------------------------

/// Werness dithering - hybrid noise-threshold + error absorption.
///
/// Invented by Brent Werness for Return of the Obra Dinn.
/// Each pixel absorbs weighted errors from neighbors across multiple phases.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WernessDither {
    /// Number of quantization levels.
    pub levels: u32,
    /// Number of iterations.
    pub iterations: u32,
}

impl WernessDither {
    /// Creates a new Werness dither operation.
    pub fn new(levels: u32) -> Self {
        Self {
            levels: levels.clamp(2, 256),
            iterations: 4,
        }
    }

    /// Sets the number of iterations.
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations.max(1);
        self
    }

    /// Applies Werness dithering to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        werness_impl(image, self)
    }
}

/// Internal implementation of Werness dithering.
fn werness_impl(image: &ImageField, config: &WernessDither) -> ImageField {
    let (width, height) = image.dimensions();
    let quantize = Quantize::new(config.levels);

    // Initialize with image luminance + noise seeding
    let mut values: Vec<f32> = Vec::with_capacity((width * height) as usize);
    let mut alphas: Vec<f32> = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let v = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];

            // Add noise seeding
            let noise = fract(52.9829189 * fract(0.06711056 * x as f32 + 0.00583715 * y as f32));
            let seeded = v + (noise - 0.5) * 0.1;

            values.push(seeded);
            alphas.push(pixel[3]);
        }
    }

    let mut output: Vec<f32> = vec![0.0; (width * height) as usize];
    let mut errors: Vec<f32> = vec![0.0; (width * height) as usize];

    // Absorption kernel
    let kernel: &[(i32, i32, f32)] = &[
        (1, 0, 1.0 / 8.0),
        (2, 0, 1.0 / 8.0),
        (-1, 1, 1.0 / 8.0),
        (0, 1, 1.0 / 8.0),
        (1, 1, 1.0 / 8.0),
        (0, 2, 1.0 / 8.0),
        (-1, 0, 1.0 / 8.0),
        (-2, 0, 1.0 / 8.0),
        (1, -1, 1.0 / 8.0),
        (0, -1, 1.0 / 8.0),
        (-1, -1, 1.0 / 8.0),
        (0, -2, 1.0 / 8.0),
    ];

    for iteration in 0..config.iterations {
        for phase_y in 0..3i32 {
            for phase_x in 0..3i32 {
                let mut y = phase_y as u32;
                while y < height {
                    let mut x = phase_x as u32;
                    while x < width {
                        let idx = (y * width + x) as usize;

                        let mut error_sum = 0.0f32;
                        for &(dx, dy, weight) in kernel {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                let nidx = (ny as u32 * width + nx as u32) as usize;
                                error_sum += errors[nidx] * weight;
                            }
                        }

                        let adjusted = if iteration == 0 {
                            values[idx] + error_sum
                        } else {
                            output[idx] + error_sum
                        };

                        let quantized = quantize.apply(adjusted);
                        output[idx] = quantized;
                        errors[idx] = values[idx] - quantized;

                        x += 3;
                    }
                    y += 3;
                }
            }
        }
    }

    // Build final image (grayscale)
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let v = output[idx];
            data.push([v, v, v, alphas[idx]]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Helper: fractional part of a float.
#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

// -----------------------------------------------------------------------------
// Blue Noise Generation
// -----------------------------------------------------------------------------

/// Generates a 2D blue noise texture using the void-and-cluster algorithm.
///
/// Blue noise has optimal spectral properties for dithering - it minimizes
/// low-frequency content while maintaining uniform energy distribution.
///
/// # Arguments
/// * `size` - Width and height of the texture (should be power of 2)
///
/// # Note
/// This is a simplified implementation. For production use, consider
/// precomputed blue noise textures which have better quality.
pub fn generate_blue_noise_2d(size: u32) -> ImageField {
    let size = size.max(4).min(256);
    let total = (size * size) as usize;

    // Initialize with random binary pattern
    let mut pattern: Vec<bool> = (0..total)
        .map(|i| (i * 7919 + i * i * 104729) % total < total / 2)
        .collect();

    // Void-and-cluster iterations to improve blue noise quality
    let iterations = 10;
    for _ in 0..iterations {
        // Find tightest cluster (densest area of 1s)
        let cluster_idx = find_tightest_cluster(&pattern, size);
        pattern[cluster_idx] = false;

        // Find largest void (sparsest area of 1s)
        let void_idx = find_largest_void(&pattern, size);
        pattern[void_idx] = true;
    }

    // Convert binary pattern to ranking
    let mut ranking = vec![0usize; total];

    // Remove pixels one by one, recording removal order
    let mut temp_pattern = pattern.clone();
    for i in 0..total / 2 {
        let idx = find_tightest_cluster(&temp_pattern, size);
        temp_pattern[idx] = false;
        ranking[idx] = total / 2 - 1 - i;
    }

    // Add pixels one by one, recording addition order
    temp_pattern = pattern;
    for p in &mut temp_pattern {
        *p = !*p;
    }
    for i in 0..total / 2 {
        let idx = find_largest_void(&temp_pattern, size);
        temp_pattern[idx] = true;
        ranking[idx] = total / 2 + i;
    }

    // Convert ranking to grayscale image
    let data: Vec<[f32; 4]> = ranking
        .iter()
        .map(|&r| {
            let v = r as f32 / total as f32;
            [v, v, v, 1.0]
        })
        .collect();

    ImageField::from_raw(data, size, size)
}

/// Find the index of the tightest cluster (highest local density of 1s).
fn find_tightest_cluster(pattern: &[bool], size: u32) -> usize {
    let mut max_density = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if is_set {
            let density = calculate_density(pattern, i, size, true);
            if density > max_density {
                max_density = density;
                max_idx = i;
            }
        }
    }
    max_idx
}

/// Find the index of the largest void (lowest local density of 1s).
fn find_largest_void(pattern: &[bool], size: u32) -> usize {
    let mut min_density = f32::INFINITY;
    let mut min_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if !is_set {
            let density = calculate_density(pattern, i, size, false);
            if density < min_density {
                min_density = density;
                min_idx = i;
            }
        }
    }
    min_idx
}

/// Calculate local density around a pixel using Gaussian weighting.
fn calculate_density(pattern: &[bool], center_idx: usize, size: u32, include_self: bool) -> f32 {
    let cx = (center_idx % size as usize) as i32;
    let cy = (center_idx / size as usize) as i32;
    let size_i = size as i32;

    let sigma = 1.5f32;
    let sigma_sq_2 = 2.0 * sigma * sigma;
    let mut density = 0.0f32;

    // Sample neighborhood
    let radius = 3i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if !include_self && dx == 0 && dy == 0 {
                continue;
            }

            // Toroidal wrapping
            let nx = ((cx + dx) % size_i + size_i) % size_i;
            let ny = ((cy + dy) % size_i + size_i) % size_i;
            let idx = (ny * size_i + nx) as usize;

            if pattern[idx] {
                let dist_sq = (dx * dx + dy * dy) as f32;
                density += (-dist_sq / sigma_sq_2).exp();
            }
        }
    }

    density
}

/// Generate 1D blue noise as a Vec<f32>.
///
/// Blue noise in 1D produces well-distributed random values without clumping.
/// Useful for audio dithering and 1D sampling patterns.
///
/// # Arguments
///
/// * `size` - Number of samples (clamped to 4..=4096)
///
/// # Returns
///
/// Vector of f32 values in [0, 1] with blue noise distribution.
pub fn generate_blue_noise_1d(size: u32) -> Vec<f32> {
    let size = size.max(4).min(4096) as usize;

    // Initialize with random binary pattern
    let mut pattern: Vec<bool> = (0..size)
        .map(|i| (i * 7919 + i * i * 104729) % size < size / 2)
        .collect();

    // Void-and-cluster iterations
    let iterations = 10;
    for _ in 0..iterations {
        // Find tightest cluster
        let cluster_idx = find_tightest_cluster_1d(&pattern);
        pattern[cluster_idx] = false;

        // Find largest void
        let void_idx = find_largest_void_1d(&pattern);
        pattern[void_idx] = true;
    }

    // Convert to ranking
    let mut ranking = vec![0usize; size];
    let mut temp_pattern = pattern.clone();

    for i in 0..size / 2 {
        let idx = find_tightest_cluster_1d(&temp_pattern);
        temp_pattern[idx] = false;
        ranking[idx] = size / 2 - 1 - i;
    }

    temp_pattern = pattern;
    for i in 0..size - size / 2 {
        let idx = find_largest_void_1d(&temp_pattern);
        temp_pattern[idx] = true;
        ranking[idx] = size / 2 + i;
    }

    ranking.iter().map(|&r| r as f32 / size as f32).collect()
}

fn find_tightest_cluster_1d(pattern: &[bool]) -> usize {
    let mut max_density = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if is_set {
            let density = calculate_density_1d(pattern, i, true);
            if density > max_density {
                max_density = density;
                max_idx = i;
            }
        }
    }
    if max_idx == 0 && max_density == f32::NEG_INFINITY {
        // Fallback: find any set bit
        pattern.iter().position(|&b| b).unwrap_or(0)
    } else {
        max_idx
    }
}

fn find_largest_void_1d(pattern: &[bool]) -> usize {
    let mut min_density = f32::INFINITY;
    let mut min_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if !is_set {
            let density = calculate_density_1d(pattern, i, false);
            if density < min_density {
                min_density = density;
                min_idx = i;
            }
        }
    }
    if min_idx == 0 && min_density == f32::INFINITY {
        // Fallback: find any unset bit
        pattern.iter().position(|&b| !b).unwrap_or(0)
    } else {
        min_idx
    }
}

fn calculate_density_1d(pattern: &[bool], center: usize, include_self: bool) -> f32 {
    let size = pattern.len() as i32;
    let sigma = 1.5f32;
    let sigma_sq_2 = 2.0 * sigma * sigma;
    let mut density = 0.0f32;

    let radius = 5i32;
    for d in -radius..=radius {
        if !include_self && d == 0 {
            continue;
        }
        // Toroidal wrapping
        let idx = ((center as i32 + d) % size + size) % size;
        if pattern[idx as usize] {
            let dist_sq = (d * d) as f32;
            density += (-dist_sq / sigma_sq_2).exp();
        }
    }
    density
}

/// Generate 3D blue noise.
///
/// **WARNING**: This is computationally expensive! O(n³) complexity.
/// For a 32x32x32 volume, this processes 32,768 voxels.
/// Consider using pre-computed blue noise textures for production.
///
/// # Arguments
///
/// * `size` - Size of each dimension (clamped to 4..=32 due to cost)
///
/// # Returns
///
/// 3D array of f32 values in [0, 1] as a flattened Vec (x + y*size + z*size*size).
pub fn generate_blue_noise_3d(size: u32) -> Vec<f32> {
    // Clamp to reasonable sizes - 3D is very expensive
    let size = size.max(4).min(32) as usize;
    let total = size * size * size;

    // Initialize with random binary pattern
    let mut pattern: Vec<bool> = (0..total)
        .map(|i| (i * 7919 + i * i * 104729) % total < total / 2)
        .collect();

    // Fewer iterations for 3D due to cost
    let iterations = 5;
    for _ in 0..iterations {
        let cluster_idx = find_tightest_cluster_3d(&pattern, size);
        pattern[cluster_idx] = false;

        let void_idx = find_largest_void_3d(&pattern, size);
        pattern[void_idx] = true;
    }

    // Convert to ranking
    let mut ranking = vec![0usize; total];
    let mut temp_pattern = pattern.clone();

    for i in 0..total / 2 {
        let idx = find_tightest_cluster_3d(&temp_pattern, size);
        temp_pattern[idx] = false;
        ranking[idx] = total / 2 - 1 - i;
    }

    temp_pattern = pattern;
    for i in 0..total - total / 2 {
        let idx = find_largest_void_3d(&temp_pattern, size);
        temp_pattern[idx] = true;
        ranking[idx] = total / 2 + i;
    }

    ranking.iter().map(|&r| r as f32 / total as f32).collect()
}

fn find_tightest_cluster_3d(pattern: &[bool], size: usize) -> usize {
    let mut max_density = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if is_set {
            let density = calculate_density_3d(pattern, i, size, true);
            if density > max_density {
                max_density = density;
                max_idx = i;
            }
        }
    }
    if max_idx == 0 && max_density == f32::NEG_INFINITY {
        pattern.iter().position(|&b| b).unwrap_or(0)
    } else {
        max_idx
    }
}

fn find_largest_void_3d(pattern: &[bool], size: usize) -> usize {
    let mut min_density = f32::INFINITY;
    let mut min_idx = 0;

    for (i, &is_set) in pattern.iter().enumerate() {
        if !is_set {
            let density = calculate_density_3d(pattern, i, size, false);
            if density < min_density {
                min_density = density;
                min_idx = i;
            }
        }
    }
    if min_idx == 0 && min_density == f32::INFINITY {
        pattern.iter().position(|&b| !b).unwrap_or(0)
    } else {
        min_idx
    }
}

fn calculate_density_3d(
    pattern: &[bool],
    center_idx: usize,
    size: usize,
    include_self: bool,
) -> f32 {
    let size_i = size as i32;
    let cx = (center_idx % size) as i32;
    let cy = ((center_idx / size) % size) as i32;
    let cz = (center_idx / (size * size)) as i32;

    let sigma = 1.5f32;
    let sigma_sq_2 = 2.0 * sigma * sigma;
    let mut density = 0.0f32;

    // Smaller radius for 3D to keep it tractable
    let radius = 2i32;
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if !include_self && dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }

                // Toroidal wrapping
                let nx = ((cx + dx) % size_i + size_i) % size_i;
                let ny = ((cy + dy) % size_i + size_i) % size_i;
                let nz = ((cz + dz) % size_i + size_i) % size_i;
                let idx = (nz * size_i * size_i + ny * size_i + nx) as usize;

                if pattern[idx] {
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    density += (-dist_sq / sigma_sq_2).exp();
                }
            }
        }
    }
    density
}

// -----------------------------------------------------------------------------
// Temporal Dithering
// -----------------------------------------------------------------------------

/// Bayer dithering pattern with temporal offset for animation.
///
/// Each frame uses a different offset into the Bayer pattern, reducing
/// temporal flickering when frames are viewed in sequence.
///
/// The offset cycles through all positions in the Bayer matrix over `size²` frames.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalBayer {
    /// Matrix size (2, 4, or 8).
    pub size: u32,
    /// Current frame index.
    pub frame: u32,
}

impl TemporalBayer {
    /// Creates a temporal Bayer field with given size and frame.
    pub fn new(size: u32, frame: u32) -> Self {
        let size = match size {
            0..=2 => 2,
            3..=5 => 4,
            _ => 8,
        };
        Self { size, frame }
    }

    /// Creates a 4x4 temporal Bayer field (default size).
    pub fn bayer4x4(frame: u32) -> Self {
        Self::new(4, frame)
    }

    /// Creates an 8x8 temporal Bayer field.
    pub fn bayer8x8(frame: u32) -> Self {
        Self::new(8, frame)
    }
}

impl Field<Vec2, f32> for TemporalBayer {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let size = self.size as usize;

        // Convert UV to pixel coordinates
        let px = (input.x.abs() * 1000.0) as usize;
        let py = (input.y.abs() * 1000.0) as usize;

        // Temporal offset: shift pattern position each frame
        let frame_offset = self.frame as usize;
        let x = (px + frame_offset) % size;
        let y = (py + frame_offset / size) % size;

        match self.size {
            2 => BAYER_2X2[y % 2][x % 2],
            4 => BAYER_4X4[y % 4][x % 4],
            _ => BAYER_8X8[y % 8][x % 8],
        }
    }
}

/// Interleaved Gradient Noise (IGN) for temporal dithering.
///
/// A low-discrepancy noise pattern commonly used in real-time graphics.
/// Produces well-distributed noise that varies smoothly with frame index,
/// making it ideal for temporal anti-aliasing and dithering in animation.
///
/// Based on Jorge Jimenez's algorithm from "Next Generation Post Processing in Call of Duty".
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InterleavedGradientNoise {
    /// Current frame index.
    pub frame: u32,
}

impl InterleavedGradientNoise {
    /// Creates a new IGN field for the given frame.
    pub fn new(frame: u32) -> Self {
        Self { frame }
    }
}

impl Field<Vec2, f32> for InterleavedGradientNoise {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Convert UV to pixel coordinates (assume 1000x1000 resolution for consistency)
        let x = input.x.abs() * 1000.0;
        let y = input.y.abs() * 1000.0;

        // Temporal rotation using golden ratio
        let frame_offset = self.frame as f32 * 5.588238;
        let rotated_x = x + frame_offset;
        let rotated_y = y + frame_offset;

        // IGN formula: fract(52.9829189 * fract(0.06711056 * x + 0.00583715 * y))
        fract(52.9829189 * fract(0.06711056 * rotated_x + 0.00583715 * rotated_y))
    }
}

/// Temporal blue noise dithering using 3D blue noise with frame as Z coordinate.
///
/// This wrapper provides explicit frame-based access to `BlueNoise3D`,
/// making the temporal dithering use case clearer.
///
/// Blue noise has optimal spectral properties - the pattern varies per-frame
/// but maintains consistent distribution, minimizing visible flickering.
#[derive(Debug, Clone)]
pub struct TemporalBlueNoise {
    /// The underlying 3D blue noise.
    pub noise: BlueNoise3D,
    /// Current frame index.
    pub frame: u32,
}

impl TemporalBlueNoise {
    /// Creates a temporal blue noise field from existing 3D noise.
    pub fn from_noise(noise: BlueNoise3D, frame: u32) -> Self {
        Self { noise, frame }
    }

    /// Generates a new temporal blue noise field.
    ///
    /// **Note**: Generation is expensive. Pre-generate and reuse the `BlueNoise3D`.
    pub fn generate(size: u32, frame: u32) -> Self {
        Self {
            noise: BlueNoise3D::generate(size),
            frame,
        }
    }

    /// Returns a new instance for a different frame (shares the noise data).
    pub fn at_frame(&self, frame: u32) -> Self {
        Self {
            noise: self.noise.clone(),
            frame,
        }
    }
}

impl Field<Vec2, f32> for TemporalBlueNoise {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        // Map frame to z coordinate, wrapping at noise size
        let z = (self.frame % self.noise.size) as f32 / self.noise.size as f32;
        self.noise.sample(Vec3::new(input.x, input.y, z), ctx)
    }
}

/// Quantizes with temporal dithering threshold.
///
/// Like `QuantizeWithThreshold` but takes a frame parameter for temporal variation.
/// This reduces temporal flickering in animated sequences by ensuring the
/// dithering pattern changes smoothly between frames.
#[derive(Clone)]
pub struct QuantizeWithTemporalThreshold<F, T> {
    /// The input color field.
    pub input: F,
    /// The threshold field (takes Vec3 where z = frame/total_frames).
    pub threshold: T,
    /// Number of quantization levels.
    pub levels: u32,
    /// Current frame index.
    pub frame: u32,
    /// Total frames in the animation (for z-coordinate normalization).
    pub total_frames: u32,
}

impl<F, T> QuantizeWithTemporalThreshold<F, T> {
    /// Creates a new temporal quantize operation.
    ///
    /// # Arguments
    /// * `input` - The color field to quantize
    /// * `threshold` - A 3D threshold field (x, y, frame_normalized)
    /// * `levels` - Number of quantization levels
    /// * `frame` - Current frame (0-indexed)
    /// * `total_frames` - Total frames in animation (used to normalize z)
    pub fn new(input: F, threshold: T, levels: u32, frame: u32, total_frames: u32) -> Self {
        Self {
            input,
            threshold,
            levels: levels.clamp(2, 256),
            frame,
            total_frames: total_frames.max(1),
        }
    }
}

impl<F, T> Field<Vec2, Rgba> for QuantizeWithTemporalThreshold<F, T>
where
    F: Field<Vec2, Rgba>,
    T: Field<Vec3, f32>,
{
    fn sample(&self, pos: Vec2, ctx: &EvalContext) -> Rgba {
        let color = self.input.sample(pos, ctx);

        // Sample threshold with frame as z-coordinate
        let z = self.frame as f32 / self.total_frames as f32;
        let thresh = self.threshold.sample(Vec3::new(pos.x, pos.y, z), ctx);

        let quantize = Quantize::new(self.levels);
        let offset = (thresh - 0.5) * quantize.spread();

        Rgba::new(
            quantize.apply(color.r + offset),
            quantize.apply(color.g + offset),
            quantize.apply(color.b + offset),
            color.a,
        )
    }
}
