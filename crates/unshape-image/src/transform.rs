use glam::{Mat3, Mat4, Vec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::expr::{ColorExpr, map_pixels, remap_uv_fn};
use crate::{FilterMode, ImageField};

/// Apply a 4x4 color matrix transform to an image.
///
/// The matrix transforms RGBA values: `[r', g', b', a'] = matrix * [r, g, b, a]`.
/// This can be used for color correction, channel mixing, sepia tones, etc.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, color_matrix};
/// use glam::Mat4;
///
/// let image = ImageField::from_raw(vec![[1.0, 0.5, 0.25, 1.0]; 16], 4, 4);
///
/// // Grayscale conversion matrix (luminance weights)
/// let grayscale = Mat4::from_cols_array(&[
///     0.299, 0.299, 0.299, 0.0,
///     0.587, 0.587, 0.587, 0.0,
///     0.114, 0.114, 0.114, 0.0,
///     0.0,   0.0,   0.0,   1.0,
/// ]);
///
/// let result = color_matrix(&image, grayscale);
/// ```
pub fn color_matrix(image: &ImageField, matrix: Mat4) -> ImageField {
    // Convert column-major Mat4 to row-major [[f32; 4]; 4] for ColorExpr
    // transpose() + to_cols_array_2d() gives us the rows as arrays
    let m = matrix.transpose().to_cols_array_2d();
    map_pixels(image, &ColorExpr::matrix(m))
}

/// Configuration for image position transformation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransformConfig {
    /// 3x3 transformation matrix for UV coordinates.
    /// Treats UV as homogeneous coordinates [u, v, 1].
    pub matrix: [[f32; 3]; 3],
    /// Whether to use bilinear filtering regardless of image setting.
    pub filter: bool,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            filter: true,
        }
    }
}

impl TransformConfig {
    /// Create an identity transform.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create a translation transform.
    pub fn translate(dx: f32, dy: f32) -> Self {
        Self {
            matrix: [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            filter: true,
        }
    }

    /// Create a scale transform around the center.
    pub fn scale(sx: f32, sy: f32) -> Self {
        // Scale around center: translate to origin, scale, translate back
        Self {
            matrix: [
                [sx, 0.0, 0.5 - 0.5 * sx],
                [0.0, sy, 0.5 - 0.5 * sy],
                [0.0, 0.0, 1.0],
            ],
            filter: true,
        }
    }

    /// Create a rotation transform around the center (radians).
    pub fn rotate(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        // Rotate around center: translate to origin, rotate, translate back
        Self {
            matrix: [
                [c, -s, 0.5 - 0.5 * c + 0.5 * s],
                [s, c, 0.5 - 0.5 * s - 0.5 * c],
                [0.0, 0.0, 1.0],
            ],
            filter: true,
        }
    }

    /// Create from a Mat3.
    pub fn from_mat3(m: Mat3) -> Self {
        Self {
            matrix: [
                m.x_axis.to_array(),
                m.y_axis.to_array(),
                m.z_axis.to_array(),
            ],
            filter: true,
        }
    }

    /// Convert to a Mat3.
    pub fn to_mat3(&self) -> Mat3 {
        Mat3::from_cols_array_2d(&self.matrix)
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        transform_image(image, self)
    }

    /// Returns the UV remapping function for this transformation.
    ///
    /// Uses the inverse matrix to map output UV → source UV.
    pub fn uv_fn(&self) -> impl Fn(f32, f32) -> (f32, f32) {
        let inv = self.to_mat3().inverse();

        move |u, v| {
            let src = inv * Vec3::new(u, v, 1.0);
            (src.x / src.z, src.y / src.z)
        }
    }
}

/// Apply a 2D affine transformation to image pixel positions.
///
/// This function is sugar over [`remap_uv_fn`] with the inverse transform matrix.
///
/// The transformation is applied to UV coordinates, effectively warping the image.
/// Uses inverse mapping: for each output pixel, find the corresponding input position.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, TransformConfig, transform_image};
///
/// let image = ImageField::from_raw(vec![[1.0, 0.0, 0.0, 1.0]; 64 * 64], 64, 64);
///
/// // Rotate 45 degrees around center
/// let config = TransformConfig::rotate(std::f32::consts::FRAC_PI_4);
/// let rotated = transform_image(&image, &config);
/// ```
pub fn transform_image(image: &ImageField, config: &TransformConfig) -> ImageField {
    // Prepare source image with desired filter mode
    let source = if config.filter && image.filter_mode != FilterMode::Bilinear {
        image.clone().with_filter_mode(FilterMode::Bilinear)
    } else {
        image.clone()
    };

    let mut result = remap_uv_fn(&source, config.uv_fn());

    // Restore original filter mode in result
    result.filter_mode = image.filter_mode;
    result
}

/// 1D lookup table for color grading.
///
/// Each channel (R, G, B) has its own curve mapping input [0, 1] to output [0, 1].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lut1D {
    /// Red channel LUT entries.
    pub red: Vec<f32>,
    /// Green channel LUT entries.
    pub green: Vec<f32>,
    /// Blue channel LUT entries.
    pub blue: Vec<f32>,
}

impl Lut1D {
    /// Create a linear (identity) LUT with the given size.
    pub fn linear(size: usize) -> Self {
        let entries: Vec<f32> = (0..size).map(|i| i as f32 / (size - 1) as f32).collect();
        Self {
            red: entries.clone(),
            green: entries.clone(),
            blue: entries,
        }
    }

    /// Create a contrast curve LUT.
    pub fn contrast(size: usize, amount: f32) -> Self {
        let entries: Vec<f32> = (0..size)
            .map(|i| {
                let t = i as f32 / (size - 1) as f32;
                // S-curve using smoothstep-like function
                let centered = t - 0.5;
                let curved = centered * amount;
                (curved + 0.5).clamp(0.0, 1.0)
            })
            .collect();
        Self {
            red: entries.clone(),
            green: entries.clone(),
            blue: entries,
        }
    }

    /// Create a gamma correction LUT.
    pub fn gamma(size: usize, gamma: f32) -> Self {
        let entries: Vec<f32> = (0..size)
            .map(|i| {
                let t = i as f32 / (size - 1) as f32;
                t.powf(1.0 / gamma)
            })
            .collect();
        Self {
            red: entries.clone(),
            green: entries.clone(),
            blue: entries,
        }
    }

    /// Sample the LUT for a given input value (with linear interpolation).
    fn sample(&self, lut: &[f32], value: f32) -> f32 {
        let clamped = value.clamp(0.0, 1.0);
        let scaled = clamped * (lut.len() - 1) as f32;
        let idx = scaled as usize;
        let frac = scaled - idx as f32;

        if idx + 1 >= lut.len() {
            lut[lut.len() - 1]
        } else {
            lut[idx] * (1.0 - frac) + lut[idx + 1] * frac
        }
    }

    /// Apply the 1D LUT to a color value.
    pub fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        [
            self.sample(&self.red, color[0]),
            self.sample(&self.green, color[1]),
            self.sample(&self.blue, color[2]),
            color[3],
        ]
    }
}

/// Apply a 1D LUT to an image.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Lut1D, apply_lut_1d};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
/// let lut = Lut1D::gamma(256, 2.2);
/// let corrected = apply_lut_1d(&image, &lut);
/// ```
pub fn apply_lut_1d(image: &ImageField, lut: &Lut1D) -> ImageField {
    let data: Vec<[f32; 4]> = image.data.iter().map(|pixel| lut.apply(*pixel)).collect();

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// 3D lookup table for color grading.
///
/// Maps RGB input to RGB output via a 3D grid with trilinear interpolation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lut3D {
    /// LUT data as [R][G][B] -> [r, g, b].
    pub data: Vec<[f32; 3]>,
    /// Size of each dimension.
    pub size: usize,
}

impl Lut3D {
    /// Create an identity 3D LUT with the given size.
    pub fn identity(size: usize) -> Self {
        let mut data = Vec::with_capacity(size * size * size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push([
                        r as f32 / (size - 1) as f32,
                        g as f32 / (size - 1) as f32,
                        b as f32 / (size - 1) as f32,
                    ]);
                }
            }
        }
        Self { data, size }
    }

    /// Sample the 3D LUT with trilinear interpolation.
    pub fn sample(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let size = self.size;
        let max_idx = (size - 1) as f32;

        // Scale and clamp input coordinates
        let r_scaled = (r.clamp(0.0, 1.0) * max_idx).min(max_idx - 0.001);
        let g_scaled = (g.clamp(0.0, 1.0) * max_idx).min(max_idx - 0.001);
        let b_scaled = (b.clamp(0.0, 1.0) * max_idx).min(max_idx - 0.001);

        // Integer and fractional parts
        let r0 = r_scaled as usize;
        let g0 = g_scaled as usize;
        let b0 = b_scaled as usize;
        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        let rf = r_scaled - r0 as f32;
        let gf = g_scaled - g0 as f32;
        let bf = b_scaled - b0 as f32;

        // Index helper
        let idx = |r: usize, g: usize, b: usize| b * size * size + g * size + r;

        // Sample 8 corners of the cube
        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        // Trilinear interpolation
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                lerp(a[0], b[0], t),
                lerp(a[1], b[1], t),
                lerp(a[2], b[2], t),
            ]
        };

        let c00 = lerp3(c000, c100, rf);
        let c10 = lerp3(c010, c110, rf);
        let c01 = lerp3(c001, c101, rf);
        let c11 = lerp3(c011, c111, rf);

        let c0 = lerp3(c00, c10, gf);
        let c1 = lerp3(c01, c11, gf);

        lerp3(c0, c1, bf)
    }

    /// Apply the 3D LUT to a color value.
    pub fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        let [r, g, b] = self.sample(color[0], color[1], color[2]);
        [r, g, b, color[3]]
    }
}

/// Apply a 3D LUT to an image.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Lut3D, apply_lut_3d};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
/// let lut = Lut3D::identity(17); // Standard .cube LUT size
/// let graded = apply_lut_3d(&image, &lut);
/// ```
pub fn apply_lut_3d(image: &ImageField, lut: &Lut3D) -> ImageField {
    let data: Vec<[f32; 4]> = image.data.iter().map(|pixel| lut.apply(*pixel)).collect();

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}
