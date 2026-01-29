#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ImageField;
use crate::expr::{UvExpr, remap_uv, remap_uv_fn};

/// Applies radial lens distortion (barrel or pincushion).
///
/// Barrel distortion (positive strength) makes the image bulge outward.
/// Pincushion distortion (negative strength) makes it pinch inward.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct LensDistortion {
    /// Distortion strength. Positive = barrel, negative = pincushion.
    pub strength: f32,
    /// Center point for distortion (normalized coordinates).
    pub center: (f32, f32),
}

impl Default for LensDistortion {
    fn default() -> Self {
        Self {
            strength: 0.0,
            center: (0.5, 0.5),
        }
    }
}

impl LensDistortion {
    /// Creates barrel distortion (bulging outward).
    pub fn barrel(strength: f32) -> Self {
        Self {
            strength: strength.abs(),
            center: (0.5, 0.5),
        }
    }

    /// Creates pincushion distortion (pinching inward).
    pub fn pincushion(strength: f32) -> Self {
        Self {
            strength: -strength.abs(),
            center: (0.5, 0.5),
        }
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        lens_distortion(image, self)
    }

    /// Converts this distortion to a `UvExpr` for use with `remap_uv`.
    ///
    /// This allows the distortion to be serialized, composed with other effects,
    /// and potentially compiled to GPU shaders.
    ///
    /// # Formula
    ///
    /// The radial distortion formula is:
    /// ```text
    /// delta = uv - center
    /// r = length(delta)
    /// distortion = 1 + strength * r²
    /// result = center + delta * distortion
    /// ```
    pub fn to_uv_expr(&self) -> UvExpr {
        let center = UvExpr::Constant2(self.center.0, self.center.1);

        // delta = uv - center
        let delta = UvExpr::Sub(Box::new(UvExpr::Uv), Box::new(center.clone()));

        // r² = delta.x² + delta.y² = dot(delta, delta)
        // Since Length returns (len, len) and we need r², we use Dot
        let r_squared = UvExpr::Dot(Box::new(delta.clone()), Box::new(delta.clone()));

        // distortion = 1 + strength * r²
        let distortion = UvExpr::Add(
            Box::new(UvExpr::Constant(1.0)),
            Box::new(UvExpr::Mul(
                Box::new(UvExpr::Constant(self.strength)),
                Box::new(r_squared),
            )),
        );

        // result = center + delta * distortion
        UvExpr::Add(
            Box::new(center),
            Box::new(UvExpr::Mul(Box::new(delta), Box::new(distortion))),
        )
    }
}

/// Backwards-compatible type alias.
pub type LensDistortionConfig = LensDistortion;

/// Applies radial lens distortion (barrel or pincushion).
///
/// Barrel distortion (positive strength) makes the image bulge outward.
/// Pincushion distortion (negative strength) makes it pinch inward.
///
/// This function is sugar over [`remap_uv`] with the expression from
/// [`LensDistortion::to_uv_expr`].
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, lens_distortion, LensDistortionConfig};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let barrel = lens_distortion(&img, &LensDistortionConfig::barrel(0.3));
/// let pincushion = lens_distortion(&img, &LensDistortionConfig::pincushion(0.3));
/// ```
pub fn lens_distortion(image: &ImageField, config: &LensDistortion) -> ImageField {
    remap_uv(image, &config.to_uv_expr())
}

/// Applies wave distortion to an image.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct WaveDistortion {
    /// Amplitude in X direction (as fraction of image size).
    pub amplitude_x: f32,
    /// Amplitude in Y direction.
    pub amplitude_y: f32,
    /// Frequency of waves in X direction.
    pub frequency_x: f32,
    /// Frequency of waves in Y direction.
    pub frequency_y: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl Default for WaveDistortion {
    fn default() -> Self {
        Self {
            amplitude_x: 0.02,
            amplitude_y: 0.02,
            frequency_x: 4.0,
            frequency_y: 4.0,
            phase: 0.0,
        }
    }
}

impl WaveDistortion {
    /// Creates a horizontal wave distortion.
    pub fn horizontal(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude_x: amplitude,
            amplitude_y: 0.0,
            frequency_x: frequency,
            frequency_y: 0.0,
            phase: 0.0,
        }
    }

    /// Creates a vertical wave distortion.
    pub fn vertical(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude_x: 0.0,
            amplitude_y: amplitude,
            frequency_x: 0.0,
            frequency_y: frequency,
            phase: 0.0,
        }
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        wave_distortion(image, self)
    }

    /// Converts this distortion to a `UvExpr` for use with `remap_uv`.
    ///
    /// This allows the distortion to be serialized, composed with other effects,
    /// and potentially compiled to GPU shaders.
    ///
    /// # Formula
    ///
    /// The wave distortion formula is:
    /// ```text
    /// offset_x = amplitude_x * sin(v * frequency_y * 2π + phase)
    /// offset_y = amplitude_y * sin(u * frequency_x * 2π + phase)
    /// result = uv + offset
    /// ```
    pub fn to_uv_expr(&self) -> UvExpr {
        let two_pi = std::f32::consts::PI * 2.0;

        // offset_x = amplitude_x * sin(v * frequency_y * 2π + phase)
        let offset_x = UvExpr::Mul(
            Box::new(UvExpr::Constant(self.amplitude_x)),
            Box::new(UvExpr::Sin(Box::new(UvExpr::Add(
                Box::new(UvExpr::Mul(
                    Box::new(UvExpr::V),
                    Box::new(UvExpr::Constant(self.frequency_y * two_pi)),
                )),
                Box::new(UvExpr::Constant(self.phase)),
            )))),
        );

        // offset_y = amplitude_y * sin(u * frequency_x * 2π + phase)
        let offset_y = UvExpr::Mul(
            Box::new(UvExpr::Constant(self.amplitude_y)),
            Box::new(UvExpr::Sin(Box::new(UvExpr::Add(
                Box::new(UvExpr::Mul(
                    Box::new(UvExpr::U),
                    Box::new(UvExpr::Constant(self.frequency_x * two_pi)),
                )),
                Box::new(UvExpr::Constant(self.phase)),
            )))),
        );

        // result = uv + vec2(offset_x, offset_y)
        UvExpr::Add(
            Box::new(UvExpr::Uv),
            Box::new(UvExpr::Vec2 {
                x: Box::new(offset_x),
                y: Box::new(offset_y),
            }),
        )
    }
}

/// Backwards-compatible type alias.
pub type WaveDistortionConfig = WaveDistortion;

/// Applies wave distortion to an image.
///
/// This function is sugar over [`remap_uv`] with the expression from
/// [`WaveDistortion::to_uv_expr`].
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, wave_distortion, WaveDistortionConfig};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let wavy = wave_distortion(&img, &WaveDistortionConfig::horizontal(0.05, 3.0));
/// ```
pub fn wave_distortion(image: &ImageField, config: &WaveDistortion) -> ImageField {
    remap_uv(image, &config.to_uv_expr())
}

/// Applies displacement using another image as a map.
///
/// The displacement map's red channel controls X offset, green controls Y offset.
/// Values are mapped from [0, 1] to [-strength, +strength].
///
/// # Arguments
/// * `image` - Source image to distort
/// * `displacement_map` - Image controlling displacement (R=X, G=Y)
/// * `strength` - Maximum displacement as fraction of image size
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, displace};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
/// let map = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// let displaced = displace(&img, &map, 0.1);
/// ```
pub fn displace(image: &ImageField, displacement_map: &ImageField, strength: f32) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            // Sample displacement map
            let disp = displacement_map.sample_uv(u, v);

            // Map [0, 1] to [-strength, +strength]
            let offset_x = (disp.r - 0.5) * 2.0 * strength;
            let offset_y = (disp.g - 0.5) * 2.0 * strength;

            let src_u = u + offset_x;
            let src_v = v + offset_y;

            let color = image.sample_uv(src_u, src_v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Configuration for swirl/twist distortion.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct Swirl {
    /// Maximum rotation in radians at center.
    pub angle: f32,
    /// Radius of effect (normalized, 1.0 = half image size).
    pub radius: f32,
    /// Center point (normalized coordinates).
    pub center: (f32, f32),
}

impl Default for Swirl {
    fn default() -> Self {
        Self {
            angle: 1.0,
            radius: 0.5,
            center: (0.5, 0.5),
        }
    }
}

impl Swirl {
    /// Creates a new swirl distortion centered on the image.
    pub fn new(angle: f32, radius: f32) -> Self {
        Self {
            angle,
            radius,
            center: (0.5, 0.5),
        }
    }

    /// Creates a swirl with custom center.
    pub fn with_center(mut self, center: (f32, f32)) -> Self {
        self.center = center;
        self
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        swirl(image, self.angle, self.radius, self.center)
    }

    /// Returns the UV remapping function for this distortion.
    pub fn uv_fn(&self) -> impl Fn(f32, f32) -> (f32, f32) {
        let angle = self.angle;
        let radius = self.radius;
        let radius_sq = radius * radius;
        let center = self.center;

        move |u, v| {
            let dx = u - center.0;
            let dy = v - center.1;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < radius_sq {
                let dist = dist_sq.sqrt();
                let factor = 1.0 - dist / radius;
                let rotation = angle * factor * factor;

                let cos_r = rotation.cos();
                let sin_r = rotation.sin();

                let new_dx = dx * cos_r - dy * sin_r;
                let new_dy = dx * sin_r + dy * cos_r;

                (center.0 + new_dx, center.1 + new_dy)
            } else {
                (u, v)
            }
        }
    }
}

/// Applies a swirl/twist distortion around a center point.
///
/// This function is sugar over [`remap_uv_fn`] with the swirl transformation.
///
/// # Arguments
/// * `angle` - Maximum rotation in radians at center
/// * `radius` - Radius of effect (normalized, 1.0 = half image size)
/// * `center` - Center point (normalized coordinates)
pub fn swirl(image: &ImageField, angle: f32, radius: f32, center: (f32, f32)) -> ImageField {
    let config = Swirl {
        angle,
        radius,
        center,
    };
    remap_uv_fn(image, config.uv_fn())
}

/// Configuration for spherize/bulge effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct Spherize {
    /// Bulge strength (positive = bulge out, negative = pinch in).
    pub strength: f32,
    /// Center point (normalized coordinates).
    pub center: (f32, f32),
}

impl Default for Spherize {
    fn default() -> Self {
        Self {
            strength: 0.5,
            center: (0.5, 0.5),
        }
    }
}

impl Spherize {
    /// Creates a new spherize effect centered on the image.
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            center: (0.5, 0.5),
        }
    }

    /// Creates a bulge effect (positive strength).
    pub fn bulge(strength: f32) -> Self {
        Self::new(strength.abs())
    }

    /// Creates a pinch effect (negative strength).
    pub fn pinch(strength: f32) -> Self {
        Self::new(-strength.abs())
    }

    /// Sets the center point.
    pub fn with_center(mut self, center: (f32, f32)) -> Self {
        self.center = center;
        self
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        spherize(image, self.strength, self.center)
    }

    /// Returns the UV remapping function for this distortion.
    pub fn uv_fn(&self) -> impl Fn(f32, f32) -> (f32, f32) {
        let strength = self.strength;
        let center = self.center;

        move |u, v| {
            let dx = u - center.0;
            let dy = v - center.1;
            let dist = (dx * dx + dy * dy).sqrt();

            let factor = if dist > 0.0001 {
                let t = dist.min(0.5) / 0.5;
                let spherize_factor = (1.0 - t * t).sqrt();
                1.0 + (spherize_factor - 1.0) * strength
            } else {
                1.0
            };

            (center.0 + dx * factor, center.1 + dy * factor)
        }
    }
}

/// Applies a spherize/bulge effect.
///
/// This function is sugar over [`remap_uv_fn`] with the spherize transformation.
///
/// # Arguments
/// * `strength` - Bulge strength (positive = bulge out, negative = pinch in)
/// * `center` - Center point (normalized coordinates)
pub fn spherize(image: &ImageField, strength: f32, center: (f32, f32)) -> ImageField {
    let config = Spherize { strength, center };
    remap_uv_fn(image, config.uv_fn())
}
