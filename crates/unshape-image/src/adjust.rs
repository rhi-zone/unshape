#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ImageField;
use crate::expr::{ColorExpr, map_pixels};

/// Applies chromatic aberration effect to an image.
///
/// This simulates lens chromatic aberration by offsetting each color channel
/// radially from the center point. Positive offsets push the channel outward,
/// negative offsets push inward.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct ChromaticAberration {
    /// Offset amount for red channel (negative = inward, positive = outward).
    pub red_offset: f32,
    /// Offset amount for green channel.
    pub green_offset: f32,
    /// Offset amount for blue channel.
    pub blue_offset: f32,
    /// Center point for radial offset (normalized coordinates, default: (0.5, 0.5)).
    pub center: (f32, f32),
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        Self {
            red_offset: 0.005,
            green_offset: 0.0,
            blue_offset: -0.005,
            center: (0.5, 0.5),
        }
    }
}

impl ChromaticAberration {
    /// Creates a new config with symmetric red/blue offset.
    ///
    /// Red is pushed outward, blue inward (typical lens aberration).
    pub fn new(strength: f32) -> Self {
        Self {
            red_offset: strength,
            green_offset: 0.0,
            blue_offset: -strength,
            center: (0.5, 0.5),
        }
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        chromatic_aberration(image, self)
    }
}

/// Backwards-compatible type alias.
pub type ChromaticAberrationConfig = ChromaticAberration;

/// Applies chromatic aberration effect to an image.
///
/// This simulates lens chromatic aberration by offsetting each color channel
/// radially from the center point. Positive offsets push the channel outward,
/// negative offsets push inward.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, chromatic_aberration, ChromaticAberrationConfig};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// // Subtle chromatic aberration
/// let config = ChromaticAberrationConfig::new(0.01);
/// let result = chromatic_aberration(&img, &config);
/// ```
pub fn chromatic_aberration(image: &ImageField, config: &ChromaticAberration) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            // Normalize coordinates to [0, 1]
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            // Vector from center to current pixel
            let dx = u - config.center.0;
            let dy = v - config.center.1;

            // Sample each channel at its offset position
            let r_u = u + dx * config.red_offset;
            let r_v = v + dy * config.red_offset;
            let r = image.sample_uv(r_u, r_v).r;

            let g_u = u + dx * config.green_offset;
            let g_v = v + dy * config.green_offset;
            let g = image.sample_uv(g_u, g_v).g;

            let b_u = u + dx * config.blue_offset;
            let b_v = v + dy * config.blue_offset;
            let b = image.sample_uv(b_u, b_v).b;

            // Alpha from original position
            let a = image.sample_uv(u, v).a;

            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a quick chromatic aberration with default red/blue fringing.
///
/// # Arguments
/// * `strength` - Amount of aberration (0.01-0.05 for subtle, higher for dramatic)
pub fn chromatic_aberration_simple(image: &ImageField, strength: f32) -> ImageField {
    chromatic_aberration(image, &ChromaticAberrationConfig::new(strength))
}

/// Applies levels adjustment to an image.
///
/// This is similar to Photoshop's Levels adjustment:
/// 1. Remap input range [input_black, input_white] to [0, 1]
/// 2. Apply gamma correction
/// 3. Remap [0, 1] to [output_black, output_white]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ImageField, output = ImageField))]
pub struct Levels {
    /// Input black point (values below this become 0). Range: 0-1.
    pub input_black: f32,
    /// Input white point (values above this become 1). Range: 0-1.
    pub input_white: f32,
    /// Gamma correction (1.0 = linear, <1 = brighten, >1 = darken).
    pub gamma: f32,
    /// Output black point. Range: 0-1.
    pub output_black: f32,
    /// Output white point. Range: 0-1.
    pub output_white: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Self {
            input_black: 0.0,
            input_white: 1.0,
            gamma: 1.0,
            output_black: 0.0,
            output_white: 1.0,
        }
    }
}

impl Levels {
    /// Creates a new levels config with only gamma adjustment.
    pub fn gamma(gamma: f32) -> Self {
        Self {
            gamma,
            ..Default::default()
        }
    }

    /// Creates a levels config that remaps black/white points.
    pub fn remap(input_black: f32, input_white: f32) -> Self {
        Self {
            input_black,
            input_white,
            ..Default::default()
        }
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        adjust_levels(image, self)
    }
}

/// Backwards-compatible type alias.
pub type LevelsConfig = Levels;

/// Applies levels adjustment to an image.
///
/// This is similar to Photoshop's Levels adjustment:
/// 1. Remap input range [input_black, input_white] to [0, 1]
/// 2. Apply gamma correction
/// 3. Remap [0, 1] to [output_black, output_white]
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, adjust_levels, LevelsConfig};
///
/// let data = vec![[0.3, 0.5, 0.7, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// // Increase contrast by pulling in black/white points
/// let config = LevelsConfig::remap(0.2, 0.8);
/// let result = adjust_levels(&img, &config);
/// ```
pub fn adjust_levels(image: &ImageField, config: &Levels) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    let input_range = (config.input_white - config.input_black).max(0.001);
    let output_range = config.output_white - config.output_black;
    // Gamma < 1 brightens (raises values), gamma > 1 darkens (lowers values)
    let gamma = config.gamma.max(0.001);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);

            let adjust = |v: f32| -> f32 {
                // Remap input
                let normalized = ((v - config.input_black) / input_range).clamp(0.0, 1.0);
                // Apply gamma (gamma < 1 brightens, gamma > 1 darkens)
                let gamma_corrected = normalized.powf(gamma);
                // Remap output
                (gamma_corrected * output_range + config.output_black).clamp(0.0, 1.0)
            };

            data.push([
                adjust(pixel[0]),
                adjust(pixel[1]),
                adjust(pixel[2]),
                pixel[3],
            ]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Adjusts brightness and contrast of an image.
///
/// # Arguments
/// * `brightness` - Brightness adjustment (-1 to 1, 0 = no change)
/// * `contrast` - Contrast adjustment (-1 to 1, 0 = no change)
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, adjust_brightness_contrast};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let result = adjust_brightness_contrast(&img, 0.1, 0.2);
/// ```
pub fn adjust_brightness_contrast(
    image: &ImageField,
    brightness: f32,
    contrast: f32,
) -> ImageField {
    map_pixels(image, &ColorExpr::brightness_contrast(brightness, contrast))
}

/// Configuration for HSL adjustments.
#[derive(Debug, Clone, Copy, Default)]
pub struct HslAdjustment {
    /// Hue shift (-0.5 to 0.5, wraps around the color wheel).
    pub hue_shift: f32,
    /// Saturation adjustment (-1 = grayscale, 0 = no change, 1 = double saturation).
    pub saturation: f32,
    /// Lightness adjustment (-1 = black, 0 = no change, 1 = white).
    pub lightness: f32,
}

impl HslAdjustment {
    /// Creates a hue shift adjustment.
    pub fn hue(shift: f32) -> Self {
        Self {
            hue_shift: shift,
            saturation: 0.0,
            lightness: 0.0,
        }
    }

    /// Creates a saturation adjustment.
    pub fn saturation(amount: f32) -> Self {
        Self {
            hue_shift: 0.0,
            saturation: amount,
            lightness: 0.0,
        }
    }

    /// Creates a lightness adjustment.
    pub fn lightness(amount: f32) -> Self {
        Self {
            hue_shift: 0.0,
            saturation: 0.0,
            lightness: amount,
        }
    }
}

/// Adjusts hue, saturation, and lightness of an image.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, adjust_hsl, HslAdjustment};
///
/// let data = vec![[1.0, 0.5, 0.0, 1.0]; 4]; // Orange
/// let img = ImageField::from_raw(data, 2, 2);
///
/// // Shift hue by 180 degrees (complement)
/// let result = adjust_hsl(&img, &HslAdjustment::hue(0.5));
/// ```
pub fn adjust_hsl(image: &ImageField, adjustment: &HslAdjustment) -> ImageField {
    // Convert HslAdjustment to ColorExpr parameters:
    // - saturation in HslAdjustment is additive (-1 to +1) but ColorExpr uses multiplicative
    // - So we convert: saturation_mult = 1.0 + adjustment.saturation
    let saturation_mult = 1.0 + adjustment.saturation;
    map_pixels(
        image,
        &ColorExpr::hsl_adjust(adjustment.hue_shift, saturation_mult, adjustment.lightness),
    )
}

/// Converts an image to grayscale using luminance.
///
/// Uses ITU-R BT.709 coefficients: 0.2126 R + 0.7152 G + 0.0722 B
pub fn grayscale(image: &ImageField) -> ImageField {
    map_pixels(image, &ColorExpr::grayscale())
}

/// Inverts the colors of an image.
///
/// Each RGB channel is inverted (1 - value). Alpha is preserved.
pub fn invert(image: &ImageField) -> ImageField {
    map_pixels(image, &ColorExpr::invert())
}

/// Applies a posterization effect, reducing the number of color levels.
///
/// # Arguments
/// * `levels` - Number of levels per channel (2-256, typically 2-8 for visible effect)
pub fn posterize(image: &ImageField, levels: u32) -> ImageField {
    map_pixels(image, &ColorExpr::posterize(levels))
}

/// Applies a threshold effect, converting to black and white.
///
/// Pixels with luminance above the threshold become white, below become black.
pub fn threshold(image: &ImageField, thresh: f32) -> ImageField {
    map_pixels(image, &ColorExpr::threshold(thresh))
}
