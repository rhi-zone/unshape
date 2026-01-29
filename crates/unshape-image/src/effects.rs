#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::BlendMode;
use crate::ImageField;
use crate::channel::{Channel, extract_channel};
use crate::composite::composite;
use crate::kernel::blur;
use crate::pyramid::{downsample, resize};
use crate::transform::{TransformConfig, transform_image};

/// Configuration for drop shadow effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DropShadow {
    /// Horizontal offset in pixels (positive = right).
    pub offset_x: f32,
    /// Vertical offset in pixels (positive = down).
    pub offset_y: f32,
    /// Blur radius (number of blur passes).
    pub blur: u32,
    /// Shadow color (RGB, alpha controls shadow density).
    pub color: [f32; 4],
}

impl Default for DropShadow {
    fn default() -> Self {
        Self {
            offset_x: 4.0,
            offset_y: 4.0,
            blur: 3,
            color: [0.0, 0.0, 0.0, 0.5],
        }
    }
}

impl DropShadow {
    /// Creates a drop shadow with the given offset.
    pub fn new(offset_x: f32, offset_y: f32) -> Self {
        Self {
            offset_x,
            offset_y,
            ..Default::default()
        }
    }

    /// Sets the blur amount.
    pub fn with_blur(mut self, blur: u32) -> Self {
        self.blur = blur;
        self
    }

    /// Sets the shadow color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }
}

/// Applies a drop shadow effect to an image.
///
/// The shadow is created by:
/// 1. Extracting the alpha channel as a mask
/// 2. Offsetting (translating) the mask
/// 3. Blurring the mask
/// 4. Tinting the mask with the shadow color
/// 5. Compositing the shadow under the original image
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, DropShadow, drop_shadow};
///
/// let image = ImageField::solid_sized(100, 100, [1.0, 0.0, 0.0, 1.0]);
/// let config = DropShadow::new(5.0, 5.0).with_blur(4).with_color(0.0, 0.0, 0.0, 0.6);
/// let result = drop_shadow(&image, &config);
/// ```
pub fn drop_shadow(image: &ImageField, config: &DropShadow) -> ImageField {
    let (width, height) = image.dimensions();

    // 1. Extract alpha as shadow mask
    let alpha = extract_channel(image, Channel::Alpha);

    // 2. Offset the shadow
    let offset_config = TransformConfig::translate(
        config.offset_x / width as f32,
        config.offset_y / height as f32,
    );
    let offset_alpha = transform_image(&alpha, &offset_config);

    // 3. Blur the shadow
    let blurred = blur(&offset_alpha, config.blur);

    // 4. Tint with shadow color (multiply grayscale alpha by color)
    let mut shadow_data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let a = blurred.get_pixel(x, y)[0]; // grayscale value = alpha
            shadow_data.push([
                config.color[0],
                config.color[1],
                config.color[2],
                a * config.color[3],
            ]);
        }
    }
    let shadow = ImageField::from_raw(shadow_data, width, height);

    // 5. Composite: shadow under original
    let with_shadow = composite(&shadow, image, BlendMode::Normal, 1.0);

    with_shadow
}

/// Configuration for glow effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Glow {
    /// Blur radius (number of blur passes).
    pub blur: u32,
    /// Glow intensity (multiplier for the glow).
    pub intensity: f32,
    /// Optional glow color. If None, uses the image's own colors.
    pub color: Option<[f32; 3]>,
    /// Threshold for what counts as "bright" (0.0-1.0). Only pixels above this glow.
    pub threshold: f32,
}

impl Default for Glow {
    fn default() -> Self {
        Self {
            blur: 5,
            intensity: 1.0,
            color: None,
            threshold: 0.0,
        }
    }
}

impl Glow {
    /// Creates a glow effect with the given blur radius.
    pub fn new(blur: u32) -> Self {
        Self {
            blur,
            ..Default::default()
        }
    }

    /// Sets the glow intensity.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Sets a fixed glow color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Some([r, g, b]);
        self
    }

    /// Sets the brightness threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

/// Applies a glow effect to an image.
///
/// The glow is created by:
/// 1. Optionally thresholding to extract bright areas
/// 2. Blurring the image/threshold
/// 3. Optionally tinting with a glow color
/// 4. Additively compositing back onto the original
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Glow, glow};
///
/// let image = ImageField::solid_sized(100, 100, [1.0, 1.0, 1.0, 1.0]);
/// let config = Glow::new(6).with_intensity(1.5).with_color(1.0, 0.8, 0.2);
/// let result = glow(&image, &config);
/// ```
pub fn glow(image: &ImageField, config: &Glow) -> ImageField {
    let (width, height) = image.dimensions();

    // 1. Extract glow source (threshold if specified)
    let glow_source = if config.threshold > 0.0 {
        // Extract pixels above threshold
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
                if lum > config.threshold {
                    data.push(pixel);
                } else {
                    data.push([0.0, 0.0, 0.0, 0.0]);
                }
            }
        }
        ImageField::from_raw(data, width, height)
    } else {
        image.clone()
    };

    // 2. Blur the glow source
    let blurred = blur(&glow_source, config.blur);

    // 3. Tint if color specified, and apply intensity
    let tinted = if let Some(color) = config.color {
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = blurred.get_pixel(x, y);
                let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
                data.push([
                    color[0] * lum * config.intensity,
                    color[1] * lum * config.intensity,
                    color[2] * lum * config.intensity,
                    pixel[3],
                ]);
            }
        }
        ImageField::from_raw(data, width, height)
    } else {
        // Just apply intensity
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = blurred.get_pixel(x, y);
                data.push([
                    pixel[0] * config.intensity,
                    pixel[1] * config.intensity,
                    pixel[2] * config.intensity,
                    pixel[3],
                ]);
            }
        }
        ImageField::from_raw(data, width, height)
    };

    // 4. Additive composite
    composite(image, &tinted, BlendMode::Add, 1.0)
}

/// Configuration for bloom effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bloom {
    /// Brightness threshold (0.0-1.0). Only pixels above this bloom.
    pub threshold: f32,
    /// Number of blur passes at each scale.
    pub blur_passes: u32,
    /// Number of scales (pyramid levels) for the bloom.
    pub scales: u32,
    /// Overall bloom intensity.
    pub intensity: f32,
}

impl Default for Bloom {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            blur_passes: 3,
            scales: 4,
            intensity: 1.0,
        }
    }
}

impl Bloom {
    /// Creates a bloom effect with the given threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Sets the number of blur passes per scale.
    pub fn with_blur_passes(mut self, passes: u32) -> Self {
        self.blur_passes = passes;
        self
    }

    /// Sets the number of pyramid scales.
    pub fn with_scales(mut self, scales: u32) -> Self {
        self.scales = scales;
        self
    }

    /// Sets the bloom intensity.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }
}

/// Applies a bloom effect to an image.
///
/// Bloom creates a "glow" around bright areas using multi-scale blurring:
/// 1. Threshold to extract bright pixels
/// 2. Build a blur pyramid at multiple scales
/// 3. Combine all scales
/// 4. Additively composite back onto the original
///
/// This is more physically accurate than simple glow as it simulates
/// light scattering at multiple distances.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Bloom, bloom};
///
/// let image = ImageField::solid_sized(100, 100, [1.0, 1.0, 1.0, 1.0]);
/// let config = Bloom::new(0.7).with_scales(5).with_intensity(0.8);
/// let result = bloom(&image, &config);
/// ```
pub fn bloom(image: &ImageField, config: &Bloom) -> ImageField {
    let (width, height) = image.dimensions();

    // 1. Threshold to extract bright areas
    let mut bright_data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            if lum > config.threshold {
                // Soft threshold: keep amount above threshold
                let excess = lum - config.threshold;
                let scale = excess / (1.0 - config.threshold + 0.001);
                bright_data.push([
                    pixel[0] * scale,
                    pixel[1] * scale,
                    pixel[2] * scale,
                    pixel[3],
                ]);
            } else {
                bright_data.push([0.0, 0.0, 0.0, 0.0]);
            }
        }
    }
    let bright = ImageField::from_raw(bright_data, width, height);

    // 2. Build blur pyramid and accumulate
    let mut accumulated = ImageField::from_raw(
        vec![[0.0, 0.0, 0.0, 0.0]; (width * height) as usize],
        width,
        height,
    );
    let mut current = bright;

    for scale in 0..config.scales {
        // Blur at this scale
        let blurred = blur(&current, config.blur_passes);

        // Upsample back to original size if needed
        let to_add = if scale > 0 {
            resize(&blurred, width, height)
        } else {
            blurred
        };

        // Accumulate (weighted by scale - larger scales contribute less)
        let weight = 1.0 / (scale as f32 + 1.0);
        let mut acc_data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let acc_pixel = accumulated.get_pixel(x, y);
                let add_pixel = to_add.get_pixel(x, y);
                acc_data.push([
                    acc_pixel[0] + add_pixel[0] * weight,
                    acc_pixel[1] + add_pixel[1] * weight,
                    acc_pixel[2] + add_pixel[2] * weight,
                    acc_pixel[3].max(add_pixel[3] * weight),
                ]);
            }
        }
        accumulated = ImageField::from_raw(acc_data, width, height);

        // Downsample for next scale
        current = downsample(&current);
        if current.dimensions().0 < 4 || current.dimensions().1 < 4 {
            break;
        }
    }

    // 3. Apply intensity and composite
    let mut final_bloom_data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = accumulated.get_pixel(x, y);
            final_bloom_data.push([
                pixel[0] * config.intensity,
                pixel[1] * config.intensity,
                pixel[2] * config.intensity,
                pixel[3],
            ]);
        }
    }
    let final_bloom = ImageField::from_raw(final_bloom_data, width, height);

    composite(image, &final_bloom, BlendMode::Add, 1.0)
}
