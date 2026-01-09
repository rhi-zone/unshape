//! Image-based fields for texture sampling.
//!
//! Provides `ImageField` which loads an image and exposes it as a `Field<Vec2, Color>`.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_image::{ImageField, WrapMode, FilterMode};
//! use rhizome_resin_field::{Field, EvalContext};
//! use glam::Vec2;
//!
//! let field = ImageField::from_file("texture.png")?;
//! let ctx = EvalContext::new();
//!
//! // Sample at UV coordinates (0.5, 0.5)
//! let color = field.sample(Vec2::new(0.5, 0.5), &ctx);
//! ```

use std::io::Read;
use std::path::Path;

use glam::{Vec2, Vec4};
use image::{DynamicImage, GenericImageView, ImageError};

use rhizome_resin_color::Rgba;
use rhizome_resin_field::{EvalContext, Field};

/// How to handle UV coordinates outside [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WrapMode {
    /// Repeat the texture (fract of coordinate).
    #[default]
    Repeat,
    /// Clamp coordinates to [0, 1].
    Clamp,
    /// Mirror the texture at boundaries.
    Mirror,
}

/// How to sample between pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Use the nearest pixel (blocky).
    Nearest,
    /// Bilinear interpolation (smooth).
    #[default]
    Bilinear,
}

/// An image that can be sampled as a field.
///
/// UV coordinates go from (0, 0) at the top-left to (1, 1) at the bottom-right.
#[derive(Clone)]
pub struct ImageField {
    /// Image pixel data as RGBA.
    data: Vec<[f32; 4]>,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// How to handle coordinates outside [0, 1].
    pub wrap_mode: WrapMode,
    /// How to interpolate between pixels.
    pub filter_mode: FilterMode,
}

impl std::fmt::Debug for ImageField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageField")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("wrap_mode", &self.wrap_mode)
            .field("filter_mode", &self.filter_mode)
            .finish()
    }
}

/// Errors that can occur when loading images.
#[derive(Debug)]
pub enum ImageFieldError {
    /// Failed to load the image file.
    ImageError(ImageError),
    /// I/O error reading the file.
    IoError(std::io::Error),
}

impl std::fmt::Display for ImageFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageFieldError::ImageError(e) => write!(f, "Image error: {}", e),
            ImageFieldError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for ImageFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ImageFieldError::ImageError(e) => Some(e),
            ImageFieldError::IoError(e) => Some(e),
        }
    }
}

impl From<ImageError> for ImageFieldError {
    fn from(e: ImageError) -> Self {
        ImageFieldError::ImageError(e)
    }
}

impl From<std::io::Error> for ImageFieldError {
    fn from(e: std::io::Error) -> Self {
        ImageFieldError::IoError(e)
    }
}

impl ImageField {
    /// Creates an image field from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ImageFieldError> {
        let img = image::open(path)?;
        Ok(Self::from_image(img))
    }

    /// Creates an image field from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ImageFieldError> {
        let img = image::load_from_memory(bytes)?;
        Ok(Self::from_image(img))
    }

    /// Creates an image field from a reader.
    pub fn from_reader<R: Read + std::io::BufRead + std::io::Seek>(
        reader: R,
    ) -> Result<Self, ImageFieldError> {
        let img = image::ImageReader::new(reader)
            .with_guessed_format()?
            .decode()?;
        Ok(Self::from_image(img))
    }

    /// Creates an image field from a DynamicImage.
    pub fn from_image(img: DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        // Convert to f32 RGBA
        let data: Vec<[f32; 4]> = rgba
            .pixels()
            .map(|p| {
                [
                    p.0[0] as f32 / 255.0,
                    p.0[1] as f32 / 255.0,
                    p.0[2] as f32 / 255.0,
                    p.0[3] as f32 / 255.0,
                ]
            })
            .collect();

        Self {
            data,
            width,
            height,
            wrap_mode: WrapMode::default(),
            filter_mode: FilterMode::default(),
        }
    }

    /// Creates a solid color image field.
    pub fn solid(color: Rgba) -> Self {
        Self {
            data: vec![[color.r, color.g, color.b, color.a]],
            width: 1,
            height: 1,
            wrap_mode: WrapMode::Repeat,
            filter_mode: FilterMode::Nearest,
        }
    }

    /// Creates an image field from raw pixel data.
    ///
    /// `data` should be in row-major order, RGBA format, with values in [0, 1].
    pub fn from_raw(data: Vec<[f32; 4]>, width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            (width * height) as usize,
            "Data length must match width * height"
        );
        Self {
            data,
            width,
            height,
            wrap_mode: WrapMode::default(),
            filter_mode: FilterMode::default(),
        }
    }

    /// Sets the wrap mode for this image field.
    pub fn with_wrap_mode(mut self, mode: WrapMode) -> Self {
        self.wrap_mode = mode;
        self
    }

    /// Sets the filter mode for this image field.
    pub fn with_filter_mode(mut self, mode: FilterMode) -> Self {
        self.filter_mode = mode;
        self
    }

    /// Returns the image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Wraps a coordinate according to the wrap mode.
    fn wrap(&self, mut t: f32) -> f32 {
        match self.wrap_mode {
            WrapMode::Repeat => {
                t = t.rem_euclid(1.0);
                if t < 0.0 { t + 1.0 } else { t }
            }
            WrapMode::Clamp => t.clamp(0.0, 1.0),
            WrapMode::Mirror => {
                t = t.rem_euclid(2.0);
                if t > 1.0 { 2.0 - t } else { t }
            }
        }
    }

    /// Gets a pixel at integer coordinates (clamped to bounds).
    fn get_pixel(&self, x: u32, y: u32) -> [f32; 4] {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let idx = (y * self.width + x) as usize;
        self.data[idx]
    }

    /// Samples the image at normalized UV coordinates.
    pub fn sample_uv(&self, u: f32, v: f32) -> Rgba {
        let u = self.wrap(u);
        let v = self.wrap(v);

        match self.filter_mode {
            FilterMode::Nearest => {
                let x = (u * self.width as f32).floor() as u32;
                let y = (v * self.height as f32).floor() as u32;
                let pixel = self.get_pixel(x, y);
                Rgba::new(pixel[0], pixel[1], pixel[2], pixel[3])
            }
            FilterMode::Bilinear => {
                // Map to pixel coordinates
                let px = u * self.width as f32 - 0.5;
                let py = v * self.height as f32 - 0.5;

                let x0 = px.floor() as i32;
                let y0 = py.floor() as i32;
                let fx = px - px.floor();
                let fy = py - py.floor();

                // Get four surrounding pixels (with wrapping)
                let x0u = x0.rem_euclid(self.width as i32) as u32;
                let x1u = (x0 + 1).rem_euclid(self.width as i32) as u32;
                let y0u = y0.rem_euclid(self.height as i32) as u32;
                let y1u = (y0 + 1).rem_euclid(self.height as i32) as u32;

                let p00 = self.get_pixel(x0u, y0u);
                let p10 = self.get_pixel(x1u, y0u);
                let p01 = self.get_pixel(x0u, y1u);
                let p11 = self.get_pixel(x1u, y1u);

                // Bilinear interpolation
                let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                let lerp_pixel = |a: [f32; 4], b: [f32; 4], t: f32| {
                    [
                        lerp(a[0], b[0], t),
                        lerp(a[1], b[1], t),
                        lerp(a[2], b[2], t),
                        lerp(a[3], b[3], t),
                    ]
                };

                let top = lerp_pixel(p00, p10, fx);
                let bottom = lerp_pixel(p01, p11, fx);
                let result = lerp_pixel(top, bottom, fy);

                Rgba::new(result[0], result[1], result[2], result[3])
            }
        }
    }

    /// Samples the image and returns a Vec4 (useful for RGBA operations).
    pub fn sample_vec4(&self, u: f32, v: f32) -> Vec4 {
        let color = self.sample_uv(u, v);
        Vec4::new(color.r, color.g, color.b, color.a)
    }
}

impl Field<Vec2, Rgba> for ImageField {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Rgba {
        self.sample_uv(input.x, input.y)
    }
}

impl Field<Vec2, Vec4> for ImageField {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec4 {
        self.sample_vec4(input.x, input.y)
    }
}

impl Field<Vec2, f32> for ImageField {
    /// Samples the image as grayscale (luminance).
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let color = self.sample_uv(input.x, input.y);
        // Standard luminance coefficients (ITU-R BT.709)
        0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
    }
}

// ============================================================================
// Texture baking - render Field to Image
// ============================================================================

/// Configuration for texture baking.
#[derive(Debug, Clone)]
pub struct BakeConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of samples per pixel for anti-aliasing (1 = no AA).
    pub samples: u32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            samples: 1,
        }
    }
}

impl BakeConfig {
    /// Creates a new bake config with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            samples: 1,
        }
    }

    /// Sets the number of anti-aliasing samples per pixel.
    pub fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples.max(1);
        self
    }
}

/// Bakes a scalar field (Field<Vec2, f32>) to a grayscale image.
///
/// UV coordinates go from (0, 0) at top-left to (1, 1) at bottom-right.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{bake_scalar, BakeConfig};
/// use rhizome_resin_field::{Perlin2D, Field, EvalContext};
///
/// let noise = Perlin2D::new().scale(4.0);
/// let config = BakeConfig::new(256, 256);
/// let ctx = EvalContext::new();
///
/// let image = bake_scalar(&noise, &config, &ctx);
/// assert_eq!(image.dimensions(), (256, 256));
/// ```
pub fn bake_scalar<F: Field<Vec2, f32>>(
    field: &F,
    config: &BakeConfig,
    ctx: &EvalContext,
) -> ImageField {
    let mut data = Vec::with_capacity((config.width * config.height) as usize);

    for y in 0..config.height {
        for x in 0..config.width {
            let value = if config.samples == 1 {
                // Single sample at pixel center
                let u = (x as f32 + 0.5) / config.width as f32;
                let v = (y as f32 + 0.5) / config.height as f32;
                field.sample(Vec2::new(u, v), ctx)
            } else {
                // Multi-sample anti-aliasing
                let mut sum = 0.0;
                let samples_sqrt = (config.samples as f32).sqrt().ceil() as u32;
                let actual_samples = samples_sqrt * samples_sqrt;

                for sy in 0..samples_sqrt {
                    for sx in 0..samples_sqrt {
                        let u = (x as f32 + (sx as f32 + 0.5) / samples_sqrt as f32)
                            / config.width as f32;
                        let v = (y as f32 + (sy as f32 + 0.5) / samples_sqrt as f32)
                            / config.height as f32;
                        sum += field.sample(Vec2::new(u, v), ctx);
                    }
                }
                sum / actual_samples as f32
            };

            let clamped = value.clamp(0.0, 1.0);
            data.push([clamped, clamped, clamped, 1.0]);
        }
    }

    ImageField::from_raw(data, config.width, config.height)
}

/// Bakes an RGBA field (Field<Vec2, Rgba>) to an image.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_image::{bake_rgba, BakeConfig};
/// use rhizome_resin_field::{Field, EvalContext};
///
/// let field = MyColorField::new();
/// let config = BakeConfig::new(512, 512);
/// let ctx = EvalContext::new();
///
/// let image = bake_rgba(&field, &config, &ctx);
/// ```
pub fn bake_rgba<F: Field<Vec2, Rgba>>(
    field: &F,
    config: &BakeConfig,
    ctx: &EvalContext,
) -> ImageField {
    let mut data = Vec::with_capacity((config.width * config.height) as usize);

    for y in 0..config.height {
        for x in 0..config.width {
            let color = if config.samples == 1 {
                let u = (x as f32 + 0.5) / config.width as f32;
                let v = (y as f32 + 0.5) / config.height as f32;
                field.sample(Vec2::new(u, v), ctx)
            } else {
                let mut sum = Rgba::new(0.0, 0.0, 0.0, 0.0);
                let samples_sqrt = (config.samples as f32).sqrt().ceil() as u32;
                let actual_samples = samples_sqrt * samples_sqrt;

                for sy in 0..samples_sqrt {
                    for sx in 0..samples_sqrt {
                        let u = (x as f32 + (sx as f32 + 0.5) / samples_sqrt as f32)
                            / config.width as f32;
                        let v = (y as f32 + (sy as f32 + 0.5) / samples_sqrt as f32)
                            / config.height as f32;
                        let c = field.sample(Vec2::new(u, v), ctx);
                        sum.r += c.r;
                        sum.g += c.g;
                        sum.b += c.b;
                        sum.a += c.a;
                    }
                }
                let n = actual_samples as f32;
                Rgba::new(sum.r / n, sum.g / n, sum.b / n, sum.a / n)
            };

            data.push([
                color.r.clamp(0.0, 1.0),
                color.g.clamp(0.0, 1.0),
                color.b.clamp(0.0, 1.0),
                color.a.clamp(0.0, 1.0),
            ]);
        }
    }

    ImageField::from_raw(data, config.width, config.height)
}

/// Bakes a Vec4 field (Field<Vec2, Vec4>) to an image.
pub fn bake_vec4<F: Field<Vec2, Vec4>>(
    field: &F,
    config: &BakeConfig,
    ctx: &EvalContext,
) -> ImageField {
    let mut data = Vec::with_capacity((config.width * config.height) as usize);

    for y in 0..config.height {
        for x in 0..config.width {
            let color = if config.samples == 1 {
                let u = (x as f32 + 0.5) / config.width as f32;
                let v = (y as f32 + 0.5) / config.height as f32;
                field.sample(Vec2::new(u, v), ctx)
            } else {
                let mut sum = Vec4::ZERO;
                let samples_sqrt = (config.samples as f32).sqrt().ceil() as u32;
                let actual_samples = samples_sqrt * samples_sqrt;

                for sy in 0..samples_sqrt {
                    for sx in 0..samples_sqrt {
                        let u = (x as f32 + (sx as f32 + 0.5) / samples_sqrt as f32)
                            / config.width as f32;
                        let v = (y as f32 + (sy as f32 + 0.5) / samples_sqrt as f32)
                            / config.height as f32;
                        sum += field.sample(Vec2::new(u, v), ctx);
                    }
                }
                sum / actual_samples as f32
            };

            data.push([
                color.x.clamp(0.0, 1.0),
                color.y.clamp(0.0, 1.0),
                color.z.clamp(0.0, 1.0),
                color.w.clamp(0.0, 1.0),
            ]);
        }
    }

    ImageField::from_raw(data, config.width, config.height)
}

/// Exports an ImageField to a PNG file.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_image::{bake_scalar, BakeConfig, export_png};
/// use rhizome_resin_field::{Perlin2D, EvalContext};
///
/// let noise = Perlin2D::new().scale(4.0);
/// let config = BakeConfig::new(256, 256);
/// let ctx = EvalContext::new();
///
/// let image = bake_scalar(&noise, &config, &ctx);
/// export_png(&image, "noise.png").unwrap();
/// ```
pub fn export_png<P: AsRef<Path>>(image: &ImageField, path: P) -> Result<(), ImageFieldError> {
    let (width, height) = image.dimensions();
    let mut img_buf = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let color = image.sample_uv(u, v);

            img_buf.put_pixel(
                x,
                y,
                image::Rgba([
                    (color.r * 255.0) as u8,
                    (color.g * 255.0) as u8,
                    (color.b * 255.0) as u8,
                    (color.a * 255.0) as u8,
                ]),
            );
        }
    }

    img_buf.save(path)?;
    Ok(())
}

// ============================================================================
// Animation export - image sequences and GIF
// ============================================================================

/// Configuration for animation rendering.
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of frames.
    pub num_frames: usize,
    /// Frame duration in seconds.
    pub frame_duration: f32,
    /// Anti-aliasing samples (1 = no AA).
    pub samples: u32,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            num_frames: 60,
            frame_duration: 1.0 / 30.0,
            samples: 1,
        }
    }
}

impl AnimationConfig {
    /// Creates a new animation config.
    pub fn new(width: u32, height: u32, num_frames: usize) -> Self {
        Self {
            width,
            height,
            num_frames,
            ..Default::default()
        }
    }

    /// Sets the frame rate.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.frame_duration = 1.0 / fps;
        self
    }

    /// Sets the anti-aliasing samples.
    pub fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples.max(1);
        self
    }

    /// Returns the total animation duration in seconds.
    pub fn duration(&self) -> f32 {
        self.num_frames as f32 * self.frame_duration
    }
}

/// Renders an animation from a time-varying field to a sequence of images.
///
/// The field receives a Vec3 where xy are UV coordinates and z is time (0 to duration).
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_image::{render_animation, AnimationConfig};
///
/// let config = AnimationConfig::new(256, 256, 60).with_fps(30.0);
/// let frames = render_animation(&my_field, &config);
/// ```
pub fn render_animation<F: Field<glam::Vec3, Rgba>>(
    field: &F,
    config: &AnimationConfig,
) -> Vec<ImageField> {
    let ctx = EvalContext::new();
    let bake_config = BakeConfig {
        width: config.width,
        height: config.height,
        samples: config.samples,
    };

    (0..config.num_frames)
        .map(|frame| {
            let t = frame as f32 * config.frame_duration;

            // Create a wrapper field that adds time to the input
            let frame_field = TimeSliceField {
                inner: field,
                time: t,
            };

            bake_rgba(&frame_field, &bake_config, &ctx)
        })
        .collect()
}

/// Helper struct to slice a 3D field at a specific time.
struct TimeSliceField<'a, F> {
    inner: &'a F,
    time: f32,
}

impl<'a, F: Field<glam::Vec3, Rgba>> Field<Vec2, Rgba> for TimeSliceField<'a, F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> Rgba {
        self.inner
            .sample(glam::Vec3::new(input.x, input.y, self.time), ctx)
    }
}

/// Exports animation frames as a numbered image sequence.
///
/// Files are named `{prefix}_{frame:04}.png`.
pub fn export_image_sequence<P: AsRef<Path>>(
    frames: &[ImageField],
    directory: P,
    prefix: &str,
) -> Result<(), ImageFieldError> {
    let dir = directory.as_ref();
    std::fs::create_dir_all(dir)?;

    for (i, frame) in frames.iter().enumerate() {
        let filename = format!("{}_{:04}.png", prefix, i);
        let path = dir.join(filename);
        export_png(frame, path)?;
    }

    Ok(())
}

/// Exports animation frames as an animated GIF.
///
/// # Arguments
/// * `frames` - The frames to export
/// * `path` - Output file path
/// * `frame_delay_ms` - Delay between frames in milliseconds
pub fn export_gif<P: AsRef<Path>>(
    frames: &[ImageField],
    path: P,
    frame_delay_ms: u16,
) -> Result<(), ImageFieldError> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame};
    use std::fs::File;

    if frames.is_empty() {
        return Ok(());
    }

    let file = File::create(path)?;
    let mut encoder = GifEncoder::new(file);
    encoder.set_repeat(Repeat::Infinite)?;

    for img_field in frames {
        let (width, height) = img_field.dimensions();
        let mut rgba_buf = image::RgbaImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let color = img_field.sample_uv(u, v);

                rgba_buf.put_pixel(
                    x,
                    y,
                    image::Rgba([
                        (color.r * 255.0) as u8,
                        (color.g * 255.0) as u8,
                        (color.b * 255.0) as u8,
                        (color.a * 255.0) as u8,
                    ]),
                );
            }
        }

        let delay = Delay::from_numer_denom_ms(frame_delay_ms as u32, 1);
        let frame = Frame::from_parts(rgba_buf, 0, 0, delay);
        encoder.encode_frame(frame)?;
    }

    Ok(())
}

/// Renders a scalar field animation (grayscale).
pub fn render_animation_scalar<F: Field<glam::Vec3, f32>>(
    field: &F,
    config: &AnimationConfig,
) -> Vec<ImageField> {
    let ctx = EvalContext::new();
    let bake_config = BakeConfig {
        width: config.width,
        height: config.height,
        samples: config.samples,
    };

    (0..config.num_frames)
        .map(|frame| {
            let t = frame as f32 * config.frame_duration;

            let frame_field = TimeSliceFieldScalar {
                inner: field,
                time: t,
            };

            bake_scalar(&frame_field, &bake_config, &ctx)
        })
        .collect()
}

struct TimeSliceFieldScalar<'a, F> {
    inner: &'a F,
    time: f32,
}

impl<'a, F: Field<glam::Vec3, f32>> Field<Vec2, f32> for TimeSliceFieldScalar<'a, F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        self.inner
            .sample(glam::Vec3::new(input.x, input.y, self.time), ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image() -> ImageField {
        // 2x2 test image
        let data = vec![
            [1.0, 0.0, 0.0, 1.0], // Red
            [0.0, 1.0, 0.0, 1.0], // Green
            [0.0, 0.0, 1.0, 1.0], // Blue
            [1.0, 1.0, 1.0, 1.0], // White
        ];
        ImageField::from_raw(data, 2, 2)
    }

    #[test]
    fn test_nearest_sampling() {
        let img = create_test_image().with_filter_mode(FilterMode::Nearest);

        let tl = img.sample_uv(0.0, 0.0);
        assert!((tl.r - 1.0).abs() < 0.001);

        let tr = img.sample_uv(0.99, 0.0);
        assert!((tr.g - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bilinear_sampling() {
        let img = create_test_image().with_filter_mode(FilterMode::Bilinear);

        let center = img.sample_uv(0.5, 0.5);
        assert!(center.r > 0.1 && center.r < 0.9);
    }

    #[test]
    fn test_field_trait() {
        let img = create_test_image().with_filter_mode(FilterMode::Nearest);
        let ctx = EvalContext::new();

        let color: Rgba = img.sample(Vec2::new(0.0, 0.0), &ctx);
        assert!(color.r > 0.5);
    }

    // Texture baking tests

    /// A simple field that returns UV coordinates as grayscale.
    struct GradientField;

    impl Field<Vec2, f32> for GradientField {
        fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
            (input.x + input.y) / 2.0
        }
    }

    /// A simple field that returns UV coordinates as color.
    struct ColorGradientField;

    impl Field<Vec2, Rgba> for ColorGradientField {
        fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Rgba {
            Rgba::new(input.x, input.y, 0.5, 1.0)
        }
    }

    impl Field<Vec2, Vec4> for ColorGradientField {
        fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec4 {
            Vec4::new(input.x, input.y, 0.5, 1.0)
        }
    }

    #[test]
    fn test_bake_scalar() {
        let field = GradientField;
        let config = BakeConfig::new(4, 4);
        let ctx = EvalContext::new();

        let image = bake_scalar(&field, &config, &ctx);
        assert_eq!(image.dimensions(), (4, 4));

        // Top-left should be darker, bottom-right should be brighter
        let tl = image.sample_uv(0.125, 0.125); // First pixel center
        let br = image.sample_uv(0.875, 0.875); // Last pixel center
        assert!(tl.r < br.r);
    }

    #[test]
    fn test_bake_scalar_with_aa() {
        let field = GradientField;
        let config = BakeConfig::new(4, 4).with_samples(4);
        let ctx = EvalContext::new();

        let image = bake_scalar(&field, &config, &ctx);
        assert_eq!(image.dimensions(), (4, 4));
    }

    #[test]
    fn test_bake_rgba() {
        let field = ColorGradientField;
        let config = BakeConfig::new(4, 4);
        let ctx = EvalContext::new();

        let image = bake_rgba(&field, &config, &ctx);
        assert_eq!(image.dimensions(), (4, 4));

        // Check that colors vary as expected
        let tl = image.sample_uv(0.125, 0.125);
        let br = image.sample_uv(0.875, 0.875);
        assert!(tl.r < br.r); // Red increases with X
        assert!(tl.g < br.g); // Green increases with Y
    }

    #[test]
    fn test_bake_vec4() {
        let field = ColorGradientField;
        let config = BakeConfig::new(4, 4);
        let ctx = EvalContext::new();

        let image = bake_vec4(&field, &config, &ctx);
        assert_eq!(image.dimensions(), (4, 4));
    }

    #[test]
    fn test_bake_config_builder() {
        let config = BakeConfig::new(512, 256).with_samples(16);
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 256);
        assert_eq!(config.samples, 16);
    }

    #[test]
    fn test_bake_config_default() {
        let config = BakeConfig::default();
        assert_eq!(config.width, 256);
        assert_eq!(config.height, 256);
        assert_eq!(config.samples, 1);
    }

    // Animation tests

    /// A time-varying color field for animation testing.
    struct AnimatedField;

    impl Field<glam::Vec3, Rgba> for AnimatedField {
        fn sample(&self, input: glam::Vec3, _ctx: &EvalContext) -> Rgba {
            // Color changes with time (z coordinate)
            let r = (input.x + input.z) % 1.0;
            let g = input.y;
            let b = input.z;
            Rgba::new(r, g, b, 1.0)
        }
    }

    /// A time-varying scalar field.
    struct AnimatedScalarField;

    impl Field<glam::Vec3, f32> for AnimatedScalarField {
        fn sample(&self, input: glam::Vec3, _ctx: &EvalContext) -> f32 {
            // Value oscillates with time
            ((input.x + input.z * 10.0) * std::f32::consts::PI).sin() * 0.5 + 0.5
        }
    }

    #[test]
    fn test_animation_config() {
        let config = AnimationConfig::new(128, 128, 30).with_fps(60.0);
        assert_eq!(config.width, 128);
        assert_eq!(config.height, 128);
        assert_eq!(config.num_frames, 30);
        assert!((config.frame_duration - 1.0 / 60.0).abs() < 0.0001);
        assert!((config.duration() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_render_animation() {
        let field = AnimatedField;
        let config = AnimationConfig::new(4, 4, 5).with_fps(10.0);

        let frames = render_animation(&field, &config);
        assert_eq!(frames.len(), 5);

        // Each frame should have the correct dimensions
        for frame in &frames {
            assert_eq!(frame.dimensions(), (4, 4));
        }

        // Frames should be different (animation is happening)
        let first = frames[0].sample_uv(0.5, 0.5);
        let last = frames[4].sample_uv(0.5, 0.5);
        // Blue channel changes with time
        assert!((first.b - last.b).abs() > 0.001);
    }

    #[test]
    fn test_render_animation_scalar() {
        let field = AnimatedScalarField;
        let config = AnimationConfig::new(4, 4, 3);

        let frames = render_animation_scalar(&field, &config);
        assert_eq!(frames.len(), 3);
    }
}
