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

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use image::{DynamicImage, GenericImageView, ImageError};

pub use rhizome_resin_color::BlendMode;
use rhizome_resin_color::{Rgba, blend_with_alpha};
use rhizome_resin_field::{EvalContext, Field};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
#[derive(Debug, thiserror::Error)]
pub enum ImageFieldError {
    /// Failed to load the image file.
    #[error("Image error: {0}")]
    ImageError(#[from] ImageError),
    /// I/O error reading the file.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
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

    /// Creates a solid color image field (1x1, tiles via wrap mode).
    pub fn solid(color: Rgba) -> Self {
        Self {
            data: vec![[color.r, color.g, color.b, color.a]],
            width: 1,
            height: 1,
            wrap_mode: WrapMode::Repeat,
            filter_mode: FilterMode::Nearest,
        }
    }

    /// Creates a solid color image field with specific dimensions.
    pub fn solid_sized(width: u32, height: u32, color: [f32; 4]) -> Self {
        Self {
            data: vec![color; (width * height) as usize],
            width,
            height,
            wrap_mode: WrapMode::default(),
            filter_mode: FilterMode::default(),
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
    pub fn get_pixel(&self, x: u32, y: u32) -> [f32; 4] {
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = BakeConfig))]
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

    /// Applies this configuration (returns self as a generator op).
    pub fn apply(&self) -> BakeConfig {
        self.clone()
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = AnimationConfig))]
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

    /// Applies this configuration (returns self as a generator op).
    pub fn apply(&self) -> AnimationConfig {
        self.clone()
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

// ============================================================================
// Convolution filters
// ============================================================================

/// A convolution kernel for image filtering.
///
/// Kernels are square matrices of odd dimensions (3x3, 5x5, etc.).
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Kernel weights in row-major order.
    pub weights: Vec<f32>,
    /// Kernel size (width and height).
    pub size: usize,
}

impl Kernel {
    /// Creates a new kernel from weights.
    ///
    /// The weights array must have length `size * size`.
    pub fn new(weights: Vec<f32>, size: usize) -> Self {
        assert_eq!(
            weights.len(),
            size * size,
            "Kernel weights must be size×size"
        );
        assert!(size % 2 == 1, "Kernel size must be odd");
        Self { weights, size }
    }

    /// Creates a 3x3 identity kernel (no change).
    pub fn identity() -> Self {
        Self::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3)
    }

    /// Creates a 3x3 box blur kernel.
    pub fn box_blur() -> Self {
        let w = 1.0 / 9.0;
        Self::new(vec![w, w, w, w, w, w, w, w, w], 3)
    }

    /// Creates a 3x3 Gaussian blur kernel.
    pub fn gaussian_blur_3x3() -> Self {
        let weights = vec![
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        ];
        Self::new(weights, 3)
    }

    /// Creates a 5x5 Gaussian blur kernel.
    pub fn gaussian_blur_5x5() -> Self {
        let weights = vec![
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, 36.0, 24.0, 6.0, 4.0,
            16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ]
        .iter()
        .map(|&x| x / 256.0)
        .collect();
        Self::new(weights, 5)
    }

    /// Creates a 3x3 sharpen kernel.
    pub fn sharpen() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], 3)
    }

    /// Creates a 3x3 unsharp mask kernel.
    pub fn unsharp_mask() -> Self {
        Self::new(
            vec![
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                17.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
            ],
            3,
        )
    }

    /// Creates a 3x3 Sobel edge detection kernel (horizontal edges).
    pub fn sobel_horizontal() -> Self {
        Self::new(vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3)
    }

    /// Creates a 3x3 Sobel edge detection kernel (vertical edges).
    pub fn sobel_vertical() -> Self {
        Self::new(vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3)
    }

    /// Creates a 3x3 Laplacian edge detection kernel.
    pub fn laplacian() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0], 3)
    }

    /// Creates a 3x3 Laplacian of Gaussian (LoG) approximation.
    pub fn laplacian_of_gaussian() -> Self {
        Self::new(vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0], 3)
    }

    /// Creates a 3x3 emboss kernel.
    pub fn emboss() -> Self {
        Self::new(vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3)
    }

    /// Creates a 3x3 edge enhancement kernel.
    pub fn edge_enhance() -> Self {
        Self::new(vec![0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3)
    }

    /// Returns the kernel radius (distance from center to edge).
    pub fn radius(&self) -> usize {
        self.size / 2
    }
}

/// Applies a convolution kernel to an image.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Kernel, convolve};
/// use rhizome_resin_color::Rgba;
///
/// // Create a simple 3x3 test image
/// let data = vec![
///     [0.5, 0.5, 0.5, 1.0]; 9
/// ];
/// let img = ImageField::from_raw(data, 3, 3);
///
/// let blurred = convolve(&img, &Kernel::box_blur());
/// ```
pub fn convolve(image: &ImageField, kernel: &Kernel) -> ImageField {
    let (width, height) = image.dimensions();
    let radius = kernel.radius() as i32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for ky in 0..kernel.size {
                for kx in 0..kernel.size {
                    let weight = kernel.weights[ky * kernel.size + kx];

                    // Sample with clamping at edges
                    let sx = (x as i32 + kx as i32 - radius).clamp(0, width as i32 - 1) as u32;
                    let sy = (y as i32 + ky as i32 - radius).clamp(0, height as i32 - 1) as u32;

                    let pixel = image.get_pixel(sx, sy);
                    r += pixel[0] * weight;
                    g += pixel[1] * weight;
                    b += pixel[2] * weight;
                    a += pixel[3] * weight;
                }
            }

            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies Sobel edge detection and returns the edge magnitude.
///
/// Combines horizontal and vertical Sobel kernels to detect edges in all directions.
pub fn detect_edges(image: &ImageField) -> ImageField {
    let horizontal = convolve(image, &Kernel::sobel_horizontal());
    let vertical = convolve(image, &Kernel::sobel_vertical());

    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let h = horizontal.get_pixel(x, y);
            let v = vertical.get_pixel(x, y);

            // Compute magnitude for each channel
            let mag_r = (h[0] * h[0] + v[0] * v[0]).sqrt();
            let mag_g = (h[1] * h[1] + v[1] * v[1]).sqrt();
            let mag_b = (h[2] * h[2] + v[2] * v[2]).sqrt();

            // Average the channels for grayscale edge output
            let mag = (mag_r + mag_g + mag_b) / 3.0;
            data.push([mag, mag, mag, 1.0]);
        }
    }

    ImageField::from_raw(data, width, height)
}

/// Applies a Gaussian blur with the specified number of passes.
///
/// Multiple passes of a small kernel approximate a larger blur radius.
pub fn blur(image: &ImageField, passes: u32) -> ImageField {
    let kernel = Kernel::gaussian_blur_3x3();
    let mut result = image.clone();

    for _ in 0..passes {
        result = convolve(&result, &kernel);
    }

    result
}

/// Sharpens an image.
pub fn sharpen(image: &ImageField) -> ImageField {
    convolve(image, &Kernel::sharpen())
}

/// Applies emboss effect to an image.
pub fn emboss(image: &ImageField) -> ImageField {
    let embossed = convolve(image, &Kernel::emboss());

    // Normalize emboss output to visible range (add 0.5 bias)
    let (width, height) = embossed.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = embossed.get_pixel(x, y);
            data.push([
                (pixel[0] + 0.5).clamp(0.0, 1.0),
                (pixel[1] + 0.5).clamp(0.0, 1.0),
                (pixel[2] + 0.5).clamp(0.0, 1.0),
                pixel[3].clamp(0.0, 1.0),
            ]);
        }
    }

    ImageField::from_raw(data, width, height)
}

// ============================================================================
// Channel operations
// ============================================================================

/// Which channel to extract or operate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Alpha channel.
    Alpha,
}

/// Extracts a single channel as a grayscale image.
///
/// The extracted channel is stored in all RGB channels of the output,
/// with alpha set to 1.0.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Channel, extract_channel};
///
/// let data = vec![[1.0, 0.5, 0.25, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let red = extract_channel(&img, Channel::Red);
/// assert_eq!(red.get_pixel(0, 0)[0], 1.0);
///
/// let green = extract_channel(&img, Channel::Green);
/// assert_eq!(green.get_pixel(0, 0)[0], 0.5);
/// ```
pub fn extract_channel(image: &ImageField, channel: Channel) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    let idx = match channel {
        Channel::Red => 0,
        Channel::Green => 1,
        Channel::Blue => 2,
        Channel::Alpha => 3,
    };

    for y in 0..height {
        for x in 0..width {
            let v = image.get_pixel(x, y)[idx];
            data.push([v, v, v, 1.0]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Splits an image into separate R, G, B, A grayscale images.
///
/// Returns a tuple of (red, green, blue, alpha) images.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, split_channels};
///
/// let data = vec![[1.0, 0.5, 0.25, 0.75]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let (r, g, b, a) = split_channels(&img);
/// assert_eq!(r.get_pixel(0, 0)[0], 1.0);
/// assert_eq!(g.get_pixel(0, 0)[0], 0.5);
/// assert_eq!(b.get_pixel(0, 0)[0], 0.25);
/// assert_eq!(a.get_pixel(0, 0)[0], 0.75);
/// ```
pub fn split_channels(image: &ImageField) -> (ImageField, ImageField, ImageField, ImageField) {
    (
        extract_channel(image, Channel::Red),
        extract_channel(image, Channel::Green),
        extract_channel(image, Channel::Blue),
        extract_channel(image, Channel::Alpha),
    )
}

/// Merges separate grayscale images into a single RGBA image.
///
/// Each input image's red channel is used as that channel's value.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, merge_channels};
///
/// let r = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
/// let g = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 4], 2, 2);
/// let b = ImageField::from_raw(vec![[0.25, 0.25, 0.25, 1.0]; 4], 2, 2);
/// let a = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
///
/// let merged = merge_channels(&r, &g, &b, &a);
/// let pixel = merged.get_pixel(0, 0);
/// assert_eq!(pixel[0], 1.0);
/// assert_eq!(pixel[1], 0.5);
/// assert_eq!(pixel[2], 0.25);
/// ```
pub fn merge_channels(
    red: &ImageField,
    green: &ImageField,
    blue: &ImageField,
    alpha: &ImageField,
) -> ImageField {
    let (width, height) = red.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = red.get_pixel(x, y)[0];
            let g = green.get_pixel(x, y)[0];
            let b = blue.get_pixel(x, y)[0];
            let a = alpha.get_pixel(x, y)[0];
            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(red.wrap_mode)
        .with_filter_mode(red.filter_mode)
}

/// Replaces a single channel in an image.
///
/// The source image's red channel is used as the replacement value.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Channel, set_channel};
///
/// let img = ImageField::from_raw(vec![[0.0, 0.0, 0.0, 1.0]; 4], 2, 2);
/// let new_red = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
///
/// let result = set_channel(&img, Channel::Red, &new_red);
/// assert_eq!(result.get_pixel(0, 0)[0], 1.0);
/// assert_eq!(result.get_pixel(0, 0)[1], 0.0); // Green unchanged
/// ```
pub fn set_channel(image: &ImageField, channel: Channel, source: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    let idx = match channel {
        Channel::Red => 0,
        Channel::Green => 1,
        Channel::Blue => 2,
        Channel::Alpha => 3,
    };

    for y in 0..height {
        for x in 0..width {
            let mut pixel = image.get_pixel(x, y);
            pixel[idx] = source.get_pixel(x, y)[0];
            data.push(pixel);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a function to a single channel of an image.
///
/// This primitive extracts a channel as a grayscale image, transforms it,
/// and puts it back. Useful for per-channel effects like independent blur,
/// noise, or distortion.
///
/// # Arguments
///
/// * `image` - Source image
/// * `channel` - Channel to transform
/// * `f` - Function that transforms a grayscale `ImageField` and returns a new one
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Channel, map_channel, convolve, Kernel};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 64], 8, 8);
///
/// // Blur only the red channel
/// let result = map_channel(&img, Channel::Red, |ch| {
///     convolve(&ch, &Kernel::box_blur())
/// });
///
/// // Apply noise only to the blue channel
/// let result = map_channel(&img, Channel::Blue, |ch| {
///     // ch is a grayscale image of the blue channel
///     ch // return transformed channel
/// });
/// ```
pub fn map_channel(
    image: &ImageField,
    channel: Channel,
    f: impl FnOnce(ImageField) -> ImageField,
) -> ImageField {
    let extracted = extract_channel(image, channel);
    let transformed = f(extracted);
    set_channel(image, channel, &transformed)
}

/// Swaps two channels in an image.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Channel, swap_channels};
///
/// let img = ImageField::from_raw(vec![[1.0, 0.5, 0.0, 1.0]; 4], 2, 2);
/// let swapped = swap_channels(&img, Channel::Red, Channel::Blue);
///
/// assert_eq!(swapped.get_pixel(0, 0)[0], 0.0);  // Was blue
/// assert_eq!(swapped.get_pixel(0, 0)[2], 1.0);  // Was red
/// ```
pub fn swap_channels(image: &ImageField, a: Channel, b: Channel) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    let idx_a = match a {
        Channel::Red => 0,
        Channel::Green => 1,
        Channel::Blue => 2,
        Channel::Alpha => 3,
    };
    let idx_b = match b {
        Channel::Red => 0,
        Channel::Green => 1,
        Channel::Blue => 2,
        Channel::Alpha => 3,
    };

    for y in 0..height {
        for x in 0..width {
            let mut pixel = image.get_pixel(x, y);
            pixel.swap(idx_a, idx_b);
            data.push(pixel);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

// ============================================================================
// Colorspace Decomposition
// ============================================================================

/// Colorspace for decomposition/reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Colorspace {
    /// RGB (Red, Green, Blue) - the native colorspace.
    Rgb,
    /// HSL (Hue, Saturation, Lightness).
    Hsl,
    /// HSV (Hue, Saturation, Value).
    Hsv,
    /// HWB (Hue, Whiteness, Blackness) - CSS Color Level 4.
    Hwb,
    /// YCbCr (Luma, Blue-difference, Red-difference chroma).
    YCbCr,
    /// LAB (CIE L*a*b* perceptual colorspace).
    Lab,
    /// LCH (Lightness, Chroma, Hue) - cylindrical LAB.
    Lch,
    /// OkLab (perceptually uniform colorspace).
    OkLab,
    /// OkLCH (cylindrical OkLab) - CSS Color Level 4.
    OkLch,
}

/// Decomposed colorspace channels.
///
/// Contains three channels representing the colorspace components.
/// Channel values are normalized to [0, 1] for storage.
pub struct ColorspaceChannels {
    /// First channel (H/Y/L depending on colorspace).
    pub c0: ImageField,
    /// Second channel (S/Cb/a depending on colorspace).
    pub c1: ImageField,
    /// Third channel (L/V/Cr/b depending on colorspace).
    pub c2: ImageField,
    /// Alpha channel (preserved from original).
    pub alpha: ImageField,
    /// Which colorspace these channels represent.
    pub colorspace: Colorspace,
}

/// Decomposes an RGB image into the specified colorspace.
///
/// Returns separate grayscale images for each channel, normalized to [0, 1].
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, Colorspace, decompose_colorspace, reconstruct_colorspace};
///
/// let image = ImageField::solid_sized(64, 64, [0.8, 0.4, 0.2, 1.0]);
///
/// // Decompose to HSL
/// let channels = decompose_colorspace(&image, Colorspace::Hsl);
///
/// // Modify saturation channel...
///
/// // Reconstruct back to RGB
/// let result = reconstruct_colorspace(&channels);
/// ```
pub fn decompose_colorspace(image: &ImageField, colorspace: Colorspace) -> ColorspaceChannels {
    let (width, height) = image.dimensions();
    let size = (width * height) as usize;

    let mut c0_data = Vec::with_capacity(size);
    let mut c1_data = Vec::with_capacity(size);
    let mut c2_data = Vec::with_capacity(size);
    let mut alpha_data = Vec::with_capacity(size);

    for i in 0..size {
        let pixel = image.data[i];
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let a = pixel[3];

        let (v0, v1, v2) = match colorspace {
            Colorspace::Rgb => (r, g, b),
            Colorspace::Hsl => rgb_to_hsl(r, g, b),
            Colorspace::Hsv => rgb_to_hsv(r, g, b),
            Colorspace::Hwb => rgb_to_hwb(r, g, b),
            Colorspace::YCbCr => rgb_to_ycbcr(r, g, b),
            Colorspace::Lab => rgb_to_lab(r, g, b),
            Colorspace::Lch => rgb_to_lch(r, g, b),
            Colorspace::OkLab => rgb_to_oklab(r, g, b),
            Colorspace::OkLch => rgb_to_oklch(r, g, b),
        };

        c0_data.push([v0, v0, v0, 1.0]);
        c1_data.push([v1, v1, v1, 1.0]);
        c2_data.push([v2, v2, v2, 1.0]);
        alpha_data.push([a, a, a, 1.0]);
    }

    ColorspaceChannels {
        c0: ImageField::from_raw(c0_data, width, height),
        c1: ImageField::from_raw(c1_data, width, height),
        c2: ImageField::from_raw(c2_data, width, height),
        alpha: ImageField::from_raw(alpha_data, width, height),
        colorspace,
    }
}

/// Reconstructs an RGB image from colorspace channels.
pub fn reconstruct_colorspace(channels: &ColorspaceChannels) -> ImageField {
    let (width, height) = channels.c0.dimensions();
    let size = (width * height) as usize;

    let mut data = Vec::with_capacity(size);

    for i in 0..size {
        let v0 = channels.c0.data[i][0];
        let v1 = channels.c1.data[i][0];
        let v2 = channels.c2.data[i][0];
        let a = channels.alpha.data[i][0];

        let (r, g, b) = match channels.colorspace {
            Colorspace::Rgb => (v0, v1, v2),
            Colorspace::Hsl => hsl_to_rgb(v0, v1, v2),
            Colorspace::Hsv => hsv_to_rgb(v0, v1, v2),
            Colorspace::Hwb => hwb_to_rgb(v0, v1, v2),
            Colorspace::YCbCr => ycbcr_to_rgb(v0, v1, v2),
            Colorspace::Lab => lab_to_rgb(v0, v1, v2),
            Colorspace::Lch => lch_to_rgb(v0, v1, v2),
            Colorspace::OkLab => oklab_to_rgb(v0, v1, v2),
            Colorspace::OkLch => oklch_to_rgb(v0, v1, v2),
        };

        data.push([r, g, b, a]);
    }

    ImageField::from_raw(data, width, height)
}

// --- Colorspace conversion helpers ---

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let hue_to_rgb = |t: f32| -> f32 {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    (
        hue_to_rgb(h + 1.0 / 3.0),
        hue_to_rgb(h),
        hue_to_rgb(h - 1.0 / 3.0),
    )
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let v = max;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, v);
    }

    let d = max - min;
    let s = d / max;

    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (v, v, v);
    }

    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - h.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = (-0.169 * r - 0.331 * g + 0.500 * b) + 0.5;
    let cr = (0.500 * r - 0.419 * g - 0.081 * b) + 0.5;
    (y, cb, cr)
}

fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let cb = cb - 0.5;
    let cr = cr - 0.5;
    let r = (y + 1.402 * cr).clamp(0.0, 1.0);
    let g = (y - 0.344 * cb - 0.714 * cr).clamp(0.0, 1.0);
    let b = (y + 1.772 * cb).clamp(0.0, 1.0);
    (r, g, b)
}

fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB to linear RGB
    let to_linear = |v: f32| -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };

    let r = to_linear(r);
    let g = to_linear(g);
    let b = to_linear(b);

    // Linear RGB to XYZ (D65 illuminant)
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // XYZ to Lab (D65 reference white)
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;

    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    let l = (116.0 * fy - 16.0) / 100.0; // Normalize L* to [0, 1]
    let a = ((500.0 * (fx - fy)) + 128.0) / 255.0; // Normalize a* to [0, 1]
    let lab_b = ((200.0 * (fy - fz)) + 128.0) / 255.0; // Normalize b* to [0, 1]

    (l.clamp(0.0, 1.0), a.clamp(0.0, 1.0), lab_b.clamp(0.0, 1.0))
}

fn lab_to_rgb(l: f32, a: f32, lab_b: f32) -> (f32, f32, f32) {
    // Denormalize
    let l = l * 100.0;
    let a = a * 255.0 - 128.0;
    let b = lab_b * 255.0 - 128.0;

    // Lab to XYZ
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;

    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let f_inv = |t: f32| -> f32 {
        if t > 0.206893 {
            t * t * t
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };

    let x = xn * f_inv(fx);
    let y = yn * f_inv(fy);
    let z = zn * f_inv(fz);

    // XYZ to linear RGB
    let r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

    // Linear RGB to sRGB
    let from_linear = |v: f32| -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    (
        from_linear(r).clamp(0.0, 1.0),
        from_linear(g).clamp(0.0, 1.0),
        from_linear(b).clamp(0.0, 1.0),
    )
}

fn rgb_to_hwb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (h, _, _) = rgb_to_hsl(r, g, b);
    let w = r.min(g).min(b);
    let b_val = 1.0 - r.max(g).max(b);
    (h, w, b_val)
}

fn hwb_to_rgb(h: f32, w: f32, b: f32) -> (f32, f32, f32) {
    // If w + b >= 1, result is gray
    if w + b >= 1.0 {
        let gray = w / (w + b);
        return (gray, gray, gray);
    }

    // Convert via HSV: HWB(h, w, b) = HSV(h, 1 - w/(1-b), 1-b)
    let v = 1.0 - b;
    let s = 1.0 - w / v;
    hsv_to_rgb(h, s, v)
}

fn rgb_to_lch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l, a, b_val) = rgb_to_lab(r, g, b);
    // Denormalize a and b from [0,1] to [-128, 127] range for math
    let a_real = a * 255.0 - 128.0;
    let b_real = b_val * 255.0 - 128.0;

    let c = (a_real * a_real + b_real * b_real).sqrt();
    let h = b_real.atan2(a_real);
    // Normalize: L is already [0,1], C to [0,1] (max ~181), H to [0,1]
    let c_norm = (c / 181.0).clamp(0.0, 1.0);
    let h_norm = (h / std::f32::consts::TAU).rem_euclid(1.0);
    (l, c_norm, h_norm)
}

fn lch_to_rgb(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    // Denormalize
    let c_real = c * 181.0;
    let h_real = h * std::f32::consts::TAU;

    let a_real = c_real * h_real.cos();
    let b_real = c_real * h_real.sin();

    // Normalize back to [0,1] for lab_to_rgb
    let a = (a_real + 128.0) / 255.0;
    let b_val = (b_real + 128.0) / 255.0;

    lab_to_rgb(l, a, b_val)
}

fn rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB to linear
    let to_linear = |v: f32| -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };

    let r = to_linear(r);
    let g = to_linear(g);
    let b = to_linear(b);

    // Linear RGB to OkLab via LMS
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    // Normalize: L is [0,1], a/b are roughly [-0.4, 0.4], normalize to [0,1]
    (
        lab_l.clamp(0.0, 1.0),
        ((lab_a + 0.4) / 0.8).clamp(0.0, 1.0),
        ((lab_b + 0.4) / 0.8).clamp(0.0, 1.0),
    )
}

fn oklab_to_rgb(l: f32, a: f32, b_val: f32) -> (f32, f32, f32) {
    // Denormalize a and b
    let lab_a = a * 0.8 - 0.4;
    let lab_b = b_val * 0.8 - 0.4;

    let l_ = l + 0.3963377774 * lab_a + 0.2158037573 * lab_b;
    let m_ = l - 0.1055613458 * lab_a - 0.0638541728 * lab_b;
    let s_ = l - 0.0894841775 * lab_a - 1.2914855480 * lab_b;

    let l_cubed = l_ * l_ * l_;
    let m_cubed = m_ * m_ * m_;
    let s_cubed = s_ * s_ * s_;

    let r = 4.0767416621 * l_cubed - 3.3077115913 * m_cubed + 0.2309699292 * s_cubed;
    let g = -1.2684380046 * l_cubed + 2.6097574011 * m_cubed - 0.3413193965 * s_cubed;
    let b = -0.0041960863 * l_cubed - 0.7034186147 * m_cubed + 1.7076147010 * s_cubed;

    // Linear to sRGB
    let from_linear = |v: f32| -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    (
        from_linear(r).clamp(0.0, 1.0),
        from_linear(g).clamp(0.0, 1.0),
        from_linear(b).clamp(0.0, 1.0),
    )
}

fn rgb_to_oklch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l, a, b_val) = rgb_to_oklab(r, g, b);
    // Denormalize a and b
    let a_real = a * 0.8 - 0.4;
    let b_real = b_val * 0.8 - 0.4;

    let c = (a_real * a_real + b_real * b_real).sqrt();
    let h = b_real.atan2(a_real);
    // Normalize: C max is about 0.4, H to [0,1]
    let c_norm = (c / 0.4).clamp(0.0, 1.0);
    let h_norm = (h / std::f32::consts::TAU).rem_euclid(1.0);
    (l, c_norm, h_norm)
}

fn oklch_to_rgb(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let c_real = c * 0.4;
    let h_real = h * std::f32::consts::TAU;

    let a_real = c_real * h_real.cos();
    let b_real = c_real * h_real.sin();

    // Normalize back to [0,1] for oklab_to_rgb
    let a = (a_real + 0.4) / 0.8;
    let b_val = (b_real + 0.4) / 0.8;

    oklab_to_rgb(l, a, b_val)
}

// ============================================================================
// Chromatic aberration
// ============================================================================

/// Applies chromatic aberration effect to an image.
///
/// This simulates lens chromatic aberration by offsetting each color channel
/// radially from the center point. Positive offsets push the channel outward,
/// negative offsets push inward.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
/// use rhizome_resin_image::{ImageField, chromatic_aberration, ChromaticAberrationConfig};
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

// ============================================================================
// Color adjustments
// ============================================================================

/// Applies levels adjustment to an image.
///
/// This is similar to Photoshop's Levels adjustment:
/// 1. Remap input range [input_black, input_white] to [0, 1]
/// 2. Apply gamma correction
/// 3. Remap [0, 1] to [output_black, output_white]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
/// use rhizome_resin_image::{ImageField, adjust_levels, LevelsConfig};
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
/// use rhizome_resin_image::{ImageField, adjust_brightness_contrast};
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
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    // Convert contrast to multiplier: 0 = 1x, 1 = 2x, -1 = 0x
    let contrast_factor = (1.0 + contrast).max(0.0);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);

            let adjust = |v: f32| -> f32 {
                // Apply contrast around midpoint, then brightness
                let contrasted = (v - 0.5) * contrast_factor + 0.5;
                (contrasted + brightness).clamp(0.0, 1.0)
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
/// use rhizome_resin_image::{ImageField, adjust_hsl, HslAdjustment};
///
/// let data = vec![[1.0, 0.5, 0.0, 1.0]; 4]; // Orange
/// let img = ImageField::from_raw(data, 2, 2);
///
/// // Shift hue by 180 degrees (complement)
/// let result = adjust_hsl(&img, &HslAdjustment::hue(0.5));
/// ```
pub fn adjust_hsl(image: &ImageField, adjustment: &HslAdjustment) -> ImageField {
    use rhizome_resin_color::{Hsl, LinearRgb};

    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);

            // Convert to HSL
            let rgb = LinearRgb::new(pixel[0], pixel[1], pixel[2]);
            let hsl = rgb.to_hsl();

            // Apply adjustments
            let new_h = (hsl.h + adjustment.hue_shift).rem_euclid(1.0);
            let new_s = (hsl.s * (1.0 + adjustment.saturation)).clamp(0.0, 1.0);
            let new_l = (hsl.l + adjustment.lightness).clamp(0.0, 1.0);

            // Convert back
            let new_rgb = Hsl::new(new_h, new_s, new_l).to_rgb();

            data.push([new_rgb.r, new_rgb.g, new_rgb.b, pixel[3]]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
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

// ============================================================================
// Dithering - Decomposed Primitives
// ============================================================================

// -----------------------------------------------------------------------------
// Quantize - primitive operation
// -----------------------------------------------------------------------------

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
fn hilbert_d2xy(order: u32, d: u32) -> (u32, u32) {
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

// ============================================================================
// Compositing
// ============================================================================

/// Composites an overlay image onto a base image using the specified blend mode.
///
/// This is the fundamental primitive for combining two images. Higher-level
/// effects like drop shadow, glow, and bloom are built on top of this.
///
/// # Arguments
///
/// * `base` - The background image
/// * `overlay` - The foreground image to composite on top
/// * `mode` - The blend mode to use (Normal, Multiply, Screen, Add, etc.)
/// * `opacity` - Overall opacity of the overlay (0.0 = invisible, 1.0 = full)
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, composite, BlendMode};
///
/// let base = ImageField::solid_sized(100, 100, [0.2, 0.2, 0.2, 1.0]);
/// let overlay = ImageField::solid_sized(100, 100, [1.0, 0.0, 0.0, 0.5]);
///
/// // Normal blend at 80% opacity
/// let result = composite(&base, &overlay, BlendMode::Normal, 0.8);
///
/// // Additive blend for glow effects
/// let glow = composite(&base, &overlay, BlendMode::Add, 1.0);
/// ```
pub fn composite(
    base: &ImageField,
    overlay: &ImageField,
    mode: BlendMode,
    opacity: f32,
) -> ImageField {
    let (width, height) = base.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let base_pixel = base.get_pixel(x, y);
            // Sample overlay at same position, handling size mismatch via UV sampling
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let overlay_pixel = overlay.sample_uv(u, v);

            let base_rgba = Rgba::new(base_pixel[0], base_pixel[1], base_pixel[2], base_pixel[3]);
            let result = blend_with_alpha(base_rgba, overlay_pixel, mode, opacity);

            data.push([result.r, result.g, result.b, result.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(base.wrap_mode)
        .with_filter_mode(base.filter_mode)
}

// ============================================================================
// High-Level Effects (composed from primitives)
// ============================================================================

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
/// use rhizome_resin_image::{ImageField, DropShadow, drop_shadow};
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
/// use rhizome_resin_image::{ImageField, Glow, glow};
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
/// use rhizome_resin_image::{ImageField, Bloom, bloom};
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

// ============================================================================
// Distortion effects
// ============================================================================

/// Applies radial lens distortion (barrel or pincushion).
///
/// Barrel distortion (positive strength) makes the image bulge outward.
/// Pincushion distortion (negative strength) makes it pinch inward.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
/// use rhizome_resin_image::{ImageField, lens_distortion, LensDistortionConfig};
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
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
/// use rhizome_resin_image::{ImageField, wave_distortion, WaveDistortionConfig};
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
/// use rhizome_resin_image::{ImageField, displace};
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
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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

// ============================================================================
// Image pyramid
// ============================================================================

/// Downsamples an image by half using box filtering (averaging 2x2 blocks).
///
/// The output dimensions are `(width / 2, height / 2)`.
/// If dimensions are odd, the last row/column is included in the final average.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, downsample};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let half = downsample(&img);
/// assert_eq!(half.dimensions(), (2, 2));
/// ```
pub fn downsample(image: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let new_width = (width / 2).max(1);
    let new_height = (height / 2).max(1);

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    for y in 0..new_height {
        for x in 0..new_width {
            // Average 2x2 block
            let x0 = x * 2;
            let y0 = y * 2;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let p00 = image.get_pixel(x0, y0);
            let p10 = image.get_pixel(x1, y0);
            let p01 = image.get_pixel(x0, y1);
            let p11 = image.get_pixel(x1, y1);

            let avg = [
                (p00[0] + p10[0] + p01[0] + p11[0]) / 4.0,
                (p00[1] + p10[1] + p01[1] + p11[1]) / 4.0,
                (p00[2] + p10[2] + p01[2] + p11[2]) / 4.0,
                (p00[3] + p10[3] + p01[3] + p11[3]) / 4.0,
            ];

            data.push(avg);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Upsamples an image by 2x using bilinear interpolation.
///
/// The output dimensions are `(width * 2, height * 2)`.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, upsample};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let double = upsample(&img);
/// assert_eq!(double.dimensions(), (4, 4));
/// ```
pub fn upsample(image: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let new_width = width * 2;
    let new_height = height * 2;

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    // Use bilinear sampling
    let bilinear_image = ImageField {
        filter_mode: FilterMode::Bilinear,
        ..image.clone()
    };

    for y in 0..new_height {
        for x in 0..new_width {
            // Map to source coordinates
            let u = (x as f32 + 0.5) / new_width as f32;
            let v = (y as f32 + 0.5) / new_height as f32;

            let color = bilinear_image.sample_uv(u, v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// An image pyramid for multi-scale processing.
///
/// Contains progressively downsampled versions of the original image.
#[derive(Debug, Clone)]
pub struct ImagePyramid {
    /// Pyramid levels, from finest (original) to coarsest.
    pub levels: Vec<ImageField>,
}

impl ImagePyramid {
    /// Creates a Gaussian pyramid by repeatedly downsampling.
    ///
    /// # Arguments
    /// * `image` - Source image (becomes level 0)
    /// * `num_levels` - Total number of pyramid levels (including original)
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_resin_image::{ImageField, ImagePyramid};
    ///
    /// let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
    /// let img = ImageField::from_raw(data, 8, 8);
    ///
    /// let pyramid = ImagePyramid::gaussian(&img, 4);
    /// assert_eq!(pyramid.levels.len(), 4);
    /// assert_eq!(pyramid.levels[0].dimensions(), (8, 8));
    /// assert_eq!(pyramid.levels[1].dimensions(), (4, 4));
    /// ```
    pub fn gaussian(image: &ImageField, num_levels: usize) -> Self {
        let num_levels = num_levels.max(1);
        let mut levels = Vec::with_capacity(num_levels);

        // Level 0 is the original (optionally blurred)
        let blurred = blur(image, 1);
        levels.push(blurred);

        // Build remaining levels
        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let (w, h) = prev.dimensions();

            // Stop if we can't downsample further
            if w <= 1 && h <= 1 {
                break;
            }

            let downsampled = downsample(prev);
            let blurred = blur(&downsampled, 1);
            levels.push(blurred);
        }

        Self { levels }
    }

    /// Creates a Laplacian pyramid (difference-of-Gaussians).
    ///
    /// Each level stores the difference between consecutive Gaussian levels,
    /// which captures detail at that scale.
    ///
    /// The final level stores the coarsest Gaussian level (the residual).
    pub fn laplacian(image: &ImageField, num_levels: usize) -> Self {
        let gaussian = Self::gaussian(image, num_levels);
        let mut levels = Vec::with_capacity(gaussian.levels.len());

        for i in 0..gaussian.levels.len() - 1 {
            let current = &gaussian.levels[i];
            let next_upsampled = upsample(&gaussian.levels[i + 1]);

            // Compute difference (detail at this level)
            let (width, height) = current.dimensions();
            let mut diff_data = Vec::with_capacity((width * height) as usize);

            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;

                    let c1 = current.get_pixel(x, y);
                    let c2 = next_upsampled.sample_uv(u, v);

                    // Store difference + 0.5 offset to keep in [0, 1] range
                    diff_data.push([
                        (c1[0] - c2.r) * 0.5 + 0.5,
                        (c1[1] - c2.g) * 0.5 + 0.5,
                        (c1[2] - c2.b) * 0.5 + 0.5,
                        c1[3],
                    ]);
                }
            }

            levels.push(
                ImageField::from_raw(diff_data, width, height)
                    .with_wrap_mode(current.wrap_mode)
                    .with_filter_mode(current.filter_mode),
            );
        }

        // Final level is the residual (coarsest Gaussian level)
        levels.push(gaussian.levels.last().unwrap().clone());

        Self { levels }
    }

    /// Returns the number of levels in the pyramid.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Returns true if the pyramid is empty.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Returns the finest (largest) level.
    pub fn finest(&self) -> Option<&ImageField> {
        self.levels.first()
    }

    /// Returns the coarsest (smallest) level.
    pub fn coarsest(&self) -> Option<&ImageField> {
        self.levels.last()
    }

    /// Reconstructs an image from a Laplacian pyramid.
    ///
    /// Starts from the coarsest level and progressively adds detail.
    pub fn reconstruct_laplacian(&self) -> Option<ImageField> {
        if self.levels.is_empty() {
            return None;
        }

        // Start with the coarsest (residual) level
        let mut current = self.levels.last().unwrap().clone();

        // Add detail from each level
        for i in (0..self.levels.len() - 1).rev() {
            let detail = &self.levels[i];
            let (width, height) = detail.dimensions();

            // Upsample current
            let upsampled = upsample(&current);

            // Add detail
            let mut data = Vec::with_capacity((width * height) as usize);

            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;

                    let base = upsampled.sample_uv(u, v);
                    let diff = detail.get_pixel(x, y);

                    // Undo the 0.5 offset and add
                    data.push([
                        base.r + (diff[0] - 0.5) * 2.0,
                        base.g + (diff[1] - 0.5) * 2.0,
                        base.b + (diff[2] - 0.5) * 2.0,
                        diff[3],
                    ]);
                }
            }

            current = ImageField::from_raw(data, width, height)
                .with_wrap_mode(detail.wrap_mode)
                .with_filter_mode(detail.filter_mode);
        }

        Some(current)
    }
}

/// Resizes an image to a specific size using bilinear interpolation.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, resize};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let resized = resize(&img, 8, 6);
/// assert_eq!(resized.dimensions(), (8, 6));
/// ```
pub fn resize(image: &ImageField, new_width: u32, new_height: u32) -> ImageField {
    let new_width = new_width.max(1);
    let new_height = new_height.max(1);

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    // Use bilinear sampling
    let bilinear_image = ImageField {
        filter_mode: FilterMode::Bilinear,
        ..image.clone()
    };

    for y in 0..new_height {
        for x in 0..new_width {
            let u = (x as f32 + 0.5) / new_width as f32;
            let v = (y as f32 + 0.5) / new_height as f32;

            let color = bilinear_image.sample_uv(u, v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

// ============================================================================
// Normal map generation
// ============================================================================

/// Generates a normal map from a heightfield/grayscale image.
///
/// Uses Sobel operators to compute gradients, then constructs normal vectors.
/// The output is an RGB image where:
/// - R = X component of normal (mapped to 0-1)
/// - G = Y component of normal (mapped to 0-1)
/// - B = Z component of normal (mapped to 0-1)
///
/// # Arguments
/// * `heightfield` - Grayscale image where brightness = height
/// * `strength` - How pronounced the normals should be (typically 1.0-10.0)
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, heightfield_to_normal_map};
///
/// // Create a simple gradient heightfield
/// let data: Vec<_> = (0..16).map(|i| {
///     let v = (i % 4) as f32 / 3.0;
///     [v, v, v, 1.0]
/// }).collect();
/// let heightfield = ImageField::from_raw(data, 4, 4);
///
/// let normal_map = heightfield_to_normal_map(&heightfield, 2.0);
/// ```
pub fn heightfield_to_normal_map(heightfield: &ImageField, strength: f32) -> ImageField {
    let (width, height) = heightfield.dimensions();

    // Compute gradients using Sobel operators
    let dx = convolve(heightfield, &Kernel::sobel_vertical());
    let dy = convolve(heightfield, &Kernel::sobel_horizontal());

    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            // Get gradient values (use red channel for grayscale)
            let gx = dx.get_pixel(x, y)[0] * strength;
            let gy = dy.get_pixel(x, y)[0] * strength;

            // Construct normal vector: (-gx, -gy, 1) and normalize
            let len = (gx * gx + gy * gy + 1.0).sqrt();
            let nx = -gx / len;
            let ny = -gy / len;
            let nz = 1.0 / len;

            // Map from [-1, 1] to [0, 1] for storage
            let r = nx * 0.5 + 0.5;
            let g = ny * 0.5 + 0.5;
            let b = nz * 0.5 + 0.5;

            data.push([r, g, b, 1.0]);
        }
    }

    ImageField::from_raw(data, width, height)
}

/// Generates a normal map from a Field<Vec2, f32> heightfield.
///
/// This samples the field at the specified resolution and generates normals.
///
/// # Arguments
/// * `field` - A 2D scalar field representing height
/// * `config` - Bake configuration for resolution
/// * `strength` - Normal map strength
pub fn field_to_normal_map<F: Field<Vec2, f32>>(
    field: &F,
    config: &BakeConfig,
    strength: f32,
) -> ImageField {
    let ctx = EvalContext::new();

    // First bake the heightfield
    let heightfield = bake_scalar(field, config, &ctx);

    // Then convert to normal map
    heightfield_to_normal_map(&heightfield, strength)
}

// ============================================================================
// Inpainting
// ============================================================================

/// Configuration for diffusion-based inpainting operations.
///
/// Note: Inpainting takes two images (source + mask), so this is not a simple
/// Image -> Image op and does not derive Op.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Inpaint {
    /// Number of iterations for diffusion-based inpainting.
    pub iterations: u32,
    /// Diffusion rate (0.0-1.0). Higher values spread color faster.
    pub diffusion_rate: f32,
}

impl Default for Inpaint {
    fn default() -> Self {
        Self {
            iterations: 100,
            diffusion_rate: 0.25,
        }
    }
}

impl Inpaint {
    /// Creates a new inpaint configuration with the specified iterations.
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            ..Default::default()
        }
    }

    /// Sets the diffusion rate.
    pub fn with_diffusion_rate(mut self, rate: f32) -> Self {
        self.diffusion_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Applies this inpainting operation to an image with a mask.
    pub fn apply(&self, image: &ImageField, mask: &ImageField) -> ImageField {
        inpaint_diffusion(image, mask, self)
    }
}

/// Backwards-compatible type alias.
pub type InpaintConfig = Inpaint;

/// Fills masked regions using diffusion-based inpainting.
///
/// This algorithm propagates color values from the boundary of the mask inward
/// using a heat-equation-like diffusion process. Works well for small holes and
/// smooth regions.
///
/// # Arguments
///
/// * `image` - The source image to inpaint
/// * `mask` - A grayscale mask where white (1.0) indicates areas to fill
/// * `config` - Inpainting configuration
///
/// # Example
///
/// ```ignore
/// let mask = create_mask_for_scratch(&image);
/// let config = InpaintConfig::new(200);
/// let repaired = inpaint_diffusion(&image, &mask, &config);
/// ```
pub fn inpaint_diffusion(image: &ImageField, mask: &ImageField, config: &Inpaint) -> ImageField {
    let width = image.width;
    let height = image.height;

    // Create working buffer initialized with original image
    let mut result = image.data.clone();

    // Precompute mask as booleans (true = needs inpainting)
    let mask_flags: Vec<bool> = mask
        .data
        .iter()
        .map(|c| c[0] > 0.5 || c[1] > 0.5 || c[2] > 0.5)
        .collect();

    let rate = config.diffusion_rate;

    for _ in 0..config.iterations {
        let prev = result.clone();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Skip pixels that don't need inpainting
                if !mask_flags[idx] {
                    continue;
                }

                // Gather neighbors
                let mut sum = [0.0f32; 4];
                let mut count = 0.0;

                // 4-connected neighbors
                let neighbors = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];

                for (nx, ny) in neighbors {
                    if nx < width && ny < height {
                        let nidx = (ny * width + nx) as usize;
                        let neighbor = prev[nidx];
                        sum[0] += neighbor[0];
                        sum[1] += neighbor[1];
                        sum[2] += neighbor[2];
                        sum[3] += neighbor[3];
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    let avg = [
                        sum[0] / count,
                        sum[1] / count,
                        sum[2] / count,
                        sum[3] / count,
                    ];

                    // Blend toward average
                    let current = prev[idx];
                    result[idx] = [
                        current[0] + rate * (avg[0] - current[0]),
                        current[1] + rate * (avg[1] - current[1]),
                        current[2] + rate * (avg[2] - current[2]),
                        current[3] + rate * (avg[3] - current[3]),
                    ];
                }
            }
        }
    }

    ImageField {
        data: result,
        width,
        height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Configuration for PatchMatch-based inpainting.
///
/// Note: Inpainting takes two images (source + mask), so this is not a simple
/// Image -> Image op and does not derive Op.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatchMatch {
    /// Size of patches to match (must be odd).
    pub patch_size: u32,
    /// Number of pyramid levels for multi-scale processing.
    pub pyramid_levels: u32,
    /// Number of iterations per pyramid level.
    pub iterations: u32,
}

impl Default for PatchMatch {
    fn default() -> Self {
        Self {
            patch_size: 7,
            pyramid_levels: 4,
            iterations: 5,
        }
    }
}

impl PatchMatch {
    /// Creates a new PatchMatch configuration.
    pub fn new(patch_size: u32) -> Self {
        Self {
            patch_size: if patch_size % 2 == 0 {
                patch_size + 1
            } else {
                patch_size
            },
            ..Default::default()
        }
    }

    /// Sets the number of pyramid levels.
    pub fn with_pyramid_levels(mut self, levels: u32) -> Self {
        self.pyramid_levels = levels.max(1);
        self
    }

    /// Sets iterations per level.
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations.max(1);
        self
    }

    /// Applies this PatchMatch inpainting operation to an image with a mask.
    pub fn apply(&self, image: &ImageField, mask: &ImageField) -> ImageField {
        inpaint_patchmatch(image, mask, self)
    }
}

/// Backwards-compatible type alias.
pub type PatchMatchConfig = PatchMatch;

/// Fills masked regions using multi-scale PatchMatch inpainting.
///
/// This algorithm finds similar patches from known regions and copies them
/// to fill holes. Uses a coarse-to-fine approach for better coherence.
/// Good for texture synthesis and larger hole filling.
///
/// # Arguments
///
/// * `image` - The source image to inpaint
/// * `mask` - A grayscale mask where white (1.0) indicates areas to fill
/// * `config` - PatchMatch configuration
///
/// # Example
///
/// ```ignore
/// let mask = create_mask_for_object(&image);
/// let config = PatchMatchConfig::new(9).with_pyramid_levels(5);
/// let filled = inpaint_patchmatch(&image, &mask, &config);
/// ```
pub fn inpaint_patchmatch(
    image: &ImageField,
    mask: &ImageField,
    config: &PatchMatch,
) -> ImageField {
    // Build image pyramid
    let mut image_pyramid = vec![image.clone()];
    let mut mask_pyramid = vec![mask.clone()];

    for _ in 1..config.pyramid_levels {
        let last_img = image_pyramid.last().unwrap();
        let last_mask = mask_pyramid.last().unwrap();

        if last_img.width <= config.patch_size * 2 || last_img.height <= config.patch_size * 2 {
            break;
        }

        image_pyramid.push(downsample(last_img));
        mask_pyramid.push(downsample(last_mask));
    }

    // Process from coarse to fine
    let mut result = image_pyramid.last().unwrap().clone();
    let levels = image_pyramid.len();

    for level in (0..levels).rev() {
        let target_width = image_pyramid[level].width;
        let target_height = image_pyramid[level].height;

        // Upsample result if not at coarsest level
        if level < levels - 1 {
            result = upsample(&result);
            // Resize to exact target dimensions
            if result.width != target_width || result.height != target_height {
                result = resize(&result, target_width, target_height);
            }
        }

        // Copy known pixels from original
        let original = &image_pyramid[level];
        let level_mask = &mask_pyramid[level];

        for y in 0..target_height {
            for x in 0..target_width {
                let idx = (y * target_width + x) as usize;
                let mask_val = level_mask.data[idx];
                if mask_val[0] < 0.5 && mask_val[1] < 0.5 && mask_val[2] < 0.5 {
                    result.data[idx] = original.data[idx];
                }
            }
        }

        // Run PatchMatch iterations at this level
        result = patchmatch_iteration(&result, &image_pyramid[level], level_mask, config);
    }

    result
}

/// Single iteration of PatchMatch for one pyramid level.
fn patchmatch_iteration(
    current: &ImageField,
    original: &ImageField,
    mask: &ImageField,
    config: &PatchMatchConfig,
) -> ImageField {
    let width = current.width;
    let height = current.height;
    let half_patch = (config.patch_size / 2) as i32;

    // Build list of valid source patches (not in mask)
    let mut valid_sources: Vec<(u32, u32)> = Vec::new();
    for y in half_patch as u32..(height - half_patch as u32) {
        for x in half_patch as u32..(width - half_patch as u32) {
            let idx = (y * width + x) as usize;
            if mask.data[idx][0] < 0.5 {
                // Check if entire patch is valid
                let mut patch_valid = true;
                'patch_check: for py in -half_patch..=half_patch {
                    for px in -half_patch..=half_patch {
                        let check_x = (x as i32 + px) as u32;
                        let check_y = (y as i32 + py) as u32;
                        let check_idx = (check_y * width + check_x) as usize;
                        if mask.data[check_idx][0] >= 0.5 {
                            patch_valid = false;
                            break 'patch_check;
                        }
                    }
                }
                if patch_valid {
                    valid_sources.push((x, y));
                }
            }
        }
    }

    if valid_sources.is_empty() {
        // No valid sources, return current
        return current.clone();
    }

    let mut result = current.data.clone();

    // Initialize nearest neighbor field with random assignments
    let mut nnf: Vec<(u32, u32)> = Vec::with_capacity((width * height) as usize);
    let mut rng_state: u64 = 12345;

    for _ in 0..(width * height) {
        // Simple LCG for deterministic randomness
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng_state >> 32) as usize % valid_sources.len();
        nnf.push(valid_sources[idx]);
    }

    for _ in 0..config.iterations {
        // Forward pass
        for y in half_patch as u32..(height - half_patch as u32) {
            for x in half_patch as u32..(width - half_patch as u32) {
                let idx = (y * width + x) as usize;
                if mask.data[idx][0] < 0.5 {
                    continue; // Skip known pixels
                }

                let mut best_match = nnf[idx];
                let mut best_dist = patch_distance(
                    current,
                    original,
                    x,
                    y,
                    best_match.0,
                    best_match.1,
                    half_patch,
                );

                // Propagation: check neighbors
                if x > half_patch as u32 {
                    let left_idx = (y * width + x - 1) as usize;
                    let (sx, sy) = nnf[left_idx];
                    if sx + 1 < width - half_patch as u32 {
                        let dist = patch_distance(current, original, x, y, sx + 1, sy, half_patch);
                        if dist < best_dist {
                            best_dist = dist;
                            best_match = (sx + 1, sy);
                        }
                    }
                }

                if y > half_patch as u32 {
                    let up_idx = ((y - 1) * width + x) as usize;
                    let (sx, sy) = nnf[up_idx];
                    if sy + 1 < height - half_patch as u32 {
                        let dist = patch_distance(current, original, x, y, sx, sy + 1, half_patch);
                        if dist < best_dist {
                            best_dist = dist;
                            best_match = (sx, sy + 1);
                        }
                    }
                }

                // Random search
                let mut search_radius = width.max(height) as i32;
                while search_radius > 1 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let rand_idx = (rng_state >> 32) as usize % valid_sources.len();
                    let (rx, ry) = valid_sources[rand_idx];

                    let dist = patch_distance(current, original, x, y, rx, ry, half_patch);
                    if dist < best_dist {
                        best_dist = dist;
                        best_match = (rx, ry);
                    }

                    search_radius /= 2;
                }

                nnf[idx] = best_match;
            }
        }

        // Copy pixels from best matches
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if mask.data[idx][0] >= 0.5 {
                    let (sx, sy) = nnf[idx];
                    let src_idx = (sy * width + sx) as usize;
                    result[idx] = original.data[src_idx];
                }
            }
        }
    }

    ImageField {
        data: result,
        width,
        height,
        wrap_mode: current.wrap_mode,
        filter_mode: current.filter_mode,
    }
}

/// Computes squared color distance between two patches.
fn patch_distance(
    target: &ImageField,
    source: &ImageField,
    tx: u32,
    ty: u32,
    sx: u32,
    sy: u32,
    half_patch: i32,
) -> f32 {
    let width = target.width;
    let mut total = 0.0;

    for py in -half_patch..=half_patch {
        for px in -half_patch..=half_patch {
            let target_x = (tx as i32 + px) as u32;
            let target_y = (ty as i32 + py) as u32;
            let source_x = (sx as i32 + px) as u32;
            let source_y = (sy as i32 + py) as u32;

            let tidx = (target_y * width + target_x) as usize;
            let sidx = (source_y * width + source_x) as usize;

            let tc = target.data[tidx];
            let sc = source.data[sidx];

            let dr = tc[0] - sc[0];
            let dg = tc[1] - sc[1];
            let db = tc[2] - sc[2];

            total += dr * dr + dg * dg + db * db;
        }
    }

    total
}

/// Creates a simple mask from an image based on a color key.
///
/// Pixels close to the key color (within tolerance) are marked for inpainting.
///
/// # Arguments
///
/// * `image` - The source image
/// * `key_color` - The color to key out (e.g., magenta for marked regions)
/// * `tolerance` - Color distance threshold (0.0-1.0)
pub fn create_color_key_mask(image: &ImageField, key_color: Rgba, tolerance: f32) -> ImageField {
    let tol_sq = tolerance * tolerance * 3.0; // Scale for RGB distance

    let data: Vec<[f32; 4]> = image
        .data
        .iter()
        .map(|c| {
            let dr = c[0] - key_color.r;
            let dg = c[1] - key_color.g;
            let db = c[2] - key_color.b;
            let dist_sq = dr * dr + dg * dg + db * db;

            if dist_sq <= tol_sq {
                [1.0, 1.0, 1.0, 1.0] // Mark for inpainting
            } else {
                [0.0, 0.0, 0.0, 1.0] // Keep original
            }
        })
        .collect();

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Dilates a mask by the specified radius.
///
/// Useful for expanding inpainting regions to cover edges.
pub fn dilate_mask(mask: &ImageField, radius: u32) -> ImageField {
    let width = mask.width;
    let height = mask.height;
    let r = radius as i32;

    let data: Vec<[f32; 4]> = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                // Check if any pixel within radius is white
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy <= r * r {
                            let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                            let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                            let idx = (ny * width + nx) as usize;
                            if mask.data[idx][0] > 0.5 {
                                return [1.0, 1.0, 1.0, 1.0];
                            }
                        }
                    }
                }
                [0.0, 0.0, 0.0, 1.0]
            })
        })
        .collect();

    ImageField {
        data,
        width,
        height,
        wrap_mode: mask.wrap_mode,
        filter_mode: mask.filter_mode,
    }
}

// ============================================================================
// Glitch Effects
// ============================================================================

/// Pixel sorting configuration.
#[derive(Debug, Clone)]
pub struct PixelSort {
    /// Sort direction.
    pub direction: SortDirection,
    /// What to sort by.
    pub sort_by: SortBy,
    /// Threshold for starting a sort span (0-1).
    pub threshold_min: f32,
    /// Threshold for ending a sort span (0-1).
    pub threshold_max: f32,
    /// Reverse sort order.
    pub reverse: bool,
}

/// Direction to sort pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    /// Sort along rows (left to right).
    #[default]
    Horizontal,
    /// Sort along columns (top to bottom).
    Vertical,
}

/// Metric to sort pixels by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortBy {
    /// Sort by brightness (luminance).
    #[default]
    Brightness,
    /// Sort by hue.
    Hue,
    /// Sort by saturation.
    Saturation,
    /// Sort by red channel.
    Red,
    /// Sort by green channel.
    Green,
    /// Sort by blue channel.
    Blue,
}

impl Default for PixelSort {
    fn default() -> Self {
        Self {
            direction: SortDirection::Horizontal,
            sort_by: SortBy::Brightness,
            threshold_min: 0.25,
            threshold_max: 0.75,
            reverse: false,
        }
    }
}

/// Sorts pixels in the image based on brightness or other metrics.
///
/// Creates a distinctive glitch art aesthetic by sorting spans of pixels.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_image::{pixel_sort, PixelSort, SortBy};
///
/// let config = PixelSort {
///     sort_by: SortBy::Brightness,
///     threshold_min: 0.2,
///     threshold_max: 0.8,
///     ..Default::default()
/// };
/// let sorted = pixel_sort(&image, &config);
/// ```
pub fn pixel_sort(image: &ImageField, config: &PixelSort) -> ImageField {
    let width = image.width as usize;
    let height = image.height as usize;
    let mut data = image.data.clone();

    let get_sort_value = |pixel: &[f32; 4]| -> f32 {
        match config.sort_by {
            SortBy::Brightness => 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2],
            SortBy::Red => pixel[0],
            SortBy::Green => pixel[1],
            SortBy::Blue => pixel[2],
            SortBy::Hue => {
                let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                if (max - min).abs() < 1e-6 {
                    0.0
                } else if (max - r).abs() < 1e-6 {
                    ((g - b) / (max - min)).rem_euclid(6.0) / 6.0
                } else if (max - g).abs() < 1e-6 {
                    ((b - r) / (max - min) + 2.0) / 6.0
                } else {
                    ((r - g) / (max - min) + 4.0) / 6.0
                }
            }
            SortBy::Saturation => {
                let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                if max < 1e-6 { 0.0 } else { (max - min) / max }
            }
        }
    };

    match config.direction {
        SortDirection::Horizontal => {
            for y in 0..height {
                let row_start = y * width;
                let row = &mut data[row_start..row_start + width];

                // Find spans to sort
                let mut spans: Vec<(usize, usize)> = Vec::new();
                let mut span_start: Option<usize> = None;

                for (x, pixel) in row.iter().enumerate() {
                    let value = get_sort_value(pixel);
                    let in_range = value >= config.threshold_min && value <= config.threshold_max;

                    match (span_start, in_range) {
                        (None, true) => span_start = Some(x),
                        (Some(start), false) => {
                            if x > start + 1 {
                                spans.push((start, x));
                            }
                            span_start = None;
                        }
                        _ => {}
                    }
                }
                if let Some(start) = span_start {
                    if width > start + 1 {
                        spans.push((start, width));
                    }
                }

                // Sort each span
                for (start, end) in spans {
                    let span = &mut row[start..end];
                    span.sort_by(|a, b| {
                        let va = get_sort_value(a);
                        let vb = get_sort_value(b);
                        if config.reverse {
                            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    });
                }
            }
        }
        SortDirection::Vertical => {
            for x in 0..width {
                // Extract column
                let mut column: Vec<[f32; 4]> = (0..height).map(|y| data[y * width + x]).collect();

                // Find spans to sort
                let mut spans: Vec<(usize, usize)> = Vec::new();
                let mut span_start: Option<usize> = None;

                for (y, pixel) in column.iter().enumerate() {
                    let value = get_sort_value(pixel);
                    let in_range = value >= config.threshold_min && value <= config.threshold_max;

                    match (span_start, in_range) {
                        (None, true) => span_start = Some(y),
                        (Some(start), false) => {
                            if y > start + 1 {
                                spans.push((start, y));
                            }
                            span_start = None;
                        }
                        _ => {}
                    }
                }
                if let Some(start) = span_start {
                    if height > start + 1 {
                        spans.push((start, height));
                    }
                }

                // Sort each span
                for (start, end) in spans {
                    let span = &mut column[start..end];
                    span.sort_by(|a, b| {
                        let va = get_sort_value(a);
                        let vb = get_sort_value(b);
                        if config.reverse {
                            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    });
                }

                // Write column back
                for (y, pixel) in column.into_iter().enumerate() {
                    data[y * width + x] = pixel;
                }
            }
        }
    }

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// RGB channel shift configuration.
#[derive(Debug, Clone)]
pub struct RgbShift {
    /// Red channel offset (x, y) in pixels.
    pub red_offset: (i32, i32),
    /// Green channel offset (x, y) in pixels.
    pub green_offset: (i32, i32),
    /// Blue channel offset (x, y) in pixels.
    pub blue_offset: (i32, i32),
}

impl Default for RgbShift {
    fn default() -> Self {
        Self {
            red_offset: (-5, 0),
            green_offset: (0, 0),
            blue_offset: (5, 0),
        }
    }
}

/// Shifts RGB channels independently for a glitch effect.
///
/// Creates color fringing similar to analog video distortion.
pub fn rgb_shift(image: &ImageField, config: &RgbShift) -> ImageField {
    let width = image.width as i32;
    let height = image.height as i32;

    let sample = |x: i32, y: i32| -> [f32; 4] {
        let wx = x.rem_euclid(width) as usize;
        let wy = y.rem_euclid(height) as usize;
        image.data[wy * width as usize + wx]
    };

    let mut data = Vec::with_capacity(image.data.len());

    for y in 0..height {
        for x in 0..width {
            let r = sample(x + config.red_offset.0, y + config.red_offset.1)[0];
            let g = sample(x + config.green_offset.0, y + config.green_offset.1)[1];
            let b = sample(x + config.blue_offset.0, y + config.blue_offset.1)[2];
            let a = image.data[(y * width + x) as usize][3];
            data.push([r, g, b, a]);
        }
    }

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Scan lines configuration.
#[derive(Debug, Clone)]
pub struct ScanLines {
    /// Gap between scan lines in pixels.
    pub gap: u32,
    /// Thickness of dark lines in pixels.
    pub thickness: u32,
    /// Darkness of the lines (0 = transparent, 1 = black).
    pub intensity: f32,
    /// Vertical offset.
    pub offset: u32,
}

impl Default for ScanLines {
    fn default() -> Self {
        Self {
            gap: 2,
            thickness: 1,
            intensity: 0.5,
            offset: 0,
        }
    }
}

/// Adds CRT-style scan lines to an image.
pub fn scan_lines(image: &ImageField, config: &ScanLines) -> ImageField {
    let width = image.width as usize;
    let height = image.height as usize;
    let period = config.gap + config.thickness;

    let mut data = image.data.clone();

    for y in 0..height {
        let line_pos = ((y as u32 + config.offset) % period) as u32;
        if line_pos < config.thickness {
            let factor = 1.0 - config.intensity;
            for x in 0..width {
                let pixel = &mut data[y * width + x];
                pixel[0] *= factor;
                pixel[1] *= factor;
                pixel[2] *= factor;
            }
        }
    }

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Adds random noise/static to an image.
///
/// # Arguments
/// * `image` - Input image
/// * `intensity` - Noise intensity (0-1)
/// * `seed` - Random seed for reproducibility
pub fn static_noise(image: &ImageField, intensity: f32, seed: u32) -> ImageField {
    let mut data = image.data.clone();
    let intensity = intensity.clamp(0.0, 1.0);

    for (i, pixel) in data.iter_mut().enumerate() {
        // Simple hash for deterministic noise
        let hash = simple_hash(seed.wrapping_add(i as u32));
        let noise = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0; // -1 to 1

        pixel[0] = (pixel[0] + noise * intensity).clamp(0.0, 1.0);
        pixel[1] = (pixel[1] + noise * intensity).clamp(0.0, 1.0);
        pixel[2] = (pixel[2] + noise * intensity).clamp(0.0, 1.0);
    }

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// VHS tracking distortion configuration.
#[derive(Debug, Clone)]
pub struct VhsTracking {
    /// Maximum horizontal displacement in pixels.
    pub displacement: f32,
    /// Frequency of displacement bands.
    pub frequency: f32,
    /// Color bleeding amount (0-1).
    pub color_bleed: f32,
    /// Vertical scroll offset.
    pub scroll: f32,
    /// Random seed.
    pub seed: u32,
}

impl Default for VhsTracking {
    fn default() -> Self {
        Self {
            displacement: 10.0,
            frequency: 0.1,
            color_bleed: 0.3,
            scroll: 0.0,
            seed: 42,
        }
    }
}

/// Applies VHS tracking distortion effect.
///
/// Simulates analog video tracking errors with horizontal displacement
/// bands and color bleeding.
pub fn vhs_tracking(image: &ImageField, config: &VhsTracking) -> ImageField {
    let width = image.width as i32;
    let height = image.height as i32;

    let sample = |x: i32, y: i32| -> [f32; 4] {
        let wx = x.clamp(0, width - 1) as usize;
        let wy = y.clamp(0, height - 1) as usize;
        image.data[wy * width as usize + wx]
    };

    let mut data = Vec::with_capacity(image.data.len());

    for y in 0..height {
        // Calculate displacement for this row
        let y_norm = (y as f32 + config.scroll) / height as f32;
        let hash = simple_hash(config.seed.wrapping_add((y_norm * 1000.0) as u32));
        let noise = (hash as f32 / u32::MAX as f32) * 2.0 - 1.0;

        let wave = (y_norm * config.frequency * std::f32::consts::TAU).sin();
        let displacement = ((wave + noise * 0.5) * config.displacement) as i32;

        for x in 0..width {
            let base = sample(x + displacement, y);

            // Color bleeding - offset red channel slightly more
            let bleed_offset = (config.color_bleed * 3.0) as i32;
            let r = if config.color_bleed > 0.0 {
                let left = sample(x + displacement - bleed_offset, y)[0];
                base[0] * (1.0 - config.color_bleed) + left * config.color_bleed
            } else {
                base[0]
            };

            data.push([r, base[1], base[2], base[3]]);
        }
    }

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Simple hash for deterministic noise.
fn simple_hash(x: u32) -> u32 {
    let mut h = x;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

// =============================================================================
// JPEG Artifacts (DCT-based compression artifacts)
// =============================================================================

/// Configuration for JPEG artifact effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JpegArtifacts {
    /// Quality level (1-100). Lower = more artifacts.
    pub quality: u8,
    /// Whether to add color subsampling artifacts (4:2:0 chroma).
    pub chroma_subsampling: bool,
}

impl Default for JpegArtifacts {
    fn default() -> Self {
        Self {
            quality: 10,
            chroma_subsampling: true,
        }
    }
}

impl JpegArtifacts {
    /// Creates JPEG artifacts with the given quality.
    pub fn new(quality: u8) -> Self {
        Self {
            quality: quality.clamp(1, 100),
            ..Default::default()
        }
    }

    /// Enables or disables chroma subsampling.
    pub fn with_chroma_subsampling(mut self, enabled: bool) -> Self {
        self.chroma_subsampling = enabled;
        self
    }
}

/// Standard JPEG luminance quantization table.
const JPEG_LUMA_QUANT: [f32; 64] = [
    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0, 12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
    14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0, 14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0, 24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0,
    92.0, 49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0, 72.0, 92.0, 95.0, 98.0, 112.0, 100.0,
    103.0, 99.0,
];

/// Standard JPEG chrominance quantization table.
const JPEG_CHROMA_QUANT: [f32; 64] = [
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0, 18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0, 47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

/// Applies 2D DCT to an 8x8 block.
fn dct_8x8(block: &[f32; 64]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    let pi = std::f32::consts::PI;

    for v in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };

            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    let cos_u = ((2 * x + 1) as f32 * u as f32 * pi / 16.0).cos();
                    let cos_v = ((2 * y + 1) as f32 * v as f32 * pi / 16.0).cos();
                    sum += block[y * 8 + x] * cos_u * cos_v;
                }
            }
            result[v * 8 + u] = 0.25 * cu * cv * sum;
        }
    }
    result
}

/// Applies inverse 2D DCT to an 8x8 block.
fn idct_8x8(block: &[f32; 64]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    let pi = std::f32::consts::PI;

    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0;
            for v in 0..8 {
                for u in 0..8 {
                    let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                    let cos_u = ((2 * x + 1) as f32 * u as f32 * pi / 16.0).cos();
                    let cos_v = ((2 * y + 1) as f32 * v as f32 * pi / 16.0).cos();
                    sum += cu * cv * block[v * 8 + u] * cos_u * cos_v;
                }
            }
            result[y * 8 + x] = 0.25 * sum;
        }
    }
    result
}

/// Quantizes DCT coefficients using the given quantization table and quality.
fn quantize_block(block: &[f32; 64], quant_table: &[f32; 64], quality: u8) -> [f32; 64] {
    let scale = if quality < 50 {
        5000.0 / quality as f32
    } else {
        200.0 - 2.0 * quality as f32
    } / 100.0;

    let mut result = [0.0f32; 64];
    for i in 0..64 {
        let q = (quant_table[i] * scale).max(1.0);
        result[i] = (block[i] / q).round() * q;
    }
    result
}

/// Applies JPEG-like compression artifacts to an image.
///
/// Simulates JPEG compression by:
/// 1. Converting to YCbCr colorspace
/// 2. Splitting into 8x8 blocks
/// 3. Applying DCT to each block
/// 4. Quantizing coefficients (this creates the artifacts)
/// 5. Applying inverse DCT
/// 6. Converting back to RGB
///
/// Lower quality values create more visible block artifacts and color banding.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, JpegArtifacts, jpeg_artifacts};
///
/// let image = ImageField::solid_sized(64, 64, [0.5, 0.3, 0.7, 1.0]);
/// let config = JpegArtifacts::new(5); // Very low quality = heavy artifacts
/// let result = jpeg_artifacts(&image, &config);
/// ```
pub fn jpeg_artifacts(image: &ImageField, config: &JpegArtifacts) -> ImageField {
    let (width, height) = image.dimensions();

    // Convert to YCbCr
    let mut y_channel = vec![0.0f32; (width * height) as usize];
    let mut cb_channel = vec![0.0f32; (width * height) as usize];
    let mut cr_channel = vec![0.0f32; (width * height) as usize];

    for i in 0..(width * height) as usize {
        let pixel = image.data[i];
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        // RGB to YCbCr
        y_channel[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb_channel[i] = -0.169 * r - 0.331 * g + 0.500 * b + 0.5;
        cr_channel[i] = 0.500 * r - 0.419 * g - 0.081 * b + 0.5;
    }

    // Process in 8x8 blocks
    let process_channel = |channel: &mut [f32], quant_table: &[f32; 64]| {
        for by in (0..height).step_by(8) {
            for bx in (0..width).step_by(8) {
                // Extract block
                let mut block = [0.0f32; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        let px = (bx + x).min(width - 1) as usize;
                        let py = (by + y).min(height - 1) as usize;
                        block[y as usize * 8 + x as usize] = channel[py * width as usize + px];
                    }
                }

                // Level shift (center around 0)
                for v in &mut block {
                    *v -= 0.5;
                }

                // DCT -> Quantize -> IDCT
                let dct = dct_8x8(&block);
                let quantized = quantize_block(&dct, quant_table, config.quality);
                let reconstructed = idct_8x8(&quantized);

                // Level shift back and write block
                for y in 0..8 {
                    for x in 0..8 {
                        let px = (bx + x) as usize;
                        let py = (by + y) as usize;
                        if px < width as usize && py < height as usize {
                            channel[py * width as usize + px] =
                                (reconstructed[y as usize * 8 + x as usize] + 0.5).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    };

    // Process Y with luminance table
    process_channel(&mut y_channel, &JPEG_LUMA_QUANT);

    // Process Cb/Cr with chrominance table
    process_channel(&mut cb_channel, &JPEG_CHROMA_QUANT);
    process_channel(&mut cr_channel, &JPEG_CHROMA_QUANT);

    // Optional: Simulate 4:2:0 chroma subsampling
    if config.chroma_subsampling {
        // Downsample and upsample chroma (nearest neighbor for blocky look)
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                let idx00 = (y * width + x) as usize;
                let idx01 = (y * width + (x + 1).min(width - 1)) as usize;
                let idx10 = ((y + 1).min(height - 1) * width + x) as usize;
                let idx11 = ((y + 1).min(height - 1) * width + (x + 1).min(width - 1)) as usize;

                // Average the 2x2 block
                let cb_avg =
                    (cb_channel[idx00] + cb_channel[idx01] + cb_channel[idx10] + cb_channel[idx11])
                        / 4.0;
                let cr_avg =
                    (cr_channel[idx00] + cr_channel[idx01] + cr_channel[idx10] + cr_channel[idx11])
                        / 4.0;

                // Write back to all 4 pixels (blocky upsampling)
                cb_channel[idx00] = cb_avg;
                cb_channel[idx01] = cb_avg;
                cb_channel[idx10] = cb_avg;
                cb_channel[idx11] = cb_avg;
                cr_channel[idx00] = cr_avg;
                cr_channel[idx01] = cr_avg;
                cr_channel[idx10] = cr_avg;
                cr_channel[idx11] = cr_avg;
            }
        }
    }

    // Convert back to RGB
    let mut data = Vec::with_capacity((width * height) as usize);
    for i in 0..(width * height) as usize {
        let y = y_channel[i];
        let cb = cb_channel[i] - 0.5;
        let cr = cr_channel[i] - 0.5;

        // YCbCr to RGB
        let r = y + 1.402 * cr;
        let g = y - 0.344 * cb - 0.714 * cr;
        let b = y + 1.772 * cb;

        data.push([
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            image.data[i][3],
        ]);
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

// =============================================================================
// Bit Manipulation Effects
// =============================================================================

/// Configuration for bit manipulation effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BitManip {
    /// Operation to perform.
    pub operation: BitOperation,
    /// Value to use for the operation (interpreted as u8 for each channel).
    pub value: u8,
    /// Which channels to affect (R, G, B, A).
    pub channels: [bool; 4],
}

/// Bit manipulation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BitOperation {
    /// XOR each byte with the value.
    Xor,
    /// AND each byte with the value.
    And,
    /// OR each byte with the value.
    Or,
    /// NOT (invert bits, value is ignored).
    Not,
    /// Shift bits left by value amount.
    ShiftLeft,
    /// Shift bits right by value amount.
    ShiftRight,
}

impl Default for BitManip {
    fn default() -> Self {
        Self {
            operation: BitOperation::Xor,
            value: 0x55,                         // Checkerboard pattern
            channels: [true, true, true, false], // RGB only
        }
    }
}

impl BitManip {
    /// Creates a bit manipulation config with the given operation and value.
    pub fn new(operation: BitOperation, value: u8) -> Self {
        Self {
            operation,
            value,
            ..Default::default()
        }
    }

    /// Sets which channels to affect.
    pub fn with_channels(mut self, r: bool, g: bool, b: bool, a: bool) -> Self {
        self.channels = [r, g, b, a];
        self
    }
}

/// Applies bit manipulation to image pixel data.
///
/// Treats pixel values as 8-bit integers and applies bitwise operations,
/// creating digital glitch effects.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, BitManip, BitOperation, bit_manip};
///
/// let image = ImageField::solid_sized(64, 64, [0.5, 0.3, 0.7, 1.0]);
///
/// // XOR with checkerboard pattern
/// let glitched = bit_manip(&image, &BitManip::new(BitOperation::Xor, 0xAA));
///
/// // AND to create color bands
/// let banded = bit_manip(&image, &BitManip::new(BitOperation::And, 0xF0));
/// ```
pub fn bit_manip(image: &ImageField, config: &BitManip) -> ImageField {
    let operation = config.operation;
    let value = config.value;
    let channels = config.channels;

    map_pixels_fn(image, move |pixel| {
        let apply_op = |v: f32| -> f32 {
            let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
            let result = match operation {
                BitOperation::Xor => byte ^ value,
                BitOperation::And => byte & value,
                BitOperation::Or => byte | value,
                BitOperation::Not => !byte,
                BitOperation::ShiftLeft => byte.wrapping_shl(value as u32),
                BitOperation::ShiftRight => byte.wrapping_shr(value as u32),
            };
            result as f32 / 255.0
        };

        let mut result = pixel;
        for (i, &should_apply) in channels.iter().enumerate() {
            if should_apply {
                result[i] = apply_op(pixel[i]);
            }
        }
        result
    })
}

/// Corrupts random bytes in the image data.
///
/// Simulates file corruption by randomly modifying pixel values.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ByteCorrupt {
    /// Probability of corrupting each byte (0.0-1.0).
    pub probability: f32,
    /// Random seed for reproducible corruption.
    pub seed: u32,
    /// Corruption mode.
    pub mode: CorruptMode,
}

/// Byte corruption modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CorruptMode {
    /// Replace with random value.
    Random,
    /// Swap with adjacent byte.
    Swap,
    /// Zero out the byte.
    Zero,
    /// Set to maximum value.
    Max,
}

impl Default for ByteCorrupt {
    fn default() -> Self {
        Self {
            probability: 0.01,
            seed: 42,
            mode: CorruptMode::Random,
        }
    }
}

impl ByteCorrupt {
    /// Creates a byte corruption config with the given probability.
    pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Sets the corruption mode.
    pub fn with_mode(mut self, mode: CorruptMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }
}

/// Corrupts random bytes in an image for glitch effects.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, ByteCorrupt, CorruptMode, byte_corrupt};
///
/// let image = ImageField::solid_sized(64, 64, [0.5, 0.3, 0.7, 1.0]);
/// let config = ByteCorrupt::new(0.05).with_mode(CorruptMode::Random);
/// let corrupted = byte_corrupt(&image, &config);
/// ```
pub fn byte_corrupt(image: &ImageField, config: &ByteCorrupt) -> ImageField {
    let mut rng_state = config.seed;
    let next_random = |state: &mut u32| -> f32 {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (*state as f32) / (u32::MAX as f32)
    };

    let mut data = image.data.clone();

    for pixel in &mut data {
        for channel in pixel.iter_mut() {
            if next_random(&mut rng_state) < config.probability {
                let byte = (*channel * 255.0) as u8;
                let corrupted = match config.mode {
                    CorruptMode::Random => (next_random(&mut rng_state) * 255.0) as u8,
                    CorruptMode::Swap => byte.rotate_left(4), // Swap nibbles
                    CorruptMode::Zero => 0,
                    CorruptMode::Max => 255,
                };
                *channel = corrupted as f32 / 255.0;
            }
        }
    }

    ImageField::from_raw(data, image.width, image.height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

// =============================================================================
// Color and Transform Operations
// =============================================================================

/// Apply a 4x4 color matrix transform to an image.
///
/// The matrix transforms RGBA values: `[r', g', b', a'] = matrix * [r, g, b, a]`.
/// This can be used for color correction, channel mixing, sepia tones, etc.
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, color_matrix};
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
    let data: Vec<[f32; 4]> = image
        .data
        .iter()
        .map(|pixel| {
            let v = Vec4::from_array(*pixel);
            let transformed = matrix * v;
            transformed.to_array()
        })
        .collect();

    ImageField {
        data,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
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
/// use rhizome_resin_image::{ImageField, TransformConfig, transform_image};
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
/// use rhizome_resin_image::{ImageField, Lut1D, apply_lut_1d};
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
/// use rhizome_resin_image::{ImageField, Lut3D, apply_lut_3d};
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

/// Registers all image operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of image ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<BakeConfig>("resin::BakeConfig");
    registry.register_type::<AnimationConfig>("resin::AnimationConfig");
    registry.register_type::<ChromaticAberration>("resin::ChromaticAberration");
    registry.register_type::<Levels>("resin::Levels");
    registry.register_type::<LensDistortion>("resin::LensDistortion");
    registry.register_type::<WaveDistortion>("resin::WaveDistortion");
}

// ============================================================================
// Expression-Based Primitives
// ============================================================================

/// A typed expression AST for UV coordinate remapping (Vec2 → Vec2).
///
/// This is the expression language for the `remap_uv` primitive. Each variant
/// represents an operation that transforms UV coordinates.
///
/// # Design
///
/// Unlike raw closures, `UvExpr` is:
/// - **Serializable** - Save/load effect pipelines
/// - **Interpretable** - Direct CPU evaluation
/// - **Inspectable** - Debug and optimize transforms
/// - **Future JIT/GPU** - Will compile to Cranelift/WGSL when dew-linalg is ready
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, UvExpr, remap_uv};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Wave distortion: offset U by sin(V * frequency) * amplitude
/// let wave = UvExpr::Add(
///     Box::new(UvExpr::Uv),
///     Box::new(UvExpr::Vec2 {
///         x: Box::new(UvExpr::Mul(
///             Box::new(UvExpr::Constant(0.05)),
///             Box::new(UvExpr::Sin(Box::new(UvExpr::Mul(
///                 Box::new(UvExpr::V),
///                 Box::new(UvExpr::Constant(6.28 * 4.0)),
///             )))),
///         )),
///         y: Box::new(UvExpr::Constant(0.0)),
///     }),
/// );
///
/// let result = remap_uv(&image, &wave);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UvExpr {
    // === Coordinates ===
    /// The input UV coordinate as Vec2.
    Uv,
    /// Just the U coordinate (x).
    U,
    /// Just the V coordinate (y).
    V,

    // === Constructors ===
    /// Construct a Vec2 from two scalar expressions.
    Vec2 { x: Box<UvExpr>, y: Box<UvExpr> },

    // === Literals ===
    /// A constant scalar value.
    Constant(f32),
    /// A constant Vec2 value.
    Constant2(f32, f32),

    // === Vec2 operations ===
    /// Component-wise addition of two Vec2 expressions.
    Add(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise subtraction.
    Sub(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise multiplication.
    Mul(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise division.
    Div(Box<UvExpr>, Box<UvExpr>),
    /// Negate (flip sign).
    Neg(Box<UvExpr>),

    // === Scalar math functions (applied component-wise or to scalars) ===
    /// Sine.
    Sin(Box<UvExpr>),
    /// Cosine.
    Cos(Box<UvExpr>),
    /// Absolute value.
    Abs(Box<UvExpr>),
    /// Floor.
    Floor(Box<UvExpr>),
    /// Fractional part.
    Fract(Box<UvExpr>),
    /// Square root.
    Sqrt(Box<UvExpr>),
    /// Power.
    Pow(Box<UvExpr>, Box<UvExpr>),
    /// Minimum.
    Min(Box<UvExpr>, Box<UvExpr>),
    /// Maximum.
    Max(Box<UvExpr>, Box<UvExpr>),
    /// Clamp to range.
    Clamp {
        value: Box<UvExpr>,
        min: Box<UvExpr>,
        max: Box<UvExpr>,
    },
    /// Linear interpolation.
    Lerp {
        a: Box<UvExpr>,
        b: Box<UvExpr>,
        t: Box<UvExpr>,
    },

    // === Vec2-specific operations ===
    /// Length of the vector.
    Length(Box<UvExpr>),
    /// Normalize to unit vector.
    Normalize(Box<UvExpr>),
    /// Dot product.
    Dot(Box<UvExpr>, Box<UvExpr>),
    /// Distance between two points.
    Distance(Box<UvExpr>, Box<UvExpr>),

    // === Common UV transforms (sugar for complex expressions) ===
    /// Rotate around a center point by angle (radians).
    Rotate {
        center: Box<UvExpr>,
        angle: Box<UvExpr>,
    },
    /// Scale around a center point.
    Scale {
        center: Box<UvExpr>,
        scale: Box<UvExpr>,
    },
}

impl UvExpr {
    /// Evaluate the expression at the given UV coordinate.
    ///
    /// Returns the transformed UV as (u', v').
    pub fn eval(&self, u: f32, v: f32) -> (f32, f32) {
        match self {
            // Coordinates
            Self::Uv => (u, v),
            Self::U => (u, u), // scalar → vec2 broadcast
            Self::V => (v, v),

            // Constructors
            Self::Vec2 { x, y } => {
                let (xu, _) = x.eval(u, v);
                let (yu, _) = y.eval(u, v);
                (xu, yu)
            }

            // Literals
            Self::Constant(c) => (*c, *c),
            Self::Constant2(x, y) => (*x, *y),

            // Vec2 operations
            Self::Add(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au + bu, av + bv)
            }
            Self::Sub(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au - bu, av - bv)
            }
            Self::Mul(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au * bu, av * bv)
            }
            Self::Div(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au / bu, av / bv)
            }
            Self::Neg(a) => {
                let (au, av) = a.eval(u, v);
                (-au, -av)
            }

            // Scalar math
            Self::Sin(a) => {
                let (au, av) = a.eval(u, v);
                (au.sin(), av.sin())
            }
            Self::Cos(a) => {
                let (au, av) = a.eval(u, v);
                (au.cos(), av.cos())
            }
            Self::Abs(a) => {
                let (au, av) = a.eval(u, v);
                (au.abs(), av.abs())
            }
            Self::Floor(a) => {
                let (au, av) = a.eval(u, v);
                (au.floor(), av.floor())
            }
            Self::Fract(a) => {
                let (au, av) = a.eval(u, v);
                (au.fract(), av.fract())
            }
            Self::Sqrt(a) => {
                let (au, av) = a.eval(u, v);
                (au.sqrt(), av.sqrt())
            }
            Self::Pow(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.powf(bu), av.powf(bv))
            }
            Self::Min(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.min(bu), av.min(bv))
            }
            Self::Max(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.max(bu), av.max(bv))
            }
            Self::Clamp { value, min, max } => {
                let (vu, vv) = value.eval(u, v);
                let (minu, minv) = min.eval(u, v);
                let (maxu, maxv) = max.eval(u, v);
                (vu.clamp(minu, maxu), vv.clamp(minv, maxv))
            }
            Self::Lerp { a, b, t } => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let (tu, tv) = t.eval(u, v);
                (au + (bu - au) * tu, av + (bv - av) * tv)
            }

            // Vec2-specific
            Self::Length(a) => {
                let (au, av) = a.eval(u, v);
                let len = (au * au + av * av).sqrt();
                (len, len)
            }
            Self::Normalize(a) => {
                let (au, av) = a.eval(u, v);
                let len = (au * au + av * av).sqrt();
                if len > 0.0 {
                    (au / len, av / len)
                } else {
                    (0.0, 0.0)
                }
            }
            Self::Dot(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let d = au * bu + av * bv;
                (d, d)
            }
            Self::Distance(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let dx = au - bu;
                let dy = av - bv;
                let d = (dx * dx + dy * dy).sqrt();
                (d, d)
            }

            // Common transforms
            Self::Rotate { center, angle } => {
                let (cx, cy) = center.eval(u, v);
                let (angle_val, _) = angle.eval(u, v);
                let dx = u - cx;
                let dy = v - cy;
                let cos_a = angle_val.cos();
                let sin_a = angle_val.sin();
                (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            }
            Self::Scale { center, scale } => {
                let (cx, cy) = center.eval(u, v);
                let (sx, sy) = scale.eval(u, v);
                (cx + (u - cx) * sx, cy + (v - cy) * sy)
            }
        }
    }

    /// Creates an identity transform (returns UV unchanged).
    pub fn identity() -> Self {
        Self::Uv
    }

    /// Creates a translation transform.
    pub fn translate(offset_x: f32, offset_y: f32) -> Self {
        Self::Add(
            Box::new(Self::Uv),
            Box::new(Self::Constant2(offset_x, offset_y)),
        )
    }

    /// Creates a scale transform around the center (0.5, 0.5).
    pub fn scale_centered(scale_x: f32, scale_y: f32) -> Self {
        Self::Scale {
            center: Box::new(Self::Constant2(0.5, 0.5)),
            scale: Box::new(Self::Constant2(scale_x, scale_y)),
        }
    }

    /// Creates a rotation transform around the center (0.5, 0.5).
    pub fn rotate_centered(angle: f32) -> Self {
        Self::Rotate {
            center: Box::new(Self::Constant2(0.5, 0.5)),
            angle: Box::new(Self::Constant(angle)),
        }
    }

    /// Converts this expression to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `uv` Vec2 variable and returns Vec2.
    /// Use with `dew-linalg` for evaluation or compilation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rhizome_dew_linalg::{Value, eval, linalg_registry};
    /// use std::collections::HashMap;
    ///
    /// let expr = UvExpr::translate(0.1, 0.0);
    /// let ast = expr.to_dew_ast();
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("uv".into(), Value::Vec2([0.5, 0.5]));
    ///
    /// let result = eval(&ast, &vars, &linalg_registry()).unwrap();
    /// // result = Value::Vec2([0.6, 0.5])
    /// ```
    #[cfg(feature = "dew")]
    pub fn to_dew_ast(&self) -> rhizome_dew_core::Ast {
        use rhizome_dew_core::{Ast, BinOp, UnaryOp};

        match self {
            // Coordinates - uv is a Vec2 variable
            Self::Uv => Ast::Var("uv".into()),
            // U and V need component extraction - use helper functions
            Self::U => Ast::Call("x".into(), vec![Ast::Var("uv".into())]),
            Self::V => Ast::Call("y".into(), vec![Ast::Var("uv".into())]),

            // Constructors
            Self::Vec2 { x, y } => Ast::Call("vec2".into(), vec![x.to_dew_ast(), y.to_dew_ast()]),

            // Literals
            Self::Constant(c) => Ast::Num(*c as f64),
            Self::Constant2(x, y) => Ast::Call(
                "vec2".into(),
                vec![Ast::Num(*x as f64), Ast::Num(*y as f64)],
            ),

            // Binary operations
            Self::Add(a, b) => Ast::BinOp(
                BinOp::Add,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Sub(a, b) => Ast::BinOp(
                BinOp::Sub,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mul(a, b) => Ast::BinOp(
                BinOp::Mul,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Div(a, b) => Ast::BinOp(
                BinOp::Div,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            // Math functions
            Self::Sin(a) => Ast::Call("sin".into(), vec![a.to_dew_ast()]),
            Self::Cos(a) => Ast::Call("cos".into(), vec![a.to_dew_ast()]),
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Fract(a) => Ast::Call("fract".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Min(a, b) => Ast::Call("min".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Max(a, b) => Ast::Call("max".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Clamp { value, min, max } => Ast::Call(
                "clamp".into(),
                vec![value.to_dew_ast(), min.to_dew_ast(), max.to_dew_ast()],
            ),
            Self::Lerp { a, b, t } => Ast::Call(
                "lerp".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), t.to_dew_ast()],
            ),

            // Vec2-specific operations
            Self::Length(a) => Ast::Call("length".into(), vec![a.to_dew_ast()]),
            Self::Normalize(a) => Ast::Call("normalize".into(), vec![a.to_dew_ast()]),
            Self::Dot(a, b) => Ast::Call("dot".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Distance(a, b) => {
                Ast::Call("distance".into(), vec![a.to_dew_ast(), b.to_dew_ast()])
            }

            // Transforms - these expand to their mathematical equivalents
            Self::Rotate { center, angle } => {
                // center + rotate_vec(uv - center, angle)
                // where rotate_vec(v, a) = vec2(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a))
                let c = center.to_dew_ast();
                let a = angle.to_dew_ast();
                let delta = Ast::BinOp(
                    BinOp::Sub,
                    Box::new(Ast::Var("uv".into())),
                    Box::new(c.clone()),
                );
                // For now, use a rotate2d function that backends should implement
                let rotated = Ast::Call("rotate2d".into(), vec![delta, a]);
                Ast::BinOp(BinOp::Add, Box::new(c), Box::new(rotated))
            }
            Self::Scale { center, scale } => {
                // center + (uv - center) * scale
                let c = center.to_dew_ast();
                let s = scale.to_dew_ast();
                let delta = Ast::BinOp(
                    BinOp::Sub,
                    Box::new(Ast::Var("uv".into())),
                    Box::new(c.clone()),
                );
                let scaled = Ast::BinOp(BinOp::Mul, Box::new(delta), Box::new(s));
                Ast::BinOp(BinOp::Add, Box::new(c), Box::new(scaled))
            }
        }
    }
}

/// A typed expression AST for per-pixel color transforms (Vec4 → Vec4).
///
/// This is the expression language for the `map_pixels` primitive. Each variant
/// represents an operation that transforms RGBA color values.
///
/// # Design
///
/// Unlike raw closures, `ColorExpr` is:
/// - **Serializable** - Save/load effect pipelines
/// - **Interpretable** - Direct CPU evaluation
/// - **Inspectable** - Debug and optimize transforms
/// - **Future JIT/GPU** - Will compile to Cranelift/WGSL when dew-linalg is ready
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, ColorExpr, map_pixels};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
///
/// // Grayscale: luminance weighted average
/// let grayscale = ColorExpr::grayscale();
/// let result = map_pixels(&image, &grayscale);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColorExpr {
    // === Input ===
    /// The input RGBA color as Vec4.
    Rgba,
    /// Just the red channel.
    R,
    /// Just the green channel.
    G,
    /// Just the blue channel.
    B,
    /// Just the alpha channel.
    A,
    /// Computed luminance (0.2126*R + 0.7152*G + 0.0722*B).
    Luminance,

    // === Constructors ===
    /// Construct RGBA from four scalar expressions.
    Vec4 {
        r: Box<ColorExpr>,
        g: Box<ColorExpr>,
        b: Box<ColorExpr>,
        a: Box<ColorExpr>,
    },
    /// Construct RGB with explicit alpha.
    Vec3A {
        r: Box<ColorExpr>,
        g: Box<ColorExpr>,
        b: Box<ColorExpr>,
        a: Box<ColorExpr>,
    },

    // === Literals ===
    /// A constant scalar value (broadcasts to all channels).
    Constant(f32),
    /// A constant RGBA value.
    Constant4(f32, f32, f32, f32),

    // === Arithmetic ===
    /// Component-wise addition.
    Add(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise subtraction.
    Sub(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise multiplication.
    Mul(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise division.
    Div(Box<ColorExpr>, Box<ColorExpr>),
    /// Negate.
    Neg(Box<ColorExpr>),

    // === Math functions ===
    /// Absolute value.
    Abs(Box<ColorExpr>),
    /// Floor.
    Floor(Box<ColorExpr>),
    /// Fractional part.
    Fract(Box<ColorExpr>),
    /// Square root.
    Sqrt(Box<ColorExpr>),
    /// Power.
    Pow(Box<ColorExpr>, Box<ColorExpr>),
    /// Minimum.
    Min(Box<ColorExpr>, Box<ColorExpr>),
    /// Maximum.
    Max(Box<ColorExpr>, Box<ColorExpr>),
    /// Clamp to range.
    Clamp {
        value: Box<ColorExpr>,
        min: Box<ColorExpr>,
        max: Box<ColorExpr>,
    },
    /// Linear interpolation (mix).
    Lerp {
        a: Box<ColorExpr>,
        b: Box<ColorExpr>,
        t: Box<ColorExpr>,
    },
    /// Smooth step.
    SmoothStep {
        edge0: Box<ColorExpr>,
        edge1: Box<ColorExpr>,
        x: Box<ColorExpr>,
    },
    /// Step function.
    Step {
        edge: Box<ColorExpr>,
        x: Box<ColorExpr>,
    },

    // === Conditionals ===
    /// If-then-else based on comparison > 0.5.
    IfThenElse {
        condition: Box<ColorExpr>,
        then_expr: Box<ColorExpr>,
        else_expr: Box<ColorExpr>,
    },
    /// Greater than (returns 1.0 or 0.0).
    Gt(Box<ColorExpr>, Box<ColorExpr>),
    /// Less than (returns 1.0 or 0.0).
    Lt(Box<ColorExpr>, Box<ColorExpr>),

    // === Colorspace conversions ===
    // These operate on RGB channels, preserving alpha.
    // Input: vec4 RGBA, Output: vec4 where RGB is converted, A is preserved.
    /// Convert RGB to HSL (Hue, Saturation, Lightness).
    RgbToHsl(Box<ColorExpr>),
    /// Convert HSL to RGB.
    HslToRgb(Box<ColorExpr>),
    /// Convert RGB to HSV (Hue, Saturation, Value).
    RgbToHsv(Box<ColorExpr>),
    /// Convert HSV to RGB.
    HsvToRgb(Box<ColorExpr>),
    /// Convert RGB to HWB (Hue, Whiteness, Blackness).
    RgbToHwb(Box<ColorExpr>),
    /// Convert HWB to RGB.
    HwbToRgb(Box<ColorExpr>),
    /// Convert RGB to CIE LAB (Lightness, a*, b*).
    RgbToLab(Box<ColorExpr>),
    /// Convert CIE LAB to RGB.
    LabToRgb(Box<ColorExpr>),
    /// Convert RGB to LCH (Lightness, Chroma, Hue) - cylindrical LAB.
    RgbToLch(Box<ColorExpr>),
    /// Convert LCH to RGB.
    LchToRgb(Box<ColorExpr>),
    /// Convert RGB to OkLab (perceptually uniform).
    RgbToOklab(Box<ColorExpr>),
    /// Convert OkLab to RGB.
    OklabToRgb(Box<ColorExpr>),
    /// Convert RGB to OkLCH (cylindrical OkLab).
    RgbToOklch(Box<ColorExpr>),
    /// Convert OkLCH to RGB.
    OklchToRgb(Box<ColorExpr>),
    /// Convert RGB to YCbCr (luma, chroma).
    RgbToYcbcr(Box<ColorExpr>),
    /// Convert YCbCr to RGB.
    YcbcrToRgb(Box<ColorExpr>),
}

impl ColorExpr {
    /// Evaluate the expression for the given RGBA color.
    ///
    /// Returns the transformed color as [r', g', b', a'].
    pub fn eval(&self, r: f32, g: f32, b: f32, a: f32) -> [f32; 4] {
        match self {
            // Input
            Self::Rgba => [r, g, b, a],
            Self::R => [r, r, r, r],
            Self::G => [g, g, g, g],
            Self::B => [b, b, b, b],
            Self::A => [a, a, a, a],
            Self::Luminance => {
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                [lum, lum, lum, lum]
            }

            // Constructors
            Self::Vec4 {
                r: er,
                g: eg,
                b: eb,
                a: ea,
            } => {
                let [rv, _, _, _] = er.eval(r, g, b, a);
                let [gv, _, _, _] = eg.eval(r, g, b, a);
                let [bv, _, _, _] = eb.eval(r, g, b, a);
                let [av, _, _, _] = ea.eval(r, g, b, a);
                [rv, gv, bv, av]
            }
            Self::Vec3A {
                r: er,
                g: eg,
                b: eb,
                a: ea,
            } => {
                let [rv, _, _, _] = er.eval(r, g, b, a);
                let [gv, _, _, _] = eg.eval(r, g, b, a);
                let [bv, _, _, _] = eb.eval(r, g, b, a);
                let [av, _, _, _] = ea.eval(r, g, b, a);
                [rv, gv, bv, av]
            }

            // Literals
            Self::Constant(c) => [*c, *c, *c, *c],
            Self::Constant4(cr, cg, cb, ca) => [*cr, *cg, *cb, *ca],

            // Arithmetic
            Self::Add(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar + br, ag + bg, ab + bb, aa + ba]
            }
            Self::Sub(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar - br, ag - bg, ab - bb, aa - ba]
            }
            Self::Mul(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar * br, ag * bg, ab * bb, aa * ba]
            }
            Self::Div(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar / br, ag / bg, ab / bb, aa / ba]
            }
            Self::Neg(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [-er, -eg, -eb, -ea]
            }

            // Math functions
            Self::Abs(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.abs(), eg.abs(), eb.abs(), ea.abs()]
            }
            Self::Floor(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.floor(), eg.floor(), eb.floor(), ea.floor()]
            }
            Self::Fract(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.fract(), eg.fract(), eb.fract(), ea.fract()]
            }
            Self::Sqrt(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.sqrt(), eg.sqrt(), eb.sqrt(), ea.sqrt()]
            }
            Self::Pow(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.powf(br), ag.powf(bg), ab.powf(bb), aa.powf(ba)]
            }
            Self::Min(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.min(br), ag.min(bg), ab.min(bb), aa.min(ba)]
            }
            Self::Max(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.max(br), ag.max(bg), ab.max(bb), aa.max(ba)]
            }
            Self::Clamp { value, min, max } => {
                let [vr, vg, vb, va] = value.eval(r, g, b, a);
                let [minr, ming, minb, mina] = min.eval(r, g, b, a);
                let [maxr, maxg, maxb, maxa] = max.eval(r, g, b, a);
                [
                    vr.clamp(minr, maxr),
                    vg.clamp(ming, maxg),
                    vb.clamp(minb, maxb),
                    va.clamp(mina, maxa),
                ]
            }
            Self::Lerp {
                a: ea,
                b: eb,
                t: et,
            } => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                let [tr, tg, tb, ta] = et.eval(r, g, b, a);
                [
                    ar + (br - ar) * tr,
                    ag + (bg - ag) * tg,
                    ab + (bb - ab) * tb,
                    aa + (ba - aa) * ta,
                ]
            }
            Self::SmoothStep { edge0, edge1, x } => {
                let [e0r, e0g, e0b, e0a] = edge0.eval(r, g, b, a);
                let [e1r, e1g, e1b, e1a] = edge1.eval(r, g, b, a);
                let [xr, xg, xb, xa] = x.eval(r, g, b, a);

                let smooth = |x: f32, e0: f32, e1: f32| {
                    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };

                [
                    smooth(xr, e0r, e1r),
                    smooth(xg, e0g, e1g),
                    smooth(xb, e0b, e1b),
                    smooth(xa, e0a, e1a),
                ]
            }
            Self::Step { edge, x } => {
                let [er, eg, eb, ea] = edge.eval(r, g, b, a);
                let [xr, xg, xb, xa] = x.eval(r, g, b, a);
                [
                    if xr < er { 0.0 } else { 1.0 },
                    if xg < eg { 0.0 } else { 1.0 },
                    if xb < eb { 0.0 } else { 1.0 },
                    if xa < ea { 0.0 } else { 1.0 },
                ]
            }

            // Conditionals
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                let [cr, cg, cb, ca] = condition.eval(r, g, b, a);
                let [tr, tg, tb, ta] = then_expr.eval(r, g, b, a);
                let [er, eg, eb, ea] = else_expr.eval(r, g, b, a);
                [
                    if cr > 0.5 { tr } else { er },
                    if cg > 0.5 { tg } else { eg },
                    if cb > 0.5 { tb } else { eb },
                    if ca > 0.5 { ta } else { ea },
                ]
            }
            Self::Gt(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [
                    if ar > br { 1.0 } else { 0.0 },
                    if ag > bg { 1.0 } else { 0.0 },
                    if ab > bb { 1.0 } else { 0.0 },
                    if aa > ba { 1.0 } else { 0.0 },
                ]
            }
            Self::Lt(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [
                    if ar < br { 1.0 } else { 0.0 },
                    if ag < bg { 1.0 } else { 0.0 },
                    if ab < bb { 1.0 } else { 0.0 },
                    if aa < ba { 1.0 } else { 0.0 },
                ]
            }

            // Colorspace conversions
            Self::RgbToHsl(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, s, l) = rgb_to_hsl(er, eg, eb);
                [h, s, l, ea]
            }
            Self::HslToRgb(e) => {
                let [eh, es, el, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hsl_to_rgb(eh, es, el);
                [r, g, b, ea]
            }
            Self::RgbToHsv(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, s, v) = rgb_to_hsv(er, eg, eb);
                [h, s, v, ea]
            }
            Self::HsvToRgb(e) => {
                let [eh, es, ev, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hsv_to_rgb(eh, es, ev);
                [r, g, b, ea]
            }
            Self::RgbToHwb(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, w, b_val) = rgb_to_hwb(er, eg, eb);
                [h, w, b_val, ea]
            }
            Self::HwbToRgb(e) => {
                let [eh, ew, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hwb_to_rgb(eh, ew, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToLab(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, a_val, b_val) = rgb_to_lab(er, eg, eb);
                [l, a_val, b_val, ea]
            }
            Self::LabToRgb(e) => {
                let [el, ea_val, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = lab_to_rgb(el, ea_val, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToLch(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, c, h) = rgb_to_lch(er, eg, eb);
                [l, c, h, ea]
            }
            Self::LchToRgb(e) => {
                let [el, ec, eh, ea] = e.eval(r, g, b, a);
                let (r, g, b) = lch_to_rgb(el, ec, eh);
                [r, g, b, ea]
            }
            Self::RgbToOklab(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, a_val, b_val) = rgb_to_oklab(er, eg, eb);
                [l, a_val, b_val, ea]
            }
            Self::OklabToRgb(e) => {
                let [el, ea_val, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = oklab_to_rgb(el, ea_val, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToOklch(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, c, h) = rgb_to_oklch(er, eg, eb);
                [l, c, h, ea]
            }
            Self::OklchToRgb(e) => {
                let [el, ec, eh, ea] = e.eval(r, g, b, a);
                let (r, g, b) = oklch_to_rgb(el, ec, eh);
                [r, g, b, ea]
            }
            Self::RgbToYcbcr(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (y, cb, cr) = rgb_to_ycbcr(er, eg, eb);
                [y, cb, cr, ea]
            }
            Self::YcbcrToRgb(e) => {
                let [ey, ecb, ecr, ea] = e.eval(r, g, b, a);
                let (r, g, b) = ycbcr_to_rgb(ey, ecb, ecr);
                [r, g, b, ea]
            }
        }
    }

    /// Creates an identity transform (returns RGBA unchanged).
    pub fn identity() -> Self {
        Self::Rgba
    }

    /// Creates a grayscale transform using ITU-R BT.709 luminance.
    pub fn grayscale() -> Self {
        // luminance, luminance, luminance, alpha
        Self::Vec4 {
            r: Box::new(Self::Luminance),
            g: Box::new(Self::Luminance),
            b: Box::new(Self::Luminance),
            a: Box::new(Self::A),
        }
    }

    /// Creates an invert transform (1 - RGB, preserves alpha).
    pub fn invert() -> Self {
        Self::Vec4 {
            r: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::R))),
            g: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::G))),
            b: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::B))),
            a: Box::new(Self::A),
        }
    }

    /// Creates a threshold transform (luminance > threshold ? white : black).
    pub fn threshold(threshold: f32) -> Self {
        let lum_gt_threshold = Self::Gt(
            Box::new(Self::Luminance),
            Box::new(Self::Constant(threshold)),
        );
        Self::Vec4 {
            r: Box::new(lum_gt_threshold.clone()),
            g: Box::new(lum_gt_threshold.clone()),
            b: Box::new(lum_gt_threshold),
            a: Box::new(Self::A),
        }
    }

    /// Creates a brightness adjustment (multiply RGB by factor).
    pub fn brightness(factor: f32) -> Self {
        Self::Vec4 {
            r: Box::new(Self::Mul(
                Box::new(Self::R),
                Box::new(Self::Constant(factor)),
            )),
            g: Box::new(Self::Mul(
                Box::new(Self::G),
                Box::new(Self::Constant(factor)),
            )),
            b: Box::new(Self::Mul(
                Box::new(Self::B),
                Box::new(Self::Constant(factor)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Creates a contrast adjustment (scale RGB around 0.5).
    pub fn contrast(factor: f32) -> Self {
        // (color - 0.5) * factor + 0.5
        let adjust = |channel: Self| {
            Self::Add(
                Box::new(Self::Mul(
                    Box::new(Self::Sub(Box::new(channel), Box::new(Self::Constant(0.5)))),
                    Box::new(Self::Constant(factor)),
                )),
                Box::new(Self::Constant(0.5)),
            )
        };
        Self::Vec4 {
            r: Box::new(adjust(Self::R)),
            g: Box::new(adjust(Self::G)),
            b: Box::new(adjust(Self::B)),
            a: Box::new(Self::A),
        }
    }

    /// Creates a posterize effect (quantize to N levels).
    pub fn posterize(levels: u32) -> Self {
        let factor = (levels.max(2) - 1) as f32;
        // floor(color * factor) / factor
        let quantize = |channel: Self| {
            Self::Div(
                Box::new(Self::Floor(Box::new(Self::Mul(
                    Box::new(channel),
                    Box::new(Self::Constant(factor)),
                )))),
                Box::new(Self::Constant(factor)),
            )
        };
        Self::Vec4 {
            r: Box::new(quantize(Self::R)),
            g: Box::new(quantize(Self::G)),
            b: Box::new(quantize(Self::B)),
            a: Box::new(Self::A),
        }
    }

    /// Creates a gamma correction transform.
    pub fn gamma(gamma: f32) -> Self {
        let inv_gamma = 1.0 / gamma;
        Self::Vec4 {
            r: Box::new(Self::Pow(
                Box::new(Self::R),
                Box::new(Self::Constant(inv_gamma)),
            )),
            g: Box::new(Self::Pow(
                Box::new(Self::G),
                Box::new(Self::Constant(inv_gamma)),
            )),
            b: Box::new(Self::Pow(
                Box::new(Self::B),
                Box::new(Self::Constant(inv_gamma)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Creates a color tint (multiply by a color).
    pub fn tint(tint_r: f32, tint_g: f32, tint_b: f32) -> Self {
        Self::Vec4 {
            r: Box::new(Self::Mul(
                Box::new(Self::R),
                Box::new(Self::Constant(tint_r)),
            )),
            g: Box::new(Self::Mul(
                Box::new(Self::G),
                Box::new(Self::Constant(tint_g)),
            )),
            b: Box::new(Self::Mul(
                Box::new(Self::B),
                Box::new(Self::Constant(tint_b)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Converts this expression to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects an `rgba` Vec4 variable and returns Vec4.
    /// Use with `dew-linalg` for evaluation or compilation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rhizome_dew_linalg::{Value, eval, linalg_registry};
    /// use std::collections::HashMap;
    ///
    /// let expr = ColorExpr::grayscale();
    /// let ast = expr.to_dew_ast();
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("rgba".into(), Value::Vec4([1.0, 0.0, 0.0, 1.0]));
    ///
    /// let result = eval(&ast, &vars, &linalg_registry()).unwrap();
    /// // result = Value::Vec4([0.2126, 0.2126, 0.2126, 1.0])
    /// ```
    #[cfg(feature = "dew")]
    pub fn to_dew_ast(&self) -> rhizome_dew_core::Ast {
        use rhizome_dew_core::{Ast, BinOp, UnaryOp};

        match self {
            // Input - rgba is a Vec4 variable
            Self::Rgba => Ast::Var("rgba".into()),
            // Component extraction using x/y/z/w (or could use r/g/b/a if supported)
            Self::R => Ast::Call("x".into(), vec![Ast::Var("rgba".into())]),
            Self::G => Ast::Call("y".into(), vec![Ast::Var("rgba".into())]),
            Self::B => Ast::Call("z".into(), vec![Ast::Var("rgba".into())]),
            Self::A => Ast::Call("w".into(), vec![Ast::Var("rgba".into())]),
            // Luminance: 0.2126*R + 0.7152*G + 0.0722*B
            Self::Luminance => {
                let r = Ast::Call("x".into(), vec![Ast::Var("rgba".into())]);
                let g = Ast::Call("y".into(), vec![Ast::Var("rgba".into())]);
                let b = Ast::Call("z".into(), vec![Ast::Var("rgba".into())]);
                let term_r = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.2126)), Box::new(r));
                let term_g = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.7152)), Box::new(g));
                let term_b = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.0722)), Box::new(b));
                let sum_rg = Ast::BinOp(BinOp::Add, Box::new(term_r), Box::new(term_g));
                Ast::BinOp(BinOp::Add, Box::new(sum_rg), Box::new(term_b))
            }

            // Constructors
            Self::Vec4 { r, g, b, a } => Ast::Call(
                "vec4".into(),
                vec![
                    r.to_dew_ast(),
                    g.to_dew_ast(),
                    b.to_dew_ast(),
                    a.to_dew_ast(),
                ],
            ),
            Self::Vec3A { r, g, b, a } => {
                // Same as Vec4 - construct vec4 from components
                Ast::Call(
                    "vec4".into(),
                    vec![
                        r.to_dew_ast(),
                        g.to_dew_ast(),
                        b.to_dew_ast(),
                        a.to_dew_ast(),
                    ],
                )
            }

            // Literals
            Self::Constant(c) => Ast::Num(*c as f64),
            Self::Constant4(r, g, b, a) => Ast::Call(
                "vec4".into(),
                vec![
                    Ast::Num(*r as f64),
                    Ast::Num(*g as f64),
                    Ast::Num(*b as f64),
                    Ast::Num(*a as f64),
                ],
            ),

            // Binary operations
            Self::Add(a, b) => Ast::BinOp(
                BinOp::Add,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Sub(a, b) => Ast::BinOp(
                BinOp::Sub,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mul(a, b) => Ast::BinOp(
                BinOp::Mul,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Div(a, b) => Ast::BinOp(
                BinOp::Div,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            // Math functions
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Fract(a) => Ast::Call("fract".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Min(a, b) => Ast::Call("min".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Max(a, b) => Ast::Call("max".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Clamp { value, min, max } => Ast::Call(
                "clamp".into(),
                vec![value.to_dew_ast(), min.to_dew_ast(), max.to_dew_ast()],
            ),
            Self::Lerp { a, b, t } => Ast::Call(
                "lerp".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), t.to_dew_ast()],
            ),
            Self::SmoothStep { edge0, edge1, x } => Ast::Call(
                "smoothstep".into(),
                vec![edge0.to_dew_ast(), edge1.to_dew_ast(), x.to_dew_ast()],
            ),
            Self::Step { edge, x } => {
                Ast::Call("step".into(), vec![edge.to_dew_ast(), x.to_dew_ast()])
            }

            // Conditionals - these map to select/mix based on comparison
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                // select(else, then, condition > 0.5)
                // or we can use: lerp(else, then, step(0.5, condition))
                let cond_gt_half =
                    Ast::Call("step".into(), vec![Ast::Num(0.5), condition.to_dew_ast()]);
                Ast::Call(
                    "lerp".into(),
                    vec![else_expr.to_dew_ast(), then_expr.to_dew_ast(), cond_gt_half],
                )
            }
            Self::Gt(a, b) => {
                // step(b, a) returns 1.0 if a >= b, 0.0 otherwise
                // We want a > b strictly, but step is >=, close enough for floats
                Ast::Call("step".into(), vec![b.to_dew_ast(), a.to_dew_ast()])
            }
            Self::Lt(a, b) => {
                // step(a, b) returns 1.0 if b >= a, i.e., a <= b
                Ast::Call("step".into(), vec![a.to_dew_ast(), b.to_dew_ast()])
            }

            // Colorspace conversions - emit function calls that must be registered
            Self::RgbToHsl(e) => Ast::Call("rgb_to_hsl".into(), vec![e.to_dew_ast()]),
            Self::HslToRgb(e) => Ast::Call("hsl_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToHsv(e) => Ast::Call("rgb_to_hsv".into(), vec![e.to_dew_ast()]),
            Self::HsvToRgb(e) => Ast::Call("hsv_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToHwb(e) => Ast::Call("rgb_to_hwb".into(), vec![e.to_dew_ast()]),
            Self::HwbToRgb(e) => Ast::Call("hwb_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToLab(e) => Ast::Call("rgb_to_lab".into(), vec![e.to_dew_ast()]),
            Self::LabToRgb(e) => Ast::Call("lab_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToLch(e) => Ast::Call("rgb_to_lch".into(), vec![e.to_dew_ast()]),
            Self::LchToRgb(e) => Ast::Call("lch_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToOklab(e) => Ast::Call("rgb_to_oklab".into(), vec![e.to_dew_ast()]),
            Self::OklabToRgb(e) => Ast::Call("oklab_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToOklch(e) => Ast::Call("rgb_to_oklch".into(), vec![e.to_dew_ast()]),
            Self::OklchToRgb(e) => Ast::Call("oklch_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToYcbcr(e) => Ast::Call("rgb_to_ycbcr".into(), vec![e.to_dew_ast()]),
            Self::YcbcrToRgb(e) => Ast::Call("ycbcr_to_rgb".into(), vec![e.to_dew_ast()]),
        }
    }
}

// ============================================================================
// Colorspace dew function registration
// ============================================================================

/// Colorspace conversion functions for dew expression evaluation.
///
/// These functions allow colorspace conversions to be used in dew expressions
/// when evaluated via `rhizome_dew_linalg`.
///
/// # Example
///
/// ```ignore
/// use rhizome_dew_linalg::{linalg_registry, eval, Value};
/// use rhizome_resin_image::register_colorspace;
///
/// let mut registry = linalg_registry();
/// register_colorspace(&mut registry);
///
/// // Now you can use rgb_to_hsl, hsl_to_rgb, etc. in expressions
/// ```
#[cfg(feature = "dew")]
pub mod colorspace_dew {
    use num_traits::NumCast;
    use rhizome_dew_core::Numeric;
    use rhizome_dew_linalg::{FunctionRegistry, LinalgFn, LinalgValue, Signature, Type};

    macro_rules! colorspace_fn {
        ($name:ident, $fn_name:literal, $convert:expr) => {
            /// Colorspace conversion function for dew.
            pub struct $name;

            impl<T, V> LinalgFn<T, V> for $name
            where
                T: Numeric,
                V: LinalgValue<T>,
            {
                fn name(&self) -> &str {
                    $fn_name
                }

                fn signatures(&self) -> Vec<Signature> {
                    // Takes vec4 (RGBA), returns vec4 (converted RGB + preserved A)
                    vec![Signature {
                        args: vec![Type::Vec4],
                        ret: Type::Vec4,
                    }]
                }

                fn call(&self, args: &[V]) -> V {
                    let rgba = args[0].as_vec4().unwrap();
                    let r: f32 = NumCast::from(rgba[0]).unwrap_or(0.0);
                    let g: f32 = NumCast::from(rgba[1]).unwrap_or(0.0);
                    let b: f32 = NumCast::from(rgba[2]).unwrap_or(0.0);
                    let a: f32 = NumCast::from(rgba[3]).unwrap_or(1.0);

                    let convert: fn(f32, f32, f32) -> (f32, f32, f32) = $convert;
                    let (c0, c1, c2) = convert(r, g, b);

                    V::from_vec4([
                        NumCast::from(c0).unwrap_or_else(T::zero),
                        NumCast::from(c1).unwrap_or_else(T::zero),
                        NumCast::from(c2).unwrap_or_else(T::zero),
                        NumCast::from(a).unwrap_or_else(T::one),
                    ])
                }
            }
        };
    }

    colorspace_fn!(RgbToHsl, "rgb_to_hsl", super::rgb_to_hsl);
    colorspace_fn!(HslToRgb, "hsl_to_rgb", super::hsl_to_rgb);
    colorspace_fn!(RgbToHsv, "rgb_to_hsv", super::rgb_to_hsv);
    colorspace_fn!(HsvToRgb, "hsv_to_rgb", super::hsv_to_rgb);
    colorspace_fn!(RgbToHwb, "rgb_to_hwb", super::rgb_to_hwb);
    colorspace_fn!(HwbToRgb, "hwb_to_rgb", super::hwb_to_rgb);
    colorspace_fn!(RgbToLab, "rgb_to_lab", super::rgb_to_lab);
    colorspace_fn!(LabToRgb, "lab_to_rgb", super::lab_to_rgb);
    colorspace_fn!(RgbToLch, "rgb_to_lch", super::rgb_to_lch);
    colorspace_fn!(LchToRgb, "lch_to_rgb", super::lch_to_rgb);
    colorspace_fn!(RgbToOklab, "rgb_to_oklab", super::rgb_to_oklab);
    colorspace_fn!(OklabToRgb, "oklab_to_rgb", super::oklab_to_rgb);
    colorspace_fn!(RgbToOklch, "rgb_to_oklch", super::rgb_to_oklch);
    colorspace_fn!(OklchToRgb, "oklch_to_rgb", super::oklch_to_rgb);
    colorspace_fn!(RgbToYcbcr, "rgb_to_ycbcr", super::rgb_to_ycbcr);
    colorspace_fn!(YcbcrToRgb, "ycbcr_to_rgb", super::ycbcr_to_rgb);

    /// Registers all colorspace conversion functions into a dew-linalg registry.
    pub fn register_colorspace<T, V>(registry: &mut FunctionRegistry<T, V>)
    where
        T: Numeric,
        V: LinalgValue<T>,
    {
        registry.register(RgbToHsl);
        registry.register(HslToRgb);
        registry.register(RgbToHsv);
        registry.register(HsvToRgb);
        registry.register(RgbToHwb);
        registry.register(HwbToRgb);
        registry.register(RgbToLab);
        registry.register(LabToRgb);
        registry.register(RgbToLch);
        registry.register(LchToRgb);
        registry.register(RgbToOklab);
        registry.register(OklabToRgb);
        registry.register(RgbToOklch);
        registry.register(OklchToRgb);
        registry.register(RgbToYcbcr);
        registry.register(YcbcrToRgb);
    }
}

#[cfg(feature = "dew")]
pub use colorspace_dew::register_colorspace;

/// Remaps UV coordinates using an expression.
///
/// This is a fundamental primitive for geometric image transforms. All UV-based
/// effects (distortions, transforms, warps) can be expressed through this function.
///
/// # Arguments
///
/// * `image` - The source image to sample from
/// * `expr` - A `UvExpr` that maps (u, v) → (u', v')
///
/// # How It Works
///
/// For each output pixel at position (u, v):
/// 1. Evaluate `expr` to get the source coordinates (u', v')
/// 2. Sample the source image at (u', v')
/// 3. Write that color to the output pixel
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, UvExpr, remap_uv};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Simple translation
/// let translated = remap_uv(&image, &UvExpr::translate(0.1, 0.0));
///
/// // Rotation around center
/// let rotated = remap_uv(&image, &UvExpr::rotate_centered(0.5));
/// ```
pub fn remap_uv(image: &ImageField, expr: &UvExpr) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            let (src_u, src_v) = expr.eval(u, v);
            let color = image.sample_uv(src_u, src_v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a per-pixel color transform using an expression.
///
/// This is a fundamental primitive for color image transforms. All per-pixel
/// color effects (grayscale, invert, threshold, color grading) can be expressed
/// through this function.
///
/// # Arguments
///
/// * `image` - The source image to transform
/// * `expr` - A `ColorExpr` that maps (r, g, b, a) → (r', g', b', a')
///
/// # How It Works
///
/// For each pixel in the image:
/// 1. Read the RGBA color
/// 2. Evaluate `expr` to transform the color
/// 3. Write the transformed color back
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, ColorExpr, map_pixels};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
///
/// // Convert to grayscale
/// let gray = map_pixels(&image, &ColorExpr::grayscale());
///
/// // Invert colors
/// let inverted = map_pixels(&image, &ColorExpr::invert());
///
/// // Apply threshold
/// let binary = map_pixels(&image, &ColorExpr::threshold(0.5));
/// ```
pub fn map_pixels(image: &ImageField, expr: &ColorExpr) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let [r, g, b, a] = expr.eval(pixel[0], pixel[1], pixel[2], pixel[3]);
            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a UV coordinate remapping using a closure.
///
/// This is the runtime closure variant of [`remap_uv`]. Use this when:
/// - The transformation doesn't need to be serialized
/// - The transformation is a one-off custom effect
/// - The transformation references external state (like another image in [`displace`])
///
/// For serializable/compilable transforms, use [`remap_uv`] with [`UvExpr`] instead.
///
/// # Arguments
///
/// * `image` - The source image to transform
/// * `f` - A function that maps output UV → source UV coordinates
///
/// # Example
///
/// ```
/// use rhizome_resin_image::{ImageField, remap_uv_fn};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Custom swirl effect
/// let swirled = remap_uv_fn(&image, |u, v| {
///     let dx = u - 0.5;
///     let dy = v - 0.5;
///     let dist = (dx * dx + dy * dy).sqrt();
///     let angle = dist * 3.0;
///     let cos_a = angle.cos();
///     let sin_a = angle.sin();
///     (0.5 + dx * cos_a - dy * sin_a, 0.5 + dx * sin_a + dy * cos_a)
/// });
/// ```
pub fn remap_uv_fn(image: &ImageField, f: impl Fn(f32, f32) -> (f32, f32)) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            let (src_u, src_v) = f(u, v);
            let color = image.sample_uv(src_u, src_v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Internal: applies a per-pixel color transform using a closure.
///
/// This is for internal use where ColorExpr doesn't support the operation
/// (e.g., bit manipulation). Public API should use [`map_pixels`] with [`ColorExpr`].
pub(crate) fn map_pixels_fn(image: &ImageField, f: impl Fn([f32; 4]) -> [f32; 4]) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            data.push(f(pixel));
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
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

    // Convolution filter tests

    fn create_convolution_test_image() -> ImageField {
        // 5x5 image with a bright center for testing filters
        let mut data = vec![[0.0, 0.0, 0.0, 1.0]; 25];
        data[12] = [1.0, 1.0, 1.0, 1.0]; // Center pixel is white
        ImageField::from_raw(data, 5, 5)
    }

    #[test]
    fn test_kernel_identity() {
        let img = create_convolution_test_image();
        let result = convolve(&img, &Kernel::identity());

        // Identity kernel should not change the image
        let center = result.get_pixel(2, 2);
        assert!((center[0] - 1.0).abs() < 0.001);

        let corner = result.get_pixel(0, 0);
        assert!(corner[0].abs() < 0.001);
    }

    #[test]
    fn test_kernel_box_blur() {
        let img = create_convolution_test_image();
        let result = convolve(&img, &Kernel::box_blur());

        // Box blur should spread the center pixel's value
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.0 && center[0] < 1.0);

        // Neighbors should also have some value now
        let neighbor = result.get_pixel(2, 1);
        assert!(neighbor[0] > 0.0);
    }

    #[test]
    fn test_kernel_gaussian_blur() {
        let img = create_convolution_test_image();
        let result = convolve(&img, &Kernel::gaussian_blur_3x3());

        // Gaussian blur should smooth the image
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.0 && center[0] < 1.0);
    }

    #[test]
    fn test_kernel_sharpen() {
        // Create an image with gradual variation
        let data: Vec<_> = (0..25)
            .map(|i| {
                let v = (i as f32) / 24.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 5, 5);
        let result = convolve(&img, &Kernel::sharpen());

        // Sharpening should increase contrast at edges
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_kernel_sobel() {
        // Create an image with a vertical edge
        let mut data = vec![[0.0, 0.0, 0.0, 1.0]; 25];
        for y in 0..5 {
            data[y * 5 + 3] = [1.0, 1.0, 1.0, 1.0];
            data[y * 5 + 4] = [1.0, 1.0, 1.0, 1.0];
        }
        let img = ImageField::from_raw(data, 5, 5);

        let result = convolve(&img, &Kernel::sobel_vertical());

        // Vertical Sobel should detect the edge
        let edge_pixel = result.get_pixel(2, 2);
        assert!(edge_pixel[0].abs() > 0.1);
    }

    #[test]
    fn test_detect_edges() {
        let img = create_convolution_test_image();
        let edges = detect_edges(&img);

        assert_eq!(edges.dimensions(), (5, 5));
        // Edge detection should produce non-negative values
        let pixel = edges.get_pixel(2, 2);
        assert!(pixel[0] >= 0.0);
    }

    #[test]
    fn test_blur_function() {
        let img = create_convolution_test_image();
        let blurred = blur(&img, 2);

        // Multiple blur passes should spread the bright pixel more
        let center = blurred.get_pixel(2, 2);
        assert!(center[0] > 0.0 && center[0] < 0.5);
    }

    #[test]
    fn test_sharpen_function() {
        let img = create_convolution_test_image();
        let sharpened = sharpen(&img);

        assert_eq!(sharpened.dimensions(), (5, 5));
    }

    #[test]
    fn test_emboss_function() {
        let img = create_convolution_test_image();
        let embossed = emboss(&img);

        assert_eq!(embossed.dimensions(), (5, 5));
        // Emboss output should be normalized to visible range
        let pixel = embossed.get_pixel(2, 2);
        assert!(pixel[0] >= 0.0 && pixel[0] <= 1.0);
    }

    #[test]
    fn test_kernel_5x5() {
        let img = create_convolution_test_image();
        let result = convolve(&img, &Kernel::gaussian_blur_5x5());

        // 5x5 kernel should still work
        assert_eq!(result.dimensions(), (5, 5));
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.0);
    }

    // Normal map tests

    #[test]
    fn test_heightfield_to_normal_map_flat() {
        // Flat heightfield should produce normals pointing straight up (0.5, 0.5, 1.0)
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 9];
        let heightfield = ImageField::from_raw(data, 3, 3);

        let normal_map = heightfield_to_normal_map(&heightfield, 1.0);
        assert_eq!(normal_map.dimensions(), (3, 3));

        // Center pixel should have normal pointing up (encoded as ~0.5, ~0.5, ~1.0)
        let center = normal_map.get_pixel(1, 1);
        assert!((center[0] - 0.5).abs() < 0.1); // X ~= 0
        assert!((center[1] - 0.5).abs() < 0.1); // Y ~= 0
        assert!(center[2] > 0.9); // Z ~= 1 (pointing up)
    }

    #[test]
    fn test_heightfield_to_normal_map_slope() {
        // Create a slope (gradient from left to right)
        let data: Vec<_> = (0..9)
            .map(|i| {
                let v = (i % 3) as f32 / 2.0;
                [v, v, v, 1.0]
            })
            .collect();
        let heightfield = ImageField::from_raw(data, 3, 3);

        let normal_map = heightfield_to_normal_map(&heightfield, 2.0);
        assert_eq!(normal_map.dimensions(), (3, 3));

        // Normals should tilt in the X direction
        let center = normal_map.get_pixel(1, 1);
        // X component should be non-zero due to slope
        assert!(center[2] > 0.5); // Z still positive (pointing somewhat up)
    }

    #[test]
    fn test_heightfield_to_normal_map_strength() {
        // Same heightfield with different strengths
        let data: Vec<_> = (0..9)
            .map(|i| {
                let v = (i % 3) as f32 / 2.0;
                [v, v, v, 1.0]
            })
            .collect();
        let heightfield = ImageField::from_raw(data, 3, 3);

        let weak = heightfield_to_normal_map(&heightfield, 1.0);
        let strong = heightfield_to_normal_map(&heightfield, 5.0);

        // Stronger normals should have lower Z (more tilted)
        let weak_z = weak.get_pixel(1, 1)[2];
        let strong_z = strong.get_pixel(1, 1)[2];
        assert!(weak_z >= strong_z);
    }

    #[test]
    fn test_field_to_normal_map() {
        // Use a simple gradient field
        let config = BakeConfig::new(4, 4);
        let normal_map = field_to_normal_map(&GradientField, &config, 2.0);

        assert_eq!(normal_map.dimensions(), (4, 4));
        // All pixels should have valid normal values in [0, 1]
        for y in 0..4 {
            for x in 0..4 {
                let pixel = normal_map.get_pixel(x, y);
                assert!(pixel[0] >= 0.0 && pixel[0] <= 1.0);
                assert!(pixel[1] >= 0.0 && pixel[1] <= 1.0);
                assert!(pixel[2] >= 0.0 && pixel[2] <= 1.0);
            }
        }
    }

    // Channel operation tests

    #[test]
    fn test_extract_channel() {
        let data = vec![[1.0, 0.5, 0.25, 0.75]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        let red = extract_channel(&img, Channel::Red);
        assert_eq!(red.get_pixel(0, 0)[0], 1.0);
        assert_eq!(red.get_pixel(0, 0)[1], 1.0); // Grayscale - all channels same

        let green = extract_channel(&img, Channel::Green);
        assert_eq!(green.get_pixel(0, 0)[0], 0.5);

        let blue = extract_channel(&img, Channel::Blue);
        assert_eq!(blue.get_pixel(0, 0)[0], 0.25);

        let alpha = extract_channel(&img, Channel::Alpha);
        assert_eq!(alpha.get_pixel(0, 0)[0], 0.75);
    }

    #[test]
    fn test_split_channels() {
        let data = vec![[1.0, 0.5, 0.25, 0.75]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        let (r, g, b, a) = split_channels(&img);

        assert_eq!(r.get_pixel(0, 0)[0], 1.0);
        assert_eq!(g.get_pixel(0, 0)[0], 0.5);
        assert_eq!(b.get_pixel(0, 0)[0], 0.25);
        assert_eq!(a.get_pixel(0, 0)[0], 0.75);
    }

    #[test]
    fn test_merge_channels() {
        let r = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
        let g = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 4], 2, 2);
        let b = ImageField::from_raw(vec![[0.25, 0.25, 0.25, 1.0]; 4], 2, 2);
        let a = ImageField::from_raw(vec![[0.75, 0.75, 0.75, 1.0]; 4], 2, 2);

        let merged = merge_channels(&r, &g, &b, &a);
        let pixel = merged.get_pixel(0, 0);

        assert_eq!(pixel[0], 1.0);
        assert_eq!(pixel[1], 0.5);
        assert_eq!(pixel[2], 0.25);
        assert_eq!(pixel[3], 0.75);
    }

    #[test]
    fn test_set_channel() {
        let img = ImageField::from_raw(vec![[0.0, 0.0, 0.0, 1.0]; 4], 2, 2);
        let new_val = ImageField::from_raw(vec![[0.8, 0.8, 0.8, 1.0]; 4], 2, 2);

        let result = set_channel(&img, Channel::Red, &new_val);
        assert_eq!(result.get_pixel(0, 0)[0], 0.8);
        assert_eq!(result.get_pixel(0, 0)[1], 0.0); // Unchanged

        let result = set_channel(&img, Channel::Green, &new_val);
        assert_eq!(result.get_pixel(0, 0)[1], 0.8);
        assert_eq!(result.get_pixel(0, 0)[0], 0.0); // Unchanged
    }

    #[test]
    fn test_swap_channels() {
        let img = ImageField::from_raw(vec![[1.0, 0.5, 0.0, 1.0]; 4], 2, 2);
        let swapped = swap_channels(&img, Channel::Red, Channel::Blue);

        assert_eq!(swapped.get_pixel(0, 0)[0], 0.0); // Was blue
        assert_eq!(swapped.get_pixel(0, 0)[1], 0.5); // Unchanged
        assert_eq!(swapped.get_pixel(0, 0)[2], 1.0); // Was red
    }

    #[test]
    fn test_split_merge_roundtrip() {
        let data = vec![
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.3, 0.4, 0.5, 0.6],
        ];
        let img = ImageField::from_raw(data.clone(), 2, 2);

        let (r, g, b, a) = split_channels(&img);
        let merged = merge_channels(&r, &g, &b, &a);

        for (i, original) in data.iter().enumerate() {
            let x = (i % 2) as u32;
            let y = (i / 2) as u32;
            let pixel = merged.get_pixel(x, y);
            assert!((pixel[0] - original[0]).abs() < 0.001);
            assert!((pixel[1] - original[1]).abs() < 0.001);
            assert!((pixel[2] - original[2]).abs() < 0.001);
            assert!((pixel[3] - original[3]).abs() < 0.001);
        }
    }

    // Chromatic aberration tests

    #[test]
    fn test_chromatic_aberration_config() {
        let config = ChromaticAberrationConfig::new(0.02);
        assert_eq!(config.red_offset, 0.02);
        assert_eq!(config.green_offset, 0.0);
        assert_eq!(config.blue_offset, -0.02);
        assert_eq!(config.center, (0.5, 0.5));
    }

    #[test]
    fn test_chromatic_aberration_config_builder() {
        let config = ChromaticAberrationConfig {
            red_offset: 0.02,
            green_offset: 0.01,
            blue_offset: -0.01,
            center: (0.3, 0.7),
        };

        assert_eq!(config.center, (0.3, 0.7));
        assert_eq!(config.red_offset, 0.02);
        assert_eq!(config.green_offset, 0.01);
        assert_eq!(config.blue_offset, -0.01);
    }

    #[test]
    fn test_chromatic_aberration_preserves_dimensions() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let result = chromatic_aberration(&img, &ChromaticAberrationConfig::new(0.1));
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_chromatic_aberration_zero_strength() {
        // Zero strength should leave image unchanged
        let data = vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let img = ImageField::from_raw(data.clone(), 2, 2);

        let config = ChromaticAberrationConfig::new(0.0);
        let result = chromatic_aberration(&img, &config);

        for (i, original) in data.iter().enumerate() {
            let x = (i % 2) as u32;
            let y = (i / 2) as u32;
            let pixel = result.get_pixel(x, y);
            assert!((pixel[0] - original[0]).abs() < 0.01);
            assert!((pixel[1] - original[1]).abs() < 0.01);
            assert!((pixel[2] - original[2]).abs() < 0.01);
        }
    }

    #[test]
    fn test_chromatic_aberration_simple() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let result = chromatic_aberration_simple(&img, 0.05);
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_chromatic_aberration_center_unchanged() {
        // At the center, all channels should sample from nearly the same location
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 9];
        let img = ImageField::from_raw(data, 3, 3);

        let result = chromatic_aberration(&img, &ChromaticAberrationConfig::new(0.1));

        // Center pixel should be very similar to original
        let center = result.get_pixel(1, 1);
        assert!((center[0] - 0.5).abs() < 0.1);
        assert!((center[1] - 0.5).abs() < 0.1);
        assert!((center[2] - 0.5).abs() < 0.1);
    }

    // Color adjustment tests

    #[test]
    fn test_levels_default() {
        let data = vec![[0.3, 0.5, 0.7, 1.0]; 4];
        let img = ImageField::from_raw(data.clone(), 2, 2);

        // Default config should not change the image
        let result = adjust_levels(&img, &LevelsConfig::default());
        let pixel = result.get_pixel(0, 0);
        assert!((pixel[0] - 0.3).abs() < 0.01);
        assert!((pixel[1] - 0.5).abs() < 0.01);
        assert!((pixel[2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_levels_gamma() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // Gamma < 1 brightens
        let brightened = adjust_levels(&img, &LevelsConfig::gamma(0.5));
        assert!(brightened.get_pixel(0, 0)[0] > 0.5);

        // Gamma > 1 darkens
        let darkened = adjust_levels(&img, &LevelsConfig::gamma(2.0));
        assert!(darkened.get_pixel(0, 0)[0] < 0.5);
    }

    #[test]
    fn test_levels_remap() {
        let data = vec![[0.3, 0.5, 0.7, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // Remap [0.2, 0.8] to [0, 1] - should increase contrast
        let config = LevelsConfig::remap(0.2, 0.8);
        let result = adjust_levels(&img, &config);

        // 0.3 should map to ~0.167 (below black point)
        // 0.5 should map to ~0.5 (middle)
        // 0.7 should map to ~0.833 (above black point)
        let pixel = result.get_pixel(0, 0);
        assert!(pixel[0] < 0.3); // Darker than original
        assert!((pixel[1] - 0.5).abs() < 0.01); // About the same
        assert!(pixel[2] > 0.7); // Brighter than original
    }

    #[test]
    fn test_brightness_contrast() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // Brightness only
        let brighter = adjust_brightness_contrast(&img, 0.2, 0.0);
        assert!((brighter.get_pixel(0, 0)[0] - 0.7).abs() < 0.01);

        // Contrast only - midpoint unchanged
        let contrasted = adjust_brightness_contrast(&img, 0.0, 0.5);
        assert!((contrasted.get_pixel(0, 0)[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_brightness_contrast_edges() {
        let data = vec![[0.0, 0.25, 0.75, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // High contrast should push values toward 0 and 1
        let contrasted = adjust_brightness_contrast(&img, 0.0, 0.5);
        let pixel = contrasted.get_pixel(0, 0);
        assert!(pixel[0] < 0.0 + 0.01); // Should be clamped to 0
        assert!(pixel[1] < 0.25); // Should be pushed darker
        assert!(pixel[2] > 0.75); // Should be pushed brighter
        assert!(pixel[3] > 0.99); // Should be clamped to 1
    }

    #[test]
    fn test_hsl_adjustment_hue() {
        let data = vec![[1.0, 0.0, 0.0, 1.0]; 4]; // Pure red
        let img = ImageField::from_raw(data, 2, 2);

        // Shift hue by 1/3 (120 degrees) - red -> green
        let result = adjust_hsl(&img, &HslAdjustment::hue(1.0 / 3.0));
        let pixel = result.get_pixel(0, 0);
        assert!(pixel[1] > pixel[0]); // Green should dominate
        assert!(pixel[1] > pixel[2]); // Green > blue
    }

    #[test]
    fn test_hsl_adjustment_saturation() {
        let data = vec![[1.0, 0.5, 0.5, 1.0]; 4]; // Pinkish
        let img = ImageField::from_raw(data, 2, 2);

        // Desaturate
        let desaturated = adjust_hsl(&img, &HslAdjustment::saturation(-0.5));
        let pixel = desaturated.get_pixel(0, 0);
        // Should be more gray - channels closer together
        let range = pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
        assert!(range < 0.5); // Range should be reduced
    }

    #[test]
    fn test_grayscale() {
        let data = vec![[1.0, 0.0, 0.0, 1.0]; 4]; // Red
        let img = ImageField::from_raw(data, 2, 2);

        let gray = grayscale(&img);
        let pixel = gray.get_pixel(0, 0);

        // All channels should be equal
        assert!((pixel[0] - pixel[1]).abs() < 0.001);
        assert!((pixel[1] - pixel[2]).abs() < 0.001);

        // Should be approximately 0.2126 (red luminance coefficient)
        assert!((pixel[0] - 0.2126).abs() < 0.01);
    }

    #[test]
    fn test_invert() {
        let data = vec![[0.2, 0.5, 0.8, 0.9]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        let inverted = invert(&img);
        let pixel = inverted.get_pixel(0, 0);

        assert!((pixel[0] - 0.8).abs() < 0.001);
        assert!((pixel[1] - 0.5).abs() < 0.001);
        assert!((pixel[2] - 0.2).abs() < 0.001);
        assert!((pixel[3] - 0.9).abs() < 0.001); // Alpha unchanged
    }

    #[test]
    fn test_posterize() {
        let data = vec![[0.33, 0.66, 0.99, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // 2 levels = only 0 or 1
        let posterized = posterize(&img, 2);
        let pixel = posterized.get_pixel(0, 0);
        assert!(pixel[0] == 0.0 || pixel[0] == 1.0);
        assert!(pixel[1] == 0.0 || pixel[1] == 1.0);
        assert!(pixel[2] == 0.0 || pixel[2] == 1.0);
    }

    #[test]
    fn test_threshold() {
        let data = vec![
            [0.3, 0.3, 0.3, 1.0],
            [0.7, 0.7, 0.7, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
        ];
        let img = ImageField::from_raw(data, 2, 2);

        let result = threshold(&img, 0.5);

        // 0.3 luminance < 0.5 -> black
        assert!(result.get_pixel(0, 0)[0] < 0.01);
        // 0.7 luminance > 0.5 -> white
        assert!(result.get_pixel(1, 0)[0] > 0.99);
    }

    // Dithering tests - decomposed primitives

    #[test]
    fn test_quantize_primitive() {
        let q = Quantize::new(2);
        assert_eq!(q.apply(0.0), 0.0);
        assert_eq!(q.apply(1.0), 1.0);
        assert_eq!(q.apply(0.3), 0.0);
        assert_eq!(q.apply(0.7), 1.0);
        assert_eq!(q.apply(0.5), 1.0); // rounds to nearest

        let q4 = Quantize::new(4);
        assert!((q4.apply(0.4) - 1.0 / 3.0).abs() < 0.01);
        assert!((q4.apply(0.6) - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bayer_field() {
        let bayer = BayerField::bayer4x4();
        let ctx = EvalContext::new();

        // Sample at different positions - should get values in 0-1 range
        for i in 0..16 {
            let x = (i % 4) as f32 / 1000.0;
            let y = (i / 4) as f32 / 1000.0;
            let v = bayer.sample(Vec2::new(x, y), &ctx);
            assert!(v >= 0.0 && v <= 1.0, "Bayer value out of range: {}", v);
        }
    }

    #[test]
    fn test_quantize_with_threshold_field() {
        // Create simple solid color field
        let img = ImageField::solid(Rgba::new(0.5, 0.5, 0.5, 1.0));
        let bayer = BayerField::bayer4x4();
        let ctx = EvalContext::new();

        let dithered = QuantizeWithThreshold::new(img, bayer, 2);

        // Sample at various positions - should get 0 or 1
        for i in 0..16 {
            let x = (i % 4) as f32 * 0.001;
            let y = (i / 4) as f32 * 0.001;
            let color = dithered.sample(Vec2::new(x, y), &ctx);
            assert!(
                color.r == 0.0 || color.r == 1.0,
                "Expected binary output, got {}",
                color.r
            );
        }
    }

    #[test]
    fn test_error_diffuse_floyd_steinberg() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);

        let result = ErrorDiffuse::floyd_steinberg(2).apply(&img);

        // Should have mix of black and white
        let mut black_count = 0;
        let mut white_count = 0;
        for y in 0..8 {
            for x in 0..8 {
                if result.get_pixel(x, y)[0] < 0.5 {
                    black_count += 1;
                } else {
                    white_count += 1;
                }
            }
        }
        assert!(black_count > 20, "Expected more black pixels for 50% gray");
        assert!(white_count > 20, "Expected more white pixels for 50% gray");
    }

    #[test]
    fn test_error_diffuse_preserves_alpha() {
        let data = vec![[0.5, 0.5, 0.5, 0.7]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let result = ErrorDiffuse::atkinson(2).apply(&img);

        for y in 0..4 {
            for x in 0..4 {
                assert!(
                    (result.get_pixel(x, y)[3] - 0.7).abs() < 0.001,
                    "Alpha should be preserved"
                );
            }
        }
    }

    #[test]
    fn test_curve_diffuse_riemersma() {
        let data: Vec<_> = (0..64)
            .map(|i| {
                let v = i as f32 / 63.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 8, 8);

        let result = CurveDiffuse::new(2).apply(&img);

        // All pixels should be quantized to 0 or 1
        for y in 0..8 {
            for x in 0..8 {
                let v = result.get_pixel(x, y)[0];
                assert!(
                    v == 0.0 || v == 1.0,
                    "CurveDiffuse should produce binary output, got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_werness_dither() {
        let data: Vec<_> = (0..64)
            .map(|i| {
                let v = i as f32 / 63.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 8, 8);

        let result = WernessDither::new(2).apply(&img);

        // All pixels should be quantized to 0 or 1
        for y in 0..8 {
            for x in 0..8 {
                let v = result.get_pixel(x, y)[0];
                assert!(
                    v == 0.0 || v == 1.0,
                    "Werness dither should produce binary output, got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_werness_preserves_alpha() {
        let data = vec![[0.5, 0.5, 0.5, 0.7]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let result = WernessDither::new(2).apply(&img);

        for y in 0..4 {
            for x in 0..4 {
                assert!(
                    (result.get_pixel(x, y)[3] - 0.7).abs() < 0.001,
                    "Alpha should be preserved"
                );
            }
        }
    }

    #[test]
    fn test_generate_blue_noise_2d() {
        let noise = generate_blue_noise_2d(16);
        assert_eq!(noise.dimensions(), (16, 16));

        // Check that values are in valid range
        for y in 0..16 {
            for x in 0..16 {
                let v = noise.get_pixel(x, y)[0];
                assert!(v >= 0.0 && v <= 1.0, "Blue noise value out of range: {}", v);
            }
        }

        // Check that we have variety in values (not constant or degenerate)
        let mut values: Vec<f32> = Vec::with_capacity(256);
        for y in 0..16 {
            for x in 0..16 {
                values.push(noise.get_pixel(x, y)[0]);
            }
        }
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Should have a reasonable range of values
        assert!(
            max - min > 0.5,
            "Blue noise should have good range, got min={}, max={}",
            min,
            max
        );
    }

    #[test]
    fn test_blue_noise_field() {
        let noise = generate_blue_noise_2d(16);
        let field = BlueNoise2D::from_texture(noise);
        let ctx = EvalContext::new();

        // Check that it returns values in [0, 1]
        for y in 0..16 {
            for x in 0..16 {
                let uv = Vec2::new(x as f32 / 16.0, y as f32 / 16.0);
                let v = field.sample(uv, &ctx);
                assert!(v >= 0.0 && v <= 1.0, "Blue noise value out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_generate_blue_noise_1d() {
        let noise = generate_blue_noise_1d(64);
        assert_eq!(noise.len(), 64);

        // Check range
        for &v in &noise {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Blue noise 1D value out of range: {}",
                v
            );
        }

        // Check variety
        let min = noise.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = noise.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.5, "Blue noise 1D should have good range");
    }

    #[test]
    fn test_generate_blue_noise_3d() {
        // Small size due to cost
        let noise = generate_blue_noise_3d(8);
        assert_eq!(noise.len(), 8 * 8 * 8);

        // Check range
        for &v in &noise {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Blue noise 3D value out of range: {}",
                v
            );
        }

        // Check variety
        let min = noise.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = noise.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.3, "Blue noise 3D should have good range");
    }

    #[test]
    fn test_threshold_dither_with_blue_noise() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);
        let noise = generate_blue_noise_2d(8);
        let ctx = EvalContext::new();

        let dithered = QuantizeWithThreshold::new(img, BlueNoise2D::from_texture(noise), 2);

        // Check for mix of black and white
        let mut has_black = false;
        let mut has_white = false;
        for y in 0..8 {
            for x in 0..8 {
                let uv = Vec2::new(x as f32 / 8.0, y as f32 / 8.0);
                let v = dithered.sample(uv, &ctx).r;
                if v < 0.5 {
                    has_black = true;
                } else {
                    has_white = true;
                }
            }
        }
        assert!(
            has_black && has_white,
            "Blue noise dithering should produce mix of values"
        );
    }

    #[test]
    fn test_temporal_bayer_varies_by_frame() {
        let ctx = EvalContext::new();
        let pos = Vec2::new(0.5, 0.5);

        // Different frames should produce different thresholds
        let mut values = Vec::new();
        for frame in 0..16 {
            let bayer = TemporalBayer::bayer4x4(frame);
            values.push(bayer.sample(pos, &ctx));
        }

        // Should have multiple distinct values (not all the same)
        let mut unique = values.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        assert!(unique.len() > 1, "Temporal Bayer should vary across frames");
    }

    #[test]
    fn test_temporal_bayer_range() {
        let ctx = EvalContext::new();

        // All values should be in [0, 1)
        for frame in 0..10 {
            let bayer = TemporalBayer::bayer8x8(frame);
            for i in 0..10 {
                for j in 0..10 {
                    let pos = Vec2::new(i as f32 / 10.0, j as f32 / 10.0);
                    let v = bayer.sample(pos, &ctx);
                    assert!(v >= 0.0 && v < 1.0, "Bayer value {} out of range", v);
                }
            }
        }
    }

    #[test]
    fn test_ign_varies_by_frame() {
        let ctx = EvalContext::new();
        let pos = Vec2::new(0.5, 0.5);

        // Different frames should produce different values
        let mut values = Vec::new();
        for frame in 0..10 {
            let ign = InterleavedGradientNoise::new(frame);
            values.push(ign.sample(pos, &ctx));
        }

        // Should have distinct values
        let mut unique = values.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        assert!(unique.len() > 5, "IGN should vary across frames");
    }

    #[test]
    fn test_ign_range() {
        let ctx = EvalContext::new();

        // All values should be in [0, 1)
        for frame in 0..10 {
            let ign = InterleavedGradientNoise::new(frame);
            for i in 0..10 {
                for j in 0..10 {
                    let pos = Vec2::new(i as f32 / 10.0, j as f32 / 10.0);
                    let v = ign.sample(pos, &ctx);
                    assert!(v >= 0.0 && v < 1.0, "IGN value {} out of range", v);
                }
            }
        }
    }

    #[test]
    fn test_temporal_blue_noise_varies_by_frame() {
        let ctx = EvalContext::new();
        let noise = BlueNoise3D::generate(8);
        let pos = Vec2::new(0.5, 0.5);

        // Different frames should produce different values
        let mut values = Vec::new();
        for frame in 0..8 {
            let temporal = TemporalBlueNoise::from_noise(noise.clone(), frame);
            values.push(temporal.sample(pos, &ctx));
        }

        // Should have distinct values
        let mut unique = values.clone();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        assert!(
            unique.len() > 3,
            "Temporal blue noise should vary across frames"
        );
    }

    #[test]
    fn test_temporal_blue_noise_at_frame() {
        let ctx = EvalContext::new();
        let noise = BlueNoise3D::generate(8);
        let pos = Vec2::new(0.25, 0.75);

        let temporal1 = TemporalBlueNoise::from_noise(noise.clone(), 3);
        let temporal2 = temporal1.at_frame(3);

        // Same frame should give same value
        let v1 = temporal1.sample(pos, &ctx);
        let v2 = temporal2.sample(pos, &ctx);
        assert!(
            (v1 - v2).abs() < 0.001,
            "at_frame should preserve noise data"
        );
    }

    #[test]
    fn test_hilbert_curve_coverage() {
        // Verify Hilbert curve covers all points in a 4x4 grid
        let order = 2u32; // 2^2 = 4x4
        let size = 1u32 << order;
        let mut visited = vec![vec![false; size as usize]; size as usize];

        for d in 0..(size * size) {
            let (x, y) = hilbert_d2xy(order, d);
            assert!(x < size && y < size, "Hilbert point out of bounds");
            visited[y as usize][x as usize] = true;
        }

        // All points should be visited
        for y in 0..size as usize {
            for x in 0..size as usize {
                assert!(
                    visited[y][x],
                    "Point ({}, {}) not visited by Hilbert curve",
                    x, y
                );
            }
        }
    }

    // Distortion tests

    #[test]
    fn test_lens_distortion_barrel() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let result = lens_distortion(&img, &LensDistortionConfig::barrel(0.5));
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_lens_distortion_pincushion() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let result = lens_distortion(&img, &LensDistortionConfig::pincushion(0.5));
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_lens_distortion_zero_strength() {
        let data: Vec<_> = (0..16).map(|i| [i as f32 / 15.0; 4]).collect();
        let img = ImageField::from_raw(data.clone(), 4, 4);

        let result = lens_distortion(&img, &LensDistortionConfig::default());

        // Zero strength should not significantly change the image
        for i in 0..16 {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let orig = img.get_pixel(x, y)[0];
            let new = result.get_pixel(x, y)[0];
            assert!((orig - new).abs() < 0.1);
        }
    }

    #[test]
    fn test_wave_distortion_horizontal() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let result = wave_distortion(&img, &WaveDistortionConfig::horizontal(0.1, 2.0));
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_wave_distortion_vertical() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let result = wave_distortion(&img, &WaveDistortionConfig::vertical(0.1, 2.0));
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_displace_neutral() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data.clone(), 4, 4);

        // Displacement map with all 0.5 = no displacement
        let disp_map = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
        let result = displace(&img, &disp_map, 0.2);

        // Should be unchanged
        for i in 0..16 {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let pixel = result.get_pixel(x, y);
            assert!((pixel[0] - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_displace_offset() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        // Red > 0.5 = offset right, Green > 0.5 = offset down
        let disp_map = ImageField::from_raw(vec![[1.0, 1.0, 0.5, 1.0]; 16], 4, 4);
        let result = displace(&img, &disp_map, 0.1);

        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_swirl() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let result = swirl(&img, std::f32::consts::PI, 0.5, (0.5, 0.5));
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_swirl_zero_angle() {
        let data: Vec<_> = (0..16).map(|i| [i as f32 / 15.0; 4]).collect();
        let img = ImageField::from_raw(data.clone(), 4, 4);

        // Zero angle swirl should not change the image
        let result = swirl(&img, 0.0, 0.5, (0.5, 0.5));

        for i in 0..16 {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let orig = img.get_pixel(x, y)[0];
            let new = result.get_pixel(x, y)[0];
            assert!((orig - new).abs() < 0.01);
        }
    }

    #[test]
    fn test_spherize() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let bulge = spherize(&img, 0.5, (0.5, 0.5));
        let pinch = spherize(&img, -0.5, (0.5, 0.5));

        assert_eq!(bulge.dimensions(), (5, 5));
        assert_eq!(pinch.dimensions(), (5, 5));
    }

    #[test]
    fn test_spherize_zero() {
        let data: Vec<_> = (0..16).map(|i| [i as f32 / 15.0; 4]).collect();
        let img = ImageField::from_raw(data.clone(), 4, 4);

        // Zero strength should not significantly change the image
        let result = spherize(&img, 0.0, (0.5, 0.5));

        for i in 0..16 {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let orig = img.get_pixel(x, y)[0];
            let new = result.get_pixel(x, y)[0];
            assert!((orig - new).abs() < 0.1);
        }
    }

    // Image pyramid tests

    #[test]
    fn test_downsample() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let half = downsample(&img);
        assert_eq!(half.dimensions(), (2, 2));

        // All pixels should still be ~0.5
        for y in 0..2 {
            for x in 0..2 {
                let pixel = half.get_pixel(x, y);
                assert!((pixel[0] - 0.5).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_downsample_averaging() {
        // Create image with distinct quadrants
        let data = vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let img = ImageField::from_raw(data, 2, 2);

        let half = downsample(&img);
        assert_eq!(half.dimensions(), (1, 1));

        // Should average to 0.5
        let pixel = half.get_pixel(0, 0);
        assert!((pixel[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_upsample() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        let double = upsample(&img);
        assert_eq!(double.dimensions(), (4, 4));
    }

    #[test]
    fn test_upsample_downsample_roundtrip() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let down = downsample(&img);
        let up = upsample(&down);

        // Should be back to original size
        assert_eq!(up.dimensions(), (4, 4));

        // Values should be similar
        for y in 0..4 {
            for x in 0..4 {
                let pixel = up.get_pixel(x, y);
                assert!((pixel[0] - 0.5).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_gaussian_pyramid() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);

        let pyramid = ImagePyramid::gaussian(&img, 4);

        assert_eq!(pyramid.len(), 4);
        assert!(!pyramid.is_empty());

        // Check dimensions decrease
        assert_eq!(pyramid.levels[0].dimensions(), (8, 8));
        assert_eq!(pyramid.levels[1].dimensions(), (4, 4));
        assert_eq!(pyramid.levels[2].dimensions(), (2, 2));
        assert_eq!(pyramid.levels[3].dimensions(), (1, 1));
    }

    #[test]
    fn test_gaussian_pyramid_finest_coarsest() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);

        let pyramid = ImagePyramid::gaussian(&img, 4);

        assert_eq!(pyramid.finest().unwrap().dimensions(), (8, 8));
        assert_eq!(pyramid.coarsest().unwrap().dimensions(), (1, 1));
    }

    #[test]
    fn test_laplacian_pyramid() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);

        let pyramid = ImagePyramid::laplacian(&img, 4);

        assert_eq!(pyramid.len(), 4);
    }

    #[test]
    fn test_laplacian_reconstruct() {
        let data: Vec<_> = (0..64)
            .map(|i| {
                let v = i as f32 / 63.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 8, 8);

        let pyramid = ImagePyramid::laplacian(&img, 3);
        let reconstructed = pyramid.reconstruct_laplacian().unwrap();

        assert_eq!(reconstructed.dimensions(), (8, 8));

        // Reconstruction should be close to original
        // (some loss is expected due to blur/downsample/upsample)
    }

    #[test]
    fn test_resize_up() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let resized = resize(&img, 8, 6);
        assert_eq!(resized.dimensions(), (8, 6));
    }

    #[test]
    fn test_resize_down() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
        let img = ImageField::from_raw(data, 8, 8);

        let resized = resize(&img, 3, 3);
        assert_eq!(resized.dimensions(), (3, 3));
    }

    #[test]
    fn test_resize_preserves_values() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
        let img = ImageField::from_raw(data, 4, 4);

        let resized = resize(&img, 8, 8);

        // Values should be similar
        for y in 0..8 {
            for x in 0..8 {
                let pixel = resized.get_pixel(x, y);
                assert!((pixel[0] - 0.5).abs() < 0.1);
            }
        }
    }

    // ========== Inpainting tests ==========

    #[test]
    fn test_inpaint_config_default() {
        let config = InpaintConfig::default();
        assert_eq!(config.iterations, 100);
        assert!((config.diffusion_rate - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_inpaint_config_builder() {
        let config = InpaintConfig::new(50).with_diffusion_rate(0.5);
        assert_eq!(config.iterations, 50);
        assert!((config.diffusion_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_inpaint_diffusion_basic() {
        // Create a simple 8x8 image with a hole in the center
        let mut data = vec![[0.5, 0.5, 0.5, 1.0]; 64];

        // Create mask (center 2x2 needs inpainting)
        let mut mask_data = vec![[0.0, 0.0, 0.0, 1.0]; 64];
        mask_data[27] = [1.0, 1.0, 1.0, 1.0]; // (3, 3)
        mask_data[28] = [1.0, 1.0, 1.0, 1.0]; // (4, 3)
        mask_data[35] = [1.0, 1.0, 1.0, 1.0]; // (3, 4)
        mask_data[36] = [1.0, 1.0, 1.0, 1.0]; // (4, 4)

        // Set hole to black
        data[27] = [0.0, 0.0, 0.0, 1.0];
        data[28] = [0.0, 0.0, 0.0, 1.0];
        data[35] = [0.0, 0.0, 0.0, 1.0];
        data[36] = [0.0, 0.0, 0.0, 1.0];

        let img = ImageField::from_raw(data, 8, 8);
        let mask = ImageField::from_raw(mask_data, 8, 8);

        let config = InpaintConfig::new(50);
        let result = inpaint_diffusion(&img, &mask, &config);

        // After diffusion, the hole should be filled with values closer to 0.5
        let p1 = result.get_pixel(3, 3);
        let p2 = result.get_pixel(4, 4);

        // Should have moved toward the surrounding gray
        assert!(p1[0] > 0.1, "Pixel should be filled, got {}", p1[0]);
        assert!(p2[0] > 0.1, "Pixel should be filled, got {}", p2[0]);
    }

    #[test]
    fn test_inpaint_diffusion_preserves_known() {
        let data = vec![[0.8, 0.2, 0.4, 1.0]; 64];
        let mut mask_data = vec![[0.0, 0.0, 0.0, 1.0]; 64];
        mask_data[27] = [1.0, 1.0, 1.0, 1.0]; // Only one pixel to fill

        let img = ImageField::from_raw(data, 8, 8);
        let mask = ImageField::from_raw(mask_data, 8, 8);

        let config = InpaintConfig::new(10);
        let result = inpaint_diffusion(&img, &mask, &config);

        // Known pixels should be preserved
        let known = result.get_pixel(0, 0);
        assert!((known[0] - 0.8).abs() < 1e-6);
        assert!((known[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_patchmatch_config_default() {
        let config = PatchMatchConfig::default();
        assert_eq!(config.patch_size, 7);
        assert_eq!(config.pyramid_levels, 4);
        assert_eq!(config.iterations, 5);
    }

    #[test]
    fn test_patchmatch_config_ensures_odd() {
        let config = PatchMatchConfig::new(8);
        assert_eq!(config.patch_size, 9); // Should round up to odd
    }

    #[test]
    fn test_inpaint_patchmatch_basic() {
        // Create a simple textured image
        let data: Vec<[f32; 4]> = (0..256)
            .map(|i| {
                let x = i % 16;
                let y = i / 16;
                let checker = ((x / 2 + y / 2) % 2) as f32;
                [checker, checker, checker, 1.0]
            })
            .collect();

        // Mask out a small region
        let mut mask_data = vec![[0.0, 0.0, 0.0, 1.0]; 256];
        for y in 6..10 {
            for x in 6..10 {
                mask_data[y * 16 + x] = [1.0, 1.0, 1.0, 1.0];
            }
        }

        let img = ImageField::from_raw(data, 16, 16);
        let mask = ImageField::from_raw(mask_data, 16, 16);

        let config = PatchMatchConfig::new(3)
            .with_pyramid_levels(2)
            .with_iterations(2);
        let result = inpaint_patchmatch(&img, &mask, &config);

        // Result should have same dimensions
        assert_eq!(result.dimensions(), (16, 16));
    }

    #[test]
    fn test_create_color_key_mask() {
        let data: Vec<[f32; 4]> = vec![
            [1.0, 0.0, 1.0, 1.0], // magenta - should be masked
            [0.9, 0.1, 0.9, 1.0], // near magenta - within tolerance
            [0.5, 0.5, 0.5, 1.0], // gray - should not be masked
            [1.0, 1.0, 1.0, 1.0], // white - should not be masked
        ];

        let img = ImageField::from_raw(data, 2, 2);
        let key = Rgba::new(1.0, 0.0, 1.0, 1.0); // magenta

        let mask = create_color_key_mask(&img, key, 0.2);

        // First two pixels should be masked (white in mask)
        assert!(mask.get_pixel(0, 0)[0] > 0.5);
        assert!(mask.get_pixel(1, 0)[0] > 0.5);

        // Last two should not be masked (black in mask)
        assert!(mask.get_pixel(0, 1)[0] < 0.5);
        assert!(mask.get_pixel(1, 1)[0] < 0.5);
    }

    #[test]
    fn test_dilate_mask() {
        // Create a mask with a single white pixel in center
        let mut mask_data = vec![[0.0, 0.0, 0.0, 1.0]; 25];
        mask_data[12] = [1.0, 1.0, 1.0, 1.0]; // Center of 5x5

        let mask = ImageField::from_raw(mask_data, 5, 5);
        let dilated = dilate_mask(&mask, 1);

        // Center should still be white
        assert!(dilated.get_pixel(2, 2)[0] > 0.5);

        // Immediate neighbors (4-connected) should be white
        assert!(dilated.get_pixel(1, 2)[0] > 0.5);
        assert!(dilated.get_pixel(3, 2)[0] > 0.5);
        assert!(dilated.get_pixel(2, 1)[0] > 0.5);
        assert!(dilated.get_pixel(2, 3)[0] > 0.5);

        // Corners of 5x5 should still be black with radius 1
        assert!(dilated.get_pixel(0, 0)[0] < 0.5);
        assert!(dilated.get_pixel(4, 4)[0] < 0.5);
    }

    #[test]
    fn test_dilate_mask_radius_2() {
        let mut mask_data = vec![[0.0, 0.0, 0.0, 1.0]; 49];
        mask_data[24] = [1.0, 1.0, 1.0, 1.0]; // Center of 7x7

        let mask = ImageField::from_raw(mask_data, 7, 7);
        let dilated = dilate_mask(&mask, 2);

        // Points within radius 2 should be white
        assert!(dilated.get_pixel(3, 3)[0] > 0.5); // center
        assert!(dilated.get_pixel(3, 1)[0] > 0.5); // 2 up
        assert!(dilated.get_pixel(5, 3)[0] > 0.5); // 2 right

        // Corners should still be black
        assert!(dilated.get_pixel(0, 0)[0] < 0.5);
    }

    // ========== Expression-based primitive tests ==========

    #[test]
    fn test_uv_expr_identity() {
        let expr = UvExpr::identity();
        assert_eq!(expr.eval(0.3, 0.7), (0.3, 0.7));
        assert_eq!(expr.eval(0.0, 1.0), (0.0, 1.0));
    }

    #[test]
    fn test_uv_expr_translate() {
        let expr = UvExpr::translate(0.1, -0.2);
        let (u, v) = expr.eval(0.5, 0.5);
        assert!((u - 0.6).abs() < 1e-6);
        assert!((v - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_uv_expr_scale_centered() {
        let expr = UvExpr::scale_centered(2.0, 2.0);
        // At center, should stay the same
        let (u, v) = expr.eval(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);

        // At (0, 0), scales outward from center
        let (u, v) = expr.eval(0.0, 0.0);
        assert!((u - (-0.5)).abs() < 1e-6);
        assert!((v - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_uv_expr_rotate_centered() {
        use std::f32::consts::PI;
        let expr = UvExpr::rotate_centered(PI); // 180 degrees
        // At center, should stay the same
        let (u, v) = expr.eval(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-5);
        assert!((v - 0.5).abs() < 1e-5);

        // At (1, 0.5), should map to (0, 0.5)
        let (u, v) = expr.eval(1.0, 0.5);
        assert!((u - 0.0).abs() < 1e-5);
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_uv_expr_vec2_constructor() {
        // Swap U and V
        let expr = UvExpr::Vec2 {
            x: Box::new(UvExpr::V),
            y: Box::new(UvExpr::U),
        };
        let (u, v) = expr.eval(0.3, 0.7);
        assert!((u - 0.7).abs() < 1e-6);
        assert!((v - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_uv_expr_math_ops() {
        // Test sin
        let sin_expr = UvExpr::Sin(Box::new(UvExpr::Constant(0.0)));
        let (u, _) = sin_expr.eval(0.0, 0.0);
        assert!(u.abs() < 1e-6);

        // Test length
        let len_expr = UvExpr::Length(Box::new(UvExpr::Constant2(3.0, 4.0)));
        let (len, _) = len_expr.eval(0.0, 0.0);
        assert!((len - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_expr_identity() {
        let expr = ColorExpr::identity();
        let result = expr.eval(0.2, 0.4, 0.6, 0.8);
        assert_eq!(result, [0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_color_expr_grayscale() {
        let expr = ColorExpr::grayscale();
        let result = expr.eval(1.0, 0.0, 0.0, 1.0); // Pure red
        // Luminance of pure red = 0.2126
        assert!((result[0] - 0.2126).abs() < 1e-4);
        assert!((result[1] - 0.2126).abs() < 1e-4);
        assert!((result[2] - 0.2126).abs() < 1e-4);
        assert!((result[3] - 1.0).abs() < 1e-6); // Alpha preserved
    }

    #[test]
    fn test_color_expr_invert() {
        let expr = ColorExpr::invert();
        let result = expr.eval(0.2, 0.3, 0.4, 0.9);
        assert!((result[0] - 0.8).abs() < 1e-6);
        assert!((result[1] - 0.7).abs() < 1e-6);
        assert!((result[2] - 0.6).abs() < 1e-6);
        assert!((result[3] - 0.9).abs() < 1e-6); // Alpha preserved
    }

    #[test]
    fn test_color_expr_threshold() {
        let expr = ColorExpr::threshold(0.5);

        // Dark pixel (luminance < 0.5)
        let dark = expr.eval(0.2, 0.2, 0.2, 1.0);
        assert!(dark[0] < 0.01);

        // Bright pixel (luminance > 0.5)
        let bright = expr.eval(0.8, 0.8, 0.8, 1.0);
        assert!(bright[0] > 0.99);
    }

    #[test]
    fn test_color_expr_brightness() {
        let expr = ColorExpr::brightness(2.0);
        let result = expr.eval(0.25, 0.5, 0.125, 1.0);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_color_expr_posterize() {
        let expr = ColorExpr::posterize(4); // 4 levels: 0, 0.33, 0.67, 1.0
        // formula: floor(color * factor) / factor, where factor = 3
        // 0.4 * 3 = 1.2 -> floor = 1 -> 1/3 = 0.333
        // 0.6 * 3 = 1.8 -> floor = 1 -> 1/3 = 0.333
        // 0.9 * 3 = 2.7 -> floor = 2 -> 2/3 = 0.667
        let result = expr.eval(0.4, 0.6, 0.9, 1.0);
        assert!((result[0] - 0.333).abs() < 0.1);
        assert!((result[1] - 0.333).abs() < 0.1);
        assert!((result[2] - 0.667).abs() < 0.1);
    }

    #[test]
    fn test_remap_uv_identity() {
        let data: Vec<_> = (0..16).map(|i| [i as f32 / 15.0, 0.0, 0.0, 1.0]).collect();
        let img = ImageField::from_raw(data.clone(), 4, 4);

        let result = remap_uv(&img, &UvExpr::identity());

        // Should be essentially unchanged
        for i in 0..16 {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let orig = img.get_pixel(x, y)[0];
            let new = result.get_pixel(x, y)[0];
            assert!(
                (orig - new).abs() < 0.01,
                "Pixel ({}, {}) changed: {} -> {}",
                x,
                y,
                orig,
                new
            );
        }
    }

    #[test]
    fn test_map_pixels_identity() {
        let data = vec![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let img = ImageField::from_raw(data.clone(), 2, 1);

        let result = map_pixels(&img, &ColorExpr::identity());

        let p0 = result.get_pixel(0, 0);
        let p1 = result.get_pixel(1, 0);
        assert!((p0[0] - 0.1).abs() < 1e-6);
        assert!((p1[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_map_pixels_grayscale() {
        let data = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
        let img = ImageField::from_raw(data, 2, 1);

        let result = map_pixels(&img, &ColorExpr::grayscale());

        // Red pixel -> luminance 0.2126
        let p0 = result.get_pixel(0, 0);
        assert!((p0[0] - 0.2126).abs() < 1e-4);
        assert!((p0[1] - 0.2126).abs() < 1e-4);

        // Green pixel -> luminance 0.7152
        let p1 = result.get_pixel(1, 0);
        assert!((p1[0] - 0.7152).abs() < 1e-4);
    }

    #[test]
    fn test_lens_distortion_to_uv_expr() {
        let config = LensDistortion::barrel(0.3);
        let expr = config.to_uv_expr();

        // At center, should return center
        let (u, v) = expr.eval(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);

        // Away from center, should be distorted
        let (u, _) = expr.eval(0.8, 0.5);
        assert!(u != 0.8, "Distortion should modify coordinates");
    }

    #[test]
    fn test_wave_distortion_to_uv_expr() {
        // Use default config which has both amplitude_x/y and frequency_x/y set
        let config = WaveDistortion {
            amplitude_x: 0.1,
            amplitude_y: 0.0,
            frequency_x: 0.0,
            frequency_y: 2.0, // This controls the X offset wave
            phase: 0.0,
        };
        let expr = config.to_uv_expr();

        // At V=0, the sine wave should be at phase=0, so offset_x = 0
        let (u, v) = expr.eval(0.5, 0.0);
        assert!((u - 0.5).abs() < 1e-5, "At v=0, u should be ~unchanged");
        assert!((v - 0.0).abs() < 1e-5);

        // At V=0.125 (1/4 cycle for freq=2), sine should be at peak
        // offset_x = 0.1 * sin(0.125 * 2 * 2π) = 0.1 * sin(π/2) = 0.1 * 1 = 0.1
        let (u, _) = expr.eval(0.5, 0.125);
        assert!(
            (u - 0.6).abs() < 0.02,
            "At v=0.125, should have positive offset, got u={}",
            u
        );
    }

    #[test]
    fn test_lens_distortion_uses_remap_uv() {
        // Verify that lens_distortion produces the same result as remap_uv
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);
        let config = LensDistortion::barrel(0.5);

        let result1 = lens_distortion(&img, &config);
        let result2 = remap_uv(&img, &config.to_uv_expr());

        for y in 0..5 {
            for x in 0..5 {
                let p1 = result1.get_pixel(x, y);
                let p2 = result2.get_pixel(x, y);
                assert!((p1[0] - p2[0]).abs() < 1e-6, "Mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_wave_distortion_uses_remap_uv() {
        let data = vec![[0.5, 0.5, 0.5, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);
        let config = WaveDistortion::horizontal(0.05, 3.0);

        let result1 = wave_distortion(&img, &config);
        let result2 = remap_uv(&img, &config.to_uv_expr());

        for y in 0..5 {
            for x in 0..5 {
                let p1 = result1.get_pixel(x, y);
                let p2 = result2.get_pixel(x, y);
                assert!((p1[0] - p2[0]).abs() < 1e-6, "Mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_remap_uv_fn_identity() {
        let img = create_test_image();

        // Identity transform should preserve pixels
        let result = remap_uv_fn(&img, |u, v| (u, v));

        for y in 0..2 {
            for x in 0..2 {
                let orig = img.get_pixel(x, y);
                let new = result.get_pixel(x, y);
                assert!((orig[0] - new[0]).abs() < 1e-6);
                assert!((orig[1] - new[1]).abs() < 1e-6);
                assert!((orig[2] - new[2]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_remap_uv_fn_flip_horizontal() {
        let img = create_test_image().with_filter_mode(FilterMode::Nearest);

        // Flip horizontally: sample from (1-u, v)
        let result = remap_uv_fn(&img, |u, v| (1.0 - u, v));

        // Top-left becomes top-right, etc.
        let tl = result.sample_uv(0.25, 0.25); // Should get what was at (0.75, 0.25)
        let tr = result.sample_uv(0.75, 0.25); // Should get what was at (0.25, 0.25)

        // Original: TL=red, TR=green
        // Flipped: TL=green, TR=red
        assert!(tl.g > 0.5, "Top-left should be green after flip");
        assert!(tr.r > 0.5, "Top-right should be red after flip");
    }

    #[test]
    fn test_map_pixels_fn_identity() {
        let img = create_test_image();

        let result = map_pixels_fn(&img, |pixel| pixel);

        for y in 0..2 {
            for x in 0..2 {
                let orig = img.get_pixel(x, y);
                let new = result.get_pixel(x, y);
                assert_eq!(orig, new);
            }
        }
    }

    #[test]
    fn test_map_pixels_fn_invert() {
        let img = create_test_image();

        let result = map_pixels_fn(&img, |[r, g, b, a]| [1.0 - r, 1.0 - g, 1.0 - b, a]);

        // Red pixel (1, 0, 0) -> Cyan (0, 1, 1)
        let inverted_red = result.get_pixel(0, 0);
        assert!((inverted_red[0] - 0.0).abs() < 1e-6);
        assert!((inverted_red[1] - 1.0).abs() < 1e-6);
        assert!((inverted_red[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_pixels_fn_grayscale() {
        let img = create_test_image();

        let result = map_pixels_fn(&img, |[r, g, b, a]| {
            let lum = r * 0.299 + g * 0.587 + b * 0.114;
            [lum, lum, lum, a]
        });

        // All channels should be equal for each pixel
        for y in 0..2 {
            for x in 0..2 {
                let pixel = result.get_pixel(x, y);
                assert!((pixel[0] - pixel[1]).abs() < 1e-6);
                assert!((pixel[1] - pixel[2]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_map_channel_identity() {
        let img = create_test_image();

        // Identity transform on red channel
        let result = map_channel(&img, Channel::Red, |ch| ch);

        for y in 0..2 {
            for x in 0..2 {
                let orig = img.get_pixel(x, y);
                let new = result.get_pixel(x, y);
                assert_eq!(orig, new);
            }
        }
    }

    #[test]
    fn test_map_channel_invert_red() {
        // Create an image with known red values
        let data = vec![
            [1.0, 0.5, 0.3, 1.0],
            [0.0, 0.5, 0.3, 1.0],
            [0.5, 0.5, 0.3, 1.0],
            [0.25, 0.5, 0.3, 1.0],
        ];
        let img = ImageField::from_raw(data, 2, 2);

        // Invert only the red channel
        let result = map_channel(&img, Channel::Red, |ch| {
            // Invert the channel (which appears in R, G, B of the grayscale)
            map_pixels_fn(&ch, |[r, _, _, a]| [1.0 - r, 1.0 - r, 1.0 - r, a])
        });

        // Red channel should be inverted
        assert!((result.get_pixel(0, 0)[0] - 0.0).abs() < 1e-6); // 1.0 -> 0.0
        assert!((result.get_pixel(1, 0)[0] - 1.0).abs() < 1e-6); // 0.0 -> 1.0
        assert!((result.get_pixel(0, 1)[0] - 0.5).abs() < 1e-6); // 0.5 -> 0.5

        // Green and blue should be unchanged
        assert!((result.get_pixel(0, 0)[1] - 0.5).abs() < 1e-6);
        assert!((result.get_pixel(0, 0)[2] - 0.3).abs() < 1e-6);
    }

    // ========================================================================
    // Colorspace conversion tests
    // ========================================================================

    fn assert_rgb_close(a: (f32, f32, f32), b: (f32, f32, f32), epsilon: f32) {
        assert!(
            (a.0 - b.0).abs() < epsilon
                && (a.1 - b.1).abs() < epsilon
                && (a.2 - b.2).abs() < epsilon,
            "RGB values differ: {:?} vs {:?}",
            a,
            b
        );
    }

    #[test]
    fn test_hwb_roundtrip() {
        let test_colors = [
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (1.0, 1.0, 1.0), // White
            (0.0, 0.0, 0.0), // Black
            (0.5, 0.5, 0.5), // Gray
            (0.8, 0.4, 0.2), // Orange-ish
        ];

        for (r, g, b) in test_colors {
            let (h, w, b_val) = rgb_to_hwb(r, g, b);
            let (r2, g2, b2) = hwb_to_rgb(h, w, b_val);
            assert_rgb_close((r, g, b), (r2, g2, b2), 0.01);
        }
    }

    #[test]
    fn test_lch_roundtrip() {
        let test_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.8, 0.4, 0.2),
        ];

        for (r, g, b) in test_colors {
            let (l, c, h) = rgb_to_lch(r, g, b);
            let (r2, g2, b2) = lch_to_rgb(l, c, h);
            assert_rgb_close((r, g, b), (r2, g2, b2), 0.02);
        }
    }

    #[test]
    fn test_oklab_roundtrip() {
        let test_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.8, 0.4, 0.2),
        ];

        for (r, g, b) in test_colors {
            let (l, a, b_val) = rgb_to_oklab(r, g, b);
            let (r2, g2, b2) = oklab_to_rgb(l, a, b_val);
            assert_rgb_close((r, g, b), (r2, g2, b2), 0.01);
        }
    }

    #[test]
    fn test_oklch_roundtrip() {
        let test_colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.8, 0.4, 0.2),
        ];

        for (r, g, b) in test_colors {
            let (l, c, h) = rgb_to_oklch(r, g, b);
            let (r2, g2, b2) = oklch_to_rgb(l, c, h);
            assert_rgb_close((r, g, b), (r2, g2, b2), 0.02);
        }
    }

    #[test]
    fn test_decompose_reconstruct_all_colorspaces() {
        let img = ImageField::solid_sized(4, 4, [0.8, 0.4, 0.2, 0.9]);

        for colorspace in [
            Colorspace::Rgb,
            Colorspace::Hsl,
            Colorspace::Hsv,
            Colorspace::Hwb,
            Colorspace::YCbCr,
            Colorspace::Lab,
            Colorspace::Lch,
            Colorspace::OkLab,
            Colorspace::OkLch,
        ] {
            let channels = decompose_colorspace(&img, colorspace);
            let reconstructed = reconstruct_colorspace(&channels);

            // Check that roundtrip preserves the image
            let orig = img.get_pixel(0, 0);
            let result = reconstructed.get_pixel(0, 0);

            assert!(
                (orig[0] - result[0]).abs() < 0.02
                    && (orig[1] - result[1]).abs() < 0.02
                    && (orig[2] - result[2]).abs() < 0.02
                    && (orig[3] - result[3]).abs() < 0.001,
                "Colorspace {:?} roundtrip failed: {:?} vs {:?}",
                colorspace,
                orig,
                result
            );
        }
    }

    #[test]
    fn test_decompose_rgb_is_identity() {
        let img = create_test_image();
        let channels = decompose_colorspace(&img, Colorspace::Rgb);

        // c0 should be red channel, c1 green, c2 blue
        let red_pixel = channels.c0.get_pixel(0, 0);
        assert!((red_pixel[0] - 1.0).abs() < 0.001); // First pixel is red

        let green_pixel = channels.c1.get_pixel(1, 0);
        assert!((green_pixel[0] - 1.0).abs() < 0.001); // Second pixel is green
    }

    #[test]
    fn test_color_expr_colorspace_roundtrip() {
        // Test that ColorExpr::RgbToHsl followed by HslToRgb is identity
        let expr = ColorExpr::HslToRgb(Box::new(ColorExpr::RgbToHsl(Box::new(ColorExpr::Rgba))));

        let test_colors = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.8, 0.4, 0.2, 0.9),
        ];

        for (r, g, b, a) in test_colors {
            let [r2, g2, b2, a2] = expr.eval(r, g, b, a);
            assert!(
                (r - r2).abs() < 0.01
                    && (g - g2).abs() < 0.01
                    && (b - b2).abs() < 0.01
                    && (a - a2).abs() < 0.001,
                "ColorExpr HSL roundtrip failed: ({}, {}, {}, {}) vs ({}, {}, {}, {})",
                r,
                g,
                b,
                a,
                r2,
                g2,
                b2,
                a2
            );
        }
    }

    #[test]
    fn test_color_expr_oklab_roundtrip() {
        let expr =
            ColorExpr::OklabToRgb(Box::new(ColorExpr::RgbToOklab(Box::new(ColorExpr::Rgba))));

        let [r, g, b, a] = expr.eval(0.8, 0.4, 0.2, 1.0);
        assert!((r - 0.8).abs() < 0.01);
        assert!((g - 0.4).abs() < 0.01);
        assert!((b - 0.2).abs() < 0.01);
        assert!((a - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_color_expr_preserves_alpha() {
        // Alpha should be preserved through colorspace conversions
        let expr = ColorExpr::RgbToHsl(Box::new(ColorExpr::Rgba));

        let [_, _, _, a] = expr.eval(0.5, 0.5, 0.5, 0.7);
        assert!((a - 0.7).abs() < 0.001, "Alpha not preserved: {}", a);
    }

    #[cfg(feature = "dew")]
    #[test]
    fn test_colorspace_dew_registration() {
        use rhizome_dew_linalg::{Value, linalg_registry};

        let mut registry = linalg_registry::<f32>();
        register_colorspace(&mut registry);

        // Check that functions are registered
        assert!(registry.get("rgb_to_hsl").is_some());
        assert!(registry.get("hsl_to_rgb").is_some());
        assert!(registry.get("rgb_to_oklab").is_some());
        assert!(registry.get("oklab_to_rgb").is_some());
        assert!(registry.get("rgb_to_hwb").is_some());
        assert!(registry.get("hwb_to_rgb").is_some());
    }

    #[cfg(feature = "dew")]
    #[test]
    fn test_colorspace_dew_eval() {
        use rhizome_dew_linalg::{LinalgFn, Value, linalg_registry};

        let mut registry = linalg_registry::<f32>();
        register_colorspace(&mut registry);

        // Test rgb_to_hsl function directly
        let rgb_to_hsl_fn = registry.get("rgb_to_hsl").unwrap();
        let result = rgb_to_hsl_fn.call(&[Value::Vec4([1.0, 0.0, 0.0, 1.0])]);

        if let Value::Vec4([h, s, l, a]) = result {
            // Pure red: H=0, S=1, L=0.5
            assert!(h.abs() < 0.01, "Hue should be ~0 for red, got {}", h);
            assert!(
                (s - 1.0).abs() < 0.01,
                "Saturation should be ~1 for red, got {}",
                s
            );
            assert!(
                (l - 0.5).abs() < 0.01,
                "Lightness should be ~0.5 for red, got {}",
                l
            );
            assert!((a - 1.0).abs() < 0.001, "Alpha should be preserved");
        } else {
            panic!("Expected Vec4 result");
        }
    }
}

// ============================================================================
// Invariant tests - statistical property validation
// ============================================================================

#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    // =========================================================================
    // Blue noise distribution tests
    // =========================================================================

    /// Compute autocorrelation at a given lag for 1D data
    fn autocorrelation_1d(values: &[f32], lag: usize) -> f32 {
        if lag >= values.len() {
            return 0.0;
        }
        let n = values.len() - lag;
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;

        let mut cov = 0.0f32;
        let mut var = 0.0f32;

        for i in 0..n {
            let a = values[i] - mean;
            let b = values[i + lag] - mean;
            cov += a * b;
        }

        for v in values {
            let d = v - mean;
            var += d * d;
        }

        if var < 1e-10 {
            return 0.0;
        }

        cov / var
    }

    /// Compute 2D autocorrelation at a given (dx, dy) pixel offset
    fn autocorrelation_2d(image: &ImageField, dx: i32, dy: i32) -> f32 {
        use rhizome_resin_field::Field;

        let (width, height) = image.dimensions();
        let ctx = rhizome_resin_field::EvalContext::default();

        let mut sum_product = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut mean = 0.0f32;
        let mut count = 0;

        // First pass: compute mean using normalized UV coordinates
        for y in 0..height {
            for x in 0..width {
                let uv = Vec2::new(
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                );
                let color: Rgba = image.sample(uv, &ctx);
                mean += color.r;
                count += 1;
            }
        }
        mean /= count as f32;

        // Second pass: compute autocorrelation
        for y in 0..height {
            for x in 0..width {
                let nx = ((x as i32 + dx) % width as i32 + width as i32) % width as i32;
                let ny = ((y as i32 + dy) % height as i32 + height as i32) % height as i32;

                let uv1 = Vec2::new(
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                );
                let uv2 = Vec2::new(
                    (nx as f32 + 0.5) / width as f32,
                    (ny as f32 + 0.5) / height as f32,
                );

                let c1: Rgba = image.sample(uv1, &ctx);
                let c2: Rgba = image.sample(uv2, &ctx);
                let v1 = c1.r - mean;
                let v2 = c2.r - mean;

                sum_product += v1 * v2;
                sum_sq += v1 * v1;
            }
        }

        if sum_sq < 1e-10 {
            return 0.0;
        }

        sum_product / sum_sq
    }

    #[test]
    fn test_blue_noise_1d_negative_autocorrelation() {
        // Blue noise should have negative autocorrelation at lag 1
        // (nearby values should be anti-correlated)
        let noise = generate_blue_noise_1d(256);

        let ac1 = autocorrelation_1d(&noise, 1);
        let ac2 = autocorrelation_1d(&noise, 2);

        // Blue noise should have negative or near-zero autocorrelation
        // at small lags (values spread out, not clumped)
        assert!(
            ac1 < 0.1,
            "Blue noise 1D autocorrelation(1) should be negative or near-zero, got {}",
            ac1
        );
        assert!(
            ac2 < 0.2,
            "Blue noise 1D autocorrelation(2) should be low, got {}",
            ac2
        );
    }

    #[test]
    fn test_blue_noise_2d_negative_autocorrelation() {
        // Blue noise should have negative autocorrelation at small offsets
        let noise = generate_blue_noise_2d(32);

        let ac_10 = autocorrelation_2d(&noise, 1, 0);
        let ac_01 = autocorrelation_2d(&noise, 0, 1);
        let ac_11 = autocorrelation_2d(&noise, 1, 1);

        // Blue noise should have negative or near-zero autocorrelation
        assert!(
            ac_10 < 0.1,
            "Blue noise 2D autocorrelation(1,0) should be low, got {}",
            ac_10
        );
        assert!(
            ac_01 < 0.1,
            "Blue noise 2D autocorrelation(0,1) should be low, got {}",
            ac_01
        );
        assert!(
            ac_11 < 0.2,
            "Blue noise 2D autocorrelation(1,1) should be low, got {}",
            ac_11
        );
    }

    #[test]
    fn test_blue_noise_uniform_distribution() {
        // Blue noise should be uniformly distributed in [0, 1]
        let noise = generate_blue_noise_1d(1024);

        let mean: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
        let variance: f32 =
            noise.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / noise.len() as f32;

        // Uniform [0,1] has mean=0.5, variance=1/12≈0.0833
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Blue noise mean should be ~0.5, got {}",
            mean
        );
        assert!(
            (variance - 0.0833).abs() < 0.02,
            "Blue noise variance should be ~0.083, got {}",
            variance
        );
    }

    #[test]
    fn test_blue_noise_2d_uniform_distribution() {
        use rhizome_resin_field::Field;

        let noise = generate_blue_noise_2d(32);
        let (width, height) = noise.dimensions();
        let ctx = rhizome_resin_field::EvalContext::default();

        let mut values = Vec::new();
        for y in 0..height {
            for x in 0..width {
                // Use normalized UV coordinates [0, 1]
                let uv = Vec2::new(
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                );
                let color: Rgba = noise.sample(uv, &ctx);
                values.push(color.r);
            }
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        // Blue noise from void-and-cluster may not have perfect uniform mean
        // due to the ranking algorithm, but should be reasonably close
        assert!(
            (mean - 0.5).abs() < 0.15,
            "Blue noise 2D mean should be ~0.5, got {}",
            mean
        );
        // Variance should be moderate (not all same value, not extreme)
        assert!(
            variance > 0.01 && variance < 0.20,
            "Blue noise 2D variance should be moderate, got {}",
            variance
        );
    }

    // =========================================================================
    // Blur kernel tests
    // =========================================================================

    #[test]
    fn test_blur_kernels_sum_to_one() {
        // All blur kernels should sum to 1 to preserve brightness
        let kernels = [
            ("box_blur", Kernel::box_blur()),
            ("gaussian_3x3", Kernel::gaussian_blur_3x3()),
            ("gaussian_5x5", Kernel::gaussian_blur_5x5()),
        ];

        for (name, kernel) in kernels {
            let sum: f32 = kernel.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "{} kernel should sum to 1.0, got {}",
                name,
                sum
            );
        }
    }

    #[test]
    fn test_blur_preserves_uniform_image() {
        use rhizome_resin_field::Field;

        // Blurring a uniform image should not change it
        let uniform_value = 0.42f32;
        let data = vec![[uniform_value, uniform_value, uniform_value, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let blurred = convolve(&img, &Kernel::gaussian_blur_3x3());
        let ctx = rhizome_resin_field::EvalContext::default();

        // Check center pixel using normalized UV
        let center: Rgba = blurred.sample(Vec2::new(0.5, 0.5), &ctx);
        assert!(
            (center.r - uniform_value).abs() < 0.01,
            "Blur should preserve uniform image, got {} instead of {}",
            center.r,
            uniform_value
        );
    }

    #[test]
    fn test_blur_reduces_variance() {
        // Blurring should reduce variance (smooth the image)
        // Create a noisy image with actual variation
        let data: Vec<[f32; 4]> = (0..64)
            .map(|i| {
                let v = ((i * 7919) % 256) as f32 / 255.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 8, 8);

        let blurred = blur(&img, 3);
        let ctx = rhizome_resin_field::EvalContext::default();

        // Compute variance of original and blurred using normalized UVs
        fn compute_variance(img: &ImageField, ctx: &rhizome_resin_field::EvalContext) -> f32 {
            use rhizome_resin_field::Field;
            let (w, h) = img.dimensions();
            let mut values = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let uv = Vec2::new((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let color: Rgba = img.sample(uv, ctx);
                    values.push(color.r);
                }
            }
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
        }

        let var_original = compute_variance(&img, &ctx);
        let var_blurred = compute_variance(&blurred, &ctx);

        assert!(
            var_blurred < var_original,
            "Blur should reduce variance: original={}, blurred={}",
            var_original,
            var_blurred
        );
    }

    // =========================================================================
    // Dithering tests
    // =========================================================================

    #[test]
    fn test_dither_preserves_average_brightness() {
        use rhizome_resin_field::Field;

        // Dithering should approximately preserve average brightness
        // Using a 16x16 gray image to get better sampling of Bayer pattern
        let gray_level = 0.4f32;
        let data = vec![[gray_level, gray_level, gray_level, 1.0]; 256];
        let img = ImageField::from_raw(data, 16, 16);
        let bayer = BayerField::bayer4x4();

        let dithered = QuantizeWithThreshold::new(img.clone(), bayer, 2);
        let ctx = rhizome_resin_field::EvalContext::default();

        // BayerField uses UV * 1000, so 0.001 UV step = 1 Bayer pixel
        // Sample at UV coords that align with Bayer pattern
        let mut sum = 0.0f32;
        let mut count = 0;
        for y in 0..16 {
            for x in 0..16 {
                // Use Bayer-aligned coordinates
                let uv = Vec2::new(x as f32 * 0.001, y as f32 * 0.001);
                let color: Rgba = dithered.sample(uv, &ctx);
                sum += color.r;
                count += 1;
            }
        }
        let avg = sum / count as f32;

        // Allow tolerance since dithering is discrete (binary 0/1 outputs)
        assert!(
            (avg - gray_level).abs() < 0.2,
            "Dithered average brightness should be ~{}, got {}",
            gray_level,
            avg
        );
    }

    #[test]
    fn test_dither_produces_binary_output() {
        use rhizome_resin_field::Field;

        // Quantize to 2 levels should produce only 0 or 1
        let data: Vec<[f32; 4]> = (0..64)
            .map(|i| {
                let v = i as f32 / 64.0;
                [v, v, v, 1.0]
            })
            .collect();
        let img = ImageField::from_raw(data, 8, 8);
        let bayer = BayerField::bayer4x4();

        let dithered = QuantizeWithThreshold::new(img, bayer, 2);
        let ctx = rhizome_resin_field::EvalContext::default();

        for y in 0..8 {
            for x in 0..8 {
                let uv = Vec2::new((x as f32 + 0.5) / 8.0, (y as f32 + 0.5) / 8.0);
                let color: Rgba = dithered.sample(uv, &ctx);
                assert!(
                    color.r == 0.0 || color.r == 1.0,
                    "Binary dither should produce 0 or 1, got {} at ({}, {})",
                    color.r,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_bayer_field_range() {
        use rhizome_resin_field::Field;

        // Bayer field values should be in [0, 1)
        // BayerField tiles at UV * 1000.0, so sample at small UV steps
        let bayer = BayerField::bayer8x8();
        let ctx = rhizome_resin_field::EvalContext::default();

        for y in 0..8 {
            for x in 0..8 {
                // BayerField converts UV to pixels via * 1000, then mods by size
                // So 0.001 UV step = 1 pixel step
                let uv = Vec2::new(x as f32 * 0.001, y as f32 * 0.001);
                let v: f32 = bayer.sample(uv, &ctx);
                assert!(
                    v >= 0.0 && v < 1.0,
                    "Bayer value should be in [0, 1), got {} at ({}, {})",
                    v,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_bayer_field_unique_values() {
        use rhizome_resin_field::Field;

        // Each value in an nxn Bayer matrix should be unique within the tile
        // BayerField converts UV to pixels via * 1000, then mods by size
        let bayer = BayerField::bayer4x4();
        let ctx = rhizome_resin_field::EvalContext::default();

        let mut values: Vec<f32> = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                // 0.001 UV step = 1 pixel step in Bayer
                let uv = Vec2::new(x as f32 * 0.001, y as f32 * 0.001);
                values.push(bayer.sample(uv, &ctx));
            }
        }

        // Sort and check for duplicates
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..values.len() {
            assert!(
                (values[i] - values[i - 1]).abs() > 1e-6,
                "Bayer matrix should have unique values, found duplicates: {:?}",
                values
            );
        }
    }

    // =========================================================================
    // Color transform invertibility tests
    // =========================================================================

    #[test]
    fn test_grayscale_idempotent() {
        use rhizome_resin_field::Field;

        // Applying grayscale twice should give the same result
        let data = vec![
            [0.2, 0.5, 0.8, 1.0],
            [0.1, 0.9, 0.3, 1.0],
            [0.7, 0.2, 0.6, 1.0],
            [0.4, 0.4, 0.4, 1.0],
        ];
        let img = ImageField::from_raw(data, 2, 2);

        let gray1 = grayscale(&img);
        let gray2 = grayscale(&gray1);

        let ctx = rhizome_resin_field::EvalContext::default();
        for y in 0..2 {
            for x in 0..2 {
                let uv = Vec2::new((x as f32 + 0.5) / 2.0, (y as f32 + 0.5) / 2.0);
                let v1: Rgba = gray1.sample(uv, &ctx);
                let v2: Rgba = gray2.sample(uv, &ctx);
                assert!(
                    (v1.r - v2.r).abs() < 1e-5,
                    "Grayscale should be idempotent at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_invert_is_involution() {
        use rhizome_resin_field::Field;

        // Inverting twice should give the original
        let data = vec![
            [0.2, 0.5, 0.8, 1.0],
            [0.1, 0.9, 0.3, 1.0],
            [0.7, 0.2, 0.6, 1.0],
            [0.0, 1.0, 0.5, 1.0],
        ];
        let img = ImageField::from_raw(data.clone(), 2, 2);

        let inv1 = invert(&img);
        let inv2 = invert(&inv1);

        let ctx = rhizome_resin_field::EvalContext::default();
        for y in 0..2 {
            for x in 0..2 {
                let uv = Vec2::new((x as f32 + 0.5) / 2.0, (y as f32 + 0.5) / 2.0);
                let original: Rgba = img.sample(uv, &ctx);
                let double_inv: Rgba = inv2.sample(uv, &ctx);
                assert!(
                    (original.r - double_inv.r).abs() < 1e-5,
                    "Double invert should restore original R at ({}, {})",
                    x,
                    y
                );
                assert!(
                    (original.g - double_inv.g).abs() < 1e-5,
                    "Double invert should restore original G at ({}, {})",
                    x,
                    y
                );
                assert!(
                    (original.b - double_inv.b).abs() < 1e-5,
                    "Double invert should restore original B at ({}, {})",
                    x,
                    y
                );
            }
        }
    }
}
