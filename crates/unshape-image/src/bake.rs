use std::path::Path;

use glam::{Vec2, Vec3, Vec4};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_color::Rgba;
use unshape_field::{EvalContext, Field};

use crate::{ImageField, ImageFieldError};

/// Configuration for texture baking.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
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
/// use unshape_image::{bake_scalar, BakeConfig};
/// use unshape_field::{Perlin2D, Field, EvalContext};
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
/// use unshape_image::{bake_rgba, BakeConfig};
/// use unshape_field::{Field, EvalContext};
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
/// use unshape_image::{bake_scalar, BakeConfig, export_png};
/// use unshape_field::{Perlin2D, EvalContext};
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

/// Configuration for animation rendering.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
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
/// use unshape_image::{render_animation, AnimationConfig};
///
/// let config = AnimationConfig::new(256, 256, 60).with_fps(30.0);
/// let frames = render_animation(&my_field, &config);
/// ```
pub fn render_animation<F: Field<Vec3, Rgba>>(
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

impl<'a, F: Field<Vec3, Rgba>> Field<Vec2, Rgba> for TimeSliceField<'a, F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> Rgba {
        self.inner
            .sample(Vec3::new(input.x, input.y, self.time), ctx)
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
pub fn render_animation_scalar<F: Field<Vec3, f32>>(
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

impl<'a, F: Field<Vec3, f32>> Field<Vec2, f32> for TimeSliceFieldScalar<'a, F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        self.inner
            .sample(Vec3::new(input.x, input.y, self.time), ctx)
    }
}
