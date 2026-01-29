//! Image-based fields for texture sampling.
//!
//! Provides `ImageField` which loads an image and exposes it as a `Field<Vec2, Color>`.
//!
//! # Example
//!
//! ```ignore
//! use unshape_image::{ImageField, WrapMode, FilterMode};
//! use unshape_field::{Field, EvalContext};
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

pub use unshape_color::BlendMode;
use unshape_color::Rgba;
use unshape_field::{EvalContext, Field};

mod adjust;
mod bake;
mod channel;
mod colorspace;
mod composite;
mod distort;
mod dither;
mod effects;
mod expr;
mod freq;
mod glitch;
mod inpaint;
mod int_ops;
mod kernel;
mod normal_map;
mod pyramid;
mod transform;

pub use adjust::*;
pub use bake::*;
pub use channel::*;
pub use colorspace::*;
pub use composite::*;
pub use distort::*;
pub use dither::*;
pub use effects::*;
pub use expr::*;
pub use freq::*;
pub use glitch::*;
pub use inpaint::*;
pub use int_ops::*;
pub use kernel::*;
pub use normal_map::*;
pub use pyramid::*;
pub use transform::*;

// Re-import pub(crate) items needed by tests
#[cfg(test)]
use dither::hilbert_d2xy;

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
    pub(crate) data: Vec<[f32; 4]>,
    /// Image width in pixels.
    pub(crate) width: u32,
    /// Image height in pixels.
    pub(crate) height: u32,
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

/// Registers all image operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of image ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<BakeConfig>("resin::BakeConfig");
    registry.register_type::<AnimationConfig>("resin::AnimationConfig");
    registry.register_type::<ChromaticAberration>("resin::ChromaticAberration");
    registry.register_type::<Levels>("resin::Levels");
    registry.register_type::<LensDistortion>("resin::LensDistortion");
    registry.register_type::<WaveDistortion>("resin::WaveDistortion");
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
    fn test_color_expr_brightness_contrast() {
        // No change: brightness=0, contrast=0
        let expr = ColorExpr::brightness_contrast(0.0, 0.0);
        let result = expr.eval(0.5, 0.5, 0.5, 1.0);
        assert!((result[0] - 0.5).abs() < 1e-6);

        // Brightness only: add 0.2 to each channel
        let expr = ColorExpr::brightness_contrast(0.2, 0.0);
        let result = expr.eval(0.5, 0.5, 0.5, 1.0);
        assert!((result[0] - 0.7).abs() < 1e-6);

        // Contrast only: midpoint unchanged, edges amplified
        // contrast = 0.5 means factor = 1.5
        // (0.25 - 0.5) * 1.5 + 0.5 = -0.375 + 0.5 = 0.125
        let expr = ColorExpr::brightness_contrast(0.0, 0.5);
        let result = expr.eval(0.25, 0.5, 0.75, 1.0);
        assert!((result[0] - 0.125).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6); // midpoint unchanged
        assert!((result[2] - 0.875).abs() < 1e-6);

        // Alpha should be preserved
        assert!((result[3] - 1.0).abs() < 1e-6);
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
    fn test_color_expr_matrix() {
        // Identity matrix - no change
        let identity = ColorExpr::matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let result = identity.eval(0.5, 0.3, 0.8, 1.0);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.3).abs() < 1e-6);
        assert!((result[2] - 0.8).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);

        // Grayscale matrix (luminance)
        let gray = ColorExpr::matrix([
            [0.299, 0.587, 0.114, 0.0],
            [0.299, 0.587, 0.114, 0.0],
            [0.299, 0.587, 0.114, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let result = gray.eval(1.0, 0.0, 0.0, 1.0);
        assert!((result[0] - 0.299).abs() < 1e-6);
        assert!((result[1] - 0.299).abs() < 1e-6);
        assert!((result[2] - 0.299).abs() < 1e-6);

        // Channel swap: RGB -> BGR
        let swap = ColorExpr::matrix([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let result = swap.eval(0.2, 0.5, 0.8, 1.0);
        assert!((result[0] - 0.8).abs() < 1e-6); // B -> R
        assert!((result[1] - 0.5).abs() < 1e-6); // G -> G
        assert!((result[2] - 0.2).abs() < 1e-6); // R -> B
    }

    #[test]
    fn test_color_matrix_function() {
        use glam::Mat4;

        let data = vec![[1.0, 0.5, 0.25, 1.0]; 4];
        let img = ImageField::from_raw(data, 2, 2);

        // Identity matrix
        let result = color_matrix(&img, Mat4::IDENTITY);
        let pixel = result.get_pixel(0, 0);
        assert!((pixel[0] - 1.0).abs() < 1e-6);
        assert!((pixel[1] - 0.5).abs() < 1e-6);
        assert!((pixel[2] - 0.25).abs() < 1e-6);

        // Grayscale using Mat4
        let grayscale = Mat4::from_cols_array(&[
            0.299, 0.299, 0.299, 0.0, // column 0
            0.587, 0.587, 0.587, 0.0, // column 1
            0.114, 0.114, 0.114, 0.0, // column 2
            0.0, 0.0, 0.0, 1.0, // column 3
        ]);
        let result = color_matrix(&img, grayscale);
        let pixel = result.get_pixel(0, 0);
        let expected_lum = 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.25;
        assert!((pixel[0] - expected_lum).abs() < 1e-5);
        assert!((pixel[1] - expected_lum).abs() < 1e-5);
        assert!((pixel[2] - expected_lum).abs() < 1e-5);
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
            map_pixels(&ch, &ColorExpr::invert())
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

    // =========================================================================
    // Datamosh tests
    // =========================================================================

    #[test]
    fn test_datamosh_preserves_dimensions() {
        let img = ImageField::solid_sized(32, 24, [0.5, 0.5, 0.5, 1.0]);
        let config = Datamosh::default();
        let result = datamosh(&img, &config);

        assert_eq!(result.width, 32);
        assert_eq!(result.height, 24);
    }

    #[test]
    fn test_datamosh_with_zero_iterations() {
        let img = ImageField::solid_sized(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let config = Datamosh::new(0);
        let result = datamosh(&img, &config);

        // With zero iterations, should be close to original
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_datamosh_various_patterns() {
        let img = ImageField::solid_sized(32, 32, [0.5, 0.3, 0.7, 1.0]);

        for pattern in [
            MotionPattern::Random,
            MotionPattern::Directional,
            MotionPattern::Radial,
            MotionPattern::Vortex,
            MotionPattern::Brightness,
        ] {
            let config = Datamosh::new(2).pattern(pattern);
            let result = datamosh(&img, &config);
            assert_eq!(result.width, 32);
            assert_eq!(result.height, 32);
        }
    }

    #[test]
    fn test_datamosh_builder_pattern() {
        let img = ImageField::solid_sized(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let config = Datamosh::new(3)
            .block_size(8)
            .intensity(0.7)
            .decay(0.8)
            .freeze(0.2)
            .seed(12345)
            .pattern(MotionPattern::Radial);

        let result = config.apply(&img);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_datamosh_frames_preserves_dimensions() {
        let frame1 = ImageField::solid_sized(16, 16, [0.3, 0.4, 0.5, 1.0]);
        let frame2 = ImageField::solid_sized(16, 16, [0.5, 0.6, 0.7, 1.0]);
        let config = Datamosh::new(1);

        let result = datamosh_frames(&frame1, &frame2, &config);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }

    #[test]
    fn test_datamosh_frames_size_mismatch_fallback() {
        let frame1 = ImageField::solid_sized(16, 16, [0.3, 0.4, 0.5, 1.0]);
        let frame2 = ImageField::solid_sized(32, 32, [0.5, 0.6, 0.7, 1.0]);
        let config = Datamosh::default();

        // Should fallback to single-frame datamosh on frame2
        let result = datamosh_frames(&frame1, &frame2, &config);
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
    }

    #[test]
    fn test_datamosh_reproducible() {
        let img = ImageField::solid_sized(32, 32, [0.5, 0.3, 0.7, 1.0]);

        let config1 = Datamosh::new(3).seed(42);
        let config2 = Datamosh::new(3).seed(42);

        let result1 = datamosh(&img, &config1);
        let result2 = datamosh(&img, &config2);

        // Same seed should produce same result
        for (p1, p2) in result1.data.iter().zip(result2.data.iter()) {
            assert!((p1[0] - p2[0]).abs() < 1e-6);
            assert!((p1[1] - p2[1]).abs() < 1e-6);
            assert!((p1[2] - p2[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_datamosh_different_seeds() {
        // Create a gradient image so there's actual variation
        let mut data = Vec::new();
        for y in 0..32 {
            for x in 0..32 {
                let r = x as f32 / 31.0;
                let g = y as f32 / 31.0;
                let b = 0.5;
                data.push([r, g, b, 1.0]);
            }
        }
        let img = ImageField::from_raw(data, 32, 32);

        let config1 = Datamosh::new(3).seed(42).intensity(0.8);
        let config2 = Datamosh::new(3).seed(999).intensity(0.8);

        let result1 = datamosh(&img, &config1);
        let result2 = datamosh(&img, &config2);

        // Different seeds should produce different results
        // (at least some pixels should differ)
        let mut any_different = false;
        for (p1, p2) in result1.data.iter().zip(result2.data.iter()) {
            if (p1[0] - p2[0]).abs() > 0.001
                || (p1[1] - p2[1]).abs() > 0.001
                || (p1[2] - p2[2]).abs() > 0.001
            {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different results"
        );
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
        use unshape_field::Field;

        let (width, height) = image.dimensions();
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

        let noise = generate_blue_noise_2d(32);
        let (width, height) = noise.dimensions();
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

        // Blurring a uniform image should not change it
        let uniform_value = 0.42f32;
        let data = vec![[uniform_value, uniform_value, uniform_value, 1.0]; 25];
        let img = ImageField::from_raw(data, 5, 5);

        let blurred = convolve(&img, &Kernel::gaussian_blur_3x3());
        let ctx = unshape_field::EvalContext::default();

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
        let ctx = unshape_field::EvalContext::default();

        // Compute variance of original and blurred using normalized UVs
        fn compute_variance(img: &ImageField, ctx: &unshape_field::EvalContext) -> f32 {
            use unshape_field::Field;
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
        use unshape_field::Field;

        // Dithering should approximately preserve average brightness
        // Using a 16x16 gray image to get better sampling of Bayer pattern
        let gray_level = 0.4f32;
        let data = vec![[gray_level, gray_level, gray_level, 1.0]; 256];
        let img = ImageField::from_raw(data, 16, 16);
        let bayer = BayerField::bayer4x4();

        let dithered = QuantizeWithThreshold::new(img.clone(), bayer, 2);
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

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
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

        // Bayer field values should be in [0, 1)
        // BayerField tiles at UV * 1000.0, so sample at small UV steps
        let bayer = BayerField::bayer8x8();
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

        // Each value in an nxn Bayer matrix should be unique within the tile
        // BayerField converts UV to pixels via * 1000, then mods by size
        let bayer = BayerField::bayer4x4();
        let ctx = unshape_field::EvalContext::default();

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
        use unshape_field::Field;

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

        let ctx = unshape_field::EvalContext::default();
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
        use unshape_field::Field;

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

        let ctx = unshape_field::EvalContext::default();
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
