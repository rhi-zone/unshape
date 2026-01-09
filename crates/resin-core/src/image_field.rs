//! Image-based fields for texture sampling.
//!
//! Provides `ImageField` which loads an image and exposes it as a `Field<Vec2, Color>`.
//!
//! # Example
//!
//! ```ignore
//! use resin_core::{ImageField, Field, EvalContext, WrapMode, FilterMode};
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

use crate::context::EvalContext;
use crate::field::Field;
use resin_color::Rgba;

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

        // Sample corners
        let tl = img.sample_uv(0.0, 0.0);
        assert!((tl.r - 1.0).abs() < 0.001); // Red

        let tr = img.sample_uv(0.99, 0.0);
        assert!((tr.g - 1.0).abs() < 0.001); // Green

        let bl = img.sample_uv(0.0, 0.99);
        assert!((bl.b - 1.0).abs() < 0.001); // Blue

        let br = img.sample_uv(0.99, 0.99);
        assert!((br.r - 1.0).abs() < 0.001); // White
        assert!((br.g - 1.0).abs() < 0.001);
        assert!((br.b - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bilinear_sampling() {
        let img = create_test_image().with_filter_mode(FilterMode::Bilinear);

        // Sample center - should be mix of all four colors
        let center = img.sample_uv(0.5, 0.5);
        // All channels should be mixed
        assert!(center.r > 0.1 && center.r < 0.9);
        assert!(center.g > 0.1 && center.g < 0.9);
        assert!(center.b > 0.1 && center.b < 0.9);
    }

    #[test]
    fn test_wrap_repeat() {
        let img = create_test_image()
            .with_filter_mode(FilterMode::Nearest)
            .with_wrap_mode(WrapMode::Repeat);

        // UV > 1.0 should wrap
        let at_1_0 = img.sample_uv(0.0, 0.0);
        let at_2_0 = img.sample_uv(1.0, 0.0);
        assert!((at_1_0.r - at_2_0.r).abs() < 0.001);
    }

    #[test]
    fn test_wrap_clamp() {
        let img = create_test_image()
            .with_filter_mode(FilterMode::Nearest)
            .with_wrap_mode(WrapMode::Clamp);

        // UV > 1.0 should clamp to edge
        let at_1 = img.sample_uv(0.99, 0.99);
        let at_2 = img.sample_uv(2.0, 2.0);
        assert!((at_1.r - at_2.r).abs() < 0.001);
        assert!((at_1.g - at_2.g).abs() < 0.001);
    }

    #[test]
    fn test_wrap_mirror() {
        let img = create_test_image()
            .with_filter_mode(FilterMode::Nearest)
            .with_wrap_mode(WrapMode::Mirror);

        // Test that mirroring works
        let at_0_5 = img.sample_uv(0.25, 0.0);
        let at_1_5 = img.sample_uv(1.75, 0.0); // Should mirror back
        assert!((at_0_5.r - at_1_5.r).abs() < 0.001);
    }

    #[test]
    fn test_solid_color() {
        let color = Rgba::new(0.5, 0.3, 0.8, 1.0);
        let img = ImageField::solid(color);

        let sampled = img.sample_uv(0.5, 0.5);
        assert!((sampled.r - 0.5).abs() < 0.001);
        assert!((sampled.g - 0.3).abs() < 0.001);
        assert!((sampled.b - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_field_trait() {
        let img = create_test_image().with_filter_mode(FilterMode::Nearest);
        let ctx = EvalContext::new();

        let color: Rgba = img.sample(Vec2::new(0.0, 0.0), &ctx);
        assert!(color.r > 0.5); // Red corner

        let vec4: Vec4 = img.sample(Vec2::new(0.0, 0.0), &ctx);
        assert!(vec4.x > 0.5); // Red corner

        let gray: f32 = img.sample(Vec2::new(0.0, 0.0), &ctx);
        // Red has some luminance
        assert!(gray > 0.0 && gray < 1.0);
    }

    #[test]
    fn test_dimensions() {
        let img = create_test_image();
        assert_eq!(img.dimensions(), (2, 2));
    }
}
