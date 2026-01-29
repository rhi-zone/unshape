use unshape_color::{Rgba, blend_with_alpha};

use crate::{BlendMode, Composite, ImageField};

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
/// use unshape_image::{ImageField, composite, BlendMode};
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
    Composite::new(mode, opacity).apply(base, overlay)
}

/// Internal composite implementation.
pub(crate) fn composite_impl(
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
