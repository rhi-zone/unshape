use glam::Vec2;
use unshape_field::{EvalContext, Field};

use crate::ImageField;
use crate::bake::{BakeConfig, bake_scalar};
use crate::kernel::{Kernel, convolve};

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
/// use unshape_image::{ImageField, heightfield_to_normal_map};
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
