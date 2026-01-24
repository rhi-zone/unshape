//! Noise texture generation demo.
//!
//! Generates a noise texture using layered Perlin noise (fBm)
//! and exports it as a PNG.
//!
//! Run with: `cargo run --example noise_texture`

use glam::Vec2;
use unshape_field::{EvalContext, Fbm2D, Field, Perlin2D};
use unshape_image::{ImageField, export_png};

fn main() {
    println!("Generating noise texture...");

    // Create layered Perlin noise (fractional Brownian motion)
    let noise = Fbm2D::new(Perlin2D::with_seed(42))
        .octaves(6)
        .lacunarity(2.0)
        .gain(0.5);

    // Bake to image
    let width = 512u32;
    let height = 512u32;

    let ctx = EvalContext::new();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            // Scale UV to get nice detail
            let uv = Vec2::new(
                x as f32 / width as f32 * 8.0,
                y as f32 / height as f32 * 8.0,
            );

            // Sample noise (Fbm2D returns 0-1)
            let value = noise.sample(uv, &ctx);

            pixels.push([value, value, value, 1.0]);
        }
    }

    let image = ImageField::from_raw(pixels, width, height);

    let output_path = "noise_texture_output.png";
    match export_png(&image, output_path) {
        Ok(_) => println!("Wrote {}", output_path),
        Err(e) => eprintln!("Failed to write PNG: {}", e),
    }

    // Also generate a colored version using domain warping
    println!("Generating warped color texture...");

    let noise2 = Perlin2D::with_seed(123);
    let mut color_pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(
                x as f32 / width as f32 * 4.0,
                y as f32 / height as f32 * 4.0,
            );

            // Domain warping: offset UV by noise
            let warp_x = noise.sample(uv, &ctx) * 0.5;
            let warp_y = noise.sample(uv + Vec2::new(5.2, 1.3), &ctx) * 0.5;
            let warped_uv = uv + Vec2::new(warp_x, warp_y);

            // Sample at warped position
            let n1 = noise.sample(warped_uv, &ctx);
            let n2 = (noise2.sample(warped_uv * 2.0, &ctx) + 1.0) * 0.5; // Perlin returns -1 to 1

            // Create organic-looking colors
            let r = (n1 * 0.6 + 0.2).clamp(0.0, 1.0);
            let g = (n1 * 0.4 + n2 * 0.3 + 0.1).clamp(0.0, 1.0);
            let b = (n2 * 0.5 + 0.3).clamp(0.0, 1.0);

            color_pixels.push([r, g, b, 1.0]);
        }
    }

    let color_image = ImageField::from_raw(color_pixels, width, height);

    let color_output_path = "noise_texture_color_output.png";
    match export_png(&color_image, color_output_path) {
        Ok(_) => println!("Wrote {}", color_output_path),
        Err(e) => eprintln!("Failed to write PNG: {}", e),
    }
}
