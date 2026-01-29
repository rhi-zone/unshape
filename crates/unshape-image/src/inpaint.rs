#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_color::Rgba;

use crate::ImageField;
use crate::kernel::Resize;
use crate::pyramid::{downsample, resize};

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
            let (rw, rh) = (result.width, result.height);
            result = Resize::new(rw * 2, rh * 2).apply(&result);
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
