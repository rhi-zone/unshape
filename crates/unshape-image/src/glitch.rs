#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ImageField;

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
/// use unshape_image::{pixel_sort, PixelSort, SortBy};
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

/// Configuration for datamosh glitch effect.
///
/// Datamosh simulates video codec artifacts caused by P-frame accumulation
/// without I-frame refreshes, creating motion smearing and visual corruption.
#[derive(Debug, Clone)]
pub struct Datamosh {
    /// Block size for motion compensation (typical values: 8, 16, 32).
    pub block_size: u32,
    /// Number of "frames" to accumulate artifacts (more = more corruption).
    pub iterations: u32,
    /// Motion intensity - how much blocks shift between iterations (0-1).
    pub motion_intensity: f32,
    /// Decay factor - how much previous frame influences result (0-1).
    /// Higher values create more ghosting/smearing.
    pub decay: f32,
    /// Random seed for reproducible motion vectors.
    pub seed: u32,
    /// Motion pattern to use.
    pub motion: MotionPattern,
    /// Probability of a block "sticking" and not updating (0-1).
    /// Creates freeze artifacts where parts of the image get stuck.
    pub freeze_probability: f32,
}

/// Pattern for motion vector generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionPattern {
    /// Random motion vectors per block (most chaotic).
    #[default]
    Random,
    /// Directional flow (all blocks move in similar direction).
    Directional,
    /// Radial motion from center (expanding or contracting).
    Radial,
    /// Spiral/vortex motion.
    Vortex,
    /// Motion based on pixel brightness (bright areas move more).
    Brightness,
}

impl Default for Datamosh {
    fn default() -> Self {
        Self {
            block_size: 16,
            iterations: 3,
            motion_intensity: 0.5,
            decay: 0.7,
            seed: 42,
            motion: MotionPattern::Random,
            freeze_probability: 0.1,
        }
    }
}

impl Datamosh {
    /// Creates a datamosh config with the given number of iterations.
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            ..Default::default()
        }
    }

    /// Sets the block size.
    pub fn block_size(mut self, size: u32) -> Self {
        self.block_size = size.max(4);
        self
    }

    /// Sets the motion intensity.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.motion_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Sets the decay factor.
    pub fn decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(0.0, 1.0);
        self
    }

    /// Sets the motion pattern.
    pub fn pattern(mut self, pattern: MotionPattern) -> Self {
        self.motion = pattern;
        self
    }

    /// Sets the freeze probability.
    pub fn freeze(mut self, prob: f32) -> Self {
        self.freeze_probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Sets the random seed.
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Applies this datamosh configuration to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        datamosh(image, self)
    }
}

/// Applies datamosh glitch effect to an image.
///
/// Simulates video codec artifacts from P-frame accumulation without I-frame
/// refreshes, creating motion smearing and block corruption typical of corrupted
/// video files.
///
/// # Example
///
/// ```ignore
/// use unshape_image::{datamosh, Datamosh, MotionPattern};
///
/// // Basic datamosh with default settings
/// let glitched = datamosh(&image, &Datamosh::default());
///
/// // More intense datamosh with radial motion
/// let config = Datamosh::new(5)
///     .intensity(0.8)
///     .pattern(MotionPattern::Radial)
///     .decay(0.9);
/// let glitched = datamosh(&image, &config);
/// ```
pub fn datamosh(image: &ImageField, config: &Datamosh) -> ImageField {
    let width = image.width as usize;
    let height = image.height as usize;
    let block_size = config.block_size.max(4) as usize;

    // Calculate number of blocks
    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;

    // Start with the original image data
    let mut current = image.data.clone();
    let mut motion_vectors: Vec<(i32, i32)> = vec![(0, 0); blocks_x * blocks_y];
    let mut frozen: Vec<bool> = vec![false; blocks_x * blocks_y];

    // Initialize motion vectors based on pattern
    initialize_motion_vectors(&mut motion_vectors, &mut frozen, blocks_x, blocks_y, config);

    // Iterate to accumulate artifacts
    for iteration in 0..config.iterations {
        let mut next = vec![[0.0f32; 4]; width * height];

        // Update motion vectors for each iteration (add some drift)
        update_motion_vectors(&mut motion_vectors, blocks_x, blocks_y, iteration, config);

        // Process each block
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let (mvx, mvy) = motion_vectors[block_idx];

                // Calculate block boundaries
                let x_start = bx * block_size;
                let y_start = by * block_size;
                let x_end = (x_start + block_size).min(width);
                let y_end = (y_start + block_size).min(height);

                // If frozen, don't update this block
                if frozen[block_idx] {
                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            let idx = y * width + x;
                            next[idx] = current[idx];
                        }
                    }
                    continue;
                }

                // Apply motion compensation
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let idx = y * width + x;

                        // Source position (with motion vector applied)
                        let src_x = (x as i32 + mvx).clamp(0, (width - 1) as i32) as usize;
                        let src_y = (y as i32 + mvy).clamp(0, (height - 1) as i32) as usize;
                        let src_idx = src_y * width + src_x;

                        // Blend with decay
                        let src_pixel = current[src_idx];
                        let cur_pixel = current[idx];

                        next[idx] = [
                            src_pixel[0] * config.decay + cur_pixel[0] * (1.0 - config.decay),
                            src_pixel[1] * config.decay + cur_pixel[1] * (1.0 - config.decay),
                            src_pixel[2] * config.decay + cur_pixel[2] * (1.0 - config.decay),
                            cur_pixel[3], // Preserve alpha
                        ];
                    }
                }
            }
        }

        current = next;
    }

    ImageField {
        data: current,
        width: image.width,
        height: image.height,
        wrap_mode: image.wrap_mode,
        filter_mode: image.filter_mode,
    }
}

/// Initialize motion vectors based on the pattern.
fn initialize_motion_vectors(
    vectors: &mut [(i32, i32)],
    frozen: &mut [bool],
    blocks_x: usize,
    blocks_y: usize,
    config: &Datamosh,
) {
    let max_motion = (config.block_size as f32 * config.motion_intensity * 2.0) as i32;
    let cx = blocks_x as f32 / 2.0;
    let cy = blocks_y as f32 / 2.0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = by * blocks_x + bx;

            // Check for freeze
            let freeze_hash = simple_hash(config.seed.wrapping_add(idx as u32).wrapping_mul(7919));
            let freeze_prob = freeze_hash as f32 / u32::MAX as f32;
            frozen[idx] = freeze_prob < config.freeze_probability;

            if frozen[idx] {
                vectors[idx] = (0, 0);
                continue;
            }

            let (mvx, mvy) = match config.motion {
                MotionPattern::Random => {
                    let hash_x = simple_hash(config.seed.wrapping_add(idx as u32));
                    let hash_y =
                        simple_hash(config.seed.wrapping_add(idx as u32).wrapping_mul(31337));
                    let mvx = ((hash_x as f32 / u32::MAX as f32) * 2.0 - 1.0) * max_motion as f32;
                    let mvy = ((hash_y as f32 / u32::MAX as f32) * 2.0 - 1.0) * max_motion as f32;
                    (mvx as i32, mvy as i32)
                }
                MotionPattern::Directional => {
                    // Mostly horizontal motion with some variance
                    let hash = simple_hash(config.seed.wrapping_add(idx as u32));
                    let variance = ((hash as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.3;
                    let mvx = max_motion as f32 * (1.0 + variance);
                    let mvy = max_motion as f32 * variance * 0.5;
                    (mvx as i32, mvy as i32)
                }
                MotionPattern::Radial => {
                    // Motion away from center
                    let dx = bx as f32 - cx;
                    let dy = by as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.001);
                    let scale = config.motion_intensity * config.block_size as f32 / dist;
                    ((dx * scale) as i32, (dy * scale) as i32)
                }
                MotionPattern::Vortex => {
                    // Spiral motion around center
                    let dx = bx as f32 - cx;
                    let dy = by as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.001);
                    let scale = config.motion_intensity * config.block_size as f32 / (dist + 1.0);
                    // Perpendicular + some radial
                    (
                        (-dy * scale + dx * scale * 0.3) as i32,
                        (dx * scale + dy * scale * 0.3) as i32,
                    )
                }
                MotionPattern::Brightness => {
                    // Will be updated in update_motion_vectors based on block brightness
                    (0, 0)
                }
            };

            vectors[idx] = (mvx, mvy);
        }
    }
}

/// Update motion vectors for subsequent iterations.
fn update_motion_vectors(
    vectors: &mut [(i32, i32)],
    blocks_x: usize,
    blocks_y: usize,
    iteration: u32,
    config: &Datamosh,
) {
    // Add some drift/noise to motion vectors over iterations
    let drift_scale = 0.2 * config.motion_intensity;
    let max_drift = (config.block_size as f32 * drift_scale) as i32;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = by * blocks_x + bx;
            let (mvx, mvy) = vectors[idx];

            // Skip frozen blocks
            if mvx == 0 && mvy == 0 && config.freeze_probability > 0.0 {
                continue;
            }

            // Add drift based on iteration
            let hash = simple_hash(
                config
                    .seed
                    .wrapping_add(idx as u32)
                    .wrapping_mul(iteration.wrapping_add(1)),
            );
            let drift_x = ((hash as f32 / u32::MAX as f32) * 2.0 - 1.0) * max_drift as f32;
            let drift_y = (((hash >> 16) as f32 / u16::MAX as f32) * 2.0 - 1.0) * max_drift as f32;

            vectors[idx] = (mvx + drift_x as i32, mvy + drift_y as i32);
        }
    }
}

/// Applies datamosh effect between two frames.
///
/// This variant takes a "previous frame" and applies motion estimation
/// between the two frames, more closely mimicking actual video codec behavior.
///
/// # Example
///
/// ```ignore
/// use unshape_image::{datamosh_frames, Datamosh};
///
/// let glitched = datamosh_frames(&frame1, &frame2, &Datamosh::default());
/// ```
pub fn datamosh_frames(
    prev_frame: &ImageField,
    curr_frame: &ImageField,
    config: &Datamosh,
) -> ImageField {
    if prev_frame.width != curr_frame.width || prev_frame.height != curr_frame.height {
        // If sizes don't match, fall back to single-frame datamosh
        return datamosh(curr_frame, config);
    }

    let width = curr_frame.width as usize;
    let height = curr_frame.height as usize;
    let block_size = config.block_size.max(4) as usize;

    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;

    let mut result = vec![[0.0f32; 4]; width * height];

    // For each block, estimate motion from prev to curr frame
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let x_start = bx * block_size;
            let y_start = by * block_size;
            let x_end = (x_start + block_size).min(width);
            let y_end = (y_start + block_size).min(height);

            // Check for freeze
            let block_idx = by * blocks_x + bx;
            let freeze_hash = simple_hash(
                config
                    .seed
                    .wrapping_add(block_idx as u32)
                    .wrapping_mul(7919),
            );
            let freeze_prob = freeze_hash as f32 / u32::MAX as f32;

            if freeze_prob < config.freeze_probability {
                // Frozen block - use previous frame
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let idx = y * width + x;
                        result[idx] = prev_frame.data[idx];
                    }
                }
                continue;
            }

            // Estimate motion vector by finding average color difference
            let (mvx, mvy) = estimate_block_motion(
                prev_frame,
                curr_frame,
                x_start,
                y_start,
                x_end - x_start,
                y_end - y_start,
                config,
                block_idx,
            );

            // Apply motion-compensated prediction with artifacts
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = y * width + x;

                    // Use motion vector from previous frame
                    let src_x = (x as i32 + mvx).clamp(0, (width - 1) as i32) as usize;
                    let src_y = (y as i32 + mvy).clamp(0, (height - 1) as i32) as usize;
                    let src_idx = src_y * width + src_x;

                    let prev_pixel = prev_frame.data[src_idx];
                    let curr_pixel = curr_frame.data[idx];

                    // Blend based on decay (simulates incomplete refresh)
                    result[idx] = [
                        prev_pixel[0] * config.decay + curr_pixel[0] * (1.0 - config.decay),
                        prev_pixel[1] * config.decay + curr_pixel[1] * (1.0 - config.decay),
                        prev_pixel[2] * config.decay + curr_pixel[2] * (1.0 - config.decay),
                        curr_pixel[3],
                    ];
                }
            }
        }
    }

    ImageField {
        data: result,
        width: curr_frame.width,
        height: curr_frame.height,
        wrap_mode: curr_frame.wrap_mode,
        filter_mode: curr_frame.filter_mode,
    }
}

/// Estimate motion vector for a block based on brightness changes.
fn estimate_block_motion(
    prev_frame: &ImageField,
    curr_frame: &ImageField,
    x_start: usize,
    y_start: usize,
    block_w: usize,
    block_h: usize,
    config: &Datamosh,
    block_idx: usize,
) -> (i32, i32) {
    let width = curr_frame.width as usize;

    // Calculate brightness gradient in the block
    let mut dx_sum = 0.0f32;
    let mut dy_sum = 0.0f32;
    let mut count = 0.0;

    for dy in 0..block_h {
        for dx in 0..block_w {
            let x = x_start + dx;
            let y = y_start + dy;
            let idx = y * width + x;

            let prev_bright = brightness(&prev_frame.data[idx]);
            let curr_bright = brightness(&curr_frame.data[idx]);
            let diff = curr_bright - prev_bright;

            // Estimate motion direction from brightness change
            // (very simplified - real codecs use much more sophisticated algorithms)
            if diff.abs() > 0.01 {
                // Add some directional bias based on position in block
                let bx = dx as f32 / block_w as f32 - 0.5;
                let by = dy as f32 / block_h as f32 - 0.5;
                dx_sum += diff * bx;
                dy_sum += diff * by;
                count += diff.abs();
            }
        }
    }

    if count < 0.001 {
        // Low motion - add some noise for visual interest
        let hash = simple_hash(config.seed.wrapping_add(block_idx as u32));
        let noise_x = ((hash as f32 / u32::MAX as f32) * 2.0 - 1.0) * config.motion_intensity;
        let noise_y =
            (((hash >> 16) as f32 / u16::MAX as f32) * 2.0 - 1.0) * config.motion_intensity;
        return (
            (noise_x * config.block_size as f32) as i32,
            (noise_y * config.block_size as f32) as i32,
        );
    }

    // Scale motion vector
    let scale = config.motion_intensity * config.block_size as f32 * 4.0;
    (
        (dx_sum / count * scale) as i32,
        (dy_sum / count * scale) as i32,
    )
}

/// Calculate brightness of a pixel.
fn brightness(pixel: &[f32; 4]) -> f32 {
    0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
}

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
/// use unshape_image::{ImageField, JpegArtifacts, jpeg_artifacts};
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
/// use unshape_image::{ImageField, BitManip, BitOperation, bit_manip};
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
    let op = config.operation;
    let value = config.value;
    let channels = config.channels;

    // Perf guess: inline match probably better than function pointer or closure.
    // - Function pointer: indirect call overhead on every pixel?
    // - Closure: same, plus potential capture overhead?
    // - Inline match: LLVM *might* see `op` is loop-invariant and hoist
    // Not benchmarked - branch prediction probably makes it all moot anyway.

    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let mut result = pixel;
            for (i, &should_apply) in channels.iter().enumerate() {
                if should_apply {
                    let byte = (pixel[i].clamp(0.0, 1.0) * 255.0) as u8;
                    let out = match op {
                        BitOperation::Xor => byte ^ value,
                        BitOperation::And => byte & value,
                        BitOperation::Or => byte | value,
                        BitOperation::Not => !byte,
                        BitOperation::ShiftLeft => byte.wrapping_shl(value as u32),
                        BitOperation::ShiftRight => byte.wrapping_shr(value as u32),
                    };
                    result[i] = out as f32 / 255.0;
                }
            }
            data.push(result);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
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
/// use unshape_image::{ImageField, ByteCorrupt, CorruptMode, byte_corrupt};
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
