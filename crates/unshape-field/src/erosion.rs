use glam::Vec2;

use crate::{EvalContext, Field};

/// A heightmap for terrain erosion simulation.
///
/// Stores height values as a 2D grid.
#[derive(Clone)]
pub struct Heightmap {
    /// Height values (row-major).
    pub data: Vec<f32>,
    /// Width of the heightmap.
    pub width: usize,
    /// Height of the heightmap.
    pub height: usize,
}

impl Heightmap {
    /// Creates a new heightmap filled with zeros.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Creates a heightmap from a Field.
    pub fn from_field<F: Field<Vec2, f32>>(
        field: &F,
        width: usize,
        height: usize,
        scale: f32,
    ) -> Self {
        let ctx = EvalContext::new();
        let mut data = vec![0.0; width * height];

        for y in 0..height {
            for x in 0..width {
                let pos = Vec2::new(x as f32 * scale, y as f32 * scale);
                data[y * width + x] = field.sample(pos, &ctx);
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Gets the height at grid coordinates.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Sets the height at grid coordinates.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Gets the height with bilinear interpolation.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let h00 = self.get_clamped(x0, y0);
        let h10 = self.get_clamped(x0 + 1, y0);
        let h01 = self.get_clamped(x0, y0 + 1);
        let h11 = self.get_clamped(x0 + 1, y0 + 1);

        let h0 = h00 * (1.0 - fx) + h10 * fx;
        let h1 = h01 * (1.0 - fx) + h11 * fx;

        h0 * (1.0 - fy) + h1 * fy
    }

    /// Gets height with boundary clamping.
    fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Computes the gradient at a position.
    pub fn gradient(&self, x: f32, y: f32) -> Vec2 {
        let h = 0.5;
        let dx = (self.sample(x + h, y) - self.sample(x - h, y)) / (2.0 * h);
        let dy = (self.sample(x, y + h) - self.sample(x, y - h)) / (2.0 * h);
        Vec2::new(dx, dy)
    }

    /// Adds sediment at a position (with bilinear distribution).
    fn add_sediment(&mut self, x: f32, y: f32, amount: f32) {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        // Bilinear weights
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        self.add_clamped(x0, y0, amount * w00);
        self.add_clamped(x0 + 1, y0, amount * w10);
        self.add_clamped(x0, y0 + 1, amount * w01);
        self.add_clamped(x0 + 1, y0 + 1, amount * w11);
    }

    /// Adds to height with boundary clamping.
    fn add_clamped(&mut self, x: i32, y: i32, amount: f32) {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            let idx = y as usize * self.width + x as usize;
            self.data[idx] += amount;
        }
    }

    /// Gets the minimum and maximum heights.
    pub fn bounds(&self) -> (f32, f32) {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Normalizes heights to 0-1 range.
    pub fn normalize(&mut self) {
        let (min, max) = self.bounds();
        let range = max - min;
        if range > 0.0001 {
            for h in &mut self.data {
                *h = (*h - min) / range;
            }
        }
    }
}

/// Hydraulic erosion simulation operation.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Heightmap))]
pub struct HydraulicErosion {
    /// Number of water droplets to simulate.
    pub iterations: usize,
    /// Maximum steps per droplet lifetime.
    pub max_lifetime: usize,
    /// Initial water volume per droplet.
    pub initial_water: f32,
    /// Initial speed of droplets.
    pub initial_speed: f32,
    /// Inertia factor (0 = follow gradient exactly, 1 = maintain direction).
    pub inertia: f32,
    /// Minimum slope for erosion to occur.
    pub min_slope: f32,
    /// Sediment capacity multiplier.
    pub capacity_factor: f32,
    /// Rate of sediment pickup.
    pub erosion_rate: f32,
    /// Rate of sediment deposition.
    pub deposition_rate: f32,
    /// Water evaporation rate per step.
    pub evaporation_rate: f32,
    /// Gravity strength.
    pub gravity: f32,
    /// Erosion brush radius.
    pub brush_radius: usize,
    /// Random seed for simulation.
    pub seed: u64,
}

impl Default for HydraulicErosion {
    fn default() -> Self {
        Self {
            iterations: 10000,
            max_lifetime: 64,
            initial_water: 1.0,
            initial_speed: 1.0,
            inertia: 0.3,
            min_slope: 0.01,
            capacity_factor: 4.0,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            evaporation_rate: 0.01,
            gravity: 4.0,
            brush_radius: 3,
            seed: 0,
        }
    }
}

impl HydraulicErosion {
    /// Applies this erosion operation to a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Heightmap {
        let mut result = heightmap.clone();
        hydraulic_erosion(&mut result, self, self.seed);
        result
    }
}

/// Backwards-compatible type alias.
pub type HydraulicErosionConfig = HydraulicErosion;

/// Simulates hydraulic erosion on a heightmap.
///
/// Water droplets flow downhill, picking up sediment and depositing it
/// as they slow down, creating realistic river valleys and gullies.
pub fn hydraulic_erosion(heightmap: &mut Heightmap, config: &HydraulicErosion, seed: u64) {
    let mut rng = SimpleRng::new(seed);
    let width = heightmap.width;
    let height = heightmap.height;

    // Pre-compute erosion brush weights
    let brush = compute_erosion_brush(config.brush_radius);

    for _ in 0..config.iterations {
        // Random starting position
        let mut x = rng.next_f32() * (width - 1) as f32;
        let mut y = rng.next_f32() * (height - 1) as f32;

        // Initial direction (random)
        let angle = rng.next_f32() * std::f32::consts::TAU;
        let mut dir = Vec2::new(angle.cos(), angle.sin());

        let mut speed = config.initial_speed;
        let mut water = config.initial_water;
        let mut sediment = 0.0;

        for _ in 0..config.max_lifetime {
            let xi = x as usize;
            let yi = y as usize;

            // Get gradient at current position
            let gradient = heightmap.gradient(x, y);
            let gradient_len = gradient.length();

            // Update direction with inertia
            if gradient_len > 0.0001 {
                dir = dir * config.inertia - gradient.normalize() * (1.0 - config.inertia);
                let dir_len = dir.length();
                if dir_len > 0.0001 {
                    dir /= dir_len;
                }
            }

            // Move to next position
            let new_x = x + dir.x;
            let new_y = y + dir.y;

            // Check boundaries
            if new_x < 0.0
                || new_x >= (width - 1) as f32
                || new_y < 0.0
                || new_y >= (height - 1) as f32
            {
                break;
            }

            // Calculate height difference
            let old_height = heightmap.sample(x, y);
            let new_height = heightmap.sample(new_x, new_y);
            let height_diff = new_height - old_height;

            // Calculate sediment capacity
            let slope = (-height_diff).max(config.min_slope);
            let capacity = slope * speed * water * config.capacity_factor;

            if sediment > capacity || height_diff > 0.0 {
                // Deposit sediment
                let deposit_amount = if height_diff > 0.0 {
                    // Moving uphill - deposit all sediment
                    height_diff.min(sediment)
                } else {
                    // Deposit excess sediment
                    (sediment - capacity) * config.deposition_rate
                };

                sediment -= deposit_amount;
                heightmap.add_sediment(x, y, deposit_amount);
            } else {
                // Erode terrain
                let erode_amount = ((capacity - sediment) * config.erosion_rate).min(-height_diff);

                // Apply erosion with brush
                apply_erosion_brush(heightmap, xi, yi, erode_amount, &brush);
                sediment += erode_amount;
            }

            // Update physics
            speed = (speed * speed + height_diff.abs() * config.gravity)
                .sqrt()
                .max(0.1);
            water *= 1.0 - config.evaporation_rate;

            // Stop if water dried up
            if water < 0.01 {
                break;
            }

            x = new_x;
            y = new_y;
        }
    }
}

/// Brush weights for erosion.
pub struct ErosionBrush {
    /// Grid offsets relative to the center.
    pub offsets: Vec<(i32, i32)>,
    /// Corresponding weights (sum to 1.0).
    pub weights: Vec<f32>,
}

pub fn compute_erosion_brush(radius: usize) -> ErosionBrush {
    let mut offsets = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0.0;
    let r = radius as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist <= radius as f32 {
                let weight = 1.0 - dist / (radius as f32 + 1.0);
                offsets.push((dx, dy));
                weights.push(weight);
                total_weight += weight;
            }
        }
    }

    // Normalize weights
    if total_weight > 0.0 {
        for w in &mut weights {
            *w /= total_weight;
        }
    }

    ErosionBrush { offsets, weights }
}

fn apply_erosion_brush(
    heightmap: &mut Heightmap,
    cx: usize,
    cy: usize,
    amount: f32,
    brush: &ErosionBrush,
) {
    for (i, (dx, dy)) in brush.offsets.iter().enumerate() {
        let x = cx as i32 + dx;
        let y = cy as i32 + dy;
        if x >= 0 && x < heightmap.width as i32 && y >= 0 && y < heightmap.height as i32 {
            let idx = y as usize * heightmap.width + x as usize;
            heightmap.data[idx] -= amount * brush.weights[i];
        }
    }
}

/// Thermal erosion simulation operation.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Heightmap))]
pub struct ThermalErosion {
    /// Number of iterations.
    pub iterations: usize,
    /// Maximum slope angle (as tangent) before material slides.
    pub talus_angle: f32,
    /// Rate of material transfer per iteration.
    pub transfer_rate: f32,
}

impl Default for ThermalErosion {
    fn default() -> Self {
        Self {
            iterations: 50,
            talus_angle: 0.8, // ~40 degrees
            transfer_rate: 0.5,
        }
    }
}

impl ThermalErosion {
    /// Applies this erosion operation to a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Heightmap {
        let mut result = heightmap.clone();
        thermal_erosion(&mut result, self);
        result
    }
}

/// Backwards-compatible type alias.
pub type ThermalErosionConfig = ThermalErosion;

/// Simulates thermal erosion (talus slopes).
///
/// Material slides down when slopes exceed the talus angle,
/// creating realistic scree slopes and smoothing sharp features.
pub fn thermal_erosion(heightmap: &mut Heightmap, config: &ThermalErosion) {
    let width = heightmap.width;
    let height = heightmap.height;

    // Neighbor offsets
    let neighbors: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    // Distance for each neighbor (diagonal = sqrt(2))
    let distances: [f32; 8] = [
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        1.0,
        std::f32::consts::SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
    ];

    let mut deltas = vec![0.0; width * height];

    for _ in 0..config.iterations {
        // Clear deltas
        deltas.fill(0.0);

        // Calculate transfers
        for y in 0..height {
            for x in 0..width {
                let h = heightmap.get(x, y);
                let mut total_diff: f32 = 0.0;
                let mut max_diff: f32 = 0.0;

                // Find total height difference to lower neighbors
                for (i, (dx, dy)) in neighbors.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nh = heightmap.get(nx as usize, ny as usize);
                        let slope = (h - nh) / distances[i];

                        if slope > config.talus_angle {
                            let diff = slope - config.talus_angle;
                            total_diff += diff;
                            max_diff = max_diff.max(diff);
                        }
                    }
                }

                if total_diff > 0.0 {
                    // Distribute material to lower neighbors
                    let transfer_amount = max_diff * config.transfer_rate * 0.5;

                    for (i, (dx, dy)) in neighbors.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nh = heightmap.get(nx as usize, ny as usize);
                            let slope = (h - nh) / distances[i];

                            if slope > config.talus_angle {
                                let diff = slope - config.talus_angle;
                                let ratio = diff / total_diff;
                                let amount = transfer_amount * ratio;

                                deltas[y * width + x] -= amount;
                                deltas[ny as usize * width + nx as usize] += amount;
                            }
                        }
                    }
                }
            }
        }

        // Apply deltas
        for (i, delta) in deltas.iter().enumerate() {
            heightmap.data[i] += delta;
        }
    }
}

/// Simple random number generator for erosion simulation.
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

/// Computes accumulated flow for each point in a heightmap.
/// Higher values indicate more upstream area draining through that point.
pub fn compute_flow_accumulation(heightmap: &Heightmap) -> Heightmap {
    let (width, height) = (heightmap.width, heightmap.height);
    let mut flow = Heightmap::new(width, height);

    // Initialize with 1 (each cell contributes its own area)
    for i in 0..flow.data.len() {
        flow.data[i] = 1.0;
    }

    // Sort cells by height (highest first)
    let mut cells: Vec<(usize, usize, f32)> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            cells.push((x, y, heightmap.get(x, y)));
        }
    }
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Process cells from high to low
    for (x, y, _) in cells {
        let h = heightmap.get(x, y);
        let cell_flow = flow.get(x, y);

        // Find lowest neighbor
        let mut best_drop = 0.0;
        let mut best_x = x;
        let mut best_y = y;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nh = heightmap.get(nx as usize, ny as usize);
                    let drop = h - nh;
                    if drop > best_drop {
                        best_drop = drop;
                        best_x = nx as usize;
                        best_y = ny as usize;
                    }
                }
            }
        }

        // Transfer flow to lowest neighbor
        if best_drop > 0.0 && (best_x != x || best_y != y) {
            let current = flow.get(best_x, best_y);
            flow.set(best_x, best_y, current + cell_flow);
        }
    }

    flow
}
