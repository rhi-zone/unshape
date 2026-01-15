//! Terrain generation and erosion simulation.
//!
//! Provides heightfield-based terrain generation with:
//! - Heightfield mesh generation
//! - Hydraulic erosion (water flow simulation)
//! - Thermal erosion (material slippage)
//! - Diamond-square fractal noise
//!
//! # Example
//!
//! ```
//! use rhizome_resin_mesh::{Heightfield, HydraulicErosion, ThermalErosion};
//!
//! // Create a heightfield
//! let mut heightfield = Heightfield::new(64, 64);
//! heightfield.apply_diamond_square(0.5, 12345);
//!
//! // Apply erosion
//! let mut erosion = HydraulicErosion::default();
//! erosion.erode(&mut heightfield, 10000);
//!
//! // Convert to mesh
//! let mesh = heightfield.to_mesh(10.0, 2.0);
//! ```

use crate::Mesh;
use glam::{Vec2, Vec3};

/// A 2D heightfield for terrain generation.
#[derive(Debug, Clone)]
pub struct Heightfield {
    /// Height values (row-major order).
    heights: Vec<f32>,
    /// Width in samples.
    width: usize,
    /// Height in samples (depth).
    depth: usize,
}

impl Heightfield {
    /// Creates a new flat heightfield.
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            heights: vec![0.0; width * depth],
            width,
            depth,
        }
    }

    /// Creates a heightfield from existing data.
    pub fn from_data(heights: Vec<f32>, width: usize, depth: usize) -> Self {
        assert_eq!(heights.len(), width * depth);
        Self {
            heights,
            width,
            depth,
        }
    }

    /// Returns the width in samples.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the depth in samples.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Gets the height at (x, z) coordinates.
    pub fn get(&self, x: usize, z: usize) -> f32 {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x]
        } else {
            0.0
        }
    }

    /// Sets the height at (x, z) coordinates.
    pub fn set(&mut self, x: usize, z: usize, height: f32) {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x] = height;
        }
    }

    /// Adds to the height at (x, z) coordinates.
    pub fn add(&mut self, x: usize, z: usize, delta: f32) {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x] += delta;
        }
    }

    /// Returns the raw height data.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Returns mutable access to raw height data.
    pub fn heights_mut(&mut self) -> &mut [f32] {
        &mut self.heights
    }

    /// Samples height with bilinear interpolation.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = u * (self.width - 1) as f32;
        let z = v * (self.depth - 1) as f32;

        let x0 = (x.floor() as usize).min(self.width - 1);
        let x1 = (x0 + 1).min(self.width - 1);
        let z0 = (z.floor() as usize).min(self.depth - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let fx = x - x.floor();
        let fz = z - z.floor();

        let h00 = self.get(x0, z0);
        let h10 = self.get(x1, z0);
        let h01 = self.get(x0, z1);
        let h11 = self.get(x1, z1);

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;

        h0 + (h1 - h0) * fz
    }

    /// Computes the gradient at (x, z) using central differences.
    pub fn gradient(&self, x: usize, z: usize) -> Vec2 {
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(self.width - 1);
        let z0 = z.saturating_sub(1);
        let z1 = (z + 1).min(self.depth - 1);

        let dx = (self.get(x1, z) - self.get(x0, z)) / ((x1 - x0).max(1) as f32);
        let dz = (self.get(x, z1) - self.get(x, z0)) / ((z1 - z0).max(1) as f32);

        Vec2::new(dx, dz)
    }

    /// Computes the normal at (x, z).
    pub fn normal(&self, x: usize, z: usize) -> Vec3 {
        let grad = self.gradient(x, z);
        Vec3::new(-grad.x, 1.0, -grad.y).normalize()
    }

    /// Fills with a constant value.
    pub fn fill(&mut self, height: f32) {
        self.heights.fill(height);
    }

    /// Applies diamond-square fractal noise.
    pub fn apply_diamond_square(&mut self, roughness: f32, seed: u64) {
        let size = self.width.max(self.depth);
        let mut rng = SimpleRng::new(seed);

        // Initialize corners
        self.set(0, 0, rng.next_f32() - 0.5);
        self.set(self.width - 1, 0, rng.next_f32() - 0.5);
        self.set(0, self.depth - 1, rng.next_f32() - 0.5);
        self.set(self.width - 1, self.depth - 1, rng.next_f32() - 0.5);

        let mut step = size - 1;
        let mut scale = roughness;

        while step > 1 {
            let half = step / 2;

            // Diamond step
            for z in (0..self.depth - 1).step_by(step) {
                for x in (0..self.width - 1).step_by(step) {
                    let x1 = (x + step).min(self.width - 1);
                    let z1 = (z + step).min(self.depth - 1);

                    let avg =
                        (self.get(x, z) + self.get(x1, z) + self.get(x, z1) + self.get(x1, z1))
                            / 4.0;

                    let cx = x + half;
                    let cz = z + half;
                    if cx < self.width && cz < self.depth {
                        self.set(cx, cz, avg + (rng.next_f32() - 0.5) * scale);
                    }
                }
            }

            // Square step
            for z in 0..self.depth {
                let offset = if (z / half) % 2 == 0 { half } else { 0 };
                for x in (offset..self.width).step_by(step) {
                    let mut sum = 0.0;
                    let mut count = 0;

                    if x >= half {
                        sum += self.get(x - half, z);
                        count += 1;
                    }
                    if x + half < self.width {
                        sum += self.get(x + half, z);
                        count += 1;
                    }
                    if z >= half {
                        sum += self.get(x, z - half);
                        count += 1;
                    }
                    if z + half < self.depth {
                        sum += self.get(x, z + half);
                        count += 1;
                    }

                    if count > 0 {
                        let avg = sum / count as f32;
                        self.set(x, z, avg + (rng.next_f32() - 0.5) * scale);
                    }
                }
            }

            step = half;
            scale *= 0.5;
        }
    }

    /// Normalizes heights to [0, 1] range.
    pub fn normalize(&mut self) {
        let min = self.heights.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .heights
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range > 0.0 {
            for h in &mut self.heights {
                *h = (*h - min) / range;
            }
        }
    }

    /// Clamps heights to a range.
    pub fn clamp(&mut self, min: f32, max: f32) {
        for h in &mut self.heights {
            *h = h.clamp(min, max);
        }
    }

    /// Applies Gaussian blur for smoothing.
    pub fn blur(&mut self, radius: usize) {
        let sigma = radius as f32 / 2.0;
        let kernel_size = radius * 2 + 1;

        // Build Gaussian kernel
        let mut kernel = Vec::with_capacity(kernel_size);
        let mut sum = 0.0;
        for i in 0..kernel_size {
            let x = i as f32 - radius as f32;
            let w = (-x * x / (2.0 * sigma * sigma)).exp();
            kernel.push(w);
            sum += w;
        }
        for k in &mut kernel {
            *k /= sum;
        }

        // Horizontal pass
        let mut temp = self.heights.clone();
        for z in 0..self.depth {
            for x in 0..self.width {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let sx = (x as isize + i as isize - radius as isize)
                        .clamp(0, self.width as isize - 1) as usize;
                    acc += self.get(sx, z) * w;
                }
                temp[z * self.width + x] = acc;
            }
        }

        // Vertical pass
        for z in 0..self.depth {
            for x in 0..self.width {
                let mut acc = 0.0;
                for (i, &w) in kernel.iter().enumerate() {
                    let sz = (z as isize + i as isize - radius as isize)
                        .clamp(0, self.depth as isize - 1) as usize;
                    acc += temp[sz * self.width + x] * w;
                }
                self.heights[z * self.width + x] = acc;
            }
        }
    }

    /// Converts the heightfield to a triangle mesh.
    ///
    /// # Arguments
    /// * `scale_xz` - Horizontal scale
    /// * `scale_y` - Vertical scale (height multiplier)
    pub fn to_mesh(&self, scale_xz: f32, scale_y: f32) -> Mesh {
        let num_vertices = self.width * self.depth;
        let num_triangles = (self.width - 1) * (self.depth - 1) * 2;

        let mut positions = Vec::with_capacity(num_vertices);
        let mut normals = Vec::with_capacity(num_vertices);
        let mut uvs = Vec::with_capacity(num_vertices);
        let mut indices = Vec::with_capacity(num_triangles * 3);

        // Generate vertices
        for z in 0..self.depth {
            for x in 0..self.width {
                let px = (x as f32 / (self.width - 1) as f32 - 0.5) * scale_xz;
                let py = self.get(x, z) * scale_y;
                let pz = (z as f32 / (self.depth - 1) as f32 - 0.5) * scale_xz;

                positions.push(Vec3::new(px, py, pz));
                normals.push(self.normal(x, z));
                uvs.push(Vec2::new(
                    x as f32 / (self.width - 1) as f32,
                    z as f32 / (self.depth - 1) as f32,
                ));
            }
        }

        // Generate triangles
        for z in 0..self.depth - 1 {
            for x in 0..self.width - 1 {
                let i00 = (z * self.width + x) as u32;
                let i10 = (z * self.width + x + 1) as u32;
                let i01 = ((z + 1) * self.width + x) as u32;
                let i11 = ((z + 1) * self.width + x + 1) as u32;

                // First triangle
                indices.push(i00);
                indices.push(i01);
                indices.push(i10);

                // Second triangle
                indices.push(i10);
                indices.push(i01);
                indices.push(i11);
            }
        }

        Mesh {
            positions,
            normals,
            uvs,
            indices,
        }
    }
}

/// Hydraulic erosion simulation.
///
/// Simulates water flow and sediment transport.
#[derive(Debug, Clone)]
pub struct HydraulicErosion {
    /// Inertia - how much the droplet retains its direction (0-1).
    pub inertia: f32,
    /// Sediment capacity multiplier.
    pub capacity: f32,
    /// Deposition rate (0-1).
    pub deposition: f32,
    /// Erosion rate (0-1).
    pub erosion: f32,
    /// Evaporation rate (0-1).
    pub evaporation: f32,
    /// Minimum slope for erosion.
    pub min_slope: f32,
    /// Gravity strength.
    pub gravity: f32,
    /// Erosion radius.
    pub radius: usize,
    /// Maximum droplet lifetime.
    pub max_lifetime: usize,
    /// Initial water amount.
    pub initial_water: f32,
    /// Initial speed.
    pub initial_speed: f32,
}

impl Default for HydraulicErosion {
    fn default() -> Self {
        Self {
            inertia: 0.05,
            capacity: 4.0,
            deposition: 0.3,
            erosion: 0.3,
            evaporation: 0.01,
            min_slope: 0.01,
            gravity: 4.0,
            radius: 3,
            max_lifetime: 30,
            initial_water: 1.0,
            initial_speed: 1.0,
        }
    }
}

impl HydraulicErosion {
    /// Creates with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets inertia.
    pub fn with_inertia(mut self, inertia: f32) -> Self {
        self.inertia = inertia.clamp(0.0, 1.0);
        self
    }

    /// Sets erosion rate.
    pub fn with_erosion(mut self, erosion: f32) -> Self {
        self.erosion = erosion.clamp(0.0, 1.0);
        self
    }

    /// Runs erosion simulation with the given number of droplets.
    pub fn erode(&self, heightfield: &mut Heightfield, iterations: usize) {
        self.erode_with_seed(heightfield, iterations, 12345);
    }

    /// Runs erosion simulation with a specific seed.
    pub fn erode_with_seed(&self, heightfield: &mut Heightfield, iterations: usize, seed: u64) {
        let mut rng = SimpleRng::new(seed);

        for _ in 0..iterations {
            // Spawn droplet at random position
            let mut pos_x = rng.next_f32() * (heightfield.width() - 1) as f32;
            let mut pos_z = rng.next_f32() * (heightfield.depth() - 1) as f32;
            let mut dir_x = 0.0f32;
            let mut dir_z = 0.0f32;
            let mut speed = self.initial_speed;
            let mut water = self.initial_water;
            let mut sediment = 0.0f32;

            for _ in 0..self.max_lifetime {
                let cell_x = pos_x as usize;
                let cell_z = pos_z as usize;

                if cell_x >= heightfield.width() - 1 || cell_z >= heightfield.depth() - 1 {
                    break;
                }

                // Calculate gradient
                let grad = heightfield.gradient(cell_x, cell_z);

                // Update direction with inertia
                dir_x = dir_x * self.inertia - grad.x * (1.0 - self.inertia);
                dir_z = dir_z * self.inertia - grad.y * (1.0 - self.inertia);

                // Normalize direction
                let len = (dir_x * dir_x + dir_z * dir_z).sqrt();
                if len < 0.0001 {
                    // Random direction if stuck
                    dir_x = rng.next_f32() * 2.0 - 1.0;
                    dir_z = rng.next_f32() * 2.0 - 1.0;
                } else {
                    dir_x /= len;
                    dir_z /= len;
                }

                // Move droplet
                let new_pos_x = pos_x + dir_x;
                let new_pos_z = pos_z + dir_z;

                // Check bounds
                if new_pos_x < 0.0
                    || new_pos_x >= (heightfield.width() - 1) as f32
                    || new_pos_z < 0.0
                    || new_pos_z >= (heightfield.depth() - 1) as f32
                {
                    break;
                }

                // Calculate height difference
                let old_height = heightfield.sample(
                    pos_x / (heightfield.width() - 1) as f32,
                    pos_z / (heightfield.depth() - 1) as f32,
                );
                let new_height = heightfield.sample(
                    new_pos_x / (heightfield.width() - 1) as f32,
                    new_pos_z / (heightfield.depth() - 1) as f32,
                );
                let height_diff = new_height - old_height;

                // Calculate sediment capacity
                let capacity =
                    ((-height_diff).max(self.min_slope) * speed * water * self.capacity).max(0.0);

                if sediment > capacity || height_diff > 0.0 {
                    // Deposit sediment
                    let deposit_amount = if height_diff > 0.0 {
                        sediment.min(height_diff)
                    } else {
                        (sediment - capacity) * self.deposition
                    };

                    sediment -= deposit_amount;
                    self.deposit(heightfield, pos_x, pos_z, deposit_amount);
                } else {
                    // Erode terrain
                    let erode_amount =
                        ((capacity - sediment) * self.erosion).min(-height_diff.min(0.0));

                    self.erode_at(heightfield, pos_x, pos_z, erode_amount);
                    sediment += erode_amount;
                }

                // Update speed
                speed = (speed * speed + height_diff * self.gravity).abs().sqrt();

                // Evaporate water
                water *= 1.0 - self.evaporation;

                if water < 0.01 {
                    break;
                }

                pos_x = new_pos_x;
                pos_z = new_pos_z;
            }
        }
    }

    fn deposit(&self, heightfield: &mut Heightfield, x: f32, z: f32, amount: f32) {
        let cell_x = x as usize;
        let cell_z = z as usize;

        // Bilinear weights
        let fx = x - x.floor();
        let fz = z - z.floor();

        heightfield.add(cell_x, cell_z, amount * (1.0 - fx) * (1.0 - fz));
        if cell_x + 1 < heightfield.width() {
            heightfield.add(cell_x + 1, cell_z, amount * fx * (1.0 - fz));
        }
        if cell_z + 1 < heightfield.depth() {
            heightfield.add(cell_x, cell_z + 1, amount * (1.0 - fx) * fz);
        }
        if cell_x + 1 < heightfield.width() && cell_z + 1 < heightfield.depth() {
            heightfield.add(cell_x + 1, cell_z + 1, amount * fx * fz);
        }
    }

    fn erode_at(&self, heightfield: &mut Heightfield, x: f32, z: f32, amount: f32) {
        let center_x = x as isize;
        let center_z = z as isize;
        let radius = self.radius as isize;

        // Compute weights based on distance
        let mut total_weight = 0.0;
        let mut weights = Vec::new();

        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let px = center_x + dx;
                let pz = center_z + dz;

                if px >= 0
                    && px < heightfield.width() as isize
                    && pz >= 0
                    && pz < heightfield.depth() as isize
                {
                    let dist = ((dx * dx + dz * dz) as f32).sqrt();
                    if dist <= radius as f32 {
                        let w = (1.0 - dist / radius as f32).max(0.0);
                        weights.push((px as usize, pz as usize, w));
                        total_weight += w;
                    }
                }
            }
        }

        // Apply erosion
        if total_weight > 0.0 {
            for (px, pz, w) in weights {
                heightfield.add(px, pz, -amount * w / total_weight);
            }
        }
    }
}

/// Thermal erosion simulation.
///
/// Simulates material slippage on steep slopes.
#[derive(Debug, Clone)]
pub struct ThermalErosion {
    /// Talus angle (maximum stable slope in radians).
    pub talus_angle: f32,
    /// Amount of material transferred per iteration.
    pub transfer_rate: f32,
}

impl Default for ThermalErosion {
    fn default() -> Self {
        Self {
            talus_angle: 0.5, // ~28.6 degrees
            transfer_rate: 0.5,
        }
    }
}

impl ThermalErosion {
    /// Creates with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the transfer rate.
    pub fn with_transfer_rate(mut self, rate: f32) -> Self {
        self.transfer_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Runs thermal erosion for the given number of iterations.
    pub fn erode(&self, heightfield: &mut Heightfield, iterations: usize) {
        let max_slope = self.talus_angle.tan();

        for _ in 0..iterations {
            let mut deltas = vec![0.0f32; heightfield.heights().len()];

            for z in 0..heightfield.depth() {
                for x in 0..heightfield.width() {
                    let h = heightfield.get(x, z);

                    // Check neighbors
                    let neighbors = [
                        (x.wrapping_sub(1), z, 1.0),
                        (x + 1, z, 1.0),
                        (x, z.wrapping_sub(1), 1.0),
                        (x, z + 1, 1.0),
                        (
                            x.wrapping_sub(1),
                            z.wrapping_sub(1),
                            std::f32::consts::SQRT_2,
                        ),
                        (x + 1, z.wrapping_sub(1), std::f32::consts::SQRT_2),
                        (x.wrapping_sub(1), z + 1, std::f32::consts::SQRT_2),
                        (x + 1, z + 1, std::f32::consts::SQRT_2),
                    ];

                    let mut max_diff = 0.0f32;
                    let mut max_neighbor = None;

                    for &(nx, nz, dist) in &neighbors {
                        if nx < heightfield.width() && nz < heightfield.depth() {
                            let nh = heightfield.get(nx, nz);
                            let slope = (h - nh) / dist;

                            if slope > max_slope && slope > max_diff {
                                max_diff = slope;
                                max_neighbor = Some((nx, nz, dist));
                            }
                        }
                    }

                    // Transfer material if slope exceeds threshold
                    if let Some((nx, nz, dist)) = max_neighbor {
                        let transfer = (max_diff - max_slope) * dist * self.transfer_rate * 0.5;
                        let idx = z * heightfield.width() + x;
                        let nidx = nz * heightfield.width() + nx;

                        deltas[idx] -= transfer;
                        deltas[nidx] += transfer;
                    }
                }
            }

            // Apply deltas
            for (h, d) in heightfield.heights_mut().iter_mut().zip(deltas.iter()) {
                *h += d;
            }
        }
    }
}

/// Combined erosion with both hydraulic and thermal effects.
#[derive(Debug, Clone)]
pub struct CombinedErosion {
    /// Hydraulic erosion settings.
    pub hydraulic: HydraulicErosion,
    /// Thermal erosion settings.
    pub thermal: ThermalErosion,
    /// Hydraulic iterations per step.
    pub hydraulic_iterations: usize,
    /// Thermal iterations per step.
    pub thermal_iterations: usize,
}

impl Default for CombinedErosion {
    fn default() -> Self {
        Self {
            hydraulic: HydraulicErosion::default(),
            thermal: ThermalErosion::default(),
            hydraulic_iterations: 5000,
            thermal_iterations: 10,
        }
    }
}

impl CombinedErosion {
    /// Runs combined erosion.
    pub fn erode(&self, heightfield: &mut Heightfield, steps: usize) {
        self.erode_with_seed(heightfield, steps, 12345);
    }

    /// Runs combined erosion with a specific seed.
    pub fn erode_with_seed(&self, heightfield: &mut Heightfield, steps: usize, seed: u64) {
        for i in 0..steps {
            self.hydraulic.erode_with_seed(
                heightfield,
                self.hydraulic_iterations,
                seed.wrapping_add(i as u64 * 1000),
            );
            self.thermal.erode(heightfield, self.thermal_iterations);
        }
    }
}

/// Simple PRNG for terrain generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightfield_creation() {
        let hf = Heightfield::new(10, 10);
        assert_eq!(hf.width(), 10);
        assert_eq!(hf.depth(), 10);
        assert_eq!(hf.get(5, 5), 0.0);
    }

    #[test]
    fn test_heightfield_set_get() {
        let mut hf = Heightfield::new(5, 5);
        hf.set(2, 3, 1.5);
        assert_eq!(hf.get(2, 3), 1.5);
    }

    #[test]
    fn test_heightfield_add() {
        let mut hf = Heightfield::new(5, 5);
        hf.set(2, 2, 1.0);
        hf.add(2, 2, 0.5);
        assert!((hf.get(2, 2) - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_heightfield_sample() {
        let mut hf = Heightfield::new(3, 3);
        // Set all corners and middle to create a predictable gradient
        for z in 0..3 {
            for x in 0..3 {
                hf.set(x, z, (x + z) as f32 / 4.0);
            }
        }

        // Sample at center should be average
        let center = hf.sample(0.5, 0.5);
        assert!(center > 0.0 && center < 1.0);
    }

    #[test]
    fn test_heightfield_gradient() {
        let mut hf = Heightfield::new(5, 5);
        // Create a slope in x direction
        for x in 0..5 {
            for z in 0..5 {
                hf.set(x, z, x as f32);
            }
        }

        let grad = hf.gradient(2, 2);
        assert!(grad.x > 0.5); // Positive x gradient
        assert!(grad.y.abs() < 0.1); // No z gradient
    }

    #[test]
    fn test_heightfield_normal() {
        let hf = Heightfield::new(5, 5);
        let normal = hf.normal(2, 2);

        // Flat surface should have upward normal
        assert!(normal.y > 0.99);
    }

    #[test]
    fn test_diamond_square() {
        let mut hf = Heightfield::new(17, 17);
        hf.apply_diamond_square(0.5, 12345);

        // Should have varied heights
        let min = hf.heights().iter().copied().fold(f32::INFINITY, f32::min);
        let max = hf
            .heights()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max > min);
    }

    #[test]
    fn test_heightfield_normalize() {
        let mut hf = Heightfield::new(5, 5);
        hf.set(0, 0, -10.0);
        hf.set(4, 4, 10.0);
        hf.normalize();

        let min = hf.heights().iter().copied().fold(f32::INFINITY, f32::min);
        let max = hf
            .heights()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_heightfield_blur() {
        let mut hf = Heightfield::new(10, 10);
        hf.set(5, 5, 1.0); // Single spike

        hf.blur(2);

        // Spike should be reduced
        assert!(hf.get(5, 5) < 1.0);
        // Neighbors should have some height
        assert!(hf.get(4, 5) > 0.0);
    }

    #[test]
    fn test_heightfield_to_mesh() {
        let mut hf = Heightfield::new(4, 4);
        hf.apply_diamond_square(0.3, 12345);

        let mesh = hf.to_mesh(10.0, 2.0);

        assert_eq!(mesh.positions.len(), 16);
        assert_eq!(mesh.normals.len(), 16);
        assert_eq!(mesh.uvs.len(), 16);
        assert_eq!(mesh.indices.len(), 3 * 3 * 2 * 3); // 3x3 quads, 2 tris each
    }

    #[test]
    fn test_hydraulic_erosion() {
        let mut hf = Heightfield::new(32, 32);
        hf.apply_diamond_square(0.5, 12345);
        let original_heights = hf.heights().to_vec();

        let erosion = HydraulicErosion::default();
        erosion.erode(&mut hf, 1000);

        // Heights should change
        let changed = hf
            .heights()
            .iter()
            .zip(original_heights.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed);
    }

    #[test]
    fn test_thermal_erosion() {
        let mut hf = Heightfield::new(10, 10);
        // Create a steep spike
        hf.set(5, 5, 10.0);

        let erosion = ThermalErosion::default();
        erosion.erode(&mut hf, 100);

        // Spike should be reduced
        assert!(hf.get(5, 5) < 10.0);
        // Material should spread to neighbors
        assert!(hf.get(4, 5) > 0.0);
    }

    #[test]
    fn test_thermal_erosion_settings() {
        let mut erosion = ThermalErosion::new();
        erosion.talus_angle = 0.3;
        erosion = erosion.with_transfer_rate(0.8);

        assert_eq!(erosion.talus_angle, 0.3);
        assert_eq!(erosion.transfer_rate, 0.8);
    }

    #[test]
    fn test_hydraulic_erosion_settings() {
        let mut erosion = HydraulicErosion::new().with_inertia(0.1).with_erosion(0.5);
        erosion.radius = 5;

        assert_eq!(erosion.inertia, 0.1);
        assert_eq!(erosion.erosion, 0.5);
        assert_eq!(erosion.radius, 5);
    }

    #[test]
    fn test_combined_erosion() {
        let mut hf = Heightfield::new(32, 32);
        hf.apply_diamond_square(0.5, 12345);

        let erosion = CombinedErosion::default();
        erosion.erode(&mut hf, 1);

        // Should complete without error
        assert!(hf.heights().iter().all(|h| h.is_finite()));
    }

    #[test]
    fn test_heightfield_fill() {
        let mut hf = Heightfield::new(5, 5);
        hf.fill(0.5);

        assert!(hf.heights().iter().all(|&h| (h - 0.5).abs() < 0.001));
    }

    #[test]
    fn test_heightfield_clamp() {
        let mut hf = Heightfield::new(5, 5);
        hf.set(0, 0, -1.0);
        hf.set(4, 4, 2.0);
        hf.clamp(0.0, 1.0);

        assert_eq!(hf.get(0, 0), 0.0);
        assert_eq!(hf.get(4, 4), 1.0);
    }

    #[test]
    fn test_deterministic() {
        let mut hf1 = Heightfield::new(16, 16);
        let mut hf2 = Heightfield::new(16, 16);

        hf1.apply_diamond_square(0.5, 99999);
        hf2.apply_diamond_square(0.5, 99999);

        assert_eq!(hf1.heights(), hf2.heights());
    }
}
