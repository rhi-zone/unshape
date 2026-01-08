//! Instancing and scattering system for distributing objects in space.
//!
//! # Example
//!
//! ```ignore
//! use resin_core::{ScatterConfig, scatter_grid, scatter_random, Instance};
//! use glam::Vec3;
//!
//! // Random scatter in a box
//! let instances = scatter_random(Vec3::ZERO, Vec3::ONE, 100, 42);
//!
//! // Grid scatter
//! let grid_instances = scatter_grid(Vec3::ZERO, Vec3::ONE, [10, 10, 10]);
//! ```

use glam::{Mat4, Quat, Vec2, Vec3};

/// A single instance with transform data.
#[derive(Debug, Clone, Copy)]
pub struct Instance {
    /// Instance position.
    pub position: Vec3,
    /// Instance rotation as a quaternion.
    pub rotation: Quat,
    /// Instance scale.
    pub scale: Vec3,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Instance {
    /// Creates a new instance at the given position.
    pub fn at(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Creates an instance with position and scale.
    pub fn with_scale(position: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            scale,
            ..Default::default()
        }
    }

    /// Creates an instance with position and uniform scale.
    pub fn with_uniform_scale(position: Vec3, scale: f32) -> Self {
        Self {
            position,
            scale: Vec3::splat(scale),
            ..Default::default()
        }
    }

    /// Creates an instance with full transform.
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Returns the transform matrix for this instance.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Transforms a point by this instance's transform.
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.position + self.rotation * (point * self.scale)
    }
}

/// Configuration for scattering operations.
#[derive(Debug, Clone)]
pub struct ScatterConfig {
    /// Random seed.
    pub seed: u64,
    /// Minimum scale (for random scaling).
    pub min_scale: f32,
    /// Maximum scale (for random scaling).
    pub max_scale: f32,
    /// Whether to apply random rotation.
    pub random_rotation: bool,
    /// Alignment axis for oriented scatter (e.g., up vector).
    pub align_axis: Option<Vec3>,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            min_scale: 1.0,
            max_scale: 1.0,
            random_rotation: false,
            align_axis: None,
        }
    }
}

impl ScatterConfig {
    /// Sets the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets scale range.
    pub fn with_scale_range(mut self, min: f32, max: f32) -> Self {
        self.min_scale = min;
        self.max_scale = max;
        self
    }

    /// Enables random rotation.
    pub fn with_random_rotation(mut self) -> Self {
        self.random_rotation = true;
        self
    }

    /// Sets alignment axis.
    pub fn with_alignment(mut self, axis: Vec3) -> Self {
        self.align_axis = Some(axis);
        self
    }
}

/// Simple LCG random number generator for deterministic scattering.
struct Rng {
    state: u64,
}

impl Rng {
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
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Scatters instances randomly within a box volume.
pub fn scatter_random(min: Vec3, max: Vec3, count: usize, seed: u64) -> Vec<Instance> {
    scatter_random_with_config(min, max, count, ScatterConfig::default().with_seed(seed))
}

/// Scatters instances randomly with full configuration.
pub fn scatter_random_with_config(
    min: Vec3,
    max: Vec3,
    count: usize,
    config: ScatterConfig,
) -> Vec<Instance> {
    let mut rng = Rng::new(config.seed);
    let mut instances = Vec::with_capacity(count);

    for _ in 0..count {
        let position = Vec3::new(
            rng.range(min.x, max.x),
            rng.range(min.y, max.y),
            rng.range(min.z, max.z),
        );

        let scale = Vec3::splat(rng.range(config.min_scale, config.max_scale));

        let rotation = if config.random_rotation {
            random_rotation(&mut rng, config.align_axis)
        } else {
            Quat::IDENTITY
        };

        instances.push(Instance::new(position, rotation, scale));
    }

    instances
}

/// Scatters instances on a regular 3D grid.
pub fn scatter_grid(min: Vec3, max: Vec3, resolution: [usize; 3]) -> Vec<Instance> {
    let mut instances = Vec::with_capacity(resolution[0] * resolution[1] * resolution[2]);

    let step = (max - min)
        / Vec3::new(
            (resolution[0].max(1) - 1).max(1) as f32,
            (resolution[1].max(1) - 1).max(1) as f32,
            (resolution[2].max(1) - 1).max(1) as f32,
        );

    for z in 0..resolution[2] {
        for y in 0..resolution[1] {
            for x in 0..resolution[0] {
                let position = min + step * Vec3::new(x as f32, y as f32, z as f32);
                instances.push(Instance::at(position));
            }
        }
    }

    instances
}

/// Scatters instances on a 2D grid (XZ plane).
pub fn scatter_grid_2d(min: Vec2, max: Vec2, resolution: [usize; 2], y: f32) -> Vec<Instance> {
    let mut instances = Vec::with_capacity(resolution[0] * resolution[1]);

    let step = (max - min)
        / Vec2::new(
            (resolution[0].max(1) - 1).max(1) as f32,
            (resolution[1].max(1) - 1).max(1) as f32,
        );

    for z in 0..resolution[1] {
        for x in 0..resolution[0] {
            let pos_2d = min + step * Vec2::new(x as f32, z as f32);
            let position = Vec3::new(pos_2d.x, y, pos_2d.y);
            instances.push(Instance::at(position));
        }
    }

    instances
}

/// Scatters instances on a sphere surface.
pub fn scatter_sphere(center: Vec3, radius: f32, count: usize, seed: u64) -> Vec<Instance> {
    let mut rng = Rng::new(seed);
    let mut instances = Vec::with_capacity(count);

    for _ in 0..count {
        // Fibonacci sphere distribution for even spacing
        let theta = rng.range(0.0, std::f32::consts::TAU);
        let phi = (1.0 - 2.0 * rng.next_f32()).acos();

        let position = center
            + radius * Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());

        // Align to surface normal
        let normal = (position - center).normalize_or_zero();
        let rotation = Quat::from_rotation_arc(Vec3::Y, normal);

        instances.push(Instance::new(position, rotation, Vec3::ONE));
    }

    instances
}

/// Scatters instances using Poisson disk sampling for even distribution.
pub fn scatter_poisson_2d(
    min: Vec2,
    max: Vec2,
    min_distance: f32,
    seed: u64,
    max_attempts: usize,
) -> Vec<Instance> {
    let mut rng = Rng::new(seed);
    let mut instances = Vec::new();
    let mut active: Vec<usize> = Vec::new();

    // Cell size for spatial hashing
    let cell_size = min_distance / std::f32::consts::SQRT_2;
    let grid_width = ((max.x - min.x) / cell_size).ceil() as usize + 1;
    let grid_height = ((max.y - min.y) / cell_size).ceil() as usize + 1;
    let mut grid: Vec<Option<usize>> = vec![None; grid_width * grid_height];

    // Helper to get grid cell
    let get_cell = |p: Vec2| -> (usize, usize) {
        (
            ((p.x - min.x) / cell_size) as usize,
            ((p.y - min.y) / cell_size) as usize,
        )
    };

    // Start with random point
    let first = Vec2::new(rng.range(min.x, max.x), rng.range(min.y, max.y));
    instances.push(Instance::at(Vec3::new(first.x, 0.0, first.y)));
    active.push(0);

    let (cx, cy) = get_cell(first);
    if cx < grid_width && cy < grid_height {
        grid[cy * grid_width + cx] = Some(0);
    }

    while !active.is_empty() {
        let idx = (rng.next_u64() as usize) % active.len();
        let active_idx = active[idx];
        let base_pos = Vec2::new(
            instances[active_idx].position.x,
            instances[active_idx].position.z,
        );

        let mut found = false;

        for _ in 0..max_attempts {
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let dist = rng.range(min_distance, min_distance * 2.0);
            let candidate = base_pos + dist * Vec2::new(angle.cos(), angle.sin());

            if candidate.x < min.x
                || candidate.x > max.x
                || candidate.y < min.y
                || candidate.y > max.y
            {
                continue;
            }

            let (cx, cy) = get_cell(candidate);

            // Check neighboring cells
            let mut valid = true;
            let search_radius = 2i32;

            'outer: for dy in -search_radius..=search_radius {
                for dx in -search_radius..=search_radius {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;

                    if nx < 0 || ny < 0 || nx >= grid_width as i32 || ny >= grid_height as i32 {
                        continue;
                    }

                    if let Some(neighbor_idx) = grid[ny as usize * grid_width + nx as usize] {
                        let neighbor_pos = Vec2::new(
                            instances[neighbor_idx].position.x,
                            instances[neighbor_idx].position.z,
                        );
                        if candidate.distance(neighbor_pos) < min_distance {
                            valid = false;
                            break 'outer;
                        }
                    }
                }
            }

            if valid {
                let new_idx = instances.len();
                instances.push(Instance::at(Vec3::new(candidate.x, 0.0, candidate.y)));
                active.push(new_idx);

                if cx < grid_width && cy < grid_height {
                    grid[cy * grid_width + cx] = Some(new_idx);
                }

                found = true;
                break;
            }
        }

        if !found {
            active.swap_remove(idx);
        }
    }

    instances
}

/// Scatters instances along a line.
pub fn scatter_line(start: Vec3, end: Vec3, count: usize) -> Vec<Instance> {
    let mut instances = Vec::with_capacity(count);

    for i in 0..count {
        let t = if count > 1 {
            i as f32 / (count - 1) as f32
        } else {
            0.5
        };
        let position = start.lerp(end, t);
        instances.push(Instance::at(position));
    }

    instances
}

/// Scatters instances in a circular pattern.
pub fn scatter_circle(center: Vec3, radius: f32, count: usize) -> Vec<Instance> {
    let mut instances = Vec::with_capacity(count);
    let step = std::f32::consts::TAU / count as f32;

    for i in 0..count {
        let angle = i as f32 * step;
        let position = center + radius * Vec3::new(angle.cos(), 0.0, angle.sin());

        // Face outward from center
        let rotation = Quat::from_rotation_y(-angle);

        instances.push(Instance::new(position, rotation, Vec3::ONE));
    }

    instances
}

/// Generates a random rotation quaternion.
fn random_rotation(rng: &mut Rng, align_axis: Option<Vec3>) -> Quat {
    if let Some(axis) = align_axis {
        // Random rotation around alignment axis
        let angle = rng.range(0.0, std::f32::consts::TAU);
        Quat::from_axis_angle(axis.normalize_or_zero(), angle)
    } else {
        // Full random rotation
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();
        let u3 = rng.next_f32();

        let sqrt_u1 = u1.sqrt();
        let sqrt_1_u1 = (1.0 - u1).sqrt();

        let tau = std::f32::consts::TAU;
        Quat::from_xyzw(
            sqrt_1_u1 * (tau * u2).sin(),
            sqrt_1_u1 * (tau * u2).cos(),
            sqrt_u1 * (tau * u3).sin(),
            sqrt_u1 * (tau * u3).cos(),
        )
    }
}

/// Applies random scale variation to existing instances.
pub fn randomize_scale(instances: &mut [Instance], min: f32, max: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    for inst in instances {
        let scale = rng.range(min, max);
        inst.scale *= scale;
    }
}

/// Applies random rotation to existing instances.
pub fn randomize_rotation(instances: &mut [Instance], seed: u64) {
    let mut rng = Rng::new(seed);
    for inst in instances {
        inst.rotation = random_rotation(&mut rng, None);
    }
}

/// Jitters instance positions by a random offset.
pub fn jitter_positions(instances: &mut [Instance], amount: Vec3, seed: u64) {
    let mut rng = Rng::new(seed);
    for inst in instances {
        inst.position += Vec3::new(
            rng.range(-amount.x, amount.x),
            rng.range(-amount.y, amount.y),
            rng.range(-amount.z, amount.z),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_default() {
        let inst = Instance::default();
        assert_eq!(inst.position, Vec3::ZERO);
        assert_eq!(inst.rotation, Quat::IDENTITY);
        assert_eq!(inst.scale, Vec3::ONE);
    }

    #[test]
    fn test_instance_matrix() {
        let inst = Instance::with_uniform_scale(Vec3::new(1.0, 2.0, 3.0), 2.0);
        let mat = inst.matrix();

        // Transform origin should give position
        let transformed = mat.transform_point3(Vec3::ZERO);
        assert!((transformed - inst.position).length() < 0.001);
    }

    #[test]
    fn test_scatter_random() {
        let instances = scatter_random(Vec3::ZERO, Vec3::ONE, 100, 42);

        assert_eq!(instances.len(), 100);

        // All instances should be within bounds
        for inst in &instances {
            assert!(inst.position.x >= 0.0 && inst.position.x <= 1.0);
            assert!(inst.position.y >= 0.0 && inst.position.y <= 1.0);
            assert!(inst.position.z >= 0.0 && inst.position.z <= 1.0);
        }
    }

    #[test]
    fn test_scatter_grid() {
        let instances = scatter_grid(Vec3::ZERO, Vec3::ONE, [3, 3, 3]);

        assert_eq!(instances.len(), 27);
    }

    #[test]
    fn test_scatter_grid_2d() {
        let instances = scatter_grid_2d(Vec2::ZERO, Vec2::ONE, [5, 5], 0.0);

        assert_eq!(instances.len(), 25);

        // All should be at y=0
        for inst in &instances {
            assert!((inst.position.y - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_scatter_sphere() {
        let instances = scatter_sphere(Vec3::ZERO, 1.0, 50, 42);

        assert_eq!(instances.len(), 50);

        // All should be at approximately radius distance
        for inst in &instances {
            let dist = inst.position.length();
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_scatter_poisson() {
        let instances = scatter_poisson_2d(Vec2::ZERO, Vec2::splat(10.0), 1.0, 42, 30);

        // Should have generated some instances
        assert!(instances.len() > 0);

        // Check minimum distance constraint
        for i in 0..instances.len() {
            for j in (i + 1)..instances.len() {
                let pi = Vec2::new(instances[i].position.x, instances[i].position.z);
                let pj = Vec2::new(instances[j].position.x, instances[j].position.z);
                let dist = pi.distance(pj);
                assert!(dist >= 0.99, "Points too close: {} < 1.0", dist);
            }
        }
    }

    #[test]
    fn test_scatter_line() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 11);

        assert_eq!(instances.len(), 11);

        // Check evenly spaced
        for (i, inst) in instances.iter().enumerate() {
            assert!((inst.position.x - i as f32).abs() < 0.001);
        }
    }

    #[test]
    fn test_scatter_circle() {
        let instances = scatter_circle(Vec3::ZERO, 1.0, 8);

        assert_eq!(instances.len(), 8);

        // All should be at radius distance from center
        for inst in &instances {
            let dist = Vec2::new(inst.position.x, inst.position.z).length();
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_jitter_positions() {
        let mut instances = vec![Instance::at(Vec3::ZERO), Instance::at(Vec3::ONE)];

        jitter_positions(&mut instances, Vec3::splat(0.1), 42);

        // Positions should have changed
        assert!(instances[0].position.length() > 0.0);
    }

    #[test]
    fn test_randomize_scale() {
        let mut instances = vec![Instance::at(Vec3::ZERO), Instance::at(Vec3::ONE)];

        randomize_scale(&mut instances, 0.5, 1.5, 42);

        // Scales should be in range
        for inst in &instances {
            assert!(inst.scale.x >= 0.5 && inst.scale.x <= 1.5);
        }
    }

    #[test]
    fn test_deterministic() {
        // Same seed should give same results
        let a = scatter_random(Vec3::ZERO, Vec3::ONE, 10, 42);
        let b = scatter_random(Vec3::ZERO, Vec3::ONE, 10, 42);

        for (ia, ib) in a.iter().zip(b.iter()) {
            assert!((ia.position - ib.position).length() < 0.0001);
        }
    }
}
