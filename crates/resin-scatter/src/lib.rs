//! Instancing and scattering system for distributing objects in space.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_core::{ScatterConfig, scatter_grid, scatter_random, Instance};
//! use glam::Vec3;
//!
//! // Random scatter in a box
//! let instances = scatter_random(Vec3::ZERO, Vec3::ONE, 100, 42);
//!
//! // Grid scatter
//! let grid_instances = scatter_grid(Vec3::ZERO, Vec3::ONE, [10, 10, 10]);
//! ```

use glam::{Mat4, Quat, Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single instance with transform data.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// Scatters instances randomly within a box volume.
///
/// A complete operation that combines volume bounds, count, and scatter
/// configuration into a single serializable struct.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Vec<Instance>))]
pub struct Scatter {
    /// Minimum bounds of the scatter volume.
    pub min: Vec3,
    /// Maximum bounds of the scatter volume.
    pub max: Vec3,
    /// Number of instances to generate.
    pub count: usize,
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

impl Default for Scatter {
    fn default() -> Self {
        Self {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            count: 100,
            seed: 42,
            min_scale: 1.0,
            max_scale: 1.0,
            random_rotation: false,
            align_axis: None,
        }
    }
}

impl Scatter {
    /// Creates a new scatter operation with given bounds and count.
    pub fn new(min: Vec3, max: Vec3, count: usize) -> Self {
        Self {
            min,
            max,
            count,
            ..Default::default()
        }
    }

    /// Applies this operation to generate instances.
    pub fn apply(&self) -> Vec<Instance> {
        scatter_random_with_config(self.min, self.max, self.count, self.clone())
    }
}

/// Backwards-compatible type alias.
pub type ScatterConfig = Scatter;

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
    scatter_random_with_config(
        min,
        max,
        count,
        ScatterConfig {
            seed,
            ..Default::default()
        },
    )
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

// ===========================================================================
// Stagger Timing (Animation Offset)
// ===========================================================================

/// Pattern for distributing stagger delays across instances.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StaggerPattern {
    /// Linear distribution from first to last instance.
    #[default]
    Linear,
    /// Reversed linear distribution (last to first).
    Reverse,
    /// Start from the center, spread outward.
    CenterOut,
    /// Start from edges, converge to center.
    EdgesIn,
    /// Random delays (requires a seed).
    Random(u64),
    /// Based on distance from a reference point.
    FromPoint(Vec3),
    /// Based on 2D position (XZ plane distance from point).
    FromPoint2D(Vec2),
}

/// Configuration for stagger timing generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Stagger {
    /// Delay between each instance (in seconds or frames).
    pub delay: f32,
    /// Total duration to spread instances over (overrides delay if set).
    pub total_duration: Option<f32>,
    /// Pattern for distributing delays.
    pub pattern: StaggerPattern,
    /// Easing function for delay distribution (0 = linear, positive = ease-in, negative = ease-out).
    pub easing: f32,
}

impl Default for Stagger {
    fn default() -> Self {
        Self {
            delay: 0.1,
            total_duration: None,
            pattern: StaggerPattern::Linear,
            easing: 0.0,
        }
    }
}

impl Stagger {
    /// Creates a stagger config with a fixed delay between instances.
    pub fn with_delay(delay: f32) -> Self {
        Self {
            delay,
            ..Default::default()
        }
    }

    /// Creates a stagger config that spreads instances over a total duration.
    pub fn with_duration(duration: f32) -> Self {
        Self {
            delay: 0.0,
            total_duration: Some(duration),
            ..Default::default()
        }
    }

    /// Sets the stagger pattern.
    pub fn pattern(mut self, pattern: StaggerPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Sets the easing factor.
    ///
    /// Positive values ease-in (slow start, fast end).
    /// Negative values ease-out (fast start, slow end).
    /// Zero is linear distribution.
    pub fn easing(mut self, easing: f32) -> Self {
        self.easing = easing;
        self
    }

    /// Sets linear pattern (first to last).
    pub fn linear(mut self) -> Self {
        self.pattern = StaggerPattern::Linear;
        self
    }

    /// Sets reverse pattern (last to first).
    pub fn reverse(mut self) -> Self {
        self.pattern = StaggerPattern::Reverse;
        self
    }

    /// Sets center-out pattern.
    pub fn center_out(mut self) -> Self {
        self.pattern = StaggerPattern::CenterOut;
        self
    }

    /// Sets edges-in pattern.
    pub fn edges_in(mut self) -> Self {
        self.pattern = StaggerPattern::EdgesIn;
        self
    }

    /// Sets random pattern with a seed.
    pub fn random(mut self, seed: u64) -> Self {
        self.pattern = StaggerPattern::Random(seed);
        self
    }

    /// Sets from-point pattern (3D distance).
    pub fn from_point(mut self, point: Vec3) -> Self {
        self.pattern = StaggerPattern::FromPoint(point);
        self
    }

    /// Sets from-point pattern (2D distance on XZ plane).
    pub fn from_point_2d(mut self, point: Vec2) -> Self {
        self.pattern = StaggerPattern::FromPoint2D(point);
        self
    }

    /// Calculates stagger delays for a set of instances.
    pub fn apply(&self, instances: &[Instance]) -> Vec<f32> {
        stagger_timing(instances, self)
    }

    /// Calculates the delay for a single index given a count.
    ///
    /// This is useful when you don't have Instance data and just need
    /// timing offsets for a known number of items.
    pub fn delay_for_index(&self, index: usize, count: usize) -> f32 {
        stagger_index(index, count, self)
    }
}

/// Result of stagger timing calculation.
#[derive(Debug, Clone)]
pub struct StaggerTiming {
    /// Delay values for each instance.
    pub delays: Vec<f32>,
    /// Total duration from first to last start.
    pub total_duration: f32,
}

impl StaggerTiming {
    /// Gets the delay for an instance at the given index.
    pub fn delay(&self, index: usize) -> f32 {
        self.delays.get(index).copied().unwrap_or(0.0)
    }

    /// Returns an iterator over (index, delay) pairs sorted by delay.
    pub fn sorted_by_delay(&self) -> Vec<(usize, f32)> {
        let mut indexed: Vec<_> = self.delays.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
    }
}

/// Calculates stagger timing for instances.
///
/// Returns a vector of delay values (in the same unit as `config.delay`).
/// Use these delays to offset animation start times.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_scatter::{scatter_circle, stagger_timing, Stagger};
/// use glam::Vec3;
///
/// let instances = scatter_circle(Vec3::ZERO, 100.0, 8);
/// let stagger = Stagger::with_delay(0.1).center_out();
/// let delays = stagger_timing(&instances, &stagger);
///
/// // Use in animation loop
/// for (i, inst) in instances.iter().enumerate() {
///     let start_time = delays[i];
///     // Animate instance starting at start_time
/// }
/// ```
pub fn stagger_timing(instances: &[Instance], config: &Stagger) -> Vec<f32> {
    let count = instances.len();
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0.0];
    }

    // Calculate normalized t values (0 to 1) based on pattern
    let mut t_values: Vec<f32> = match config.pattern {
        StaggerPattern::Linear => (0..count).map(|i| i as f32 / (count - 1) as f32).collect(),

        StaggerPattern::Reverse => (0..count)
            .map(|i| 1.0 - i as f32 / (count - 1) as f32)
            .collect(),

        StaggerPattern::CenterOut => {
            let center = (count - 1) as f32 / 2.0;
            let max_dist = center;
            (0..count)
                .map(|i| {
                    let dist = (i as f32 - center).abs();
                    dist / max_dist
                })
                .collect()
        }

        StaggerPattern::EdgesIn => {
            let center = (count - 1) as f32 / 2.0;
            let max_dist = center;
            (0..count)
                .map(|i| {
                    let dist = (i as f32 - center).abs();
                    1.0 - dist / max_dist
                })
                .collect()
        }

        StaggerPattern::Random(seed) => {
            let mut rng = Rng::new(seed);
            let mut values: Vec<f32> = (0..count).map(|_| rng.next_f32()).collect();
            // Normalize to 0-1 range
            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = (max - min).max(0.001);
            for v in &mut values {
                *v = (*v - min) / range;
            }
            values
        }

        StaggerPattern::FromPoint(point) => {
            let distances: Vec<f32> = instances
                .iter()
                .map(|inst| inst.position.distance(point))
                .collect();
            let max_dist = distances.iter().cloned().fold(0.0f32, f32::max).max(0.001);
            distances.iter().map(|d| d / max_dist).collect()
        }

        StaggerPattern::FromPoint2D(point) => {
            let distances: Vec<f32> = instances
                .iter()
                .map(|inst| Vec2::new(inst.position.x, inst.position.z).distance(point))
                .collect();
            let max_dist = distances.iter().cloned().fold(0.0f32, f32::max).max(0.001);
            distances.iter().map(|d| d / max_dist).collect()
        }
    };

    // Apply easing
    if config.easing != 0.0 {
        for t in &mut t_values {
            *t = apply_easing(*t, config.easing);
        }
    }

    // Convert to actual delays
    let max_delay = config
        .total_duration
        .unwrap_or((count - 1) as f32 * config.delay);

    t_values.iter().map(|t| t * max_delay).collect()
}

/// Calculates stagger timing and returns detailed results.
pub fn stagger_timing_detailed(instances: &[Instance], config: &Stagger) -> StaggerTiming {
    let delays = stagger_timing(instances, config);
    let total_duration = delays.iter().cloned().fold(0.0f32, f32::max);
    StaggerTiming {
        delays,
        total_duration,
    }
}

/// Calculates the stagger delay for a single index without instance data.
///
/// Useful when you just need timing offsets for a sequence of items.
pub fn stagger_index(index: usize, count: usize, config: &Stagger) -> f32 {
    if count == 0 || count == 1 {
        return 0.0;
    }

    let t = match config.pattern {
        StaggerPattern::Linear => index as f32 / (count - 1) as f32,
        StaggerPattern::Reverse => 1.0 - index as f32 / (count - 1) as f32,
        StaggerPattern::CenterOut => {
            let center = (count - 1) as f32 / 2.0;
            (index as f32 - center).abs() / center
        }
        StaggerPattern::EdgesIn => {
            let center = (count - 1) as f32 / 2.0;
            1.0 - (index as f32 - center).abs() / center
        }
        StaggerPattern::Random(seed) => {
            // Generate a deterministic random value for this index
            let mut rng = Rng::new(seed.wrapping_add(index as u64));
            rng.next_f32()
        }
        StaggerPattern::FromPoint(_) | StaggerPattern::FromPoint2D(_) => {
            // These patterns require instance positions, fall back to linear
            index as f32 / (count - 1) as f32
        }
    };

    let t = if config.easing != 0.0 {
        apply_easing(t, config.easing)
    } else {
        t
    };

    let max_delay = config
        .total_duration
        .unwrap_or((count - 1) as f32 * config.delay);

    t * max_delay
}

/// Applies a simple power easing to a t value.
fn apply_easing(t: f32, easing: f32) -> f32 {
    if easing > 0.0 {
        // Ease-in: slow start
        t.powf(1.0 + easing)
    } else if easing < 0.0 {
        // Ease-out: slow end
        1.0 - (1.0 - t).powf(1.0 - easing)
    } else {
        t
    }
}

/// Creates stagger delays for simple sequential animation.
///
/// This is a convenience function for the common case of linear stagger.
pub fn stagger_linear(count: usize, delay: f32) -> Vec<f32> {
    (0..count).map(|i| i as f32 * delay).collect()
}

/// Creates stagger delays that spread over a total duration.
pub fn stagger_spread(count: usize, total_duration: f32) -> Vec<f32> {
    if count <= 1 {
        return vec![0.0; count];
    }
    (0..count)
        .map(|i| i as f32 / (count - 1) as f32 * total_duration)
        .collect()
}

/// Registers all scatter operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of scatter ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<Scatter>("resin::Scatter");
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

    // Stagger tests

    #[test]
    fn test_stagger_default() {
        let stagger = Stagger::default();
        assert_eq!(stagger.delay, 0.1);
        assert_eq!(stagger.pattern, StaggerPattern::Linear);
    }

    #[test]
    fn test_stagger_with_delay() {
        let stagger = Stagger::with_delay(0.5);
        assert_eq!(stagger.delay, 0.5);
    }

    #[test]
    fn test_stagger_with_duration() {
        let stagger = Stagger::with_duration(2.0);
        assert_eq!(stagger.total_duration, Some(2.0));
    }

    #[test]
    fn test_stagger_linear() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_delay(0.1).apply(&instances);

        assert_eq!(delays.len(), 5);
        assert!((delays[0] - 0.0).abs() < 0.001);
        assert!((delays[1] - 0.1).abs() < 0.001);
        assert!((delays[2] - 0.2).abs() < 0.001);
        assert!((delays[3] - 0.3).abs() < 0.001);
        assert!((delays[4] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_stagger_reverse() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_delay(0.1).reverse().apply(&instances);

        assert!((delays[0] - 0.4).abs() < 0.001);
        assert!((delays[4] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_stagger_center_out() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_delay(0.1).center_out().apply(&instances);

        // Center (index 2) should start first
        assert!((delays[2] - 0.0).abs() < 0.001);
        // Edges should start last (max_delay = 4 * 0.1 = 0.4)
        assert!((delays[0] - 0.4).abs() < 0.001);
        assert!((delays[4] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_stagger_edges_in() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_delay(0.1).edges_in().apply(&instances);

        // Edges should start first
        assert!((delays[0] - 0.0).abs() < 0.001);
        assert!((delays[4] - 0.0).abs() < 0.001);
        // Center should start last (max_delay = 4 * 0.1 = 0.4)
        assert!((delays[2] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_stagger_with_duration_override() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_duration(2.0).apply(&instances);

        assert!((delays[0] - 0.0).abs() < 0.001);
        assert!((delays[4] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_stagger_from_point() {
        let instances = scatter_circle(Vec3::ZERO, 10.0, 8);
        let delays = Stagger::with_duration(1.0)
            .from_point(Vec3::ZERO)
            .apply(&instances);

        // All instances should have similar delays (all same distance from center)
        for d in &delays {
            assert!((*d - delays[0]).abs() < 0.001);
        }
    }

    #[test]
    fn test_stagger_random() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let delays = Stagger::with_delay(0.1).random(42).apply(&instances);

        // Should have 5 delays
        assert_eq!(delays.len(), 5);

        // Not all delays should be the same
        let all_same = delays.windows(2).all(|w| (w[0] - w[1]).abs() < 0.001);
        assert!(!all_same);
    }

    #[test]
    fn test_stagger_index() {
        let config = Stagger::with_delay(0.1);

        let d0 = stagger_index(0, 5, &config);
        let d2 = stagger_index(2, 5, &config);
        let d4 = stagger_index(4, 5, &config);

        assert!((d0 - 0.0).abs() < 0.001);
        assert!((d2 - 0.2).abs() < 0.001);
        assert!((d4 - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_stagger_linear_fn() {
        let delays = stagger_linear(5, 0.2);

        assert_eq!(delays.len(), 5);
        assert!((delays[0] - 0.0).abs() < 0.001);
        assert!((delays[4] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_stagger_spread() {
        let delays = stagger_spread(5, 2.0);

        assert_eq!(delays.len(), 5);
        assert!((delays[0] - 0.0).abs() < 0.001);
        assert!((delays[4] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_stagger_timing_detailed() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let timing = stagger_timing_detailed(&instances, &Stagger::with_delay(0.1));

        assert_eq!(timing.delays.len(), 5);
        assert!((timing.total_duration - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_stagger_sorted_by_delay() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 5);
        let timing = stagger_timing_detailed(&instances, &Stagger::with_delay(0.1).reverse());

        let sorted = timing.sorted_by_delay();

        // Should be sorted from smallest to largest delay
        assert_eq!(sorted[0].0, 4); // Last instance first
        assert_eq!(sorted[4].0, 0); // First instance last
    }

    #[test]
    fn test_stagger_single_instance() {
        let instances = vec![Instance::at(Vec3::ZERO)];
        let delays = Stagger::with_delay(0.1).apply(&instances);

        assert_eq!(delays.len(), 1);
        assert_eq!(delays[0], 0.0);
    }

    #[test]
    fn test_stagger_empty() {
        let instances: Vec<Instance> = vec![];
        let delays = Stagger::with_delay(0.1).apply(&instances);

        assert!(delays.is_empty());
    }

    #[test]
    fn test_stagger_easing_positive() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 3);
        let delays = Stagger::with_duration(1.0).easing(1.0).apply(&instances);

        // With positive easing, middle value should be < 0.5 (slow start)
        assert!(delays[1] < 0.5);
    }

    #[test]
    fn test_stagger_easing_negative() {
        let instances = scatter_line(Vec3::ZERO, Vec3::X * 10.0, 3);
        let delays = Stagger::with_duration(1.0).easing(-1.0).apply(&instances);

        // With negative easing, middle value should be > 0.5 (fast start)
        assert!(delays[1] > 0.5);
    }
}
