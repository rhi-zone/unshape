//! Cloth simulation with collision support.
//!
//! Position-based dynamics cloth simulation with support for
//! collision against spheres, planes, and capsules.

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for cloth simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = ClothConfig))]
pub struct ClothConfig {
    /// Number of constraint solver iterations.
    pub iterations: u32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Global damping factor (0-1).
    pub damping: f32,
    /// Stretch stiffness (0-1).
    pub stretch_stiffness: f32,
    /// Bend stiffness (0-1).
    pub bend_stiffness: f32,
    /// Collision margin (added to collider radii).
    pub collision_margin: f32,
    /// Friction coefficient for collisions.
    pub friction: f32,
}

impl Default for ClothConfig {
    fn default() -> Self {
        Self {
            iterations: 4,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            stretch_stiffness: 0.9,
            bend_stiffness: 0.5,
            collision_margin: 0.01,
            friction: 0.3,
        }
    }
}

impl ClothConfig {
    /// Applies this configuration (returns self as a generator op).
    pub fn apply(&self) -> ClothConfig {
        self.clone()
    }
}

/// A particle in the cloth simulation.
#[derive(Debug, Clone)]
pub struct ClothParticle {
    /// Current position.
    pub position: Vec3,
    /// Previous position (for Verlet integration).
    pub prev_position: Vec3,
    /// Inverse mass (0 = pinned).
    pub inv_mass: f32,
    /// Velocity (computed from positions).
    pub velocity: Vec3,
}

impl ClothParticle {
    /// Creates a new cloth particle.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            prev_position: position,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            velocity: Vec3::ZERO,
        }
    }

    /// Creates a pinned (immovable) particle.
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            inv_mass: 0.0,
            velocity: Vec3::ZERO,
        }
    }

    /// Returns whether this particle is pinned.
    pub fn is_pinned(&self) -> bool {
        self.inv_mass == 0.0
    }

    /// Pins this particle in place.
    pub fn pin(&mut self) {
        self.inv_mass = 0.0;
    }

    /// Unpins this particle with the given mass.
    pub fn unpin(&mut self, mass: f32) {
        self.inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    }
}

/// A distance constraint between two particles.
#[derive(Debug, Clone, Copy)]
pub struct DistanceConstraint {
    /// First particle index.
    pub p0: usize,
    /// Second particle index.
    pub p1: usize,
    /// Rest length.
    pub rest_length: f32,
    /// Stiffness (0-1).
    pub stiffness: f32,
}

impl DistanceConstraint {
    /// Creates a new distance constraint.
    pub fn new(p0: usize, p1: usize, rest_length: f32, stiffness: f32) -> Self {
        Self {
            p0,
            p1,
            rest_length,
            stiffness,
        }
    }
}

/// Collision shape for cloth interaction.
#[derive(Debug, Clone)]
pub enum ClothCollider {
    /// Sphere collider.
    Sphere {
        /// Center position.
        center: Vec3,
        /// Radius.
        radius: f32,
    },
    /// Infinite plane.
    Plane {
        /// Point on plane.
        point: Vec3,
        /// Plane normal (pointing outward).
        normal: Vec3,
    },
    /// Capsule collider.
    Capsule {
        /// Start point of axis.
        start: Vec3,
        /// End point of axis.
        end: Vec3,
        /// Radius.
        radius: f32,
    },
    /// Axis-aligned box.
    Box {
        /// Minimum corner.
        min: Vec3,
        /// Maximum corner.
        max: Vec3,
    },
}

impl ClothCollider {
    /// Creates a sphere collider.
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        Self::Sphere { center, radius }
    }

    /// Creates a plane collider.
    pub fn plane(point: Vec3, normal: Vec3) -> Self {
        Self::Plane {
            point,
            normal: normal.normalize(),
        }
    }

    /// Creates a ground plane at y=0.
    pub fn ground() -> Self {
        Self::plane(Vec3::ZERO, Vec3::Y)
    }

    /// Creates a capsule collider.
    pub fn capsule(start: Vec3, end: Vec3, radius: f32) -> Self {
        Self::Capsule { start, end, radius }
    }

    /// Creates a box collider.
    pub fn aabb(min: Vec3, max: Vec3) -> Self {
        Self::Box { min, max }
    }
}

/// Result of a collision query.
#[derive(Debug, Clone, Copy)]
pub struct CollisionResult {
    /// Whether collision occurred.
    pub collided: bool,
    /// Closest point on collider surface.
    pub closest_point: Vec3,
    /// Normal at collision point (pointing away from collider).
    pub normal: Vec3,
    /// Penetration depth (positive if inside).
    pub depth: f32,
}

impl CollisionResult {
    /// No collision result.
    pub fn none() -> Self {
        Self {
            collided: false,
            closest_point: Vec3::ZERO,
            normal: Vec3::ZERO,
            depth: 0.0,
        }
    }
}

/// A cloth mesh simulation.
#[derive(Debug, Clone)]
pub struct Cloth {
    /// Particles in the cloth.
    pub particles: Vec<ClothParticle>,
    /// Distance constraints (stretch).
    pub stretch_constraints: Vec<DistanceConstraint>,
    /// Bending constraints.
    pub bend_constraints: Vec<DistanceConstraint>,
    /// Colliders to interact with.
    pub colliders: Vec<ClothCollider>,
    /// Simulation configuration.
    pub config: ClothConfig,
    /// Grid width (for rectangular cloth).
    pub width: usize,
    /// Grid height (for rectangular cloth).
    pub height: usize,
}

impl Cloth {
    /// Creates a rectangular cloth grid.
    ///
    /// # Arguments
    /// * `origin` - Top-left corner of the cloth
    /// * `width_size` - Width of cloth in world units
    /// * `height_size` - Height of cloth in world units
    /// * `width_segments` - Number of particles along width
    /// * `height_segments` - Number of particles along height
    /// * `mass` - Total mass of cloth (distributed among particles)
    pub fn grid(
        origin: Vec3,
        width_size: f32,
        height_size: f32,
        width_segments: usize,
        height_segments: usize,
        mass: f32,
    ) -> Self {
        let total_particles = width_segments * height_segments;
        let particle_mass = mass / total_particles as f32;

        let dx = width_size / (width_segments - 1).max(1) as f32;
        let dy = height_size / (height_segments - 1).max(1) as f32;

        // Create particles
        let mut particles = Vec::with_capacity(total_particles);
        for y in 0..height_segments {
            for x in 0..width_segments {
                let pos = origin + Vec3::new(x as f32 * dx, 0.0, y as f32 * dy);
                particles.push(ClothParticle::new(pos, particle_mass));
            }
        }

        // Create stretch constraints (horizontal and vertical)
        let mut stretch_constraints = Vec::new();
        for y in 0..height_segments {
            for x in 0..width_segments {
                let idx = y * width_segments + x;

                // Horizontal
                if x < width_segments - 1 {
                    stretch_constraints.push(DistanceConstraint::new(idx, idx + 1, dx, 1.0));
                }

                // Vertical
                if y < height_segments - 1 {
                    stretch_constraints.push(DistanceConstraint::new(
                        idx,
                        idx + width_segments,
                        dy,
                        1.0,
                    ));
                }

                // Diagonal (for shear)
                if x < width_segments - 1 && y < height_segments - 1 {
                    let diag_len = (dx * dx + dy * dy).sqrt();
                    stretch_constraints.push(DistanceConstraint::new(
                        idx,
                        idx + width_segments + 1,
                        diag_len,
                        0.8,
                    ));
                    stretch_constraints.push(DistanceConstraint::new(
                        idx + 1,
                        idx + width_segments,
                        diag_len,
                        0.8,
                    ));
                }
            }
        }

        // Create bend constraints (skip one particle)
        let mut bend_constraints = Vec::new();
        for y in 0..height_segments {
            for x in 0..width_segments {
                let idx = y * width_segments + x;

                // Horizontal bend
                if x < width_segments - 2 {
                    bend_constraints.push(DistanceConstraint::new(idx, idx + 2, dx * 2.0, 1.0));
                }

                // Vertical bend
                if y < height_segments - 2 {
                    bend_constraints.push(DistanceConstraint::new(
                        idx,
                        idx + width_segments * 2,
                        dy * 2.0,
                        1.0,
                    ));
                }
            }
        }

        Self {
            particles,
            stretch_constraints,
            bend_constraints,
            colliders: Vec::new(),
            config: ClothConfig::default(),
            width: width_segments,
            height: height_segments,
        }
    }

    /// Creates cloth from arbitrary particle positions and triangles.
    pub fn from_mesh(positions: &[Vec3], triangles: &[[usize; 3]], mass: f32) -> Self {
        let particle_mass = mass / positions.len() as f32;

        let particles: Vec<ClothParticle> = positions
            .iter()
            .map(|&pos| ClothParticle::new(pos, particle_mass))
            .collect();

        // Build edge set from triangles
        let mut edges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for tri in triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                edges.insert(edge);
            }
        }

        // Create stretch constraints from edges
        let stretch_constraints: Vec<DistanceConstraint> = edges
            .iter()
            .map(|&(a, b)| {
                let rest_len = (positions[a] - positions[b]).length();
                DistanceConstraint::new(a, b, rest_len, 1.0)
            })
            .collect();

        Self {
            particles,
            stretch_constraints,
            bend_constraints: Vec::new(),
            colliders: Vec::new(),
            config: ClothConfig::default(),
            width: 0,
            height: 0,
        }
    }

    /// Adds a collider.
    pub fn add_collider(&mut self, collider: ClothCollider) {
        self.colliders.push(collider);
    }

    /// Pins particles at specific indices.
    pub fn pin(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.particles.len() {
                self.particles[idx].pin();
            }
        }
    }

    /// Pins particles in the top row (for rectangular cloth).
    pub fn pin_top_row(&mut self) {
        for x in 0..self.width {
            self.particles[x].pin();
        }
    }

    /// Pins corners of a rectangular cloth.
    pub fn pin_corners(&mut self) {
        if self.width > 0 && self.height > 0 {
            self.particles[0].pin(); // Top-left
            self.particles[self.width - 1].pin(); // Top-right
        }
    }

    /// Gets particle index in grid layout.
    pub fn grid_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Steps the simulation forward.
    pub fn step(&mut self, dt: f32) {
        let gravity = self.config.gravity;
        let damping = self.config.damping;

        // Apply external forces and integrate
        for particle in &mut self.particles {
            if particle.is_pinned() {
                continue;
            }

            // Compute velocity from position difference
            particle.velocity = particle.position - particle.prev_position;

            // Apply damping
            particle.velocity *= 1.0 - damping;

            // Store previous position
            particle.prev_position = particle.position;

            // Verlet integration: x_new = x + v + a*dt^2
            particle.position += particle.velocity + gravity * dt * dt;
        }

        // Solve constraints iteratively
        for _ in 0..self.config.iterations {
            // Stretch constraints
            let stiffness = self.config.stretch_stiffness;
            self.solve_constraints(&self.stretch_constraints.clone(), stiffness);

            // Bend constraints
            let bend_stiffness = self.config.bend_stiffness;
            self.solve_constraints(&self.bend_constraints.clone(), bend_stiffness);

            // Collision constraints
            self.solve_collisions();
        }
    }

    /// Solves distance constraints.
    fn solve_constraints(&mut self, constraints: &[DistanceConstraint], global_stiffness: f32) {
        for constraint in constraints {
            let p0 = &self.particles[constraint.p0];
            let p1 = &self.particles[constraint.p1];

            let delta = p1.position - p0.position;
            let current_length = delta.length();

            if current_length < 0.0001 {
                continue;
            }

            let error = current_length - constraint.rest_length;
            let direction = delta / current_length;

            let w0 = p0.inv_mass;
            let w1 = p1.inv_mass;
            let total_weight = w0 + w1;

            if total_weight < 0.0001 {
                continue;
            }

            let stiffness = constraint.stiffness * global_stiffness;
            let correction = direction * error * stiffness / total_weight;

            // Apply corrections
            if !self.particles[constraint.p0].is_pinned() {
                self.particles[constraint.p0].position += correction * w0;
            }
            if !self.particles[constraint.p1].is_pinned() {
                self.particles[constraint.p1].position -= correction * w1;
            }
        }
    }

    /// Solves collision constraints.
    fn solve_collisions(&mut self) {
        let margin = self.config.collision_margin;
        let friction = self.config.friction;

        for particle in &mut self.particles {
            if particle.is_pinned() {
                continue;
            }

            for collider in &self.colliders {
                let result = query_collision(collider, particle.position, margin);

                if result.collided && result.depth > 0.0 {
                    // Push particle out of collider
                    particle.position += result.normal * result.depth;

                    // Apply friction
                    let velocity = particle.position - particle.prev_position;
                    let vn = velocity.dot(result.normal);
                    let tangent_velocity = velocity - result.normal * vn;

                    particle.position -= tangent_velocity * friction;
                }
            }
        }
    }

    /// Gets all particle positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Gets particle normals (for rendering).
    pub fn normals(&self) -> Vec<Vec3> {
        if self.width == 0 || self.height == 0 {
            return vec![Vec3::Y; self.particles.len()];
        }

        let mut normals = vec![Vec3::ZERO; self.particles.len()];

        // Compute normals from adjacent particles in grid
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.grid_index(x, y);
                let pos = self.particles[idx].position;

                let mut normal = Vec3::ZERO;
                let mut count = 0;

                // Use cross products of adjacent edges
                if x > 0 && y > 0 {
                    let left = self.particles[self.grid_index(x - 1, y)].position;
                    let up = self.particles[self.grid_index(x, y - 1)].position;
                    normal += (left - pos).cross(up - pos);
                    count += 1;
                }

                if x < self.width - 1 && y > 0 {
                    let right = self.particles[self.grid_index(x + 1, y)].position;
                    let up = self.particles[self.grid_index(x, y - 1)].position;
                    normal += (up - pos).cross(right - pos);
                    count += 1;
                }

                if x < self.width - 1 && y < self.height - 1 {
                    let right = self.particles[self.grid_index(x + 1, y)].position;
                    let down = self.particles[self.grid_index(x, y + 1)].position;
                    normal += (right - pos).cross(down - pos);
                    count += 1;
                }

                if x > 0 && y < self.height - 1 {
                    let left = self.particles[self.grid_index(x - 1, y)].position;
                    let down = self.particles[self.grid_index(x, y + 1)].position;
                    normal += (down - pos).cross(left - pos);
                    count += 1;
                }

                if count > 0 {
                    normals[idx] = (normal / count as f32).normalize_or_zero();
                } else {
                    normals[idx] = Vec3::Y;
                }
            }
        }

        normals
    }

    /// Applies wind force to particles.
    pub fn apply_wind(&mut self, wind: Vec3, dt: f32) {
        let normals = self.normals();

        for (i, particle) in self.particles.iter_mut().enumerate() {
            if particle.is_pinned() {
                continue;
            }

            // Wind force depends on facing direction
            let normal = normals[i];
            let wind_dir = wind.normalize_or_zero();
            // Use abs so both sides of cloth catch wind
            let facing = normal.dot(wind_dir).abs().max(0.1); // Minimum facing factor
            let force = wind * facing * dt * dt;

            particle.position += force;
        }
    }

    /// Resets cloth to initial positions.
    pub fn reset(&mut self) {
        for particle in &mut self.particles {
            particle.position = particle.prev_position;
            particle.velocity = Vec3::ZERO;
        }
    }

    /// Moves pinned particles to new positions.
    pub fn set_pinned_positions(&mut self, positions: &[(usize, Vec3)]) {
        for &(idx, pos) in positions {
            if idx < self.particles.len() && self.particles[idx].is_pinned() {
                self.particles[idx].position = pos;
                self.particles[idx].prev_position = pos;
            }
        }
    }
}

/// Queries collision between a point and a collider.
pub fn query_collision(collider: &ClothCollider, point: Vec3, margin: f32) -> CollisionResult {
    match collider {
        ClothCollider::Sphere { center, radius } => {
            let delta = point - *center;
            let distance = delta.length();
            let effective_radius = radius + margin;

            if distance < effective_radius {
                let normal = if distance > 0.0001 {
                    delta / distance
                } else {
                    Vec3::Y
                };

                CollisionResult {
                    collided: true,
                    closest_point: *center + normal * *radius,
                    normal,
                    depth: effective_radius - distance,
                }
            } else {
                CollisionResult::none()
            }
        }

        ClothCollider::Plane {
            point: plane_pt,
            normal,
        } => {
            let dist = (point - *plane_pt).dot(*normal);

            if dist < margin {
                CollisionResult {
                    collided: true,
                    closest_point: point - *normal * dist,
                    normal: *normal,
                    depth: margin - dist,
                }
            } else {
                CollisionResult::none()
            }
        }

        ClothCollider::Capsule { start, end, radius } => {
            // Find closest point on capsule axis
            let axis = *end - *start;
            let len_sq = axis.length_squared();

            let t = if len_sq > 0.0001 {
                ((point - *start).dot(axis) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let closest_on_axis = *start + axis * t;
            let delta = point - closest_on_axis;
            let distance = delta.length();
            let effective_radius = radius + margin;

            if distance < effective_radius {
                let normal = if distance > 0.0001 {
                    delta / distance
                } else {
                    Vec3::Y
                };

                CollisionResult {
                    collided: true,
                    closest_point: closest_on_axis + normal * *radius,
                    normal,
                    depth: effective_radius - distance,
                }
            } else {
                CollisionResult::none()
            }
        }

        ClothCollider::Box { min, max } => {
            // Clamp point to box
            let clamped = Vec3::new(
                point.x.clamp(min.x, max.x),
                point.y.clamp(min.y, max.y),
                point.z.clamp(min.z, max.z),
            );

            let delta = point - clamped;
            let distance = delta.length();

            // Check if inside box
            let inside = point.x >= min.x
                && point.x <= max.x
                && point.y >= min.y
                && point.y <= max.y
                && point.z >= min.z
                && point.z <= max.z;

            if inside {
                // Find closest face
                let to_min = point - *min;
                let to_max = *max - point;
                let distances = [to_min.x, to_max.x, to_min.y, to_max.y, to_min.z, to_max.z];
                let normals = [-Vec3::X, Vec3::X, -Vec3::Y, Vec3::Y, -Vec3::Z, Vec3::Z];

                let (min_idx, &min_dist) = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                CollisionResult {
                    collided: true,
                    closest_point: point + normals[min_idx] * min_dist,
                    normal: normals[min_idx],
                    depth: min_dist + margin,
                }
            } else if distance < margin {
                let normal = if distance > 0.0001 {
                    delta / distance
                } else {
                    Vec3::Y
                };

                CollisionResult {
                    collided: true,
                    closest_point: clamped,
                    normal,
                    depth: margin - distance,
                }
            } else {
                CollisionResult::none()
            }
        }
    }
}

/// Self-collision detection grid.
pub struct SelfCollisionGrid {
    cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SelfCollisionGrid {
    /// Creates a new self-collision grid.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    /// Clears the grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Inserts a particle into the grid.
    pub fn insert(&mut self, index: usize, position: Vec3) {
        let cell = self.cell_coords(position);
        self.cells.entry(cell).or_default().push(index);
    }

    /// Gets cell coordinates for a position.
    fn cell_coords(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        )
    }

    /// Gets potential collision pairs for a particle.
    pub fn query_neighbors(&self, position: Vec3) -> Vec<usize> {
        let (cx, cy, cz) = self.cell_coords(position);
        let mut neighbors = Vec::new();

        // Check 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = (cx + dx, cy + dy, cz + dz);
                    if let Some(indices) = self.cells.get(&cell) {
                        neighbors.extend(indices);
                    }
                }
            }
        }

        neighbors
    }
}

/// Enables self-collision for a cloth.
pub fn solve_self_collision(cloth: &mut Cloth, min_distance: f32, grid: &mut SelfCollisionGrid) {
    grid.clear();

    // Build spatial hash
    for (i, particle) in cloth.particles.iter().enumerate() {
        grid.insert(i, particle.position);
    }

    // Check collisions
    for i in 0..cloth.particles.len() {
        if cloth.particles[i].is_pinned() {
            continue;
        }

        let pos_i = cloth.particles[i].position;
        let neighbors = grid.query_neighbors(pos_i);

        for &j in &neighbors {
            if j <= i || cloth.particles[j].is_pinned() {
                continue;
            }

            let pos_j = cloth.particles[j].position;
            let delta = pos_j - pos_i;
            let dist = delta.length();

            if dist > 0.0001 && dist < min_distance {
                let correction = delta.normalize() * (min_distance - dist) * 0.5;

                cloth.particles[i].position -= correction;
                cloth.particles[j].position += correction;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloth_config_default() {
        let config = ClothConfig::default();
        assert!(config.iterations > 0);
        assert!(config.stretch_stiffness > 0.0);
    }

    #[test]
    fn test_cloth_particle_new() {
        let particle = ClothParticle::new(Vec3::new(1.0, 2.0, 3.0), 1.0);
        assert_eq!(particle.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(!particle.is_pinned());
    }

    #[test]
    fn test_cloth_particle_pinned() {
        let particle = ClothParticle::pinned(Vec3::ZERO);
        assert!(particle.is_pinned());
    }

    #[test]
    fn test_cloth_particle_pin_unpin() {
        let mut particle = ClothParticle::new(Vec3::ZERO, 1.0);
        assert!(!particle.is_pinned());

        particle.pin();
        assert!(particle.is_pinned());

        particle.unpin(1.0);
        assert!(!particle.is_pinned());
    }

    #[test]
    fn test_cloth_grid_creation() {
        let cloth = Cloth::grid(Vec3::ZERO, 2.0, 2.0, 3, 3, 1.0);

        assert_eq!(cloth.particles.len(), 9);
        assert_eq!(cloth.width, 3);
        assert_eq!(cloth.height, 3);
        assert!(!cloth.stretch_constraints.is_empty());
    }

    #[test]
    fn test_cloth_pin_top_row() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 2.0, 2.0, 3, 3, 1.0);
        cloth.pin_top_row();

        // First row should be pinned
        assert!(cloth.particles[0].is_pinned());
        assert!(cloth.particles[1].is_pinned());
        assert!(cloth.particles[2].is_pinned());

        // Other rows should not be
        assert!(!cloth.particles[3].is_pinned());
    }

    #[test]
    fn test_cloth_pin_corners() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 2.0, 2.0, 3, 3, 1.0);
        cloth.pin_corners();

        assert!(cloth.particles[0].is_pinned());
        assert!(cloth.particles[2].is_pinned());
        assert!(!cloth.particles[1].is_pinned());
    }

    #[test]
    fn test_cloth_step() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 3, 3, 1.0);
        cloth.pin_top_row();

        let initial_y = cloth.particles[6].position.y; // Bottom center

        // Step simulation
        for _ in 0..60 {
            cloth.step(1.0 / 60.0);
        }

        // Bottom should have fallen due to gravity
        let final_y = cloth.particles[6].position.y;
        assert!(final_y < initial_y);
    }

    #[test]
    fn test_cloth_ground_collision() {
        let mut cloth = Cloth::grid(Vec3::Y * 0.5, 1.0, 1.0, 3, 3, 1.0);
        cloth.add_collider(ClothCollider::ground());
        cloth.pin_top_row();

        // Step simulation
        for _ in 0..120 {
            cloth.step(1.0 / 60.0);
        }

        // All particles should be above ground
        for particle in &cloth.particles {
            assert!(
                particle.position.y >= -0.1,
                "y = {} should be above ground",
                particle.position.y
            );
        }
    }

    #[test]
    fn test_cloth_sphere_collision() {
        let mut cloth = Cloth::grid(Vec3::new(-0.5, 2.0, -0.5), 1.0, 1.0, 5, 5, 1.0);
        cloth.add_collider(ClothCollider::sphere(Vec3::Y, 0.5));
        cloth.pin_top_row();

        // Step simulation
        for _ in 0..120 {
            cloth.step(1.0 / 60.0);
        }

        // Check that particles drape around sphere
        let center_idx = cloth.grid_index(2, 4);
        let center_pos = cloth.particles[center_idx].position;

        // Bottom center should be near sphere level or pushed aside
        let dist_from_sphere = (center_pos - Vec3::Y).length();
        assert!(dist_from_sphere >= 0.4, "Should be outside sphere");
    }

    #[test]
    fn test_query_collision_sphere() {
        let collider = ClothCollider::sphere(Vec3::ZERO, 1.0);

        // Point inside sphere
        let result = query_collision(&collider, Vec3::new(0.5, 0.0, 0.0), 0.0);
        assert!(result.collided);
        assert!(result.depth > 0.0);

        // Point outside sphere
        let result = query_collision(&collider, Vec3::new(2.0, 0.0, 0.0), 0.0);
        assert!(!result.collided);
    }

    #[test]
    fn test_query_collision_plane() {
        let collider = ClothCollider::ground();

        // Point below plane
        let result = query_collision(&collider, Vec3::new(0.0, -0.5, 0.0), 0.0);
        assert!(result.collided);

        // Point above plane
        let result = query_collision(&collider, Vec3::new(0.0, 0.5, 0.0), 0.0);
        assert!(!result.collided);
    }

    #[test]
    fn test_query_collision_capsule() {
        let collider = ClothCollider::capsule(Vec3::ZERO, Vec3::Y * 2.0, 0.5);

        // Point near capsule
        let result = query_collision(&collider, Vec3::new(0.3, 1.0, 0.0), 0.0);
        assert!(result.collided);

        // Point far from capsule
        let result = query_collision(&collider, Vec3::new(2.0, 1.0, 0.0), 0.0);
        assert!(!result.collided);
    }

    #[test]
    fn test_query_collision_box() {
        let collider = ClothCollider::aabb(Vec3::ZERO, Vec3::ONE);

        // Point inside box
        let result = query_collision(&collider, Vec3::new(0.5, 0.5, 0.5), 0.0);
        assert!(result.collided);

        // Point outside box
        let result = query_collision(&collider, Vec3::new(2.0, 0.5, 0.5), 0.0);
        assert!(!result.collided);
    }

    #[test]
    fn test_cloth_from_mesh() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 1.0),
        ];
        let triangles = vec![[0, 1, 2]];

        let cloth = Cloth::from_mesh(&positions, &triangles, 1.0);

        assert_eq!(cloth.particles.len(), 3);
        assert_eq!(cloth.stretch_constraints.len(), 3); // 3 edges
    }

    #[test]
    fn test_cloth_normals() {
        let cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 3, 3, 1.0);
        let normals = cloth.normals();

        assert_eq!(normals.len(), 9);
        // Flat cloth should have mostly upward normals
        for normal in &normals {
            assert!(normal.length() > 0.9);
        }
    }

    #[test]
    fn test_cloth_wind() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 3, 3, 1.0);
        cloth.config.gravity = Vec3::ZERO; // No gravity for this test
        cloth.pin_top_row();

        let initial_x = cloth.particles[6].position.x;

        // Apply wind
        for _ in 0..60 {
            cloth.apply_wind(Vec3::X * 10.0, 1.0 / 60.0);
            cloth.step(1.0 / 60.0);
        }

        // Should have moved in wind direction
        assert!(cloth.particles[6].position.x > initial_x);
    }

    #[test]
    fn test_self_collision_grid() {
        let mut grid = SelfCollisionGrid::new(0.5);

        grid.insert(0, Vec3::ZERO);
        grid.insert(1, Vec3::new(0.1, 0.0, 0.0));
        grid.insert(2, Vec3::new(2.0, 0.0, 0.0));

        let neighbors = grid.query_neighbors(Vec3::ZERO);

        // Should find nearby particles but not far ones
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
    }

    #[test]
    fn test_self_collision() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 5, 5, 1.0);
        let mut grid = SelfCollisionGrid::new(0.1);

        // Manually move particles close together
        cloth.particles[0].position = Vec3::ZERO;
        cloth.particles[0].unpin(1.0);
        cloth.particles[1].position = Vec3::new(0.01, 0.0, 0.0);
        cloth.particles[1].unpin(1.0);

        solve_self_collision(&mut cloth, 0.05, &mut grid);

        // Particles should be pushed apart
        let dist = (cloth.particles[0].position - cloth.particles[1].position).length();
        assert!(dist >= 0.04, "dist = {} should be >= 0.05", dist);
    }

    #[test]
    fn test_cloth_positions() {
        let cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 2, 2, 1.0);
        let positions = cloth.positions();

        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_distance_constraint() {
        let constraint = DistanceConstraint::new(0, 1, 1.0, 0.5);
        assert_eq!(constraint.p0, 0);
        assert_eq!(constraint.p1, 1);
        assert_eq!(constraint.rest_length, 1.0);
        assert_eq!(constraint.stiffness, 0.5);
    }

    #[test]
    fn test_cloth_set_pinned_positions() {
        let mut cloth = Cloth::grid(Vec3::ZERO, 1.0, 1.0, 3, 3, 1.0);
        cloth.pin(&[0, 2]);

        let new_pos = Vec3::new(5.0, 5.0, 5.0);
        cloth.set_pinned_positions(&[(0, new_pos)]);

        assert_eq!(cloth.particles[0].position, new_pos);
    }
}
