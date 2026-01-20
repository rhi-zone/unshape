//! Spring physics and Verlet integration for soft body simulation.
//!
//! Provides position-based dynamics for cloth, rope, chains, and soft bodies.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_spring::{SpringSystem, SpringConfig};
//! use glam::Vec3;
//!
//! // Create a simple rope
//! let mut system = SpringSystem::new();
//!
//! // Add particles in a line
//! for i in 0..10 {
//!     let id = system.add_particle(Vec3::new(i as f32, 0.0, 0.0), 1.0);
//!     if i == 0 {
//!         system.pin_particle(id);
//!     }
//! }
//!
//! // Connect with springs
//! for i in 0..9 {
//!     system.add_spring(i, i + 1, SpringConfig::default());
//! }
//!
//! // Simulate
//! system.set_gravity(Vec3::new(0.0, -9.8, 0.0));
//! for _ in 0..100 {
//!     system.step(0.016);
//! }
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use glam::Vec3;

/// Registers all spring operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of spring ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<SpringConfig>("resin::SpringConfig");
}

/// A particle in the spring system.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position.
    pub position: Vec3,
    /// Previous position (for Verlet integration).
    pub prev_position: Vec3,
    /// Acceleration accumulator.
    pub acceleration: Vec3,
    /// Mass (0 = infinite mass / pinned).
    pub mass: f32,
    /// Inverse mass (cached).
    inv_mass: f32,
    /// Whether this particle is pinned (cannot move).
    pub pinned: bool,
}

impl Particle {
    /// Creates a new particle.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            prev_position: position,
            acceleration: Vec3::ZERO,
            mass,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            pinned: false,
        }
    }

    /// Returns the velocity (estimated from position difference).
    pub fn velocity(&self) -> Vec3 {
        self.position - self.prev_position
    }
}

/// Configuration for a spring constraint.
///
/// Operations on spring systems use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = SpringConfig))]
pub struct SpringConfig {
    /// Rest length of the spring.
    pub rest_length: f32,
    /// Stiffness (0-1, higher = stiffer).
    pub stiffness: f32,
    /// Damping (0-1, reduces oscillation).
    pub damping: f32,
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self {
            rest_length: 1.0,
            stiffness: 0.8,
            damping: 0.02,
        }
    }
}

impl SpringConfig {
    /// Creates a spring config with automatic rest length (set when connected).
    pub fn with_stiffness(stiffness: f32) -> Self {
        Self {
            rest_length: 0.0, // Will be set automatically
            stiffness,
            damping: 0.02,
        }
    }

    /// Applies this configuration, returning a copy of self.
    ///
    /// This is the identity operation for config structs - the config
    /// is the value itself. Useful for serialization pipelines.
    pub fn apply(&self) -> SpringConfig {
        *self
    }
}

/// A spring constraint between two particles.
#[derive(Debug, Clone)]
pub struct Spring {
    /// First particle index.
    pub a: usize,
    /// Second particle index.
    pub b: usize,
    /// Spring configuration.
    pub config: SpringConfig,
}

/// Spring-mass system with Verlet integration.
#[derive(Debug, Clone)]
pub struct SpringSystem {
    /// Particles in the system.
    particles: Vec<Particle>,
    /// Spring constraints.
    springs: Vec<Spring>,
    /// Global gravity.
    gravity: Vec3,
    /// Constraint iteration count.
    constraint_iterations: usize,
    /// Global damping.
    damping: f32,
}

impl SpringSystem {
    /// Creates a new empty spring system.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            gravity: Vec3::ZERO,
            constraint_iterations: 3,
            damping: 0.99,
        }
    }

    /// Adds a particle and returns its index.
    pub fn add_particle(&mut self, position: Vec3, mass: f32) -> usize {
        let id = self.particles.len();
        self.particles.push(Particle::new(position, mass));
        id
    }

    /// Pins a particle (prevents movement).
    pub fn pin_particle(&mut self, id: usize) {
        if id < self.particles.len() {
            self.particles[id].pinned = true;
            self.particles[id].inv_mass = 0.0;
        }
    }

    /// Unpins a particle.
    pub fn unpin_particle(&mut self, id: usize) {
        if id < self.particles.len() {
            let p = &mut self.particles[id];
            p.pinned = false;
            p.inv_mass = if p.mass > 0.0 { 1.0 / p.mass } else { 0.0 };
        }
    }

    /// Adds a spring constraint between two particles.
    pub fn add_spring(&mut self, a: usize, b: usize, mut config: SpringConfig) {
        if a >= self.particles.len() || b >= self.particles.len() {
            return;
        }

        // Auto-calculate rest length if not set
        if config.rest_length <= 0.0 {
            config.rest_length = (self.particles[a].position - self.particles[b].position).length();
        }

        self.springs.push(Spring { a, b, config });
    }

    /// Sets the global gravity.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Sets the number of constraint iterations.
    pub fn set_constraint_iterations(&mut self, iterations: usize) {
        self.constraint_iterations = iterations;
    }

    /// Sets global damping (0-1).
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
    }

    /// Returns the number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Returns the number of springs.
    pub fn spring_count(&self) -> usize {
        self.springs.len()
    }

    /// Gets a particle by index.
    pub fn particle(&self, id: usize) -> Option<&Particle> {
        self.particles.get(id)
    }

    /// Gets a mutable particle by index.
    pub fn particle_mut(&mut self, id: usize) -> Option<&mut Particle> {
        self.particles.get_mut(id)
    }

    /// Returns all particles.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Returns all springs.
    pub fn springs(&self) -> &[Spring] {
        &self.springs
    }

    /// Gets all particle positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Applies a force to a particle.
    pub fn apply_force(&mut self, id: usize, force: Vec3) {
        if let Some(p) = self.particles.get_mut(id) {
            p.acceleration += force * p.inv_mass;
        }
    }

    /// Applies a force to all particles.
    pub fn apply_force_all(&mut self, force: Vec3) {
        for p in &mut self.particles {
            p.acceleration += force * p.inv_mass;
        }
    }

    /// Advances the simulation by one time step using Verlet integration.
    pub fn step(&mut self, dt: f32) {
        // Apply gravity
        for p in &mut self.particles {
            if !p.pinned {
                p.acceleration += self.gravity;
            }
        }

        // Verlet integration
        let dt2 = dt * dt;
        for p in &mut self.particles {
            if p.pinned {
                continue;
            }

            let velocity = (p.position - p.prev_position) * self.damping;
            let new_pos = p.position + velocity + p.acceleration * dt2;

            p.prev_position = p.position;
            p.position = new_pos;
            p.acceleration = Vec3::ZERO;
        }

        // Solve constraints
        for _ in 0..self.constraint_iterations {
            self.solve_constraints();
        }
    }

    /// Solves spring constraints.
    fn solve_constraints(&mut self) {
        for spring in &self.springs.clone() {
            let pa = &self.particles[spring.a];
            let pb = &self.particles[spring.b];

            let delta = pb.position - pa.position;
            let dist = delta.length();

            if dist < 1e-10 {
                continue;
            }

            let diff = (dist - spring.config.rest_length) / dist;
            let correction = delta * diff * spring.config.stiffness;

            let inv_mass_sum = pa.inv_mass + pb.inv_mass;
            if inv_mass_sum <= 0.0 {
                continue;
            }

            let wa = pa.inv_mass / inv_mass_sum;
            let wb = pb.inv_mass / inv_mass_sum;

            // Apply damping to velocity
            let va = pa.velocity();
            let vb = pb.velocity();
            let relative_velocity = vb - va;
            let damping_force = relative_velocity * spring.config.damping;

            self.particles[spring.a].position += correction * wa + damping_force * wa;
            self.particles[spring.b].position -= correction * wb - damping_force * wb;
        }
    }

    /// Advances multiple time steps.
    pub fn steps(&mut self, dt: f32, n: usize) {
        for _ in 0..n {
            self.step(dt);
        }
    }
}

impl Default for SpringSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a rope (1D chain of particles).
pub fn create_rope(start: Vec3, end: Vec3, segments: usize, config: SpringConfig) -> SpringSystem {
    let mut system = SpringSystem::new();

    let delta = (end - start) / segments as f32;

    for i in 0..=segments {
        let pos = start + delta * i as f32;
        system.add_particle(pos, 1.0);
    }

    for i in 0..segments {
        system.add_spring(i, i + 1, config);
    }

    system
}

/// Creates a cloth (2D grid of particles).
pub fn create_cloth(
    origin: Vec3,
    width: f32,
    height: f32,
    cols: usize,
    rows: usize,
    config: SpringConfig,
) -> SpringSystem {
    let mut system = SpringSystem::new();

    let dx = width / (cols - 1).max(1) as f32;
    let dy = height / (rows - 1).max(1) as f32;

    // Create particles in a grid
    for row in 0..rows {
        for col in 0..cols {
            let pos = origin + Vec3::new(col as f32 * dx, 0.0, row as f32 * dy);
            system.add_particle(pos, 1.0);
        }
    }

    // Connect horizontally
    for row in 0..rows {
        for col in 0..cols - 1 {
            let a = row * cols + col;
            let b = a + 1;
            system.add_spring(a, b, config);
        }
    }

    // Connect vertically
    for row in 0..rows - 1 {
        for col in 0..cols {
            let a = row * cols + col;
            let b = a + cols;
            system.add_spring(a, b, config);
        }
    }

    // Diagonal springs for shear stability
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let a = row * cols + col;
            // Diagonal \
            system.add_spring(a, a + cols + 1, config);
            // Diagonal /
            system.add_spring(a + 1, a + cols, config);
        }
    }

    system
}

/// Creates a soft body sphere.
pub fn create_soft_sphere(center: Vec3, radius: f32, subdivisions: usize) -> SpringSystem {
    let mut system = SpringSystem::new();

    // Generate vertices on a sphere using icosphere-like distribution
    let mut points = Vec::new();

    // Add poles
    points.push(center + Vec3::Y * radius);
    points.push(center - Vec3::Y * radius);

    // Add rings
    let rings = subdivisions.max(2);
    let segments = (subdivisions * 2).max(4);

    for ring in 1..rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        let y = phi.cos() * radius;
        let ring_radius = phi.sin() * radius;

        for seg in 0..segments {
            let theta = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
            let x = theta.cos() * ring_radius;
            let z = theta.sin() * ring_radius;
            points.push(center + Vec3::new(x, y, z));
        }
    }

    // Add all particles
    for point in &points {
        system.add_particle(*point, 1.0);
    }

    // Connect nearby particles
    // Use a threshold based on expected neighbor distance
    let expected_arc = 2.0 * std::f32::consts::PI * radius / segments as f32;
    let threshold = expected_arc * 2.5; // Allow connections to nearby neighbors
    let config = SpringConfig {
        stiffness: 0.5,
        damping: 0.05,
        ..SpringConfig::default()
    };

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist = (points[i] - points[j]).length();
            if dist < threshold {
                system.add_spring(i, j, config);
            }
        }
    }

    system
}

/// Distance constraint (keeps two particles at a fixed distance).
#[derive(Debug, Clone)]
pub struct DistanceConstraint {
    /// First particle index.
    pub a: usize,
    /// Second particle index.
    pub b: usize,
    /// Target distance.
    pub distance: f32,
}

/// Verlet particle (simpler, standalone particle for basic simulations).
#[derive(Debug, Clone, Copy)]
pub struct VerletParticle {
    /// Current position.
    pub position: Vec3,
    /// Previous position.
    pub prev_position: Vec3,
    /// Is pinned.
    pub pinned: bool,
}

impl VerletParticle {
    /// Creates a new Verlet particle.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            pinned: false,
        }
    }

    /// Creates a pinned particle.
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            pinned: true,
        }
    }

    /// Returns the velocity.
    pub fn velocity(&self) -> Vec3 {
        self.position - self.prev_position
    }

    /// Integrates the particle with gravity.
    pub fn integrate(&mut self, gravity: Vec3, dt: f32, damping: f32) {
        if self.pinned {
            return;
        }

        let velocity = self.velocity() * damping;
        let new_pos = self.position + velocity + gravity * dt * dt;
        self.prev_position = self.position;
        self.position = new_pos;
    }
}

/// Solves a distance constraint between two Verlet particles.
pub fn solve_distance_constraint(
    a: &mut VerletParticle,
    b: &mut VerletParticle,
    target_distance: f32,
    stiffness: f32,
) {
    let delta = b.position - a.position;
    let dist = delta.length();

    if dist < 1e-10 {
        return;
    }

    let diff = (dist - target_distance) / dist;
    let correction = delta * diff * stiffness * 0.5;

    if !a.pinned {
        a.position += correction;
    }
    if !b.pinned {
        b.position -= correction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(Vec3::new(1.0, 2.0, 3.0), 1.0);
        assert_eq!(p.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.mass, 1.0);
        assert!(!p.pinned);
    }

    #[test]
    fn test_spring_system_creation() {
        let system = SpringSystem::new();
        assert_eq!(system.particle_count(), 0);
        assert_eq!(system.spring_count(), 0);
    }

    #[test]
    fn test_add_particle() {
        let mut system = SpringSystem::new();
        let id = system.add_particle(Vec3::ZERO, 1.0);
        assert_eq!(id, 0);
        assert_eq!(system.particle_count(), 1);
    }

    #[test]
    fn test_pin_particle() {
        let mut system = SpringSystem::new();
        let id = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(id);

        assert!(system.particle(id).unwrap().pinned);
    }

    #[test]
    fn test_add_spring() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::ZERO, 1.0);
        system.add_particle(Vec3::X, 1.0);
        system.add_spring(0, 1, SpringConfig::default());

        assert_eq!(system.spring_count(), 1);
    }

    #[test]
    fn test_auto_rest_length() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::ZERO, 1.0);
        system.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);
        system.add_spring(0, 1, SpringConfig::with_stiffness(0.5));

        // Rest length should be automatically set to 2.0
        assert!((system.springs()[0].config.rest_length - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_step_with_gravity() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::ZERO, 1.0);
        system.set_gravity(Vec3::new(0.0, -9.8, 0.0));

        let initial_y = system.particle(0).unwrap().position.y;
        system.step(0.016);
        let after_y = system.particle(0).unwrap().position.y;

        // Particle should fall
        assert!(after_y < initial_y);
    }

    #[test]
    fn test_pinned_particle_doesnt_move() {
        let mut system = SpringSystem::new();
        let id = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(id);
        system.set_gravity(Vec3::new(0.0, -9.8, 0.0));

        system.step(0.016);

        assert_eq!(system.particle(id).unwrap().position, Vec3::ZERO);
    }

    #[test]
    fn test_spring_constraint() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::ZERO, 1.0);
        system.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);
        system.add_spring(
            0,
            1,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 1.0,
                damping: 0.0,
            },
        );

        // Initial distance is 2.0, rest length is 1.0
        // After solving constraints, particles should move closer
        system.step(0.016);

        let dist =
            (system.particle(0).unwrap().position - system.particle(1).unwrap().position).length();
        assert!(dist < 2.0);
    }

    #[test]
    fn test_create_rope() {
        let rope = create_rope(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            10,
            SpringConfig::default(),
        );

        assert_eq!(rope.particle_count(), 11);
        assert_eq!(rope.spring_count(), 10);
    }

    #[test]
    fn test_create_cloth() {
        let cloth = create_cloth(Vec3::ZERO, 5.0, 5.0, 4, 4, SpringConfig::default());

        assert_eq!(cloth.particle_count(), 16); // 4x4 grid

        // Should have structural + shear springs
        // Horizontal: 3*4 = 12
        // Vertical: 4*3 = 12
        // Diagonal: 3*3*2 = 18
        // Total: 42
        assert_eq!(cloth.spring_count(), 42);
    }

    #[test]
    fn test_create_soft_sphere() {
        let sphere = create_soft_sphere(Vec3::ZERO, 1.0, 3);
        assert!(sphere.particle_count() > 2);
        assert!(sphere.spring_count() > 0);
    }

    #[test]
    fn test_verlet_particle() {
        let mut p = VerletParticle::new(Vec3::new(0.0, 10.0, 0.0));
        p.integrate(Vec3::new(0.0, -9.8, 0.0), 0.016, 0.99);

        // Particle should fall
        assert!(p.position.y < 10.0);
    }

    #[test]
    fn test_distance_constraint() {
        let mut a = VerletParticle::new(Vec3::ZERO);
        let mut b = VerletParticle::new(Vec3::new(2.0, 0.0, 0.0));

        // Target distance 1.0, current 2.0
        solve_distance_constraint(&mut a, &mut b, 1.0, 1.0);

        let dist = (a.position - b.position).length();
        assert!(dist < 2.0);
    }

    #[test]
    fn test_apply_force() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::ZERO, 1.0);
        system.apply_force(0, Vec3::new(10.0, 0.0, 0.0));

        assert_eq!(
            system.particle(0).unwrap().acceleration,
            Vec3::new(10.0, 0.0, 0.0)
        );
    }

    #[test]
    fn test_positions() {
        let mut system = SpringSystem::new();
        system.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
        system.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);

        let positions = system.positions();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(positions[1], Vec3::new(2.0, 0.0, 0.0));
    }
}

/// Invariant tests for spring physics.
///
/// Run with: cargo test -p rhizome-resin-spring --features invariant-tests
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    /// Helper to compute total kinetic energy of a spring system.
    fn kinetic_energy(system: &SpringSystem) -> f32 {
        system
            .particles()
            .iter()
            .map(|p| {
                let v = p.velocity();
                0.5 * p.mass * v.length_squared()
            })
            .sum()
    }

    /// Helper to compute total potential energy stored in springs.
    fn spring_potential_energy(system: &SpringSystem) -> f32 {
        system
            .springs()
            .iter()
            .map(|s| {
                let pa = &system.particles()[s.a];
                let pb = &system.particles()[s.b];
                let dist = (pb.position - pa.position).length();
                let stretch = dist - s.config.rest_length;
                // Simple harmonic potential: 0.5 * k * x^2
                // Using stiffness as a proxy for spring constant
                0.5 * s.config.stiffness * stretch * stretch
            })
            .sum()
    }

    /// Critically damped springs should converge to target position.
    ///
    /// When a spring connects a pinned particle to a free particle,
    /// the free particle should eventually settle at the rest length distance.
    #[test]
    fn invariant_spring_convergence() {
        let mut system = SpringSystem::new();

        // Pinned anchor at origin
        let anchor = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(anchor);

        // Free particle displaced from rest position
        let free = system.add_particle(Vec3::new(3.0, 0.0, 0.0), 1.0);

        // Connect with high stiffness, high damping spring (rest length 1.0)
        system.add_spring(
            anchor,
            free,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 0.9,
                damping: 0.3, // High damping for faster settling
            },
        );

        system.set_damping(0.95); // Global damping

        // Simulate for many steps
        for _ in 0..500 {
            system.step(0.016);
        }

        // Check that free particle has converged near rest length from anchor
        let final_dist = system.particle(free).unwrap().position.length();
        let error = (final_dist - 1.0).abs();

        assert!(
            error < 0.1,
            "Spring should converge to rest length. Expected ~1.0, got {}, error {}",
            final_dist,
            error
        );
    }

    /// Underdamped springs should oscillate with decreasing amplitude.
    ///
    /// With low damping, a spring system should show oscillation where
    /// each successive peak has lower amplitude than the previous.
    #[test]
    fn invariant_underdamped_decreasing_amplitude() {
        let mut system = SpringSystem::new();

        // Pinned anchor at origin
        let anchor = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(anchor);

        // Free particle displaced from rest position
        let free = system.add_particle(Vec3::new(3.0, 0.0, 0.0), 1.0);

        // Low damping spring - should oscillate
        system.add_spring(
            anchor,
            free,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 0.8,
                damping: 0.01, // Low damping
            },
        );

        system.set_damping(0.999); // Very little global damping

        // Track maximum displacements (peaks)
        let mut peaks: Vec<f32> = Vec::new();
        let mut prev_dist = 3.0_f32;
        let mut increasing = false;

        for _ in 0..300 {
            system.step(0.016);
            let dist = system.particle(free).unwrap().position.length();

            // Detect peaks (local maxima of distance from rest)
            let displacement = (dist - 1.0).abs();
            let prev_displacement = (prev_dist - 1.0).abs();

            if increasing && displacement < prev_displacement {
                // We just passed a peak
                peaks.push(prev_displacement);
            }

            increasing = displacement > prev_displacement;
            prev_dist = dist;
        }

        // We should have at least 2 peaks to compare
        assert!(
            peaks.len() >= 2,
            "Expected at least 2 oscillation peaks, got {}",
            peaks.len()
        );

        // Each peak should be smaller than or equal to the previous
        for i in 1..peaks.len() {
            assert!(
                peaks[i] <= peaks[i - 1] + 0.01, // Small tolerance for numerical error
                "Amplitude should decrease: peak {} ({}) should be <= peak {} ({})",
                i,
                peaks[i],
                i - 1,
                peaks[i - 1]
            );
        }
    }

    /// Spring positions should remain bounded (no explosion).
    ///
    /// A properly damped spring system should never have particles
    /// fly off to infinity, even after many simulation steps.
    #[test]
    fn invariant_position_bounds() {
        let mut system = SpringSystem::new();

        // Create a chain of springs
        for i in 0..10 {
            system.add_particle(Vec3::new(i as f32, 0.0, 0.0), 1.0);
        }

        // Pin the first particle
        system.pin_particle(0);

        // Connect with springs
        for i in 0..9 {
            system.add_spring(i, i + 1, SpringConfig::default());
        }

        // Apply gravity and simulate
        system.set_gravity(Vec3::new(0.0, -9.8, 0.0));

        let initial_bound = 100.0; // Reasonable bound for this setup

        for step in 0..1000 {
            system.step(0.016);

            // Check all particles are within bounds
            for (i, p) in system.particles().iter().enumerate() {
                let dist = p.position.length();
                assert!(
                    dist < initial_bound,
                    "Particle {} position exploded at step {}: distance {} exceeds bound {}",
                    i,
                    step,
                    dist,
                    initial_bound
                );
            }
        }
    }

    /// Higher global damping should result in faster settling.
    ///
    /// A system with higher global damping should have lower total
    /// displacement over time compared to one with lower damping.
    #[test]
    fn invariant_damping_reduces_motion() {
        // Test with low global damping
        let total_displacement_low = measure_total_displacement(0.99);

        // Test with high global damping
        let total_displacement_high = measure_total_displacement(0.90);

        assert!(
            total_displacement_high < total_displacement_low,
            "Higher damping should reduce total motion. Low damping: {}, High damping: {}",
            total_displacement_low,
            total_displacement_high
        );
    }

    /// Helper to measure total displacement during spring simulation.
    fn measure_total_displacement(global_damping: f32) -> f32 {
        let mut system = SpringSystem::new();

        let anchor = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(anchor);

        let free = system.add_particle(Vec3::new(3.0, 0.0, 0.0), 1.0);

        system.add_spring(
            anchor,
            free,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 0.8,
                damping: 0.0, // No spring damping to isolate global damping effect
            },
        );

        system.set_damping(global_damping);

        let mut total_displacement = 0.0_f32;
        let mut prev_pos = system.particle(free).unwrap().position;

        for _ in 0..200 {
            system.step(0.016);
            let pos = system.particle(free).unwrap().position;
            total_displacement += (pos - prev_pos).length();
            prev_pos = pos;
        }

        total_displacement
    }

    /// Damped spring system should lose energy over time.
    ///
    /// With damping, total mechanical energy should decrease over time
    /// as energy is dissipated.
    #[test]
    fn invariant_energy_dissipation_damped() {
        let mut system = SpringSystem::new();

        // Create a system with initial potential energy (stretched spring)
        let anchor = system.add_particle(Vec3::ZERO, 1.0);
        system.pin_particle(anchor);

        // Start particle far from rest position to give it potential energy
        system.add_particle(Vec3::new(3.0, 0.0, 0.0), 1.0);

        system.add_spring(
            0,
            1,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 0.8,
                damping: 0.1, // Moderate damping
            },
        );

        system.set_damping(0.98); // Global damping

        // Measure energy at different time points
        let mut energies: Vec<f32> = Vec::new();

        for step in 0..200 {
            system.step(0.016);

            // Sample energy every 20 steps
            if step % 20 == 0 {
                let ke = kinetic_energy(&system);
                let pe = spring_potential_energy(&system);
                energies.push(ke + pe);
            }
        }

        // Energy should generally decrease over time
        // Check that final energy is less than initial
        let initial_energy = energies.first().copied().unwrap_or(0.0);
        let final_energy = energies.last().copied().unwrap_or(0.0);

        assert!(
            final_energy < initial_energy,
            "Damped system should dissipate energy. Initial: {}, Final: {}",
            initial_energy,
            final_energy
        );
    }

    /// Verlet particle should maintain velocity direction under constant force.
    ///
    /// A particle under constant acceleration should move in the direction
    /// of that acceleration.
    #[test]
    fn invariant_verlet_velocity_direction() {
        let mut p = VerletParticle::new(Vec3::ZERO);
        let gravity = Vec3::new(0.0, -9.8, 0.0);

        for _ in 0..10 {
            p.integrate(gravity, 0.016, 1.0);
        }

        // Velocity should be pointing downward (same direction as gravity)
        let velocity = p.velocity();
        assert!(
            velocity.y < 0.0,
            "Velocity should be in gravity direction, got {:?}",
            velocity
        );

        // Velocity should be mostly in Y direction
        let normalized = velocity.normalize();
        assert!(
            normalized.y.abs() > 0.99,
            "Velocity should be aligned with gravity, got {:?}",
            normalized
        );
    }

    /// Distance constraint should preserve distance over time.
    ///
    /// After many applications of a distance constraint, the two
    /// particles should be at approximately the target distance.
    #[test]
    fn invariant_distance_constraint_convergence() {
        let mut a = VerletParticle::new(Vec3::ZERO);
        let mut b = VerletParticle::new(Vec3::new(5.0, 0.0, 0.0));

        let target = 2.0;

        // Apply constraint many times
        for _ in 0..50 {
            solve_distance_constraint(&mut a, &mut b, target, 1.0);
        }

        let final_dist = (a.position - b.position).length();
        let error = (final_dist - target).abs();

        assert!(
            error < 0.01,
            "Distance constraint should converge to target {}. Got {}, error {}",
            target,
            final_dist,
            error
        );
    }

    /// Pinned particles should not move under any circumstance.
    #[test]
    fn invariant_pinned_immobility() {
        let mut system = SpringSystem::new();

        // Create pinned particle
        let pinned = system.add_particle(Vec3::new(1.0, 2.0, 3.0), 1.0);
        system.pin_particle(pinned);

        // Create free particle connected to it
        let free = system.add_particle(Vec3::new(10.0, 0.0, 0.0), 1.0);
        system.add_spring(
            pinned,
            free,
            SpringConfig {
                rest_length: 1.0,
                stiffness: 1.0,
                damping: 0.0,
            },
        );

        // Apply forces and gravity
        system.set_gravity(Vec3::new(0.0, -100.0, 0.0));
        system.apply_force(pinned, Vec3::new(1000.0, 1000.0, 1000.0));

        let initial_pos = system.particle(pinned).unwrap().position;

        // Simulate many steps
        for _ in 0..100 {
            system.step(0.016);
        }

        let final_pos = system.particle(pinned).unwrap().position;

        assert_eq!(
            initial_pos, final_pos,
            "Pinned particle should not move. Initial: {:?}, Final: {:?}",
            initial_pos, final_pos
        );
    }

    /// Cloth simulation should maintain structural integrity.
    ///
    /// A cloth under gravity should not have any springs stretched
    /// beyond a reasonable multiple of their rest length.
    #[test]
    fn invariant_cloth_structural_integrity() {
        let mut cloth = create_cloth(Vec3::ZERO, 2.0, 2.0, 4, 4, SpringConfig::default());

        // Pin top row
        for i in 0..4 {
            cloth.pin_particle(i);
        }

        cloth.set_gravity(Vec3::new(0.0, -9.8, 0.0));

        // Simulate
        for _ in 0..200 {
            cloth.step(0.016);
        }

        // Check that no spring is stretched more than 3x its rest length
        let max_stretch_ratio = 3.0;

        for spring in cloth.springs() {
            let pa = &cloth.particles()[spring.a];
            let pb = &cloth.particles()[spring.b];
            let actual_length = (pb.position - pa.position).length();
            let ratio = actual_length / spring.config.rest_length;

            assert!(
                ratio < max_stretch_ratio,
                "Spring stretched too much: {:.2}x rest length (max allowed: {}x)",
                ratio,
                max_stretch_ratio
            );
        }
    }
}
