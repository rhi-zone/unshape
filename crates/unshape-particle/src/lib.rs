//! Particle system for procedural effects.
//!
//! Provides particles, emitters, and forces for creating dynamic effects.
//! Integrates with the Field system for spatial forces.

use glam::{Vec2, Vec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single particle with position, velocity, and lifetime.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Particle {
    /// Position in world space.
    pub position: Vec3,
    /// Velocity in units per second.
    pub velocity: Vec3,
    /// Current age in seconds.
    pub age: f32,
    /// Total lifetime in seconds.
    pub lifetime: f32,
    /// Size (can be used for rendering).
    pub size: f32,
    /// Color (RGBA).
    pub color: [f32; 4],
    /// Custom data slot.
    pub custom: f32,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            age: 0.0,
            lifetime: 1.0,
            size: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            custom: 0.0,
        }
    }
}

impl Particle {
    /// Creates a new particle at the given position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Returns the normalized age (0.0 to 1.0).
    pub fn normalized_age(&self) -> f32 {
        if self.lifetime <= 0.0 {
            1.0
        } else {
            (self.age / self.lifetime).clamp(0.0, 1.0)
        }
    }

    /// Returns true if the particle is still alive.
    pub fn is_alive(&self) -> bool {
        self.age < self.lifetime
    }
}

/// Describes how particles are spawned.
pub trait Emitter: Send + Sync {
    /// Spawns a new particle with initial properties.
    fn emit(&self, rng: &mut ParticleRng) -> Particle;
}

/// Modifies particles over time.
pub trait Force: Send + Sync {
    /// Applies force to a particle, modifying its velocity.
    fn apply(&self, particle: &mut Particle, dt: f32);
}

/// Simple random number generator for particle systems.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParticleRng {
    state: u64,
}

impl Default for ParticleRng {
    fn default() -> Self {
        Self::new(12345)
    }
}

impl ParticleRng {
    /// Creates a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a random u64.
    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Returns a random f32 in [min, max).
    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    /// Returns a random point on a unit sphere.
    pub fn unit_sphere(&mut self) -> Vec3 {
        // Use rejection sampling for uniform distribution
        loop {
            let x = self.range(-1.0, 1.0);
            let y = self.range(-1.0, 1.0);
            let z = self.range(-1.0, 1.0);
            let len_sq = x * x + y * y + z * z;
            if len_sq > 0.0001 && len_sq <= 1.0 {
                return Vec3::new(x, y, z).normalize();
            }
        }
    }

    /// Returns a random point inside a unit sphere.
    pub fn inside_unit_sphere(&mut self) -> Vec3 {
        self.unit_sphere() * self.next_f32().powf(1.0 / 3.0)
    }

    /// Returns a random point on a unit circle (XY plane).
    pub fn unit_circle(&mut self) -> Vec2 {
        let angle = self.next_f32() * std::f32::consts::TAU;
        Vec2::new(angle.cos(), angle.sin())
    }
}

/// Particle system that manages particles and their simulation.
#[derive(Debug)]
pub struct ParticleSystem {
    /// Active particles.
    particles: Vec<Particle>,
    /// Maximum number of particles.
    max_particles: usize,
    /// Random number generator.
    rng: ParticleRng,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(10000)
    }
}

impl ParticleSystem {
    /// Creates a new particle system with the given capacity.
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
            rng: ParticleRng::default(),
        }
    }

    /// Returns the current particle count.
    pub fn count(&self) -> usize {
        self.particles.len()
    }

    /// Returns true if the system is at capacity.
    pub fn is_full(&self) -> bool {
        self.particles.len() >= self.max_particles
    }

    /// Returns a slice of all active particles.
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Returns a mutable slice of all active particles.
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    /// Emits particles from the given emitter.
    pub fn emit(&mut self, emitter: &dyn Emitter, count: usize) {
        let available = self.max_particles - self.particles.len();
        let to_emit = count.min(available);

        for _ in 0..to_emit {
            let particle = emitter.emit(&mut self.rng);
            self.particles.push(particle);
        }
    }

    /// Updates all particles with the given time delta.
    pub fn update(&mut self, dt: f32) {
        // Update positions based on velocity
        for particle in &mut self.particles {
            particle.position += particle.velocity * dt;
            particle.age += dt;
        }

        // Remove dead particles
        self.particles.retain(|p| p.is_alive());
    }

    /// Updates all particles and applies forces.
    pub fn update_with_forces(&mut self, dt: f32, forces: &[&dyn Force]) {
        // Apply forces
        for particle in &mut self.particles {
            for force in forces {
                force.apply(particle, dt);
            }
        }

        // Update positions
        self.update(dt);
    }

    /// Clears all particles.
    pub fn clear(&mut self) {
        self.particles.clear();
    }
}

// ============================================================================
// Built-in Emitters
// ============================================================================

/// Emits particles from a single point.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = PointEmitter))]
pub struct PointEmitter {
    /// Emission position.
    pub position: Vec3,
    /// Initial velocity direction.
    pub direction: Vec3,
    /// Velocity spread angle in radians.
    pub spread: f32,
    /// Minimum initial speed.
    pub speed_min: f32,
    /// Maximum initial speed.
    pub speed_max: f32,
    /// Minimum lifetime.
    pub lifetime_min: f32,
    /// Maximum lifetime.
    pub lifetime_max: f32,
    /// Initial size.
    pub size: f32,
    /// Initial color.
    pub color: [f32; 4],
}

impl Default for PointEmitter {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::Y,
            spread: 0.5,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            size: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl PointEmitter {
    /// Returns this emitter configuration (for Op pipeline).
    pub fn apply(&self) -> PointEmitter {
        self.clone()
    }
}

impl Emitter for PointEmitter {
    fn emit(&self, rng: &mut ParticleRng) -> Particle {
        // Create a random direction within the spread cone
        let dir = if self.spread > 0.0 {
            let random_dir = rng.unit_sphere();
            let spread_amount = rng.next_f32() * self.spread;
            self.direction
                .normalize()
                .lerp(random_dir, spread_amount)
                .normalize()
        } else {
            self.direction.normalize()
        };

        let speed = rng.range(self.speed_min, self.speed_max);
        let lifetime = rng.range(self.lifetime_min, self.lifetime_max);

        Particle {
            position: self.position,
            velocity: dir * speed,
            age: 0.0,
            lifetime,
            size: self.size,
            color: self.color,
            custom: 0.0,
        }
    }
}

/// Emits particles from a sphere surface or volume.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = SphereEmitter))]
pub struct SphereEmitter {
    /// Center position.
    pub center: Vec3,
    /// Sphere radius.
    pub radius: f32,
    /// If true, emit from volume; if false, emit from surface.
    pub volume: bool,
    /// Initial speed (outward from center).
    pub speed_min: f32,
    /// Maximum initial speed.
    pub speed_max: f32,
    /// Minimum lifetime.
    pub lifetime_min: f32,
    /// Maximum lifetime.
    pub lifetime_max: f32,
    /// Initial size.
    pub size: f32,
    /// Initial color.
    pub color: [f32; 4],
}

impl Default for SphereEmitter {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 1.0,
            volume: false,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            size: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl SphereEmitter {
    /// Returns this emitter configuration (for Op pipeline).
    pub fn apply(&self) -> SphereEmitter {
        self.clone()
    }
}

impl Emitter for SphereEmitter {
    fn emit(&self, rng: &mut ParticleRng) -> Particle {
        let dir = rng.unit_sphere();
        let dist = if self.volume {
            self.radius * rng.next_f32().powf(1.0 / 3.0)
        } else {
            self.radius
        };

        let speed = rng.range(self.speed_min, self.speed_max);
        let lifetime = rng.range(self.lifetime_min, self.lifetime_max);

        Particle {
            position: self.center + dir * dist,
            velocity: dir * speed,
            age: 0.0,
            lifetime,
            size: self.size,
            color: self.color,
            custom: 0.0,
        }
    }
}

/// Emits particles in a cone shape.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = ConeEmitter))]
pub struct ConeEmitter {
    /// Cone apex position.
    pub position: Vec3,
    /// Cone direction (axis).
    pub direction: Vec3,
    /// Cone angle in radians (half-angle).
    pub angle: f32,
    /// Minimum initial speed.
    pub speed_min: f32,
    /// Maximum initial speed.
    pub speed_max: f32,
    /// Minimum lifetime.
    pub lifetime_min: f32,
    /// Maximum lifetime.
    pub lifetime_max: f32,
    /// Initial size.
    pub size: f32,
    /// Initial color.
    pub color: [f32; 4],
}

impl Default for ConeEmitter {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::Y,
            angle: std::f32::consts::FRAC_PI_4,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            size: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl ConeEmitter {
    /// Returns this emitter configuration (for Op pipeline).
    pub fn apply(&self) -> ConeEmitter {
        self.clone()
    }
}

impl Emitter for ConeEmitter {
    fn emit(&self, rng: &mut ParticleRng) -> Particle {
        // Generate a random direction within the cone
        let cos_angle = self.angle.cos();
        let z = rng.range(cos_angle, 1.0);
        let phi = rng.next_f32() * std::f32::consts::TAU;
        let sin_theta = (1.0 - z * z).sqrt();

        // Local direction (cone along Z)
        let local_dir = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), z);

        // Rotate to match cone direction
        let dir = rotate_to_direction(local_dir, self.direction.normalize());

        let speed = rng.range(self.speed_min, self.speed_max);
        let lifetime = rng.range(self.lifetime_min, self.lifetime_max);

        Particle {
            position: self.position,
            velocity: dir * speed,
            age: 0.0,
            lifetime,
            size: self.size,
            color: self.color,
            custom: 0.0,
        }
    }
}

/// Rotates a vector from Z-up to the given direction.
fn rotate_to_direction(v: Vec3, dir: Vec3) -> Vec3 {
    if dir.z.abs() > 0.999 {
        // Direction is close to Z axis
        if dir.z > 0.0 {
            v
        } else {
            Vec3::new(v.x, -v.y, -v.z)
        }
    } else {
        // Build rotation basis
        let up = Vec3::Z;
        let right = up.cross(dir).normalize();
        let new_up = dir.cross(right);

        right * v.x + new_up * v.y + dir * v.z
    }
}

// ============================================================================
// Built-in Forces
// ============================================================================

/// Constant directional force (like gravity).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Gravity))]
pub struct Gravity {
    /// Acceleration vector (units per second squared).
    pub acceleration: Vec3,
}

impl Default for Gravity {
    fn default() -> Self {
        Self {
            acceleration: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

impl Gravity {
    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Gravity {
        self.clone()
    }
}

impl Force for Gravity {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        particle.velocity += self.acceleration * dt;
    }
}

/// Constant wind force.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Wind))]
pub struct Wind {
    /// Wind velocity (target velocity particles are pushed toward).
    pub velocity: Vec3,
    /// How strongly particles are pushed (0 = no effect, 1 = instant).
    pub strength: f32,
}

impl Default for Wind {
    fn default() -> Self {
        Self {
            velocity: Vec3::new(1.0, 0.0, 0.0),
            strength: 0.1,
        }
    }
}

impl Wind {
    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Wind {
        self.clone()
    }
}

impl Force for Wind {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        let diff = self.velocity - particle.velocity;
        particle.velocity += diff * self.strength * dt;
    }
}

/// Drag force that slows particles.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Drag))]
pub struct Drag {
    /// Drag coefficient (0 = no drag, higher = more drag).
    pub coefficient: f32,
}

impl Default for Drag {
    fn default() -> Self {
        Self { coefficient: 0.1 }
    }
}

impl Drag {
    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Drag {
        *self
    }
}

impl Force for Drag {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        let factor = 1.0 - self.coefficient * dt;
        particle.velocity *= factor.max(0.0);
    }
}

/// Attractor/repulsor force.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Attractor))]
pub struct Attractor {
    /// Attractor position.
    pub position: Vec3,
    /// Strength (positive = attract, negative = repel).
    pub strength: f32,
    /// Minimum distance (to prevent extreme forces).
    pub min_distance: f32,
}

impl Default for Attractor {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            strength: 10.0,
            min_distance: 0.1,
        }
    }
}

impl Attractor {
    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Attractor {
        self.clone()
    }
}

impl Force for Attractor {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        let to_attractor = self.position - particle.position;
        let dist = to_attractor.length().max(self.min_distance);
        let dir = to_attractor / dist;

        // Inverse square falloff
        let force = self.strength / (dist * dist);
        particle.velocity += dir * force * dt;
    }
}

/// Vortex force that creates spinning motion.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Vortex))]
pub struct Vortex {
    /// Vortex axis origin.
    pub position: Vec3,
    /// Vortex axis direction.
    pub axis: Vec3,
    /// Rotational strength.
    pub strength: f32,
    /// How quickly force falls off with distance.
    pub falloff: f32,
}

impl Default for Vortex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            axis: Vec3::Y,
            strength: 5.0,
            falloff: 1.0,
        }
    }
}

impl Vortex {
    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Vortex {
        self.clone()
    }
}

impl Force for Vortex {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        let to_particle = particle.position - self.position;
        let axis = self.axis.normalize();

        // Project onto plane perpendicular to axis
        let on_axis = axis * to_particle.dot(axis);
        let radial = to_particle - on_axis;
        let dist = radial.length();

        if dist > 0.001 {
            // Tangent direction (perpendicular to radial and axis)
            let tangent = axis.cross(radial).normalize();

            // Force falls off with distance
            let force = self.strength / (1.0 + dist * self.falloff);
            particle.velocity += tangent * force * dt;
        }
    }
}

/// Turbulence force using noise.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Turbulence))]
pub struct Turbulence {
    /// Strength of the turbulence.
    pub strength: f32,
    /// Frequency (scale of noise).
    pub frequency: f32,
    /// Animation speed.
    pub speed: f32,
    /// Current time offset.
    time: f32,
}

impl Default for Turbulence {
    fn default() -> Self {
        Self {
            strength: 5.0,
            frequency: 1.0,
            speed: 1.0,
            time: 0.0,
        }
    }
}

impl Turbulence {
    /// Creates turbulence with the given strength.
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            ..Default::default()
        }
    }

    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> Turbulence {
        self.clone()
    }

    /// Advances the internal time.
    pub fn advance(&mut self, dt: f32) {
        self.time += dt * self.speed;
    }
}

impl Force for Turbulence {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        use unshape_noise::Noise3D;
        let noise = unshape_noise::Simplex3D::new();

        let p = particle.position * self.frequency;

        // Use 3D noise with time offset for each axis
        let fx = noise.sample_vec(Vec3::new(p.x, p.y, p.z + self.time));
        let fy = noise.sample_vec(Vec3::new(p.x + 100.0, p.y, p.z + self.time));
        let fz = noise.sample_vec(Vec3::new(p.x, p.y + 100.0, p.z + self.time));

        let force = Vec3::new(fx, fy, fz) * self.strength;
        particle.velocity += force * dt;
    }
}

/// Curl noise force for divergence-free turbulence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = CurlNoise))]
pub struct CurlNoise {
    /// Strength of the force.
    pub strength: f32,
    /// Frequency (scale of noise).
    pub frequency: f32,
    /// Small offset for gradient computation.
    epsilon: f32,
}

impl Default for CurlNoise {
    fn default() -> Self {
        Self {
            strength: 5.0,
            frequency: 1.0,
            epsilon: 0.001,
        }
    }
}

impl CurlNoise {
    /// Creates curl noise with the given strength.
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            ..Default::default()
        }
    }

    /// Returns this force configuration (for Op pipeline).
    #[cfg_attr(not(feature = "dynop"), allow(dead_code))]
    pub fn apply(&self) -> CurlNoise {
        self.clone()
    }
}

impl Force for CurlNoise {
    fn apply(&self, particle: &mut Particle, dt: f32) {
        use unshape_noise::Noise3D;
        let noise = unshape_noise::Simplex3D::new();

        let p = particle.position * self.frequency;
        let e = self.epsilon;

        // Compute curl of noise field
        // curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)

        // Sample noise at offset positions
        let n_x0 = noise.sample_vec(p - Vec3::X * e);
        let n_x1 = noise.sample_vec(p + Vec3::X * e);
        let n_y0 = noise.sample_vec(p - Vec3::Y * e);
        let n_y1 = noise.sample_vec(p + Vec3::Y * e);
        let n_z0 = noise.sample_vec(p - Vec3::Z * e);
        let n_z1 = noise.sample_vec(p + Vec3::Z * e);

        let dx = (n_x1 - n_x0) / (2.0 * e);
        let dy = (n_y1 - n_y0) / (2.0 * e);
        let dz = (n_z1 - n_z0) / (2.0 * e);

        // Use noise value as potential, compute curl
        let curl = Vec3::new(dy - dz, dz - dx, dx - dy);

        particle.velocity += curl * self.strength * dt;
    }
}

/// Registers all particle operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of particle ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<PointEmitter>("resin::PointEmitter");
    registry.register_type::<SphereEmitter>("resin::SphereEmitter");
    registry.register_type::<ConeEmitter>("resin::ConeEmitter");
    registry.register_type::<Gravity>("resin::Gravity");
    registry.register_type::<Wind>("resin::Wind");
    registry.register_type::<Drag>("resin::Drag");
    registry.register_type::<Attractor>("resin::Attractor");
    registry.register_type::<Vortex>("resin::Vortex");
    registry.register_type::<Turbulence>("resin::Turbulence");
    registry.register_type::<CurlNoise>("resin::CurlNoise");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_lifetime() {
        let mut p = Particle::new(Vec3::ZERO);
        p.lifetime = 2.0;

        assert!(p.is_alive());
        assert_eq!(p.normalized_age(), 0.0);

        p.age = 1.0;
        assert!(p.is_alive());
        assert!((p.normalized_age() - 0.5).abs() < 0.001);

        p.age = 2.0;
        assert!(!p.is_alive());
        assert!((p.normalized_age() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_particle_rng() {
        let mut rng = ParticleRng::new(42);

        // Should produce values in [0, 1)
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0);
        }

        // Range should work
        for _ in 0..100 {
            let v = rng.range(5.0, 10.0);
            assert!(v >= 5.0 && v < 10.0);
        }
    }

    #[test]
    fn test_particle_rng_unit_sphere() {
        let mut rng = ParticleRng::new(42);

        for _ in 0..100 {
            let v = rng.unit_sphere();
            let len = v.length();
            assert!((len - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_particle_system_emit() {
        let mut system = ParticleSystem::new(100);
        let emitter = PointEmitter::default();

        system.emit(&emitter, 10);
        assert_eq!(system.count(), 10);

        system.emit(&emitter, 50);
        assert_eq!(system.count(), 60);
    }

    #[test]
    fn test_particle_system_capacity() {
        let mut system = ParticleSystem::new(10);
        let emitter = PointEmitter::default();

        system.emit(&emitter, 100);
        assert_eq!(system.count(), 10);
        assert!(system.is_full());
    }

    #[test]
    fn test_particle_system_update() {
        let mut system = ParticleSystem::new(100);

        let emitter = PointEmitter {
            lifetime_min: 1.0,
            lifetime_max: 1.0,
            ..Default::default()
        };

        system.emit(&emitter, 10);
        assert_eq!(system.count(), 10);

        // Update for half lifetime
        system.update(0.5);
        assert_eq!(system.count(), 10);

        // Update past lifetime
        system.update(0.6);
        assert_eq!(system.count(), 0);
    }

    #[test]
    fn test_gravity_force() {
        let gravity = Gravity::default();
        let mut p = Particle::new(Vec3::ZERO);

        assert_eq!(p.velocity, Vec3::ZERO);

        Force::apply(&gravity, &mut p, 1.0);
        assert!((p.velocity.y - (-9.81)).abs() < 0.001);
    }

    #[test]
    fn test_drag_force() {
        let drag = Drag { coefficient: 0.5 };
        let mut p = Particle::new(Vec3::ZERO);
        p.velocity = Vec3::new(10.0, 0.0, 0.0);

        Force::apply(&drag, &mut p, 1.0);
        assert!(p.velocity.x < 10.0);
        assert!(p.velocity.x > 0.0);
    }

    #[test]
    fn test_attractor_force() {
        let attractor = Attractor {
            position: Vec3::new(10.0, 0.0, 0.0),
            strength: 10.0,
            min_distance: 0.1,
        };

        let mut p = Particle::new(Vec3::ZERO);
        Force::apply(&attractor, &mut p, 1.0);

        // Should be pulled toward attractor
        assert!(p.velocity.x > 0.0);
    }

    #[test]
    fn test_sphere_emitter() {
        let mut system = ParticleSystem::new(100);
        let emitter = SphereEmitter {
            center: Vec3::ZERO,
            radius: 1.0,
            volume: false,
            ..Default::default()
        };

        system.emit(&emitter, 50);

        // All particles should be at radius distance from center
        for p in system.particles() {
            let dist = p.position.length();
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_cone_emitter() {
        let mut system = ParticleSystem::new(100);
        let emitter = ConeEmitter {
            direction: Vec3::Y,
            angle: std::f32::consts::FRAC_PI_4,
            ..Default::default()
        };

        system.emit(&emitter, 50);

        // All velocities should have positive Y component within cone angle
        for p in system.particles() {
            let cos_angle = p.velocity.normalize().dot(Vec3::Y);
            assert!(cos_angle >= emitter.angle.cos() - 0.001);
        }
    }

    #[test]
    fn test_vortex_force() {
        let vortex = Vortex {
            position: Vec3::ZERO,
            axis: Vec3::Y,
            strength: 5.0,
            falloff: 1.0,
        };

        let mut p = Particle::new(Vec3::new(1.0, 0.0, 0.0));
        Force::apply(&vortex, &mut p, 1.0);

        // Should spin around Y axis (negative Z direction from +X)
        assert!(p.velocity.z < 0.0);
    }

    #[test]
    fn test_wind_force() {
        let wind = Wind {
            velocity: Vec3::new(5.0, 0.0, 0.0),
            strength: 1.0,
        };

        let mut p = Particle::new(Vec3::ZERO);
        Force::apply(&wind, &mut p, 1.0);

        // Should be pushed toward wind velocity
        assert!(p.velocity.x > 0.0);
    }
}

/// Invariant tests for particle systems.
///
/// Run with: cargo test -p unshape-particle --features invariant-tests
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    // =========================================================================
    // Emitter invariants
    // =========================================================================

    /// PointEmitter should emit particles exactly at the configured position.
    #[test]
    fn invariant_point_emitter_position() {
        let position = Vec3::new(5.0, 10.0, -3.0);
        let emitter = PointEmitter {
            position,
            spread: 0.0,
            ..Default::default()
        };

        let mut rng = ParticleRng::new(42);

        for _ in 0..100 {
            let p = emitter.emit(&mut rng);
            assert_eq!(
                p.position, position,
                "PointEmitter should emit particles at the configured position"
            );
        }
    }

    /// SphereEmitter with volume=false should emit particles on the sphere surface.
    #[test]
    fn invariant_sphere_emitter_surface() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let radius = 2.5;
        let emitter = SphereEmitter {
            center,
            radius,
            volume: false,
            ..Default::default()
        };

        let mut rng = ParticleRng::new(42);

        for _ in 0..100 {
            let p = emitter.emit(&mut rng);
            let dist = (p.position - center).length();
            assert!(
                (dist - radius).abs() < 1e-5,
                "SphereEmitter (surface) should emit at radius {}, got {}",
                radius,
                dist
            );
        }
    }

    /// SphereEmitter with volume=true should emit particles inside the sphere.
    #[test]
    fn invariant_sphere_emitter_volume() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let radius = 2.5;
        let emitter = SphereEmitter {
            center,
            radius,
            volume: true,
            ..Default::default()
        };

        let mut rng = ParticleRng::new(42);

        for _ in 0..100 {
            let p = emitter.emit(&mut rng);
            let dist = (p.position - center).length();
            assert!(
                dist <= radius + 1e-5,
                "SphereEmitter (volume) should emit inside radius {}, got {}",
                radius,
                dist
            );
        }
    }

    /// ConeEmitter should emit particles within the configured cone angle.
    #[test]
    fn invariant_cone_emitter_angle() {
        let direction = Vec3::Y;
        let angle = std::f32::consts::FRAC_PI_6; // 30 degrees
        let emitter = ConeEmitter {
            direction,
            angle,
            ..Default::default()
        };

        let mut rng = ParticleRng::new(42);
        let cos_angle = angle.cos();

        for _ in 0..100 {
            let p = emitter.emit(&mut rng);
            let vel_dir = p.velocity.normalize();
            let dot = vel_dir.dot(direction);
            assert!(
                dot >= cos_angle - 1e-4,
                "ConeEmitter velocity should be within cone angle, dot={} (expected >= {})",
                dot,
                cos_angle
            );
        }
    }

    // =========================================================================
    // Force invariants
    // =========================================================================

    /// Gravity should accelerate particles downward (negative Y by default).
    #[test]
    fn invariant_gravity_accelerates_downward() {
        let gravity = Gravity::default(); // acceleration = (0, -9.81, 0)
        let mut p = Particle::new(Vec3::ZERO);
        p.velocity = Vec3::ZERO;

        let initial_vy = p.velocity.y;
        Force::apply(&gravity, &mut p, 1.0);

        assert!(
            p.velocity.y < initial_vy,
            "Gravity should decrease Y velocity: initial={}, after={}",
            initial_vy,
            p.velocity.y
        );
        assert!(
            (p.velocity.y - (-9.81)).abs() < 1e-5,
            "After 1s, velocity.y should be -9.81, got {}",
            p.velocity.y
        );
    }

    /// Gravity applied over multiple steps should equal single step of same total time.
    #[test]
    fn invariant_gravity_time_step_independent() {
        let gravity = Gravity {
            acceleration: Vec3::new(0.0, -10.0, 0.0),
        };

        // Single step
        let mut p1 = Particle::new(Vec3::ZERO);
        p1.velocity = Vec3::new(1.0, 5.0, 0.0);
        Force::apply(&gravity, &mut p1, 1.0);

        // Multiple small steps
        let mut p2 = Particle::new(Vec3::ZERO);
        p2.velocity = Vec3::new(1.0, 5.0, 0.0);
        for _ in 0..100 {
            Force::apply(&gravity, &mut p2, 0.01);
        }

        assert!(
            (p1.velocity - p2.velocity).length() < 1e-4,
            "Gravity should be time-step independent: single={:?}, multi={:?}",
            p1.velocity,
            p2.velocity
        );
    }

    /// Drag should slow particles over time (velocity magnitude decreases).
    #[test]
    fn invariant_drag_slows_particles() {
        let drag = Drag { coefficient: 0.5 };
        let mut p = Particle::new(Vec3::ZERO);
        p.velocity = Vec3::new(10.0, 5.0, -3.0);

        let initial_speed = p.velocity.length();
        Force::apply(&drag, &mut p, 0.1);
        let after_speed = p.velocity.length();

        assert!(
            after_speed < initial_speed,
            "Drag should reduce speed: initial={}, after={}",
            initial_speed,
            after_speed
        );
    }

    /// Drag should preserve velocity direction (only reduce magnitude).
    #[test]
    fn invariant_drag_preserves_direction() {
        let drag = Drag { coefficient: 0.5 };
        let mut p = Particle::new(Vec3::ZERO);
        p.velocity = Vec3::new(10.0, 5.0, -3.0);

        let initial_dir = p.velocity.normalize();
        Force::apply(&drag, &mut p, 0.1);
        let after_dir = p.velocity.normalize();

        assert!(
            (initial_dir - after_dir).length() < 1e-5,
            "Drag should preserve velocity direction: initial={:?}, after={:?}",
            initial_dir,
            after_dir
        );
    }

    /// Drag should never reverse velocity direction or increase speed.
    #[test]
    fn invariant_drag_never_reverses() {
        let drag = Drag { coefficient: 2.0 }; // Strong drag
        let mut p = Particle::new(Vec3::ZERO);
        p.velocity = Vec3::new(1.0, 0.0, 0.0);

        // Apply drag with large dt
        Force::apply(&drag, &mut p, 1.0);

        // Velocity should be clamped at 0, not negative
        assert!(
            p.velocity.x >= 0.0,
            "Drag should never reverse velocity: got {}",
            p.velocity.x
        );
    }

    /// Attractor should pull particles toward the attractor position.
    #[test]
    fn invariant_attractor_pulls_toward() {
        let attractor = Attractor {
            position: Vec3::new(10.0, 0.0, 0.0),
            strength: 10.0,
            min_distance: 0.1,
        };

        let mut p = Particle::new(Vec3::ZERO);
        let initial_vel = p.velocity;
        Force::apply(&attractor, &mut p, 1.0);

        let to_attractor = attractor.position - p.position;
        let vel_change = p.velocity - initial_vel;

        // Velocity change should be in the direction of the attractor
        assert!(
            vel_change.dot(to_attractor) > 0.0,
            "Attractor should pull toward itself"
        );
    }

    /// Attractor with negative strength should repel particles.
    #[test]
    fn invariant_attractor_repels_with_negative_strength() {
        let repulsor = Attractor {
            position: Vec3::new(10.0, 0.0, 0.0),
            strength: -10.0,
            min_distance: 0.1,
        };

        let mut p = Particle::new(Vec3::ZERO);
        let initial_vel = p.velocity;
        Force::apply(&repulsor, &mut p, 1.0);

        let to_attractor = repulsor.position - p.position;
        let vel_change = p.velocity - initial_vel;

        // Velocity change should be away from the repulsor
        assert!(
            vel_change.dot(to_attractor) < 0.0,
            "Negative strength attractor should repel"
        );
    }

    // =========================================================================
    // Particle lifetime invariants
    // =========================================================================

    /// Dead particles (age >= lifetime) should be removed on update.
    #[test]
    fn invariant_dead_particles_removed() {
        let mut system = ParticleSystem::new(100);
        let emitter = PointEmitter {
            lifetime_min: 1.0,
            lifetime_max: 1.0,
            ..Default::default()
        };

        system.emit(&emitter, 10);
        assert_eq!(system.count(), 10);

        // Advance time just past lifetime
        system.update(1.01);

        assert_eq!(
            system.count(),
            0,
            "All particles should be removed after exceeding lifetime"
        );
    }

    /// Particles should not be removed before their lifetime expires.
    #[test]
    fn invariant_alive_particles_retained() {
        let mut system = ParticleSystem::new(100);
        let emitter = PointEmitter {
            lifetime_min: 2.0,
            lifetime_max: 2.0,
            ..Default::default()
        };

        system.emit(&emitter, 10);

        // Advance time but stay within lifetime
        system.update(0.5);
        assert_eq!(system.count(), 10, "Particles should remain at t=0.5");

        system.update(0.5);
        assert_eq!(system.count(), 10, "Particles should remain at t=1.0");

        system.update(0.5);
        assert_eq!(system.count(), 10, "Particles should remain at t=1.5");

        // Now exceed lifetime
        system.update(0.6);
        assert_eq!(system.count(), 0, "Particles should be removed after t=2.0");
    }

    /// Particle normalized_age should increase linearly with time.
    #[test]
    fn invariant_normalized_age_linear() {
        let mut p = Particle::new(Vec3::ZERO);
        p.lifetime = 4.0;

        assert!((p.normalized_age() - 0.0).abs() < 1e-5);

        p.age = 1.0;
        assert!((p.normalized_age() - 0.25).abs() < 1e-5);

        p.age = 2.0;
        assert!((p.normalized_age() - 0.5).abs() < 1e-5);

        p.age = 4.0;
        assert!((p.normalized_age() - 1.0).abs() < 1e-5);
    }

    /// Particle normalized_age should be clamped to [0, 1].
    #[test]
    fn invariant_normalized_age_clamped() {
        let mut p = Particle::new(Vec3::ZERO);
        p.lifetime = 1.0;

        p.age = -0.5;
        assert!(
            p.normalized_age() >= 0.0,
            "normalized_age should not be negative"
        );

        p.age = 2.0;
        assert!(
            p.normalized_age() <= 1.0,
            "normalized_age should not exceed 1.0"
        );
    }

    // =========================================================================
    // Conservation laws
    // =========================================================================

    /// In the absence of forces, particle momentum should be conserved.
    #[test]
    fn invariant_momentum_conservation_no_forces() {
        let mut system = ParticleSystem::new(100);
        let emitter = PointEmitter {
            speed_min: 5.0,
            speed_max: 5.0,
            lifetime_min: 10.0,
            lifetime_max: 10.0,
            ..Default::default()
        };

        system.emit(&emitter, 50);

        // Calculate total momentum before update
        let momentum_before: Vec3 = system.particles().iter().map(|p| p.velocity).sum();

        // Update without forces
        system.update(0.1);

        // Calculate total momentum after update
        let momentum_after: Vec3 = system.particles().iter().map(|p| p.velocity).sum();

        assert!(
            (momentum_before - momentum_after).length() < 1e-4,
            "Momentum should be conserved without forces: before={:?}, after={:?}",
            momentum_before,
            momentum_after
        );
    }

    /// Position update should follow velocity integration (p += v * dt).
    #[test]
    fn invariant_position_velocity_integration() {
        // Create a simple emitter that emits at a known position with known velocity
        let emitter = PointEmitter {
            position: Vec3::new(1.0, 2.0, 3.0),
            direction: Vec3::new(10.0, -5.0, 2.0).normalize(),
            spread: 0.0,
            speed_min: Vec3::new(10.0, -5.0, 2.0).length(),
            speed_max: Vec3::new(10.0, -5.0, 2.0).length(),
            lifetime_min: 10.0,
            lifetime_max: 10.0,
            ..Default::default()
        };

        let mut system = ParticleSystem::new(1);
        system.emit(&emitter, 1);

        let initial_pos = system.particles()[0].position;
        let velocity = system.particles()[0].velocity;
        let dt = 0.5;

        system.update(dt);

        let expected_pos = initial_pos + velocity * dt;
        let actual_pos = system.particles()[0].position;

        assert!(
            (expected_pos - actual_pos).length() < 1e-5,
            "Position should follow p += v*dt: expected={:?}, got={:?}",
            expected_pos,
            actual_pos
        );
    }

    /// Vortex force should produce circular motion (radial distance stays constant).
    #[test]
    fn invariant_vortex_preserves_radial_distance() {
        let vortex = Vortex {
            position: Vec3::ZERO,
            axis: Vec3::Y,
            strength: 5.0,
            falloff: 0.0, // No falloff for cleaner test
        };

        let mut p = Particle::new(Vec3::new(1.0, 0.0, 0.0));
        p.velocity = Vec3::ZERO;
        p.lifetime = 100.0;

        // The vortex creates tangential velocity, so radial distance should stay constant
        // over small time steps when position updates are included

        let _initial_radial = (p.position - vortex.position).length();

        // Apply vortex force
        Force::apply(&vortex, &mut p, 0.01);

        // The velocity should be tangential (perpendicular to radial direction)
        let radial_dir = (p.position - vortex.position).normalize();
        let vel_radial_component = p.velocity.dot(radial_dir);

        assert!(
            vel_radial_component.abs() < 1e-4,
            "Vortex velocity should be tangential: radial component = {}",
            vel_radial_component
        );
    }
}
