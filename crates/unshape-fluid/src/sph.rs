use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration for SPH simulation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Sph))]
pub struct Sph {
    /// Rest density of the fluid.
    pub rest_density: f32,
    /// Gas constant for pressure calculation.
    pub gas_constant: f32,
    /// Viscosity coefficient.
    pub viscosity: f32,
    /// Smoothing radius (kernel size).
    pub h: f32,
    /// Time step.
    pub dt: f32,
    /// Gravity.
    pub gravity: Vec2,
    /// Boundary damping.
    pub boundary_damping: f32,
}

impl Default for Sph {
    fn default() -> Self {
        Self {
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 250.0,
            h: 16.0,
            dt: 0.0007,
            gravity: Vec2::new(0.0, -9.81 * 1000.0),
            boundary_damping: 0.3,
        }
    }
}

impl Sph {
    /// Applies this configuration, returning it as-is.
    pub fn apply(&self) -> Sph {
        self.clone()
    }
}

/// Backwards-compatible type alias.
pub type SphConfig = Sph;

/// A single SPH particle.
#[derive(Clone, Debug)]
pub struct SphParticle2D {
    /// Current position.
    pub position: Vec2,
    /// Current velocity.
    pub velocity: Vec2,
    /// Accumulated force this timestep.
    pub force: Vec2,
    /// Computed density at this particle.
    pub density: f32,
    /// Computed pressure at this particle.
    pub pressure: f32,
    /// Particle mass.
    pub mass: f32,
}

impl SphParticle2D {
    /// Create a new particle at rest.
    pub fn new(position: Vec2, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec2::ZERO,
            force: Vec2::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass,
        }
    }
}

/// 2D SPH fluid simulation.
pub struct Sph2D {
    /// All particles in the simulation.
    pub particles: Vec<SphParticle2D>,
    /// Simulation parameters.
    pub config: SphConfig,
    /// Simulation bounds (min, max).
    pub bounds: (Vec2, Vec2),
}

impl Sph2D {
    /// Create a new SPH simulation.
    pub fn new(config: SphConfig, bounds: (Vec2, Vec2)) -> Self {
        Self {
            particles: Vec::new(),
            config,
            bounds,
        }
    }

    /// Add a particle.
    pub fn add_particle(&mut self, position: Vec2, mass: f32) {
        self.particles.push(SphParticle2D::new(position, mass));
    }

    /// Add a block of particles.
    pub fn add_block(&mut self, min: Vec2, max: Vec2, spacing: f32, mass: f32) {
        let mut y = min.y;
        while y <= max.y {
            let mut x = min.x;
            while x <= max.x {
                self.add_particle(Vec2::new(x, y), mass);
                x += spacing;
            }
            y += spacing;
        }
    }

    /// Step the simulation forward.
    pub fn step(&mut self) {
        self.compute_density_pressure();
        self.compute_forces();
        self.integrate();
    }

    fn compute_density_pressure(&mut self) {
        let h = self.config.h;
        let h2 = h * h;
        let poly6_coeff = 315.0 / (64.0 * PI * h.powi(9));

        for i in 0..self.particles.len() {
            let mut density = 0.0;
            let pos_i = self.particles[i].position;

            for j in 0..self.particles.len() {
                let r = pos_i - self.particles[j].position;
                let r2 = r.length_squared();

                if r2 < h2 {
                    density += self.particles[j].mass * poly6_coeff * (h2 - r2).powi(3);
                }
            }

            self.particles[i].density = density;
            self.particles[i].pressure =
                self.config.gas_constant * (density - self.config.rest_density);
        }
    }

    fn compute_forces(&mut self) {
        let h = self.config.h;
        let spiky_coeff = -45.0 / (PI * h.powi(6));
        let visc_coeff = 45.0 / (PI * h.powi(6));

        for i in 0..self.particles.len() {
            let mut f_pressure = Vec2::ZERO;
            let mut f_viscosity = Vec2::ZERO;

            let pos_i = self.particles[i].position;
            let vel_i = self.particles[i].velocity;
            let pressure_i = self.particles[i].pressure;
            let density_i = self.particles[i].density;

            for j in 0..self.particles.len() {
                if i == j {
                    continue;
                }

                let r = pos_i - self.particles[j].position;
                let r_len = r.length();

                if r_len < h && r_len > 0.0 {
                    let r_norm = r / r_len;

                    // Pressure force
                    f_pressure += -r_norm
                        * self.particles[j].mass
                        * (pressure_i + self.particles[j].pressure)
                        / (2.0 * self.particles[j].density)
                        * spiky_coeff
                        * (h - r_len).powi(2);

                    // Viscosity force
                    f_viscosity += self.config.viscosity
                        * self.particles[j].mass
                        * (self.particles[j].velocity - vel_i)
                        / self.particles[j].density
                        * visc_coeff
                        * (h - r_len);
                }
            }

            // Gravity
            let f_gravity = self.config.gravity * density_i;

            self.particles[i].force = f_pressure + f_viscosity + f_gravity;
        }
    }

    fn integrate(&mut self) {
        let dt = self.config.dt;
        let (min, max) = self.bounds;
        let damping = self.config.boundary_damping;

        for particle in &mut self.particles {
            // Integration
            if particle.density > 0.0 {
                particle.velocity += dt * particle.force / particle.density;
            }
            particle.position += dt * particle.velocity;

            // Boundary conditions
            if particle.position.x < min.x {
                particle.position.x = min.x;
                particle.velocity.x *= -damping;
            }
            if particle.position.x > max.x {
                particle.position.x = max.x;
                particle.velocity.x *= -damping;
            }
            if particle.position.y < min.y {
                particle.position.y = min.y;
                particle.velocity.y *= -damping;
            }
            if particle.position.y > max.y {
                particle.position.y = max.y;
                particle.velocity.y *= -damping;
            }
        }
    }

    /// Get particle positions.
    pub fn positions(&self) -> Vec<Vec2> {
        self.particles.iter().map(|p| p.position).collect()
    }

    /// Get particle velocities.
    pub fn velocities(&self) -> Vec<Vec2> {
        self.particles.iter().map(|p| p.velocity).collect()
    }

    /// Get particle densities.
    pub fn densities(&self) -> Vec<f32> {
        self.particles.iter().map(|p| p.density).collect()
    }
}

/// A single 3D SPH particle.
#[derive(Clone, Debug)]
pub struct SphParticle3D {
    /// Current position.
    pub position: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
    /// Accumulated force this timestep.
    pub force: Vec3,
    /// Computed density at this particle.
    pub density: f32,
    /// Computed pressure at this particle.
    pub pressure: f32,
    /// Particle mass.
    pub mass: f32,
}

impl SphParticle3D {
    /// Create a new particle at rest.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass,
        }
    }
}

/// Configuration for 3D SPH simulation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = SphParams3D))]
pub struct SphParams3D {
    /// Rest density of the fluid.
    pub rest_density: f32,
    /// Gas constant for pressure calculation.
    pub gas_constant: f32,
    /// Viscosity coefficient.
    pub viscosity: f32,
    /// Smoothing radius (kernel size).
    pub h: f32,
    /// Time step.
    pub dt: f32,
    /// Gravity.
    pub gravity: Vec3,
    /// Boundary damping.
    pub boundary_damping: f32,
}

impl Default for SphParams3D {
    fn default() -> Self {
        Self {
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 250.0,
            h: 0.1,
            dt: 0.0001,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            boundary_damping: 0.3,
        }
    }
}

impl SphParams3D {
    /// Applies this configuration, returning it as-is.
    pub fn apply(&self) -> SphParams3D {
        self.clone()
    }
}

/// Backwards-compatible type alias.
pub type SphConfig3D = SphParams3D;

/// 3D SPH fluid simulation.
pub struct Sph3D {
    /// All particles in the simulation.
    pub particles: Vec<SphParticle3D>,
    /// Simulation parameters.
    pub config: SphConfig3D,
    /// Simulation bounds (min, max).
    pub bounds: (Vec3, Vec3),
}

impl Sph3D {
    /// Create a new 3D SPH simulation.
    pub fn new(config: SphConfig3D, bounds: (Vec3, Vec3)) -> Self {
        Self {
            particles: Vec::new(),
            config,
            bounds,
        }
    }

    /// Add a particle.
    pub fn add_particle(&mut self, position: Vec3, mass: f32) {
        self.particles.push(SphParticle3D::new(position, mass));
    }

    /// Add a block of particles.
    pub fn add_block(&mut self, min: Vec3, max: Vec3, spacing: f32, mass: f32) {
        let mut z = min.z;
        while z <= max.z {
            let mut y = min.y;
            while y <= max.y {
                let mut x = min.x;
                while x <= max.x {
                    self.add_particle(Vec3::new(x, y, z), mass);
                    x += spacing;
                }
                y += spacing;
            }
            z += spacing;
        }
    }

    /// Step the simulation forward.
    pub fn step(&mut self) {
        self.compute_density_pressure();
        self.compute_forces();
        self.integrate();
    }

    fn compute_density_pressure(&mut self) {
        let h = self.config.h;
        let h2 = h * h;
        let poly6_coeff = 315.0 / (64.0 * PI * h.powi(9));

        for i in 0..self.particles.len() {
            let mut density = 0.0;
            let pos_i = self.particles[i].position;

            for j in 0..self.particles.len() {
                let r = pos_i - self.particles[j].position;
                let r2 = r.length_squared();

                if r2 < h2 {
                    density += self.particles[j].mass * poly6_coeff * (h2 - r2).powi(3);
                }
            }

            self.particles[i].density = density;
            self.particles[i].pressure =
                self.config.gas_constant * (density - self.config.rest_density);
        }
    }

    fn compute_forces(&mut self) {
        let h = self.config.h;
        let spiky_coeff = -45.0 / (PI * h.powi(6));
        let visc_coeff = 45.0 / (PI * h.powi(6));

        for i in 0..self.particles.len() {
            let mut f_pressure = Vec3::ZERO;
            let mut f_viscosity = Vec3::ZERO;

            let pos_i = self.particles[i].position;
            let vel_i = self.particles[i].velocity;
            let pressure_i = self.particles[i].pressure;
            let density_i = self.particles[i].density;

            for j in 0..self.particles.len() {
                if i == j {
                    continue;
                }

                let r = pos_i - self.particles[j].position;
                let r_len = r.length();

                if r_len < h && r_len > 0.0 {
                    let r_norm = r / r_len;

                    // Pressure force
                    f_pressure += -r_norm
                        * self.particles[j].mass
                        * (pressure_i + self.particles[j].pressure)
                        / (2.0 * self.particles[j].density)
                        * spiky_coeff
                        * (h - r_len).powi(2);

                    // Viscosity force
                    f_viscosity += self.config.viscosity
                        * self.particles[j].mass
                        * (self.particles[j].velocity - vel_i)
                        / self.particles[j].density
                        * visc_coeff
                        * (h - r_len);
                }
            }

            // Gravity
            let f_gravity = self.config.gravity * density_i;

            self.particles[i].force = f_pressure + f_viscosity + f_gravity;
        }
    }

    fn integrate(&mut self) {
        let dt = self.config.dt;
        let (min, max) = self.bounds;
        let damping = self.config.boundary_damping;

        for particle in &mut self.particles {
            // Integration
            if particle.density > 0.0 {
                particle.velocity += dt * particle.force / particle.density;
            }
            particle.position += dt * particle.velocity;

            // Boundary conditions
            if particle.position.x < min.x {
                particle.position.x = min.x;
                particle.velocity.x *= -damping;
            }
            if particle.position.x > max.x {
                particle.position.x = max.x;
                particle.velocity.x *= -damping;
            }
            if particle.position.y < min.y {
                particle.position.y = min.y;
                particle.velocity.y *= -damping;
            }
            if particle.position.y > max.y {
                particle.position.y = max.y;
                particle.velocity.y *= -damping;
            }
            if particle.position.z < min.z {
                particle.position.z = min.z;
                particle.velocity.z *= -damping;
            }
            if particle.position.z > max.z {
                particle.position.z = max.z;
                particle.velocity.z *= -damping;
            }
        }
    }

    /// Get particle positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.particles.iter().map(|p| p.position).collect()
    }
}
