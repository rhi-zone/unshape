use glam::Vec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::helpers::*;

/// Configuration for smoke simulation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Smoke))]
pub struct Smoke {
    /// Diffusion rate for velocity.
    pub diffusion: f32,
    /// Number of iterations for solver.
    pub iterations: u32,
    /// Time step.
    pub dt: f32,
    /// Buoyancy coefficient (how much hot gas rises).
    pub buoyancy: f32,
    /// Ambient temperature.
    pub ambient_temperature: f32,
    /// Temperature dissipation rate (cooling).
    pub temperature_dissipation: f32,
    /// Density dissipation rate.
    pub density_dissipation: f32,
}

impl Default for Smoke {
    fn default() -> Self {
        Self {
            diffusion: 0.0,
            iterations: 20,
            dt: 0.1,
            buoyancy: 1.0,
            ambient_temperature: 0.0,
            temperature_dissipation: 0.01,
            density_dissipation: 0.005,
        }
    }
}

impl Smoke {
    /// Applies this configuration, returning it as-is.
    pub fn apply(&self) -> Smoke {
        self.clone()
    }
}

/// Backwards-compatible type alias.
pub type SmokeConfig = Smoke;

/// 2D smoke/gas simulation with buoyancy.
#[derive(Clone)]
pub struct SmokeGrid2D {
    width: usize,
    height: usize,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vx0: Vec<f32>,
    vy0: Vec<f32>,
    density: Vec<f32>,
    density0: Vec<f32>,
    temperature: Vec<f32>,
    temperature0: Vec<f32>,
    config: SmokeConfig,
}

impl SmokeGrid2D {
    /// Create a new 2D smoke grid.
    pub fn new(width: usize, height: usize, config: SmokeConfig) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            density: vec![0.0; size],
            density0: vec![0.0; size],
            temperature: vec![config.ambient_temperature; size],
            temperature0: vec![config.ambient_temperature; size],
            config,
        }
    }

    /// Get grid dimensions.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get density at a grid cell.
    pub fn density(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.density[idx2d(x, y, self.width)]
        } else {
            0.0
        }
    }

    /// Get temperature at a grid cell.
    pub fn temperature(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.temperature[idx2d(x, y, self.width)]
        } else {
            self.config.ambient_temperature
        }
    }

    /// Get velocity at a grid cell.
    pub fn velocity(&self, x: usize, y: usize) -> Vec2 {
        if x < self.width && y < self.height {
            let i = idx2d(x, y, self.width);
            Vec2::new(self.vx[i], self.vy[i])
        } else {
            Vec2::ZERO
        }
    }

    /// Add smoke (density + temperature) at a position.
    pub fn add_smoke(&mut self, x: usize, y: usize, density: f32, temperature: f32) {
        if x < self.width && y < self.height {
            let i = idx2d(x, y, self.width);
            self.density[i] += density;
            self.temperature[i] += temperature;
        }
    }

    /// Add velocity at a position.
    pub fn add_velocity(&mut self, x: usize, y: usize, vx: f32, vy: f32) {
        if x < self.width && y < self.height {
            let i = idx2d(x, y, self.width);
            self.vx[i] += vx;
            self.vy[i] += vy;
        }
    }

    /// Step the simulation forward.
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let diff = self.config.diffusion;
        let iters = self.config.iterations;
        let w = self.width;
        let h = self.height;

        // Apply buoyancy force
        self.apply_buoyancy();

        // Velocity step
        std::mem::swap(&mut self.vx, &mut self.vx0);
        std::mem::swap(&mut self.vy, &mut self.vy0);

        diffuse_2d(1, &mut self.vx, &self.vx0, diff, dt, iters, w, h);
        diffuse_2d(2, &mut self.vy, &self.vy0, diff, dt, iters, w, h);

        project_2d(
            &mut self.vx,
            &mut self.vy,
            &mut self.vx0,
            &mut self.vy0,
            iters,
            w,
            h,
        );

        std::mem::swap(&mut self.vx, &mut self.vx0);
        std::mem::swap(&mut self.vy, &mut self.vy0);

        advect_2d(1, &mut self.vx, &self.vx0, &self.vx0, &self.vy0, dt, w, h);
        advect_2d(2, &mut self.vy, &self.vy0, &self.vx0, &self.vy0, dt, w, h);

        project_2d(
            &mut self.vx,
            &mut self.vy,
            &mut self.vx0,
            &mut self.vy0,
            iters,
            w,
            h,
        );

        // Density step
        std::mem::swap(&mut self.density, &mut self.density0);
        diffuse_2d(0, &mut self.density, &self.density0, diff, dt, iters, w, h);

        std::mem::swap(&mut self.density, &mut self.density0);
        advect_2d(
            0,
            &mut self.density,
            &self.density0,
            &self.vx,
            &self.vy,
            dt,
            w,
            h,
        );

        // Temperature step
        std::mem::swap(&mut self.temperature, &mut self.temperature0);
        diffuse_2d(
            0,
            &mut self.temperature,
            &self.temperature0,
            diff,
            dt,
            iters,
            w,
            h,
        );

        std::mem::swap(&mut self.temperature, &mut self.temperature0);
        advect_2d(
            0,
            &mut self.temperature,
            &self.temperature0,
            &self.vx,
            &self.vy,
            dt,
            w,
            h,
        );

        // Apply dissipation
        self.apply_dissipation();
    }

    fn apply_buoyancy(&mut self) {
        let buoyancy = self.config.buoyancy;
        let ambient = self.config.ambient_temperature;

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = idx2d(i, j, self.width);
                let temp_diff = self.temperature[idx] - ambient;
                // Hot gas rises (positive y is up)
                self.vy[idx] += buoyancy * temp_diff * self.config.dt;
            }
        }
    }

    fn apply_dissipation(&mut self) {
        let density_factor = 1.0 - self.config.density_dissipation;
        let temp_factor = 1.0 - self.config.temperature_dissipation;
        let ambient = self.config.ambient_temperature;

        for i in 0..self.density.len() {
            self.density[i] *= density_factor;
            // Cool towards ambient
            self.temperature[i] = ambient + (self.temperature[i] - ambient) * temp_factor;
        }
    }

    /// Clear all fields.
    pub fn clear(&mut self) {
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vx0.fill(0.0);
        self.vy0.fill(0.0);
        self.density.fill(0.0);
        self.density0.fill(0.0);
        let ambient = self.config.ambient_temperature;
        self.temperature.fill(ambient);
        self.temperature0.fill(ambient);
    }

    /// Get the density field as a slice.
    pub fn density_field(&self) -> &[f32] {
        &self.density
    }

    /// Get the temperature field as a slice.
    pub fn temperature_field(&self) -> &[f32] {
        &self.temperature
    }
}

/// 3D smoke/gas simulation with buoyancy.
#[derive(Clone)]
pub struct SmokeGrid3D {
    width: usize,
    height: usize,
    depth: usize,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    vx0: Vec<f32>,
    vy0: Vec<f32>,
    vz0: Vec<f32>,
    density: Vec<f32>,
    density0: Vec<f32>,
    temperature: Vec<f32>,
    temperature0: Vec<f32>,
    config: SmokeConfig,
}

impl SmokeGrid3D {
    /// Create a new 3D smoke grid.
    pub fn new(width: usize, height: usize, depth: usize, config: SmokeConfig) -> Self {
        let size = width * height * depth;
        Self {
            width,
            height,
            depth,
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            vz: vec![0.0; size],
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            vz0: vec![0.0; size],
            density: vec![0.0; size],
            density0: vec![0.0; size],
            temperature: vec![config.ambient_temperature; size],
            temperature0: vec![config.ambient_temperature; size],
            config,
        }
    }

    /// Get grid dimensions.
    pub fn size(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    /// Get density at a grid cell.
    pub fn density(&self, x: usize, y: usize, z: usize) -> f32 {
        if x < self.width && y < self.height && z < self.depth {
            self.density[idx3d(x, y, z, self.width, self.height)]
        } else {
            0.0
        }
    }

    /// Get temperature at a grid cell.
    pub fn temperature(&self, x: usize, y: usize, z: usize) -> f32 {
        if x < self.width && y < self.height && z < self.depth {
            self.temperature[idx3d(x, y, z, self.width, self.height)]
        } else {
            self.config.ambient_temperature
        }
    }

    /// Add smoke (density + temperature) at a position.
    pub fn add_smoke(&mut self, x: usize, y: usize, z: usize, density: f32, temperature: f32) {
        if x < self.width && y < self.height && z < self.depth {
            let i = idx3d(x, y, z, self.width, self.height);
            self.density[i] += density;
            self.temperature[i] += temperature;
        }
    }

    /// Add velocity at a position.
    pub fn add_velocity(&mut self, x: usize, y: usize, z: usize, vel: glam::Vec3) {
        if x < self.width && y < self.height && z < self.depth {
            let i = idx3d(x, y, z, self.width, self.height);
            self.vx[i] += vel.x;
            self.vy[i] += vel.y;
            self.vz[i] += vel.z;
        }
    }

    /// Step the simulation forward.
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let diff = self.config.diffusion;
        let iters = self.config.iterations;
        let w = self.width;
        let h = self.height;
        let d = self.depth;

        // Apply buoyancy force
        self.apply_buoyancy();

        // Velocity step
        std::mem::swap(&mut self.vx, &mut self.vx0);
        std::mem::swap(&mut self.vy, &mut self.vy0);
        std::mem::swap(&mut self.vz, &mut self.vz0);

        diffuse_3d(1, &mut self.vx, &self.vx0, diff, dt, iters, w, h, d);
        diffuse_3d(2, &mut self.vy, &self.vy0, diff, dt, iters, w, h, d);
        diffuse_3d(3, &mut self.vz, &self.vz0, diff, dt, iters, w, h, d);

        project_3d(
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
            &mut self.vx0,
            &mut self.vy0,
            iters,
            w,
            h,
            d,
        );

        std::mem::swap(&mut self.vx, &mut self.vx0);
        std::mem::swap(&mut self.vy, &mut self.vy0);
        std::mem::swap(&mut self.vz, &mut self.vz0);

        advect_3d(
            1,
            &mut self.vx,
            &self.vx0,
            &self.vx0,
            &self.vy0,
            &self.vz0,
            dt,
            w,
            h,
            d,
        );
        advect_3d(
            2,
            &mut self.vy,
            &self.vy0,
            &self.vx0,
            &self.vy0,
            &self.vz0,
            dt,
            w,
            h,
            d,
        );
        advect_3d(
            3,
            &mut self.vz,
            &self.vz0,
            &self.vx0,
            &self.vy0,
            &self.vz0,
            dt,
            w,
            h,
            d,
        );

        project_3d(
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
            &mut self.vx0,
            &mut self.vy0,
            iters,
            w,
            h,
            d,
        );

        // Density step
        std::mem::swap(&mut self.density, &mut self.density0);
        diffuse_3d(
            0,
            &mut self.density,
            &self.density0,
            diff,
            dt,
            iters,
            w,
            h,
            d,
        );

        std::mem::swap(&mut self.density, &mut self.density0);
        advect_3d(
            0,
            &mut self.density,
            &self.density0,
            &self.vx,
            &self.vy,
            &self.vz,
            dt,
            w,
            h,
            d,
        );

        // Temperature step
        std::mem::swap(&mut self.temperature, &mut self.temperature0);
        diffuse_3d(
            0,
            &mut self.temperature,
            &self.temperature0,
            diff,
            dt,
            iters,
            w,
            h,
            d,
        );

        std::mem::swap(&mut self.temperature, &mut self.temperature0);
        advect_3d(
            0,
            &mut self.temperature,
            &self.temperature0,
            &self.vx,
            &self.vy,
            &self.vz,
            dt,
            w,
            h,
            d,
        );

        // Apply dissipation
        self.apply_dissipation();
    }

    fn apply_buoyancy(&mut self) {
        let buoyancy = self.config.buoyancy;
        let ambient = self.config.ambient_temperature;

        for k in 1..self.depth - 1 {
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    let idx = idx3d(i, j, k, self.width, self.height);
                    let temp_diff = self.temperature[idx] - ambient;
                    // Hot gas rises (positive y is up)
                    self.vy[idx] += buoyancy * temp_diff * self.config.dt;
                }
            }
        }
    }

    fn apply_dissipation(&mut self) {
        let density_factor = 1.0 - self.config.density_dissipation;
        let temp_factor = 1.0 - self.config.temperature_dissipation;
        let ambient = self.config.ambient_temperature;

        for i in 0..self.density.len() {
            self.density[i] *= density_factor;
            self.temperature[i] = ambient + (self.temperature[i] - ambient) * temp_factor;
        }
    }

    /// Clear all fields.
    pub fn clear(&mut self) {
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vz.fill(0.0);
        self.vx0.fill(0.0);
        self.vy0.fill(0.0);
        self.vz0.fill(0.0);
        self.density.fill(0.0);
        self.density0.fill(0.0);
        let ambient = self.config.ambient_temperature;
        self.temperature.fill(ambient);
        self.temperature0.fill(ambient);
    }

    /// Get the density field as a slice.
    pub fn density_field(&self) -> &[f32] {
        &self.density
    }

    /// Get the temperature field as a slice.
    pub fn temperature_field(&self) -> &[f32] {
        &self.temperature
    }
}
