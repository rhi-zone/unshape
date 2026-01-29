use glam::Vec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::helpers::*;

/// Configuration for grid-based fluid simulation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Fluid))]
pub struct Fluid {
    /// Diffusion rate (viscosity).
    pub diffusion: f32,
    /// Number of iterations for Gauss-Seidel solver.
    pub iterations: u32,
    /// Time step for simulation.
    pub dt: f32,
}

impl Default for Fluid {
    fn default() -> Self {
        Self {
            diffusion: 0.0001,
            iterations: 20,
            dt: 0.1,
        }
    }
}

impl Fluid {
    /// Applies this configuration, returning it as-is.
    pub fn apply(&self) -> Fluid {
        self.clone()
    }
}

/// Backwards-compatible type alias.
pub type FluidConfig = Fluid;

/// 2D grid-based fluid simulation using stable fluids method.
///
/// Based on Jos Stam's "Stable Fluids" (1999).
#[derive(Clone)]
pub struct FluidGrid2D {
    width: usize,
    height: usize,
    /// Velocity field (x component).
    vx: Vec<f32>,
    /// Velocity field (y component).
    vy: Vec<f32>,
    /// Previous velocity (x).
    vx0: Vec<f32>,
    /// Previous velocity (y).
    vy0: Vec<f32>,
    /// Density field.
    density: Vec<f32>,
    /// Previous density.
    density0: Vec<f32>,
    config: FluidConfig,
}

impl FluidGrid2D {
    /// Create a new 2D fluid grid.
    pub fn new(width: usize, height: usize, config: FluidConfig) -> Self {
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

    /// Get velocity at a grid cell.
    pub fn velocity(&self, x: usize, y: usize) -> Vec2 {
        if x < self.width && y < self.height {
            let i = idx2d(x, y, self.width);
            Vec2::new(self.vx[i], self.vy[i])
        } else {
            Vec2::ZERO
        }
    }

    /// Sample density with bilinear interpolation.
    pub fn sample_density(&self, pos: Vec2) -> f32 {
        bilinear_sample_2d(&self.density, pos, self.width, self.height)
    }

    /// Sample velocity with bilinear interpolation.
    pub fn sample_velocity(&self, pos: Vec2) -> Vec2 {
        let vx = bilinear_sample_2d(&self.vx, pos, self.width, self.height);
        let vy = bilinear_sample_2d(&self.vy, pos, self.width, self.height);
        Vec2::new(vx, vy)
    }

    /// Add density at a position.
    pub fn add_density(&mut self, x: usize, y: usize, amount: f32) {
        if x < self.width && y < self.height {
            let i = idx2d(x, y, self.width);
            self.density[i] += amount;
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
    }

    /// Clear all fields.
    pub fn clear(&mut self) {
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vx0.fill(0.0);
        self.vy0.fill(0.0);
        self.density.fill(0.0);
        self.density0.fill(0.0);
    }

    /// Get the density field as a slice.
    pub fn density_field(&self) -> &[f32] {
        &self.density
    }

    /// Get the velocity field as slices.
    pub fn velocity_field(&self) -> (&[f32], &[f32]) {
        (&self.vx, &self.vy)
    }
}

/// 3D grid-based fluid simulation.
#[derive(Clone)]
pub struct FluidGrid3D {
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
    config: FluidConfig,
}

impl FluidGrid3D {
    /// Create a new 3D fluid grid.
    pub fn new(width: usize, height: usize, depth: usize, config: FluidConfig) -> Self {
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

    /// Get velocity at a grid cell.
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> glam::Vec3 {
        if x < self.width && y < self.height && z < self.depth {
            let i = idx3d(x, y, z, self.width, self.height);
            glam::Vec3::new(self.vx[i], self.vy[i], self.vz[i])
        } else {
            glam::Vec3::ZERO
        }
    }

    /// Add density at a position.
    pub fn add_density(&mut self, x: usize, y: usize, z: usize, amount: f32) {
        if x < self.width && y < self.height && z < self.depth {
            let i = idx3d(x, y, z, self.width, self.height);
            self.density[i] += amount;
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
    }

    /// Get the density field as a slice.
    pub fn density_field(&self) -> &[f32] {
        &self.density
    }
}
