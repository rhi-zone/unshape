//! Fluid simulation for resin.
//!
//! Provides grid-based and particle-based fluid simulation:
//! - `FluidGrid2D` - 2D Eulerian grid-based simulation (stable fluids)
//! - `FluidGrid3D` - 3D Eulerian grid-based simulation
//! - `Sph2D` - 2D Smoothed Particle Hydrodynamics
//! - `Sph3D` - 3D Smoothed Particle Hydrodynamics

use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ============================================================================
// Grid-based 2D Fluid Simulation (Stable Fluids)
// ============================================================================

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

// 2D helper functions

fn idx2d(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

fn bilinear_sample_2d(field: &[f32], pos: Vec2, width: usize, height: usize) -> f32 {
    let x = pos.x.clamp(0.0, (width - 1) as f32);
    let y = pos.y.clamp(0.0, (height - 1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    let v00 = field[idx2d(x0, y0, width)];
    let v10 = field[idx2d(x1, y0, width)];
    let v01 = field[idx2d(x0, y1, width)];
    let v11 = field[idx2d(x1, y1, width)];

    let v0 = v00 * (1.0 - sx) + v10 * sx;
    let v1 = v01 * (1.0 - sx) + v11 * sx;

    v0 * (1.0 - sy) + v1 * sy
}

fn diffuse_2d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    diff: f32,
    dt: f32,
    iters: u32,
    width: usize,
    height: usize,
) {
    let a = dt * diff * (width * height) as f32;
    lin_solve_2d(b, x, x0, a, 1.0 + 4.0 * a, iters, width, height);
}

fn lin_solve_2d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    a: f32,
    c: f32,
    iters: u32,
    width: usize,
    height: usize,
) {
    let c_recip = 1.0 / c;

    for _ in 0..iters {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx2d(i, j, width);
                x[idx] = (x0[idx]
                    + a * (x[idx2d(i + 1, j, width)]
                        + x[idx2d(i - 1, j, width)]
                        + x[idx2d(i, j + 1, width)]
                        + x[idx2d(i, j - 1, width)]))
                    * c_recip;
            }
        }
        set_bnd_2d(b, x, width, height);
    }
}

fn project_2d(
    vx: &mut [f32],
    vy: &mut [f32],
    p: &mut [f32],
    div: &mut [f32],
    iters: u32,
    width: usize,
    height: usize,
) {
    let h = 1.0 / width.max(height) as f32;

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            div[idx] = -0.5
                * h
                * (vx[idx2d(i + 1, j, width)] - vx[idx2d(i - 1, j, width)]
                    + vy[idx2d(i, j + 1, width)]
                    - vy[idx2d(i, j - 1, width)]);
            p[idx] = 0.0;
        }
    }

    set_bnd_2d(0, div, width, height);
    set_bnd_2d(0, p, width, height);
    lin_solve_2d(0, p, div, 1.0, 4.0, iters, width, height);

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            vx[idx] -= 0.5 * (p[idx2d(i + 1, j, width)] - p[idx2d(i - 1, j, width)]) / h;
            vy[idx] -= 0.5 * (p[idx2d(i, j + 1, width)] - p[idx2d(i, j - 1, width)]) / h;
        }
    }

    set_bnd_2d(1, vx, width, height);
    set_bnd_2d(2, vy, width, height);
}

fn advect_2d(
    b: i32,
    d: &mut [f32],
    d0: &[f32],
    vx: &[f32],
    vy: &[f32],
    dt: f32,
    width: usize,
    height: usize,
) {
    let dt0 = dt * width.max(height) as f32;

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            let idx = idx2d(i, j, width);
            let x = (i as f32 - dt0 * vx[idx]).clamp(0.5, width as f32 - 1.5);
            let y = (j as f32 - dt0 * vy[idx]).clamp(0.5, height as f32 - 1.5);

            let i0 = x.floor() as usize;
            let i1 = i0 + 1;
            let j0 = y.floor() as usize;
            let j1 = j0 + 1;

            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;

            d[idx] = s0 * (t0 * d0[idx2d(i0, j0, width)] + t1 * d0[idx2d(i0, j1, width)])
                + s1 * (t0 * d0[idx2d(i1, j0, width)] + t1 * d0[idx2d(i1, j1, width)]);
        }
    }

    set_bnd_2d(b, d, width, height);
}

fn set_bnd_2d(b: i32, x: &mut [f32], width: usize, height: usize) {
    // Top and bottom boundaries
    for i in 1..width - 1 {
        x[idx2d(i, 0, width)] = if b == 2 {
            -x[idx2d(i, 1, width)]
        } else {
            x[idx2d(i, 1, width)]
        };
        x[idx2d(i, height - 1, width)] = if b == 2 {
            -x[idx2d(i, height - 2, width)]
        } else {
            x[idx2d(i, height - 2, width)]
        };
    }

    // Left and right boundaries
    for j in 1..height - 1 {
        x[idx2d(0, j, width)] = if b == 1 {
            -x[idx2d(1, j, width)]
        } else {
            x[idx2d(1, j, width)]
        };
        x[idx2d(width - 1, j, width)] = if b == 1 {
            -x[idx2d(width - 2, j, width)]
        } else {
            x[idx2d(width - 2, j, width)]
        };
    }

    // Corners
    x[idx2d(0, 0, width)] = 0.5 * (x[idx2d(1, 0, width)] + x[idx2d(0, 1, width)]);
    x[idx2d(0, height - 1, width)] =
        0.5 * (x[idx2d(1, height - 1, width)] + x[idx2d(0, height - 2, width)]);
    x[idx2d(width - 1, 0, width)] =
        0.5 * (x[idx2d(width - 2, 0, width)] + x[idx2d(width - 1, 1, width)]);
    x[idx2d(width - 1, height - 1, width)] =
        0.5 * (x[idx2d(width - 2, height - 1, width)] + x[idx2d(width - 1, height - 2, width)]);
}

// ============================================================================
// Grid-based 3D Fluid Simulation
// ============================================================================

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
    pub fn velocity(&self, x: usize, y: usize, z: usize) -> Vec3 {
        if x < self.width && y < self.height && z < self.depth {
            let i = idx3d(x, y, z, self.width, self.height);
            Vec3::new(self.vx[i], self.vy[i], self.vz[i])
        } else {
            Vec3::ZERO
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
    pub fn add_velocity(&mut self, x: usize, y: usize, z: usize, vel: Vec3) {
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

// 3D helper functions

fn idx3d(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    z * width * height + y * width + x
}

fn diffuse_3d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    diff: f32,
    dt: f32,
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let a = dt * diff * (width * height * depth) as f32;
    lin_solve_3d(b, x, x0, a, 1.0 + 6.0 * a, iters, width, height, depth);
}

fn lin_solve_3d(
    b: i32,
    x: &mut [f32],
    x0: &[f32],
    a: f32,
    c: f32,
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let c_recip = 1.0 / c;

    for _ in 0..iters {
        for k in 1..depth - 1 {
            for j in 1..height - 1 {
                for i in 1..width - 1 {
                    let idx = idx3d(i, j, k, width, height);
                    x[idx] = (x0[idx]
                        + a * (x[idx3d(i + 1, j, k, width, height)]
                            + x[idx3d(i - 1, j, k, width, height)]
                            + x[idx3d(i, j + 1, k, width, height)]
                            + x[idx3d(i, j - 1, k, width, height)]
                            + x[idx3d(i, j, k + 1, width, height)]
                            + x[idx3d(i, j, k - 1, width, height)]))
                        * c_recip;
                }
            }
        }
        set_bnd_3d(b, x, width, height, depth);
    }
}

fn project_3d(
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    p: &mut [f32],
    div: &mut [f32],
    iters: u32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let h = 1.0 / (width.max(height).max(depth)) as f32;

    // Calculate divergence
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                div[idx] = -0.5
                    * h
                    * (vx[idx3d(i + 1, j, k, width, height)]
                        - vx[idx3d(i - 1, j, k, width, height)]
                        + vy[idx3d(i, j + 1, k, width, height)]
                        - vy[idx3d(i, j - 1, k, width, height)]
                        + vz[idx3d(i, j, k + 1, width, height)]
                        - vz[idx3d(i, j, k - 1, width, height)]);
                p[idx] = 0.0;
            }
        }
    }

    set_bnd_3d(0, div, width, height, depth);
    set_bnd_3d(0, p, width, height, depth);
    lin_solve_3d(0, p, div, 1.0, 6.0, iters, width, height, depth);

    // Subtract pressure gradient
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                vx[idx] -= 0.5
                    * (p[idx3d(i + 1, j, k, width, height)] - p[idx3d(i - 1, j, k, width, height)])
                    / h;
                vy[idx] -= 0.5
                    * (p[idx3d(i, j + 1, k, width, height)] - p[idx3d(i, j - 1, k, width, height)])
                    / h;
                vz[idx] -= 0.5
                    * (p[idx3d(i, j, k + 1, width, height)] - p[idx3d(i, j, k - 1, width, height)])
                    / h;
            }
        }
    }

    set_bnd_3d(1, vx, width, height, depth);
    set_bnd_3d(2, vy, width, height, depth);
    set_bnd_3d(3, vz, width, height, depth);
}

fn advect_3d(
    b: i32,
    d: &mut [f32],
    d0: &[f32],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    dt: f32,
    width: usize,
    height: usize,
    depth: usize,
) {
    let dt0 = dt * (width.max(height).max(depth)) as f32;

    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                let idx = idx3d(i, j, k, width, height);
                let x = (i as f32 - dt0 * vx[idx]).clamp(0.5, width as f32 - 1.5);
                let y = (j as f32 - dt0 * vy[idx]).clamp(0.5, height as f32 - 1.5);
                let z = (k as f32 - dt0 * vz[idx]).clamp(0.5, depth as f32 - 1.5);

                let i0 = x.floor() as usize;
                let i1 = i0 + 1;
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;
                let k0 = z.floor() as usize;
                let k1 = k0 + 1;

                let s1 = x - i0 as f32;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f32;
                let t0 = 1.0 - t1;
                let u1 = z - k0 as f32;
                let u0 = 1.0 - u1;

                d[idx] = s0
                    * (t0
                        * (u0 * d0[idx3d(i0, j0, k0, width, height)]
                            + u1 * d0[idx3d(i0, j0, k1, width, height)])
                        + t1 * (u0 * d0[idx3d(i0, j1, k0, width, height)]
                            + u1 * d0[idx3d(i0, j1, k1, width, height)]))
                    + s1 * (t0
                        * (u0 * d0[idx3d(i1, j0, k0, width, height)]
                            + u1 * d0[idx3d(i1, j0, k1, width, height)])
                        + t1 * (u0 * d0[idx3d(i1, j1, k0, width, height)]
                            + u1 * d0[idx3d(i1, j1, k1, width, height)]));
            }
        }
    }

    set_bnd_3d(b, d, width, height, depth);
}

fn set_bnd_3d(b: i32, x: &mut [f32], width: usize, height: usize, depth: usize) {
    // Face boundaries
    for k in 1..depth - 1 {
        for j in 1..height - 1 {
            x[idx3d(0, j, k, width, height)] = if b == 1 {
                -x[idx3d(1, j, k, width, height)]
            } else {
                x[idx3d(1, j, k, width, height)]
            };
            x[idx3d(width - 1, j, k, width, height)] = if b == 1 {
                -x[idx3d(width - 2, j, k, width, height)]
            } else {
                x[idx3d(width - 2, j, k, width, height)]
            };
        }
    }

    for k in 1..depth - 1 {
        for i in 1..width - 1 {
            x[idx3d(i, 0, k, width, height)] = if b == 2 {
                -x[idx3d(i, 1, k, width, height)]
            } else {
                x[idx3d(i, 1, k, width, height)]
            };
            x[idx3d(i, height - 1, k, width, height)] = if b == 2 {
                -x[idx3d(i, height - 2, k, width, height)]
            } else {
                x[idx3d(i, height - 2, k, width, height)]
            };
        }
    }

    for j in 1..height - 1 {
        for i in 1..width - 1 {
            x[idx3d(i, j, 0, width, height)] = if b == 3 {
                -x[idx3d(i, j, 1, width, height)]
            } else {
                x[idx3d(i, j, 1, width, height)]
            };
            x[idx3d(i, j, depth - 1, width, height)] = if b == 3 {
                -x[idx3d(i, j, depth - 2, width, height)]
            } else {
                x[idx3d(i, j, depth - 2, width, height)]
            };
        }
    }
}

// ============================================================================
// SPH (Smoothed Particle Hydrodynamics) - 2D
// ============================================================================

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

// ============================================================================
// SPH (Smoothed Particle Hydrodynamics) - 3D
// ============================================================================

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

// ============================================================================
// Smoke/Gas Simulation - 2D
// ============================================================================

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
    pub fn add_velocity(&mut self, x: usize, y: usize, z: usize, vel: Vec3) {
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

/// Registers all fluid operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of fluid ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Fluid>("resin::Fluid");
    registry.register_type::<Sph>("resin::Sph");
    registry.register_type::<SphParams3D>("resin::SphParams3D");
    registry.register_type::<Smoke>("resin::Smoke");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_grid_2d_creation() {
        let fluid = FluidGrid2D::new(64, 64, FluidConfig::default());
        assert_eq!(fluid.size(), (64, 64));
    }

    #[test]
    fn test_fluid_grid_2d_add_density() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);
        assert_eq!(fluid.density(16, 16), 100.0);
    }

    #[test]
    fn test_fluid_grid_2d_add_velocity() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_velocity(16, 16, 10.0, 5.0);
        let vel = fluid.velocity(16, 16);
        assert_eq!(vel.x, 10.0);
        assert_eq!(vel.y, 5.0);
    }

    #[test]
    fn test_fluid_grid_2d_step() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);
        fluid.add_velocity(16, 16, 5.0, 0.0);

        // Step simulation
        fluid.step();

        // Density should have spread/advected
        // Just verify it runs without panic
    }

    #[test]
    fn test_fluid_grid_2d_sample() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);

        let sampled = fluid.sample_density(Vec2::new(16.0, 16.0));
        assert!(sampled > 0.0);
    }

    #[test]
    fn test_fluid_grid_3d_creation() {
        let fluid = FluidGrid3D::new(16, 16, 16, FluidConfig::default());
        assert_eq!(fluid.size(), (16, 16, 16));
    }

    #[test]
    fn test_fluid_grid_3d_step() {
        let mut fluid = FluidGrid3D::new(16, 16, 16, FluidConfig::default());
        fluid.add_density(8, 8, 8, 100.0);
        fluid.add_velocity(8, 8, 8, Vec3::new(1.0, 0.0, 0.0));
        fluid.step();
        // Just verify it runs
    }

    #[test]
    fn test_sph_2d_creation() {
        let sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        assert_eq!(sph.particles.len(), 0);
    }

    #[test]
    fn test_sph_2d_add_particle() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_particle(Vec2::new(50.0, 50.0), 1.0);
        assert_eq!(sph.particles.len(), 1);
    }

    #[test]
    fn test_sph_2d_add_block() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_block(Vec2::new(10.0, 10.0), Vec2::new(30.0, 30.0), 5.0, 1.0);
        assert!(sph.particles.len() > 0);
    }

    #[test]
    fn test_sph_2d_step() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_block(Vec2::new(20.0, 50.0), Vec2::new(40.0, 70.0), 8.0, 1.0);

        let initial_pos = sph.positions();
        sph.step();
        let final_pos = sph.positions();

        // Particles should have moved (gravity)
        assert!(initial_pos != final_pos);
    }

    #[test]
    fn test_sph_3d_creation() {
        let sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        assert_eq!(sph.particles.len(), 0);
    }

    #[test]
    fn test_sph_3d_add_block() {
        let mut sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        sph.add_block(
            Vec3::new(0.1, 0.5, 0.1),
            Vec3::new(0.3, 0.7, 0.3),
            0.05,
            0.001,
        );
        assert!(sph.particles.len() > 0);
    }

    #[test]
    fn test_sph_3d_step() {
        let mut sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        sph.add_block(
            Vec3::new(0.2, 0.5, 0.2),
            Vec3::new(0.4, 0.7, 0.4),
            0.05,
            0.001,
        );

        let initial_pos = sph.positions();
        sph.step();
        let final_pos = sph.positions();

        // Particles should have moved
        assert!(initial_pos != final_pos);
    }

    #[test]
    fn test_smoke_2d_creation() {
        let smoke = SmokeGrid2D::new(64, 64, SmokeConfig::default());
        assert_eq!(smoke.size(), (64, 64));
    }

    #[test]
    fn test_smoke_2d_add_smoke() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        smoke.add_smoke(16, 16, 100.0, 50.0);
        assert_eq!(smoke.density(16, 16), 100.0);
        assert_eq!(smoke.temperature(16, 16), 50.0);
    }

    #[test]
    fn test_smoke_2d_buoyancy() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        // Add hot smoke
        smoke.add_smoke(16, 5, 100.0, 100.0);

        // Step simulation
        for _ in 0..10 {
            smoke.step();
        }

        // Hot smoke should have risen (positive y velocity somewhere)
        let vel = smoke.velocity(16, 10);
        assert!(vel.y > 0.0 || smoke.density(16, 10) > 0.0);
    }

    #[test]
    fn test_smoke_2d_dissipation() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        smoke.add_smoke(16, 16, 100.0, 100.0);

        let initial_density = smoke.density(16, 16);
        smoke.step();
        let final_density = smoke.density(16, 16);

        // Density should have decreased due to dissipation
        assert!(final_density < initial_density);
    }

    #[test]
    fn test_smoke_3d_creation() {
        let smoke = SmokeGrid3D::new(16, 16, 16, SmokeConfig::default());
        assert_eq!(smoke.size(), (16, 16, 16));
    }

    #[test]
    fn test_smoke_3d_step() {
        let mut smoke = SmokeGrid3D::new(16, 16, 16, SmokeConfig::default());
        smoke.add_smoke(8, 4, 8, 100.0, 100.0);

        // Step simulation
        smoke.step();

        // Just verify it runs
    }
}
