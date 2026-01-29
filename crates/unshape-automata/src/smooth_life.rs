#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::SimpleRng;

/// SmoothLife configuration parameters.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLifeConfig {
    /// Inner radius for the "inner disk" (self + close neighbors).
    pub inner_radius: f32,
    /// Outer radius for the "outer ring" (neighbor annulus).
    pub outer_radius: f32,
    /// Birth threshold low (birth if filling in range [b1, b2]).
    pub birth_lo: f32,
    /// Birth threshold high.
    pub birth_hi: f32,
    /// Death threshold low (survive if filling in range [d1, d2]).
    pub death_lo: f32,
    /// Death threshold high.
    pub death_hi: f32,
    /// Sigmoid steepness (higher = sharper transition).
    pub alpha_n: f32,
    /// Sigmoid steepness for state mixing.
    pub alpha_m: f32,
}

impl Default for SmoothLifeConfig {
    fn default() -> Self {
        Self {
            inner_radius: 3.0,
            outer_radius: 9.0,
            birth_lo: 0.278,
            birth_hi: 0.365,
            death_lo: 0.267,
            death_hi: 0.445,
            alpha_n: 0.028,
            alpha_m: 0.147,
        }
    }
}

impl SmoothLifeConfig {
    /// Creates a SmoothLife config optimized for the "standard" smooth Life look.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Creates a config for more fluid-like behavior.
    pub fn fluid() -> Self {
        Self {
            inner_radius: 4.0,
            outer_radius: 12.0,
            birth_lo: 0.257,
            birth_hi: 0.336,
            death_lo: 0.365,
            death_hi: 0.550,
            alpha_n: 0.028,
            alpha_m: 0.147,
        }
    }

    /// Creates a config for slower, more stable patterns.
    pub fn slow() -> Self {
        Self {
            inner_radius: 5.0,
            outer_radius: 15.0,
            birth_lo: 0.269,
            birth_hi: 0.340,
            death_lo: 0.262,
            death_hi: 0.428,
            alpha_n: 0.020,
            alpha_m: 0.100,
        }
    }
}

/// SmoothLife - continuous-state cellular automaton.
///
/// A continuous generalization of Conway's Game of Life:
/// - Cell states are continuous (0.0 to 1.0) instead of binary
/// - Neighbor counting uses smooth disk/ring integrals
/// - State transitions use smooth sigmoid functions
///
/// This produces organic, fluid-like patterns that evolve smoothly.
///
/// # Example
///
/// ```
/// use unshape_automata::{SmoothLife, SmoothLifeConfig};
///
/// let mut sl = SmoothLife::new(100, 100, SmoothLifeConfig::default());
/// sl.randomize(12345, 0.3);
/// sl.step(0.1);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLife {
    /// Current cell states (0.0 to 1.0).
    cells: Vec<Vec<f32>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Configuration parameters.
    config: SmoothLifeConfig,
    /// Precomputed inner disk weights.
    inner_weights: Vec<Vec<f32>>,
    /// Precomputed outer ring weights.
    outer_weights: Vec<Vec<f32>>,
    /// Inner disk total weight (for normalization).
    inner_total: f32,
    /// Outer ring total weight (for normalization).
    outer_total: f32,
    /// Kernel radius (in cells).
    kernel_radius: i32,
}

impl SmoothLife {
    /// Creates a new SmoothLife simulation.
    pub fn new(width: usize, height: usize, config: SmoothLifeConfig) -> Self {
        let kernel_radius = config.outer_radius.ceil() as i32 + 1;
        let (inner_weights, outer_weights, inner_total, outer_total) =
            Self::compute_weights(&config, kernel_radius);

        Self {
            cells: vec![vec![0.0; width]; height],
            width,
            height,
            config,
            inner_weights,
            outer_weights,
            inner_total,
            outer_total,
            kernel_radius,
        }
    }

    /// Computes disk/ring weights for neighbor sampling.
    fn compute_weights(
        config: &SmoothLifeConfig,
        kernel_radius: i32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32, f32) {
        let size = (kernel_radius * 2 + 1) as usize;
        let mut inner = vec![vec![0.0f32; size]; size];
        let mut outer = vec![vec![0.0f32; size]; size];
        let mut inner_total = 0.0f32;
        let mut outer_total = 0.0f32;

        let ri = config.inner_radius;
        let ro = config.outer_radius;

        for dy in -kernel_radius..=kernel_radius {
            for dx in -kernel_radius..=kernel_radius {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let ux = (dx + kernel_radius) as usize;
                let uy = (dy + kernel_radius) as usize;

                // Inner disk (excluding center for some variants, but we include it)
                if dist <= ri {
                    // Smooth falloff at edge
                    let w = Self::smooth_step(ri - dist, 0.0, 1.0);
                    inner[uy][ux] = w;
                    inner_total += w;
                }

                // Outer ring (annulus between ri and ro)
                if dist > ri && dist <= ro {
                    let w = Self::smooth_step(ro - dist, 0.0, 1.0);
                    outer[uy][ux] = w;
                    outer_total += w;
                }
            }
        }

        (inner, outer, inner_total.max(1.0), outer_total.max(1.0))
    }

    /// Smooth step function (smoothstep).
    fn smooth_step(x: f32, edge0: f32, edge1: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Sigmoid function for smooth transitions.
    fn sigmoid(x: f32, a: f32, alpha: f32) -> f32 {
        1.0 / (1.0 + ((a - x) / alpha).exp())
    }

    /// Smooth interval membership (1 if x in [a,b], 0 otherwise, with smooth edges).
    fn sigmoid_interval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
        Self::sigmoid(x, a, alpha) * (1.0 - Self::sigmoid(x, b, alpha))
    }

    /// Transition function: computes new state from inner filling (m) and outer filling (n).
    fn transition(&self, n: f32, m: f32) -> f32 {
        let c = &self.config;

        // Birth: outer ring filling in [birth_lo, birth_hi]
        let birth = Self::sigmoid_interval(n, c.birth_lo, c.birth_hi, c.alpha_n);

        // Death threshold varies based on current state
        let death_lo = c.death_lo + (c.birth_lo - c.death_lo) * m;
        let death_hi = c.death_hi + (c.birth_hi - c.death_hi) * m;

        // Survival: outer ring filling in [death_lo, death_hi]
        let survival = Self::sigmoid_interval(n, death_lo, death_hi, c.alpha_n);

        // Mix based on current state
        let alive = Self::sigmoid(m, 0.5, c.alpha_m);
        birth * (1.0 - alive) + survival * alive
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(0.0)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if y < self.height && x < self.width {
            self.cells[y][x] = value.clamp(0.0, 1.0);
        }
    }

    /// Clears all cells to 0.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(0.0);
        }
    }

    /// Randomizes cells with given density (probability of being close to 1.0).
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for row in &mut self.cells {
            for cell in row {
                *cell = if rng.next_f32() < density {
                    0.5 + rng.next_f32() * 0.5
                } else {
                    rng.next_f32() * 0.2
                };
            }
        }
    }

    /// Computes inner disk (m) and outer ring (n) filling for a cell.
    fn compute_filling(&self, x: usize, y: usize) -> (f32, f32) {
        let mut inner_sum = 0.0f32;
        let mut outer_sum = 0.0f32;

        let kr = self.kernel_radius;

        for dy in -kr..=kr {
            for dx in -kr..=kr {
                let nx = ((x as i32 + dx).rem_euclid(self.width as i32)) as usize;
                let ny = ((y as i32 + dy).rem_euclid(self.height as i32)) as usize;

                let wx = (dx + kr) as usize;
                let wy = (dy + kr) as usize;

                let cell_value = self.cells[ny][nx];

                inner_sum += self.inner_weights[wy][wx] * cell_value;
                outer_sum += self.outer_weights[wy][wx] * cell_value;
            }
        }

        let m = inner_sum / self.inner_total;
        let n = outer_sum / self.outer_total;

        (n, m)
    }

    /// Advances the simulation by one step.
    ///
    /// The `dt` parameter controls the rate of change (0.0 to 1.0).
    /// Use ~0.1 for smooth animation, 1.0 for discrete steps.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, dt: f32) {
        let mut next = vec![vec![0.0f32; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let (n, m) = self.compute_filling(x, y);
                let target = self.transition(n, m);
                let current = self.cells[y][x];

                // Smooth interpolation toward target
                next[y][x] = current + (target - current) * dt;
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize, dt: f32) {
        for _ in 0..n {
            self.step(dt);
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<f32>> {
        &self.cells
    }

    /// Returns the average cell value.
    pub fn average_value(&self) -> f32 {
        let sum: f32 = self.cells.iter().flat_map(|row| row.iter()).sum();
        sum / (self.width * self.height) as f32
    }
}
