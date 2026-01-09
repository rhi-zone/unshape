//! Reaction-diffusion simulation for procedural pattern generation.
//!
//! Implements the Gray-Scott model for creating organic patterns like
//! animal markings, coral, fingerprints, and abstract textures.
//!
//! # Example
//!
//! ```
//! use resin_rd::{ReactionDiffusion, GrayScottPreset};
//!
//! // Create a 100x100 simulation with coral preset
//! let mut rd = ReactionDiffusion::new(100, 100);
//! rd.set_preset(GrayScottPreset::Coral);
//!
//! // Add some initial seed pattern
//! rd.add_seed_circle(50, 50, 5);
//!
//! // Simulate for many steps
//! for _ in 0..1000 {
//!     rd.step();
//! }
//!
//! // Get the pattern as a normalized grid
//! let pattern = rd.get_v_normalized();
//! ```

/// Gray-Scott reaction-diffusion simulation.
#[derive(Debug, Clone)]
pub struct ReactionDiffusion {
    /// Width of the simulation grid.
    width: usize,
    /// Height of the simulation grid.
    height: usize,
    /// Chemical U concentration.
    u: Vec<f32>,
    /// Chemical V concentration.
    v: Vec<f32>,
    /// Scratch buffer for U.
    u_next: Vec<f32>,
    /// Scratch buffer for V.
    v_next: Vec<f32>,
    /// Diffusion rate of U.
    pub du: f32,
    /// Diffusion rate of V.
    pub dv: f32,
    /// Feed rate.
    pub feed: f32,
    /// Kill rate.
    pub kill: f32,
    /// Time step.
    pub dt: f32,
}

/// Preset parameters for common Gray-Scott patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrayScottPreset {
    /// Mitosis-like dividing cells.
    Mitosis,
    /// Coral-like branching patterns.
    Coral,
    /// Maze-like patterns.
    Maze,
    /// Soliton spots.
    Solitons,
    /// Worm-like patterns.
    Worms,
    /// Fingerprint-like patterns.
    Fingerprint,
    /// Spots pattern.
    Spots,
    /// Unstable chaotic pattern.
    Chaos,
    /// Moving spots.
    MovingSpots,
}

impl GrayScottPreset {
    /// Returns (feed, kill) parameters for this preset.
    pub fn parameters(&self) -> (f32, f32) {
        match self {
            GrayScottPreset::Mitosis => (0.028, 0.062),
            GrayScottPreset::Coral => (0.037, 0.060),
            GrayScottPreset::Maze => (0.029, 0.057),
            GrayScottPreset::Solitons => (0.030, 0.062),
            GrayScottPreset::Worms => (0.078, 0.061),
            GrayScottPreset::Fingerprint => (0.037, 0.060),
            GrayScottPreset::Spots => (0.035, 0.065),
            GrayScottPreset::Chaos => (0.026, 0.051),
            GrayScottPreset::MovingSpots => (0.014, 0.054),
        }
    }
}

impl ReactionDiffusion {
    /// Creates a new reaction-diffusion simulation.
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;

        // Initialize U to 1.0 everywhere, V to 0.0
        let u = vec![1.0; size];
        let v = vec![0.0; size];

        Self {
            width,
            height,
            u,
            v,
            u_next: vec![0.0; size],
            v_next: vec![0.0; size],
            du: 0.16,
            dv: 0.08,
            feed: 0.037,
            kill: 0.060,
            dt: 1.0,
        }
    }

    /// Sets parameters from a preset.
    pub fn set_preset(&mut self, preset: GrayScottPreset) {
        let (feed, kill) = preset.parameters();
        self.feed = feed;
        self.kill = kill;
    }

    /// Sets the feed rate.
    pub fn set_feed(&mut self, feed: f32) {
        self.feed = feed;
    }

    /// Sets the kill rate.
    pub fn set_kill(&mut self, kill: f32) {
        self.kill = kill;
    }

    /// Returns the width of the simulation grid.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the simulation grid.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the U value at a position.
    pub fn get_u(&self, x: usize, y: usize) -> f32 {
        self.u[y * self.width + x]
    }

    /// Gets the V value at a position.
    pub fn get_v(&self, x: usize, y: usize) -> f32 {
        self.v[y * self.width + x]
    }

    /// Sets the U value at a position.
    pub fn set_u(&mut self, x: usize, y: usize, value: f32) {
        self.u[y * self.width + x] = value;
    }

    /// Sets the V value at a position.
    pub fn set_v(&mut self, x: usize, y: usize, value: f32) {
        self.v[y * self.width + x] = value;
    }

    /// Adds a circular seed of chemical V.
    pub fn add_seed_circle(&mut self, cx: usize, cy: usize, radius: usize) {
        let r2 = (radius * radius) as i32;

        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                if dx * dx + dy * dy <= r2 {
                    self.set_u(x, y, 0.5);
                    self.set_v(x, y, 0.25);
                }
            }
        }
    }

    /// Adds a rectangular seed of chemical V.
    pub fn add_seed_rect(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        for y in y1..=y2.min(self.height - 1) {
            for x in x1..=x2.min(self.width - 1) {
                self.set_u(x, y, 0.5);
                self.set_v(x, y, 0.25);
            }
        }
    }

    /// Adds random seeds across the grid.
    pub fn add_random_seeds(&mut self, count: usize, radius: usize, seed: u64) {
        let mut rng = seed;

        for _ in 0..count {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (rng as usize) % self.width;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (rng as usize) % self.height;

            self.add_seed_circle(x, y, radius);
        }
    }

    /// Clears the simulation (U=1, V=0).
    pub fn clear(&mut self) {
        self.u.fill(1.0);
        self.v.fill(0.0);
    }

    /// Computes the Laplacian of a field at a position using 5-point stencil.
    fn laplacian(&self, field: &[f32], x: usize, y: usize) -> f32 {
        let w = self.width;
        let h = self.height;

        // Wrap around at edges
        let xm = if x == 0 { w - 1 } else { x - 1 };
        let xp = if x == w - 1 { 0 } else { x + 1 };
        let ym = if y == 0 { h - 1 } else { y - 1 };
        let yp = if y == h - 1 { 0 } else { y + 1 };

        let center = field[y * w + x];
        let left = field[y * w + xm];
        let right = field[y * w + xp];
        let up = field[ym * w + x];
        let down = field[yp * w + x];

        left + right + up + down - 4.0 * center
    }

    /// Advances the simulation by one time step.
    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let u = self.u[idx];
                let v = self.v[idx];

                let lap_u = self.laplacian(&self.u, x, y);
                let lap_v = self.laplacian(&self.v, x, y);

                // Reaction term
                let uvv = u * v * v;

                // Gray-Scott equations
                let du_dt = self.du * lap_u - uvv + self.feed * (1.0 - u);
                let dv_dt = self.dv * lap_v + uvv - (self.kill + self.feed) * v;

                self.u_next[idx] = (u + self.dt * du_dt).clamp(0.0, 1.0);
                self.v_next[idx] = (v + self.dt * dv_dt).clamp(0.0, 1.0);
            }
        }

        // Swap buffers
        std::mem::swap(&mut self.u, &mut self.u_next);
        std::mem::swap(&mut self.v, &mut self.v_next);
    }

    /// Advances the simulation by multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the U buffer.
    pub fn u_buffer(&self) -> &[f32] {
        &self.u
    }

    /// Returns a reference to the V buffer.
    pub fn v_buffer(&self) -> &[f32] {
        &self.v
    }

    /// Returns the V buffer normalized to [0, 1].
    pub fn get_v_normalized(&self) -> Vec<f32> {
        let min = self.v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range < 1e-10 {
            return vec![0.5; self.v.len()];
        }

        self.v.iter().map(|&v| (v - min) / range).collect()
    }

    /// Returns the U buffer normalized to [0, 1].
    pub fn get_u_normalized(&self) -> Vec<f32> {
        let min = self.u.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.u.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range < 1e-10 {
            return vec![0.5; self.u.len()];
        }

        self.u.iter().map(|&u| (u - min) / range).collect()
    }

    /// Returns the difference (U - V) as a normalized grid.
    pub fn get_difference_normalized(&self) -> Vec<f32> {
        let diff: Vec<f32> = self.u.iter().zip(&self.v).map(|(u, v)| u - v).collect();

        let min = diff.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = diff.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range < 1e-10 {
            return vec![0.5; diff.len()];
        }

        diff.iter().map(|&d| (d - min) / range).collect()
    }
}

/// Multi-channel reaction-diffusion with more complex chemistry.
#[derive(Debug, Clone)]
pub struct MultiChannelRD {
    /// Width of the simulation grid.
    width: usize,
    /// Height of the simulation grid.
    height: usize,
    /// Chemical concentrations (multiple channels).
    channels: Vec<Vec<f32>>,
    /// Diffusion rates for each channel.
    diffusion: Vec<f32>,
    // Note: Reaction function could be added with boxed closure if needed
}

impl MultiChannelRD {
    /// Creates a new multi-channel simulation.
    pub fn new(width: usize, height: usize, num_channels: usize) -> Self {
        let size = width * height;

        Self {
            width,
            height,
            channels: vec![vec![0.0; size]; num_channels],
            diffusion: vec![0.1; num_channels],
        }
    }

    /// Sets the diffusion rate for a channel.
    pub fn set_diffusion(&mut self, channel: usize, rate: f32) {
        if channel < self.diffusion.len() {
            self.diffusion[channel] = rate;
        }
    }

    /// Gets a value from a channel.
    pub fn get(&self, channel: usize, x: usize, y: usize) -> f32 {
        self.channels[channel][y * self.width + x]
    }

    /// Sets a value in a channel.
    pub fn set(&mut self, channel: usize, x: usize, y: usize, value: f32) {
        self.channels[channel][y * self.width + x] = value;
    }

    /// Returns a reference to a channel buffer.
    pub fn channel(&self, channel: usize) -> &[f32] {
        &self.channels[channel]
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let rd = ReactionDiffusion::new(100, 100);
        assert_eq!(rd.width(), 100);
        assert_eq!(rd.height(), 100);
    }

    #[test]
    fn test_preset() {
        let mut rd = ReactionDiffusion::new(50, 50);
        rd.set_preset(GrayScottPreset::Coral);
        assert_eq!(rd.feed, 0.037);
        assert_eq!(rd.kill, 0.060);
    }

    #[test]
    fn test_seed_circle() {
        let mut rd = ReactionDiffusion::new(50, 50);
        rd.add_seed_circle(25, 25, 5);

        // Center should have V > 0
        assert!(rd.get_v(25, 25) > 0.0);
        // Corner should still be 0
        assert_eq!(rd.get_v(0, 0), 0.0);
    }

    #[test]
    fn test_seed_rect() {
        let mut rd = ReactionDiffusion::new(50, 50);
        rd.add_seed_rect(10, 10, 20, 20);

        // Inside should have V > 0
        assert!(rd.get_v(15, 15) > 0.0);
        // Outside should still be 0
        assert_eq!(rd.get_v(5, 5), 0.0);
    }

    #[test]
    fn test_step() {
        let mut rd = ReactionDiffusion::new(50, 50);
        rd.add_seed_circle(25, 25, 5);

        let initial_v = rd.get_v(25, 25);
        rd.step();
        let after_v = rd.get_v(25, 25);

        // Values should change after stepping
        assert_ne!(initial_v, after_v);
    }

    #[test]
    fn test_multiple_steps() {
        let mut rd = ReactionDiffusion::new(30, 30);
        rd.add_seed_circle(15, 15, 3);
        rd.steps(100);

        // After many steps, pattern should have evolved
        let v_sum: f32 = rd.v_buffer().iter().sum();
        assert!(v_sum > 0.0);
    }

    #[test]
    fn test_normalization() {
        let mut rd = ReactionDiffusion::new(20, 20);
        rd.add_seed_circle(10, 10, 3);
        rd.steps(50);

        let normalized = rd.get_v_normalized();

        // All values should be in [0, 1]
        for v in normalized {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_clear() {
        let mut rd = ReactionDiffusion::new(20, 20);
        rd.add_seed_circle(10, 10, 3);
        rd.clear();

        // All V should be 0
        for v in rd.v_buffer() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_random_seeds() {
        let mut rd = ReactionDiffusion::new(50, 50);
        rd.add_random_seeds(5, 3, 12345);

        // Should have some V > 0 somewhere
        let v_sum: f32 = rd.v_buffer().iter().sum();
        assert!(v_sum > 0.0);
    }

    #[test]
    fn test_preset_parameters() {
        assert_eq!(GrayScottPreset::Mitosis.parameters(), (0.028, 0.062));
        assert_eq!(GrayScottPreset::Coral.parameters(), (0.037, 0.060));
        assert_eq!(GrayScottPreset::Spots.parameters(), (0.035, 0.065));
    }

    #[test]
    fn test_multi_channel() {
        let mut mc = MultiChannelRD::new(20, 20, 3);
        assert_eq!(mc.num_channels(), 3);

        mc.set(0, 10, 10, 0.5);
        assert_eq!(mc.get(0, 10, 10), 0.5);
    }
}
