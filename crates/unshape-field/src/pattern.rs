use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

/// Checkerboard pattern.
#[derive(Debug, Clone, Copy)]
pub struct Checkerboard {
    /// Pattern scale.
    pub scale: f32,
}

impl Default for Checkerboard {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl Checkerboard {
    /// Create a new checkerboard pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific scale.
    pub fn with_scale(scale: f32) -> Self {
        Self { scale }
    }
}

impl Field<Vec2, f32> for Checkerboard {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        if (x + y) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

impl Field<Vec3, f32> for Checkerboard {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let x = p.x.floor() as i32;
        let y = p.y.floor() as i32;
        let z = p.z.floor() as i32;
        if (x + y + z) % 2 == 0 { 1.0 } else { 0.0 }
    }
}

/// Stripe pattern (horizontal by default).
#[derive(Debug, Clone, Copy)]
pub struct Stripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for Stripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl Stripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }

    /// Set the direction (will be normalized).
    pub fn with_direction(mut self, direction: Vec2) -> Self {
        self.direction = direction.normalize();
        self
    }
}

impl Field<Vec2, f32> for Stripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency;
        if (t.floor() as i32) % 2 == 0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Smooth stripe pattern (sine wave).
#[derive(Debug, Clone, Copy)]
pub struct SmoothStripes {
    /// Stripe frequency.
    pub frequency: f32,
    /// Stripe direction.
    pub direction: Vec2,
}

impl Default for SmoothStripes {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            direction: Vec2::Y,
        }
    }
}

impl SmoothStripes {
    /// Create horizontal stripes.
    pub fn horizontal() -> Self {
        Self {
            direction: Vec2::Y,
            ..Self::default()
        }
    }

    /// Create vertical stripes.
    pub fn vertical() -> Self {
        Self {
            direction: Vec2::X,
            ..Self::default()
        }
    }
}

impl Field<Vec2, f32> for SmoothStripes {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let t = input.dot(self.direction) * self.frequency * std::f32::consts::TAU;
        (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

/// Brick pattern.
#[derive(Debug, Clone, Copy)]
pub struct Brick {
    /// Brick size.
    pub scale: Vec2,
    /// Mortar width.
    pub mortar: f32,
    /// Row offset (0.5 for standard brick).
    pub offset: f32,
}

impl Default for Brick {
    fn default() -> Self {
        Self {
            scale: Vec2::new(2.0, 1.0),
            mortar: 0.05,
            offset: 0.5,
        }
    }
}

impl Brick {
    /// Create a new brick pattern.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for Brick {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let row = p.y.floor() as i32;
        let x = if row % 2 == 0 { p.x } else { p.x + self.offset };

        let fx = x.fract();
        let fy = p.y.fract();

        // Check if in mortar
        if fx < self.mortar || fx > 1.0 - self.mortar || fy < self.mortar || fy > 1.0 - self.mortar
        {
            0.0
        } else {
            1.0
        }
    }
}

/// Polka dots pattern.
#[derive(Debug, Clone, Copy)]
pub struct Dots {
    /// Grid scale.
    pub scale: f32,
    /// Dot radius.
    pub radius: f32,
}

impl Default for Dots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.3,
        }
    }
}

impl Dots {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for Dots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        if d < self.radius { 1.0 } else { 0.0 }
    }
}

/// Smooth dots (falloff from center).
#[derive(Debug, Clone, Copy)]
pub struct SmoothDots {
    pub scale: f32,
    pub radius: f32,
}

impl Default for SmoothDots {
    fn default() -> Self {
        Self {
            scale: 1.0,
            radius: 0.4,
        }
    }
}

impl SmoothDots {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for SmoothDots {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let center = cell + Vec2::splat(0.5);
        let d = (p - center).length();
        (1.0 - d / self.radius).clamp(0.0, 1.0)
    }
}

/// Voronoi/Cellular noise - returns distance to nearest cell center.
#[derive(Debug, Clone, Copy)]
pub struct Voronoi {
    pub scale: f32,
    pub seed: i32,
}

impl Default for Voronoi {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl Voronoi {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash function for cell randomization.
    pub fn hash(x: i32, y: i32, seed: i32) -> Vec2 {
        // Simple hash based on prime multipliers
        let n = (x.wrapping_mul(374761393))
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed.wrapping_mul(1013904223));
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        let fx = ((n & 0xFFFF) as f32) / 65535.0;
        let fy = (((n >> 16) & 0xFFFF) as f32) / 65535.0;
        Vec2::new(fx, fy)
    }
}

impl Field<Vec2, f32> for Voronoi {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;

        // Check 3x3 neighborhood
        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Self::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                min_dist = min_dist.min(dist);
            }
        }

        // Normalize roughly to 0-1 (max distance in a cell is ~sqrt(2))
        (min_dist / 1.5).clamp(0.0, 1.0)
    }
}

/// Voronoi that returns cell ID (for coloring).
#[derive(Debug, Clone, Copy)]
pub struct VoronoiId {
    pub scale: f32,
    pub seed: i32,
}

impl Default for VoronoiId {
    fn default() -> Self {
        Self {
            scale: 1.0,
            seed: 0,
        }
    }
}

impl VoronoiId {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Field<Vec2, f32> for VoronoiId {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input * self.scale;
        let cell = Vec2::new(p.x.floor(), p.y.floor());
        let mut min_dist = f32::MAX;
        let mut closest_cell = cell;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let neighbor = cell + Vec2::new(dx as f32, dy as f32);
                let offset = Voronoi::hash(neighbor.x as i32, neighbor.y as i32, self.seed);
                let point = neighbor + offset;
                let dist = (p - point).length();
                if dist < min_dist {
                    min_dist = dist;
                    closest_cell = neighbor;
                }
            }
        }

        // Return a pseudo-random value based on cell coordinates
        let h = Voronoi::hash(
            closest_cell.x as i32,
            closest_cell.y as i32,
            self.seed + 12345,
        );
        h.x
    }
}
