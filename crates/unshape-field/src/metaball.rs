use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

/// A single metaball (blob) center for use in metaball fields.
#[derive(Debug, Clone, Copy)]
pub struct Metaball {
    /// Center position.
    pub center: Vec3,
    /// Radius of influence.
    pub radius: f32,
    /// Strength (default 1.0).
    pub strength: f32,
}

impl Metaball {
    /// Creates a new metaball at the given position with the given radius.
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            strength: 1.0,
        }
    }

    /// Creates a 2D metaball (z=0).
    pub fn new_2d(center: Vec2, radius: f32) -> Self {
        Self::new(center.extend(0.0), radius)
    }
}

/// 2D metaball field - computes the sum of influences at each point.
///
/// The field value is the sum of `strength * f(distance)` for each ball,
/// where f is the falloff function. Values above 1.0 are typically "inside"
/// the merged surface.
///
/// # Example
///
/// ```
/// use glam::Vec2;
/// use unshape_field::{Field, EvalContext, Metaball, Metaballs2D};
///
/// let balls = vec![
///     Metaball::new_2d(Vec2::new(0.0, 0.0), 1.0),
///     Metaball::new_2d(Vec2::new(1.5, 0.0), 1.0),
/// ];
///
/// let field = Metaballs2D::new(balls);
/// let ctx = EvalContext::new();
///
/// // Sample the field - values > 1.0 are "inside"
/// let value = field.sample(Vec2::new(0.75, 0.0), &ctx);
/// ```
#[derive(Debug, Clone)]
pub struct Metaballs2D {
    pub(crate) balls: Vec<Metaball>,
    pub(crate) threshold: f32,
}

impl Metaballs2D {
    /// Creates a new 2D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec2, f32> for Metaballs2D {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let ball_pos = Vec2::new(ball.center.x, ball.center.y);
            let dist_sq = (input - ball_pos).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                // Very close to center - return large value
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// 3D metaball field.
///
/// Similar to Metaballs2D but operates in 3D space.
#[derive(Debug, Clone)]
pub struct Metaballs3D {
    pub(crate) balls: Vec<Metaball>,
    pub(crate) threshold: f32,
}

impl Metaballs3D {
    /// Creates a new 3D metaball field.
    pub fn new(balls: Vec<Metaball>) -> Self {
        Self {
            balls,
            threshold: 1.0,
        }
    }

    /// Sets the threshold for the implicit surface (default 1.0).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Adds a metaball to the field.
    pub fn add_ball(&mut self, ball: Metaball) {
        self.balls.push(ball);
    }

    /// Returns the threshold value.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns a reference to the balls.
    pub fn balls(&self) -> &[Metaball] {
        &self.balls
    }
}

impl Field<Vec3, f32> for Metaballs3D {
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        let mut sum = 0.0;

        for ball in &self.balls {
            let dist_sq = (input - ball.center).length_squared();
            let radius_sq = ball.radius * ball.radius;

            if dist_sq < 0.0001 {
                sum += ball.strength * 100.0;
            } else {
                // Classic metaball falloff: r^2 / d^2
                sum += ball.strength * radius_sq / dist_sq;
            }
        }

        sum
    }
}

/// Converts a 2D metaball field to an SDF-like representation.
///
/// Returns negative values inside (where field > threshold),
/// positive values outside. This makes it compatible with SDF
/// operations like smooth union.
#[derive(Debug, Clone)]
pub struct MetaballSdf2D {
    pub(crate) field: Metaballs2D,
}

impl MetaballSdf2D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs2D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs2D::new(balls))
    }
}

impl Field<Vec2, f32> for MetaballSdf2D {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        // Convert to SDF-like: negative inside, positive outside
        // When value > threshold, we're inside, so return negative
        self.field.threshold - value
    }
}

/// Converts a 3D metaball field to an SDF-like representation.
#[derive(Debug, Clone)]
pub struct MetaballSdf3D {
    pub(crate) field: Metaballs3D,
}

impl MetaballSdf3D {
    /// Creates a new SDF from a metaball field.
    pub fn new(field: Metaballs3D) -> Self {
        Self { field }
    }

    /// Creates an SDF from balls directly.
    pub fn from_balls(balls: Vec<Metaball>) -> Self {
        Self::new(Metaballs3D::new(balls))
    }
}

impl Field<Vec3, f32> for MetaballSdf3D {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let value = self.field.sample(input, ctx);
        self.field.threshold - value
    }
}
