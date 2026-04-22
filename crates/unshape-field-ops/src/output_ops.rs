//! Output-space field combinators: Remap, Clamp, Smoothstep, Step, Abs, Negate, Pow.
//!
//! These combinators transform the output of an existing field, operating on `f32` values.

use crate::{EvalContext, Field};

// ============================================================================
// Remap
// ============================================================================

/// Remaps the output of a field from `[in_min, in_max]` to `[out_min, out_max]`,
/// clamping to the output range.
#[derive(Debug, Clone)]
pub struct Remap<F> {
    /// The inner field to remap.
    pub field: F,
    /// The minimum of the input range.
    pub in_min: f32,
    /// The maximum of the input range.
    pub in_max: f32,
    /// The minimum of the output range.
    pub out_min: f32,
    /// The maximum of the output range.
    pub out_max: f32,
}

impl<I, F> Field<I, f32> for Remap<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let v = self.field.sample(input, ctx);
        let t = if (self.in_max - self.in_min).abs() < f32::EPSILON {
            0.0
        } else {
            (v - self.in_min) / (self.in_max - self.in_min)
        };
        let result = self.out_min + t * (self.out_max - self.out_min);
        result.clamp(
            self.out_min.min(self.out_max),
            self.out_min.max(self.out_max),
        )
    }
}

// ============================================================================
// Clamp
// ============================================================================

/// Clamps the output of a field to `[min, max]`.
#[derive(Debug, Clone)]
pub struct Clamp<F> {
    /// The inner field to clamp.
    pub field: F,
    /// The minimum value.
    pub min: f32,
    /// The maximum value.
    pub max: f32,
}

impl<I, F> Field<I, f32> for Clamp<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).clamp(self.min, self.max)
    }
}

// ============================================================================
// Smoothstep
// ============================================================================

/// Applies a smoothstep mapping from `[edge0, edge1]` to `[0, 1]`.
///
/// Equivalent to GLSL `smoothstep(edge0, edge1, x)`.
#[derive(Debug, Clone)]
pub struct Smoothstep<F> {
    /// The inner field.
    pub field: F,
    /// The lower edge of the smoothstep range.
    pub edge0: f32,
    /// The upper edge of the smoothstep range.
    pub edge1: f32,
}

impl<I, F> Field<I, f32> for Smoothstep<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let v = self.field.sample(input, ctx);
        let t = ((v - self.edge0) / (self.edge1 - self.edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

// ============================================================================
// Step
// ============================================================================

/// Returns 0.0 if the field output is less than `threshold`, else 1.0.
///
/// Equivalent to GLSL `step(threshold, x)`.
#[derive(Debug, Clone)]
pub struct Step<F> {
    /// The inner field.
    pub field: F,
    /// The threshold value.
    pub threshold: f32,
}

impl<I, F> Field<I, f32> for Step<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        if self.field.sample(input, ctx) < self.threshold {
            0.0
        } else {
            1.0
        }
    }
}

// ============================================================================
// Abs
// ============================================================================

/// Returns the absolute value of the field output.
#[derive(Debug, Clone)]
pub struct Abs<F> {
    /// The inner field.
    pub field: F,
}

impl<I, F> Field<I, f32> for Abs<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).abs()
    }
}

// ============================================================================
// Negate
// ============================================================================

/// Negates the field output.
#[derive(Debug, Clone)]
pub struct Negate<F> {
    /// The inner field.
    pub field: F,
}

impl<I, F> Field<I, f32> for Negate<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        -self.field.sample(input, ctx)
    }
}

// ============================================================================
// Pow
// ============================================================================

/// Raises the field output to a power.
#[derive(Debug, Clone)]
pub struct Pow<F> {
    /// The inner field.
    pub field: F,
    /// The exponent.
    pub exponent: f32,
}

impl<I, F> Field<I, f32> for Pow<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).powf(self.exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal `Field<Vec3, f32>` to use in tests without depending on external crates.
    struct SphereSdfLocal {
        radius: f32,
    }

    impl Field<glam::Vec3, f32> for SphereSdfLocal {
        fn sample(&self, input: glam::Vec3, _ctx: &EvalContext) -> f32 {
            input.length() - self.radius
        }
    }

    #[test]
    fn test_combinators_with_vec3_input() {
        use glam::Vec3;

        // sphere SDF centered at origin with radius 1: Field<Vec3, f32>
        let field = SphereSdfLocal { radius: 1.0 };
        let ctx = EvalContext::new();

        // chain: sphere SDF → negate → clamp(-2,2) → remap(-2,2, 0,1)
        let result = Remap {
            field: Clamp {
                field: Negate { field },
                min: -2.0,
                max: 2.0,
            },
            in_min: -2.0,
            in_max: 2.0,
            out_min: 0.0,
            out_max: 1.0,
        }
        .sample(Vec3::new(0.5, 0.0, 0.0), &ctx);

        assert!(
            (0.0..=1.0).contains(&result),
            "expected result in [0,1], got {result}"
        );
    }
}
