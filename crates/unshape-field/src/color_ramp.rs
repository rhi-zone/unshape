use crate::{EvalContext, Field};

/// A single color stop in a [`ColorRamp`] gradient.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorStop {
    /// Position along the gradient, in the range [0.0, 1.0].
    pub position: f32,
    /// RGBA color at this stop.
    pub color: [f32; 4],
}

impl ColorStop {
    /// Creates a new color stop.
    pub fn new(position: f32, color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

/// Interpolation mode for a [`ColorRamp`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorRampInterp {
    /// Linearly interpolate between adjacent stops.
    #[default]
    Linear,
    /// Step to the left stop's color with no interpolation.
    Constant,
    /// Smoothstep (cubic ease) between adjacent stops.
    Ease,
}

/// Maps a scalar field output to a color via a gradient (gradient map).
///
/// This is equivalent to Blender's "Color Ramp" node or Photoshop's "Gradient Map".
/// The scalar input is clamped to [0, 1], then mapped through the sorted color stops.
///
/// # Example
///
/// ```
/// use unshape_field::{ColorRamp, ColorStop, ColorRampInterp, Constant, EvalContext, Field};
/// use glam::Vec2;
///
/// let ramp = ColorRamp::new(
///     Constant::new(0.5f32),
///     vec![
///         ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
///         ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
///     ],
/// );
///
/// let ctx = EvalContext::new();
/// let color = ramp.sample(Vec2::ZERO, &ctx);
/// assert!((color[0] - 0.5).abs() < 1e-5);
/// ```
pub struct ColorRamp<F> {
    /// The scalar field to sample.
    pub field: F,
    /// Color stops, sorted by position.
    pub stops: Vec<ColorStop>,
    /// Interpolation mode between stops.
    pub interpolation: ColorRampInterp,
}

impl<F> ColorRamp<F> {
    /// Creates a new `ColorRamp` with the given scalar field and stops.
    ///
    /// Stops are automatically sorted by position.
    pub fn new(field: F, mut stops: Vec<ColorStop>) -> Self {
        stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            field,
            stops,
            interpolation: ColorRampInterp::default(),
        }
    }

    /// Creates a `ColorRamp` that evenly distributes N colors across [0, 1].
    ///
    /// - 1 color → stop at 0.0
    /// - 2 colors → stops at 0.0, 1.0
    /// - N colors → evenly spaced from 0.0 to 1.0
    pub fn gradient(field: F, colors: Vec<[f32; 4]>) -> Self {
        let stops = match colors.len() {
            0 => vec![ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0])],
            1 => vec![ColorStop::new(0.0, colors[0])],
            n => colors
                .into_iter()
                .enumerate()
                .map(|(i, color)| ColorStop::new(i as f32 / (n - 1) as f32, color))
                .collect(),
        };
        Self {
            field,
            stops,
            interpolation: ColorRampInterp::default(),
        }
    }

    /// Sets the interpolation mode.
    pub fn with_interpolation(mut self, interpolation: ColorRampInterp) -> Self {
        self.interpolation = interpolation;
        self
    }

    /// Evaluates the gradient at position `t` (already clamped and scaled).
    fn eval_gradient(&self, t: f32) -> [f32; 4] {
        if self.stops.is_empty() {
            return [0.0, 0.0, 0.0, 1.0];
        }

        let t = t.clamp(0.0, 1.0);

        // Return first/last stop if outside their range.
        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        if t >= self.stops[self.stops.len() - 1].position {
            return self.stops[self.stops.len() - 1].color;
        }

        // Find the two surrounding stops.
        let right_idx = self.stops.partition_point(|s| s.pos_le(t));
        let right_idx = right_idx.min(self.stops.len() - 1).max(1);
        let left = &self.stops[right_idx - 1];
        let right = &self.stops[right_idx];

        let span = right.position - left.position;
        let local_t = if span > 0.0 {
            (t - left.position) / span
        } else {
            0.0
        };

        let weight = match self.interpolation {
            ColorRampInterp::Linear => local_t,
            ColorRampInterp::Constant => 0.0,
            ColorRampInterp::Ease => {
                let s = local_t;
                s * s * (3.0 - 2.0 * s)
            }
        };

        lerp_color(left.color, right.color, weight)
    }
}

impl ColorStop {
    /// Returns `true` if this stop's position is ≤ `t` (used for binary search).
    fn pos_le(&self, t: f32) -> bool {
        self.position <= t
    }
}

fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

impl<I, F> Field<I, [f32; 4]> for ColorRamp<F>
where
    F: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> [f32; 4] {
        let t = self.field.sample(input, ctx);
        self.eval_gradient(t)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use crate::{ColorRamp, ColorRampInterp, ColorStop, Constant, EvalContext, Field};

    #[test]
    fn two_stop_black_white_midpoint() {
        let ramp = ColorRamp::new(
            Constant::new(0.5f32),
            vec![
                ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
        );
        let ctx = EvalContext::new();
        let color = ramp.sample(Vec2::ZERO, &ctx);
        assert!(
            (color[0] - 0.5).abs() < 1e-5,
            "expected ~0.5, got {}",
            color[0]
        );
        assert!(
            (color[1] - 0.5).abs() < 1e-5,
            "expected ~0.5, got {}",
            color[1]
        );
    }

    #[test]
    fn gradient_distributes_three_colors() {
        let ramp = ColorRamp::gradient(
            Constant::new(0.0f32),
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        );
        assert_eq!(ramp.stops.len(), 3);
        assert!((ramp.stops[0].position - 0.0).abs() < 1e-5);
        assert!((ramp.stops[1].position - 0.5).abs() < 1e-5);
        assert!((ramp.stops[2].position - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clamps_below_first_stop() {
        let ramp = ColorRamp::new(
            Constant::new(-1.0f32),
            vec![
                ColorStop::new(0.0, [0.25, 0.25, 0.25, 1.0]),
                ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
        );
        let ctx = EvalContext::new();
        let color = ramp.sample(Vec2::ZERO, &ctx);
        assert!((color[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn constant_interpolation_stays_at_left() {
        let ramp = ColorRamp::new(
            Constant::new(0.75f32),
            vec![
                ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                ColorStop::new(0.5, [0.5, 0.5, 0.5, 1.0]),
                ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
        )
        .with_interpolation(ColorRampInterp::Constant);
        let ctx = EvalContext::new();
        let color = ramp.sample(Vec2::ZERO, &ctx);
        // t=0.75 is between stops at 0.5 and 1.0; constant → left stop [0.5, 0.5, 0.5, 1.0]
        assert!((color[0] - 0.5).abs() < 1e-5);
    }
}
