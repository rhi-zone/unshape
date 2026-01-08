//! Spline and curve types for interpolation.
//!
//! Provides various curve types for smooth interpolation:
//! - [`CubicBezier`] - Cubic Bezier curves
//! - [`CatmullRom`] - Catmull-Rom splines (pass through control points)
//! - [`BSpline`] - B-spline curves

use glam::{Vec2, Vec3};

/// Trait for types that can be interpolated along a curve.
pub trait Interpolatable:
    Clone
    + Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<f32, Output = Self>
{
}

impl Interpolatable for f32 {}
impl Interpolatable for Vec2 {}
impl Interpolatable for Vec3 {}

/// A cubic Bezier curve segment.
///
/// Defined by 4 control points: start (P0), control 1 (P1), control 2 (P2), end (P3).
/// The curve passes through P0 and P3, and is influenced by P1 and P2.
#[derive(Debug, Clone, Copy)]
pub struct CubicBezier<T: Interpolatable> {
    /// Start point.
    pub p0: T,
    /// First control point.
    pub p1: T,
    /// Second control point.
    pub p2: T,
    /// End point.
    pub p3: T,
}

impl<T: Interpolatable> CubicBezier<T> {
    /// Creates a new cubic Bezier curve.
    pub fn new(p0: T, p1: T, p2: T, p3: T) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluates the curve at parameter t (0 to 1).
    pub fn evaluate(&self, t: f32) -> T {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        // B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        self.p0 * mt3 + self.p1 * (3.0 * mt2 * t) + self.p2 * (3.0 * mt * t2) + self.p3 * t3
    }

    /// Evaluates the derivative (tangent) at parameter t.
    pub fn derivative(&self, t: f32) -> T {
        let t2 = t * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;

        // B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
        (self.p1 - self.p0) * (3.0 * mt2)
            + (self.p2 - self.p1) * (6.0 * mt * t)
            + (self.p3 - self.p2) * (3.0 * t2)
    }

    /// Splits the curve at parameter t into two curves.
    pub fn split(&self, t: f32) -> (Self, Self) {
        // De Casteljau's algorithm
        let p01 = lerp(self.p0, self.p1, t);
        let p12 = lerp(self.p1, self.p2, t);
        let p23 = lerp(self.p2, self.p3, t);

        let p012 = lerp(p01, p12, t);
        let p123 = lerp(p12, p23, t);

        let p0123 = lerp(p012, p123, t);

        (
            Self::new(self.p0, p01, p012, p0123),
            Self::new(p0123, p123, p23, self.p3),
        )
    }
}

/// A Catmull-Rom spline that passes through all control points.
///
/// Uses centripetal parameterization for better curve behavior.
#[derive(Debug, Clone)]
pub struct CatmullRom<T: Interpolatable> {
    /// Control points (the curve passes through all of them).
    pub points: Vec<T>,
    /// Tension parameter (0.0 = Catmull-Rom, 0.5 = centripetal, 1.0 = chordal).
    pub alpha: f32,
}

impl<T: Interpolatable> CatmullRom<T> {
    /// Creates a new Catmull-Rom spline.
    pub fn new(points: Vec<T>) -> Self {
        Self {
            points,
            alpha: 0.5, // centripetal
        }
    }

    /// Creates with specified alpha (tension).
    pub fn with_alpha(points: Vec<T>, alpha: f32) -> Self {
        Self { points, alpha }
    }

    /// Returns the number of segments.
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 2 {
            0
        } else {
            self.points.len() - 1
        }
    }

    /// Evaluates the spline at parameter t (0 to segment_count).
    pub fn evaluate(&self, t: f32) -> T {
        if self.points.is_empty() {
            panic!("Cannot evaluate empty spline");
        }
        if self.points.len() == 1 {
            return self.points[0];
        }

        let segment_count = self.segment_count() as f32;
        let t_clamped = t.clamp(0.0, segment_count);

        let segment = (t_clamped.floor() as usize).min(self.segment_count() - 1);
        let local_t = t_clamped - segment as f32;

        self.evaluate_segment(segment, local_t)
    }

    /// Evaluates a specific segment at local parameter t (0 to 1).
    fn evaluate_segment(&self, segment: usize, t: f32) -> T {
        let n = self.points.len();

        // Get the four points for this segment
        let i0 = if segment == 0 { 0 } else { segment - 1 };
        let i1 = segment;
        let i2 = (segment + 1).min(n - 1);
        let i3 = (segment + 2).min(n - 1);

        let p0 = self.points[i0];
        let p1 = self.points[i1];
        let p2 = self.points[i2];
        let p3 = self.points[i3];

        catmull_rom_segment(p0, p1, p2, p3, t)
    }

    /// Samples the spline at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.points.is_empty() {
            return Vec::new();
        }
        if num_samples == 1 {
            return vec![self.evaluate(0.0)];
        }

        let segment_count = self.segment_count() as f32;
        (0..num_samples)
            .map(|i| {
                let t = (i as f32 / (num_samples - 1) as f32) * segment_count;
                self.evaluate(t)
            })
            .collect()
    }
}

/// A B-spline curve.
///
/// B-splines provide smooth curves that approximate (but don't pass through)
/// control points, with local control.
#[derive(Debug, Clone)]
pub struct BSpline<T: Interpolatable> {
    /// Control points.
    pub points: Vec<T>,
    /// Degree of the spline (typically 3 for cubic).
    pub degree: usize,
    /// Knot vector (if None, uses uniform knots).
    knots: Option<Vec<f32>>,
}

impl<T: Interpolatable> BSpline<T> {
    /// Creates a cubic B-spline with uniform knots.
    pub fn cubic(points: Vec<T>) -> Self {
        Self {
            points,
            degree: 3,
            knots: None,
        }
    }

    /// Creates a B-spline with the specified degree.
    pub fn with_degree(points: Vec<T>, degree: usize) -> Self {
        Self {
            points,
            degree,
            knots: None,
        }
    }

    /// Creates a B-spline with custom knots.
    pub fn with_knots(points: Vec<T>, degree: usize, knots: Vec<f32>) -> Self {
        Self {
            points,
            degree,
            knots: Some(knots),
        }
    }

    /// Returns the knot vector (generates uniform if not set).
    fn get_knots(&self) -> Vec<f32> {
        if let Some(ref knots) = self.knots {
            knots.clone()
        } else {
            // Generate uniform clamped knots
            let n = self.points.len();
            let k = self.degree;
            let num_knots = n + k + 1;
            let mut knots = Vec::with_capacity(num_knots);

            for i in 0..num_knots {
                if i <= k {
                    knots.push(0.0);
                } else if i >= n {
                    knots.push((n - k) as f32);
                } else {
                    knots.push((i - k) as f32);
                }
            }

            knots
        }
    }

    /// Evaluates the B-spline at parameter t.
    pub fn evaluate(&self, t: f32) -> T {
        if self.points.is_empty() {
            panic!("Cannot evaluate empty spline");
        }
        if self.points.len() == 1 {
            return self.points[0];
        }

        let knots = self.get_knots();
        let n = self.points.len();
        let k = self.degree;

        // Clamp t to valid range
        let t_max = knots[n];
        let t_clamped = t.clamp(0.0, t_max - 0.0001);

        // Find the knot span
        let mut span = k;
        for i in k..n {
            if t_clamped < knots[i + 1] {
                span = i;
                break;
            }
        }

        // De Boor's algorithm
        let mut d: Vec<T> = (0..=k).map(|j| self.points[span - k + j]).collect();

        for r in 1..=k {
            for j in (r..=k).rev() {
                let i = span - k + j;
                let alpha = (t_clamped - knots[i]) / (knots[i + k + 1 - r] - knots[i]);
                d[j] = d[j - 1] * (1.0 - alpha) + d[j] * alpha;
            }
        }

        d[k]
    }

    /// Samples the spline at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.points.is_empty() {
            return Vec::new();
        }

        let knots = self.get_knots();
        let t_max = knots[self.points.len()];

        (0..num_samples)
            .map(|i| {
                let t = (i as f32 / (num_samples.max(2) - 1) as f32) * t_max;
                self.evaluate(t)
            })
            .collect()
    }
}

/// A piecewise cubic Bezier spline with continuity.
#[derive(Debug, Clone)]
pub struct BezierSpline<T: Interpolatable> {
    /// Bezier segments.
    pub segments: Vec<CubicBezier<T>>,
}

impl<T: Interpolatable> BezierSpline<T> {
    /// Creates an empty spline.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Creates from a list of points with automatic tangents.
    pub fn from_points(points: &[T]) -> Self {
        if points.len() < 2 {
            return Self::new();
        }

        let mut segments = Vec::with_capacity(points.len() - 1);

        for i in 0..points.len() - 1 {
            let p0 = points[i];
            let p3 = points[i + 1];

            // Compute tangents for smooth interpolation
            let tangent_scale = 0.25;

            let t0 = if i == 0 {
                p3 - p0
            } else {
                points[i + 1] - points[i - 1]
            };

            let t1 = if i + 2 >= points.len() {
                p3 - p0
            } else {
                points[i + 2] - points[i]
            };

            let p1 = p0 + t0 * tangent_scale;
            let p2 = p3 - t1 * tangent_scale;

            segments.push(CubicBezier::new(p0, p1, p2, p3));
        }

        Self { segments }
    }

    /// Adds a segment to the spline.
    pub fn push(&mut self, segment: CubicBezier<T>) {
        self.segments.push(segment);
    }

    /// Returns the number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns true if the spline is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Evaluates the spline at parameter t (0 to len).
    pub fn evaluate(&self, t: f32) -> T {
        if self.segments.is_empty() {
            panic!("Cannot evaluate empty spline");
        }

        let t_clamped = t.clamp(0.0, self.segments.len() as f32);
        let segment = (t_clamped.floor() as usize).min(self.segments.len() - 1);
        let local_t = t_clamped - segment as f32;

        self.segments[segment].evaluate(local_t)
    }

    /// Evaluates the derivative at parameter t.
    pub fn derivative(&self, t: f32) -> T {
        if self.segments.is_empty() {
            panic!("Cannot evaluate empty spline");
        }

        let t_clamped = t.clamp(0.0, self.segments.len() as f32);
        let segment = (t_clamped.floor() as usize).min(self.segments.len() - 1);
        let local_t = t_clamped - segment as f32;

        self.segments[segment].derivative(local_t)
    }

    /// Samples the spline at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.segments.is_empty() {
            return Vec::new();
        }

        let len = self.segments.len() as f32;
        (0..num_samples)
            .map(|i| {
                let t = (i as f32 / (num_samples.max(2) - 1) as f32) * len;
                self.evaluate(t)
            })
            .collect()
    }
}

impl<T: Interpolatable> Default for BezierSpline<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Linear interpolation between two values.
fn lerp<T: Interpolatable>(a: T, b: T, t: f32) -> T {
    a * (1.0 - t) + b * t
}

/// Evaluates a single Catmull-Rom segment.
fn catmull_rom_segment<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom basis matrix coefficients
    // P(t) = 0.5 * [(2P1) + (-P0 + P2)t + (2P0 - 5P1 + 4P2 - P3)t² + (-P0 + 3P1 - 3P2 + P3)t³]
    let c0 = p1 * 2.0;
    let c1 = p2 - p0;
    let c2 = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let c3 = p1 * 3.0 - p0 - p2 * 3.0 + p3;

    (c0 + c1 * t + c2 * t2 + c3 * t3) * 0.5
}

/// Creates a smooth curve through points using Catmull-Rom interpolation.
pub fn smooth_through_points<T: Interpolatable>(
    points: &[T],
    samples_per_segment: usize,
) -> Vec<T> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let spline = CatmullRom::new(points.to_vec());
    spline.sample((points.len() - 1) * samples_per_segment + 1)
}

/// Evaluates a quadratic Bezier curve.
pub fn quadratic_bezier<T: Interpolatable>(p0: T, p1: T, p2: T, t: f32) -> T {
    let mt = 1.0 - t;
    p0 * (mt * mt) + p1 * (2.0 * mt * t) + p2 * (t * t)
}

/// Evaluates a cubic Bezier curve.
pub fn cubic_bezier<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    CubicBezier::new(p0, p1, p2, p3).evaluate(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_bezier_endpoints() {
        let curve = CubicBezier::new(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        // Should pass through endpoints
        assert!((curve.evaluate(0.0) - Vec3::ZERO).length() < 0.001);
        assert!((curve.evaluate(1.0) - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_cubic_bezier_midpoint() {
        // Straight line
        let curve = CubicBezier::new(
            Vec3::ZERO,
            Vec3::new(0.333, 0.0, 0.0),
            Vec3::new(0.666, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let mid = curve.evaluate(0.5);
        assert!((mid.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cubic_bezier_split() {
        let curve = CubicBezier::new(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let (left, right) = curve.split(0.5);

        // Split point should match
        let split_point = curve.evaluate(0.5);
        assert!((left.evaluate(1.0) - split_point).length() < 0.001);
        assert!((right.evaluate(0.0) - split_point).length() < 0.001);
    }

    #[test]
    fn test_catmull_rom_passes_through_points() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ];

        let spline = CatmullRom::new(points.clone());

        // Should pass through all control points
        for (i, point) in points.iter().enumerate() {
            let t = i as f32;
            let eval = spline.evaluate(t);
            assert!(
                (eval - *point).length() < 0.001,
                "Point {} mismatch: {:?} vs {:?}",
                i,
                eval,
                point
            );
        }
    }

    #[test]
    fn test_catmull_rom_sampling() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let spline = CatmullRom::new(points);
        let samples = spline.sample(21);

        assert_eq!(samples.len(), 21);

        // First and last should match endpoints
        assert!((samples[0] - Vec3::ZERO).length() < 0.001);
        assert!((samples[20] - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_bspline_basic() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        let spline = BSpline::cubic(points);
        let samples = spline.sample(10);

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_bezier_spline_from_points() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let spline = BezierSpline::from_points(&points);

        assert_eq!(spline.len(), 2);

        // Should pass through endpoints
        assert!((spline.evaluate(0.0) - points[0]).length() < 0.001);
        assert!((spline.evaluate(2.0) - points[2]).length() < 0.001);
    }

    #[test]
    fn test_smooth_through_points() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let smoothed = smooth_through_points(&points, 10);

        // Should have (2 segments * 10 samples) + 1 = 21 points
        assert_eq!(smoothed.len(), 21);
    }

    #[test]
    fn test_f32_interpolation() {
        let curve = CubicBezier::new(0.0_f32, 0.25, 0.75, 1.0);

        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vec2_curves() {
        let curve = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        );

        let mid = curve.evaluate(0.5);
        assert!(mid.length() > 0.0);
    }
}
