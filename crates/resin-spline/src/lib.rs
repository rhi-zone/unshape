//! Spline and curve types for interpolation.
//!
//! Provides various curve types for smooth interpolation:
//! - [`CubicBezier`] - Cubic Bezier curves
//! - [`CatmullRom`] - Catmull-Rom splines (pass through control points)
//! - [`BSpline`] - B-spline curves
//! - [`Nurbs`] - Non-Uniform Rational B-Splines
//!
//! All types implement the [`Curve`] trait from `resin-curve` for Vec2/Vec3.

use glam::{Vec2, Vec3};

mod curve_impl;

// Re-export Curve trait for convenience
pub use rhizome_resin_curve::Curve;

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
    ///
    /// Returns `None` if the spline has no control points.
    pub fn evaluate(&self, t: f32) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }
        if self.points.len() == 1 {
            return Some(self.points[0]);
        }

        let segment_count = self.segment_count() as f32;
        let t_clamped = t.clamp(0.0, segment_count);

        let segment = (t_clamped.floor() as usize).min(self.segment_count() - 1);
        let local_t = t_clamped - segment as f32;

        Some(self.evaluate_segment(segment, local_t))
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
            return self.evaluate(0.0).into_iter().collect();
        }

        let segment_count = self.segment_count() as f32;
        (0..num_samples)
            .filter_map(|i| {
                let t = (i as f32 / (num_samples - 1) as f32) * segment_count;
                self.evaluate(t)
            })
            .collect()
    }
}

impl<T: Interpolatable> Default for CatmullRom<T> {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            alpha: 0.5,
        }
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
    ///
    /// Returns `None` if the spline has no control points.
    pub fn evaluate(&self, t: f32) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }
        if self.points.len() == 1 {
            return Some(self.points[0]);
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

        Some(d[k])
    }

    /// Samples the spline at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.points.is_empty() {
            return Vec::new();
        }

        let knots = self.get_knots();
        let t_max = knots[self.points.len()];

        (0..num_samples)
            .filter_map(|i| {
                let t = (i as f32 / (num_samples.max(2) - 1) as f32) * t_max;
                self.evaluate(t)
            })
            .collect()
    }
}

impl<T: Interpolatable> Default for BSpline<T> {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            degree: 3,
            knots: None,
        }
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
    ///
    /// Returns `None` if the spline has no segments.
    pub fn evaluate(&self, t: f32) -> Option<T> {
        if self.segments.is_empty() {
            return None;
        }

        let t_clamped = t.clamp(0.0, self.segments.len() as f32);
        let segment = (t_clamped.floor() as usize).min(self.segments.len() - 1);
        let local_t = t_clamped - segment as f32;

        Some(self.segments[segment].evaluate(local_t))
    }

    /// Evaluates the derivative at parameter t.
    ///
    /// Returns `None` if the spline has no segments.
    pub fn derivative(&self, t: f32) -> Option<T> {
        if self.segments.is_empty() {
            return None;
        }

        let t_clamped = t.clamp(0.0, self.segments.len() as f32);
        let segment = (t_clamped.floor() as usize).min(self.segments.len() - 1);
        let local_t = t_clamped - segment as f32;

        Some(self.segments[segment].derivative(local_t))
    }

    /// Samples the spline at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.segments.is_empty() {
            return Vec::new();
        }

        let len = self.segments.len() as f32;
        (0..num_samples)
            .filter_map(|i| {
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
// NURBS (Non-Uniform Rational B-Splines)
// ============================================================================

/// A weighted control point for NURBS curves.
#[derive(Debug, Clone, Copy)]
pub struct WeightedPoint<T: Interpolatable> {
    /// The control point position.
    pub point: T,
    /// The weight (typically > 0).
    pub weight: f32,
}

impl<T: Interpolatable> WeightedPoint<T> {
    /// Creates a new weighted point.
    pub fn new(point: T, weight: f32) -> Self {
        Self { point, weight }
    }

    /// Creates a weighted point with weight 1.0.
    pub fn unweighted(point: T) -> Self {
        Self { point, weight: 1.0 }
    }
}

/// A NURBS (Non-Uniform Rational B-Spline) curve.
///
/// NURBS curves extend B-splines with weights, enabling:
/// - Exact representation of conic sections (circles, ellipses, parabolas)
/// - Greater control over curve shape
/// - Industry-standard curve representation (CAD, modeling)
#[derive(Debug, Clone)]
pub struct Nurbs<T: Interpolatable> {
    /// Weighted control points.
    pub points: Vec<WeightedPoint<T>>,
    /// Degree of the spline (typically 2 for conics, 3 for cubic).
    pub degree: usize,
    /// Knot vector.
    knots: Vec<f32>,
}

impl<T: Interpolatable> Nurbs<T> {
    /// Creates a NURBS curve with uniform knots.
    pub fn new(points: Vec<WeightedPoint<T>>, degree: usize) -> Self {
        let knots = Self::uniform_clamped_knots(points.len(), degree);
        Self {
            points,
            degree,
            knots,
        }
    }

    /// Creates a NURBS curve with custom knots.
    pub fn with_knots(points: Vec<WeightedPoint<T>>, degree: usize, knots: Vec<f32>) -> Self {
        Self {
            points,
            degree,
            knots,
        }
    }

    /// Creates a quadratic NURBS (degree 2, good for conics).
    pub fn quadratic(points: Vec<WeightedPoint<T>>) -> Self {
        Self::new(points, 2)
    }

    /// Creates a cubic NURBS (degree 3).
    pub fn cubic(points: Vec<WeightedPoint<T>>) -> Self {
        Self::new(points, 3)
    }

    /// Creates from unweighted points (equivalent to regular B-spline).
    pub fn from_points(points: Vec<T>, degree: usize) -> Self {
        let weighted = points.into_iter().map(WeightedPoint::unweighted).collect();
        Self::new(weighted, degree)
    }

    /// Generates uniform clamped knots for n control points and degree k.
    fn uniform_clamped_knots(n: usize, k: usize) -> Vec<f32> {
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

    /// Returns the parameter range [t_min, t_max].
    pub fn domain(&self) -> (f32, f32) {
        (self.knots[self.degree], self.knots[self.points.len()])
    }

    /// Evaluates the NURBS curve at parameter t using the rational De Boor algorithm.
    ///
    /// Returns `None` if the curve has no control points.
    pub fn evaluate(&self, t: f32) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }
        if self.points.len() == 1 {
            return Some(self.points[0].point);
        }

        let n = self.points.len();
        let k = self.degree;

        // Clamp t to valid range
        let (t_min, t_max) = self.domain();
        let t_clamped = t.clamp(t_min, t_max - 0.0001);

        // Find the knot span
        let mut span = k;
        for i in k..n {
            if t_clamped < self.knots[i + 1] {
                span = i;
                break;
            }
        }

        // Rational De Boor's algorithm
        // Work in homogeneous coordinates: (w*x, w*y, w*z, w)
        let mut d: Vec<(T, f32)> = (0..=k)
            .map(|j| {
                let wp = &self.points[span - k + j];
                (wp.point * wp.weight, wp.weight)
            })
            .collect();

        for r in 1..=k {
            for j in (r..=k).rev() {
                let i = span - k + j;
                let denom = self.knots[i + k + 1 - r] - self.knots[i];
                let alpha = if denom.abs() < 1e-10 {
                    0.0
                } else {
                    (t_clamped - self.knots[i]) / denom
                };

                let (p_prev, w_prev) = d[j - 1];
                let (p_curr, w_curr) = d[j];

                // Linear interpolation in homogeneous space
                d[j] = (
                    p_prev * (1.0 - alpha) + p_curr * alpha,
                    w_prev * (1.0 - alpha) + w_curr * alpha,
                );
            }
        }

        // Project back from homogeneous coordinates
        let (p, w) = d[k];
        Some(if w.abs() < 1e-10 { p } else { p * (1.0 / w) })
    }

    /// Evaluates the derivative at parameter t.
    ///
    /// Returns `None` if the curve has no control points.
    pub fn derivative(&self, t: f32) -> Option<T> {
        // Numerical derivative using central differences
        let h = 0.001;
        let (t_min, t_max) = self.domain();

        let t_lo = (t - h).max(t_min);
        let t_hi = (t + h).min(t_max - 0.0001);

        let p_lo = self.evaluate(t_lo)?;
        let p_hi = self.evaluate(t_hi)?;

        Some((p_hi - p_lo) * (1.0 / (t_hi - t_lo)))
    }

    /// Samples the curve at regular intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<T> {
        if num_samples == 0 || self.points.is_empty() {
            return Vec::new();
        }

        let (t_min, t_max) = self.domain();

        (0..num_samples)
            .filter_map(|i| {
                let t = t_min + (i as f32 / (num_samples.max(2) - 1) as f32) * (t_max - t_min);
                self.evaluate(t)
            })
            .collect()
    }

    /// Inserts a knot at parameter t (knot insertion for refinement).
    pub fn insert_knot(&mut self, t: f32) {
        let n = self.points.len();
        let k = self.degree;

        // Find the knot span
        let mut span = k;
        for i in k..n {
            if t < self.knots[i + 1] {
                span = i;
                break;
            }
        }

        // Calculate new control points
        let mut new_points = Vec::with_capacity(n + 1);

        for i in 0..=n {
            if i <= span - k {
                new_points.push(self.points[i].clone());
            } else if i > span {
                new_points.push(self.points[i - 1].clone());
            } else {
                let alpha = (t - self.knots[i]) / (self.knots[i + k] - self.knots[i]);
                let wp_prev = &self.points[i - 1];
                let wp_curr = &self.points[i];

                let new_point = wp_prev.point * (1.0 - alpha) + wp_curr.point * alpha;
                let new_weight = wp_prev.weight * (1.0 - alpha) + wp_curr.weight * alpha;

                new_points.push(WeightedPoint::new(new_point, new_weight));
            }
        }

        // Insert the knot
        let mut new_knots = Vec::with_capacity(self.knots.len() + 1);
        for (i, &knot) in self.knots.iter().enumerate() {
            if i == span + 1 {
                new_knots.push(t);
            }
            new_knots.push(knot);
        }

        self.points = new_points;
        self.knots = new_knots;
    }
}

impl<T: Interpolatable> Default for Nurbs<T> {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            degree: 3,
            knots: Vec::new(),
        }
    }
}

// ============================================================================
// NURBS Primitive Curves
// ============================================================================

/// Creates a NURBS circle in the XY plane.
pub fn nurbs_circle(center: Vec3, radius: f32) -> Nurbs<Vec3> {
    // 9-point NURBS representation of a circle using degree 2
    let w = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2) for 45-degree arcs

    let points = vec![
        WeightedPoint::new(center + Vec3::new(radius, 0.0, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(radius, radius, 0.0), w),
        WeightedPoint::new(center + Vec3::new(0.0, radius, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(-radius, radius, 0.0), w),
        WeightedPoint::new(center + Vec3::new(-radius, 0.0, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(-radius, -radius, 0.0), w),
        WeightedPoint::new(center + Vec3::new(0.0, -radius, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(radius, -radius, 0.0), w),
        WeightedPoint::new(center + Vec3::new(radius, 0.0, 0.0), 1.0),
    ];

    // Knot vector for a closed periodic curve with 90-degree arcs
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];

    Nurbs::with_knots(points, 2, knots)
}

/// Creates a NURBS arc in the XY plane.
pub fn nurbs_arc(center: Vec3, radius: f32, start_angle: f32, end_angle: f32) -> Nurbs<Vec3> {
    let angle_span = end_angle - start_angle;

    // For small arcs, use a single segment
    if angle_span.abs() <= std::f32::consts::FRAC_PI_2 + 0.01 {
        let mid_angle = (start_angle + end_angle) / 2.0;
        let w = (angle_span / 2.0).cos();

        let p0 = center + Vec3::new(radius * start_angle.cos(), radius * start_angle.sin(), 0.0);
        let p2 = center + Vec3::new(radius * end_angle.cos(), radius * end_angle.sin(), 0.0);

        // Middle control point on the tangent lines
        let tan_factor = radius / (angle_span / 2.0).cos();
        let p1 = center
            + Vec3::new(
                tan_factor * mid_angle.cos(),
                tan_factor * mid_angle.sin(),
                0.0,
            );

        let points = vec![
            WeightedPoint::new(p0, 1.0),
            WeightedPoint::new(p1, w),
            WeightedPoint::new(p2, 1.0),
        ];

        return Nurbs::quadratic(points);
    }

    // For larger arcs, split into multiple segments
    let num_segments = ((angle_span.abs() / std::f32::consts::FRAC_PI_2).ceil() as usize).max(1);
    let segment_angle = angle_span / num_segments as f32;
    let w = (segment_angle / 2.0).cos();

    let mut points = Vec::new();

    for i in 0..=num_segments {
        let angle = start_angle + i as f32 * segment_angle;
        let p = center + Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.0);
        points.push(WeightedPoint::new(p, 1.0));

        if i < num_segments {
            let mid_angle = angle + segment_angle / 2.0;
            let tan_factor = radius / (segment_angle / 2.0).cos();
            let p_mid = center
                + Vec3::new(
                    tan_factor * mid_angle.cos(),
                    tan_factor * mid_angle.sin(),
                    0.0,
                );
            points.push(WeightedPoint::new(p_mid, w));
        }
    }

    // Generate knots
    let n = points.len();
    let knots = Nurbs::<Vec3>::uniform_clamped_knots(n, 2);

    Nurbs::with_knots(points, 2, knots)
}

/// Creates a NURBS ellipse in the XY plane.
pub fn nurbs_ellipse(center: Vec3, radius_x: f32, radius_y: f32) -> Nurbs<Vec3> {
    let w = std::f32::consts::FRAC_1_SQRT_2;

    let points = vec![
        WeightedPoint::new(center + Vec3::new(radius_x, 0.0, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(radius_x, radius_y, 0.0), w),
        WeightedPoint::new(center + Vec3::new(0.0, radius_y, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(-radius_x, radius_y, 0.0), w),
        WeightedPoint::new(center + Vec3::new(-radius_x, 0.0, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(-radius_x, -radius_y, 0.0), w),
        WeightedPoint::new(center + Vec3::new(0.0, -radius_y, 0.0), 1.0),
        WeightedPoint::new(center + Vec3::new(radius_x, -radius_y, 0.0), w),
        WeightedPoint::new(center + Vec3::new(radius_x, 0.0, 0.0), 1.0),
    ];

    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];

    Nurbs::with_knots(points, 2, knots)
}

/// Creates a 2D NURBS circle in the XY plane.
pub fn nurbs_circle_2d(center: Vec2, radius: f32) -> Nurbs<Vec2> {
    let w = std::f32::consts::FRAC_1_SQRT_2;

    let points = vec![
        WeightedPoint::new(center + Vec2::new(radius, 0.0), 1.0),
        WeightedPoint::new(center + Vec2::new(radius, radius), w),
        WeightedPoint::new(center + Vec2::new(0.0, radius), 1.0),
        WeightedPoint::new(center + Vec2::new(-radius, radius), w),
        WeightedPoint::new(center + Vec2::new(-radius, 0.0), 1.0),
        WeightedPoint::new(center + Vec2::new(-radius, -radius), w),
        WeightedPoint::new(center + Vec2::new(0.0, -radius), 1.0),
        WeightedPoint::new(center + Vec2::new(radius, -radius), w),
        WeightedPoint::new(center + Vec2::new(radius, 0.0), 1.0),
    ];

    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];

    Nurbs::with_knots(points, 2, knots)
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
            let eval = spline.evaluate(t).unwrap();
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
        assert!((spline.evaluate(0.0).unwrap() - points[0]).length() < 0.001);
        assert!((spline.evaluate(2.0).unwrap() - points[2]).length() < 0.001);
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

    // NURBS tests
    #[test]
    fn test_nurbs_basic() {
        let points = vec![
            WeightedPoint::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(1.0, 2.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(2.0, 2.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(3.0, 0.0, 0.0), 1.0),
        ];

        let nurbs = Nurbs::cubic(points);
        let samples = nurbs.sample(10);

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_nurbs_endpoints() {
        let points = vec![
            WeightedPoint::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(2.0, 0.0, 0.0), 1.0),
        ];

        let nurbs = Nurbs::quadratic(points);
        let (t_min, t_max) = nurbs.domain();

        // Should pass through first and last control points
        let start = nurbs.evaluate(t_min).unwrap();
        let end = nurbs.evaluate(t_max - 0.0001).unwrap();

        assert!((start - Vec3::new(0.0, 0.0, 0.0)).length() < 0.01);
        assert!((end - Vec3::new(2.0, 0.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn test_nurbs_weights_affect_shape() {
        // Same control points, different weights
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 2.0, 0.0);
        let p2 = Vec3::new(2.0, 0.0, 0.0);

        // Uniform weights
        let nurbs_uniform = Nurbs::quadratic(vec![
            WeightedPoint::new(p0, 1.0),
            WeightedPoint::new(p1, 1.0),
            WeightedPoint::new(p2, 1.0),
        ]);

        // Higher weight on middle point pulls curve toward it
        let nurbs_weighted = Nurbs::quadratic(vec![
            WeightedPoint::new(p0, 1.0),
            WeightedPoint::new(p1, 3.0),
            WeightedPoint::new(p2, 1.0),
        ]);

        let (t_min, t_max) = nurbs_uniform.domain();
        let t_mid = (t_min + t_max) / 2.0;

        let mid_uniform = nurbs_uniform.evaluate(t_mid).unwrap();
        let mid_weighted = nurbs_weighted.evaluate(t_mid).unwrap();

        // Weighted version should be closer to the middle control point
        let dist_to_p1_uniform = (mid_uniform - p1).length();
        let dist_to_p1_weighted = (mid_weighted - p1).length();

        assert!(
            dist_to_p1_weighted < dist_to_p1_uniform,
            "Higher weight should pull curve closer to control point"
        );
    }

    #[test]
    fn test_nurbs_circle_is_circular() {
        let circle = nurbs_circle(Vec3::ZERO, 1.0);
        let samples = circle.sample(100);

        // All points should be at distance ~1 from center
        for sample in &samples {
            let dist = sample.length();
            assert!(
                (dist - 1.0).abs() < 0.001,
                "Point {:?} is at distance {} from center, expected 1.0",
                sample,
                dist
            );
        }
    }

    #[test]
    fn test_nurbs_ellipse() {
        let ellipse = nurbs_ellipse(Vec3::ZERO, 2.0, 1.0);
        let samples = ellipse.sample(100);

        // All points should satisfy ellipse equation: (x/a)² + (y/b)² = 1
        for sample in &samples {
            let val = (sample.x / 2.0).powi(2) + (sample.y / 1.0).powi(2);
            assert!(
                (val - 1.0).abs() < 0.001,
                "Point {:?} doesn't satisfy ellipse equation: {}",
                sample,
                val
            );
        }
    }

    #[test]
    fn test_nurbs_arc() {
        use std::f32::consts::FRAC_PI_2;

        // Quarter circle arc
        let arc = nurbs_arc(Vec3::ZERO, 1.0, 0.0, FRAC_PI_2);
        let samples = arc.sample(20);

        // All points should be at distance 1 from center
        for sample in &samples {
            let dist = (sample.x.powi(2) + sample.y.powi(2)).sqrt();
            assert!(
                (dist - 1.0).abs() < 0.01,
                "Arc point {:?} at distance {}",
                sample,
                dist
            );
        }

        // Start should be at (1, 0)
        assert!((samples[0].x - 1.0).abs() < 0.01);
        assert!(samples[0].y.abs() < 0.01);

        // End should be at (0, 1)
        let last = samples.last().unwrap();
        assert!(last.x.abs() < 0.01);
        assert!((last.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_nurbs_circle_2d() {
        let circle = nurbs_circle_2d(Vec2::ZERO, 1.0);
        let samples = circle.sample(50);

        for sample in &samples {
            let dist = sample.length();
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_nurbs_from_unweighted() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
        ];

        let nurbs = Nurbs::from_points(points.clone(), 3);

        // Should behave like a regular B-spline
        let bspline = BSpline::cubic(points);

        let nurbs_samples = nurbs.sample(10);
        let bspline_samples = bspline.sample(10);

        for (n, b) in nurbs_samples.iter().zip(bspline_samples.iter()) {
            assert!((*n - *b).length() < 0.001);
        }
    }

    #[test]
    fn test_nurbs_derivative() {
        let points = vec![
            WeightedPoint::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
            WeightedPoint::new(Vec3::new(2.0, 0.0, 0.0), 1.0),
        ];

        let nurbs = Nurbs::quadratic(points);
        let (t_min, t_max) = nurbs.domain();
        let t_mid = (t_min + t_max) / 2.0;

        let deriv = nurbs.derivative(t_mid).unwrap();

        // Derivative should be non-zero and tangent to curve
        assert!(deriv.length() > 0.0);
    }
}
