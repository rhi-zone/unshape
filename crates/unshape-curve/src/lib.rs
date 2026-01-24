//! Unified curve trait and types for 2D/3D paths.
//!
//! This crate provides:
//! - [`VectorSpace`] - trait for types that support vector operations
//! - [`Curve`] - unified interface for all curve types
//! - Concrete curve types: [`Line`], [`QuadBezier`], [`CubicBezier`], [`Arc`]
//! - Segment enums: [`Segment2D`], [`Segment3D`]
//! - Path types: [`Path`], [`ArcLengthPath`]

use glam::{Vec2, Vec3};
use std::ops::{Add, Mul, Sub};

mod arc;
mod bezier;
mod line;
mod path;
mod segment;
mod traits;

pub use arc::Arc;
pub use bezier::{CubicBezier, QuadBezier};
pub use line::Line;
pub use path::{ArcLengthPath, Path};
pub use segment::{Segment2D, Segment3D};
pub use traits::{Curve, Curve2DExt, Curve3DExt, VectorSpace};

/// Trait for types that can be interpolated.
///
/// This is the base requirement for curve control points.
pub trait Interpolatable:
    Clone + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self>
{
}

impl Interpolatable for f32 {}
impl Interpolatable for Vec2 {}
impl Interpolatable for Vec3 {}

/// Linear interpolation between two values.
#[inline]
pub fn lerp<T: Interpolatable>(a: T, b: T, t: f32) -> T {
    a * (1.0 - t) + b * t
}

/// Axis-aligned bounding box (3D).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Self { min, max })
    }
}

/// Axis-aligned bounding rectangle (2D).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec2>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Self { min, max })
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
}

// ============================================================================
// Invariant tests
// ============================================================================

/// Invariant tests for curves.
///
/// These tests verify mathematical properties that should hold for all curve
/// types. Run with:
///
/// ```sh
/// cargo test -p unshape-curve --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use std::f32::consts::PI;

    // ========================================================================
    // Curve endpoint invariants
    // ========================================================================

    /// position_at(0) should equal start() for all curves.
    #[test]
    fn test_position_at_zero_equals_start() {
        let curves = create_test_curves_2d();
        for (name, curve) in curves {
            let pos = curve.position_at(0.0);
            let start = curve.start();
            assert!(
                (pos - start).length() < 0.001,
                "{name}: position_at(0) = {pos:?} != start() = {start:?}"
            );
        }
    }

    /// position_at(1) should equal end() for all curves.
    #[test]
    fn test_position_at_one_equals_end() {
        let curves = create_test_curves_2d();
        for (name, curve) in curves {
            let pos = curve.position_at(1.0);
            let end = curve.end();
            assert!(
                (pos - end).length() < 0.001,
                "{name}: position_at(1) = {pos:?} != end() = {end:?}"
            );
        }
    }

    // ========================================================================
    // Continuity invariants
    // ========================================================================

    /// Curves should be continuous: small t changes give small position changes.
    #[test]
    fn test_curve_continuity() {
        let curves = create_test_curves_2d();
        let epsilon = 0.001;

        for (name, curve) in curves {
            let mut prev_pos = curve.position_at(0.0);
            for i in 1..=100 {
                let t = i as f32 / 100.0;
                let pos = curve.position_at(t);
                let delta = (pos - prev_pos).length();

                // Change should be bounded (rough estimate based on curve length)
                let max_change = curve.length() / 50.0 + epsilon;
                assert!(
                    delta < max_change,
                    "{name}: discontinuity at t={t}, delta={delta}"
                );
                prev_pos = pos;
            }
        }
    }

    /// Tangent should be continuous for smooth curves.
    #[test]
    fn test_tangent_continuity() {
        // Test quadratic bezier
        let quad = QuadBezier::new(Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0));
        let mut prev_tangent = quad.tangent_at(0.01);
        for i in 2..=99 {
            let t = i as f32 / 100.0;
            let tangent = quad.tangent_at(t);
            let prev_normalized = prev_tangent.normalize();
            let curr_normalized = tangent.normalize();
            let dot = prev_normalized.dot(curr_normalized);
            assert!(dot > 0.9, "Quad tangent discontinuity at t={t}, dot={dot}");
            prev_tangent = tangent;
        }

        // Test cubic bezier
        let cubic = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(3.0, 0.0),
        );
        let mut prev_tangent = cubic.tangent_at(0.01);
        for i in 2..=99 {
            let t = i as f32 / 100.0;
            let tangent = cubic.tangent_at(t);
            let prev_normalized = prev_tangent.normalize();
            let curr_normalized = tangent.normalize();
            let dot = prev_normalized.dot(curr_normalized);
            assert!(dot > 0.9, "Cubic tangent discontinuity at t={t}, dot={dot}");
            prev_tangent = tangent;
        }
    }

    // ========================================================================
    // Arc length invariants
    // ========================================================================

    /// Arc length should be non-negative.
    #[test]
    fn test_arc_length_non_negative() {
        let curves = create_test_curves_2d();
        for (name, curve) in curves {
            let len = curve.length();
            assert!(len >= 0.0, "{name}: length = {len} < 0");
        }
    }

    /// Line length should equal Euclidean distance.
    #[test]
    fn test_line_length_is_euclidean() {
        for _ in 0..20 {
            let start = Vec2::new(rand_f32(-100.0, 100.0), rand_f32(-100.0, 100.0));
            let end = Vec2::new(rand_f32(-100.0, 100.0), rand_f32(-100.0, 100.0));
            let line = Line::new(start, end);

            let euclidean = (end - start).length();
            let arc_length = line.length();

            assert!(
                (arc_length - euclidean).abs() < 0.001,
                "Line length {} != Euclidean {}",
                arc_length,
                euclidean
            );
        }
    }

    /// Circle arc length should be r * θ.
    #[test]
    fn test_circle_arc_length() {
        for radius in [0.5, 1.0, 2.0, 5.0] {
            for sweep in [PI / 4.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI] {
                let arc = Arc::circle(Vec2::ZERO, radius, 0.0, sweep);
                let expected = radius * sweep;
                let actual = arc.length();

                assert!(
                    (actual - expected).abs() < 0.01,
                    "Circle arc (r={radius}, θ={sweep}): expected {expected}, got {actual}"
                );
            }
        }
    }

    // ========================================================================
    // Split invariants
    // ========================================================================

    /// Split should preserve endpoints: left.end() = right.start() = position_at(t).
    #[test]
    fn test_split_preserves_continuity() {
        let curves = create_test_curves_2d();

        for (name, curve) in curves {
            for t in [0.25, 0.5, 0.75] {
                let (left, right) = curve.split(t);
                let split_pos = curve.position_at(t);

                let left_end = left.position_at(1.0);
                let right_start = right.position_at(0.0);

                assert!(
                    (left_end - split_pos).length() < 0.001,
                    "{name}: left.end() != position_at({t})"
                );
                assert!(
                    (right_start - split_pos).length() < 0.001,
                    "{name}: right.start() != position_at({t})"
                );
            }
        }
    }

    /// Split curves should approximate original curve.
    #[test]
    fn test_split_approximates_original() {
        let quad = QuadBezier::new(Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0));
        let (left, right) = quad.split(0.5);

        // Test left half
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let original_t = t * 0.5; // Map [0,1] to [0,0.5]
            let original_pos = quad.position_at(original_t);
            let split_pos = left.position_at(t);
            assert!(
                (original_pos - split_pos).length() < 0.001,
                "Left split mismatch at t={t}"
            );
        }

        // Test right half
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let original_t = 0.5 + t * 0.5; // Map [0,1] to [0.5,1]
            let original_pos = quad.position_at(original_t);
            let split_pos = right.position_at(t);
            assert!(
                (original_pos - split_pos).length() < 0.001,
                "Right split mismatch at t={t}"
            );
        }
    }

    // ========================================================================
    // Bezier invariants
    // ========================================================================

    /// Quadratic degree elevation should be exact.
    #[test]
    fn test_quad_elevation_exact() {
        for _ in 0..10 {
            let quad = QuadBezier::new(
                Vec2::new(rand_f32(-10.0, 10.0), rand_f32(-10.0, 10.0)),
                Vec2::new(rand_f32(-10.0, 10.0), rand_f32(-10.0, 10.0)),
                Vec2::new(rand_f32(-10.0, 10.0), rand_f32(-10.0, 10.0)),
            );
            let cubic = quad.elevate();

            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let quad_pos = quad.position_at(t);
                let cubic_pos = cubic.position_at(t);
                assert!(
                    (quad_pos - cubic_pos).length() < 0.001,
                    "Elevation mismatch at t={t}"
                );
            }
        }
    }

    /// Control polygon length >= arc length (convex hull property).
    #[test]
    fn test_bezier_control_polygon_longer() {
        let cubic = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 3.0),
            Vec2::new(2.0, 3.0),
            Vec2::new(3.0, 0.0),
        );

        let control_len = (cubic.p1 - cubic.p0).length()
            + (cubic.p2 - cubic.p1).length()
            + (cubic.p3 - cubic.p2).length();
        let arc_len = cubic.length();

        assert!(
            control_len >= arc_len - 0.01,
            "Control polygon ({control_len}) should be >= arc length ({arc_len})"
        );
    }

    // ========================================================================
    // Arc invariants
    // ========================================================================

    /// All points on a circular arc should be at radius distance from center.
    #[test]
    fn test_circle_arc_radius_invariant() {
        for radius in [0.5, 1.0, 2.0, 5.0] {
            let arc = Arc::circle(Vec2::ZERO, radius, 0.0, PI);

            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let pos = arc.position_at(t);
                let dist = pos.length();

                assert!(
                    (dist - radius).abs() < 0.001,
                    "Arc point at t={t} has dist={dist}, expected {radius}"
                );
            }
        }
    }

    /// Tangent should be perpendicular to radius for circular arcs.
    #[test]
    fn test_circle_arc_tangent_perpendicular() {
        let arc = Arc::circle(Vec2::new(5.0, 5.0), 2.0, 0.0, PI);

        for i in 1..20 {
            let t = i as f32 / 20.0;
            let pos = arc.position_at(t);
            let tangent = arc.tangent_at(t);

            let radius_vec = pos - arc.center;
            let dot = radius_vec.normalize().dot(tangent.normalize());

            assert!(
                dot.abs() < 0.01,
                "Tangent not perpendicular at t={t}, dot={dot}"
            );
        }
    }

    // ========================================================================
    // Path invariants
    // ========================================================================

    /// Path length should equal sum of segment lengths.
    #[test]
    fn test_path_length_is_sum() {
        let segments = vec![
            Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(3.0, 0.0))),
            Segment2D::Line(Line::new(Vec2::new(3.0, 0.0), Vec2::new(3.0, 4.0))),
            Segment2D::Line(Line::new(Vec2::new(3.0, 4.0), Vec2::new(0.0, 4.0))),
        ];

        let expected_len: f32 = segments.iter().map(|s| s.length()).sum();
        let path = Path::from_segments(segments);
        let path_len = path.length();

        assert!(
            (path_len - expected_len).abs() < 0.001,
            "Path length {} != sum of segments {}",
            path_len,
            expected_len
        );
    }

    /// ArcLengthPath should give uniform speed.
    #[test]
    fn test_arc_length_path_uniform_speed() {
        // Create path with segments of different lengths
        let path = Path::from_segments(vec![
            Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0))), // length 10
            Segment2D::Line(Line::new(Vec2::new(10.0, 0.0), Vec2::new(10.0, 30.0))), // length 30
        ]);

        let arc_path = ArcLengthPath::new(path);
        let _total = arc_path.total_length();

        // Sample at equal arc-length intervals
        let samples = arc_path.sample(5);
        assert_eq!(samples.len(), 5);

        // Check that samples are approximately equidistant along the path
        // t=0.0: dist=0, t=0.25: dist=10, t=0.5: dist=20, t=0.75: dist=30, t=1.0: dist=40
        let expected_positions = [
            Vec2::new(0.0, 0.0),   // t=0.0, dist=0
            Vec2::new(10.0, 0.0),  // t=0.25, dist=10 (end of first segment)
            Vec2::new(10.0, 10.0), // t=0.5, dist=20
            Vec2::new(10.0, 20.0), // t=0.75, dist=30
            Vec2::new(10.0, 30.0), // t=1.0, dist=40
        ];

        for (i, (sample, expected)) in samples.iter().zip(expected_positions.iter()).enumerate() {
            assert!(
                (*sample - *expected).length() < 0.1,
                "Sample {i}: got {:?}, expected {:?}",
                sample,
                expected
            );
        }
    }

    // ========================================================================
    // Flatten invariants
    // ========================================================================

    /// Flattened points should start and end at curve endpoints.
    #[test]
    fn test_flatten_preserves_endpoints() {
        let curves = create_test_curves_2d();

        for (name, curve) in curves {
            let points = curve.flatten(0.1);
            assert!(!points.is_empty(), "{name}: flatten returned empty");

            let first = points.first().unwrap();
            let last = points.last().unwrap();

            assert!(
                (*first - curve.start()).length() < 0.001,
                "{name}: first flattened point != start()"
            );
            assert!(
                (*last - curve.end()).length() < 0.001,
                "{name}: last flattened point != end()"
            );
        }
    }

    /// Tighter tolerance should produce more points.
    #[test]
    fn test_flatten_more_points_with_tighter_tolerance() {
        let cubic = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 3.0),
            Vec2::new(2.0, 3.0),
            Vec2::new(3.0, 0.0),
        );

        let loose = cubic.flatten(1.0);
        let tight = cubic.flatten(0.01);

        assert!(
            tight.len() >= loose.len(),
            "Tighter tolerance should produce >= points: {} vs {}",
            tight.len(),
            loose.len()
        );
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    fn create_test_curves_2d() -> Vec<(&'static str, Segment2D)> {
        vec![
            (
                "Line",
                Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 5.0))),
            ),
            (
                "QuadBezier",
                Segment2D::Quad(QuadBezier::new(
                    Vec2::ZERO,
                    Vec2::new(5.0, 10.0),
                    Vec2::new(10.0, 0.0),
                )),
            ),
            (
                "CubicBezier",
                Segment2D::Cubic(CubicBezier::new(
                    Vec2::ZERO,
                    Vec2::new(2.0, 5.0),
                    Vec2::new(8.0, 5.0),
                    Vec2::new(10.0, 0.0),
                )),
            ),
            (
                "Arc",
                Segment2D::Arc(Arc::circle(Vec2::ZERO, 5.0, 0.0, PI / 2.0)),
            ),
        ]
    }

    /// Simple LCG random number generator for tests.
    fn rand_f32(min: f32, max: f32) -> f32 {
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = const { Cell::new(54321) };
        }
        SEED.with(|seed| {
            let s = seed.get().wrapping_mul(6364136223846793005).wrapping_add(1);
            seed.set(s);
            let t = ((s >> 33) as u32) as f32 / u32::MAX as f32;
            min + t * (max - min)
        })
    }
}
