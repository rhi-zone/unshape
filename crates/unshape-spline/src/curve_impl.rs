//! Curve trait implementations for spline types.

use crate::{BezierSpline, CubicBezier, Nurbs};
use glam::{Vec2, Vec3};
use unshape_curve::{CubicBezier as CurveCubic, Curve};

// ============================================================================
// CubicBezier<T> implements Curve for Vec2/Vec3
// ============================================================================

impl Curve for CubicBezier<Vec2> {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        self.evaluate(t)
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        self.derivative(t)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        CubicBezier::split(self, t)
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec2>> {
        vec![CurveCubic::new(self.p0, self.p1, self.p2, self.p3)]
    }

    fn start(&self) -> Vec2 {
        self.p0
    }

    fn end(&self) -> Vec2 {
        self.p3
    }
}

impl Curve for CubicBezier<Vec3> {
    type Point = Vec3;

    fn position_at(&self, t: f32) -> Vec3 {
        self.evaluate(t)
    }

    fn tangent_at(&self, t: f32) -> Vec3 {
        self.derivative(t)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        CubicBezier::split(self, t)
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec3>> {
        vec![CurveCubic::new(self.p0, self.p1, self.p2, self.p3)]
    }

    fn start(&self) -> Vec3 {
        self.p0
    }

    fn end(&self) -> Vec3 {
        self.p3
    }
}

// ============================================================================
// BezierSpline<T> implements Curve for Vec2/Vec3
// ============================================================================

impl Curve for BezierSpline<Vec2> {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        // Scale t from [0,1] to [0, len]
        let scaled_t = t * self.segments.len() as f32;
        self.evaluate(scaled_t).unwrap_or(Vec2::ZERO)
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        let scaled_t = t * self.segments.len() as f32;
        self.derivative(scaled_t).unwrap_or(Vec2::ZERO)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        if self.segments.is_empty() {
            return (Self::new(), Self::new());
        }

        let scaled_t = t * self.segments.len() as f32;
        let segment_idx = (scaled_t.floor() as usize).min(self.segments.len() - 1);
        let local_t = scaled_t - segment_idx as f32;

        let mut left_segments = self.segments[..segment_idx].to_vec();
        let mut right_segments = self.segments[segment_idx + 1..].to_vec();

        let (left_part, right_part) = self.segments[segment_idx].split(local_t);
        left_segments.push(left_part);
        right_segments.insert(0, right_part);

        (
            BezierSpline {
                segments: left_segments,
            },
            BezierSpline {
                segments: right_segments,
            },
        )
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec2>> {
        self.segments
            .iter()
            .map(|s| CurveCubic::new(s.p0, s.p1, s.p2, s.p3))
            .collect()
    }

    fn start(&self) -> Vec2 {
        self.segments.first().map(|s| s.p0).unwrap_or(Vec2::ZERO)
    }

    fn end(&self) -> Vec2 {
        self.segments.last().map(|s| s.p3).unwrap_or(Vec2::ZERO)
    }
}

impl Curve for BezierSpline<Vec3> {
    type Point = Vec3;

    fn position_at(&self, t: f32) -> Vec3 {
        let scaled_t = t * self.segments.len() as f32;
        self.evaluate(scaled_t).unwrap_or(Vec3::ZERO)
    }

    fn tangent_at(&self, t: f32) -> Vec3 {
        let scaled_t = t * self.segments.len() as f32;
        self.derivative(scaled_t).unwrap_or(Vec3::ZERO)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        if self.segments.is_empty() {
            return (Self::new(), Self::new());
        }

        let scaled_t = t * self.segments.len() as f32;
        let segment_idx = (scaled_t.floor() as usize).min(self.segments.len() - 1);
        let local_t = scaled_t - segment_idx as f32;

        let mut left_segments = self.segments[..segment_idx].to_vec();
        let mut right_segments = self.segments[segment_idx + 1..].to_vec();

        let (left_part, right_part) = self.segments[segment_idx].split(local_t);
        left_segments.push(left_part);
        right_segments.insert(0, right_part);

        (
            BezierSpline {
                segments: left_segments,
            },
            BezierSpline {
                segments: right_segments,
            },
        )
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec3>> {
        self.segments
            .iter()
            .map(|s| CurveCubic::new(s.p0, s.p1, s.p2, s.p3))
            .collect()
    }

    fn start(&self) -> Vec3 {
        self.segments.first().map(|s| s.p0).unwrap_or(Vec3::ZERO)
    }

    fn end(&self) -> Vec3 {
        self.segments.last().map(|s| s.p3).unwrap_or(Vec3::ZERO)
    }
}

// ============================================================================
// Nurbs<T> implements Curve for Vec2/Vec3
// ============================================================================

impl Curve for Nurbs<Vec2> {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        let (t_min, t_max) = self.domain();
        let mapped_t = t_min + t * (t_max - t_min);
        self.evaluate(mapped_t).unwrap_or(Vec2::ZERO)
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        let (t_min, t_max) = self.domain();
        let mapped_t = t_min + t * (t_max - t_min);
        self.derivative(mapped_t).unwrap_or(Vec2::ZERO)
    }

    fn split(&self, _t: f32) -> (Self, Self) {
        // NURBS splitting is complex - would need knot insertion
        // For now, return clones (lossy but functional)
        (self.clone(), self.clone())
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec2>> {
        // Sample the NURBS and create cubic approximations
        let samples = self.sample(33); // 32 segments
        if samples.len() < 2 {
            return vec![];
        }

        samples
            .windows(2)
            .map(|w| {
                let p0 = w[0];
                let p1 = w[1];
                // Degenerate cubic (straight line between samples)
                let third = (p1 - p0) * (1.0 / 3.0);
                CurveCubic::new(p0, p0 + third, p1 - third, p1)
            })
            .collect()
    }

    fn start(&self) -> Vec2 {
        self.points.first().map(|wp| wp.point).unwrap_or(Vec2::ZERO)
    }

    fn end(&self) -> Vec2 {
        self.points.last().map(|wp| wp.point).unwrap_or(Vec2::ZERO)
    }
}

impl Curve for Nurbs<Vec3> {
    type Point = Vec3;

    fn position_at(&self, t: f32) -> Vec3 {
        let (t_min, t_max) = self.domain();
        let mapped_t = t_min + t * (t_max - t_min);
        self.evaluate(mapped_t).unwrap_or(Vec3::ZERO)
    }

    fn tangent_at(&self, t: f32) -> Vec3 {
        let (t_min, t_max) = self.domain();
        let mapped_t = t_min + t * (t_max - t_min);
        self.derivative(mapped_t).unwrap_or(Vec3::ZERO)
    }

    fn split(&self, _t: f32) -> (Self, Self) {
        (self.clone(), self.clone())
    }

    fn to_cubics(&self) -> Vec<CurveCubic<Vec3>> {
        let samples = self.sample(33);
        if samples.len() < 2 {
            return vec![];
        }

        samples
            .windows(2)
            .map(|w| {
                let p0 = w[0];
                let p1 = w[1];
                let third = (p1 - p0) * (1.0 / 3.0);
                CurveCubic::new(p0, p0 + third, p1 - third, p1)
            })
            .collect()
    }

    fn start(&self) -> Vec3 {
        self.points.first().map(|wp| wp.point).unwrap_or(Vec3::ZERO)
    }

    fn end(&self) -> Vec3 {
        self.points.last().map(|wp| wp.point).unwrap_or(Vec3::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_bezier_curve_trait() {
        let bezier = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        );

        // Test via Curve trait
        assert!((bezier.position_at(0.0) - Vec2::ZERO).length() < 0.001);
        assert!((bezier.position_at(1.0) - Vec2::new(1.0, 0.0)).length() < 0.001);

        let tangent = bezier.tangent_at(0.0);
        assert!(tangent.length() > 0.0);
    }

    #[test]
    fn test_bezier_spline_curve_trait() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 0.0),
        ];
        let spline = BezierSpline::from_points(&points);

        assert!((spline.start() - Vec2::ZERO).length() < 0.001);
        assert!((spline.end() - Vec2::new(2.0, 0.0)).length() < 0.001);

        // Midpoint should be somewhere reasonable
        let mid = spline.position_at(0.5);
        assert!(mid.x > 0.5 && mid.x < 1.5);
    }

    #[test]
    fn test_nurbs_curve_trait() {
        use crate::WeightedPoint;

        let nurbs = Nurbs::quadratic(vec![
            WeightedPoint::new(Vec2::new(0.0, 0.0), 1.0),
            WeightedPoint::new(Vec2::new(1.0, 1.0), 1.0),
            WeightedPoint::new(Vec2::new(2.0, 0.0), 1.0),
        ]);

        assert!((nurbs.start() - Vec2::ZERO).length() < 0.001);
        assert!((nurbs.end() - Vec2::new(2.0, 0.0)).length() < 0.001);

        let cubics = nurbs.to_cubics();
        assert!(!cubics.is_empty());
    }
}
