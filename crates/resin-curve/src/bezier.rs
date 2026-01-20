//! Bézier curves.

use crate::{Curve, VectorSpace, lerp};

/// A quadratic Bézier curve.
///
/// Defined by 3 control points: start (p0), control (p1), end (p2).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "V: serde::Serialize",
        deserialize = "V: serde::de::DeserializeOwned"
    ))
)]
pub struct QuadBezier<V> {
    pub p0: V,
    pub p1: V,
    pub p2: V,
}

impl<V> QuadBezier<V> {
    /// Creates a new quadratic Bézier curve.
    pub fn new(p0: V, p1: V, p2: V) -> Self {
        Self { p0, p1, p2 }
    }
}

impl<V: VectorSpace> QuadBezier<V> {
    /// Elevates to a cubic Bézier (exact conversion).
    pub fn elevate(&self) -> CubicBezier<V> {
        // Degree elevation formulas:
        // C0 = Q0
        // C1 = Q0 + 2/3 * (Q1 - Q0) = 1/3 * Q0 + 2/3 * Q1
        // C2 = Q2 + 2/3 * (Q1 - Q2) = 2/3 * Q1 + 1/3 * Q2
        // C3 = Q2
        CubicBezier::new(
            self.p0,
            lerp(self.p0, self.p1, 2.0 / 3.0),
            lerp(self.p1, self.p2, 1.0 / 3.0),
            self.p2,
        )
    }
}

impl<V: VectorSpace> Curve for QuadBezier<V> {
    type Point = V;

    fn position_at(&self, t: f32) -> V {
        let mt = 1.0 - t;
        // B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        self.p0 * (mt * mt) + self.p1 * (2.0 * mt * t) + self.p2 * (t * t)
    }

    fn tangent_at(&self, t: f32) -> V {
        let mt = 1.0 - t;
        // B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        (self.p1 - self.p0) * (2.0 * mt) + (self.p2 - self.p1) * (2.0 * t)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        // De Casteljau's algorithm
        let p01 = lerp(self.p0, self.p1, t);
        let p12 = lerp(self.p1, self.p2, t);
        let p012 = lerp(p01, p12, t);

        (
            QuadBezier::new(self.p0, p01, p012),
            QuadBezier::new(p012, p12, self.p2),
        )
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> {
        vec![self.elevate()]
    }

    fn start(&self) -> V {
        self.p0
    }

    fn end(&self) -> V {
        self.p2
    }
}

/// A cubic Bézier curve.
///
/// Defined by 4 control points: start (p0), control 1 (p1), control 2 (p2), end (p3).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "V: serde::Serialize",
        deserialize = "V: serde::de::DeserializeOwned"
    ))
)]
pub struct CubicBezier<V> {
    pub p0: V,
    pub p1: V,
    pub p2: V,
    pub p3: V,
}

impl<V> CubicBezier<V> {
    /// Creates a new cubic Bézier curve.
    pub fn new(p0: V, p1: V, p2: V, p3: V) -> Self {
        Self { p0, p1, p2, p3 }
    }
}

impl<V: VectorSpace> Curve for CubicBezier<V> {
    type Point = V;

    fn position_at(&self, t: f32) -> V {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        // B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        self.p0 * mt3 + self.p1 * (3.0 * mt2 * t) + self.p2 * (3.0 * mt * t2) + self.p3 * t3
    }

    fn tangent_at(&self, t: f32) -> V {
        let t2 = t * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;

        // B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
        (self.p1 - self.p0) * (3.0 * mt2)
            + (self.p2 - self.p1) * (6.0 * mt * t)
            + (self.p3 - self.p2) * (3.0 * t2)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        // De Casteljau's algorithm
        let p01 = lerp(self.p0, self.p1, t);
        let p12 = lerp(self.p1, self.p2, t);
        let p23 = lerp(self.p2, self.p3, t);

        let p012 = lerp(p01, p12, t);
        let p123 = lerp(p12, p23, t);

        let p0123 = lerp(p012, p123, t);

        (
            CubicBezier::new(self.p0, p01, p012, p0123),
            CubicBezier::new(p0123, p123, p23, self.p3),
        )
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> {
        vec![*self]
    }

    fn start(&self) -> V {
        self.p0
    }

    fn end(&self) -> V {
        self.p3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    #[test]
    fn test_quad_bezier_endpoints() {
        let curve = QuadBezier::new(Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0));

        assert!((curve.position_at(0.0) - Vec2::ZERO).length() < 0.001);
        assert!((curve.position_at(1.0) - Vec2::new(2.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_quad_bezier_split() {
        let curve = QuadBezier::new(Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0));
        let (left, right) = curve.split(0.5);

        let split_point = curve.position_at(0.5);
        assert!((left.position_at(1.0) - split_point).length() < 0.001);
        assert!((right.position_at(0.0) - split_point).length() < 0.001);
    }

    #[test]
    fn test_quad_elevate() {
        let quad = QuadBezier::new(Vec2::ZERO, Vec2::new(1.0, 2.0), Vec2::new(2.0, 0.0));
        let cubic = quad.elevate();

        // Elevated cubic should match quad at several points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let q = quad.position_at(t);
            let c = cubic.position_at(t);
            assert!(
                (q - c).length() < 0.001,
                "Mismatch at t={}: {:?} vs {:?}",
                t,
                q,
                c
            );
        }
    }

    #[test]
    fn test_cubic_bezier_endpoints() {
        let curve = CubicBezier::new(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        assert!((curve.position_at(0.0) - Vec3::ZERO).length() < 0.001);
        assert!((curve.position_at(1.0) - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);
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
        let split_point = curve.position_at(0.5);

        assert!((left.position_at(1.0) - split_point).length() < 0.001);
        assert!((right.position_at(0.0) - split_point).length() < 0.001);
    }

    #[test]
    fn test_cubic_bezier_tangent() {
        let curve = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );

        // Straight line, tangent should be constant
        let t0 = curve.tangent_at(0.0);
        let t1 = curve.tangent_at(0.5);
        let t2 = curve.tangent_at(1.0);

        // All tangents should point in same direction
        assert!(t0.normalize().dot(t1.normalize()) > 0.999);
        assert!(t1.normalize().dot(t2.normalize()) > 0.999);
    }

    #[test]
    fn test_cubic_bezier_length() {
        // Straight line
        let curve = CubicBezier::new(
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        );

        let len = curve.length();
        assert!((len - 3.0).abs() < 0.1, "Expected ~3.0, got {}", len);
    }
}
