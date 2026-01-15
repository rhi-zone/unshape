//! Elliptical arc (2D only).

use crate::{CubicBezier, Curve};
use glam::Vec2;
use std::f32::consts::{FRAC_PI_2, PI};

/// A 2D elliptical arc.
///
/// Arcs are 2D-only because 3D arcs are better represented as NURBS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    /// Center of the ellipse.
    pub center: Vec2,
    /// Radii (rx, ry). For circles, rx == ry.
    pub radii: Vec2,
    /// Start angle in radians.
    pub start_angle: f32,
    /// Sweep angle in radians. Positive = CCW, negative = CW.
    pub sweep: f32,
    /// X-axis rotation in radians (for rotated ellipses).
    pub rotation: f32,
}

impl Arc {
    /// Creates a new arc.
    pub fn new(center: Vec2, radii: Vec2, start_angle: f32, sweep: f32) -> Self {
        Self {
            center,
            radii,
            start_angle,
            sweep,
            rotation: 0.0,
        }
    }

    /// Creates a circular arc.
    pub fn circle(center: Vec2, radius: f32, start_angle: f32, sweep: f32) -> Self {
        Self::new(center, Vec2::splat(radius), start_angle, sweep)
    }

    /// Creates a full circle.
    pub fn full_circle(center: Vec2, radius: f32) -> Self {
        Self::circle(center, radius, 0.0, 2.0 * PI)
    }

    /// Creates an arc with rotation.
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns the end angle.
    pub fn end_angle(&self) -> f32 {
        self.start_angle + self.sweep
    }

    /// Rotates a point by the arc's rotation angle.
    fn rotate_point(&self, p: Vec2) -> Vec2 {
        if self.rotation.abs() < 1e-10 {
            p
        } else {
            let cos_r = self.rotation.cos();
            let sin_r = self.rotation.sin();
            Vec2::new(p.x * cos_r - p.y * sin_r, p.x * sin_r + p.y * cos_r)
        }
    }
}

impl Curve for Arc {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        let angle = self.start_angle + self.sweep * t;
        let p = Vec2::new(self.radii.x * angle.cos(), self.radii.y * angle.sin());
        self.rotate_point(p) + self.center
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        let angle = self.start_angle + self.sweep * t;
        // Derivative of position with respect to t
        // d/dt [rx * cos(start + sweep * t), ry * sin(start + sweep * t)]
        // = [-rx * sweep * sin(angle), ry * sweep * cos(angle)]
        let p = Vec2::new(
            -self.radii.x * self.sweep * angle.sin(),
            self.radii.y * self.sweep * angle.cos(),
        );
        self.rotate_point(p)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        let mid_angle = self.start_angle + self.sweep * t;
        let sweep1 = self.sweep * t;
        let sweep2 = self.sweep * (1.0 - t);

        (
            Arc {
                center: self.center,
                radii: self.radii,
                start_angle: self.start_angle,
                sweep: sweep1,
                rotation: self.rotation,
            },
            Arc {
                center: self.center,
                radii: self.radii,
                start_angle: mid_angle,
                sweep: sweep2,
                rotation: self.rotation,
            },
        )
    }

    fn to_cubics(&self) -> Vec<CubicBezier<Vec2>> {
        arc_to_cubics(self)
    }

    fn length(&self) -> f32 {
        // For circles, exact formula
        if (self.radii.x - self.radii.y).abs() < 1e-10 {
            return self.radii.x * self.sweep.abs();
        }

        // For ellipses, use numerical approximation from trait default
        // (Ramanujan's approximation could be used for full ellipses)
        let default_len: f32 = {
            const WEIGHTS: [f32; 5] = [0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269];
            const POINTS: [f32; 5] = [0.0469101, 0.2307653, 0.5, 0.7692347, 0.9530899];
            WEIGHTS
                .iter()
                .zip(POINTS.iter())
                .map(|(w, t)| w * self.tangent_at(*t).length())
                .sum()
        };
        default_len
    }
}

/// Converts an arc to cubic Bézier approximations.
///
/// Each cubic handles up to 90° accurately. Larger arcs are split.
fn arc_to_cubics(arc: &Arc) -> Vec<CubicBezier<Vec2>> {
    let sweep = arc.sweep.abs();
    if sweep < 1e-10 {
        return vec![];
    }

    // Number of segments needed (each handles up to 90°)
    let num_segments = ((sweep / FRAC_PI_2).ceil() as usize).max(1);

    let mut cubics = Vec::with_capacity(num_segments);

    for i in 0..num_segments {
        let t0 = i as f32 / num_segments as f32;
        let t1 = (i + 1) as f32 / num_segments as f32;

        let angle0 = arc.start_angle + arc.sweep * t0;
        let angle1 = arc.start_angle + arc.sweep * t1;

        let cubic = arc_segment_to_cubic(arc.center, arc.radii, arc.rotation, angle0, angle1);
        cubics.push(cubic);
    }

    cubics
}

/// Converts a single arc segment (up to 90°) to a cubic Bézier.
fn arc_segment_to_cubic(
    center: Vec2,
    radii: Vec2,
    rotation: f32,
    angle0: f32,
    angle1: f32,
) -> CubicBezier<Vec2> {
    let sweep = angle1 - angle0;
    let half_sweep = sweep / 2.0;

    // Magic number for cubic Bézier approximation of circular arc
    // k = 4/3 * tan(sweep/4) for optimal approximation
    let k = (4.0 / 3.0) * (half_sweep / 2.0).tan();

    // Points on ellipse
    let cos0 = angle0.cos();
    let sin0 = angle0.sin();
    let cos1 = angle1.cos();
    let sin1 = angle1.sin();

    // Start and end points (before rotation)
    let p0 = Vec2::new(radii.x * cos0, radii.y * sin0);
    let p3 = Vec2::new(radii.x * cos1, radii.y * sin1);

    // Tangent directions (scaled by k)
    let t0 = Vec2::new(-radii.x * sin0, radii.y * cos0) * k;
    let t1 = Vec2::new(-radii.x * sin1, radii.y * cos1) * k;

    // Control points
    let p1 = p0 + t0;
    let p2 = p3 - t1;

    // Apply rotation and translate to center
    let rotate = |p: Vec2| -> Vec2 {
        if rotation.abs() < 1e-10 {
            p + center
        } else {
            let cos_r = rotation.cos();
            let sin_r = rotation.sin();
            Vec2::new(p.x * cos_r - p.y * sin_r, p.x * sin_r + p.y * cos_r) + center
        }
    };

    CubicBezier::new(rotate(p0), rotate(p1), rotate(p2), rotate(p3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_endpoints() {
        let arc = Arc::circle(Vec2::ZERO, 1.0, 0.0, FRAC_PI_2);

        let start = arc.position_at(0.0);
        let end = arc.position_at(1.0);

        assert!((start - Vec2::new(1.0, 0.0)).length() < 0.001);
        assert!((end - Vec2::new(0.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn test_arc_midpoint() {
        let arc = Arc::circle(Vec2::ZERO, 1.0, 0.0, FRAC_PI_2);

        let mid = arc.position_at(0.5);
        let expected_angle = FRAC_PI_2 / 2.0; // 45 degrees
        let expected = Vec2::new(expected_angle.cos(), expected_angle.sin());

        assert!((mid - expected).length() < 0.001);
    }

    #[test]
    fn test_arc_on_circle() {
        let arc = Arc::circle(Vec2::ZERO, 2.0, 0.0, PI);

        // All points should be at distance 2 from center
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = arc.position_at(t);
            let dist = p.length();
            assert!(
                (dist - 2.0).abs() < 0.001,
                "Point at t={} has dist {}",
                t,
                dist
            );
        }
    }

    #[test]
    fn test_arc_length_circle() {
        let arc = Arc::circle(Vec2::ZERO, 1.0, 0.0, PI);

        // Half circle should have length π
        let len = arc.length();
        assert!((len - PI).abs() < 0.01, "Expected π, got {}", len);
    }

    #[test]
    fn test_arc_to_cubics_quarter() {
        let arc = Arc::circle(Vec2::ZERO, 1.0, 0.0, FRAC_PI_2);
        let cubics = arc.to_cubics();

        assert_eq!(cubics.len(), 1);

        // Cubic should approximate arc well
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let arc_point = arc.position_at(t);
            let cubic_point = cubics[0].position_at(t);
            let error = (arc_point - cubic_point).length();
            assert!(error < 0.01, "Error at t={}: {}", t, error);
        }
    }

    #[test]
    fn test_arc_to_cubics_full_circle() {
        let arc = Arc::full_circle(Vec2::ZERO, 1.0);
        let cubics = arc.to_cubics();

        // Full circle needs 4 segments
        assert_eq!(cubics.len(), 4);

        // All cubic points should be on circle
        for (i, cubic) in cubics.iter().enumerate() {
            for t in [0.0, 0.5, 1.0] {
                let p = cubic.position_at(t);
                let dist = p.length();
                assert!(
                    (dist - 1.0).abs() < 0.01,
                    "Segment {} at t={}: dist = {}",
                    i,
                    t,
                    dist
                );
            }
        }
    }

    #[test]
    fn test_arc_split() {
        let arc = Arc::circle(Vec2::ZERO, 1.0, 0.0, PI);
        let (left, right) = arc.split(0.5);

        // Split point should match
        let mid = arc.position_at(0.5);
        assert!((left.position_at(1.0) - mid).length() < 0.001);
        assert!((right.position_at(0.0) - mid).length() < 0.001);

        // Sweeps should add up
        assert!((left.sweep + right.sweep - arc.sweep).abs() < 0.001);
    }
}
