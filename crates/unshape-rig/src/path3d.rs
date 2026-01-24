//! 3D path representation using resin-curve types.
//!
//! This module provides convenient 3D path building on top of `unshape-curve`'s
//! unified curve types. Paths are represented as `ArcLengthPath<Path<Segment3D>>`
//! for uniform-speed sampling along the path.

use glam::Vec3;
use unshape_curve::{ArcLengthPath, CubicBezier, Line, Path, Segment3D};

/// A 3D path with arc-length parameterization.
///
/// This is a type alias for the underlying unified curve types.
/// Internally stores a `Path<Segment3D>` with cached arc lengths.
pub type Path3D = ArcLengthPath<Segment3D>;

/// Result of sampling a path.
#[derive(Debug, Clone, Copy)]
pub struct PathSample {
    /// Position on the path.
    pub position: Vec3,
    /// Tangent direction (normalized).
    pub tangent: Vec3,
}

/// Extension trait for Path3D to add convenience methods.
pub trait Path3DExt {
    /// Samples position and tangent at t ∈ [0, 1].
    fn sample_at(&self, t: f32) -> PathSample;
}

impl Path3DExt for Path3D {
    fn sample_at(&self, t: f32) -> PathSample {
        let position = self.position_at(t).unwrap_or(Vec3::ZERO);
        let tangent = self
            .tangent_at(t)
            .map(|v| v.normalize_or_zero())
            .unwrap_or(Vec3::X);
        PathSample { position, tangent }
    }
}

/// Builder for constructing 3D paths.
#[derive(Debug, Clone, Default)]
pub struct Path3DBuilder {
    segments: Vec<Segment3D>,
    current: Vec3,
}

impl Path3DBuilder {
    /// Creates a new path builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves to a point without drawing.
    pub fn move_to(mut self, to: Vec3) -> Self {
        self.current = to;
        self
    }

    /// Draws a line to a point.
    pub fn line_to(mut self, to: Vec3) -> Self {
        self.segments
            .push(Segment3D::Line(Line::new(self.current, to)));
        self.current = to;
        self
    }

    /// Draws a cubic bezier curve.
    pub fn cubic_to(mut self, control1: Vec3, control2: Vec3, to: Vec3) -> Self {
        self.segments.push(Segment3D::Cubic(CubicBezier::new(
            self.current,
            control1,
            control2,
            to,
        )));
        self.current = to;
        self
    }

    /// Builds the final path.
    pub fn build(self) -> Path3D {
        let path = Path::from_segments(self.segments);
        ArcLengthPath::new(path)
    }
}

/// Creates a straight line path.
pub fn line3d(from: Vec3, to: Vec3) -> Path3D {
    Path3DBuilder::new().move_to(from).line_to(to).build()
}

/// Creates a polyline path.
pub fn polyline3d(points: &[Vec3]) -> Path3D {
    if points.is_empty() {
        return Path3DBuilder::new().build();
    }

    let mut builder = Path3DBuilder::new().move_to(points[0]);
    for &p in &points[1..] {
        builder = builder.line_to(p);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_path() {
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));

        assert!((path.total_length() - 10.0).abs() < 0.01);
        assert!((path.position_at(0.0).unwrap() - Vec3::ZERO).length() < 0.001);
        assert!((path.position_at(0.5).unwrap().x - 5.0).abs() < 0.01);
        assert!((path.position_at(1.0).unwrap().x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_line_tangent() {
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));

        let tangent = path.tangent_at(0.5).unwrap().normalize();
        assert!((tangent.x - 1.0).abs() < 0.001);
        assert!(tangent.y.abs() < 0.001);
    }

    #[test]
    fn test_polyline() {
        let path = polyline3d(&[
            Vec3::ZERO,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
        ]);

        // Total length should be 10
        assert!((path.total_length() - 10.0).abs() < 0.01);

        // Midpoint should be at the corner
        let mid = path.position_at(0.5).unwrap();
        assert!((mid.x - 5.0).abs() < 0.01);
        assert!(mid.y.abs() < 0.01);
    }

    #[test]
    fn test_cubic_path() {
        let path = Path3DBuilder::new()
            .move_to(Vec3::ZERO)
            .cubic_to(
                Vec3::new(0.0, 5.0, 0.0),
                Vec3::new(10.0, 5.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
            )
            .build();

        // Start and end positions
        assert!((path.position_at(0.0).unwrap() - Vec3::ZERO).length() < 0.001);
        let end = path.position_at(1.0).unwrap();
        assert!((end.x - 10.0).abs() < 0.01);
        assert!(end.y.abs() < 0.01);
    }

    #[test]
    fn test_sample_at() {
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let sample = path.sample_at(0.5);

        assert!((sample.position.x - 5.0).abs() < 0.01);
        assert!((sample.tangent.x - 1.0).abs() < 0.001);
    }
}
