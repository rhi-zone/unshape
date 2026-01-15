//! Segment enums for mixed-type paths.

use crate::{Aabb, Arc, CubicBezier, Curve, Curve2DExt, Curve3DExt, Line, QuadBezier, Rect};
use glam::{Vec2, Vec3};

/// A 2D path segment (line, quadratic, cubic, or arc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment2D {
    Line(Line<Vec2>),
    Quad(QuadBezier<Vec2>),
    Cubic(CubicBezier<Vec2>),
    Arc(Arc),
}

impl From<Line<Vec2>> for Segment2D {
    fn from(line: Line<Vec2>) -> Self {
        Segment2D::Line(line)
    }
}

impl From<QuadBezier<Vec2>> for Segment2D {
    fn from(quad: QuadBezier<Vec2>) -> Self {
        Segment2D::Quad(quad)
    }
}

impl From<CubicBezier<Vec2>> for Segment2D {
    fn from(cubic: CubicBezier<Vec2>) -> Self {
        Segment2D::Cubic(cubic)
    }
}

impl From<Arc> for Segment2D {
    fn from(arc: Arc) -> Self {
        Segment2D::Arc(arc)
    }
}

impl Curve for Segment2D {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        match self {
            Segment2D::Line(c) => c.position_at(t),
            Segment2D::Quad(c) => c.position_at(t),
            Segment2D::Cubic(c) => c.position_at(t),
            Segment2D::Arc(c) => c.position_at(t),
        }
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        match self {
            Segment2D::Line(c) => c.tangent_at(t),
            Segment2D::Quad(c) => c.tangent_at(t),
            Segment2D::Cubic(c) => c.tangent_at(t),
            Segment2D::Arc(c) => c.tangent_at(t),
        }
    }

    fn split(&self, t: f32) -> (Self, Self) {
        match self {
            Segment2D::Line(c) => {
                let (a, b) = c.split(t);
                (Segment2D::Line(a), Segment2D::Line(b))
            }
            Segment2D::Quad(c) => {
                let (a, b) = c.split(t);
                (Segment2D::Quad(a), Segment2D::Quad(b))
            }
            Segment2D::Cubic(c) => {
                let (a, b) = c.split(t);
                (Segment2D::Cubic(a), Segment2D::Cubic(b))
            }
            Segment2D::Arc(c) => {
                let (a, b) = c.split(t);
                (Segment2D::Arc(a), Segment2D::Arc(b))
            }
        }
    }

    fn to_cubics(&self) -> Vec<CubicBezier<Vec2>> {
        match self {
            Segment2D::Line(c) => c.to_cubics(),
            Segment2D::Quad(c) => c.to_cubics(),
            Segment2D::Cubic(c) => c.to_cubics(),
            Segment2D::Arc(c) => c.to_cubics(),
        }
    }

    fn start(&self) -> Vec2 {
        match self {
            Segment2D::Line(c) => c.start(),
            Segment2D::Quad(c) => c.start(),
            Segment2D::Cubic(c) => c.start(),
            Segment2D::Arc(c) => c.start(),
        }
    }

    fn end(&self) -> Vec2 {
        match self {
            Segment2D::Line(c) => c.end(),
            Segment2D::Quad(c) => c.end(),
            Segment2D::Cubic(c) => c.end(),
            Segment2D::Arc(c) => c.end(),
        }
    }

    fn length(&self) -> f32 {
        match self {
            Segment2D::Line(c) => c.length(),
            Segment2D::Quad(c) => c.length(),
            Segment2D::Cubic(c) => c.length(),
            Segment2D::Arc(c) => c.length(),
        }
    }
}

impl Curve2DExt for Segment2D {
    fn bounding_box(&self) -> Rect {
        // Sample points and compute bounds (could be more precise for beziers)
        let points: Vec<Vec2> = self.flatten(0.1);
        Rect::from_points(points).unwrap_or(Rect::new(self.start(), self.start()))
    }
}

/// A 3D path segment (line, quadratic, or cubic).
///
/// No Arc variant - use NURBS for 3D curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment3D {
    Line(Line<Vec3>),
    Quad(QuadBezier<Vec3>),
    Cubic(CubicBezier<Vec3>),
}

impl From<Line<Vec3>> for Segment3D {
    fn from(line: Line<Vec3>) -> Self {
        Segment3D::Line(line)
    }
}

impl From<QuadBezier<Vec3>> for Segment3D {
    fn from(quad: QuadBezier<Vec3>) -> Self {
        Segment3D::Quad(quad)
    }
}

impl From<CubicBezier<Vec3>> for Segment3D {
    fn from(cubic: CubicBezier<Vec3>) -> Self {
        Segment3D::Cubic(cubic)
    }
}

impl Curve for Segment3D {
    type Point = Vec3;

    fn position_at(&self, t: f32) -> Vec3 {
        match self {
            Segment3D::Line(c) => c.position_at(t),
            Segment3D::Quad(c) => c.position_at(t),
            Segment3D::Cubic(c) => c.position_at(t),
        }
    }

    fn tangent_at(&self, t: f32) -> Vec3 {
        match self {
            Segment3D::Line(c) => c.tangent_at(t),
            Segment3D::Quad(c) => c.tangent_at(t),
            Segment3D::Cubic(c) => c.tangent_at(t),
        }
    }

    fn split(&self, t: f32) -> (Self, Self) {
        match self {
            Segment3D::Line(c) => {
                let (a, b) = c.split(t);
                (Segment3D::Line(a), Segment3D::Line(b))
            }
            Segment3D::Quad(c) => {
                let (a, b) = c.split(t);
                (Segment3D::Quad(a), Segment3D::Quad(b))
            }
            Segment3D::Cubic(c) => {
                let (a, b) = c.split(t);
                (Segment3D::Cubic(a), Segment3D::Cubic(b))
            }
        }
    }

    fn to_cubics(&self) -> Vec<CubicBezier<Vec3>> {
        match self {
            Segment3D::Line(c) => c.to_cubics(),
            Segment3D::Quad(c) => c.to_cubics(),
            Segment3D::Cubic(c) => c.to_cubics(),
        }
    }

    fn start(&self) -> Vec3 {
        match self {
            Segment3D::Line(c) => c.start(),
            Segment3D::Quad(c) => c.start(),
            Segment3D::Cubic(c) => c.start(),
        }
    }

    fn end(&self) -> Vec3 {
        match self {
            Segment3D::Line(c) => c.end(),
            Segment3D::Quad(c) => c.end(),
            Segment3D::Cubic(c) => c.end(),
        }
    }

    fn length(&self) -> f32 {
        match self {
            Segment3D::Line(c) => c.length(),
            Segment3D::Quad(c) => c.length(),
            Segment3D::Cubic(c) => c.length(),
        }
    }
}

impl Curve3DExt for Segment3D {
    fn bounding_box(&self) -> Aabb {
        let points: Vec<Vec3> = self.flatten(0.1);
        Aabb::from_points(points).unwrap_or(Aabb::new(self.start(), self.start()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment2d_from_line() {
        let line = Line::new(Vec2::ZERO, Vec2::new(1.0, 1.0));
        let seg: Segment2D = line.into();

        assert_eq!(seg.start(), Vec2::ZERO);
        assert_eq!(seg.end(), Vec2::new(1.0, 1.0));
    }

    #[test]
    fn test_segment2d_split() {
        let seg = Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0)));
        let (left, right) = seg.split(0.5);

        assert_eq!(left.end(), Vec2::new(5.0, 0.0));
        assert_eq!(right.start(), Vec2::new(5.0, 0.0));
    }

    #[test]
    fn test_segment3d_cubic() {
        let cubic = CubicBezier::new(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let seg: Segment3D = cubic.into();

        assert_eq!(seg.start(), Vec3::ZERO);
        assert_eq!(seg.end(), Vec3::new(1.0, 0.0, 0.0));
    }
}
