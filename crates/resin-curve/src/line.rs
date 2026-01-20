//! Line segment.

use crate::{CubicBezier, Curve, VectorSpace, lerp};

/// A line segment from start to end.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "V: serde::Serialize",
        deserialize = "V: serde::de::DeserializeOwned"
    ))
)]
pub struct Line<V> {
    pub start: V,
    pub end: V,
}

impl<V> Line<V> {
    /// Creates a new line segment.
    pub fn new(start: V, end: V) -> Self {
        Self { start, end }
    }
}

impl<V: VectorSpace> Curve for Line<V> {
    type Point = V;

    #[inline]
    fn position_at(&self, t: f32) -> V {
        lerp(self.start, self.end, t)
    }

    #[inline]
    fn tangent_at(&self, _t: f32) -> V {
        self.end - self.start
    }

    fn split(&self, t: f32) -> (Self, Self) {
        let mid = self.position_at(t);
        (Line::new(self.start, mid), Line::new(mid, self.end))
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> {
        // Degenerate cubic with collinear control points
        vec![CubicBezier::new(
            self.start,
            lerp(self.start, self.end, 1.0 / 3.0),
            lerp(self.start, self.end, 2.0 / 3.0),
            self.end,
        )]
    }

    #[inline]
    fn start(&self) -> V {
        self.start
    }

    #[inline]
    fn end(&self) -> V {
        self.end
    }

    #[inline]
    fn length(&self) -> f32 {
        (self.end - self.start).length()
    }

    fn flatten(&self, _tolerance: f32) -> Vec<V> {
        vec![self.start, self.end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_line_position() {
        let line = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));

        assert_eq!(line.position_at(0.0), Vec2::ZERO);
        assert_eq!(line.position_at(1.0), Vec2::new(10.0, 0.0));
        assert_eq!(line.position_at(0.5), Vec2::new(5.0, 0.0));
    }

    #[test]
    fn test_line_length() {
        let line = Line::new(Vec2::ZERO, Vec2::new(3.0, 4.0));
        assert!((line.length() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_line_split() {
        let line = Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let (left, right) = line.split(0.5);

        assert_eq!(left.start, Vec2::ZERO);
        assert_eq!(left.end, Vec2::new(5.0, 0.0));
        assert_eq!(right.start, Vec2::new(5.0, 0.0));
        assert_eq!(right.end, Vec2::new(10.0, 0.0));
    }
}
