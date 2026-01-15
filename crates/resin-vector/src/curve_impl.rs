//! Integration with resin-curve traits.

use crate::{CurveSegment, Path, PathCommand};
use glam::Vec2;
use rhizome_resin_curve::{CubicBezier, Curve, Line, Path as CurvePath, QuadBezier, Segment2D};

// ============================================================================
// CurveSegment implements Curve
// ============================================================================

impl Curve for CurveSegment {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        self.evaluate(t)
    }

    fn tangent_at(&self, t: f32) -> Vec2 {
        self.derivative(t)
    }

    fn split(&self, t: f32) -> (Self, Self) {
        let result = CurveSegment::split(self, t);
        (result.before, result.after)
    }

    fn to_cubics(&self) -> Vec<CubicBezier<Vec2>> {
        match self {
            CurveSegment::Line { start, end } => {
                let third = (*end - *start) * (1.0 / 3.0);
                vec![CubicBezier::new(*start, *start + third, *end - third, *end)]
            }
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => {
                // Degree elevation
                let c1 = *start + (*control - *start) * (2.0 / 3.0);
                let c2 = *end + (*control - *end) * (2.0 / 3.0);
                vec![CubicBezier::new(*start, c1, c2, *end)]
            }
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => vec![CubicBezier::new(*start, *control1, *control2, *end)],
        }
    }

    fn start(&self) -> Vec2 {
        match self {
            CurveSegment::Line { start, .. } => *start,
            CurveSegment::Quadratic { start, .. } => *start,
            CurveSegment::Cubic { start, .. } => *start,
        }
    }

    fn end(&self) -> Vec2 {
        match self {
            CurveSegment::Line { end, .. } => *end,
            CurveSegment::Quadratic { end, .. } => *end,
            CurveSegment::Cubic { end, .. } => *end,
        }
    }
}

// ============================================================================
// Conversions between CurveSegment and Segment2D
// ============================================================================

impl From<CurveSegment> for Segment2D {
    fn from(seg: CurveSegment) -> Self {
        match seg {
            CurveSegment::Line { start, end } => Segment2D::Line(Line::new(start, end)),
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => Segment2D::Quad(QuadBezier::new(start, control, end)),
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => Segment2D::Cubic(CubicBezier::new(start, control1, control2, end)),
        }
    }
}

impl From<Segment2D> for CurveSegment {
    fn from(seg: Segment2D) -> Self {
        match seg {
            Segment2D::Line(l) => CurveSegment::Line {
                start: l.start,
                end: l.end,
            },
            Segment2D::Quad(q) => CurveSegment::Quadratic {
                start: q.p0,
                control: q.p1,
                end: q.p2,
            },
            Segment2D::Cubic(c) => CurveSegment::Cubic {
                start: c.p0,
                control1: c.p1,
                control2: c.p2,
                end: c.p3,
            },
            Segment2D::Arc(arc) => {
                // Convert arc to cubic approximation, take first segment
                let cubics = arc.to_cubics();
                if let Some(c) = cubics.first() {
                    CurveSegment::Cubic {
                        start: c.p0,
                        control1: c.p1,
                        control2: c.p2,
                        end: c.p3,
                    }
                } else {
                    // Fallback to line
                    CurveSegment::Line {
                        start: arc.position_at(0.0),
                        end: arc.position_at(1.0),
                    }
                }
            }
        }
    }
}

// ============================================================================
// Path to segments conversion
// ============================================================================

impl Path {
    /// Converts the path to a vector of Segment2D.
    ///
    /// MoveTo commands start new subpaths. Close commands are converted to
    /// line segments back to the subpath start.
    pub fn to_segments(&self) -> Vec<Segment2D> {
        let mut segments = Vec::new();
        let mut current_pos = Vec2::ZERO;
        let mut subpath_start = Vec2::ZERO;

        for cmd in self.commands() {
            match *cmd {
                PathCommand::MoveTo(p) => {
                    current_pos = p;
                    subpath_start = p;
                }
                PathCommand::LineTo(p) => {
                    if current_pos != p {
                        segments.push(Segment2D::Line(Line::new(current_pos, p)));
                    }
                    current_pos = p;
                }
                PathCommand::QuadTo { control, to } => {
                    segments.push(Segment2D::Quad(QuadBezier::new(current_pos, control, to)));
                    current_pos = to;
                }
                PathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    segments.push(Segment2D::Cubic(CubicBezier::new(
                        current_pos,
                        control1,
                        control2,
                        to,
                    )));
                    current_pos = to;
                }
                PathCommand::Close => {
                    if current_pos != subpath_start {
                        segments.push(Segment2D::Line(Line::new(current_pos, subpath_start)));
                    }
                    current_pos = subpath_start;
                }
            }
        }

        segments
    }

    /// Converts the path to a CurvePath (segment-based path from resin-curve).
    pub fn to_curve_path(&self) -> CurvePath<Segment2D> {
        let segments = self.to_segments();
        let closed = self
            .commands()
            .last()
            .is_some_and(|c| matches!(c, PathCommand::Close));
        if closed {
            CurvePath::closed(segments)
        } else {
            CurvePath::from_segments(segments)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PathBuilder;

    #[test]
    fn test_curve_segment_to_segment2d() {
        let cs = CurveSegment::Line {
            start: Vec2::ZERO,
            end: Vec2::new(1.0, 1.0),
        };
        let s2d: Segment2D = cs.into();

        assert_eq!(s2d.start(), Vec2::ZERO);
        assert_eq!(s2d.end(), Vec2::new(1.0, 1.0));
    }

    #[test]
    fn test_path_to_segments() {
        let path = PathBuilder::new()
            .move_to(Vec2::ZERO)
            .line_to(Vec2::new(1.0, 0.0))
            .line_to(Vec2::new(1.0, 1.0))
            .close()
            .build();

        let segments = path.to_segments();
        assert_eq!(segments.len(), 3); // Two lines + closing line
    }

    #[test]
    fn test_path_to_curve_path() {
        let path = PathBuilder::new()
            .move_to(Vec2::ZERO)
            .line_to(Vec2::new(1.0, 0.0))
            .build();

        let curve_path = path.to_curve_path();
        assert_eq!(curve_path.len(), 1);
        assert!(!curve_path.closed);
    }

    #[test]
    fn test_curve_segment_curve_trait() {
        let seg = CurveSegment::Cubic {
            start: Vec2::ZERO,
            control1: Vec2::new(0.0, 1.0),
            control2: Vec2::new(1.0, 1.0),
            end: Vec2::new(1.0, 0.0),
        };

        // Test via Curve trait
        assert!((seg.position_at(0.0) - Vec2::ZERO).length() < 0.001);
        assert!((seg.position_at(1.0) - Vec2::new(1.0, 0.0)).length() < 0.001);

        let tangent = seg.tangent_at(0.5);
        assert!(tangent.length() > 0.0);
    }
}
