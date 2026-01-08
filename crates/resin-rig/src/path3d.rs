//! 3D path representation with sampling.

use glam::Vec3;

/// A command in a 3D path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCommand3D {
    /// Move to a point without drawing.
    MoveTo(Vec3),
    /// Draw a line to a point.
    LineTo(Vec3),
    /// Cubic bezier curve to a point with two control points.
    CubicTo {
        control1: Vec3,
        control2: Vec3,
        to: Vec3,
    },
}

/// A segment of a path (for sampling).
#[derive(Debug, Clone, Copy)]
enum Segment {
    Line {
        from: Vec3,
        to: Vec3,
    },
    Cubic {
        from: Vec3,
        c1: Vec3,
        c2: Vec3,
        to: Vec3,
    },
}

impl Segment {
    fn length_estimate(&self) -> f32 {
        match self {
            Segment::Line { from, to } => (*to - *from).length(),
            Segment::Cubic { from, c1, c2, to } => {
                // Approximate with chord + control polygon average
                let chord = (*to - *from).length();
                let control_poly =
                    (*c1 - *from).length() + (*c2 - *c1).length() + (*to - *c2).length();
                (chord + control_poly) / 2.0
            }
        }
    }

    fn position_at(&self, t: f32) -> Vec3 {
        match self {
            Segment::Line { from, to } => from.lerp(*to, t),
            Segment::Cubic { from, c1, c2, to } => {
                // De Casteljau's algorithm
                let t2 = t * t;
                let t3 = t2 * t;
                let mt = 1.0 - t;
                let mt2 = mt * mt;
                let mt3 = mt2 * mt;
                *from * mt3 + *c1 * (3.0 * mt2 * t) + *c2 * (3.0 * mt * t2) + *to * t3
            }
        }
    }

    fn tangent_at(&self, t: f32) -> Vec3 {
        match self {
            Segment::Line { from, to } => (*to - *from).normalize_or_zero(),
            Segment::Cubic { from, c1, c2, to } => {
                // Derivative of cubic bezier
                let t2 = t * t;
                let mt = 1.0 - t;
                let mt2 = mt * mt;
                let d = (*c1 - *from) * (3.0 * mt2)
                    + (*c2 - *c1) * (6.0 * mt * t)
                    + (*to - *c2) * (3.0 * t2);
                d.normalize_or_zero()
            }
        }
    }
}

/// A 3D path consisting of path commands.
#[derive(Debug, Clone, Default)]
pub struct Path3D {
    commands: Vec<PathCommand3D>,
    /// Cached segments for sampling.
    segments: Vec<Segment>,
    /// Cumulative lengths for arc-length parameterization.
    cumulative_lengths: Vec<f32>,
    /// Total path length.
    total_length: f32,
}

impl Path3D {
    /// Creates an empty path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a path from commands.
    pub fn from_commands(commands: Vec<PathCommand3D>) -> Self {
        let mut path = Self {
            commands,
            segments: Vec::new(),
            cumulative_lengths: Vec::new(),
            total_length: 0.0,
        };
        path.rebuild_segments();
        path
    }

    /// Returns the path commands.
    pub fn commands(&self) -> &[PathCommand3D] {
        &self.commands
    }

    /// Returns true if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Returns the total arc length of the path.
    pub fn length(&self) -> f32 {
        self.total_length
    }

    /// Rebuilds the internal segment cache.
    fn rebuild_segments(&mut self) {
        self.segments.clear();
        self.cumulative_lengths.clear();

        let mut current = Vec3::ZERO;
        let mut length = 0.0;

        for cmd in &self.commands {
            match cmd {
                PathCommand3D::MoveTo(p) => {
                    current = *p;
                }
                PathCommand3D::LineTo(to) => {
                    let seg = Segment::Line {
                        from: current,
                        to: *to,
                    };
                    length += seg.length_estimate();
                    self.segments.push(seg);
                    self.cumulative_lengths.push(length);
                    current = *to;
                }
                PathCommand3D::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    let seg = Segment::Cubic {
                        from: current,
                        c1: *control1,
                        c2: *control2,
                        to: *to,
                    };
                    length += seg.length_estimate();
                    self.segments.push(seg);
                    self.cumulative_lengths.push(length);
                    current = *to;
                }
            }
        }

        self.total_length = length;
    }

    /// Samples position at t ∈ [0, 1] using arc-length parameterization.
    pub fn position_at(&self, t: f32) -> Vec3 {
        if self.segments.is_empty() {
            return Vec3::ZERO;
        }

        let t = t.clamp(0.0, 1.0);
        let target_length = t * self.total_length;

        // Find the segment containing this length
        let (seg_idx, seg_t) = self.find_segment(target_length);
        self.segments[seg_idx].position_at(seg_t)
    }

    /// Samples tangent at t ∈ [0, 1] using arc-length parameterization.
    pub fn tangent_at(&self, t: f32) -> Vec3 {
        if self.segments.is_empty() {
            return Vec3::X;
        }

        let t = t.clamp(0.0, 1.0);
        let target_length = t * self.total_length;

        let (seg_idx, seg_t) = self.find_segment(target_length);
        self.segments[seg_idx].tangent_at(seg_t)
    }

    /// Samples position and tangent at t ∈ [0, 1].
    pub fn sample(&self, t: f32) -> PathSample {
        if self.segments.is_empty() {
            return PathSample {
                position: Vec3::ZERO,
                tangent: Vec3::X,
            };
        }

        let t = t.clamp(0.0, 1.0);
        let target_length = t * self.total_length;

        let (seg_idx, seg_t) = self.find_segment(target_length);
        PathSample {
            position: self.segments[seg_idx].position_at(seg_t),
            tangent: self.segments[seg_idx].tangent_at(seg_t),
        }
    }

    /// Finds which segment contains the given arc length, returns (segment_index, local_t).
    fn find_segment(&self, target_length: f32) -> (usize, f32) {
        if target_length <= 0.0 {
            return (0, 0.0);
        }

        for (i, &cum_len) in self.cumulative_lengths.iter().enumerate() {
            if target_length <= cum_len {
                let prev_len = if i == 0 {
                    0.0
                } else {
                    self.cumulative_lengths[i - 1]
                };
                let seg_len = cum_len - prev_len;
                let local_t = if seg_len > 0.0 {
                    (target_length - prev_len) / seg_len
                } else {
                    0.0
                };
                return (i, local_t);
            }
        }

        // Past the end
        (self.segments.len() - 1, 1.0)
    }
}

/// Result of sampling a path.
#[derive(Debug, Clone, Copy)]
pub struct PathSample {
    /// Position on the path.
    pub position: Vec3,
    /// Tangent direction (normalized).
    pub tangent: Vec3,
}

/// Builder for constructing 3D paths.
#[derive(Debug, Clone, Default)]
pub struct Path3DBuilder {
    commands: Vec<PathCommand3D>,
    current: Vec3,
}

impl Path3DBuilder {
    /// Creates a new path builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves to a point without drawing.
    pub fn move_to(mut self, to: Vec3) -> Self {
        self.commands.push(PathCommand3D::MoveTo(to));
        self.current = to;
        self
    }

    /// Draws a line to a point.
    pub fn line_to(mut self, to: Vec3) -> Self {
        self.commands.push(PathCommand3D::LineTo(to));
        self.current = to;
        self
    }

    /// Draws a cubic bezier curve.
    pub fn cubic_to(mut self, control1: Vec3, control2: Vec3, to: Vec3) -> Self {
        self.commands.push(PathCommand3D::CubicTo {
            control1,
            control2,
            to,
        });
        self.current = to;
        self
    }

    /// Builds the final path.
    pub fn build(self) -> Path3D {
        Path3D::from_commands(self.commands)
    }
}

/// Creates a straight line path.
pub fn line3d(from: Vec3, to: Vec3) -> Path3D {
    Path3DBuilder::new().move_to(from).line_to(to).build()
}

/// Creates a polyline path.
pub fn polyline3d(points: &[Vec3]) -> Path3D {
    if points.is_empty() {
        return Path3D::new();
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

        assert!((path.length() - 10.0).abs() < 0.001);
        assert_eq!(path.position_at(0.0), Vec3::ZERO);
        assert!((path.position_at(0.5).x - 5.0).abs() < 0.001);
        assert!((path.position_at(1.0).x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_line_tangent() {
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));

        let tangent = path.tangent_at(0.5);
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
        assert!((path.length() - 10.0).abs() < 0.001);

        // Midpoint should be at the corner
        let mid = path.position_at(0.5);
        assert!((mid.x - 5.0).abs() < 0.001);
        assert!(mid.y.abs() < 0.001);
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
        assert_eq!(path.position_at(0.0), Vec3::ZERO);
        let end = path.position_at(1.0);
        assert!((end.x - 10.0).abs() < 0.001);
        assert!(end.y.abs() < 0.001);
    }

    #[test]
    fn test_sample() {
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let sample = path.sample(0.5);

        assert!((sample.position.x - 5.0).abs() < 0.001);
        assert!((sample.tangent.x - 1.0).abs() < 0.001);
    }
}
