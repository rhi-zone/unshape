//! Path types.

use crate::{Curve, Segment2D};

/// A sequence of connected curves.
#[derive(Debug, Clone)]
pub struct Path<C: Curve = Segment2D> {
    pub segments: Vec<C>,
    pub closed: bool,
}

impl<C: Curve> Path<C> {
    /// Creates a new empty path.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            closed: false,
        }
    }

    /// Creates a path from segments.
    pub fn from_segments(segments: Vec<C>) -> Self {
        Self {
            segments,
            closed: false,
        }
    }

    /// Creates a closed path from segments.
    pub fn closed(segments: Vec<C>) -> Self {
        Self {
            segments,
            closed: true,
        }
    }

    /// Returns true if the path has no segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Returns the number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns the start point of the path.
    pub fn start(&self) -> Option<C::Point> {
        self.segments.first().map(|s| s.start())
    }

    /// Returns the end point of the path.
    pub fn end(&self) -> Option<C::Point> {
        self.segments.last().map(|s| s.end())
    }

    /// Adds a segment to the path.
    pub fn push(&mut self, segment: C) {
        self.segments.push(segment);
    }

    /// Returns the total arc length of the path.
    pub fn length(&self) -> f32 {
        self.segments.iter().map(|s| s.length()).sum()
    }

    /// Returns the point at parameter t ∈ [0, 1].
    ///
    /// Parameter is scaled across all segments uniformly (not arc-length parameterized).
    pub fn position_at(&self, t: f32) -> Option<C::Point> {
        if self.segments.is_empty() {
            return None;
        }

        let t = t.clamp(0.0, 1.0);
        let scaled = t * self.segments.len() as f32;
        let index = (scaled.floor() as usize).min(self.segments.len() - 1);
        let local_t = if index == self.segments.len() - 1 && t == 1.0 {
            1.0
        } else {
            scaled.fract()
        };

        Some(self.segments[index].position_at(local_t))
    }

    /// Returns the tangent at parameter t ∈ [0, 1].
    pub fn tangent_at(&self, t: f32) -> Option<C::Point> {
        if self.segments.is_empty() {
            return None;
        }

        let t = t.clamp(0.0, 1.0);
        let scaled = t * self.segments.len() as f32;
        let index = (scaled.floor() as usize).min(self.segments.len() - 1);
        let local_t = if index == self.segments.len() - 1 && t == 1.0 {
            1.0
        } else {
            scaled.fract()
        };

        Some(self.segments[index].tangent_at(local_t))
    }

    /// Flattens the path to a polyline.
    pub fn flatten(&self, tolerance: f32) -> Vec<C::Point> {
        if self.segments.is_empty() {
            return Vec::new();
        }

        let mut points = Vec::new();
        for (i, segment) in self.segments.iter().enumerate() {
            let seg_points = segment.flatten(tolerance);
            if i == 0 {
                points.extend(seg_points);
            } else {
                // Skip first point (it's the same as previous segment's end)
                points.extend(seg_points.into_iter().skip(1));
            }
        }
        points
    }
}

impl<C: Curve> Default for Path<C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper that caches cumulative segment lengths for uniform-speed sampling.
#[derive(Debug, Clone)]
pub struct ArcLengthPath<C: Curve> {
    path: Path<C>,
    cumulative_lengths: Vec<f32>,
    total_length: f32,
}

impl<C: Curve> ArcLengthPath<C> {
    /// Creates a new arc-length parameterized path.
    pub fn new(path: Path<C>) -> Self {
        let mut cumulative_lengths = Vec::with_capacity(path.segments.len());
        let mut total = 0.0;

        for segment in &path.segments {
            total += segment.length();
            cumulative_lengths.push(total);
        }

        Self {
            path,
            cumulative_lengths,
            total_length: total,
        }
    }

    /// Returns the total arc length of the path.
    pub fn total_length(&self) -> f32 {
        self.total_length
    }

    /// Returns the underlying path.
    pub fn path(&self) -> &Path<C> {
        &self.path
    }

    /// Returns the point at arc-length parameter t ∈ [0, 1].
    ///
    /// Unlike `Path::position_at`, this gives uniform speed along the path.
    pub fn position_at(&self, t: f32) -> Option<C::Point> {
        if self.path.segments.is_empty() || self.total_length < 1e-10 {
            return self.path.start();
        }

        let t = t.clamp(0.0, 1.0);
        let target_length = t * self.total_length;

        // Binary search to find segment
        let (index, prev_length) = self.find_segment(target_length);

        let segment = &self.path.segments[index];
        let segment_length = segment.length();

        if segment_length < 1e-10 {
            return Some(segment.start());
        }

        // Local parameter within segment
        let local_dist = target_length - prev_length;
        let local_t = (local_dist / segment_length).clamp(0.0, 1.0);

        Some(segment.position_at(local_t))
    }

    /// Returns the tangent at arc-length parameter t ∈ [0, 1].
    pub fn tangent_at(&self, t: f32) -> Option<C::Point> {
        if self.path.segments.is_empty() || self.total_length < 1e-10 {
            return None;
        }

        let t = t.clamp(0.0, 1.0);
        let target_length = t * self.total_length;

        let (index, prev_length) = self.find_segment(target_length);

        let segment = &self.path.segments[index];
        let segment_length = segment.length();

        if segment_length < 1e-10 {
            return Some(segment.tangent_at(0.0));
        }

        let local_dist = target_length - prev_length;
        let local_t = (local_dist / segment_length).clamp(0.0, 1.0);

        Some(segment.tangent_at(local_t))
    }

    /// Finds the segment index and cumulative length before it.
    fn find_segment(&self, target_length: f32) -> (usize, f32) {
        // Binary search
        let mut lo = 0;
        let mut hi = self.cumulative_lengths.len();

        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cumulative_lengths[mid] < target_length {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let index = lo.min(self.path.segments.len() - 1);
        let prev_length = if index == 0 {
            0.0
        } else {
            self.cumulative_lengths[index - 1]
        };

        (index, prev_length)
    }

    /// Samples the path at uniform arc-length intervals.
    pub fn sample(&self, num_samples: usize) -> Vec<C::Point> {
        if num_samples == 0 {
            return Vec::new();
        }
        if num_samples == 1 {
            return self.position_at(0.0).into_iter().collect();
        }

        (0..num_samples)
            .filter_map(|i| {
                let t = i as f32 / (num_samples - 1) as f32;
                self.position_at(t)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Line;
    use glam::Vec2;

    #[test]
    fn test_path_empty() {
        let path: Path<Segment2D> = Path::new();
        assert!(path.is_empty());
        assert_eq!(path.len(), 0);
        assert!(path.position_at(0.5).is_none());
    }

    #[test]
    fn test_path_single_segment() {
        let seg = Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0)));
        let path = Path::from_segments(vec![seg]);

        assert_eq!(path.len(), 1);
        assert_eq!(path.position_at(0.0), Some(Vec2::ZERO));
        assert_eq!(path.position_at(1.0), Some(Vec2::new(10.0, 0.0)));
        assert_eq!(path.position_at(0.5), Some(Vec2::new(5.0, 0.0)));
    }

    #[test]
    fn test_path_multiple_segments() {
        let path = Path::from_segments(vec![
            Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0))),
            Segment2D::Line(Line::new(Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0))),
        ]);

        assert_eq!(path.len(), 2);

        // t=0.25 is middle of first segment
        assert_eq!(path.position_at(0.25), Some(Vec2::new(5.0, 0.0)));

        // t=0.5 is start of second segment
        assert_eq!(path.position_at(0.5), Some(Vec2::new(10.0, 0.0)));

        // t=0.75 is middle of second segment
        assert_eq!(path.position_at(0.75), Some(Vec2::new(10.0, 5.0)));
    }

    #[test]
    fn test_path_length() {
        let path = Path::from_segments(vec![
            Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(3.0, 0.0))),
            Segment2D::Line(Line::new(Vec2::new(3.0, 0.0), Vec2::new(3.0, 4.0))),
        ]);

        assert!((path.length() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_arc_length_path_uniform() {
        // Two segments of different lengths
        let path = Path::from_segments(vec![
            Segment2D::Line(Line::new(Vec2::ZERO, Vec2::new(10.0, 0.0))), // length 10
            Segment2D::Line(Line::new(Vec2::new(10.0, 0.0), Vec2::new(10.0, 20.0))), // length 20
        ]);

        let arc_path = ArcLengthPath::new(path);
        assert!((arc_path.total_length() - 30.0).abs() < 0.001);

        // t=0.5 should be at arc-length 15, which is in second segment
        let mid = arc_path.position_at(0.5).unwrap();
        // First segment is length 10, so we're 5 units into second segment
        assert!((mid - Vec2::new(10.0, 5.0)).length() < 0.01);
    }

    #[test]
    fn test_arc_length_path_sample() {
        let path = Path::from_segments(vec![Segment2D::Line(Line::new(
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
        ))]);

        let arc_path = ArcLengthPath::new(path);
        let samples = arc_path.sample(11);

        assert_eq!(samples.len(), 11);

        // Should be evenly spaced
        for (i, sample) in samples.iter().enumerate() {
            let expected_x = i as f32;
            assert!((sample.x - expected_x).abs() < 0.01);
        }
    }
}
