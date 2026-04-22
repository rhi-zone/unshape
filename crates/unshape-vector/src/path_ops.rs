//! Higher-level path operations: blend, array, dash.
//!
//! These ops complement the stroke/offset ops in `stroke.rs` with morphing,
//! distribution, and dash pattern operations.

use crate::stroke::{path_length, point_at_length, resample_path, tangent_at_length};
use crate::{Path, PathBuilder, PathCommand};
use glam::Vec2;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// PathBlend
// ============================================================================

/// Input for [`PathBlend`], containing two paths to interpolate between.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PathBlendInput {
    /// First path (result at `t = 0.0`).
    pub a: Path,
    /// Second path (result at `t = 1.0`).
    pub b: Path,
}

/// Operation for morphing between two paths.
///
/// Interpolates linearly between the anchor/control points of two paths.
/// If the paths have different point counts, both are resampled to the
/// larger count before blending.
///
/// # Example
/// ```ignore
/// let blend = PathBlend { t: 0.5 };
/// let mid = blend.apply(&path_a, &path_b);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PathBlendInput, output = Path))]
pub struct PathBlend {
    /// Blend factor: 0.0 = path A, 1.0 = path B.
    pub t: f32,
}

impl Default for PathBlend {
    fn default() -> Self {
        Self { t: 0.5 }
    }
}

impl PathBlend {
    /// Creates a new blend op with the given t value.
    pub fn new(t: f32) -> Self {
        Self { t }
    }

    /// Applies the blend operation using a [`PathBlendInput`].
    ///
    /// This is the primary entry point used by the dynop system.
    /// For convenience, prefer [`PathBlend::blend`] when calling directly.
    pub fn apply(&self, input: &PathBlendInput) -> Path {
        blend_paths(&input.a, &input.b, self.t)
    }

    /// Applies the blend directly to two paths.
    ///
    /// Convenience method; equivalent to wrapping `a` and `b` in a [`PathBlendInput`].
    pub fn blend(&self, a: &Path, b: &Path) -> Path {
        blend_paths(a, b, self.t)
    }
}

/// Extracts all coordinate points from a path's commands (anchors + control points).
fn path_key_points(path: &Path) -> Vec<Vec2> {
    let mut pts = Vec::new();
    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) | PathCommand::LineTo(p) => pts.push(*p),
            PathCommand::QuadTo { control, to } => {
                pts.push(*control);
                pts.push(*to);
            }
            PathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                pts.push(*control1);
                pts.push(*control2);
                pts.push(*to);
            }
            PathCommand::Close => {}
        }
    }
    pts
}

/// Resamples a path to exactly `n` evenly-spaced points along its arc length.
fn resample_to_n(path: &Path, n: usize) -> Vec<Vec2> {
    if n == 0 {
        return Vec::new();
    }
    let total = path_length(path);
    if total < 1e-6 || n == 1 {
        return point_at_length(path, 0.0).into_iter().collect();
    }

    (0..n)
        .filter_map(|i| {
            let dist = total * (i as f32) / ((n - 1) as f32);
            point_at_length(path, dist)
        })
        .collect()
}

/// Core blend implementation.
fn blend_paths(a: &Path, b: &Path, t: f32) -> Path {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }

    let t = t.clamp(0.0, 1.0);

    let pts_a = path_key_points(a);
    let pts_b = path_key_points(b);

    // Use the longer point count, resample both if needed.
    let n = pts_a.len().max(pts_b.len());

    let pts_a = if pts_a.len() == n {
        pts_a
    } else {
        resample_to_n(a, n)
    };
    let pts_b = if pts_b.len() == n {
        pts_b
    } else {
        resample_to_n(b, n)
    };

    if pts_a.is_empty() || pts_b.is_empty() {
        return Path::new();
    }

    let is_closed = a.commands().iter().any(|c| matches!(c, PathCommand::Close))
        || b.commands().iter().any(|c| matches!(c, PathCommand::Close));

    let blended: Vec<Vec2> = pts_a
        .iter()
        .zip(pts_b.iter())
        .map(|(pa, pb)| pa.lerp(*pb, t))
        .collect();

    let mut builder = PathBuilder::new().move_to(blended[0]);
    for &p in &blended[1..] {
        builder = builder.line_to(p);
    }
    if is_closed {
        builder = builder.close();
    }
    builder.build()
}

// ============================================================================
// PathArray
// ============================================================================

/// Spacing mode for [`PathArray`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArraySpacing {
    /// Distribute exactly `count` copies evenly along the rail.
    #[default]
    Count,
    /// Place one copy every `distance` units along the rail.
    Distance(f32),
}

/// Input for [`PathArray`], containing the shape to copy and the rail path.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PathArrayInput {
    /// The shape to copy along the rail.
    pub shape: Path,
    /// The path along which copies are placed.
    pub rail: Path,
}

/// Operation for distributing copies of a shape along a rail path.
///
/// Places `count` copies of `shape` along `rail`, rotating each copy to follow
/// the rail tangent at its position.
///
/// # Example
/// ```ignore
/// let op = PathArray { count: 5, spacing: ArraySpacing::Count };
/// let copies = op.apply(&shape, &rail);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PathArrayInput, output = Vec<Path>))]
pub struct PathArray {
    /// Number of copies to place (used with [`ArraySpacing::Count`]).
    pub count: usize,
    /// How copies are spaced along the rail.
    pub spacing: ArraySpacing,
}

impl Default for PathArray {
    fn default() -> Self {
        Self {
            count: 5,
            spacing: ArraySpacing::Count,
        }
    }
}

impl PathArray {
    /// Creates a new path array with even count distribution.
    pub fn new(count: usize) -> Self {
        Self {
            count,
            spacing: ArraySpacing::Count,
        }
    }

    /// Creates a path array with fixed distance spacing.
    pub fn with_distance(distance: f32) -> Self {
        Self {
            count: usize::MAX,
            spacing: ArraySpacing::Distance(distance),
        }
    }

    /// Applies the array operation using a [`PathArrayInput`].
    ///
    /// This is the primary entry point used by the dynop system.
    /// For convenience, prefer [`PathArray::array`] when calling directly.
    pub fn apply(&self, input: &PathArrayInput) -> Vec<Path> {
        array_along_path(&input.shape, &input.rail, self.count, self.spacing)
    }

    /// Applies the array operation directly to a shape and rail.
    ///
    /// Returns a `Vec<Path>` where each entry is a transformed copy of `shape`
    /// placed at the corresponding position along `rail`.
    pub fn array(&self, shape: &Path, rail: &Path) -> Vec<Path> {
        array_along_path(shape, rail, self.count, self.spacing)
    }
}

/// Core array-along-path implementation.
fn array_along_path(shape: &Path, rail: &Path, count: usize, spacing: ArraySpacing) -> Vec<Path> {
    if shape.is_empty() || rail.is_empty() || count == 0 {
        return Vec::new();
    }

    let total = path_length(rail);
    if total < 1e-6 {
        return Vec::new();
    }

    // Compute the positions along the rail.
    let positions: Vec<f32> = match spacing {
        ArraySpacing::Count => {
            if count == 1 {
                vec![0.0]
            } else {
                (0..count)
                    .map(|i| total * (i as f32) / ((count - 1) as f32))
                    .collect()
            }
        }
        ArraySpacing::Distance(d) => {
            if d <= 0.0 {
                return Vec::new();
            }
            let n = (total / d).floor() as usize + 1;
            let n = n.min(count);
            (0..n).map(|i| (i as f32) * d).collect()
        }
    };

    positions
        .into_iter()
        .filter_map(|dist| {
            let pos = point_at_length(rail, dist)?;
            let tangent = tangent_at_length(rail, dist).unwrap_or(Vec2::X);
            Some(place_shape_at(shape, pos, tangent))
        })
        .collect()
}

/// Rotates and translates a shape to align with a position + tangent.
fn place_shape_at(shape: &Path, pos: Vec2, tangent: Vec2) -> Path {
    // The default shape orientation is along +X. Rotate to align tangent.
    let angle = tangent.y.atan2(tangent.x);
    let cos = angle.cos();
    let sin = angle.sin();

    let mut placed = shape.clone();
    placed.transform(|p| {
        // Rotate around origin then translate to pos
        Vec2::new(p.x * cos - p.y * sin + pos.x, p.x * sin + p.y * cos + pos.y)
    });
    placed
}

// ============================================================================
// DashPath
// ============================================================================

/// Operation for converting a path into dashed segments.
///
/// Walks the path by arc length, alternating between `dash` and `gap` lengths.
/// The `offset` parameter shifts the phase of the dash pattern.
///
/// Each dash segment is returned as a separate sub-path in the output `Vec<Path>`.
///
/// # Example
/// ```ignore
/// let op = DashPath { dash: 20.0, gap: 10.0, offset: 0.0 };
/// let segments = op.apply(&path);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Path, output = Vec<Path>))]
pub struct DashPath {
    /// Length of each dash segment.
    pub dash: f32,
    /// Length of each gap between dashes.
    pub gap: f32,
    /// Phase offset along the path (positive shifts dashes forward).
    pub offset: f32,
}

impl Default for DashPath {
    fn default() -> Self {
        Self {
            dash: 10.0,
            gap: 5.0,
            offset: 0.0,
        }
    }
}

impl DashPath {
    /// Creates a new dash op with the given dash and gap lengths.
    pub fn new(dash: f32, gap: f32) -> Self {
        Self {
            dash,
            gap,
            offset: 0.0,
        }
    }

    /// Creates a new dash op with a phase offset.
    pub fn with_offset(dash: f32, gap: f32, offset: f32) -> Self {
        Self { dash, gap, offset }
    }

    /// Applies the dash pattern to a path.
    ///
    /// Returns one `Path` per dash segment. Each path contains the points
    /// of a single dash run along the input path.
    pub fn apply(&self, path: &Path) -> Vec<Path> {
        dash_path_segments(path, self.dash, self.gap, self.offset)
    }
}

/// Core dash implementation: walks the path by arc length.
fn dash_path_segments(path: &Path, dash: f32, gap: f32, offset: f32) -> Vec<Path> {
    if path.is_empty() || dash <= 0.0 || gap < 0.0 {
        return Vec::new();
    }

    let cycle = dash + gap;
    if cycle < 1e-6 {
        return Vec::new();
    }

    // Resample to a dense polyline for arc-length traversal.
    let segment_size = dash.min(gap).max(1.0) / 4.0;
    let resampled = resample_path(path, segment_size);

    let points: Vec<Vec2> = {
        let mut pts = Vec::new();
        for cmd in resampled.commands() {
            match cmd {
                PathCommand::MoveTo(p) | PathCommand::LineTo(p) => pts.push(*p),
                PathCommand::Close | PathCommand::QuadTo { .. } | PathCommand::CubicTo { .. } => {}
            }
        }
        pts
    };

    if points.len() < 2 {
        return Vec::new();
    }

    // Normalize offset into [0, cycle).
    let mut phase = offset % cycle;
    if phase < 0.0 {
        phase += cycle;
    }

    let mut segments: Vec<Path> = Vec::new();
    let mut current_dash: Vec<Vec2> = Vec::new();

    // Determine whether we start in a dash or a gap.
    let mut in_dash = phase < dash;
    // Distance remaining in the current dash or gap phase.
    let mut remaining_in_phase = if in_dash { dash - phase } else { cycle - phase };

    for i in 0..points.len().saturating_sub(1) {
        let start = points[i];
        let end = points[i + 1];
        let seg_len = (end - start).length();
        if seg_len < 1e-9 {
            continue;
        }

        let dir = (end - start) / seg_len;
        let mut consumed = 0.0f32;

        while consumed < seg_len {
            let step = remaining_in_phase.min(seg_len - consumed);
            let p = start + dir * consumed;
            let p_end = start + dir * (consumed + step);

            if in_dash {
                if current_dash.is_empty() {
                    current_dash.push(p);
                }
                current_dash.push(p_end);
            } else if !current_dash.is_empty() {
                // Flush completed dash segment.
                let mut builder = PathBuilder::new().move_to(current_dash[0]);
                for &cp in &current_dash[1..] {
                    builder = builder.line_to(cp);
                }
                segments.push(builder.build());
                current_dash.clear();
            }

            consumed += step;
            remaining_in_phase -= step;

            if remaining_in_phase < 1e-9 {
                // Switch phase.
                in_dash = !in_dash;
                remaining_in_phase = if in_dash { dash } else { gap };
            }
        }
    }

    // Flush any trailing dash.
    if !current_dash.is_empty() {
        let mut builder = PathBuilder::new().move_to(current_dash[0]);
        for &cp in &current_dash[1..] {
            builder = builder.line_to(cp);
        }
        segments.push(builder.build());
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circle, line};
    use glam::Vec2;

    // =========================================================================
    // PathBlend tests
    // =========================================================================

    #[test]
    fn test_path_blend_t0_matches_a() {
        let a = line(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = line(Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0));

        let blend = PathBlend::new(0.0);
        let result = blend.blend(&a, &b);

        // At t=0 points should match path A
        let pts_result = path_key_points(&result);
        let pts_a = path_key_points(&a);
        assert_eq!(pts_result.len(), pts_a.len());
        for (pr, pa) in pts_result.iter().zip(pts_a.iter()) {
            assert!(
                (pr.distance(*pa)) < 0.01,
                "t=0 should match path A: {pr:?} vs {pa:?}"
            );
        }
    }

    #[test]
    fn test_path_blend_t1_matches_b() {
        let a = line(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = line(Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0));

        let blend = PathBlend::new(1.0);
        let result = blend.blend(&a, &b);

        // At t=1 points should match path B
        let pts_result = path_key_points(&result);
        let pts_b = path_key_points(&b);
        assert_eq!(pts_result.len(), pts_b.len());
        for (pr, pb) in pts_result.iter().zip(pts_b.iter()) {
            assert!(
                (pr.distance(*pb)) < 0.01,
                "t=1 should match path B: {pr:?} vs {pb:?}"
            );
        }
    }

    #[test]
    fn test_path_blend_t05_is_midpoint() {
        let a = line(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = line(Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0));

        let blend = PathBlend::new(0.5);
        let result = blend.blend(&a, &b);

        let pts = path_key_points(&result);
        // First point should be midpoint of (0,0) and (0,10) = (0,5)
        assert!(
            (pts[0].y - 5.0).abs() < 0.1,
            "mid y should be 5: {:?}",
            pts[0]
        );
    }

    #[test]
    fn test_path_blend_different_counts() {
        // Path with 2 points vs path with more points via circle
        let a = line(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let b = circle(Vec2::ZERO, 5.0);

        let blend = PathBlend::new(0.5);
        let result = blend.blend(&a, &b);
        assert!(!result.is_empty());
    }

    // =========================================================================
    // PathArray tests
    // =========================================================================

    #[test]
    fn test_path_array_count() {
        let shape = line(Vec2::ZERO, Vec2::new(1.0, 0.0));
        let rail = line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        let op = PathArray::new(5);
        let copies = op.array(&shape, &rail);

        assert_eq!(copies.len(), 5, "should produce exactly 5 copies");
    }

    #[test]
    fn test_path_array_distance() {
        let shape = line(Vec2::ZERO, Vec2::new(1.0, 0.0));
        let rail = line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        let op = PathArray {
            count: 100,
            spacing: ArraySpacing::Distance(20.0),
        };
        let copies = op.array(&shape, &rail);

        // 100 / 20 + 1 = 6 copies (0, 20, 40, 60, 80, 100)
        assert_eq!(
            copies.len(),
            6,
            "should produce 6 copies at 20-unit spacing"
        );
    }

    #[test]
    fn test_path_array_empty_rail() {
        let shape = line(Vec2::ZERO, Vec2::new(1.0, 0.0));
        let rail = Path::new();

        let op = PathArray::new(5);
        let copies = op.array(&shape, &rail);
        assert!(copies.is_empty());
    }

    // =========================================================================
    // DashPath tests
    // =========================================================================

    #[test]
    fn test_dash_path_produces_multiple_segments() {
        let path = line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let op = DashPath::new(10.0, 5.0);
        let segments = op.apply(&path);

        assert!(
            segments.len() > 1,
            "should produce more than one dash segment, got {}",
            segments.len()
        );
    }

    #[test]
    fn test_dash_path_each_segment_non_empty() {
        let path = line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let op = DashPath::new(10.0, 5.0);
        let segments = op.apply(&path);

        for (i, seg) in segments.iter().enumerate() {
            assert!(!seg.is_empty(), "segment {i} should not be empty");
        }
    }

    #[test]
    fn test_dash_path_empty_input() {
        let path = Path::new();
        let op = DashPath::new(10.0, 5.0);
        let segments = op.apply(&path);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_dash_path_with_offset() {
        let path = line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let op_no_offset = DashPath::new(10.0, 5.0);
        let op_with_offset = DashPath::with_offset(10.0, 5.0, 7.5);

        let segs_no = op_no_offset.apply(&path);
        let segs_with = op_with_offset.apply(&path);

        // Both should produce segments, but potentially different counts
        assert!(!segs_no.is_empty());
        assert!(!segs_with.is_empty());
    }
}
