//! Path stroke and offset operations.
//!
//! Provides operations for converting strokes to fills and offsetting paths.
//!
//! # Operations
//!
//! - **Offset**: Create a parallel path at a given distance (inset/outset)
//! - **Stroke to path**: Convert a stroked path to a filled outline
//! - **Dash**: Apply a dash pattern to a path
//!
//! # Usage
//!
//! ```ignore
//! let path = circle(Vec2::ZERO, 100.0);
//!
//! // Offset the path outward
//! let outer = offset_path(&path, 10.0, JoinStyle::Miter);
//!
//! // Convert stroke to fill
//! let outline = stroke_to_path(&path, 5.0, CapStyle::Round, JoinStyle::Round);
//! ```

use crate::{Path, PathBuilder, PathCommand};
use glam::Vec2;
use std::f32::consts::PI;

/// Style for line joins.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum JoinStyle {
    /// Sharp corner (may be limited by miter limit).
    #[default]
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled (flat) corner.
    Bevel,
}

/// Style for line caps.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CapStyle {
    /// Flat cap at the endpoint.
    #[default]
    Butt,
    /// Rounded cap.
    Round,
    /// Square cap extending past endpoint.
    Square,
}

/// Configuration for stroke operations.
#[derive(Debug, Clone)]
pub struct StrokeConfig {
    /// Width of the stroke.
    pub width: f32,
    /// Cap style for line ends.
    pub cap: CapStyle,
    /// Join style for corners.
    pub join: JoinStyle,
    /// Miter limit (for miter joins).
    pub miter_limit: f32,
}

impl Default for StrokeConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            cap: CapStyle::Butt,
            join: JoinStyle::Miter,
            miter_limit: 4.0,
        }
    }
}

impl StrokeConfig {
    /// Creates a stroke config with the given width.
    pub fn new(width: f32) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }

    /// Sets the cap style.
    pub fn with_cap(mut self, cap: CapStyle) -> Self {
        self.cap = cap;
        self
    }

    /// Sets the join style.
    pub fn with_join(mut self, join: JoinStyle) -> Self {
        self.join = join;
        self
    }

    /// Sets the miter limit.
    pub fn with_miter_limit(mut self, limit: f32) -> Self {
        self.miter_limit = limit;
        self
    }
}

/// Dash pattern for stroked paths.
#[derive(Debug, Clone)]
pub struct DashPattern {
    /// Array of dash and gap lengths.
    pub pattern: Vec<f32>,
    /// Offset into the pattern.
    pub offset: f32,
}

impl DashPattern {
    /// Creates a simple dash pattern (dash, gap).
    pub fn simple(dash: f32, gap: f32) -> Self {
        Self {
            pattern: vec![dash, gap],
            offset: 0.0,
        }
    }

    /// Creates a dotted pattern.
    pub fn dotted(dot_size: f32, gap: f32) -> Self {
        Self::simple(dot_size, gap)
    }

    /// Creates a dashed pattern.
    pub fn dashed(dash: f32, gap: f32) -> Self {
        Self::simple(dash, gap)
    }

    /// Creates a dash-dot pattern.
    pub fn dash_dot(dash: f32, dot: f32, gap: f32) -> Self {
        Self {
            pattern: vec![dash, gap, dot, gap],
            offset: 0.0,
        }
    }

    /// Sets the offset into the pattern.
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }
}

/// Offsets a path by a given distance.
///
/// Positive distance offsets outward, negative inward.
/// This is a simplified implementation that works best for convex paths.
pub fn offset_path(path: &Path, distance: f32, join: JoinStyle) -> Path {
    if path.is_empty() || distance.abs() < 1e-6 {
        return path.clone();
    }

    let points = path_to_points(path);
    if points.len() < 2 {
        return path.clone();
    }

    let mut builder = PathBuilder::new();
    let mut offset_points = Vec::new();

    // Calculate offset points
    for i in 0..points.len() {
        let curr = points[i];
        let prev = points[(i + points.len() - 1) % points.len()];
        let next = points[(i + 1) % points.len()];

        // Edge normals
        let edge_prev = (curr - prev).normalize_or_zero();
        let edge_next = (next - curr).normalize_or_zero();

        let normal_prev = Vec2::new(-edge_prev.y, edge_prev.x);
        let normal_next = Vec2::new(-edge_next.y, edge_next.x);

        // Average normal at vertex
        let avg_normal = (normal_prev + normal_next).normalize_or_zero();

        // Handle sharp corners
        let dot = normal_prev.dot(normal_next);
        let offset_point = if dot < -0.9 {
            // Very sharp corner, just use average
            curr + avg_normal * distance
        } else {
            // Calculate proper offset for corner
            let cos_half = ((1.0 + dot) / 2.0).sqrt();
            if cos_half > 0.1 {
                curr + avg_normal * (distance / cos_half)
            } else {
                curr + avg_normal * distance
            }
        };

        offset_points.push(offset_point);

        // Add join geometry if needed
        if matches!(join, JoinStyle::Round) && dot < 0.9 {
            // Add arc for round join
            let angle_start = normal_prev.y.atan2(normal_prev.x);
            let angle_end = normal_next.y.atan2(normal_next.x);
            let mut angle_diff = angle_end - angle_start;
            if angle_diff > PI {
                angle_diff -= 2.0 * PI;
            } else if angle_diff < -PI {
                angle_diff += 2.0 * PI;
            }

            // Add intermediate points for the arc
            let steps = ((angle_diff.abs() / (PI / 8.0)).ceil() as usize).max(1);
            for j in 1..steps {
                let t = j as f32 / steps as f32;
                let angle = angle_start + angle_diff * t;
                let arc_point = curr + Vec2::new(angle.cos(), angle.sin()) * distance.abs();
                offset_points.push(arc_point);
            }
        }
    }

    // Build the offset path
    if !offset_points.is_empty() {
        builder = builder.move_to(offset_points[0]);
        for &p in &offset_points[1..] {
            builder = builder.line_to(p);
        }
        builder = builder.close();
    }

    builder.build()
}

/// Converts a stroked path to a filled outline path.
///
/// Creates a path representing the outline of the stroke.
pub fn stroke_to_path(path: &Path, config: &StrokeConfig) -> Path {
    if path.is_empty() || config.width <= 0.0 {
        return Path::new();
    }

    let half_width = config.width / 2.0;
    let points = path_to_points(path);

    if points.len() < 2 {
        return Path::new();
    }

    let is_closed = is_path_closed(path);

    // Generate offset curves on both sides
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();

    for i in 0..points.len() {
        let curr = points[i];

        // Get edge directions
        let (prev_dir, next_dir) = if is_closed {
            let prev = points[(i + points.len() - 1) % points.len()];
            let next = points[(i + 1) % points.len()];
            (
                (curr - prev).normalize_or_zero(),
                (next - curr).normalize_or_zero(),
            )
        } else if i == 0 {
            let next = points[i + 1];
            let dir = (next - curr).normalize_or_zero();
            (dir, dir)
        } else if i == points.len() - 1 {
            let prev = points[i - 1];
            let dir = (curr - prev).normalize_or_zero();
            (dir, dir)
        } else {
            let prev = points[i - 1];
            let next = points[i + 1];
            (
                (curr - prev).normalize_or_zero(),
                (next - curr).normalize_or_zero(),
            )
        };

        // Compute normals
        let normal_prev = Vec2::new(-prev_dir.y, prev_dir.x);
        let normal_next = Vec2::new(-next_dir.y, next_dir.x);
        let avg_normal = (normal_prev + normal_next).normalize_or_zero();

        // Compute offset factor for mitered join
        let dot = prev_dir.dot(next_dir);
        let offset_factor = if dot < 0.0 {
            let cos_half = ((1.0 + dot) / 2.0).sqrt().max(0.1);
            1.0 / cos_half
        } else {
            1.0
        };

        // Apply miter limit
        let limited_factor = offset_factor.min(config.miter_limit);

        left_points.push(curr + avg_normal * half_width * limited_factor);
        right_points.push(curr - avg_normal * half_width * limited_factor);
    }

    // Build the outline path
    let mut builder = PathBuilder::new();

    // Left side (forward)
    if !left_points.is_empty() {
        builder = builder.move_to(left_points[0]);
        for &p in &left_points[1..] {
            builder = builder.line_to(p);
        }
    }

    // End cap
    if !is_closed && !right_points.is_empty() {
        builder = add_cap(
            builder,
            points.last().copied().unwrap_or(Vec2::ZERO),
            *left_points.last().unwrap_or(&Vec2::ZERO),
            *right_points.last().unwrap_or(&Vec2::ZERO),
            config.cap,
            half_width,
            true,
        );
    }

    // Right side (backward)
    for &p in right_points.iter().rev() {
        builder = builder.line_to(p);
    }

    // Start cap
    if !is_closed && !right_points.is_empty() {
        builder = add_cap(
            builder,
            points.first().copied().unwrap_or(Vec2::ZERO),
            right_points[0],
            left_points[0],
            config.cap,
            half_width,
            false,
        );
    }

    builder = builder.close();
    builder.build()
}

/// Adds a cap to the path builder.
fn add_cap(
    mut builder: PathBuilder,
    center: Vec2,
    from: Vec2,
    to: Vec2,
    style: CapStyle,
    radius: f32,
    is_end: bool,
) -> PathBuilder {
    match style {
        CapStyle::Butt => {
            // Just connect directly
            builder = builder.line_to(to);
        }
        CapStyle::Square => {
            // Extend past the endpoint
            let dir = (to - from).normalize_or_zero();
            let _perp = Vec2::new(-dir.y, dir.x);
            let ext = if is_end { dir } else { -dir } * radius;
            builder = builder.line_to(from + ext);
            builder = builder.line_to(to + ext);
        }
        CapStyle::Round => {
            // Add a semicircle
            let start_angle = (from - center).y.atan2((from - center).x);
            let end_angle = (to - center).y.atan2((to - center).x);

            let steps = 8;
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                let angle = if is_end {
                    start_angle + (end_angle - start_angle + PI) * t
                } else {
                    start_angle + (end_angle - start_angle - PI) * t
                };
                let p = center + Vec2::new(angle.cos(), angle.sin()) * radius;
                builder = builder.line_to(p);
            }
        }
    }

    builder
}

/// Applies a dash pattern to a path.
///
/// Returns a new path with gaps according to the dash pattern.
pub fn dash_path(path: &Path, pattern: &DashPattern) -> Path {
    if path.is_empty() || pattern.pattern.is_empty() {
        return path.clone();
    }

    let points = path_to_points(path);
    if points.len() < 2 {
        return path.clone();
    }

    let mut builder = PathBuilder::new();
    let mut pattern_idx = 0;
    let mut pattern_pos = pattern.offset;
    let mut drawing = true;
    let mut in_subpath = false;

    // Normalize pattern position
    let pattern_len: f32 = pattern.pattern.iter().sum();
    while pattern_pos < 0.0 {
        pattern_pos += pattern_len;
    }
    while pattern_pos >= pattern_len {
        pattern_pos -= pattern_len;
    }

    // Find starting position in pattern
    let mut remaining = pattern_pos;
    while remaining > 0.0 {
        if remaining <= pattern.pattern[pattern_idx] {
            break;
        }
        remaining -= pattern.pattern[pattern_idx];
        pattern_idx = (pattern_idx + 1) % pattern.pattern.len();
        drawing = !drawing;
    }
    let mut current_dash_remaining = pattern.pattern[pattern_idx] - remaining;

    // Process each segment
    for i in 0..points.len() - 1 {
        let start = points[i];
        let end = points[i + 1];
        let segment_vec = end - start;
        let segment_len = segment_vec.length();

        if segment_len < 1e-6 {
            continue;
        }

        let dir = segment_vec / segment_len;
        let mut pos = 0.0;

        while pos < segment_len {
            let step = current_dash_remaining.min(segment_len - pos);
            let step_start = start + dir * pos;
            let step_end = start + dir * (pos + step);

            if drawing {
                if !in_subpath {
                    builder = builder.move_to(step_start);
                    in_subpath = true;
                }
                builder = builder.line_to(step_end);
            } else {
                in_subpath = false;
            }

            pos += step;
            current_dash_remaining -= step;

            if current_dash_remaining < 1e-6 {
                pattern_idx = (pattern_idx + 1) % pattern.pattern.len();
                current_dash_remaining = pattern.pattern[pattern_idx];
                drawing = !drawing;
            }
        }
    }

    builder.build()
}

/// Extracts points from a path (flattening curves).
fn path_to_points(path: &Path) -> Vec<Vec2> {
    let mut points = Vec::new();
    let mut current = Vec2::ZERO;

    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) => {
                current = *p;
                points.push(current);
            }
            PathCommand::LineTo(p) => {
                current = *p;
                points.push(current);
            }
            PathCommand::QuadTo { control, to } => {
                // Flatten quadratic bezier
                let steps = 8;
                for i in 1..=steps {
                    let t = i as f32 / steps as f32;
                    let p = quadratic_point(current, *control, *to, t);
                    points.push(p);
                }
                current = *to;
            }
            PathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                // Flatten cubic bezier
                let steps = 12;
                for i in 1..=steps {
                    let t = i as f32 / steps as f32;
                    let p = cubic_point(current, *control1, *control2, *to, t);
                    points.push(p);
                }
                current = *to;
            }
            PathCommand::Close => {
                // Close handled separately
            }
        }
    }

    points
}

/// Evaluates a quadratic bezier at parameter t.
fn quadratic_point(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    p0 * mt2 + p1 * 2.0 * mt * t + p2 * t2
}

/// Evaluates a cubic bezier at parameter t.
fn cubic_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    p0 * mt3 + p1 * 3.0 * mt2 * t + p2 * 3.0 * mt * t2 + p3 * t3
}

/// Checks if a path is closed.
fn is_path_closed(path: &Path) -> bool {
    path.commands()
        .iter()
        .any(|cmd| matches!(cmd, PathCommand::Close))
}

/// Computes the total length of a path.
pub fn path_length(path: &Path) -> f32 {
    let points = path_to_points(path);
    if points.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;
    for i in 0..points.len() - 1 {
        length += (points[i + 1] - points[i]).length();
    }

    length
}

/// Samples a point along a path at a given distance from the start.
pub fn point_at_length(path: &Path, distance: f32) -> Option<Vec2> {
    let points = path_to_points(path);
    if points.len() < 2 {
        return None;
    }

    let mut remaining = distance;
    for i in 0..points.len() - 1 {
        let segment_len = (points[i + 1] - points[i]).length();
        if remaining <= segment_len {
            let t = remaining / segment_len;
            return Some(points[i].lerp(points[i + 1], t));
        }
        remaining -= segment_len;
    }

    Some(*points.last().unwrap())
}

/// Samples a tangent vector along a path at a given distance.
pub fn tangent_at_length(path: &Path, distance: f32) -> Option<Vec2> {
    let points = path_to_points(path);
    if points.len() < 2 {
        return None;
    }

    let mut remaining = distance;
    for i in 0..points.len() - 1 {
        let segment = points[i + 1] - points[i];
        let segment_len = segment.length();
        if remaining <= segment_len {
            return Some(segment.normalize_or_zero());
        }
        remaining -= segment_len;
    }

    // Return last segment direction
    let n = points.len();
    Some((points[n - 1] - points[n - 2]).normalize_or_zero())
}

// ===========================================================================
// Path Simplification
// ===========================================================================

/// Simplifies a path using the Ramer-Douglas-Peucker algorithm.
///
/// Reduces the number of points in a polyline while preserving its shape.
/// Points within `epsilon` distance of the simplified line are removed.
///
/// # Arguments
/// * `path` - The path to simplify.
/// * `epsilon` - Maximum distance threshold for point removal.
///
/// # Example
/// ```ignore
/// let simplified = simplify_path(&path, 1.0);
/// ```
pub fn simplify_path(path: &Path, epsilon: f32) -> Path {
    let points = path_to_points(path);
    if points.len() < 3 {
        return path.clone();
    }

    let simplified = rdp_simplify(&points, epsilon);
    points_to_path(&simplified, is_path_closed(path))
}

/// Simplifies a list of points using Ramer-Douglas-Peucker.
pub fn simplify_points(points: &[Vec2], epsilon: f32) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    rdp_simplify(points, epsilon)
}

/// Internal RDP implementation.
fn rdp_simplify(points: &[Vec2], epsilon: f32) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with maximum distance from the line between endpoints
    let start = points[0];
    let end = points[points.len() - 1];

    let mut max_dist = 0.0;
    let mut max_idx = 0;

    for (i, &point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let dist = perpendicular_distance(point, start, end);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon {
        let mut left = rdp_simplify(&points[..=max_idx], epsilon);
        let right = rdp_simplify(&points[max_idx..], epsilon);

        // Remove duplicate point at junction
        left.pop();
        left.extend(right);
        left
    } else {
        // All points between start and end can be removed
        vec![start, end]
    }
}

/// Calculates perpendicular distance from a point to a line segment.
fn perpendicular_distance(point: Vec2, line_start: Vec2, line_end: Vec2) -> f32 {
    let line_vec = line_end - line_start;
    let line_len_sq = line_vec.length_squared();

    if line_len_sq < 1e-10 {
        // Line is a point
        return (point - line_start).length();
    }

    // Project point onto line
    let t = ((point - line_start).dot(line_vec) / line_len_sq).clamp(0.0, 1.0);
    let projection = line_start + line_vec * t;

    (point - projection).length()
}

/// Converts points back to a path.
fn points_to_path(points: &[Vec2], closed: bool) -> Path {
    if points.is_empty() {
        return Path::new();
    }

    let mut builder = PathBuilder::new().move_to(points[0]);

    for &point in &points[1..] {
        builder = builder.line_to(point);
    }

    if closed {
        builder = builder.close();
    }

    builder.build()
}

/// Smooths a path using Chaikin's corner-cutting algorithm.
///
/// Creates a smoother curve by iteratively cutting corners.
///
/// # Arguments
/// * `path` - The path to smooth.
/// * `iterations` - Number of smoothing iterations (1-5 recommended).
pub fn smooth_path(path: &Path, iterations: usize) -> Path {
    let mut points = path_to_points(path);
    let closed = is_path_closed(path);

    for _ in 0..iterations {
        points = chaikin_smooth(&points, closed);
    }

    points_to_path(&points, closed)
}

/// Single iteration of Chaikin's algorithm.
fn chaikin_smooth(points: &[Vec2], closed: bool) -> Vec<Vec2> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let mut result = Vec::with_capacity(points.len() * 2);

    if closed {
        // For closed paths, include the wrap-around
        for i in 0..points.len() {
            let p0 = points[i];
            let p1 = points[(i + 1) % points.len()];

            result.push(p0.lerp(p1, 0.25));
            result.push(p0.lerp(p1, 0.75));
        }
    } else {
        // Keep first point
        result.push(points[0]);

        for i in 0..points.len() - 1 {
            let p0 = points[i];
            let p1 = points[i + 1];

            result.push(p0.lerp(p1, 0.25));
            result.push(p0.lerp(p1, 0.75));
        }

        // Keep last point
        result.push(points[points.len() - 1]);
    }

    result
}

/// Resamples a path to have evenly spaced points.
///
/// # Arguments
/// * `path` - The path to resample.
/// * `spacing` - Target distance between points.
pub fn resample_path(path: &Path, spacing: f32) -> Path {
    if spacing <= 0.0 {
        return path.clone();
    }

    let points = path_to_points(path);
    if points.len() < 2 {
        return path.clone();
    }

    let total_length = path_length(path);
    let num_points = (total_length / spacing).ceil() as usize;

    if num_points < 2 {
        return path.clone();
    }

    let mut resampled = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let t = i as f32 / (num_points - 1) as f32;
        let dist = t * total_length;

        if let Some(point) = sample_path_at_distance(&points, dist) {
            resampled.push(point);
        }
    }

    points_to_path(&resampled, is_path_closed(path))
}

/// Samples a point along a polyline at a given distance.
fn sample_path_at_distance(points: &[Vec2], distance: f32) -> Option<Vec2> {
    if points.len() < 2 {
        return points.first().copied();
    }

    let mut remaining = distance;
    for i in 0..points.len() - 1 {
        let segment_len = (points[i + 1] - points[i]).length();
        if remaining <= segment_len {
            let t = remaining / segment_len;
            return Some(points[i].lerp(points[i + 1], t));
        }
        remaining -= segment_len;
    }

    points.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circle, rect};

    #[test]
    fn test_stroke_config_default() {
        let config = StrokeConfig::default();
        assert_eq!(config.width, 1.0);
        assert_eq!(config.cap, CapStyle::Butt);
        assert_eq!(config.join, JoinStyle::Miter);
    }

    #[test]
    fn test_stroke_config_builder() {
        let config = StrokeConfig::new(5.0)
            .with_cap(CapStyle::Round)
            .with_join(JoinStyle::Bevel);
        assert_eq!(config.width, 5.0);
        assert_eq!(config.cap, CapStyle::Round);
        assert_eq!(config.join, JoinStyle::Bevel);
    }

    #[test]
    fn test_dash_pattern_simple() {
        let pattern = DashPattern::simple(10.0, 5.0);
        assert_eq!(pattern.pattern, vec![10.0, 5.0]);
        assert_eq!(pattern.offset, 0.0);
    }

    #[test]
    fn test_dash_pattern_dash_dot() {
        let pattern = DashPattern::dash_dot(10.0, 2.0, 5.0);
        assert_eq!(pattern.pattern, vec![10.0, 5.0, 2.0, 5.0]);
    }

    #[test]
    fn test_offset_path_circle() {
        let path = circle(Vec2::ZERO, 50.0);
        let offset = offset_path(&path, 10.0, JoinStyle::Round);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_path_rect() {
        let path = rect(Vec2::ZERO, Vec2::new(100.0, 50.0));
        let offset = offset_path(&path, 5.0, JoinStyle::Miter);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_path_zero() {
        let path = circle(Vec2::ZERO, 50.0);
        let offset = offset_path(&path, 0.0, JoinStyle::Miter);
        assert_eq!(offset.len(), path.len());
    }

    #[test]
    fn test_stroke_to_path_basic() {
        let path = rect(Vec2::ZERO, Vec2::new(100.0, 50.0));
        let config = StrokeConfig::new(5.0);
        let outline = stroke_to_path(&path, &config);
        assert!(!outline.is_empty());
    }

    #[test]
    fn test_stroke_to_path_round_cap() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let config = StrokeConfig::new(10.0).with_cap(CapStyle::Round);
        let outline = stroke_to_path(&path, &config);
        assert!(!outline.is_empty());
    }

    #[test]
    fn test_dash_path_basic() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let pattern = DashPattern::simple(10.0, 5.0);
        let dashed = dash_path(&path, &pattern);
        // Dashed path should have multiple subpaths
        assert!(!dashed.is_empty());
    }

    #[test]
    fn test_path_length() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let len = path_length(&path);
        assert!((len - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_point_at_length() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let point = point_at_length(&path, 50.0);
        assert!(point.is_some());
        let p = point.unwrap();
        assert!((p.x - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_tangent_at_length() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let tangent = tangent_at_length(&path, 50.0);
        assert!(tangent.is_some());
        let t = tangent.unwrap();
        assert!((t.x - 1.0).abs() < 0.1);
        assert!(t.y.abs() < 0.1);
    }

    #[test]
    fn test_simplify_path_line() {
        // A straight line with intermediate points should simplify to 2 points
        let mut builder = PathBuilder::new().move_to(Vec2::ZERO);
        for i in 1..10 {
            builder = builder.line_to(Vec2::new(i as f32 * 10.0, 0.0));
        }
        let path = builder.build();

        let simplified = simplify_path(&path, 0.1);
        let points = path_to_points(&simplified);

        // Should collapse to just start and end
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_simplify_path_zigzag() {
        // A zigzag should not simplify much with small epsilon
        let path = PathBuilder::new()
            .move_to(Vec2::ZERO)
            .line_to(Vec2::new(10.0, 10.0))
            .line_to(Vec2::new(20.0, 0.0))
            .line_to(Vec2::new(30.0, 10.0))
            .line_to(Vec2::new(40.0, 0.0))
            .build();

        let simplified = simplify_path(&path, 0.1);
        let points = path_to_points(&simplified);

        // Should keep most points due to large deviations
        assert!(points.len() >= 4);
    }

    #[test]
    fn test_simplify_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.1), // Slightly off line
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];

        let simplified = simplify_points(&points, 0.2);

        // Should collapse middle points
        assert!(simplified.len() < points.len());
        assert_eq!(simplified[0], points[0]);
        assert_eq!(simplified[simplified.len() - 1], points[points.len() - 1]);
    }

    #[test]
    fn test_smooth_path() {
        let path = rect(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let smoothed = smooth_path(&path, 2);

        // Smoothed path should have more points
        assert!(smoothed.len() > path.len());
    }

    #[test]
    fn test_smooth_path_zero_iterations() {
        let path = rect(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let smoothed = smooth_path(&path, 0);

        // No smoothing should return same number of points
        assert_eq!(smoothed.len(), path.len());
    }

    #[test]
    fn test_resample_path() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let resampled = resample_path(&path, 10.0);

        let points = path_to_points(&resampled);

        // Should have approximately 100/10 + 1 = 11 points
        assert!(points.len() >= 10 && points.len() <= 12);

        // First and last points should match original endpoints
        assert!((points[0] - Vec2::ZERO).length() < 0.1);
        assert!((points[points.len() - 1] - Vec2::new(100.0, 0.0)).length() < 0.1);
    }

    #[test]
    fn test_perpendicular_distance() {
        let start = Vec2::ZERO;
        let end = Vec2::new(10.0, 0.0);

        // Point directly above the line
        let point = Vec2::new(5.0, 5.0);
        let dist = perpendicular_distance(point, start, end);
        assert!((dist - 5.0).abs() < 0.001);

        // Point on the line
        let point_on = Vec2::new(5.0, 0.0);
        let dist_on = perpendicular_distance(point_on, start, end);
        assert!(dist_on < 0.001);
    }
}
