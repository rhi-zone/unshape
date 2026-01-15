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

use crate::bezier::{cubic_point, quadratic_point};
use crate::{Path, PathBuilder, PathCommand};
use glam::Vec2;
use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Style for line joins.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CapStyle {
    /// Flat cap at the endpoint.
    #[default]
    Butt,
    /// Rounded cap.
    Round,
    /// Square cap extending past endpoint.
    Square,
}

/// Operation for converting strokes to filled path outlines.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Path, output = Path))]
pub struct Stroke {
    /// Width of the stroke.
    pub width: f32,
    /// Cap style for line ends.
    pub cap: CapStyle,
    /// Join style for corners.
    pub join: JoinStyle,
    /// Miter limit (for miter joins).
    pub miter_limit: f32,
}

impl Default for Stroke {
    fn default() -> Self {
        Self {
            width: 1.0,
            cap: CapStyle::Butt,
            join: JoinStyle::Miter,
            miter_limit: 4.0,
        }
    }
}

impl Stroke {
    /// Creates a stroke config with the given width.
    pub fn new(width: f32) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }

    /// Applies this stroke operation to a path, converting it to a filled outline.
    pub fn apply(&self, path: &Path) -> Path {
        stroke_to_path(path, self)
    }
}

/// Backwards-compatible type alias.
pub type StrokeConfig = Stroke;

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

// ===========================================================================
// Pressure Curves (Variable Stroke Width)
// ===========================================================================

/// A point with associated pressure value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressurePoint {
    /// 2D position.
    pub position: Vec2,
    /// Pressure value (0.0 to 1.0).
    pub pressure: f32,
}

impl PressurePoint {
    /// Creates a new pressure point.
    pub fn new(position: Vec2, pressure: f32) -> Self {
        Self {
            position,
            pressure: pressure.clamp(0.0, 1.0),
        }
    }

    /// Creates a pressure point with full pressure.
    pub fn full(position: Vec2) -> Self {
        Self::new(position, 1.0)
    }
}

/// A stroke with variable width based on pressure.
#[derive(Debug, Clone)]
pub struct PressureStroke {
    /// Points with pressure values.
    pub points: Vec<PressurePoint>,
}

impl PressureStroke {
    /// Creates a new empty pressure stroke.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Creates a pressure stroke from points with uniform pressure.
    pub fn from_points(points: &[Vec2], pressure: f32) -> Self {
        Self {
            points: points
                .iter()
                .map(|&p| PressurePoint::new(p, pressure))
                .collect(),
        }
    }

    /// Creates a pressure stroke from a path with uniform pressure.
    pub fn from_path(path: &Path, pressure: f32) -> Self {
        let points = path_to_points(path);
        Self::from_points(&points, pressure)
    }

    /// Adds a point to the stroke.
    pub fn add_point(&mut self, position: Vec2, pressure: f32) {
        self.points.push(PressurePoint::new(position, pressure));
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the stroke is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Applies a taper to the start of the stroke.
    ///
    /// `length` is the distance over which to taper (0.0 to 1.0 as fraction of total length).
    pub fn taper_start(mut self, length: f32) -> Self {
        if self.points.len() < 2 {
            return self;
        }

        let total_len = self.total_length();
        let taper_dist = total_len * length.clamp(0.0, 1.0);

        let mut dist = 0.0;
        for i in 0..self.points.len() {
            if dist < taper_dist {
                let t = dist / taper_dist;
                self.points[i].pressure *= t;
            }

            if i < self.points.len() - 1 {
                dist += (self.points[i + 1].position - self.points[i].position).length();
            }
        }

        self
    }

    /// Applies a taper to the end of the stroke.
    ///
    /// `length` is the distance over which to taper (0.0 to 1.0 as fraction of total length).
    pub fn taper_end(mut self, length: f32) -> Self {
        if self.points.len() < 2 {
            return self;
        }

        let total_len = self.total_length();
        let taper_dist = total_len * length.clamp(0.0, 1.0);

        // Calculate distances from end
        let mut dists = vec![0.0; self.points.len()];
        let mut dist = 0.0;
        for i in (0..self.points.len() - 1).rev() {
            dist += (self.points[i + 1].position - self.points[i].position).length();
            dists[i] = dist;
        }

        for (i, &d) in dists.iter().enumerate() {
            if d < taper_dist {
                let t = d / taper_dist;
                self.points[i].pressure *= t;
            }
        }

        self
    }

    /// Applies smoothing to the pressure values.
    pub fn smooth_pressure(mut self, iterations: usize) -> Self {
        for _ in 0..iterations {
            let mut new_pressures = Vec::with_capacity(self.points.len());

            for i in 0..self.points.len() {
                let prev = if i > 0 {
                    self.points[i - 1].pressure
                } else {
                    self.points[i].pressure
                };
                let curr = self.points[i].pressure;
                let next = if i < self.points.len() - 1 {
                    self.points[i + 1].pressure
                } else {
                    self.points[i].pressure
                };

                new_pressures.push((prev + curr * 2.0 + next) / 4.0);
            }

            for (i, p) in new_pressures.into_iter().enumerate() {
                self.points[i].pressure = p;
            }
        }

        self
    }

    /// Calculates total length of the stroke.
    fn total_length(&self) -> f32 {
        let mut len = 0.0;
        for i in 0..self.points.len().saturating_sub(1) {
            len += (self.points[i + 1].position - self.points[i].position).length();
        }
        len
    }
}

impl Default for PressureStroke {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation for pressure-sensitive stroke rendering.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PressureStroke, output = Path))]
pub struct PressureStrokeRender {
    /// Minimum stroke width (at pressure 0).
    pub min_width: f32,
    /// Maximum stroke width (at pressure 1).
    pub max_width: f32,
    /// Cap style for stroke ends.
    pub cap: CapStyle,
    /// Join style for corners.
    pub join: JoinStyle,
    /// Miter limit.
    pub miter_limit: f32,
}

impl Default for PressureStrokeRender {
    fn default() -> Self {
        Self {
            min_width: 0.5,
            max_width: 5.0,
            cap: CapStyle::Round,
            join: JoinStyle::Round,
            miter_limit: 4.0,
        }
    }
}

impl PressureStrokeRender {
    /// Creates a new pressure stroke config.
    pub fn new(min_width: f32, max_width: f32) -> Self {
        Self {
            min_width,
            max_width,
            ..Default::default()
        }
    }

    /// Calculates width for a given pressure.
    pub fn width_for_pressure(&self, pressure: f32) -> f32 {
        self.min_width + (self.max_width - self.min_width) * pressure.clamp(0.0, 1.0)
    }

    /// Applies this operation to a pressure stroke, converting it to a filled path outline.
    pub fn apply(&self, stroke: &PressureStroke) -> Path {
        pressure_stroke_to_path(stroke, self)
    }
}

/// Backwards-compatible type alias.
pub type PressureStrokeConfig = PressureStrokeRender;

/// Converts a pressure stroke to a filled path outline.
///
/// Creates a path that represents the variable-width stroke as a filled shape.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_vector::stroke::{PressureStroke, PressureStrokeConfig, pressure_stroke_to_path};
/// use glam::Vec2;
///
/// let mut stroke = PressureStroke::new();
/// stroke.add_point(Vec2::new(0.0, 0.0), 0.5);
/// stroke.add_point(Vec2::new(50.0, 0.0), 1.0);
/// stroke.add_point(Vec2::new(100.0, 0.0), 0.2);
///
/// let config = PressureStrokeConfig::new(1.0, 10.0);
/// let outline = pressure_stroke_to_path(&stroke, &config);
/// ```
pub fn pressure_stroke_to_path(stroke: &PressureStroke, config: &PressureStrokeConfig) -> Path {
    if stroke.len() < 2 {
        return Path::new();
    }

    let points = &stroke.points;
    let mut left_points = Vec::with_capacity(points.len());
    let mut right_points = Vec::with_capacity(points.len());

    for i in 0..points.len() {
        let curr = points[i].position;
        let half_width = config.width_for_pressure(points[i].pressure) * 0.5;

        // Get edge directions
        let (prev_dir, next_dir) = if i == 0 {
            let dir = (points[i + 1].position - curr).normalize_or_zero();
            (dir, dir)
        } else if i == points.len() - 1 {
            let dir = (curr - points[i - 1].position).normalize_or_zero();
            (dir, dir)
        } else {
            let prev = points[i - 1].position;
            let next = points[i + 1].position;
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
    if !right_points.is_empty() {
        let end_width = config.width_for_pressure(points.last().unwrap().pressure) * 0.5;
        builder = add_cap(
            builder,
            points.last().unwrap().position,
            *left_points.last().unwrap(),
            *right_points.last().unwrap(),
            config.cap,
            end_width,
            true,
        );
    }

    // Right side (backward)
    for &p in right_points.iter().rev() {
        builder = builder.line_to(p);
    }

    // Start cap
    if !right_points.is_empty() {
        let start_width = config.width_for_pressure(points[0].pressure) * 0.5;
        builder = add_cap(
            builder,
            points[0].position,
            right_points[0],
            left_points[0],
            config.cap,
            start_width,
            false,
        );
    }

    builder = builder.close();
    builder.build()
}

/// Simulates pen pressure along a path based on velocity.
///
/// Faster movement results in lighter pressure (thinner strokes).
///
/// # Arguments
/// * `path` - The input path
/// * `speed_factor` - How much velocity affects pressure (0.0 = no effect, 1.0 = strong effect)
pub fn simulate_velocity_pressure(path: &Path, speed_factor: f32) -> PressureStroke {
    let points = path_to_points(path);
    if points.len() < 2 {
        return PressureStroke::from_points(&points, 1.0);
    }

    // Calculate velocities (using segment lengths as proxy)
    let mut velocities = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let vel = if i == 0 {
            (points[1] - points[0]).length()
        } else if i == points.len() - 1 {
            (points[i] - points[i - 1]).length()
        } else {
            ((points[i + 1] - points[i]).length() + (points[i] - points[i - 1]).length()) * 0.5
        };
        velocities.push(vel);
    }

    // Find max velocity for normalization
    let max_vel = velocities.iter().cloned().fold(0.0f32, f32::max).max(0.001);

    // Convert velocity to pressure (inverse relationship)
    let pressure_points: Vec<PressurePoint> = points
        .iter()
        .zip(velocities.iter())
        .map(|(&pos, &vel)| {
            let normalized_vel = vel / max_vel;
            let pressure = 1.0 - normalized_vel * speed_factor.clamp(0.0, 1.0);
            PressurePoint::new(pos, pressure)
        })
        .collect();

    PressureStroke {
        points: pressure_points,
    }
}

/// Simulates pressure with acceleration at stroke start/end.
///
/// Creates natural-looking pen strokes with gradual buildup and release.
pub fn simulate_natural_pressure(path: &Path, taper_start: f32, taper_end: f32) -> PressureStroke {
    let stroke = PressureStroke::from_path(path, 1.0);
    stroke.taper_start(taper_start).taper_end(taper_end)
}

// ===========================================================================
// Path Trim (Stroke Reveal Animation)
// ===========================================================================

/// Operation for trimming a path to a portion of its length.
///
/// This is the core building block for "stroke reveal" animations where
/// a path is drawn from 0% to 100% over time.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Path, output = Path))]
pub struct Trim {
    /// Start position (0.0 to 1.0).
    pub start: f32,
    /// End position (0.0 to 1.0).
    pub end: f32,
}

impl Default for Trim {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
        }
    }
}

impl Trim {
    /// Creates a trim operation with the given start and end positions.
    ///
    /// Both `start` and `end` are in the range 0.0 to 1.0, representing
    /// percentage along the path length.
    pub fn new(start: f32, end: f32) -> Self {
        Self { start, end }
    }

    /// Creates a trim operation starting from the beginning.
    ///
    /// Useful for "draw on" animations: `Trim::from_start(t)` where t goes 0→1.
    pub fn from_start(end: f32) -> Self {
        Self { start: 0.0, end }
    }

    /// Creates a trim operation ending at the end.
    ///
    /// Useful for "draw off" animations: `Trim::to_end(t)` where t goes 0→1.
    pub fn to_end(start: f32) -> Self {
        Self { start, end: 1.0 }
    }

    /// Applies the trim operation to a path.
    pub fn apply(&self, path: &Path) -> Path {
        trim_path(path, self.start, self.end)
    }
}

/// Result of trimming a path, including tangent information at endpoints.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// The trimmed path.
    pub path: Path,
    /// Tangent direction at the start of the trimmed path.
    pub start_tangent: Option<Vec2>,
    /// Tangent direction at the end of the trimmed path.
    pub end_tangent: Option<Vec2>,
}

/// Trims a path to a portion of its length.
///
/// Returns the portion of the path between `start` and `end`, where both
/// values are in the range 0.0 to 1.0 (percentage of total path length).
///
/// If `start > end`, the path segment is reversed.
///
/// # Arguments
/// * `path` - The path to trim.
/// * `start` - Start position (0.0 to 1.0).
/// * `end` - End position (0.0 to 1.0).
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_vector::{circle, stroke::trim_path};
/// use glam::Vec2;
///
/// let path = circle(Vec2::ZERO, 100.0);
///
/// // First half of the circle
/// let first_half = trim_path(&path, 0.0, 0.5);
///
/// // Animate stroke reveal (in animation loop)
/// let t = 0.5; // Animation progress
/// let revealed = trim_path(&path, 0.0, t);
/// ```
pub fn trim_path(path: &Path, start: f32, end: f32) -> Path {
    let start = start.clamp(0.0, 1.0);
    let end = end.clamp(0.0, 1.0);

    if (end - start).abs() < 1e-6 {
        return Path::new();
    }

    let points = path_to_points(path);
    if points.len() < 2 {
        return path.clone();
    }

    let total_len = path_length(path);
    if total_len < 1e-6 {
        return path.clone();
    }

    let (start_dist, end_dist, reverse) = if start <= end {
        (start * total_len, end * total_len, false)
    } else {
        (end * total_len, start * total_len, true)
    };

    // Sample points between start and end distances
    let mut trimmed_points = Vec::new();

    // Add start point
    if let Some(p) = sample_path_at_distance(&points, start_dist) {
        trimmed_points.push(p);
    }

    // Add intermediate points that fall within the range
    let mut accumulated_dist = 0.0;
    for i in 0..points.len() - 1 {
        let segment_len = (points[i + 1] - points[i]).length();
        let segment_end = accumulated_dist + segment_len;

        // If this point falls within our range, add it
        if accumulated_dist > start_dist && accumulated_dist < end_dist {
            // Check if this point is significantly different from the last added point
            if let Some(&last) = trimmed_points.last() {
                if (points[i] - last).length() > 1e-6 {
                    trimmed_points.push(points[i]);
                }
            }
        }

        // Same for the next point
        if segment_end > start_dist && segment_end < end_dist && i + 1 < points.len() {
            if let Some(&last) = trimmed_points.last() {
                if (points[i + 1] - last).length() > 1e-6 {
                    trimmed_points.push(points[i + 1]);
                }
            }
        }

        accumulated_dist = segment_end;
    }

    // Add end point
    if let Some(p) = sample_path_at_distance(&points, end_dist) {
        if let Some(&last) = trimmed_points.last() {
            if (p - last).length() > 1e-6 {
                trimmed_points.push(p);
            }
        } else {
            trimmed_points.push(p);
        }
    }

    if reverse {
        trimmed_points.reverse();
    }

    points_to_path(&trimmed_points, false)
}

/// Trims a path and returns additional information about the endpoints.
///
/// This is useful when you need to draw arrowheads or other decorations
/// at the trim endpoints.
pub fn trim_path_with_tangents(path: &Path, start: f32, end: f32) -> TrimResult {
    let start = start.clamp(0.0, 1.0);
    let end = end.clamp(0.0, 1.0);

    if (end - start).abs() < 1e-6 {
        return TrimResult {
            path: Path::new(),
            start_tangent: None,
            end_tangent: None,
        };
    }

    let total_len = path_length(path);
    if total_len < 1e-6 {
        return TrimResult {
            path: path.clone(),
            start_tangent: None,
            end_tangent: None,
        };
    }

    let (actual_start, actual_end, reverse) = if start <= end {
        (start, end, false)
    } else {
        (end, start, true)
    };

    let trimmed = trim_path(path, actual_start, actual_end);

    let start_tangent = tangent_at_length(path, actual_start * total_len);
    let end_tangent = tangent_at_length(path, actual_end * total_len);

    if reverse {
        TrimResult {
            path: trimmed,
            start_tangent: end_tangent.map(|t| -t),
            end_tangent: start_tangent.map(|t| -t),
        }
    } else {
        TrimResult {
            path: trimmed,
            start_tangent,
            end_tangent,
        }
    }
}

/// Creates multiple trim segments from a path (for dashed stroke reveal).
///
/// This is useful for creating animated dashed lines where each dash
/// appears progressively.
///
/// # Arguments
/// * `path` - The path to segment.
/// * `num_segments` - Number of segments to create.
/// * `gap_ratio` - Ratio of gap to segment (0.0 = no gaps, 0.5 = equal gaps).
/// * `progress` - Animation progress (0.0 to 1.0).
pub fn trim_segments(path: &Path, num_segments: usize, gap_ratio: f32, progress: f32) -> Vec<Path> {
    if num_segments == 0 || progress <= 0.0 {
        return Vec::new();
    }

    let gap_ratio = gap_ratio.clamp(0.0, 0.9);
    let segment_size = 1.0 / num_segments as f32;
    let gap_size = segment_size * gap_ratio;
    let dash_size = segment_size - gap_size;

    let mut segments = Vec::with_capacity(num_segments);

    for i in 0..num_segments {
        let segment_start = i as f32 * segment_size;
        let segment_end = segment_start + dash_size;

        // Apply progress to this segment
        let reveal_point = progress * (1.0 + gap_ratio);
        let segment_progress = ((reveal_point - segment_start) / dash_size).clamp(0.0, 1.0);

        if segment_progress > 0.0 {
            let actual_end = segment_start + dash_size * segment_progress;
            let trimmed = trim_path(path, segment_start, actual_end.min(segment_end));
            if !trimmed.is_empty() {
                segments.push(trimmed);
            }
        }
    }

    segments
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
    fn test_stroke_config_struct_init() {
        let config = StrokeConfig {
            width: 5.0,
            cap: CapStyle::Round,
            join: JoinStyle::Bevel,
            ..StrokeConfig::default()
        };
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
        let config = StrokeConfig {
            width: 10.0,
            cap: CapStyle::Round,
            ..StrokeConfig::default()
        };
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

    // Pressure stroke tests

    #[test]
    fn test_pressure_point() {
        let p = PressurePoint::new(Vec2::new(10.0, 20.0), 0.5);
        assert_eq!(p.position, Vec2::new(10.0, 20.0));
        assert_eq!(p.pressure, 0.5);

        // Pressure should be clamped
        let p_clamped = PressurePoint::new(Vec2::ZERO, 1.5);
        assert_eq!(p_clamped.pressure, 1.0);
    }

    #[test]
    fn test_pressure_stroke_basic() {
        let mut stroke = PressureStroke::new();
        assert!(stroke.is_empty());

        stroke.add_point(Vec2::ZERO, 0.5);
        stroke.add_point(Vec2::new(10.0, 0.0), 1.0);
        assert_eq!(stroke.len(), 2);
    }

    #[test]
    fn test_pressure_stroke_from_points() {
        let points = vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(20.0, 0.0)];
        let stroke = PressureStroke::from_points(&points, 0.7);

        assert_eq!(stroke.len(), 3);
        for p in &stroke.points {
            assert!((p.pressure - 0.7).abs() < 0.001);
        }
    }

    #[test]
    fn test_pressure_stroke_taper_start() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(25.0, 0.0),
            Vec2::new(50.0, 0.0),
            Vec2::new(75.0, 0.0),
            Vec2::new(100.0, 0.0),
        ];
        let stroke = PressureStroke::from_points(&points, 1.0).taper_start(0.5);

        // First point should have low pressure
        assert!(stroke.points[0].pressure < 0.1);

        // Last point should still have full pressure
        assert!(stroke.points[4].pressure > 0.9);
    }

    #[test]
    fn test_pressure_stroke_taper_end() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(25.0, 0.0),
            Vec2::new(50.0, 0.0),
            Vec2::new(75.0, 0.0),
            Vec2::new(100.0, 0.0),
        ];
        let stroke = PressureStroke::from_points(&points, 1.0).taper_end(0.5);

        // First point should still have full pressure
        assert!(stroke.points[0].pressure > 0.9);

        // Last point should have low pressure
        assert!(stroke.points[4].pressure < 0.1);
    }

    #[test]
    fn test_pressure_stroke_smooth() {
        let mut stroke = PressureStroke::new();
        stroke.add_point(Vec2::new(0.0, 0.0), 0.0);
        stroke.add_point(Vec2::new(10.0, 0.0), 1.0);
        stroke.add_point(Vec2::new(20.0, 0.0), 0.0);

        let smoothed = stroke.smooth_pressure(1);

        // Middle point should be smoothed toward neighbors
        assert!(smoothed.points[1].pressure < 1.0);
    }

    #[test]
    fn test_pressure_stroke_config() {
        let config = PressureStrokeConfig::new(1.0, 10.0);
        assert_eq!(config.min_width, 1.0);
        assert_eq!(config.max_width, 10.0);

        // Test width calculation
        assert_eq!(config.width_for_pressure(0.0), 1.0);
        assert_eq!(config.width_for_pressure(1.0), 10.0);
        assert_eq!(config.width_for_pressure(0.5), 5.5);
    }

    #[test]
    fn test_pressure_stroke_to_path() {
        let mut stroke = PressureStroke::new();
        stroke.add_point(Vec2::new(0.0, 0.0), 0.5);
        stroke.add_point(Vec2::new(50.0, 0.0), 1.0);
        stroke.add_point(Vec2::new(100.0, 0.0), 0.2);

        let config = PressureStrokeConfig::new(2.0, 10.0);
        let outline = pressure_stroke_to_path(&stroke, &config);

        // Should produce a valid closed path
        assert!(!outline.is_empty());
    }

    #[test]
    fn test_pressure_stroke_to_path_varying_width() {
        let mut stroke = PressureStroke::new();
        stroke.add_point(Vec2::new(0.0, 0.0), 0.0);
        stroke.add_point(Vec2::new(100.0, 0.0), 1.0);

        let config = PressureStrokeConfig::new(1.0, 20.0);
        let outline = pressure_stroke_to_path(&stroke, &config);

        // Should produce a wedge-shaped outline
        assert!(!outline.is_empty());
    }

    #[test]
    fn test_simulate_velocity_pressure() {
        // Create a path with varying segment lengths
        let path = PathBuilder::new()
            .move_to(Vec2::new(0.0, 0.0))
            .line_to(Vec2::new(10.0, 0.0)) // Short segment
            .line_to(Vec2::new(60.0, 0.0)) // Long segment (fast movement)
            .line_to(Vec2::new(70.0, 0.0)) // Short segment (slow movement)
            .build();

        let stroke = simulate_velocity_pressure(&path, 0.8);

        // All points should have valid pressure
        for p in &stroke.points {
            assert!(p.pressure >= 0.0 && p.pressure <= 1.0);
        }
    }

    #[test]
    fn test_simulate_natural_pressure() {
        // Use a path with multiple points for proper taper testing
        let path = PathBuilder::new()
            .move_to(Vec2::new(0.0, 0.0))
            .line_to(Vec2::new(25.0, 0.0))
            .line_to(Vec2::new(50.0, 0.0))
            .line_to(Vec2::new(75.0, 0.0))
            .line_to(Vec2::new(100.0, 0.0))
            .build();

        let stroke = simulate_natural_pressure(&path, 0.3, 0.3);

        // Should have 5 points
        assert_eq!(stroke.len(), 5);

        // First and last points should have lower pressure than middle
        let first_pressure = stroke.points[0].pressure;
        let middle_pressure = stroke.points[2].pressure;
        let last_pressure = stroke.points[4].pressure;

        assert!(first_pressure < middle_pressure);
        assert!(last_pressure < middle_pressure);
    }

    // Path trim tests

    #[test]
    fn test_trim_default() {
        let trim = Trim::default();
        assert_eq!(trim.start, 0.0);
        assert_eq!(trim.end, 1.0);
    }

    #[test]
    fn test_trim_new() {
        let trim = Trim::new(0.25, 0.75);
        assert_eq!(trim.start, 0.25);
        assert_eq!(trim.end, 0.75);
    }

    #[test]
    fn test_trim_from_start() {
        let trim = Trim::from_start(0.5);
        assert_eq!(trim.start, 0.0);
        assert_eq!(trim.end, 0.5);
    }

    #[test]
    fn test_trim_to_end() {
        let trim = Trim::to_end(0.3);
        assert_eq!(trim.start, 0.3);
        assert_eq!(trim.end, 1.0);
    }

    #[test]
    fn test_trim_path_full() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let trimmed = trim_path(&path, 0.0, 1.0);

        // Full trim should preserve path
        assert!(!trimmed.is_empty());
        let len = path_length(&trimmed);
        assert!((len - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_first_half() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let trimmed = trim_path(&path, 0.0, 0.5);

        let len = path_length(&trimmed);
        assert!((len - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_second_half() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let trimmed = trim_path(&path, 0.5, 1.0);

        let len = path_length(&trimmed);
        assert!((len - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_middle() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let trimmed = trim_path(&path, 0.25, 0.75);

        let len = path_length(&trimmed);
        assert!((len - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_empty() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // Same start and end should give empty path
        let trimmed = trim_path(&path, 0.5, 0.5);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_trim_path_reversed() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // start > end should reverse the path
        let trimmed = trim_path(&path, 0.75, 0.25);
        let len = path_length(&trimmed);
        assert!((len - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_clamping() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // Values outside 0-1 should be clamped
        let trimmed = trim_path(&path, -0.5, 1.5);
        let len = path_length(&trimmed);
        assert!((len - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_circle() {
        let path = circle(Vec2::ZERO, 50.0);
        let full_len = path_length(&path);

        // Quarter of circle
        let trimmed = trim_path(&path, 0.0, 0.25);
        let trimmed_len = path_length(&trimmed);

        assert!((trimmed_len - full_len * 0.25).abs() < 2.0);
    }

    #[test]
    fn test_trim_apply() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let trim = Trim::new(0.0, 0.5);
        let trimmed = trim.apply(&path);

        let len = path_length(&trimmed);
        assert!((len - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_trim_path_with_tangents() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let result = trim_path_with_tangents(&path, 0.0, 0.5);

        assert!(!result.path.is_empty());
        assert!(result.start_tangent.is_some());
        assert!(result.end_tangent.is_some());

        // Tangents should point along x-axis
        let start_t = result.start_tangent.unwrap();
        assert!((start_t.x - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_trim_segments_basic() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // 4 segments, no gaps, full progress
        let segments = trim_segments(&path, 4, 0.0, 1.0);
        assert_eq!(segments.len(), 4);

        // Each segment should be about 25 units
        for seg in &segments {
            let len = path_length(seg);
            assert!((len - 25.0).abs() < 2.0);
        }
    }

    #[test]
    fn test_trim_segments_with_gaps() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // 2 segments, 50% gap ratio, full progress
        let segments = trim_segments(&path, 2, 0.5, 1.0);
        assert_eq!(segments.len(), 2);

        // Each segment should be 25 units (50% of 50 unit block)
        for seg in &segments {
            let len = path_length(seg);
            assert!((len - 25.0).abs() < 2.0);
        }
    }

    #[test]
    fn test_trim_segments_partial_progress() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        // 4 segments, no gaps, 50% progress
        let segments = trim_segments(&path, 4, 0.0, 0.5);

        // Should have 2 full segments revealed
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_trim_segments_zero_progress() {
        let path = crate::line(Vec2::ZERO, Vec2::new(100.0, 0.0));

        let segments = trim_segments(&path, 4, 0.0, 0.0);
        assert!(segments.is_empty());
    }
}
