//! Hatching patterns for non-photorealistic rendering.
//!
//! Provides parallel line hatching and cross-hatching for creating
//! shading effects in 2D vector graphics.
//!
//! # Example
//!
//! ```
//! use glam::Vec2;
//! use rhizome_resin_vector::{Hatch, hatch_rect, cross_hatch_rect};
//!
//! // Simple parallel hatching
//! let config = Hatch { spacing: 5.0, angle: 45.0, ..Hatch::default() };
//! let lines = hatch_rect(Vec2::ZERO, Vec2::new(100.0, 100.0), &config);
//!
//! // Cross-hatching
//! let lines = cross_hatch_rect(Vec2::ZERO, Vec2::new(100.0, 100.0), &config, 90.0);
//! ```

use crate::Path;
use glam::Vec2;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for hatching patterns.
///
/// Note: This is a configuration struct rather than a full Op because hatching
/// requires bounds/polygon input which isn't a single serializable type.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hatch {
    /// Spacing between hatch lines.
    pub spacing: f32,
    /// Angle of hatch lines in degrees (0 = horizontal).
    pub angle: f32,
    /// Offset for the first line.
    pub offset: f32,
}

impl Default for Hatch {
    fn default() -> Self {
        Self {
            spacing: 5.0,
            angle: 45.0,
            offset: 0.0,
        }
    }
}

impl Hatch {
    /// Creates a new hatch configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates horizontal hatching (angle = 0).
    pub fn horizontal() -> Self {
        Self {
            angle: 0.0,
            ..Self::default()
        }
    }

    /// Creates vertical hatching (angle = 90).
    pub fn vertical() -> Self {
        Self {
            angle: 90.0,
            ..Self::default()
        }
    }

    /// Creates diagonal hatching (angle = 45).
    pub fn diagonal() -> Self {
        Self {
            angle: 45.0,
            ..Self::default()
        }
    }
}

/// Backwards-compatible type alias.
pub type HatchConfig = Hatch;

/// A single hatch line segment.
#[derive(Debug, Clone, Copy)]
pub struct HatchLine {
    /// Start point of the line.
    pub start: Vec2,
    /// End point of the line.
    pub end: Vec2,
}

impl HatchLine {
    /// Creates a new hatch line.
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }

    /// Returns the length of the line.
    pub fn length(&self) -> f32 {
        (self.end - self.start).length()
    }

    /// Converts to a Path.
    pub fn to_path(&self) -> Path {
        crate::line(self.start, self.end)
    }
}

/// Generates parallel hatch lines within a rectangular region.
///
/// # Arguments
///
/// * `min` - Minimum corner of the rectangle
/// * `max` - Maximum corner of the rectangle
/// * `config` - Hatching configuration
///
/// # Returns
///
/// A vector of line segments representing the hatch pattern.
pub fn hatch_rect(min: Vec2, max: Vec2, config: &HatchConfig) -> Vec<HatchLine> {
    let angle_rad = config.angle.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Direction along the hatch line
    let dir = Vec2::new(cos_a, sin_a);
    // Perpendicular direction (for spacing)
    let perp = Vec2::new(-sin_a, cos_a);

    // Find the bounding box corners
    let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];

    // Project corners onto perpendicular axis to find range
    let perp_projections: Vec<f32> = corners.iter().map(|c| c.dot(perp)).collect();
    let perp_min = perp_projections
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let perp_max = perp_projections
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Project corners onto direction axis to find line extent
    let dir_projections: Vec<f32> = corners.iter().map(|c| c.dot(dir)).collect();
    let dir_min = dir_projections
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let dir_max = dir_projections
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut lines = Vec::new();

    // Generate lines at regular spacing
    let start_perp = (perp_min / config.spacing).floor() * config.spacing + config.offset;

    let mut perp_pos = start_perp;
    while perp_pos <= perp_max {
        // Create a line at this perpendicular position
        let line_origin = perp * perp_pos;
        let line_start = line_origin + dir * dir_min;
        let line_end = line_origin + dir * dir_max;

        // Clip line to rectangle
        if let Some((clipped_start, clipped_end)) =
            clip_line_to_rect(line_start, line_end, min, max)
        {
            lines.push(HatchLine::new(clipped_start, clipped_end));
        }

        perp_pos += config.spacing;
    }

    lines
}

/// Generates cross-hatch lines (two sets of parallel lines at different angles).
///
/// # Arguments
///
/// * `min` - Minimum corner of the rectangle
/// * `max` - Maximum corner of the rectangle
/// * `config` - Configuration for the first set of lines
/// * `cross_angle` - Angle offset for the second set of lines (in degrees)
///
/// # Returns
///
/// A vector of line segments representing the cross-hatch pattern.
pub fn cross_hatch_rect(
    min: Vec2,
    max: Vec2,
    config: &HatchConfig,
    cross_angle: f32,
) -> Vec<HatchLine> {
    let mut lines = hatch_rect(min, max, config);

    let cross_config = HatchConfig {
        spacing: config.spacing,
        angle: config.angle + cross_angle,
        offset: config.offset,
    };

    lines.extend(hatch_rect(min, max, &cross_config));
    lines
}

/// Generates hatch lines within a polygon.
///
/// # Arguments
///
/// * `polygon` - Vertices of the polygon (in order)
/// * `config` - Hatching configuration
///
/// # Returns
///
/// A vector of line segments clipped to the polygon boundary.
pub fn hatch_polygon(polygon: &[Vec2], config: &HatchConfig) -> Vec<HatchLine> {
    if polygon.len() < 3 {
        return Vec::new();
    }

    // Find bounding box
    let mut min = polygon[0];
    let mut max = polygon[0];
    for &p in polygon {
        min = min.min(p);
        max = max.max(p);
    }

    // Generate lines for bounding box, then clip to polygon
    let angle_rad = config.angle.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let dir = Vec2::new(cos_a, sin_a);
    let perp = Vec2::new(-sin_a, cos_a);

    // Expand bounding box slightly for rotated lines
    let padding = (max - min).length() * 0.5;
    let expanded_min = min - Vec2::splat(padding);
    let expanded_max = max + Vec2::splat(padding);

    let corners = [
        expanded_min,
        Vec2::new(expanded_max.x, expanded_min.y),
        expanded_max,
        Vec2::new(expanded_min.x, expanded_max.y),
    ];

    let perp_projections: Vec<f32> = corners.iter().map(|c| c.dot(perp)).collect();
    let perp_min = perp_projections
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let perp_max = perp_projections
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let dir_projections: Vec<f32> = corners.iter().map(|c| c.dot(dir)).collect();
    let dir_min = dir_projections
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let dir_max = dir_projections
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut lines = Vec::new();
    let start_perp = (perp_min / config.spacing).floor() * config.spacing + config.offset;

    let mut perp_pos = start_perp;
    while perp_pos <= perp_max {
        let line_origin = perp * perp_pos;
        let line_start = line_origin + dir * dir_min;
        let line_end = line_origin + dir * dir_max;

        // Clip line to polygon
        let clipped = clip_line_to_polygon(line_start, line_end, polygon);
        lines.extend(clipped);

        perp_pos += config.spacing;
    }

    lines
}

/// Generates cross-hatch lines within a polygon.
pub fn cross_hatch_polygon(
    polygon: &[Vec2],
    config: &HatchConfig,
    cross_angle: f32,
) -> Vec<HatchLine> {
    let mut lines = hatch_polygon(polygon, config);

    let cross_config = HatchConfig {
        spacing: config.spacing,
        angle: config.angle + cross_angle,
        offset: config.offset,
    };

    lines.extend(hatch_polygon(polygon, &cross_config));
    lines
}

/// Clips a line segment to a rectangle using Cohen-Sutherland algorithm.
fn clip_line_to_rect(start: Vec2, end: Vec2, min: Vec2, max: Vec2) -> Option<(Vec2, Vec2)> {
    const INSIDE: u8 = 0;
    const LEFT: u8 = 1;
    const RIGHT: u8 = 2;
    const BOTTOM: u8 = 4;
    const TOP: u8 = 8;

    fn compute_code(p: Vec2, min: Vec2, max: Vec2) -> u8 {
        let mut code = INSIDE;
        if p.x < min.x {
            code |= LEFT;
        } else if p.x > max.x {
            code |= RIGHT;
        }
        if p.y < min.y {
            code |= BOTTOM;
        } else if p.y > max.y {
            code |= TOP;
        }
        code
    }

    let mut p0 = start;
    let mut p1 = end;
    let mut code0 = compute_code(p0, min, max);
    let mut code1 = compute_code(p1, min, max);

    loop {
        if code0 == 0 && code1 == 0 {
            // Both inside
            return Some((p0, p1));
        } else if (code0 & code1) != 0 {
            // Both outside same region
            return None;
        } else {
            // Line needs clipping
            let code_out = if code0 != 0 { code0 } else { code1 };

            let x;
            let y;

            if (code_out & TOP) != 0 {
                x = p0.x + (p1.x - p0.x) * (max.y - p0.y) / (p1.y - p0.y);
                y = max.y;
            } else if (code_out & BOTTOM) != 0 {
                x = p0.x + (p1.x - p0.x) * (min.y - p0.y) / (p1.y - p0.y);
                y = min.y;
            } else if (code_out & RIGHT) != 0 {
                y = p0.y + (p1.y - p0.y) * (max.x - p0.x) / (p1.x - p0.x);
                x = max.x;
            } else {
                y = p0.y + (p1.y - p0.y) * (min.x - p0.x) / (p1.x - p0.x);
                x = min.x;
            }

            if code_out == code0 {
                p0 = Vec2::new(x, y);
                code0 = compute_code(p0, min, max);
            } else {
                p1 = Vec2::new(x, y);
                code1 = compute_code(p1, min, max);
            }
        }
    }
}

/// Clips a line to a polygon, returning multiple segments if the line crosses in/out.
fn clip_line_to_polygon(start: Vec2, end: Vec2, polygon: &[Vec2]) -> Vec<HatchLine> {
    if polygon.len() < 3 {
        return Vec::new();
    }

    let dir = end - start;
    let len = dir.length();
    if len < 1e-10 {
        return Vec::new();
    }

    // Find all intersections with polygon edges
    let mut intersections = Vec::new();

    let n = polygon.len();
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];

        if let Some(t) = line_segment_intersection(start, end, a, b) {
            if t >= 0.0 && t <= 1.0 {
                intersections.push(t);
            }
        }
    }

    // Sort intersections by parameter
    intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Remove duplicates
    intersections.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    // Check if start is inside polygon
    let start_inside = point_in_polygon(start, polygon);

    let mut lines = Vec::new();
    let mut inside = start_inside;
    let mut enter_t = if start_inside { 0.0 } else { -1.0 };

    for t in intersections {
        if inside {
            // Exiting polygon
            let exit_point = start + dir * t;
            let enter_point = start + dir * enter_t;
            if (exit_point - enter_point).length() > 1e-6 {
                lines.push(HatchLine::new(enter_point, exit_point));
            }
        } else {
            // Entering polygon
            enter_t = t;
        }
        inside = !inside;
    }

    // Handle case where line ends inside polygon
    if inside && enter_t >= 0.0 {
        let enter_point = start + dir * enter_t;
        if (end - enter_point).length() > 1e-6 {
            lines.push(HatchLine::new(enter_point, end));
        }
    }

    lines
}

/// Finds the parameter t where two line segments intersect.
fn line_segment_intersection(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Option<f32> {
    let s1 = p1 - p0;
    let s2 = p3 - p2;

    let denom = -s2.x * s1.y + s1.x * s2.y;
    if denom.abs() < 1e-10 {
        return None;
    }

    let s = (-s1.y * (p0.x - p2.x) + s1.x * (p0.y - p2.y)) / denom;
    let t = (s2.x * (p0.y - p2.y) - s2.y * (p0.x - p2.x)) / denom;

    if s >= 0.0 && s <= 1.0 { Some(t) } else { None }
}

/// Simple point-in-polygon test using ray casting.
fn point_in_polygon(point: Vec2, polygon: &[Vec2]) -> bool {
    let n = polygon.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];

        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Converts hatch lines to paths.
pub fn hatch_lines_to_paths(lines: &[HatchLine]) -> Vec<Path> {
    lines.iter().map(|l| l.to_path()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hatch_config() {
        let config = HatchConfig {
            spacing: 10.0,
            angle: 30.0,
            ..HatchConfig::default()
        };
        assert_eq!(config.spacing, 10.0);
        assert_eq!(config.angle, 30.0);
    }

    #[test]
    fn test_hatch_rect_horizontal() {
        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::horizontal()
        };
        let lines = hatch_rect(Vec2::ZERO, Vec2::new(100.0, 50.0), &config);

        // Should have multiple horizontal lines
        assert!(!lines.is_empty());

        // All lines should be roughly horizontal
        for line in &lines {
            assert!((line.start.y - line.end.y).abs() < 0.01);
        }
    }

    #[test]
    fn test_hatch_rect_vertical() {
        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::vertical()
        };
        let lines = hatch_rect(Vec2::ZERO, Vec2::new(50.0, 100.0), &config);

        assert!(!lines.is_empty());

        // All lines should be roughly vertical
        for line in &lines {
            assert!((line.start.x - line.end.x).abs() < 0.01);
        }
    }

    #[test]
    fn test_hatch_rect_diagonal() {
        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::diagonal()
        };
        let lines = hatch_rect(Vec2::ZERO, Vec2::new(100.0, 100.0), &config);

        assert!(!lines.is_empty());
    }

    #[test]
    fn test_cross_hatch_rect() {
        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::diagonal()
        };
        let lines = cross_hatch_rect(Vec2::ZERO, Vec2::new(100.0, 100.0), &config, 90.0);

        // Should have lines from both directions
        assert!(lines.len() > 10);
    }

    #[test]
    fn test_hatch_polygon_triangle() {
        let polygon = vec![
            Vec2::new(50.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ];

        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::horizontal()
        };
        let lines = hatch_polygon(&polygon, &config);

        assert!(!lines.is_empty());

        // All lines should be within the triangle's bounding box (roughly)
        for line in &lines {
            assert!(line.start.y >= -1.0 && line.start.y <= 101.0);
            assert!(line.end.y >= -1.0 && line.end.y <= 101.0);
        }
    }

    #[test]
    fn test_hatch_polygon_square() {
        let polygon = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ];

        let config = HatchConfig {
            spacing: 10.0,
            ..HatchConfig::horizontal()
        };
        let lines = hatch_polygon(&polygon, &config);

        // Should have roughly 10 lines (100 / 10)
        assert!(lines.len() >= 8 && lines.len() <= 12);
    }

    #[test]
    fn test_clip_line_to_rect() {
        // Line fully inside
        let result = clip_line_to_rect(
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
        );
        assert!(result.is_some());

        // Line fully outside
        let result = clip_line_to_rect(
            Vec2::new(-5.0, -5.0),
            Vec2::new(-1.0, -1.0),
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
        );
        assert!(result.is_none());

        // Line crossing
        let result = clip_line_to_rect(
            Vec2::new(-5.0, 5.0),
            Vec2::new(15.0, 5.0),
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
        );
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        assert!((start.x - 0.0).abs() < 0.01);
        assert!((end.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_hatch_line_to_path() {
        let line = HatchLine::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        let path = line.to_path();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_hatch_lines_to_paths() {
        let lines = vec![
            HatchLine::new(Vec2::ZERO, Vec2::new(10.0, 0.0)),
            HatchLine::new(Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0)),
        ];

        let paths = hatch_lines_to_paths(&lines);
        assert_eq!(paths.len(), 2);
    }
}
