//! 2D path rasterization to pixels.
//!
//! Converts vector paths to pixel masks and images.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_vector::rasterize::{Rasterizer, FillRule};
//! use rhizome_resin_vector::polygon;
//! use glam::Vec2;
//!
//! // Create a triangle path
//! let path = polygon(&[
//!     Vec2::new(50.0, 10.0),
//!     Vec2::new(90.0, 90.0),
//!     Vec2::new(10.0, 90.0),
//! ]);
//!
//! // Rasterize to a pixel buffer
//! let mut rasterizer = Rasterizer::new(100, 100);
//! rasterizer.add_path(&path);
//! let mask = rasterizer.rasterize(FillRule::NonZero);
//! ```

use crate::bezier::{cubic_point, quadratic_point};
use crate::path::{Path, PathCommand};
use crate::stroke::{JoinStyle, offset_path};
use glam::Vec2;

/// Fill rule for determining inside/outside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillRule {
    /// Non-zero winding rule.
    #[default]
    NonZero,
    /// Even-odd (parity) rule.
    EvenOdd,
}

/// A rasterizer for converting paths to pixel masks.
#[derive(Debug, Clone)]
pub struct Rasterizer {
    width: usize,
    height: usize,
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    direction: i32, // +1 for up, -1 for down
}

impl Rasterizer {
    /// Creates a new rasterizer with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            edges: Vec::new(),
        }
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Clears all edges.
    pub fn clear(&mut self) {
        self.edges.clear();
    }

    /// Adds a path to the rasterizer.
    pub fn add_path(&mut self, path: &Path) {
        let points = flatten_path(path, 32);

        if points.len() < 2 {
            return;
        }

        for window in points.windows(2) {
            self.add_edge(window[0], window[1]);
        }

        // Close the path if needed
        if let (Some(first), Some(last)) = (points.first(), points.last()) {
            if (*first - *last).length() > 0.001 {
                self.add_edge(*last, *first);
            }
        }
    }

    /// Adds a polygon to the rasterizer.
    pub fn add_polygon(&mut self, points: &[Vec2]) {
        if points.len() < 2 {
            return;
        }

        for i in 0..points.len() {
            let p0 = points[i];
            let p1 = points[(i + 1) % points.len()];
            self.add_edge(p0, p1);
        }
    }

    /// Adds a single edge.
    fn add_edge(&mut self, p0: Vec2, p1: Vec2) {
        // Skip horizontal edges
        if (p0.y - p1.y).abs() < 0.0001 {
            return;
        }

        // Ensure y0 < y1
        let (y0, y1, x0, x1, direction) = if p0.y < p1.y {
            (p0.y, p1.y, p0.x, p1.x, 1)
        } else {
            (p1.y, p0.y, p1.x, p0.x, -1)
        };

        self.edges.push(Edge {
            x0,
            y0,
            x1,
            y1,
            direction,
        });
    }

    /// Rasterizes the path to a mask (0.0 - 1.0 per pixel).
    pub fn rasterize(&self, fill_rule: FillRule) -> Vec<f32> {
        let mut mask = vec![0.0f32; self.width * self.height];

        // Sort edges by top y coordinate
        let mut sorted_edges = self.edges.clone();
        sorted_edges.sort_by(|a, b| a.y0.partial_cmp(&b.y0).unwrap());

        // Scanline algorithm
        for y in 0..self.height {
            let scanline_y = y as f32 + 0.5;

            // Collect all x intersections for this scanline
            let mut intersections: Vec<(f32, i32)> = Vec::new();

            for edge in &sorted_edges {
                // Skip if edge is entirely above or below scanline
                if scanline_y < edge.y0 || scanline_y >= edge.y1 {
                    continue;
                }

                // Calculate x intersection
                let t = (scanline_y - edge.y0) / (edge.y1 - edge.y0);
                let x = edge.x0 + t * (edge.x1 - edge.x0);

                intersections.push((x, edge.direction));
            }

            // Sort intersections by x
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Fill pixels based on fill rule
            match fill_rule {
                FillRule::NonZero => {
                    let mut winding = 0i32;
                    let mut prev_x = 0.0f32;

                    for (x, direction) in intersections {
                        if winding != 0 {
                            // Fill pixels from prev_x to x
                            let start_x = (prev_x.floor() as usize).max(0);
                            let end_x = (x.ceil() as usize).min(self.width);

                            for px in start_x..end_x {
                                let px_left = px as f32;
                                let px_right = (px + 1) as f32;

                                // Calculate coverage
                                let left = prev_x.max(px_left);
                                let right = x.min(px_right);

                                if right > left {
                                    let coverage = (right - left) / 1.0;
                                    mask[y * self.width + px] =
                                        (mask[y * self.width + px] + coverage).min(1.0);
                                }
                            }
                        }
                        winding += direction;
                        prev_x = x;
                    }
                }
                FillRule::EvenOdd => {
                    let mut inside = false;
                    let mut prev_x = 0.0f32;

                    for (x, _) in intersections {
                        if inside {
                            let start_x = (prev_x.floor() as usize).max(0);
                            let end_x = (x.ceil() as usize).min(self.width);

                            for px in start_x..end_x {
                                let px_left = px as f32;
                                let px_right = (px + 1) as f32;

                                let left = prev_x.max(px_left);
                                let right = x.min(px_right);

                                if right > left {
                                    let coverage = (right - left) / 1.0;
                                    mask[y * self.width + px] =
                                        (mask[y * self.width + px] + coverage).min(1.0);
                                }
                            }
                        }
                        inside = !inside;
                        prev_x = x;
                    }
                }
            }
        }

        mask
    }

    /// Rasterizes the path to a boolean mask.
    pub fn rasterize_bool(&self, fill_rule: FillRule) -> Vec<bool> {
        self.rasterize(fill_rule)
            .into_iter()
            .map(|v| v > 0.5)
            .collect()
    }

    /// Rasterizes the path to an RGBA image (single color).
    pub fn rasterize_rgba(&self, fill_rule: FillRule, color: [u8; 4]) -> Vec<u8> {
        let mask = self.rasterize(fill_rule);
        let mut pixels = vec![0u8; self.width * self.height * 4];

        for (i, &coverage) in mask.iter().enumerate() {
            let idx = i * 4;
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
            pixels[idx + 3] = (color[3] as f32 * coverage) as u8;
        }

        pixels
    }
}

/// Flattens a path to line segments.
fn flatten_path(path: &Path, segments_per_curve: usize) -> Vec<Vec2> {
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
                let start = current;
                for i in 1..=segments_per_curve {
                    let t = i as f32 / segments_per_curve as f32;
                    let p = quadratic_point(start, *control, *to, t);
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
                let start = current;
                for i in 1..=segments_per_curve {
                    let t = i as f32 / segments_per_curve as f32;
                    let p = cubic_point(start, *control1, *control2, *to, t);
                    points.push(p);
                }
                current = *to;
            }
            PathCommand::Close => {
                // Close is handled by the caller
            }
        }
    }

    points
}

/// Rasterizes a path to a mask.
pub fn rasterize_path(path: &Path, width: usize, height: usize, fill_rule: FillRule) -> Vec<f32> {
    let mut rasterizer = Rasterizer::new(width, height);
    rasterizer.add_path(path);
    rasterizer.rasterize(fill_rule)
}

/// Rasterizes a polygon to a mask.
pub fn rasterize_polygon(
    points: &[Vec2],
    width: usize,
    height: usize,
    fill_rule: FillRule,
) -> Vec<f32> {
    let mut rasterizer = Rasterizer::new(width, height);
    rasterizer.add_polygon(points);
    rasterizer.rasterize(fill_rule)
}

/// Rasterizes multiple paths to a mask.
pub fn rasterize_paths(
    paths: &[Path],
    width: usize,
    height: usize,
    fill_rule: FillRule,
) -> Vec<f32> {
    let mut rasterizer = Rasterizer::new(width, height);
    for path in paths {
        rasterizer.add_path(path);
    }
    rasterizer.rasterize(fill_rule)
}

/// Renders a path to an RGBA image with the given color.
pub fn render_path_rgba(
    path: &Path,
    width: usize,
    height: usize,
    fill_rule: FillRule,
    color: [u8; 4],
) -> Vec<u8> {
    let mut rasterizer = Rasterizer::new(width, height);
    rasterizer.add_path(path);
    rasterizer.rasterize_rgba(fill_rule, color)
}

/// Composites a mask onto an existing image buffer.
pub fn composite_mask(
    target: &mut [u8],
    mask: &[f32],
    width: usize,
    _height: usize,
    color: [u8; 4],
) {
    for (i, &coverage) in mask.iter().enumerate() {
        if coverage <= 0.0 {
            continue;
        }

        let idx = i * 4;
        if idx + 3 >= target.len() {
            break;
        }

        let alpha = (color[3] as f32 / 255.0) * coverage;

        // Alpha blend
        let src_r = color[0] as f32 / 255.0;
        let src_g = color[1] as f32 / 255.0;
        let src_b = color[2] as f32 / 255.0;

        let dst_r = target[idx] as f32 / 255.0;
        let dst_g = target[idx + 1] as f32 / 255.0;
        let dst_b = target[idx + 2] as f32 / 255.0;
        let dst_a = target[idx + 3] as f32 / 255.0;

        let out_a = alpha + dst_a * (1.0 - alpha);
        if out_a > 0.0 {
            let out_r = (src_r * alpha + dst_r * dst_a * (1.0 - alpha)) / out_a;
            let out_g = (src_g * alpha + dst_g * dst_a * (1.0 - alpha)) / out_a;
            let out_b = (src_b * alpha + dst_b * dst_a * (1.0 - alpha)) / out_a;

            target[idx] = (out_r * 255.0) as u8;
            target[idx + 1] = (out_g * 255.0) as u8;
            target[idx + 2] = (out_b * 255.0) as u8;
            target[idx + 3] = (out_a * 255.0) as u8;
        }
    }

    let _ = width; // Used for stride calculation if needed
}

/// Creates a stroke mask from a path (outline only).
pub fn rasterize_stroke(path: &Path, width: usize, height: usize, stroke_width: f32) -> Vec<f32> {
    // Offset the path outward and inward, then combine
    let outer = offset_path(path, stroke_width / 2.0, JoinStyle::Miter);
    let inner = offset_path(path, -stroke_width / 2.0, JoinStyle::Miter);

    // Rasterize outer - inner
    let mut rasterizer = Rasterizer::new(width, height);
    rasterizer.add_path(&outer);

    let outer_mask = rasterizer.rasterize(FillRule::NonZero);

    rasterizer.clear();
    rasterizer.add_path(&inner);
    let inner_mask = rasterizer.rasterize(FillRule::NonZero);

    // Subtract inner from outer
    outer_mask
        .iter()
        .zip(inner_mask.iter())
        .map(|(&o, &i)| (o - i).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::polygon;

    fn make_triangle_path() -> Path {
        polygon(&[
            Vec2::new(50.0, 10.0),
            Vec2::new(90.0, 90.0),
            Vec2::new(10.0, 90.0),
        ])
    }

    fn make_rect_path() -> Path {
        polygon(&[
            Vec2::new(20.0, 20.0),
            Vec2::new(80.0, 20.0),
            Vec2::new(80.0, 80.0),
            Vec2::new(20.0, 80.0),
        ])
    }

    #[test]
    fn test_rasterizer_creation() {
        let rasterizer = Rasterizer::new(100, 100);
        assert_eq!(rasterizer.width(), 100);
        assert_eq!(rasterizer.height(), 100);
    }

    #[test]
    fn test_rasterize_triangle() {
        let path = make_triangle_path();
        let mask = rasterize_path(&path, 100, 100, FillRule::NonZero);

        assert_eq!(mask.len(), 100 * 100);

        // Center of triangle should be filled
        let center_idx = 50 * 100 + 50;
        assert!(mask[center_idx] > 0.5);

        // Corner should be empty
        assert!(mask[0] < 0.1);
    }

    #[test]
    fn test_rasterize_rect() {
        let path = make_rect_path();
        let mask = rasterize_path(&path, 100, 100, FillRule::NonZero);

        // Inside rect should be filled
        let inside_idx = 50 * 100 + 50;
        assert!(mask[inside_idx] > 0.5);

        // Outside rect should be empty
        let outside_idx = 10 * 100 + 10;
        assert!(mask[outside_idx] < 0.1);
    }

    #[test]
    fn test_rasterize_polygon() {
        let points = vec![
            Vec2::new(20.0, 20.0),
            Vec2::new(80.0, 20.0),
            Vec2::new(80.0, 80.0),
            Vec2::new(20.0, 80.0),
        ];

        let mask = rasterize_polygon(&points, 100, 100, FillRule::NonZero);

        // Inside should be filled
        let inside_idx = 50 * 100 + 50;
        assert!(mask[inside_idx] > 0.5);
    }

    #[test]
    fn test_rasterize_bool() {
        let path = make_rect_path();
        let mut rasterizer = Rasterizer::new(100, 100);
        rasterizer.add_path(&path);

        let bool_mask = rasterizer.rasterize_bool(FillRule::NonZero);

        let inside_idx = 50 * 100 + 50;
        assert!(bool_mask[inside_idx]);

        let outside_idx = 10 * 100 + 10;
        assert!(!bool_mask[outside_idx]);
    }

    #[test]
    fn test_rasterize_rgba() {
        let path = make_rect_path();
        let mut rasterizer = Rasterizer::new(100, 100);
        rasterizer.add_path(&path);

        let color = [255, 0, 0, 255]; // Red
        let pixels = rasterizer.rasterize_rgba(FillRule::NonZero, color);

        assert_eq!(pixels.len(), 100 * 100 * 4);

        // Check a filled pixel
        let inside_idx = (50 * 100 + 50) * 4;
        assert_eq!(pixels[inside_idx], 255); // R
        assert_eq!(pixels[inside_idx + 1], 0); // G
        assert_eq!(pixels[inside_idx + 2], 0); // B
        assert!(pixels[inside_idx + 3] > 0); // A
    }

    #[test]
    fn test_clear() {
        let mut rasterizer = Rasterizer::new(100, 100);
        rasterizer.add_path(&make_rect_path());

        assert!(!rasterizer.edges.is_empty());

        rasterizer.clear();
        assert!(rasterizer.edges.is_empty());
    }

    #[test]
    fn test_render_path_rgba() {
        let path = make_triangle_path();
        let pixels = render_path_rgba(&path, 100, 100, FillRule::NonZero, [0, 255, 0, 255]);

        assert_eq!(pixels.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_rasterize_paths() {
        let path1 = make_rect_path();
        let path2 = make_triangle_path();

        let mask = rasterize_paths(&[path1, path2], 100, 100, FillRule::NonZero);

        // Both shapes should contribute to mask
        assert_eq!(mask.len(), 100 * 100);
    }

    #[test]
    fn test_composite_mask() {
        let mut target = vec![0u8; 100 * 100 * 4];

        let mask: Vec<f32> = (0..10000)
            .map(|i| if i > 5000 { 1.0 } else { 0.0 })
            .collect();

        composite_mask(&mut target, &mask, 100, 100, [255, 0, 0, 255]);

        // Check that some pixels were modified
        let idx = 60 * 100 * 4;
        assert!(target[idx] > 0 || target[idx + 1] > 0 || target[idx + 2] > 0);
    }

    #[test]
    fn test_stroke_rasterization() {
        let path = make_rect_path();
        let mask = rasterize_stroke(&path, 100, 100, 4.0);

        assert_eq!(mask.len(), 100 * 100);
    }
}
