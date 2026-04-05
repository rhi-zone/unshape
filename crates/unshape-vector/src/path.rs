//! 2D path representation and building.

use glam::Vec2;
use std::f32::consts::TAU;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A path command in an SVG-like path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCommand {
    /// Move to a point without drawing.
    MoveTo(Vec2),
    /// Draw a line to a point.
    LineTo(Vec2),
    /// Quadratic bezier curve to a point with one control point.
    QuadTo {
        /// Control point.
        control: Vec2,
        /// End point.
        to: Vec2,
    },
    /// Cubic bezier curve to a point with two control points.
    CubicTo {
        /// First control point.
        control1: Vec2,
        /// Second control point.
        control2: Vec2,
        /// End point.
        to: Vec2,
    },
    /// Close the current subpath by drawing a line to the start.
    Close,
}

/// A 2D path consisting of path commands.
#[derive(Debug, Clone, Default)]
pub struct Path {
    commands: Vec<PathCommand>,
}

impl Path {
    /// Creates an empty path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the path commands.
    pub fn commands(&self) -> &[PathCommand] {
        &self.commands
    }

    /// Returns true if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Returns the number of commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Appends commands from another path.
    pub fn extend(&mut self, other: &Path) {
        self.commands.extend_from_slice(&other.commands);
    }

    /// Transforms all points in the path.
    pub fn transform(&mut self, f: impl Fn(Vec2) -> Vec2) {
        for cmd in &mut self.commands {
            match cmd {
                PathCommand::MoveTo(p) => *p = f(*p),
                PathCommand::LineTo(p) => *p = f(*p),
                PathCommand::QuadTo { control, to } => {
                    *control = f(*control);
                    *to = f(*to);
                }
                PathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    *control1 = f(*control1);
                    *control2 = f(*control2);
                    *to = f(*to);
                }
                PathCommand::Close => {}
            }
        }
    }

    /// Translates the path by an offset.
    pub fn translate(&mut self, offset: Vec2) {
        self.transform(|p| p + offset);
    }

    /// Scales the path by a factor.
    pub fn scale(&mut self, factor: f32) {
        self.transform(|p| p * factor);
    }

    /// Scales the path non-uniformly.
    pub fn scale_xy(&mut self, sx: f32, sy: f32) {
        self.transform(|p| Vec2::new(p.x * sx, p.y * sy));
    }

    /// Rotates the path around the origin.
    pub fn rotate(&mut self, angle: f32) {
        let cos = angle.cos();
        let sin = angle.sin();
        self.transform(|p| Vec2::new(p.x * cos - p.y * sin, p.x * sin + p.y * cos));
    }
}

/// Builder for constructing paths.
#[derive(Debug, Clone, Default)]
pub struct PathBuilder {
    path: Path,
    current: Vec2,
    start: Vec2,
}

impl PathBuilder {
    /// Creates a new path builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves to a point without drawing.
    pub fn move_to(mut self, to: Vec2) -> Self {
        self.path.commands.push(PathCommand::MoveTo(to));
        self.current = to;
        self.start = to;
        self
    }

    /// Draws a line to a point.
    pub fn line_to(mut self, to: Vec2) -> Self {
        self.path.commands.push(PathCommand::LineTo(to));
        self.current = to;
        self
    }

    /// Draws a quadratic bezier curve.
    pub fn quad_to(mut self, control: Vec2, to: Vec2) -> Self {
        self.path.commands.push(PathCommand::QuadTo { control, to });
        self.current = to;
        self
    }

    /// Draws a cubic bezier curve.
    pub fn cubic_to(mut self, control1: Vec2, control2: Vec2, to: Vec2) -> Self {
        self.path.commands.push(PathCommand::CubicTo {
            control1,
            control2,
            to,
        });
        self.current = to;
        self
    }

    /// Closes the current subpath.
    pub fn close(mut self) -> Self {
        self.path.commands.push(PathCommand::Close);
        self.current = self.start;
        self
    }

    /// Draws a horizontal line.
    pub fn h_line_to(self, x: f32) -> Self {
        let y = self.current.y;
        self.line_to(Vec2::new(x, y))
    }

    /// Draws a vertical line.
    pub fn v_line_to(self, y: f32) -> Self {
        let x = self.current.x;
        self.line_to(Vec2::new(x, y))
    }

    /// Draws a line relative to current position.
    pub fn line_by(self, delta: Vec2) -> Self {
        let to = self.current + delta;
        self.line_to(to)
    }

    /// Builds the final path.
    pub fn build(self) -> Path {
        self.path
    }
}

// Path primitives

// ============================================================================
// LinePath
// ============================================================================

/// Generates a line segment path.
///
/// Named `LinePath` to distinguish from `unshape_curve::Line`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct LinePath {
    /// Start point.
    pub from: Vec2,
    /// End point.
    pub to: Vec2,
}

impl LinePath {
    /// Creates a new line path.
    pub fn new(from: Vec2, to: Vec2) -> Self {
        Self { from, to }
    }

    /// Generates the line path.
    pub fn apply(&self) -> Path {
        PathBuilder::new()
            .move_to(self.from)
            .line_to(self.to)
            .build()
    }
}

/// Creates a line segment.
pub fn line(from: Vec2, to: Vec2) -> Path {
    LinePath { from, to }.apply()
}

// ============================================================================
// Polyline
// ============================================================================

/// Generates a polyline (connected line segments) path.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Polyline {
    /// Points to connect with line segments.
    pub points: Vec<Vec2>,
}

impl Polyline {
    /// Creates a new polyline.
    pub fn new(points: Vec<Vec2>) -> Self {
        Self { points }
    }

    /// Generates the polyline path.
    pub fn apply(&self) -> Path {
        let points = &self.points;
        if points.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new().move_to(points[0]);
        for &p in &points[1..] {
            builder = builder.line_to(p);
        }
        builder.build()
    }
}

/// Creates a polyline (connected line segments).
pub fn polyline(points: &[Vec2]) -> Path {
    Polyline {
        points: points.to_vec(),
    }
    .apply()
}

// ============================================================================
// Polygon
// ============================================================================

/// Generates a closed polygon path.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Polygon {
    /// Vertices of the polygon.
    pub points: Vec<Vec2>,
}

impl Polygon {
    /// Creates a new polygon.
    pub fn new(points: Vec<Vec2>) -> Self {
        Self { points }
    }

    /// Generates the polygon path.
    pub fn apply(&self) -> Path {
        let points = &self.points;
        if points.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new().move_to(points[0]);
        for &p in &points[1..] {
            builder = builder.line_to(p);
        }
        builder.close().build()
    }
}

/// Creates a closed polygon.
pub fn polygon(points: &[Vec2]) -> Path {
    Polygon {
        points: points.to_vec(),
    }
    .apply()
}

// ============================================================================
// Rect
// ============================================================================

/// Generates a rectangle path from min/max corners.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Rect {
    /// Minimum corner (top-left).
    pub min: Vec2,
    /// Maximum corner (bottom-right).
    pub max: Vec2,
}

impl Rect {
    /// Creates a new rectangle.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Generates the rectangle path.
    pub fn apply(&self) -> Path {
        let (min, max) = (self.min, self.max);
        PathBuilder::new()
            .move_to(min)
            .line_to(Vec2::new(max.x, min.y))
            .line_to(max)
            .line_to(Vec2::new(min.x, max.y))
            .close()
            .build()
    }
}

/// Creates a rectangle.
pub fn rect(min: Vec2, max: Vec2) -> Path {
    Rect { min, max }.apply()
}

// ============================================================================
// RectCentered
// ============================================================================

/// Generates a rectangle centered at a point.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct RectCentered {
    /// Center point.
    pub center: Vec2,
    /// Width and height.
    pub size: Vec2,
}

impl RectCentered {
    /// Creates a new centered rectangle.
    pub fn new(center: Vec2, size: Vec2) -> Self {
        Self { center, size }
    }

    /// Generates the rectangle path.
    pub fn apply(&self) -> Path {
        let half = self.size * 0.5;
        Rect {
            min: self.center - half,
            max: self.center + half,
        }
        .apply()
    }
}

/// Creates a rectangle centered at a point.
pub fn rect_centered(center: Vec2, size: Vec2) -> Path {
    RectCentered { center, size }.apply()
}

// ============================================================================
// Circle
// ============================================================================

/// Generates a circle path approximated with cubic beziers.
///
/// Uses 4 cubic bezier curves for a good approximation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Circle {
    /// Center point.
    pub center: Vec2,
    /// Radius.
    pub radius: f32,
}

impl Default for Circle {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            radius: 1.0,
        }
    }
}

impl Circle {
    /// Creates a new circle.
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Generates the circle path.
    pub fn apply(&self) -> Path {
        // Magic number for circular arc approximation with cubics
        // k = 4/3 * tan(π/8) ≈ 0.5522847498
        const K: f32 = 0.552_284_8;

        let r = self.radius;
        let c = self.center;
        let k = K * r;

        PathBuilder::new()
            .move_to(Vec2::new(c.x + r, c.y))
            .cubic_to(
                Vec2::new(c.x + r, c.y + k),
                Vec2::new(c.x + k, c.y + r),
                Vec2::new(c.x, c.y + r),
            )
            .cubic_to(
                Vec2::new(c.x - k, c.y + r),
                Vec2::new(c.x - r, c.y + k),
                Vec2::new(c.x - r, c.y),
            )
            .cubic_to(
                Vec2::new(c.x - r, c.y - k),
                Vec2::new(c.x - k, c.y - r),
                Vec2::new(c.x, c.y - r),
            )
            .cubic_to(
                Vec2::new(c.x + k, c.y - r),
                Vec2::new(c.x + r, c.y - k),
                Vec2::new(c.x + r, c.y),
            )
            .close()
            .build()
    }
}

/// Creates a circle approximated with cubic beziers.
pub fn circle(center: Vec2, radius: f32) -> Path {
    Circle { center, radius }.apply()
}

// ============================================================================
// Ellipse
// ============================================================================

/// Generates an ellipse path.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Ellipse {
    /// Center point.
    pub center: Vec2,
    /// Radii (half-width, half-height).
    pub radii: Vec2,
}

impl Ellipse {
    /// Creates a new ellipse.
    pub fn new(center: Vec2, radii: Vec2) -> Self {
        Self { center, radii }
    }

    /// Generates the ellipse path.
    pub fn apply(&self) -> Path {
        let mut path = Circle::new(Vec2::ZERO, 1.0).apply();
        path.scale_xy(self.radii.x, self.radii.y);
        path.translate(self.center);
        path
    }
}

/// Creates an ellipse.
pub fn ellipse(center: Vec2, radii: Vec2) -> Path {
    Ellipse { center, radii }.apply()
}

// ============================================================================
// RegularPolygon
// ============================================================================

/// Generates a regular polygon path with n equal sides.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct RegularPolygon {
    /// Center point.
    pub center: Vec2,
    /// Circumradius (distance from center to vertices).
    pub radius: f32,
    /// Number of sides. Minimum 3.
    pub sides: u32,
}

impl Default for RegularPolygon {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            radius: 1.0,
            sides: 6,
        }
    }
}

impl RegularPolygon {
    /// Creates a new regular polygon.
    pub fn new(center: Vec2, radius: f32, sides: u32) -> Self {
        Self {
            center,
            radius,
            sides,
        }
    }

    /// Generates the regular polygon path.
    pub fn apply(&self) -> Path {
        if self.sides < 3 {
            return Path::new();
        }

        let mut points = Vec::with_capacity(self.sides as usize);
        for i in 0..self.sides {
            let angle = TAU * (i as f32) / (self.sides as f32) - TAU / 4.0; // Start at top
            points.push(self.center + Vec2::new(angle.cos(), angle.sin()) * self.radius);
        }
        Polygon { points }.apply()
    }
}

/// Creates a regular polygon with n sides.
pub fn regular_polygon(center: Vec2, radius: f32, sides: u32) -> Path {
    RegularPolygon {
        center,
        radius,
        sides,
    }
    .apply()
}

// ============================================================================
// RoundedRect
// ============================================================================

/// Generates a rounded rectangle path with a uniform corner radius.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct RoundedRect {
    /// Minimum corner (top-left).
    pub min: Vec2,
    /// Maximum corner (bottom-right).
    pub max: Vec2,
    /// Corner radius.
    pub radius: f32,
}

impl RoundedRect {
    /// Creates a new rounded rectangle.
    pub fn new(min: Vec2, max: Vec2, radius: f32) -> Self {
        Self { min, max, radius }
    }

    /// Generates the rounded rectangle path.
    pub fn apply(&self) -> Path {
        let (min, max) = (self.min, self.max);
        let r = self
            .radius
            .min((max.x - min.x) / 2.0)
            .min((max.y - min.y) / 2.0);

        if r <= 0.0 {
            return Rect { min, max }.apply();
        }

        const K: f32 = 0.552_284_8;
        let k = K * r;

        PathBuilder::new()
            // Start at top-left, after corner
            .move_to(Vec2::new(min.x + r, min.y))
            // Top edge
            .line_to(Vec2::new(max.x - r, min.y))
            // Top-right corner
            .cubic_to(
                Vec2::new(max.x - r + k, min.y),
                Vec2::new(max.x, min.y + r - k),
                Vec2::new(max.x, min.y + r),
            )
            // Right edge
            .line_to(Vec2::new(max.x, max.y - r))
            // Bottom-right corner
            .cubic_to(
                Vec2::new(max.x, max.y - r + k),
                Vec2::new(max.x - r + k, max.y),
                Vec2::new(max.x - r, max.y),
            )
            // Bottom edge
            .line_to(Vec2::new(min.x + r, max.y))
            // Bottom-left corner
            .cubic_to(
                Vec2::new(min.x + r - k, max.y),
                Vec2::new(min.x, max.y - r + k),
                Vec2::new(min.x, max.y - r),
            )
            // Left edge
            .line_to(Vec2::new(min.x, min.y + r))
            // Top-left corner
            .cubic_to(
                Vec2::new(min.x, min.y + r - k),
                Vec2::new(min.x + r - k, min.y),
                Vec2::new(min.x + r, min.y),
            )
            .close()
            .build()
    }
}

/// Creates a rounded rectangle.
pub fn rounded_rect(min: Vec2, max: Vec2, radius: f32) -> Path {
    RoundedRect { min, max, radius }.apply()
}

// ============================================================================
// Star
// ============================================================================

/// Generates a star shape path.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Star {
    /// Center point.
    pub center: Vec2,
    /// Outer (tip) radius.
    pub outer_radius: f32,
    /// Inner (valley) radius.
    pub inner_radius: f32,
    /// Number of points. Minimum 2.
    pub points: u32,
}

impl Default for Star {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            outer_radius: 1.0,
            inner_radius: 0.5,
            points: 5,
        }
    }
}

impl Star {
    /// Creates a new star.
    pub fn new(center: Vec2, outer_radius: f32, inner_radius: f32, points: u32) -> Self {
        Self {
            center,
            outer_radius,
            inner_radius,
            points,
        }
    }

    /// Generates the star path.
    pub fn apply(&self) -> Path {
        if self.points < 2 {
            return Path::new();
        }

        let mut vertices = Vec::with_capacity((self.points * 2) as usize);
        for i in 0..(self.points * 2) {
            let angle = TAU * (i as f32) / (self.points as f32 * 2.0) - TAU / 4.0;
            let r = if i % 2 == 0 {
                self.outer_radius
            } else {
                self.inner_radius
            };
            vertices.push(self.center + Vec2::new(angle.cos(), angle.sin()) * r);
        }
        Polygon { points: vertices }.apply()
    }
}

/// Creates a star shape.
pub fn star(center: Vec2, outer_radius: f32, inner_radius: f32, points: u32) -> Path {
    Star {
        center,
        outer_radius,
        inner_radius,
        points,
    }
    .apply()
}

// ============================================================================
// Squircle
// ============================================================================

/// Generates a squircle (superellipse) shape path.
///
/// A squircle is defined by the equation: |x/a|^n + |y/b|^n = 1
///
/// - `n = 2.0` produces an ellipse
/// - `n = 4.0` is the classic "squircle" (square-circle hybrid)
/// - Higher values approach a rectangle with rounded corners
/// - Values between 0 and 2 produce star-like shapes
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Squircle {
    /// Center point.
    pub center: Vec2,
    /// Half-width and half-height (radii in x and y).
    pub size: Vec2,
    /// Exponent controlling the shape (typically 2.0 to 10.0).
    pub n: f32,
    /// Number of line segments used to approximate the curve.
    pub segments: u32,
}

impl Default for Squircle {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            size: Vec2::ONE,
            n: 4.0,
            segments: 64,
        }
    }
}

impl Squircle {
    /// Creates a new squircle.
    pub fn new(center: Vec2, size: Vec2, n: f32, segments: u32) -> Self {
        Self {
            center,
            size,
            n,
            segments,
        }
    }

    /// Creates a squircle with uniform size (same width and height).
    pub fn uniform(center: Vec2, radius: f32, n: f32) -> Self {
        Self {
            center,
            size: Vec2::splat(radius),
            n,
            segments: 64,
        }
    }

    /// Generates the squircle path.
    pub fn apply(&self) -> Path {
        if self.segments < 4 || self.n <= 0.0 {
            return Path::new();
        }

        let mut points = Vec::with_capacity(self.segments as usize);
        let inv_n = 1.0 / self.n;

        for i in 0..self.segments {
            let angle = TAU * (i as f32) / (self.segments as f32);

            // Superellipse parametric form:
            // x = a * sign(cos(t)) * |cos(t)|^(2/n)
            // y = b * sign(sin(t)) * |sin(t)|^(2/n)
            let cos_t = angle.cos();
            let sin_t = angle.sin();

            let x = self.size.x * cos_t.signum() * cos_t.abs().powf(inv_n);
            let y = self.size.y * sin_t.signum() * sin_t.abs().powf(inv_n);

            points.push(self.center + Vec2::new(x, y));
        }

        Polygon { points }.apply()
    }
}

/// Creates a squircle (superellipse) shape.
///
/// Uses 64 segments. For custom segment count, use [`squircle_with_segments`].
pub fn squircle(center: Vec2, size: Vec2, n: f32) -> Path {
    Squircle {
        center,
        size,
        n,
        segments: 64,
    }
    .apply()
}

/// Creates a squircle with a specified number of segments.
///
/// More segments produce smoother curves but larger paths.
pub fn squircle_with_segments(center: Vec2, size: Vec2, n: f32, segments: u32) -> Path {
    Squircle {
        center,
        size,
        n,
        segments,
    }
    .apply()
}

/// Creates a squircle with uniform size (same width and height).
pub fn squircle_uniform(center: Vec2, radius: f32, n: f32) -> Path {
    Squircle::uniform(center, radius, n).apply()
}

/// Corner radii for rounded rectangles.
///
/// Specifies the radius for each corner individually.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CornerRadii {
    /// Top-left corner radius.
    pub top_left: f32,
    /// Top-right corner radius.
    pub top_right: f32,
    /// Bottom-right corner radius.
    pub bottom_right: f32,
    /// Bottom-left corner radius.
    pub bottom_left: f32,
}

impl CornerRadii {
    /// Creates corner radii with the same value for all corners.
    pub fn uniform(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    /// Creates corner radii with all zeros (sharp corners).
    pub fn zero() -> Self {
        Self::uniform(0.0)
    }

    /// Creates corner radii from an array [top_left, top_right, bottom_right, bottom_left].
    pub fn from_array(radii: [f32; 4]) -> Self {
        Self {
            top_left: radii[0],
            top_right: radii[1],
            bottom_right: radii[2],
            bottom_left: radii[3],
        }
    }

    /// Creates corner radii with top corners rounded, bottom sharp.
    pub fn top(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: 0.0,
            bottom_left: 0.0,
        }
    }

    /// Creates corner radii with bottom corners rounded, top sharp.
    pub fn bottom(radius: f32) -> Self {
        Self {
            top_left: 0.0,
            top_right: 0.0,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    /// Creates corner radii with left corners rounded, right sharp.
    pub fn left(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: 0.0,
            bottom_right: 0.0,
            bottom_left: radius,
        }
    }

    /// Creates corner radii with right corners rounded, left sharp.
    pub fn right(radius: f32) -> Self {
        Self {
            top_left: 0.0,
            top_right: radius,
            bottom_right: radius,
            bottom_left: 0.0,
        }
    }
}

impl Default for CornerRadii {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<f32> for CornerRadii {
    fn from(radius: f32) -> Self {
        Self::uniform(radius)
    }
}

impl From<[f32; 4]> for CornerRadii {
    fn from(radii: [f32; 4]) -> Self {
        Self::from_array(radii)
    }
}

// ============================================================================
// RoundedRectCorners
// ============================================================================

/// Generates a rounded rectangle path with per-corner radii.
///
/// The radii are automatically clamped to fit within the rectangle dimensions.
///
/// # Example
/// ```ignore
/// // Different radius per corner
/// let path = RoundedRectCorners {
///     min: Vec2::ZERO,
///     max: Vec2::new(200.0, 100.0),
///     radii: CornerRadii::from_array([10.0, 20.0, 30.0, 0.0]),
/// }.apply();
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct RoundedRectCorners {
    /// Minimum corner (top-left in screen coordinates).
    pub min: Vec2,
    /// Maximum corner (bottom-right in screen coordinates).
    pub max: Vec2,
    /// Per-corner radii.
    pub radii: CornerRadii,
}

impl RoundedRectCorners {
    /// Creates a new rounded rectangle with per-corner radii.
    pub fn new(min: Vec2, max: Vec2, radii: CornerRadii) -> Self {
        Self { min, max, radii }
    }

    /// Generates the rounded rectangle path.
    pub fn apply(&self) -> Path {
        let (min, max) = (self.min, self.max);
        let radii = self.radii;
        let width = max.x - min.x;
        let height = max.y - min.y;

        // Clamp radii to fit within the rectangle
        let max_radius_h = width / 2.0;
        let max_radius_v = height / 2.0;

        let tl = radii.top_left.min(max_radius_h).min(max_radius_v).max(0.0);
        let tr = radii.top_right.min(max_radius_h).min(max_radius_v).max(0.0);
        let br = radii
            .bottom_right
            .min(max_radius_h)
            .min(max_radius_v)
            .max(0.0);
        let bl = radii
            .bottom_left
            .min(max_radius_h)
            .min(max_radius_v)
            .max(0.0);

        // If all radii are zero, return a simple rect
        if tl == 0.0 && tr == 0.0 && br == 0.0 && bl == 0.0 {
            return Rect { min, max }.apply();
        }

        // Magic number for circular arc approximation with cubics
        const K: f32 = 0.552_284_8;

        let mut builder = PathBuilder::new();

        // Start at top-left, after the corner
        builder = builder.move_to(Vec2::new(min.x + tl, min.y));

        // Top edge
        builder = builder.line_to(Vec2::new(max.x - tr, min.y));

        // Top-right corner
        if tr > 0.0 {
            let k = K * tr;
            builder = builder.cubic_to(
                Vec2::new(max.x - tr + k, min.y),
                Vec2::new(max.x, min.y + tr - k),
                Vec2::new(max.x, min.y + tr),
            );
        }

        // Right edge
        builder = builder.line_to(Vec2::new(max.x, max.y - br));

        // Bottom-right corner
        if br > 0.0 {
            let k = K * br;
            builder = builder.cubic_to(
                Vec2::new(max.x, max.y - br + k),
                Vec2::new(max.x - br + k, max.y),
                Vec2::new(max.x - br, max.y),
            );
        }

        // Bottom edge
        builder = builder.line_to(Vec2::new(min.x + bl, max.y));

        // Bottom-left corner
        if bl > 0.0 {
            let k = K * bl;
            builder = builder.cubic_to(
                Vec2::new(min.x + bl - k, max.y),
                Vec2::new(min.x, max.y - bl + k),
                Vec2::new(min.x, max.y - bl),
            );
        }

        // Left edge
        builder = builder.line_to(Vec2::new(min.x, min.y + tl));

        // Top-left corner
        if tl > 0.0 {
            let k = K * tl;
            builder = builder.cubic_to(
                Vec2::new(min.x, min.y + tl - k),
                Vec2::new(min.x + tl - k, min.y),
                Vec2::new(min.x + tl, min.y),
            );
        }

        builder.close().build()
    }
}

/// Creates a rounded rectangle with different radii for each corner.
///
/// The radii are automatically clamped to fit within the rectangle dimensions.
///
/// # Arguments
/// * `min` - Minimum corner (top-left in screen coordinates)
/// * `max` - Maximum corner (bottom-right in screen coordinates)
/// * `radii` - Corner radii (can be `CornerRadii`, `f32`, or `[f32; 4]`)
pub fn rounded_rect_corners(min: Vec2, max: Vec2, radii: impl Into<CornerRadii>) -> Path {
    RoundedRectCorners {
        min,
        max,
        radii: radii.into(),
    }
    .apply()
}

// ============================================================================
// Pill
// ============================================================================

/// Generates a pill shape (stadium/discorectangle) path.
///
/// A pill is a rectangle with fully rounded ends (semicircles).
/// If `width > height`, the pill is horizontal. If `height > width`, it's vertical.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Path))]
pub struct Pill {
    /// Center point.
    pub center: Vec2,
    /// Total width (including rounded ends).
    pub width: f32,
    /// Total height.
    pub height: f32,
}

impl Pill {
    /// Creates a new pill.
    pub fn new(center: Vec2, width: f32, height: f32) -> Self {
        Self {
            center,
            width,
            height,
        }
    }

    /// Generates the pill path.
    pub fn apply(&self) -> Path {
        let radius = self.width.min(self.height) / 2.0;
        let half_w = self.width / 2.0;
        let half_h = self.height / 2.0;
        let min = self.center - Vec2::new(half_w, half_h);
        let max = self.center + Vec2::new(half_w, half_h);
        RoundedRectCorners {
            min,
            max,
            radii: CornerRadii::uniform(radius),
        }
        .apply()
    }
}

/// Creates a pill shape (stadium/discorectangle).
///
/// A pill is a rectangle with fully rounded ends (semicircles).
///
/// # Arguments
/// * `center` - Center point
/// * `width` - Total width (including rounded ends)
/// * `height` - Total height
///
/// If width > height, the pill is horizontal. If height > width, it's vertical.
pub fn pill(center: Vec2, width: f32, height: f32) -> Path {
    Pill {
        center,
        width,
        height,
    }
    .apply()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_builder() {
        let path = PathBuilder::new()
            .move_to(Vec2::ZERO)
            .line_to(Vec2::new(1.0, 0.0))
            .line_to(Vec2::new(1.0, 1.0))
            .close()
            .build();

        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_rect() {
        let path = rect(Vec2::ZERO, Vec2::new(2.0, 1.0));
        assert_eq!(path.len(), 5); // move, 3 lines, close
    }

    #[test]
    fn test_circle() {
        let path = circle(Vec2::ZERO, 1.0);
        assert_eq!(path.len(), 6); // move, 4 cubics, close
    }

    #[test]
    fn test_polygon() {
        let triangle = polygon(&[
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
        ]);
        assert_eq!(triangle.len(), 4); // move, 2 lines, close
    }

    #[test]
    fn test_regular_polygon() {
        let hex = regular_polygon(Vec2::ZERO, 1.0, 6);
        assert_eq!(hex.len(), 7); // move, 5 lines, close
    }

    #[test]
    fn test_star() {
        let s = star(Vec2::ZERO, 1.0, 0.5, 5);
        assert_eq!(s.len(), 11); // move, 9 lines, close
    }

    #[test]
    fn test_transform() {
        let mut path = line(Vec2::ZERO, Vec2::new(1.0, 0.0));
        path.translate(Vec2::new(10.0, 0.0));

        if let PathCommand::LineTo(p) = path.commands()[1] {
            assert!((p.x - 11.0).abs() < 0.001);
        } else {
            panic!("expected LineTo");
        }
    }

    #[test]
    fn test_squircle_basic() {
        let path = squircle(Vec2::ZERO, Vec2::new(100.0, 100.0), 4.0);
        assert!(!path.is_empty());
        // 64 segments: 1 MoveTo + 63 LineTo + 1 Close = 65 commands
        assert_eq!(path.len(), 65);
    }

    #[test]
    fn test_squircle_with_segments() {
        let path = squircle_with_segments(Vec2::ZERO, Vec2::new(50.0, 50.0), 4.0, 32);
        assert!(!path.is_empty());
        // 32 segments: 1 MoveTo + 31 LineTo + 1 Close = 33 commands
        assert_eq!(path.len(), 33);
    }

    #[test]
    fn test_squircle_uniform() {
        let path = squircle_uniform(Vec2::ZERO, 50.0, 4.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_squircle_n2_is_ellipse_like() {
        // n=2 should produce an ellipse-like shape
        let path = squircle_with_segments(Vec2::ZERO, Vec2::new(100.0, 100.0), 2.0, 64);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_squircle_high_n_is_rect_like() {
        // High n should produce a more rectangular shape
        let path = squircle_with_segments(Vec2::ZERO, Vec2::new(100.0, 100.0), 20.0, 64);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_squircle_invalid() {
        // Invalid parameters should return empty path
        assert!(squircle_with_segments(Vec2::ZERO, Vec2::ONE, 4.0, 2).is_empty());
        assert!(squircle_with_segments(Vec2::ZERO, Vec2::ONE, 0.0, 64).is_empty());
    }

    #[test]
    fn test_corner_radii_uniform() {
        let r = CornerRadii::uniform(10.0);
        assert_eq!(r.top_left, 10.0);
        assert_eq!(r.top_right, 10.0);
        assert_eq!(r.bottom_right, 10.0);
        assert_eq!(r.bottom_left, 10.0);
    }

    #[test]
    fn test_corner_radii_from_array() {
        let r = CornerRadii::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.top_left, 1.0);
        assert_eq!(r.top_right, 2.0);
        assert_eq!(r.bottom_right, 3.0);
        assert_eq!(r.bottom_left, 4.0);
    }

    #[test]
    fn test_corner_radii_presets() {
        let top = CornerRadii::top(10.0);
        assert_eq!(top.top_left, 10.0);
        assert_eq!(top.top_right, 10.0);
        assert_eq!(top.bottom_right, 0.0);
        assert_eq!(top.bottom_left, 0.0);

        let bottom = CornerRadii::bottom(10.0);
        assert_eq!(bottom.top_left, 0.0);
        assert_eq!(bottom.bottom_right, 10.0);
    }

    #[test]
    fn test_corner_radii_struct_init() {
        let r = CornerRadii {
            top_left: 5.0,
            bottom_right: 10.0,
            ..CornerRadii::zero()
        };
        assert_eq!(r.top_left, 5.0);
        assert_eq!(r.top_right, 0.0);
        assert_eq!(r.bottom_right, 10.0);
        assert_eq!(r.bottom_left, 0.0);
    }

    #[test]
    fn test_rounded_rect_corners_uniform() {
        let path = rounded_rect_corners(Vec2::ZERO, Vec2::new(100.0, 50.0), 10.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_rounded_rect_corners_array() {
        let path = rounded_rect_corners(Vec2::ZERO, Vec2::new(100.0, 50.0), [10.0, 20.0, 5.0, 0.0]);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_rounded_rect_corners_struct() {
        let path = rounded_rect_corners(Vec2::ZERO, Vec2::new(100.0, 50.0), CornerRadii::top(15.0));
        assert!(!path.is_empty());
    }

    #[test]
    fn test_rounded_rect_corners_zero_is_rect() {
        let rounded = rounded_rect_corners(Vec2::ZERO, Vec2::new(100.0, 50.0), 0.0);
        let plain = rect(Vec2::ZERO, Vec2::new(100.0, 50.0));
        assert_eq!(rounded.len(), plain.len());
    }

    #[test]
    fn test_rounded_rect_corners_clamping() {
        // Radii too large should be clamped
        let path = rounded_rect_corners(
            Vec2::ZERO,
            Vec2::new(100.0, 50.0),
            CornerRadii::uniform(100.0), // Way too big
        );
        assert!(!path.is_empty());
    }

    #[test]
    fn test_pill_horizontal() {
        let path = pill(Vec2::ZERO, 100.0, 50.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_pill_vertical() {
        let path = pill(Vec2::ZERO, 50.0, 100.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_pill_square() {
        // Square dimensions should produce a circle
        let path = pill(Vec2::ZERO, 100.0, 100.0);
        assert!(!path.is_empty());
    }
}
