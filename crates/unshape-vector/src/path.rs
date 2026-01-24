//! 2D path representation and building.

use glam::Vec2;
use std::f32::consts::TAU;

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

/// Creates a line segment.
pub fn line(from: Vec2, to: Vec2) -> Path {
    PathBuilder::new().move_to(from).line_to(to).build()
}

/// Creates a polyline (connected line segments).
pub fn polyline(points: &[Vec2]) -> Path {
    if points.is_empty() {
        return Path::new();
    }

    let mut builder = PathBuilder::new().move_to(points[0]);
    for &p in &points[1..] {
        builder = builder.line_to(p);
    }
    builder.build()
}

/// Creates a closed polygon.
pub fn polygon(points: &[Vec2]) -> Path {
    if points.is_empty() {
        return Path::new();
    }

    let mut builder = PathBuilder::new().move_to(points[0]);
    for &p in &points[1..] {
        builder = builder.line_to(p);
    }
    builder.close().build()
}

/// Creates a rectangle.
pub fn rect(min: Vec2, max: Vec2) -> Path {
    PathBuilder::new()
        .move_to(min)
        .line_to(Vec2::new(max.x, min.y))
        .line_to(max)
        .line_to(Vec2::new(min.x, max.y))
        .close()
        .build()
}

/// Creates a rectangle centered at a point.
pub fn rect_centered(center: Vec2, size: Vec2) -> Path {
    let half = size * 0.5;
    rect(center - half, center + half)
}

/// Creates a circle approximated with cubic beziers.
///
/// Uses 4 cubic bezier curves for a good approximation.
pub fn circle(center: Vec2, radius: f32) -> Path {
    // Magic number for circular arc approximation with cubics
    // k = 4/3 * tan(π/8) ≈ 0.5522847498
    const K: f32 = 0.552_284_8;

    let r = radius;
    let c = center;
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

/// Creates an ellipse.
pub fn ellipse(center: Vec2, radii: Vec2) -> Path {
    let mut path = circle(Vec2::ZERO, 1.0);
    path.scale_xy(radii.x, radii.y);
    path.translate(center);
    path
}

/// Creates a regular polygon with n sides.
pub fn regular_polygon(center: Vec2, radius: f32, sides: u32) -> Path {
    if sides < 3 {
        return Path::new();
    }

    let mut points = Vec::with_capacity(sides as usize);
    for i in 0..sides {
        let angle = TAU * (i as f32) / (sides as f32) - TAU / 4.0; // Start at top
        points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
    }
    polygon(&points)
}

/// Creates a rounded rectangle.
pub fn rounded_rect(min: Vec2, max: Vec2, radius: f32) -> Path {
    let r = radius.min((max.x - min.x) / 2.0).min((max.y - min.y) / 2.0);

    if r <= 0.0 {
        return rect(min, max);
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

/// Creates a star shape.
pub fn star(center: Vec2, outer_radius: f32, inner_radius: f32, points: u32) -> Path {
    if points < 2 {
        return Path::new();
    }

    let mut vertices = Vec::with_capacity((points * 2) as usize);
    for i in 0..(points * 2) {
        let angle = TAU * (i as f32) / (points as f32 * 2.0) - TAU / 4.0;
        let r = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        vertices.push(center + Vec2::new(angle.cos(), angle.sin()) * r);
    }
    polygon(&vertices)
}

/// Creates a squircle (superellipse) shape.
///
/// A squircle is defined by the equation: |x/a|^n + |y/b|^n = 1
///
/// - `n = 2.0` produces an ellipse
/// - `n = 4.0` is the classic "squircle" (square-circle hybrid)
/// - Higher values approach a rectangle with rounded corners
/// - Values between 0 and 2 produce star-like shapes
///
/// # Arguments
/// * `center` - Center point of the squircle
/// * `size` - Half-width and half-height (radii in x and y)
/// * `n` - Exponent controlling the shape (typically 2.0 to 10.0)
///
/// # Example
/// ```ignore
/// // Classic squircle
/// let sq = squircle(Vec2::ZERO, Vec2::new(100.0, 100.0), 4.0);
///
/// // More rectangular
/// let sq = squircle(Vec2::ZERO, Vec2::new(100.0, 100.0), 8.0);
/// ```
pub fn squircle(center: Vec2, size: Vec2, n: f32) -> Path {
    squircle_with_segments(center, size, n, 64)
}

/// Creates a squircle with a specified number of segments.
///
/// More segments produce smoother curves but larger paths.
pub fn squircle_with_segments(center: Vec2, size: Vec2, n: f32, segments: u32) -> Path {
    if segments < 4 || n <= 0.0 {
        return Path::new();
    }

    let mut points = Vec::with_capacity(segments as usize);
    let inv_n = 1.0 / n;

    for i in 0..segments {
        let angle = TAU * (i as f32) / (segments as f32);

        // Superellipse parametric form:
        // x = a * sign(cos(t)) * |cos(t)|^(2/n)
        // y = b * sign(sin(t)) * |sin(t)|^(2/n)
        let cos_t = angle.cos();
        let sin_t = angle.sin();

        let x = size.x * cos_t.signum() * cos_t.abs().powf(inv_n);
        let y = size.y * sin_t.signum() * sin_t.abs().powf(inv_n);

        points.push(center + Vec2::new(x, y));
    }

    polygon(&points)
}

/// Creates a squircle with uniform size (same width and height).
pub fn squircle_uniform(center: Vec2, radius: f32, n: f32) -> Path {
    squircle(center, Vec2::splat(radius), n)
}

/// Corner radii for rounded rectangles.
///
/// Specifies the radius for each corner individually.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Creates a rounded rectangle with different radii for each corner.
///
/// The radii are automatically clamped to fit within the rectangle dimensions.
///
/// # Arguments
/// * `min` - Minimum corner (top-left in screen coordinates)
/// * `max` - Maximum corner (bottom-right in screen coordinates)
/// * `radii` - Corner radii (can be `CornerRadii`, `f32`, or `[f32; 4]`)
///
/// # Example
/// ```ignore
/// // Different radius per corner
/// let path = rounded_rect_corners(
///     Vec2::ZERO,
///     Vec2::new(200.0, 100.0),
///     CornerRadii::from_array([10.0, 20.0, 30.0, 0.0]),
/// );
///
/// // Only top corners rounded
/// let path = rounded_rect_corners(
///     Vec2::ZERO,
///     Vec2::new(200.0, 100.0),
///     CornerRadii::top(15.0),
/// );
/// ```
pub fn rounded_rect_corners(min: Vec2, max: Vec2, radii: impl Into<CornerRadii>) -> Path {
    let radii = radii.into();
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
        return rect(min, max);
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
    let radius = width.min(height) / 2.0;
    let half_w = width / 2.0;
    let half_h = height / 2.0;
    let min = center - Vec2::new(half_w, half_h);
    let max = center + Vec2::new(half_w, half_h);
    rounded_rect_corners(min, max, CornerRadii::uniform(radius))
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
