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
    QuadTo { control: Vec2, to: Vec2 },
    /// Cubic bezier curve to a point with two control points.
    CubicTo { control1: Vec2, control2: Vec2, to: Vec2 },
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
                PathCommand::CubicTo { control1, control2, to } => {
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
        self.path.commands.push(PathCommand::CubicTo { control1, control2, to });
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
    PathBuilder::new()
        .move_to(from)
        .line_to(to)
        .build()
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
    const K: f32 = 0.5522847498;

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

    const K: f32 = 0.5522847498;
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
        let r = if i % 2 == 0 { outer_radius } else { inner_radius };
        vertices.push(center + Vec2::new(angle.cos(), angle.sin()) * r);
    }
    polygon(&vertices)
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
}
