//! Gradient meshes for smooth color fills.
//!
//! A gradient mesh is a 2D mesh where each vertex has a position and color,
//! allowing smooth color gradients across arbitrary shapes.

use glam::Vec2;
use unshape_color::Rgba;

/// A vertex in a gradient mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientVertex {
    /// Position in 2D space.
    pub position: Vec2,
    /// Color at this vertex.
    pub color: Rgba,
}

impl GradientVertex {
    /// Creates a new gradient vertex.
    pub fn new(position: Vec2, color: Rgba) -> Self {
        Self { position, color }
    }

    /// Creates from coordinates and RGBA values.
    pub fn from_coords(x: f32, y: f32, r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            position: Vec2::new(x, y),
            color: Rgba::new(r, g, b, a),
        }
    }
}

/// A triangular face in a gradient mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientFace {
    /// Vertex indices forming the triangle.
    pub indices: [usize; 3],
}

impl GradientFace {
    /// Creates a new face from vertex indices.
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { indices: [a, b, c] }
    }
}

/// A patch in a gradient mesh (typically a quad subdivided into triangles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientPatch {
    /// Four corner vertex indices (counterclockwise: bottom-left, bottom-right, top-right, top-left).
    pub corners: [usize; 4],
}

impl GradientPatch {
    /// Creates a new quad patch.
    pub fn new(bl: usize, br: usize, tr: usize, tl: usize) -> Self {
        Self {
            corners: [bl, br, tr, tl],
        }
    }

    /// Converts to two triangular faces.
    pub fn to_faces(&self) -> [GradientFace; 2] {
        [
            GradientFace::new(self.corners[0], self.corners[1], self.corners[2]),
            GradientFace::new(self.corners[0], self.corners[2], self.corners[3]),
        ]
    }
}

/// A gradient mesh with colored vertices and triangular faces.
#[derive(Debug, Clone)]
pub struct GradientMesh {
    /// All vertices in the mesh.
    pub vertices: Vec<GradientVertex>,
    /// Triangular faces (indices into vertices).
    pub faces: Vec<GradientFace>,
}

impl GradientMesh {
    /// Creates an empty gradient mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Creates a gradient mesh from vertices and faces.
    pub fn from_parts(vertices: Vec<GradientVertex>, faces: Vec<GradientFace>) -> Self {
        Self { vertices, faces }
    }

    /// Adds a vertex and returns its index.
    pub fn add_vertex(&mut self, vertex: GradientVertex) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(vertex);
        idx
    }

    /// Adds a triangular face.
    pub fn add_face(&mut self, a: usize, b: usize, c: usize) {
        self.faces.push(GradientFace::new(a, b, c));
    }

    /// Adds a quad as two triangles.
    pub fn add_quad(&mut self, bl: usize, br: usize, tr: usize, tl: usize) {
        let patch = GradientPatch::new(bl, br, tr, tl);
        for face in patch.to_faces() {
            self.faces.push(face);
        }
    }

    /// Creates a rectangular grid gradient mesh.
    ///
    /// The grid has `cols` x `rows` cells (so `cols+1` x `rows+1` vertices).
    /// Colors are computed by bilinear interpolation of the four corner colors.
    pub fn grid(
        origin: Vec2,
        size: Vec2,
        cols: usize,
        rows: usize,
        color_bl: Rgba,
        color_br: Rgba,
        color_tr: Rgba,
        color_tl: Rgba,
    ) -> Self {
        let mut mesh = Self::new();

        // Create vertices
        for row in 0..=rows {
            for col in 0..=cols {
                let u = col as f32 / cols as f32;
                let v = row as f32 / rows as f32;

                let position = origin + Vec2::new(u * size.x, v * size.y);

                // Bilinear interpolation of colors
                let color = bilinear_color(color_bl, color_br, color_tr, color_tl, u, v);

                mesh.add_vertex(GradientVertex::new(position, color));
            }
        }

        // Create faces (two triangles per cell)
        for row in 0..rows {
            for col in 0..cols {
                let bl = row * (cols + 1) + col;
                let br = bl + 1;
                let tl = bl + (cols + 1);
                let tr = tl + 1;

                mesh.add_quad(bl, br, tr, tl);
            }
        }

        mesh
    }

    /// Creates a radial gradient mesh (circular).
    ///
    /// Creates a mesh with `rings` concentric rings and `segments` angular divisions.
    pub fn radial(
        center: Vec2,
        radius: f32,
        rings: usize,
        segments: usize,
        center_color: Rgba,
        edge_color: Rgba,
    ) -> Self {
        let mut mesh = Self::new();

        // Center vertex
        let center_idx = mesh.add_vertex(GradientVertex::new(center, center_color));

        // Create ring vertices
        for ring in 1..=rings {
            let r = radius * ring as f32 / rings as f32;
            let t = ring as f32 / rings as f32;
            let color = center_color.lerp(edge_color, t);

            for seg in 0..segments {
                let angle = std::f32::consts::TAU * seg as f32 / segments as f32;
                let position = center + Vec2::new(angle.cos(), angle.sin()) * r;
                mesh.add_vertex(GradientVertex::new(position, color));
            }
        }

        // Create inner triangles (center to first ring)
        for seg in 0..segments {
            let next_seg = (seg + 1) % segments;
            mesh.add_face(center_idx, 1 + seg, 1 + next_seg);
        }

        // Create ring quads
        for ring in 1..rings {
            let ring_start = 1 + (ring - 1) * segments;
            let next_ring_start = 1 + ring * segments;

            for seg in 0..segments {
                let next_seg = (seg + 1) % segments;

                let bl = ring_start + seg;
                let br = ring_start + next_seg;
                let tl = next_ring_start + seg;
                let tr = next_ring_start + next_seg;

                mesh.add_quad(bl, br, tr, tl);
            }
        }

        mesh
    }

    /// Samples color at a point using barycentric interpolation.
    ///
    /// Returns None if the point is outside all faces.
    pub fn sample(&self, point: Vec2) -> Option<Rgba> {
        for face in &self.faces {
            let v0 = self.vertices[face.indices[0]];
            let v1 = self.vertices[face.indices[1]];
            let v2 = self.vertices[face.indices[2]];

            if let Some(bary) = barycentric(point, v0.position, v1.position, v2.position) {
                // Point is inside this triangle
                let color = Rgba::new(
                    v0.color.r * bary.0 + v1.color.r * bary.1 + v2.color.r * bary.2,
                    v0.color.g * bary.0 + v1.color.g * bary.1 + v2.color.g * bary.2,
                    v0.color.b * bary.0 + v1.color.b * bary.1 + v2.color.b * bary.2,
                    v0.color.a * bary.0 + v1.color.a * bary.1 + v2.color.a * bary.2,
                );
                return Some(color);
            }
        }
        None
    }

    /// Samples color at a point, returning a default if outside mesh.
    pub fn sample_or(&self, point: Vec2, default: Rgba) -> Rgba {
        self.sample(point).unwrap_or(default)
    }

    /// Gets the bounding box of the mesh.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        if self.vertices.is_empty() {
            return (Vec2::ZERO, Vec2::ZERO);
        }

        let mut min = self.vertices[0].position;
        let mut max = min;

        for v in &self.vertices[1..] {
            min = min.min(v.position);
            max = max.max(v.position);
        }

        (min, max)
    }

    /// Transforms all vertex positions.
    pub fn transform(&mut self, f: impl Fn(Vec2) -> Vec2) {
        for v in &mut self.vertices {
            v.position = f(v.position);
        }
    }

    /// Translates the mesh.
    pub fn translate(&mut self, offset: Vec2) {
        self.transform(|p| p + offset);
    }

    /// Scales the mesh around the origin.
    pub fn scale(&mut self, factor: f32) {
        self.transform(|p| p * factor);
    }

    /// Scales the mesh around a center point.
    pub fn scale_around(&mut self, center: Vec2, factor: f32) {
        self.transform(|p| center + (p - center) * factor);
    }

    /// Modifies vertex colors.
    pub fn map_colors(&mut self, f: impl Fn(Rgba) -> Rgba) {
        for v in &mut self.vertices {
            v.color = f(v.color);
        }
    }

    /// Sets all vertex alpha values.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.map_colors(|c| Rgba::new(c.r, c.g, c.b, alpha));
    }
}

impl Default for GradientMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Bilinear interpolation of four corner colors.
fn bilinear_color(bl: Rgba, br: Rgba, tr: Rgba, tl: Rgba, u: f32, v: f32) -> Rgba {
    let bottom = bl.lerp(br, u);
    let top = tl.lerp(tr, u);
    bottom.lerp(top, v)
}

/// Computes barycentric coordinates of a point relative to a triangle.
///
/// Returns Some((w0, w1, w2)) if point is inside, None otherwise.
fn barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32, f32)> {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 1e-10 {
        return None;
    }

    let inv_denom = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Check if point is inside triangle
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        Some((1.0 - u - v, v, u))
    } else {
        None
    }
}

/// Creates a simple linear gradient mesh (horizontal or vertical).
pub fn linear_gradient_mesh(
    origin: Vec2,
    size: Vec2,
    start_color: Rgba,
    end_color: Rgba,
    horizontal: bool,
) -> GradientMesh {
    let (bl, br, tr, tl) = if horizontal {
        (start_color, end_color, end_color, start_color)
    } else {
        (start_color, start_color, end_color, end_color)
    };
    GradientMesh::grid(origin, size, 1, 1, bl, br, tr, tl)
}

/// Creates a diagonal gradient mesh.
pub fn diagonal_gradient_mesh(
    origin: Vec2,
    size: Vec2,
    start_color: Rgba,
    end_color: Rgba,
    subdivisions: usize,
) -> GradientMesh {
    // Diagonal from bottom-left to top-right
    let mid = start_color.lerp(end_color, 0.5);
    GradientMesh::grid(
        origin,
        size,
        subdivisions,
        subdivisions,
        start_color,
        mid,
        end_color,
        mid,
    )
}

/// Creates a four-corner gradient mesh.
pub fn four_corner_gradient_mesh(
    origin: Vec2,
    size: Vec2,
    color_bl: Rgba,
    color_br: Rgba,
    color_tr: Rgba,
    color_tl: Rgba,
    subdivisions: usize,
) -> GradientMesh {
    GradientMesh::grid(
        origin,
        size,
        subdivisions,
        subdivisions,
        color_bl,
        color_br,
        color_tr,
        color_tl,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Color helpers for tests
    const RED: Rgba = Rgba::new(1.0, 0.0, 0.0, 1.0);
    const GREEN: Rgba = Rgba::new(0.0, 1.0, 0.0, 1.0);
    const BLUE: Rgba = Rgba::new(0.0, 0.0, 1.0, 1.0);

    #[test]
    fn test_gradient_vertex() {
        let v = GradientVertex::new(Vec2::new(1.0, 2.0), Rgba::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(v.position, Vec2::new(1.0, 2.0));
        assert_eq!(v.color.r, 1.0);
    }

    #[test]
    fn test_gradient_mesh_empty() {
        let mesh = GradientMesh::new();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.faces.is_empty());
    }

    #[test]
    fn test_gradient_mesh_add_vertex() {
        let mut mesh = GradientMesh::new();
        let idx = mesh.add_vertex(GradientVertex::new(Vec2::ZERO, Rgba::WHITE));
        assert_eq!(idx, 0);
        assert_eq!(mesh.vertices.len(), 1);
    }

    #[test]
    fn test_gradient_mesh_grid() {
        let mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            2,
            2,
            RED,
            GREEN,
            BLUE,
            Rgba::WHITE,
        );

        // 3x3 vertices = 9
        assert_eq!(mesh.vertices.len(), 9);
        // 2x2 cells * 2 triangles = 8
        assert_eq!(mesh.faces.len(), 8);
    }

    #[test]
    fn test_gradient_mesh_sample_center() {
        let mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            1,
            1,
            Rgba::BLACK,
            Rgba::BLACK,
            Rgba::WHITE,
            Rgba::WHITE,
        );

        // Center should be gray
        let color = mesh.sample(Vec2::new(5.0, 5.0)).unwrap();
        assert!((color.r - 0.5).abs() < 0.1);
        assert!((color.g - 0.5).abs() < 0.1);
        assert!((color.b - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_gradient_mesh_sample_corner() {
        let mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            1,
            1,
            RED,
            GREEN,
            BLUE,
            Rgba::WHITE,
        );

        // Bottom-left should be red
        let color = mesh.sample(Vec2::new(0.1, 0.1)).unwrap();
        assert!(color.r > 0.8);
    }

    #[test]
    fn test_gradient_mesh_sample_outside() {
        let mesh = GradientMesh::grid(Vec2::ZERO, Vec2::new(10.0, 10.0), 1, 1, RED, RED, RED, RED);

        // Outside should return None
        assert!(mesh.sample(Vec2::new(-5.0, 5.0)).is_none());
        assert!(mesh.sample(Vec2::new(15.0, 5.0)).is_none());
    }

    #[test]
    fn test_radial_gradient_mesh() {
        let mesh = GradientMesh::radial(Vec2::new(5.0, 5.0), 5.0, 3, 8, Rgba::WHITE, Rgba::BLACK);

        // Center + 3 rings * 8 segments = 1 + 24 = 25 vertices
        assert_eq!(mesh.vertices.len(), 25);

        // Center should be white
        let color = mesh.sample(Vec2::new(5.0, 5.0)).unwrap();
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_linear_gradient_horizontal() {
        let mesh = linear_gradient_mesh(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            Rgba::BLACK,
            Rgba::WHITE,
            true,
        );

        // Left should be black, right should be white
        let left = mesh.sample(Vec2::new(1.0, 5.0)).unwrap();
        let right = mesh.sample(Vec2::new(9.0, 5.0)).unwrap();

        assert!(left.r < 0.3);
        assert!(right.r > 0.7);
    }

    #[test]
    fn test_linear_gradient_vertical() {
        let mesh = linear_gradient_mesh(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            Rgba::BLACK,
            Rgba::WHITE,
            false,
        );

        // Bottom should be black, top should be white
        let bottom = mesh.sample(Vec2::new(5.0, 1.0)).unwrap();
        let top = mesh.sample(Vec2::new(5.0, 9.0)).unwrap();

        assert!(bottom.r < 0.3);
        assert!(top.r > 0.7);
    }

    #[test]
    fn test_mesh_bounds() {
        let mesh = GradientMesh::grid(
            Vec2::new(5.0, 10.0),
            Vec2::new(20.0, 30.0),
            1,
            1,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
        );

        let (min, max) = mesh.bounds();
        assert_eq!(min, Vec2::new(5.0, 10.0));
        assert_eq!(max, Vec2::new(25.0, 40.0));
    }

    #[test]
    fn test_mesh_translate() {
        let mut mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            1,
            1,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
        );

        mesh.translate(Vec2::new(5.0, 5.0));

        let (min, _max) = mesh.bounds();
        assert_eq!(min, Vec2::new(5.0, 5.0));
    }

    #[test]
    fn test_mesh_scale() {
        let mut mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            1,
            1,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::WHITE,
        );

        mesh.scale(2.0);

        let (_min, max) = mesh.bounds();
        assert_eq!(max, Vec2::new(20.0, 20.0));
    }

    #[test]
    fn test_mesh_set_alpha() {
        let mut mesh = GradientMesh::grid(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            1,
            1,
            Rgba::new(1.0, 1.0, 1.0, 1.0),
            Rgba::new(1.0, 1.0, 1.0, 1.0),
            Rgba::new(1.0, 1.0, 1.0, 1.0),
            Rgba::new(1.0, 1.0, 1.0, 1.0),
        );

        mesh.set_alpha(0.5);

        for v in &mesh.vertices {
            assert_eq!(v.color.a, 0.5);
        }
    }

    #[test]
    fn test_four_corner_gradient() {
        let mesh = four_corner_gradient_mesh(
            Vec2::ZERO,
            Vec2::new(10.0, 10.0),
            RED,
            GREEN,
            BLUE,
            Rgba::WHITE,
            4,
        );

        // 5x5 vertices
        assert_eq!(mesh.vertices.len(), 25);
    }

    #[test]
    fn test_barycentric_inside() {
        let result = barycentric(
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_barycentric_outside() {
        let result = barycentric(
            Vec2::new(-1.0, -1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert!(result.is_none());
    }
}
