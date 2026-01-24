//! Delaunay triangulation and Voronoi diagrams.
//!
//! Implements Bowyer-Watson algorithm for Delaunay triangulation and computes
//! Voronoi diagrams as the dual graph.
//!
//! # Example
//!
//! ```
//! use glam::Vec2;
//! use unshape_vector::{delaunay_triangulation, voronoi_diagram};
//!
//! let points = vec![
//!     Vec2::new(0.0, 0.0),
//!     Vec2::new(1.0, 0.0),
//!     Vec2::new(0.5, 1.0),
//!     Vec2::new(0.5, 0.5),
//! ];
//!
//! // Get triangles as vertex indices
//! let triangles = delaunay_triangulation(&points);
//!
//! // Get Voronoi diagram
//! let voronoi = voronoi_diagram(&points);
//! ```

use glam::Vec2;

/// A triangle defined by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    /// First vertex index.
    pub a: usize,
    /// Second vertex index.
    pub b: usize,
    /// Third vertex index.
    pub c: usize,
}

impl Triangle {
    /// Creates a new triangle.
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Returns the vertices as an array.
    pub fn vertices(&self) -> [usize; 3] {
        [self.a, self.b, self.c]
    }

    /// Returns the edges as pairs of vertex indices.
    pub fn edges(&self) -> [(usize, usize); 3] {
        [
            (self.a.min(self.b), self.a.max(self.b)),
            (self.b.min(self.c), self.b.max(self.c)),
            (self.c.min(self.a), self.c.max(self.a)),
        ]
    }

    /// Checks if the triangle contains a vertex index.
    pub fn contains_vertex(&self, v: usize) -> bool {
        self.a == v || self.b == v || self.c == v
    }
}

/// A Voronoi cell - a region of space closest to one site.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// The site (point) this cell belongs to.
    pub site: usize,
    /// Vertices of the cell boundary in order.
    pub vertices: Vec<Vec2>,
}

/// A Voronoi diagram.
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// The original sites (points).
    pub sites: Vec<Vec2>,
    /// The cells, one per site.
    pub cells: Vec<VoronoiCell>,
    /// All Voronoi vertices (circumcenters of Delaunay triangles).
    pub vertices: Vec<Vec2>,
    /// Edges as pairs of vertex indices (None means edge goes to infinity).
    pub edges: Vec<(Option<usize>, Option<usize>)>,
}

/// Computes the Delaunay triangulation of a set of points.
///
/// Uses the Bowyer-Watson algorithm. Returns triangles as vertex indices.
///
/// # Example
///
/// ```
/// use glam::Vec2;
/// use unshape_vector::delaunay_triangulation;
///
/// let points = vec![
///     Vec2::new(0.0, 0.0),
///     Vec2::new(1.0, 0.0),
///     Vec2::new(0.5, 1.0),
/// ];
///
/// let triangles = delaunay_triangulation(&points);
/// assert_eq!(triangles.len(), 1); // One triangle for 3 points
/// ```
pub fn delaunay_triangulation(points: &[Vec2]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Create super-triangle that contains all points
    let super_tri = create_super_triangle(points);

    // Extended points list including super-triangle vertices
    let n = points.len();
    let mut all_points: Vec<Vec2> = points.to_vec();
    all_points.extend_from_slice(&super_tri);

    // Start with super-triangle
    let mut triangles = vec![TriangleInternal {
        a: n,
        b: n + 1,
        c: n + 2,
    }];

    // Insert each point
    for i in 0..n {
        let point = all_points[i];

        // Find triangles whose circumcircle contains this point
        let mut bad_triangles = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            let (center, radius_sq) = circumcircle(&all_points, tri);
            let dist_sq = (point - center).length_squared();
            if dist_sq <= radius_sq + 1e-6 {
                bad_triangles.push(ti);
            }
        }

        // Find the boundary of the polygonal hole
        let mut polygon = Vec::new();
        for &ti in &bad_triangles {
            let tri = &triangles[ti];
            let edges = [(tri.a, tri.b), (tri.b, tri.c), (tri.c, tri.a)];

            for edge in edges {
                let is_shared = bad_triangles.iter().any(|&other_ti| {
                    if other_ti == ti {
                        return false;
                    }
                    let other = &triangles[other_ti];
                    let other_edges = [(other.a, other.b), (other.b, other.c), (other.c, other.a)];
                    other_edges.contains(&edge) || other_edges.contains(&(edge.1, edge.0))
                });

                if !is_shared {
                    polygon.push(edge);
                }
            }
        }

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_unstable();
        for ti in bad_triangles.into_iter().rev() {
            triangles.swap_remove(ti);
        }

        // Create new triangles from polygon edges to the new point
        for (e1, e2) in polygon {
            triangles.push(TriangleInternal { a: e1, b: e2, c: i });
        }
    }

    // Remove triangles that share vertices with super-triangle
    triangles.retain(|tri| tri.a < n && tri.b < n && tri.c < n);

    // Convert to public Triangle type
    triangles
        .into_iter()
        .map(|t| Triangle::new(t.a, t.b, t.c))
        .collect()
}

/// Internal triangle representation.
#[derive(Clone, Copy)]
struct TriangleInternal {
    a: usize,
    b: usize,
    c: usize,
}

/// Creates a super-triangle that contains all points.
fn create_super_triangle(points: &[Vec2]) -> [Vec2; 3] {
    // Find bounding box
    let mut min = points[0];
    let mut max = points[0];
    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }

    let dx = max.x - min.x;
    let dy = max.y - min.y;
    let delta_max = dx.max(dy).max(1.0);
    let mid_x = (min.x + max.x) / 2.0;
    let mid_y = (min.y + max.y) / 2.0;

    // Create large triangle that definitely contains all points
    [
        Vec2::new(mid_x - 20.0 * delta_max, mid_y - delta_max),
        Vec2::new(mid_x, mid_y + 20.0 * delta_max),
        Vec2::new(mid_x + 20.0 * delta_max, mid_y - delta_max),
    ]
}

/// Computes circumcircle center and squared radius.
fn circumcircle(points: &[Vec2], tri: &TriangleInternal) -> (Vec2, f32) {
    let a = points[tri.a];
    let b = points[tri.b];
    let c = points[tri.c];

    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if d.abs() < 1e-10 {
        // Degenerate triangle (collinear points)
        let center = (a + b + c) / 3.0;
        let radius_sq = (a - center)
            .length_squared()
            .max((b - center).length_squared());
        return (center, radius_sq);
    }

    let a_sq = a.x * a.x + a.y * a.y;
    let b_sq = b.x * b.x + b.y * b.y;
    let c_sq = c.x * c.x + c.y * c.y;

    let ux = (a_sq * (b.y - c.y) + b_sq * (c.y - a.y) + c_sq * (a.y - b.y)) / d;
    let uy = (a_sq * (c.x - b.x) + b_sq * (a.x - c.x) + c_sq * (b.x - a.x)) / d;

    let center = Vec2::new(ux, uy);
    let radius_sq = (a - center).length_squared();

    (center, radius_sq)
}

/// Computes the Voronoi diagram from a set of points.
///
/// The Voronoi diagram is the dual of the Delaunay triangulation.
/// Each Voronoi cell contains all points closest to one site.
///
/// # Example
///
/// ```
/// use glam::Vec2;
/// use unshape_vector::voronoi_diagram;
///
/// let points = vec![
///     Vec2::new(0.0, 0.0),
///     Vec2::new(1.0, 0.0),
///     Vec2::new(0.5, 1.0),
/// ];
///
/// let voronoi = voronoi_diagram(&points);
/// assert_eq!(voronoi.cells.len(), 3); // One cell per point
/// ```
pub fn voronoi_diagram(points: &[Vec2]) -> VoronoiDiagram {
    if points.len() < 3 {
        return VoronoiDiagram {
            sites: points.to_vec(),
            cells: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
        };
    }

    let triangles = delaunay_triangulation(points);

    // Compute circumcenters (Voronoi vertices)
    let mut voronoi_vertices = Vec::new();
    for tri in &triangles {
        let a = points[tri.a];
        let b = points[tri.b];
        let c = points[tri.c];

        if let Some(center) = compute_circumcenter(a, b, c) {
            voronoi_vertices.push(center);
        } else {
            // Fallback for degenerate triangles
            voronoi_vertices.push((a + b + c) / 3.0);
        }
    }

    // Build adjacency: for each point, which triangles contain it
    let mut point_triangles: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (ti, tri) in triangles.iter().enumerate() {
        point_triangles[tri.a].push(ti);
        point_triangles[tri.b].push(ti);
        point_triangles[tri.c].push(ti);
    }

    // Build Voronoi cells
    let mut cells = Vec::with_capacity(points.len());
    for (pi, tri_indices) in point_triangles.iter().enumerate() {
        if tri_indices.is_empty() {
            cells.push(VoronoiCell {
                site: pi,
                vertices: Vec::new(),
            });
            continue;
        }

        // Sort triangles around the point to get cell vertices in order
        let mut cell_vertices = Vec::new();
        let ordered = order_triangles_around_point(pi, tri_indices, &triangles);

        for ti in ordered {
            cell_vertices.push(voronoi_vertices[ti]);
        }

        cells.push(VoronoiCell {
            site: pi,
            vertices: cell_vertices,
        });
    }

    // Build Voronoi edges from Delaunay edges
    let mut edges = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();

    for (ti, tri) in triangles.iter().enumerate() {
        let tri_edges = tri.edges();
        for &(e1, e2) in tri_edges.iter() {
            if seen_edges.contains(&(e1, e2)) {
                continue;
            }
            seen_edges.insert((e1, e2));

            // Find adjacent triangle sharing this edge
            let adjacent = triangles
                .iter()
                .enumerate()
                .find(|(other_ti, other)| *other_ti != ti && other.edges().contains(&(e1, e2)));

            match adjacent {
                Some((other_ti, _)) => {
                    edges.push((Some(ti), Some(other_ti)));
                }
                None => {
                    // Edge on convex hull - goes to infinity
                    edges.push((Some(ti), None));
                }
            }
        }
    }

    VoronoiDiagram {
        sites: points.to_vec(),
        cells,
        vertices: voronoi_vertices,
        edges,
    }
}

/// Computes the circumcenter of a triangle.
fn compute_circumcenter(a: Vec2, b: Vec2, c: Vec2) -> Option<Vec2> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if d.abs() < 1e-10 {
        return None;
    }

    let a_sq = a.x * a.x + a.y * a.y;
    let b_sq = b.x * b.x + b.y * b.y;
    let c_sq = c.x * c.x + c.y * c.y;

    let ux = (a_sq * (b.y - c.y) + b_sq * (c.y - a.y) + c_sq * (a.y - b.y)) / d;
    let uy = (a_sq * (c.x - b.x) + b_sq * (a.x - c.x) + c_sq * (b.x - a.x)) / d;

    Some(Vec2::new(ux, uy))
}

/// Orders triangles around a point to form a continuous cell boundary.
fn order_triangles_around_point(
    point_idx: usize,
    tri_indices: &[usize],
    triangles: &[Triangle],
) -> Vec<usize> {
    if tri_indices.is_empty() {
        return Vec::new();
    }

    if tri_indices.len() == 1 {
        return tri_indices.to_vec();
    }

    // Build a graph of adjacent triangles
    let mut ordered = vec![tri_indices[0]];
    let mut remaining: Vec<usize> = tri_indices[1..].to_vec();

    while !remaining.is_empty() {
        let last = *ordered.last().unwrap();
        let last_tri = &triangles[last];

        // Find a triangle that shares an edge with the last one
        let next_idx = remaining.iter().position(|&ti| {
            let tri = &triangles[ti];
            shares_edge_at_point(last_tri, tri, point_idx)
        });

        match next_idx {
            Some(idx) => {
                ordered.push(remaining.remove(idx));
            }
            None => {
                // No adjacent triangle found - might be on boundary
                // Just add remaining in arbitrary order
                ordered.extend(remaining.drain(..));
            }
        }
    }

    ordered
}

/// Checks if two triangles share an edge that includes the given point.
fn shares_edge_at_point(t1: &Triangle, t2: &Triangle, point: usize) -> bool {
    let e1 = t1.edges();
    let e2 = t2.edges();

    for edge1 in &e1 {
        if edge1.0 != point && edge1.1 != point {
            continue;
        }
        for edge2 in &e2 {
            if edge1 == edge2 || (edge1.0 == edge2.1 && edge1.1 == edge2.0) {
                return true;
            }
        }
    }

    false
}

/// Returns the triangles as flat indices (for rendering).
pub fn triangles_to_indices(triangles: &[Triangle]) -> Vec<u32> {
    let mut indices = Vec::with_capacity(triangles.len() * 3);
    for tri in triangles {
        indices.push(tri.a as u32);
        indices.push(tri.b as u32);
        indices.push(tri.c as u32);
    }
    indices
}

/// Returns Voronoi cell edges as line segments.
pub fn voronoi_to_segments(voronoi: &VoronoiDiagram) -> Vec<(Vec2, Vec2)> {
    let mut segments = Vec::new();

    for cell in &voronoi.cells {
        let n = cell.vertices.len();
        if n < 2 {
            continue;
        }

        for i in 0..n {
            let j = (i + 1) % n;
            segments.push((cell.vertices[i], cell.vertices[j]));
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_basic() {
        let tri = Triangle::new(0, 1, 2);
        assert_eq!(tri.vertices(), [0, 1, 2]);
        assert!(tri.contains_vertex(0));
        assert!(tri.contains_vertex(1));
        assert!(tri.contains_vertex(2));
        assert!(!tri.contains_vertex(3));
    }

    #[test]
    fn test_triangle_edges() {
        let tri = Triangle::new(0, 1, 2);
        let edges = tri.edges();
        // Edges are normalized to (min, max)
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(0, 2)));
    }

    #[test]
    fn test_delaunay_three_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 1);

        let tri = &triangles[0];
        assert!(tri.contains_vertex(0));
        assert!(tri.contains_vertex(1));
        assert!(tri.contains_vertex(2));
    }

    #[test]
    fn test_delaunay_four_points_square() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 2); // Square -> 2 triangles
    }

    #[test]
    fn test_delaunay_five_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.5, 0.5), // Center point
        ];

        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 4); // Center divides into 4 triangles
    }

    #[test]
    fn test_delaunay_fewer_than_three() {
        assert!(delaunay_triangulation(&[]).is_empty());
        assert!(delaunay_triangulation(&[Vec2::ZERO]).is_empty());
        assert!(delaunay_triangulation(&[Vec2::ZERO, Vec2::ONE]).is_empty());
    }

    #[test]
    fn test_voronoi_three_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];

        let voronoi = voronoi_diagram(&points);
        assert_eq!(voronoi.sites.len(), 3);
        assert_eq!(voronoi.cells.len(), 3);
        assert_eq!(voronoi.vertices.len(), 1); // One triangle = one circumcenter
    }

    #[test]
    fn test_voronoi_four_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let voronoi = voronoi_diagram(&points);
        assert_eq!(voronoi.sites.len(), 4);
        assert_eq!(voronoi.cells.len(), 4);
        assert_eq!(voronoi.vertices.len(), 2); // 2 triangles = 2 circumcenters
    }

    #[test]
    fn test_triangles_to_indices() {
        let triangles = vec![Triangle::new(0, 1, 2), Triangle::new(0, 2, 3)];

        let indices = triangles_to_indices(&triangles);
        assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn test_voronoi_to_segments() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];

        let voronoi = voronoi_diagram(&points);
        let segments = voronoi_to_segments(&voronoi);

        // Each cell should have segments if it has vertices
        assert!(!segments.is_empty() || voronoi.cells.iter().all(|c| c.vertices.len() < 2));
    }

    #[test]
    fn test_circumcenter() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.5, 1.0);

        let center = compute_circumcenter(a, b, c).unwrap();

        // Verify all points are equidistant from center
        let da = (a - center).length();
        let db = (b - center).length();
        let dc = (c - center).length();

        assert!((da - db).abs() < 0.001);
        assert!((db - dc).abs() < 0.001);
    }
}
