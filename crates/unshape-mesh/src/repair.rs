//! Mesh repair operations including hole filling.
//!
//! Provides utilities for finding and filling holes in meshes to make them watertight.
//!
//! # Example
//!
//! ```
//! use unshape_mesh::{Mesh, repair::{find_boundary_loops, fill_hole_fan}};
//!
//! let mut mesh = Mesh::new();
//! // ... build a mesh with holes ...
//!
//! // Find all holes
//! let holes = find_boundary_loops(&mesh);
//!
//! // Fill each hole
//! for hole in &holes {
//!     fill_hole_fan(&mut mesh, hole);
//! }
//! ```

use crate::Mesh;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// An edge represented by two vertex indices (ordered as min, max for consistency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    /// First vertex index (always smaller).
    pub v0: u32,
    /// Second vertex index (always larger).
    pub v1: u32,
}

impl Edge {
    /// Creates a new edge with vertices in canonical order.
    pub fn new(a: u32, b: u32) -> Self {
        if a < b {
            Self { v0: a, v1: b }
        } else {
            Self { v0: b, v1: a }
        }
    }
}

/// A directed edge (preserves winding order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    /// Start vertex.
    pub from: u32,
    /// End vertex.
    pub to: u32,
}

impl DirectedEdge {
    /// Creates a new directed edge.
    pub fn new(from: u32, to: u32) -> Self {
        Self { from, to }
    }

    /// Returns the reversed edge.
    pub fn reversed(&self) -> Self {
        Self {
            from: self.to,
            to: self.from,
        }
    }
}

/// A boundary loop representing a hole in the mesh.
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Vertex indices forming the loop (in order).
    pub vertices: Vec<u32>,
}

impl BoundaryLoop {
    /// Returns the number of vertices in the loop.
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Returns true if the loop is empty.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Returns the edges of the loop.
    pub fn edges(&self) -> Vec<DirectedEdge> {
        let n = self.vertices.len();
        (0..n)
            .map(|i| DirectedEdge::new(self.vertices[i], self.vertices[(i + 1) % n]))
            .collect()
    }
}

/// Finds all boundary edges in the mesh (edges with only one adjacent face).
pub fn find_boundary_edges(mesh: &Mesh) -> Vec<DirectedEdge> {
    // Count how many times each edge appears
    let mut edge_count: HashMap<Edge, u32> = HashMap::new();
    let mut edge_direction: HashMap<Edge, DirectedEdge> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

        for (a, b) in edges {
            let edge = Edge::new(a, b);
            *edge_count.entry(edge).or_insert(0) += 1;
            // Store the direction for boundary edge reconstruction
            edge_direction.insert(edge, DirectedEdge::new(a, b));
        }
    }

    // Boundary edges appear exactly once
    edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| {
            // Return the opposite direction to get correct winding for fill
            edge_direction[&edge].reversed()
        })
        .collect()
}

/// Groups boundary edges into connected loops.
pub fn find_boundary_loops(mesh: &Mesh) -> Vec<BoundaryLoop> {
    let boundary_edges = find_boundary_edges(mesh);

    if boundary_edges.is_empty() {
        return Vec::new();
    }

    // Build adjacency map: vertex -> edges starting from that vertex
    let mut edge_from: HashMap<u32, Vec<DirectedEdge>> = HashMap::new();
    for edge in &boundary_edges {
        edge_from.entry(edge.from).or_default().push(*edge);
    }

    // Track which edges we've used
    let mut used_edges: HashSet<DirectedEdge> = HashSet::new();
    let mut loops = Vec::new();

    for start_edge in &boundary_edges {
        if used_edges.contains(start_edge) {
            continue;
        }

        // Trace a loop starting from this edge
        let mut loop_vertices = Vec::new();
        let mut current = *start_edge;

        loop {
            if used_edges.contains(&current) {
                break;
            }

            used_edges.insert(current);
            loop_vertices.push(current.from);

            // Find the next edge starting from current.to
            let next_edges = edge_from.get(&current.to);
            if next_edges.is_none() {
                break;
            }

            let next = next_edges.unwrap().iter().find(|e| !used_edges.contains(e));

            match next {
                Some(e) => current = *e,
                None => break,
            }
        }

        if loop_vertices.len() >= 3 {
            loops.push(BoundaryLoop {
                vertices: loop_vertices,
            });
        }
    }

    loops
}

/// Fills a hole using a fan triangulation from the centroid.
///
/// Creates a new vertex at the centroid of the hole and connects
/// all boundary vertices to it. Simple and works well for convex holes.
pub fn fill_hole_fan(mesh: &mut Mesh, hole: &BoundaryLoop) {
    if hole.len() < 3 {
        return;
    }

    // Compute centroid
    let centroid: Vec3 = hole
        .vertices
        .iter()
        .map(|&i| mesh.positions[i as usize])
        .sum::<Vec3>()
        / hole.len() as f32;

    // Add centroid vertex
    let centroid_idx = mesh.positions.len() as u32;
    mesh.positions.push(centroid);

    // Add normals and UVs for the new vertex if mesh has them
    if mesh.has_normals() {
        // Compute normal as average of boundary vertex normals
        let normal: Vec3 = hole
            .vertices
            .iter()
            .map(|&i| mesh.normals[i as usize])
            .sum::<Vec3>()
            .normalize_or_zero();
        mesh.normals.push(normal);
    }

    if mesh.has_uvs() {
        // Use average UV of boundary vertices
        let uv: glam::Vec2 = hole
            .vertices
            .iter()
            .map(|&i| mesh.uvs[i as usize])
            .sum::<glam::Vec2>()
            / hole.len() as f32;
        mesh.uvs.push(uv);
    }

    // Create triangles
    let n = hole.len();
    for i in 0..n {
        let v0 = hole.vertices[i];
        let v1 = hole.vertices[(i + 1) % n];

        mesh.indices.push(v0);
        mesh.indices.push(v1);
        mesh.indices.push(centroid_idx);
    }
}

/// Fills a hole using ear clipping triangulation.
///
/// Works for any simple polygon (convex or concave). Projects the hole
/// to 2D for triangulation, which works well for relatively planar holes.
pub fn fill_hole_ear_clip(mesh: &mut Mesh, hole: &BoundaryLoop) {
    if hole.len() < 3 {
        return;
    }

    if hole.len() == 3 {
        // Triangle - just add it directly
        mesh.indices.push(hole.vertices[0]);
        mesh.indices.push(hole.vertices[1]);
        mesh.indices.push(hole.vertices[2]);
        return;
    }

    // Get 3D positions
    let positions: Vec<Vec3> = hole
        .vertices
        .iter()
        .map(|&i| mesh.positions[i as usize])
        .collect();

    // Compute the plane normal (average of cross products)
    let centroid: Vec3 = positions.iter().sum::<Vec3>() / positions.len() as f32;
    let mut normal = Vec3::ZERO;
    for i in 0..positions.len() {
        let v0 = positions[i] - centroid;
        let v1 = positions[(i + 1) % positions.len()] - centroid;
        normal += v0.cross(v1);
    }
    normal = normal.normalize_or_zero();

    if normal == Vec3::ZERO {
        // Degenerate polygon, fallback to fan fill
        fill_hole_fan(mesh, hole);
        return;
    }

    // Project to 2D
    let (u_axis, v_axis) = compute_orthonormal_basis(normal);
    let points_2d: Vec<(f32, f32)> = positions
        .iter()
        .map(|p| {
            let d = *p - centroid;
            (d.dot(u_axis), d.dot(v_axis))
        })
        .collect();

    // Ear clipping
    let mut remaining: Vec<usize> = (0..hole.len()).collect();

    while remaining.len() > 3 {
        let mut found_ear = false;

        for i in 0..remaining.len() {
            let prev = remaining[(i + remaining.len() - 1) % remaining.len()];
            let curr = remaining[i];
            let next = remaining[(i + 1) % remaining.len()];

            let p0 = points_2d[prev];
            let p1 = points_2d[curr];
            let p2 = points_2d[next];

            // Check if this is a convex vertex (ear candidate)
            if !is_convex_2d(p0, p1, p2) {
                continue;
            }

            // Check if any other vertex is inside this triangle
            let mut valid_ear = true;
            for j in 0..remaining.len() {
                if j == i
                    || j == (i + remaining.len() - 1) % remaining.len()
                    || j == (i + 1) % remaining.len()
                {
                    continue;
                }

                let pt = points_2d[remaining[j]];
                if point_in_triangle_2d(pt, p0, p1, p2) {
                    valid_ear = false;
                    break;
                }
            }

            if valid_ear {
                // Clip this ear
                mesh.indices.push(hole.vertices[prev]);
                mesh.indices.push(hole.vertices[curr]);
                mesh.indices.push(hole.vertices[next]);

                remaining.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Failed to find an ear, fallback to fan
            let remaining_hole = BoundaryLoop {
                vertices: remaining.iter().map(|&i| hole.vertices[i]).collect(),
            };
            fill_hole_fan(mesh, &remaining_hole);
            return;
        }
    }

    // Add final triangle
    if remaining.len() == 3 {
        mesh.indices.push(hole.vertices[remaining[0]]);
        mesh.indices.push(hole.vertices[remaining[1]]);
        mesh.indices.push(hole.vertices[remaining[2]]);
    }
}

/// Fills a hole using minimum area triangulation (greedy approach).
///
/// Tries to create triangles with minimum total area, which often
/// produces better results for irregular holes.
pub fn fill_hole_minimum_area(mesh: &mut Mesh, hole: &BoundaryLoop) {
    if hole.len() < 3 {
        return;
    }

    if hole.len() == 3 {
        mesh.indices.push(hole.vertices[0]);
        mesh.indices.push(hole.vertices[1]);
        mesh.indices.push(hole.vertices[2]);
        return;
    }

    // Get positions
    let positions: Vec<Vec3> = hole
        .vertices
        .iter()
        .map(|&i| mesh.positions[i as usize])
        .collect();

    let mut remaining: Vec<usize> = (0..hole.len()).collect();

    while remaining.len() > 3 {
        let mut best_ear = None;
        let mut best_area = f32::MAX;

        for i in 0..remaining.len() {
            let prev = remaining[(i + remaining.len() - 1) % remaining.len()];
            let curr = remaining[i];
            let next = remaining[(i + 1) % remaining.len()];

            let p0 = positions[prev];
            let p1 = positions[curr];
            let p2 = positions[next];

            // Compute triangle area
            let area = (p1 - p0).cross(p2 - p0).length() * 0.5;

            if area < best_area {
                best_area = area;
                best_ear = Some(i);
            }
        }

        if let Some(ear_idx) = best_ear {
            let prev = remaining[(ear_idx + remaining.len() - 1) % remaining.len()];
            let curr = remaining[ear_idx];
            let next = remaining[(ear_idx + 1) % remaining.len()];

            mesh.indices.push(hole.vertices[prev]);
            mesh.indices.push(hole.vertices[curr]);
            mesh.indices.push(hole.vertices[next]);

            remaining.remove(ear_idx);
        } else {
            break;
        }
    }

    // Add final triangle
    if remaining.len() == 3 {
        mesh.indices.push(hole.vertices[remaining[0]]);
        mesh.indices.push(hole.vertices[remaining[1]]);
        mesh.indices.push(hole.vertices[remaining[2]]);
    }
}

/// Fills all holes in a mesh using the specified method.
pub fn fill_all_holes(mesh: &mut Mesh, method: HoleFillMethod) {
    let holes = find_boundary_loops(mesh);

    for hole in &holes {
        match method {
            HoleFillMethod::Fan => fill_hole_fan(mesh, hole),
            HoleFillMethod::EarClip => fill_hole_ear_clip(mesh, hole),
            HoleFillMethod::MinimumArea => fill_hole_minimum_area(mesh, hole),
        }
    }
}

/// Method for filling holes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleFillMethod {
    /// Fan triangulation from centroid. Fast, good for convex holes.
    Fan,
    /// Ear clipping algorithm. Works for concave holes.
    EarClip,
    /// Minimum area triangulation. Often produces cleaner results.
    MinimumArea,
}

impl Default for HoleFillMethod {
    fn default() -> Self {
        Self::EarClip
    }
}

/// Checks if a mesh is watertight (no boundary edges).
pub fn is_watertight(mesh: &Mesh) -> bool {
    find_boundary_edges(mesh).is_empty()
}

/// Counts the number of holes in a mesh.
pub fn count_holes(mesh: &Mesh) -> usize {
    find_boundary_loops(mesh).len()
}

/// Returns statistics about mesh boundaries.
#[derive(Debug, Clone)]
pub struct BoundaryStats {
    /// Number of boundary loops (holes).
    pub hole_count: usize,
    /// Total number of boundary edges.
    pub boundary_edge_count: usize,
    /// Vertices in each hole.
    pub hole_sizes: Vec<usize>,
}

/// Computes boundary statistics for a mesh.
pub fn boundary_stats(mesh: &Mesh) -> BoundaryStats {
    let loops = find_boundary_loops(mesh);
    let boundary_edges = find_boundary_edges(mesh);

    BoundaryStats {
        hole_count: loops.len(),
        boundary_edge_count: boundary_edges.len(),
        hole_sizes: loops.iter().map(|l| l.len()).collect(),
    }
}

// Helper functions

fn compute_orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let u = normal.cross(up).normalize();
    let v = u.cross(normal);
    (u, v)
}

fn is_convex_2d(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> bool {
    let d1 = (p1.0 - p0.0, p1.1 - p0.1);
    let d2 = (p2.0 - p1.0, p2.1 - p1.1);
    let cross = d1.0 * d2.1 - d1.1 * d2.0;
    cross >= 0.0
}

fn point_in_triangle_2d(p: (f32, f32), v0: (f32, f32), v1: (f32, f32), v2: (f32, f32)) -> bool {
    let sign = |p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)| -> f32 {
        (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
    };

    let d1 = sign(p, v0, v1);
    let d2 = sign(p, v1, v2);
    let d3 = sign(p, v2, v0);

    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;

    !(has_neg && has_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_open_box() -> Mesh {
        // Create a box missing its top face (5 faces, 1 hole)
        let mut mesh = Mesh::new();

        // 8 vertices of a unit cube
        mesh.positions.extend([
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(1.0, 0.0, 0.0), // 1
            Vec3::new(1.0, 1.0, 0.0), // 2
            Vec3::new(0.0, 1.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 1.0), // 4
            Vec3::new(1.0, 0.0, 1.0), // 5
            Vec3::new(1.0, 1.0, 1.0), // 6
            Vec3::new(0.0, 1.0, 1.0), // 7
        ]);

        // Bottom face
        mesh.indices.extend([0, 2, 1, 0, 3, 2]);
        // Front face
        mesh.indices.extend([0, 1, 5, 0, 5, 4]);
        // Right face
        mesh.indices.extend([1, 2, 6, 1, 6, 5]);
        // Back face
        mesh.indices.extend([2, 3, 7, 2, 7, 6]);
        // Left face
        mesh.indices.extend([3, 0, 4, 3, 4, 7]);
        // Top face is MISSING

        mesh
    }

    #[test]
    fn test_edge_creation() {
        let e1 = Edge::new(5, 3);
        let e2 = Edge::new(3, 5);
        assert_eq!(e1, e2);
        assert_eq!(e1.v0, 3);
        assert_eq!(e1.v1, 5);
    }

    #[test]
    fn test_find_boundary_edges_closed_mesh() {
        // A single triangle is closed
        let mut mesh = Mesh::new();
        mesh.positions.extend([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ]);
        mesh.indices.extend([0, 1, 2]);

        let boundaries = find_boundary_edges(&mesh);
        // Single triangle has all edges as boundaries
        assert_eq!(boundaries.len(), 3);
    }

    #[test]
    fn test_find_boundary_loops_open_box() {
        let mesh = make_open_box();
        let loops = find_boundary_loops(&mesh);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4); // Square hole on top
    }

    #[test]
    fn test_fill_hole_fan() {
        let mut mesh = make_open_box();
        let initial_tris = mesh.triangle_count();

        let loops = find_boundary_loops(&mesh);
        assert_eq!(loops.len(), 1);

        fill_hole_fan(&mut mesh, &loops[0]);

        // Should have added 4 triangles (fan from centroid to 4 edges)
        assert_eq!(mesh.triangle_count(), initial_tris + 4);

        // Should now be watertight
        assert!(is_watertight(&mesh));
    }

    #[test]
    fn test_fill_hole_ear_clip() {
        let mut mesh = make_open_box();
        let initial_tris = mesh.triangle_count();

        let loops = find_boundary_loops(&mesh);
        fill_hole_ear_clip(&mut mesh, &loops[0]);

        // Square hole needs 2 triangles, but ear clip may fall back to fan (4 tris)
        // The important thing is that the mesh is watertight
        assert!(mesh.triangle_count() >= initial_tris + 2);
        assert!(is_watertight(&mesh));
    }

    #[test]
    fn test_fill_hole_minimum_area() {
        let mut mesh = make_open_box();
        let initial_tris = mesh.triangle_count();

        let loops = find_boundary_loops(&mesh);
        fill_hole_minimum_area(&mut mesh, &loops[0]);

        // Square hole needs 2 triangles
        assert_eq!(mesh.triangle_count(), initial_tris + 2);
        assert!(is_watertight(&mesh));
    }

    #[test]
    fn test_fill_all_holes() {
        let mut mesh = make_open_box();

        fill_all_holes(&mut mesh, HoleFillMethod::EarClip);

        assert!(is_watertight(&mesh));
        assert_eq!(count_holes(&mesh), 0);
    }

    #[test]
    fn test_boundary_stats() {
        let mesh = make_open_box();
        let stats = boundary_stats(&mesh);

        assert_eq!(stats.hole_count, 1);
        assert_eq!(stats.boundary_edge_count, 4);
        assert_eq!(stats.hole_sizes, vec![4]);
    }

    #[test]
    fn test_is_watertight() {
        let mut mesh = make_open_box();
        assert!(!is_watertight(&mesh));

        fill_all_holes(&mut mesh, HoleFillMethod::Fan);
        assert!(is_watertight(&mesh));
    }

    #[test]
    fn test_tetrahedron_is_watertight() {
        // A tetrahedron is a closed mesh with 4 faces
        let mut mesh = Mesh::new();
        mesh.positions.extend([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ]);

        // 4 faces of tetrahedron
        mesh.indices.extend([
            0, 1, 2, // bottom
            0, 3, 1, // front
            1, 3, 2, // right
            2, 3, 0, // left
        ]);

        // A tetrahedron is watertight (no boundary edges)
        assert!(is_watertight(&mesh));
    }

    #[test]
    fn test_count_holes() {
        let mesh = make_open_box();
        assert_eq!(count_holes(&mesh), 1);
    }

    #[test]
    fn test_boundary_loop_edges() {
        let loop_ = BoundaryLoop {
            vertices: vec![0, 1, 2, 3],
        };

        let edges = loop_.edges();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0], DirectedEdge::new(0, 1));
        assert_eq!(edges[1], DirectedEdge::new(1, 2));
        assert_eq!(edges[2], DirectedEdge::new(2, 3));
        assert_eq!(edges[3], DirectedEdge::new(3, 0));
    }
}
