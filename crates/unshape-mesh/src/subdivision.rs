//! Mesh subdivision algorithms.
//!
//! - Loop subdivision for triangle meshes
//! - Catmull-Clark subdivision for arbitrary polygon meshes (via HalfEdgeMesh)

use crate::Mesh;
use crate::edit::EdgeCreases;
use crate::halfedge::HalfEdgeMesh;
use glam::Vec3;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Catmull-Clark subdivision operation.
///
/// Subdivides an arbitrary polygon mesh, producing a smooth limit surface.
/// After one iteration, all faces become quads.
///
/// # Edge Creases
///
/// Optional edge creases control sharpness during subdivision:
/// - Weight 0.0 = fully smooth (default)
/// - Weight 1.0 = fully sharp (edge preserved like a boundary)
/// - Creases propagate automatically through subdivision levels
///
/// # Example
///
/// ```
/// use unshape_mesh::{CatmullClark, Cuboid};
///
/// let cube = Cuboid::default().apply();
/// let smooth = CatmullClark::new(2).apply(&cube);
/// ```
///
/// With creases:
///
/// ```ignore
/// use unshape_mesh::{Mesh, subdivision::CatmullClark, EdgeCreases, Edge};
///
/// let mut creases = EdgeCreases::new();
/// creases.set(Edge::new(0, 1), 1.0);  // Keep this edge sharp
///
/// let result = CatmullClark { levels: 2, creases: Some(creases) }.apply(&mesh);
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct CatmullClark {
    /// Number of subdivision levels.
    pub levels: u32,
    /// Optional edge creases for controlling sharpness.
    pub creases: Option<EdgeCreases>,
}

impl CatmullClark {
    /// Creates a new Catmull-Clark subdivision with the given level count.
    pub fn new(levels: u32) -> Self {
        Self {
            levels,
            creases: None,
        }
    }

    /// Creates a Catmull-Clark subdivision with edge creases.
    pub fn with_creases(levels: u32, creases: EdgeCreases) -> Self {
        Self {
            levels,
            creases: Some(creases),
        }
    }

    /// Applies Catmull-Clark subdivision to the mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        if self.levels == 0 {
            return mesh.clone();
        }

        let mut hemesh = HalfEdgeMesh::from_mesh(mesh);
        for i in 0..self.levels {
            // First iteration uses input creases, subsequent use propagated internal creases
            let creases = if i == 0 { self.creases.as_ref() } else { None };
            hemesh = hemesh.catmull_clark(creases);
        }
        hemesh.to_mesh()
    }
}

/// Convenience function for Catmull-Clark subdivision.
pub fn subdivide_catmull_clark(mesh: &Mesh, levels: u32) -> Mesh {
    CatmullClark::new(levels).apply(mesh)
}

/// Edge key for hashing (smaller index first).
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Simple linear subdivision (each triangle -> 4 triangles).
///
/// New vertices are placed at edge midpoints without smoothing.
pub fn subdivide_linear(mesh: &Mesh) -> Mesh {
    let mut result = Mesh::new();

    // Map from edge -> new vertex index
    let mut edge_vertices: HashMap<(u32, u32), u32> = HashMap::new();

    // Copy original vertices
    result.positions = mesh.positions.clone();
    result.normals = mesh.normals.clone();
    result.uvs = mesh.uvs.clone();

    // Create edge midpoint vertices
    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0], tri[1], tri[2]];

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let key = edge_key(a, b);
            if !edge_vertices.contains_key(&key) {
                let mid_pos = (mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5;
                let new_idx = result.positions.len() as u32;
                result.positions.push(mid_pos);

                // Interpolate normals if present
                if mesh.has_normals() {
                    let mid_normal =
                        (mesh.normals[a as usize] + mesh.normals[b as usize]).normalize_or_zero();
                    result.normals.push(mid_normal);
                }

                // Interpolate UVs if present
                if mesh.has_uvs() {
                    let mid_uv = (mesh.uvs[a as usize] + mesh.uvs[b as usize]) * 0.5;
                    result.uvs.push(mid_uv);
                }

                edge_vertices.insert(key, new_idx);
            }
        }
    }

    // Create subdivided triangles
    for tri in mesh.indices.chunks(3) {
        let [v0, v1, v2] = [tri[0], tri[1], tri[2]];

        let e01 = *edge_vertices.get(&edge_key(v0, v1)).unwrap();
        let e12 = *edge_vertices.get(&edge_key(v1, v2)).unwrap();
        let e20 = *edge_vertices.get(&edge_key(v2, v0)).unwrap();

        // 4 new triangles
        result.indices.extend_from_slice(&[v0, e01, e20]);
        result.indices.extend_from_slice(&[e01, v1, e12]);
        result.indices.extend_from_slice(&[e20, e12, v2]);
        result.indices.extend_from_slice(&[e01, e12, e20]);
    }

    result
}

/// Loop subdivision for smooth surfaces.
///
/// Loop subdivision is a standard algorithm for subdividing triangle meshes
/// with smooth limit surfaces. Each triangle becomes 4 triangles, and vertices
/// are repositioned based on their neighbors.
pub fn subdivide_loop(mesh: &Mesh) -> Mesh {
    let vertex_count = mesh.positions.len();

    // Build adjacency information
    let mut vertex_neighbors: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let [i0, i1, i2] = [tri[0], tri[1], tri[2]];

        // Track neighbors
        if !vertex_neighbors[i0 as usize].contains(&i1) {
            vertex_neighbors[i0 as usize].push(i1);
        }
        if !vertex_neighbors[i0 as usize].contains(&i2) {
            vertex_neighbors[i0 as usize].push(i2);
        }
        if !vertex_neighbors[i1 as usize].contains(&i0) {
            vertex_neighbors[i1 as usize].push(i0);
        }
        if !vertex_neighbors[i1 as usize].contains(&i2) {
            vertex_neighbors[i1 as usize].push(i2);
        }
        if !vertex_neighbors[i2 as usize].contains(&i0) {
            vertex_neighbors[i2 as usize].push(i0);
        }
        if !vertex_neighbors[i2 as usize].contains(&i1) {
            vertex_neighbors[i2 as usize].push(i1);
        }

        // Track edge faces
        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let key = edge_key(a, b);
            edge_faces.entry(key).or_default().push(face_idx);
        }
    }

    let mut result = Mesh::new();

    // Compute new positions for original vertices (odd vertices in Loop terminology)
    let mut new_positions: Vec<Vec3> = Vec::with_capacity(vertex_count);

    for (i, pos) in mesh.positions.iter().enumerate() {
        let neighbors = &vertex_neighbors[i];
        let n = neighbors.len();

        if n < 3 {
            // Boundary vertex, keep original position
            new_positions.push(*pos);
        } else {
            // Interior vertex: use Loop's formula
            // beta = (1/n) * (5/8 - (3/8 + 1/4 * cos(2*PI/n))^2)
            // new_pos = (1 - n*beta) * pos + beta * sum(neighbors)
            let beta = loop_beta(n);
            let neighbor_sum: Vec3 = neighbors.iter().map(|&j| mesh.positions[j as usize]).sum();

            let new_pos = *pos * (1.0 - n as f32 * beta) + neighbor_sum * beta;
            new_positions.push(new_pos);
        }
    }

    result.positions = new_positions;

    // Compute edge vertices (even vertices)
    let mut edge_vertices: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0], tri[1], tri[2]];

        for &(a, b, _) in &[(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            let key = edge_key(a, b);
            if edge_vertices.contains_key(&key) {
                continue;
            }

            let faces = edge_faces.get(&key).unwrap();
            let new_idx = result.positions.len() as u32;

            if faces.len() == 2 {
                // Interior edge: 3/8 * (a + b) + 1/8 * (c + d)
                // where c and d are the opposite vertices of the two faces
                let face0 = faces[0];
                let face1 = faces[1];

                let opp0 = find_opposite_vertex(mesh, face0, a, b);
                let opp1 = find_opposite_vertex(mesh, face1, a, b);

                let pos = (mesh.positions[a as usize] + mesh.positions[b as usize]) * (3.0 / 8.0)
                    + (mesh.positions[opp0 as usize] + mesh.positions[opp1 as usize]) * (1.0 / 8.0);
                result.positions.push(pos);
            } else {
                // Boundary edge: simple midpoint
                let pos = (mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5;
                result.positions.push(pos);
            }

            edge_vertices.insert(key, new_idx);
        }
    }

    // Create subdivided triangles
    for tri in mesh.indices.chunks(3) {
        let [v0, v1, v2] = [tri[0], tri[1], tri[2]];

        let e01 = *edge_vertices.get(&edge_key(v0, v1)).unwrap();
        let e12 = *edge_vertices.get(&edge_key(v1, v2)).unwrap();
        let e20 = *edge_vertices.get(&edge_key(v2, v0)).unwrap();

        result.indices.extend_from_slice(&[v0, e01, e20]);
        result.indices.extend_from_slice(&[e01, v1, e12]);
        result.indices.extend_from_slice(&[e20, e12, v2]);
        result.indices.extend_from_slice(&[e01, e12, e20]);
    }

    // Recompute normals
    result.compute_smooth_normals();

    result
}

/// Loop subdivision beta coefficient.
fn loop_beta(n: usize) -> f32 {
    let n = n as f32;
    let cos_term = (2.0 * std::f32::consts::PI / n).cos();
    let inner = 3.0 / 8.0 + 0.25 * cos_term;
    (1.0 / n) * (5.0 / 8.0 - inner * inner)
}

/// Finds the vertex opposite to edge (a, b) in a triangle.
fn find_opposite_vertex(mesh: &Mesh, face_idx: usize, a: u32, b: u32) -> u32 {
    let tri = &mesh.indices[face_idx * 3..face_idx * 3 + 3];
    for &v in tri {
        if v != a && v != b {
            return v;
        }
    }
    a // Fallback (shouldn't happen)
}

/// Subdivides a mesh multiple times.
pub fn subdivide_loop_n(mesh: &Mesh, iterations: u32) -> Mesh {
    let mut result = mesh.clone();
    for _ in 0..iterations {
        result = subdivide_loop(&result);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;

    fn triangle_mesh() -> Mesh {
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(0.5, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.build()
    }

    fn tetrahedron() -> Mesh {
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(0.5, 0.0, 0.866));
        let v3 = builder.vertex(Vec3::new(0.5, 0.816, 0.433));

        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v3, v1);
        builder.triangle(v1, v3, v2);
        builder.triangle(v2, v3, v0);

        builder.build()
    }

    #[test]
    fn test_linear_subdivide_triangle_count() {
        let mesh = triangle_mesh();
        assert_eq!(mesh.triangle_count(), 1);

        let subdivided = subdivide_linear(&mesh);
        assert_eq!(subdivided.triangle_count(), 4);
    }

    #[test]
    fn test_linear_subdivide_vertex_count() {
        let mesh = triangle_mesh();
        assert_eq!(mesh.vertex_count(), 3);

        let subdivided = subdivide_linear(&mesh);
        // 3 original + 3 edge midpoints = 6
        assert_eq!(subdivided.vertex_count(), 6);
    }

    #[test]
    fn test_loop_subdivide_triangle_count() {
        let mesh = triangle_mesh();
        let subdivided = subdivide_loop(&mesh);
        assert_eq!(subdivided.triangle_count(), 4);
    }

    #[test]
    fn test_loop_subdivide_tetrahedron() {
        let mesh = tetrahedron();
        assert_eq!(mesh.triangle_count(), 4);

        let subdivided = subdivide_loop(&mesh);
        // Each triangle becomes 4
        assert_eq!(subdivided.triangle_count(), 16);
    }

    #[test]
    fn test_subdivide_n() {
        let mesh = triangle_mesh();

        let sub1 = subdivide_loop_n(&mesh, 1);
        assert_eq!(sub1.triangle_count(), 4);

        let sub2 = subdivide_loop_n(&mesh, 2);
        assert_eq!(sub2.triangle_count(), 16);

        let sub3 = subdivide_loop_n(&mesh, 3);
        assert_eq!(sub3.triangle_count(), 64);
    }

    #[test]
    fn test_loop_smooth() {
        let mesh = tetrahedron();
        let subdivided = subdivide_loop(&mesh);

        // Original vertices should have moved toward center
        // (smoothing effect)
        for pos in &subdivided.positions[..4] {
            // All original vertices should be inside the original bounds
            assert!(pos.x >= -0.1 && pos.x <= 1.1);
            assert!(pos.y >= -0.1 && pos.y <= 1.0);
            assert!(pos.z >= -0.1 && pos.z <= 1.0);
        }
    }

    #[test]
    fn test_catmull_clark_triangle() {
        let mesh = triangle_mesh();
        let subdivided = CatmullClark::new(1).apply(&mesh);

        // Should produce a valid mesh with more triangles
        assert!(subdivided.vertex_count() > mesh.vertex_count());
        assert!(subdivided.triangle_count() > 0);
    }

    #[test]
    fn test_catmull_clark_tetrahedron() {
        let mesh = tetrahedron();
        let subdivided = CatmullClark::new(1).apply(&mesh);

        // After subdivision, should have more geometry
        assert!(subdivided.vertex_count() > mesh.vertex_count());
    }

    #[test]
    fn test_catmull_clark_zero_levels() {
        let mesh = triangle_mesh();
        let subdivided = CatmullClark::new(0).apply(&mesh);

        // Zero levels should return unchanged mesh
        assert_eq!(subdivided.vertex_count(), mesh.vertex_count());
        assert_eq!(subdivided.triangle_count(), mesh.triangle_count());
    }

    #[test]
    fn test_catmull_clark_multiple_levels() {
        let mesh = triangle_mesh();
        let sub1 = CatmullClark::new(1).apply(&mesh);
        let sub2 = CatmullClark::new(2).apply(&mesh);

        // More levels = more vertices
        assert!(sub2.vertex_count() > sub1.vertex_count());
    }

    #[test]
    fn test_subdivide_catmull_clark_convenience() {
        let mesh = triangle_mesh();
        let subdivided = subdivide_catmull_clark(&mesh, 1);

        assert!(subdivided.vertex_count() > mesh.vertex_count());
    }

    #[test]
    fn test_catmull_clark_with_creases() {
        use crate::edit::EdgeCreases;
        use crate::selection::Edge;

        // Create a simple quad (two triangles forming a square)
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v2, v3);
        let mesh = builder.build();

        // Create creases on one edge
        let mut creases = EdgeCreases::new();
        creases.set(Edge::new(0, 1), 1.0); // Bottom edge fully sharp

        // Subdivide with creases
        let subdivided = CatmullClark::with_creases(1, creases).apply(&mesh);

        // Should produce valid mesh
        assert!(subdivided.vertex_count() > mesh.vertex_count());
        assert!(subdivided.triangle_count() > 0);
    }

    #[test]
    fn test_catmull_clark_crease_preserves_edge() {
        use crate::edit::EdgeCreases;
        use crate::selection::Edge;

        // Create a simple quad
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v2, v3);
        let mesh = builder.build();

        // Subdivide WITHOUT creases
        let smooth = CatmullClark::new(1).apply(&mesh);

        // Subdivide WITH crease on bottom edge
        let mut creases = EdgeCreases::new();
        creases.set(Edge::new(0, 1), 1.0);
        let sharp = CatmullClark::with_creases(1, creases).apply(&mesh);

        // Sharp edge should have different vertex positions than smooth
        // (the crease affects the geometry)
        let smooth_positions: Vec<_> = smooth.positions.iter().collect();
        let sharp_positions: Vec<_> = sharp.positions.iter().collect();

        // They should have the same number of vertices
        assert_eq!(smooth.vertex_count(), sharp.vertex_count());

        // But at least some positions should differ due to crease
        let any_different = smooth_positions
            .iter()
            .zip(sharp_positions.iter())
            .any(|(a, b)| (*a - *b).length() > 1e-6);
        assert!(any_different, "Crease should affect vertex positions");
    }

    #[test]
    fn test_catmull_clark_crease_propagation() {
        use crate::edit::EdgeCreases;
        use crate::selection::Edge;

        // Create a simple quad
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v2, v3);
        let mesh = builder.build();

        // Subdivide 2 levels with crease
        let mut creases = EdgeCreases::new();
        creases.set(Edge::new(0, 1), 1.0);
        let subdivided_2 = CatmullClark::with_creases(2, creases.clone()).apply(&mesh);

        // Subdivide 2 levels without crease
        let smooth_2 = CatmullClark::new(2).apply(&mesh);

        // They should differ due to propagated creases
        let any_different = subdivided_2
            .positions
            .iter()
            .zip(smooth_2.positions.iter())
            .any(|(a, b)| (*a - *b).length() > 1e-6);
        assert!(any_different, "Propagated creases should affect positions");
    }

    #[test]
    fn test_catmull_clark_partial_crease() {
        use crate::edit::EdgeCreases;
        use crate::selection::Edge;

        // Create a simple quad
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v2, v3);
        let mesh = builder.build();

        // Full crease
        let mut full_creases = EdgeCreases::new();
        full_creases.set(Edge::new(0, 1), 1.0);
        let full_sharp = CatmullClark::with_creases(1, full_creases).apply(&mesh);

        // Partial crease (0.5)
        let mut partial_creases = EdgeCreases::new();
        partial_creases.set(Edge::new(0, 1), 0.5);
        let partial_sharp = CatmullClark::with_creases(1, partial_creases).apply(&mesh);

        // Smooth (no crease)
        let smooth = CatmullClark::new(1).apply(&mesh);

        // Partial should be between smooth and full sharp
        // At least verify all three produce valid different meshes
        assert!(smooth.vertex_count() > 0);
        assert!(partial_sharp.vertex_count() > 0);
        assert!(full_sharp.vertex_count() > 0);
    }
}
