//! Mesh remeshing and retopology.
//!
//! Provides algorithms for improving mesh topology, including
//! isotropic remeshing and basic quad conversion.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```ignore
//! use unshape_mesh::{Cuboid, Remesh};
//!
//! let mesh = Cuboid::default().apply();
//! let remeshed = Remesh::default().apply(&mesh);
//! ```

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performs isotropic remeshing to achieve uniform edge lengths.
///
/// Uses edge splitting/collapsing and vertex smoothing to create
/// a more uniform mesh topology.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct Remesh {
    /// Target edge length.
    pub target_edge_length: f32,
    /// Number of iterations.
    pub iterations: u32,
    /// Smoothing factor (0-1).
    pub smoothing: f32,
    /// Whether to preserve boundary edges.
    pub preserve_boundary: bool,
}

impl Default for Remesh {
    fn default() -> Self {
        Self {
            target_edge_length: 0.1,
            iterations: 5,
            smoothing: 0.5,
            preserve_boundary: true,
        }
    }
}

impl Remesh {
    /// Applies this remeshing operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        isotropic_remesh(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type RemeshConfig = Remesh;

/// Performs isotropic remeshing to achieve uniform edge lengths.
///
/// This is a simplified implementation that uses edge splitting/collapsing
/// and vertex smoothing to create a more uniform mesh.
pub fn isotropic_remesh(mesh: &Mesh, config: &Remesh) -> Mesh {
    let mut result = mesh.clone();

    let low = config.target_edge_length * 0.8;
    let high = config.target_edge_length * 1.2;

    for _ in 0..config.iterations {
        // Split long edges
        result = split_long_edges(&result, high);

        // Collapse short edges
        result = collapse_short_edges(&result, low, config.preserve_boundary);

        // Smooth vertices
        if config.smoothing > 0.0 {
            result = smooth_vertices(&result, config.smoothing, config.preserve_boundary);
        }
    }

    result
}

/// Splits edges longer than max_length.
fn split_long_edges(mesh: &Mesh, max_length: f32) -> Mesh {
    let mut positions = mesh.positions.clone();
    let mut indices: Vec<u32> = Vec::new();

    // Build edge midpoint map
    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i];
        let i1 = mesh.indices[i + 1];
        let i2 = mesh.indices[i + 2];

        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        let mut split_verts = [i0, i1, i2];

        // Check each edge for splitting
        for (idx, &(a, b)) in edges.iter().enumerate() {
            let edge_key = if a < b { (a, b) } else { (b, a) };
            let v0 = positions[a as usize];
            let v1 = positions[b as usize];
            let length = (v1 - v0).length();

            if length > max_length {
                let mid_idx = *edge_midpoints.entry(edge_key).or_insert_with(|| {
                    let mid = (v0 + v1) * 0.5;
                    let new_idx = positions.len() as u32;
                    positions.push(mid);
                    new_idx
                });

                // Record split vertex
                split_verts[idx] = mid_idx;
            }
        }

        // Output triangles (may need to subdivide)
        if split_verts[0] != i0 || split_verts[1] != i1 || split_verts[2] != i2 {
            // At least one edge was split - create new triangles
            // This is a simplified approach that handles single edge splits
            let splits: Vec<usize> = (0..3)
                .filter(|&idx| split_verts[idx] != [i0, i1, i2][idx])
                .collect();

            if splits.len() == 1 {
                // Single edge split - create 2 triangles
                let split_idx = splits[0];
                let mid = split_verts[split_idx];
                let orig = [i0, i1, i2];
                let a = orig[split_idx];
                let b = orig[(split_idx + 1) % 3];
                let c = orig[(split_idx + 2) % 3];

                indices.extend_from_slice(&[a, mid, c]);
                indices.extend_from_slice(&[mid, b, c]);
            } else {
                // Multiple splits - just output original for now
                indices.extend_from_slice(&[i0, i1, i2]);
            }
        } else {
            indices.extend_from_slice(&[i0, i1, i2]);
        }
    }

    let mut result = Mesh::new();
    result.positions = positions;
    result.indices = indices;
    result
}

/// Collapses edges shorter than min_length.
fn collapse_short_edges(mesh: &Mesh, min_length: f32, preserve_boundary: bool) -> Mesh {
    let mut positions = mesh.positions.clone();
    let mut vertex_map: Vec<u32> = (0..positions.len() as u32).collect();

    // Find boundary vertices if needed
    let boundary_vertices = if preserve_boundary {
        find_boundary_vertices(mesh)
    } else {
        std::collections::HashSet::new()
    };

    // Build edge list and check for collapse
    let mut edges_to_collapse: Vec<(u32, u32)> = Vec::new();

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i];
        let i1 = mesh.indices[i + 1];
        let i2 = mesh.indices[i + 2];

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let v0 = positions[a as usize];
            let v1 = positions[b as usize];
            let length = (v1 - v0).length();

            if length < min_length {
                let a_boundary = boundary_vertices.contains(&a);
                let b_boundary = boundary_vertices.contains(&b);

                if !preserve_boundary || (!a_boundary && !b_boundary) {
                    edges_to_collapse.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
    }

    // Remove duplicates
    edges_to_collapse.sort();
    edges_to_collapse.dedup();

    // Collapse edges (map higher index to lower)
    for (a, b) in edges_to_collapse {
        let target = vertex_map[a as usize];
        let source = vertex_map[b as usize];

        // Update all vertices pointing to source
        for v in vertex_map.iter_mut() {
            if *v == source {
                *v = target;
            }
        }

        // Move target to midpoint
        let mid = (positions[a as usize] + positions[b as usize]) * 0.5;
        positions[target as usize] = mid;
    }

    // Rebuild mesh with collapsed vertices
    let mut new_positions: Vec<Vec3> = Vec::new();
    let mut new_indices: Vec<u32> = Vec::new();
    let mut vertex_remap: HashMap<u32, u32> = HashMap::new();

    for i in (0..mesh.indices.len()).step_by(3) {
        let mut tri = [
            vertex_map[mesh.indices[i] as usize],
            vertex_map[mesh.indices[i + 1] as usize],
            vertex_map[mesh.indices[i + 2] as usize],
        ];

        // Skip degenerate triangles
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
            continue;
        }

        for v in &mut tri {
            if !vertex_remap.contains_key(v) {
                let new_idx = new_positions.len() as u32;
                new_positions.push(positions[*v as usize]);
                vertex_remap.insert(*v, new_idx);
            }
            *v = vertex_remap[v];
        }

        new_indices.extend_from_slice(&tri);
    }

    let mut result = Mesh::new();
    result.positions = new_positions;
    result.indices = new_indices;
    result
}

/// Smooths vertex positions.
fn smooth_vertices(mesh: &Mesh, factor: f32, preserve_boundary: bool) -> Mesh {
    let positions = mesh.positions.clone();

    // Find boundary vertices
    let boundary_vertices = if preserve_boundary {
        find_boundary_vertices(mesh)
    } else {
        std::collections::HashSet::new()
    };

    // Build adjacency
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i] as usize;
        let i1 = mesh.indices[i + 1] as usize;
        let i2 = mesh.indices[i + 2] as usize;

        adjacency[i0].push(i1);
        adjacency[i0].push(i2);
        adjacency[i1].push(i0);
        adjacency[i1].push(i2);
        adjacency[i2].push(i0);
        adjacency[i2].push(i1);
    }

    // Remove duplicates
    for adj in &mut adjacency {
        adj.sort();
        adj.dedup();
    }

    // Smooth
    let mut new_positions = positions.clone();

    for (i, adj) in adjacency.iter().enumerate() {
        if preserve_boundary && boundary_vertices.contains(&(i as u32)) {
            continue;
        }

        if adj.is_empty() {
            continue;
        }

        let avg: Vec3 = adj.iter().map(|&j| positions[j]).sum::<Vec3>() / adj.len() as f32;
        new_positions[i] = positions[i].lerp(avg, factor);
    }

    let mut result = mesh.clone();
    result.positions = new_positions;
    result
}

/// Finds vertices on mesh boundary.
fn find_boundary_vertices(mesh: &Mesh) -> std::collections::HashSet<u32> {
    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i];
        let i1 = mesh.indices[i + 1];
        let i2 = mesh.indices[i + 2];

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    let mut boundary = std::collections::HashSet::new();

    for ((a, b), count) in edge_count {
        if count == 1 {
            boundary.insert(a);
            boundary.insert(b);
        }
    }

    boundary
}

/// Converts triangle pairs to quads where possible.
///
/// Pairs adjacent triangles with similar normals to form quad faces.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = QuadMesh))]
pub struct Quadify {
    /// Maximum angle difference for merging triangles (degrees).
    pub max_angle: f32,
    /// Whether to preserve sharp edges.
    pub preserve_sharp: bool,
    /// Sharp edge angle threshold (degrees).
    pub sharp_angle: f32,
}

impl Default for Quadify {
    fn default() -> Self {
        Self {
            max_angle: 15.0,
            preserve_sharp: true,
            sharp_angle: 30.0,
        }
    }
}

impl Quadify {
    /// Applies this quadification operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> QuadMesh {
        quadify(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type QuadifyConfig = Quadify;

/// Converts triangle pairs to quads where possible.
///
/// Returns indices as quads (4 indices per face) for faces that were converted,
/// and triangles for those that weren't.
pub fn quadify(mesh: &Mesh, config: &Quadify) -> QuadMesh {
    let mut result = QuadMesh::new();
    result.positions = mesh.positions.clone();

    let max_cos = (config.max_angle.to_radians()).cos();
    let sharp_cos = (config.sharp_angle.to_radians()).cos();

    // Build adjacency: edge -> triangle indices
    let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for tri_idx in 0..mesh.triangle_count() {
        let base = tri_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_tris.entry(edge).or_default().push(tri_idx);
        }
    }

    // Compute triangle normals
    let normals: Vec<Vec3> = (0..mesh.triangle_count())
        .map(|i| {
            let base = i * 3;
            let v0 = mesh.positions[mesh.indices[base] as usize];
            let v1 = mesh.positions[mesh.indices[base + 1] as usize];
            let v2 = mesh.positions[mesh.indices[base + 2] as usize];
            (v1 - v0).cross(v2 - v0).normalize_or_zero()
        })
        .collect();

    let mut used = vec![false; mesh.triangle_count()];

    // Try to pair triangles into quads
    for (edge, tris) in &edge_tris {
        if tris.len() != 2 {
            continue;
        }

        let t0 = tris[0];
        let t1 = tris[1];

        if used[t0] || used[t1] {
            continue;
        }

        // Check if edge is sharp
        let dot = normals[t0].dot(normals[t1]);
        if config.preserve_sharp && dot < sharp_cos {
            continue;
        }

        // Check if normals are similar enough
        if dot < max_cos {
            continue;
        }

        // Find the quad vertices
        let base0 = t0 * 3;
        let base1 = t1 * 3;

        let tri0 = [
            mesh.indices[base0],
            mesh.indices[base0 + 1],
            mesh.indices[base0 + 2],
        ];
        let tri1 = [
            mesh.indices[base1],
            mesh.indices[base1 + 1],
            mesh.indices[base1 + 2],
        ];

        // Find the non-shared vertices
        let (a, b) = *edge;
        let v0 = tri0.iter().find(|&&v| v != a && v != b).copied();
        let v1 = tri1.iter().find(|&&v| v != a && v != b).copied();

        if let (Some(v0), Some(v1)) = (v0, v1) {
            // Order quad vertices: v0, a, v1, b (or similar ordering)
            result.quads.extend_from_slice(&[v0, a, v1, b]);
            used[t0] = true;
            used[t1] = true;
        }
    }

    // Add remaining triangles
    for i in 0..mesh.triangle_count() {
        if !used[i] {
            let base = i * 3;
            result.triangles.extend_from_slice(&[
                mesh.indices[base],
                mesh.indices[base + 1],
                mesh.indices[base + 2],
            ]);
        }
    }

    result
}

/// A mesh with both quads and triangles.
#[derive(Debug, Clone)]
pub struct QuadMesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Quad indices (4 per face).
    pub quads: Vec<u32>,
    /// Triangle indices (3 per face).
    pub triangles: Vec<u32>,
}

impl QuadMesh {
    /// Creates an empty quad mesh.
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            quads: Vec::new(),
            triangles: Vec::new(),
        }
    }

    /// Returns the number of quad faces.
    pub fn quad_count(&self) -> usize {
        self.quads.len() / 4
    }

    /// Returns the number of triangle faces.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    /// Converts to a pure triangle mesh.
    pub fn to_triangles(&self) -> Mesh {
        let mut mesh = Mesh::new();
        mesh.positions = self.positions.clone();

        // Convert quads to triangles
        for i in (0..self.quads.len()).step_by(4) {
            let q0 = self.quads[i];
            let q1 = self.quads[i + 1];
            let q2 = self.quads[i + 2];
            let q3 = self.quads[i + 3];

            mesh.indices.extend_from_slice(&[q0, q1, q2]);
            mesh.indices.extend_from_slice(&[q0, q2, q3]);
        }

        // Add existing triangles
        mesh.indices.extend_from_slice(&self.triangles);

        mesh
    }
}

impl Default for QuadMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimates average edge length in a mesh.
pub fn average_edge_length(mesh: &Mesh) -> f32 {
    if mesh.indices.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for i in (0..mesh.indices.len()).step_by(3) {
        let v0 = mesh.positions[mesh.indices[i] as usize];
        let v1 = mesh.positions[mesh.indices[i + 1] as usize];
        let v2 = mesh.positions[mesh.indices[i + 2] as usize];

        total += (v1 - v0).length();
        total += (v2 - v1).length();
        total += (v0 - v2).length();
        count += 3;
    }

    total / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_remesh_config_default() {
        let config = RemeshConfig::default();
        assert!(config.target_edge_length > 0.0);
        assert!(config.iterations > 0);
    }

    #[test]
    fn test_isotropic_remesh() {
        let mesh = Cuboid::default().apply();
        let config = RemeshConfig {
            target_edge_length: 0.3,
            iterations: 2,
            smoothing: 0.3,
            preserve_boundary: false,
        };

        let remeshed = isotropic_remesh(&mesh, &config);

        // Should still have valid triangles
        assert!(!remeshed.positions.is_empty());
        assert!(!remeshed.indices.is_empty());
        assert_eq!(remeshed.indices.len() % 3, 0);
    }

    #[test]
    fn test_average_edge_length() {
        let mesh = Cuboid::default().apply();
        let avg = average_edge_length(&mesh);

        // Box mesh edges should be around 1.0 (unit cube)
        assert!(avg > 0.5 && avg < 2.0);
    }

    #[test]
    fn test_find_boundary_vertices() {
        // Create a simple open mesh (not closed)
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.indices = vec![0, 1, 2];

        let boundary = find_boundary_vertices(&mesh);

        // All vertices should be boundary (single triangle)
        assert_eq!(boundary.len(), 3);
    }

    #[test]
    fn test_quadify_config_default() {
        let config = QuadifyConfig::default();
        assert!(config.max_angle > 0.0);
        assert!(config.sharp_angle > 0.0);
    }

    #[test]
    fn test_quadify() {
        let mesh = Cuboid::default().apply();
        let config = QuadifyConfig::default();

        let quad_mesh = quadify(&mesh, &config);

        // Should have some quads (box faces can be paired)
        // May not convert all due to angle constraints
        assert!(!quad_mesh.positions.is_empty());
    }

    #[test]
    fn test_quad_mesh_to_triangles() {
        let mut qm = QuadMesh::new();
        qm.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        qm.quads = vec![0, 1, 2, 3];

        let mesh = qm.to_triangles();

        assert_eq!(mesh.indices.len(), 6); // 2 triangles
    }

    #[test]
    fn test_smooth_vertices() {
        let mesh = Cuboid::default().apply();
        let smoothed = smooth_vertices(&mesh, 0.5, false);

        // Positions should be different
        let orig_center: Vec3 = mesh.positions.iter().sum::<Vec3>() / mesh.positions.len() as f32;
        let smooth_center: Vec3 =
            smoothed.positions.iter().sum::<Vec3>() / smoothed.positions.len() as f32;

        // Center should be roughly preserved
        assert!((orig_center - smooth_center).length() < 0.1);
    }
}
