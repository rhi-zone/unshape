//! Mesh decimation via edge collapse.
//!
//! Reduces triangle count while preserving mesh shape.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_mesh::{sphere, Decimate};
//!
//! let high_poly = sphere(32, 16);
//! let low_poly = Decimate::target_triangles(100).apply(&high_poly);
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Decimates a mesh using edge collapse.
///
/// Uses a greedy approach that collapses the shortest edges first,
/// reducing triangle count while preserving mesh shape.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct Decimate {
    /// Target number of triangles. Decimation stops when reached.
    pub target_triangles: Option<usize>,
    /// Target reduction ratio (0.0 - 1.0). 0.5 = reduce to half the triangles.
    pub target_ratio: Option<f32>,
    /// Maximum error threshold. Edges with higher error won't be collapsed.
    pub max_error: f32,
    /// Whether to preserve boundary edges (edges with only one face).
    pub preserve_boundary: bool,
}

impl Default for Decimate {
    fn default() -> Self {
        Self {
            target_triangles: None,
            target_ratio: Some(0.5),
            max_error: f32::MAX,
            preserve_boundary: true,
        }
    }
}

impl Decimate {
    /// Creates a decimation targeting a specific triangle count.
    pub fn target_triangles(count: usize) -> Self {
        Self {
            target_triangles: Some(count),
            target_ratio: None,
            ..Default::default()
        }
    }

    /// Creates a decimation targeting a reduction ratio (0.5 = half the triangles).
    pub fn target_ratio(ratio: f32) -> Self {
        Self {
            target_ratio: Some(ratio.clamp(0.0, 1.0)),
            target_triangles: None,
            ..Default::default()
        }
    }

    /// Applies this decimation operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        decimate(mesh, self.clone())
    }
}

/// Backwards-compatible type alias.
pub type DecimateConfig = Decimate;

/// Decimates a mesh using edge collapse.
///
/// Uses a greedy approach that collapses the shortest edges first.
pub fn decimate(mesh: &Mesh, config: DecimateConfig) -> Mesh {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return mesh.clone();
    }

    // Determine target triangle count
    let current_triangles = mesh.triangle_count();
    let target = if let Some(t) = config.target_triangles {
        t
    } else if let Some(r) = config.target_ratio {
        ((current_triangles as f32) * r) as usize
    } else {
        current_triangles / 2
    };

    if current_triangles <= target {
        return mesh.clone();
    }

    // Build internal representation
    let mut state = DecimationState::from_mesh(mesh, &config);

    // Perform decimation
    while state.triangle_count() > target {
        if !state.collapse_best_edge() {
            break;
        }
    }

    // Convert back to mesh
    state.to_mesh()
}

/// Internal state for decimation algorithm.
struct DecimationState {
    /// Vertex positions.
    positions: Vec<Vec3>,
    /// Vertex normals (if present).
    normals: Vec<Vec3>,
    /// UV coordinates (if present).
    uvs: Vec<glam::Vec2>,
    /// Triangles as [v0, v1, v2] indices.
    triangles: Vec<[usize; 3]>,
    /// For each vertex, which vertex it has been merged into (or itself).
    vertex_map: Vec<usize>,
    /// Set of removed triangles.
    removed_triangles: HashSet<usize>,
    /// Priority queue of edges to collapse (by length).
    edge_queue: BinaryHeap<EdgeEntry>,
    /// Set of boundary vertices.
    boundary_vertices: HashSet<usize>,
    /// Config.
    config: DecimateConfig,
}

/// Entry in the edge priority queue.
#[derive(Clone)]
struct EdgeEntry {
    /// Edge vertices (always v0 < v1).
    edge: (usize, usize),
    /// Negative cost (for max-heap to act as min-heap).
    neg_cost: std::cmp::Reverse<ordered_float::OrderedFloat<f32>>,
}

impl PartialEq for EdgeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.edge == other.edge
    }
}

impl Eq for EdgeEntry {}

impl PartialOrd for EdgeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_cost.cmp(&other.neg_cost)
    }
}

impl DecimationState {
    fn from_mesh(mesh: &Mesh, config: &DecimateConfig) -> Self {
        let positions = mesh.positions.clone();
        let normals = mesh.normals.clone();
        let uvs = mesh.uvs.clone();

        // Convert indices to triangles
        let triangles: Vec<[usize; 3]> = mesh
            .indices
            .chunks(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();

        // Initialize vertex map (each vertex maps to itself)
        let vertex_map: Vec<usize> = (0..positions.len()).collect();

        // Find boundary vertices
        let boundary_vertices = if config.preserve_boundary {
            find_boundary_vertices(&triangles, positions.len())
        } else {
            HashSet::new()
        };

        // Build edge priority queue
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for tri in &triangles {
            for i in 0..3 {
                let v0 = tri[i];
                let v1 = tri[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_set.insert(edge);
            }
        }

        let edge_queue: BinaryHeap<EdgeEntry> = edge_set
            .into_iter()
            .filter_map(|(v0, v1)| {
                // Skip boundary edges if preserving boundaries
                if config.preserve_boundary
                    && (boundary_vertices.contains(&v0) || boundary_vertices.contains(&v1))
                {
                    return None;
                }

                let cost = positions[v0].distance(positions[v1]);
                if cost > config.max_error {
                    return None;
                }

                Some(EdgeEntry {
                    edge: (v0, v1),
                    neg_cost: std::cmp::Reverse(ordered_float::OrderedFloat(cost)),
                })
            })
            .collect();

        Self {
            positions,
            normals,
            uvs,
            triangles,
            vertex_map,
            removed_triangles: HashSet::new(),
            edge_queue,
            boundary_vertices,
            config: config.clone(),
        }
    }

    /// Gets the current (canonical) vertex index, following merges.
    fn canonical_vertex(&self, mut v: usize) -> usize {
        while self.vertex_map[v] != v {
            v = self.vertex_map[v];
        }
        v
    }

    /// Returns current triangle count (excluding removed).
    fn triangle_count(&self) -> usize {
        self.triangles.len() - self.removed_triangles.len()
    }

    /// Collapses the best (lowest cost) edge. Returns false if no more edges can be collapsed.
    fn collapse_best_edge(&mut self) -> bool {
        while let Some(entry) = self.edge_queue.pop() {
            let (v0, v1) = entry.edge;

            // Skip if vertices have been merged
            let cv0 = self.canonical_vertex(v0);
            let cv1 = self.canonical_vertex(v1);

            if cv0 == cv1 {
                continue;
            }

            // Skip boundary vertices if configured
            if self.config.preserve_boundary
                && (self.boundary_vertices.contains(&cv0) || self.boundary_vertices.contains(&cv1))
            {
                continue;
            }

            // Collapse edge: merge v1 into v0
            self.collapse_edge(cv0, cv1);
            return true;
        }
        false
    }

    /// Collapses an edge by merging v1 into v0.
    fn collapse_edge(&mut self, v0: usize, v1: usize) {
        // Update vertex map
        self.vertex_map[v1] = v0;

        // Move v0 to midpoint
        let mid = (self.positions[v0] + self.positions[v1]) * 0.5;
        self.positions[v0] = mid;

        // Average normals if present
        if v0 < self.normals.len() && v1 < self.normals.len() {
            let avg_normal = (self.normals[v0] + self.normals[v1]).normalize_or_zero();
            self.normals[v0] = avg_normal;
        }

        // Average UVs if present
        if v0 < self.uvs.len() && v1 < self.uvs.len() {
            let avg_uv = (self.uvs[v0] + self.uvs[v1]) * 0.5;
            self.uvs[v0] = avg_uv;
        }

        // Update triangles and remove degenerate ones
        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        let mut triangles_to_remove: Vec<usize> = Vec::new();

        // First pass: update vertices and collect changes
        for tri_idx in 0..self.triangles.len() {
            if self.removed_triangles.contains(&tri_idx) {
                continue;
            }

            let tri = &mut self.triangles[tri_idx];

            // Update vertex references - use vertex_map directly
            for v in tri.iter_mut() {
                // Follow the chain to get canonical vertex
                let mut current = *v;
                while self.vertex_map[current] != current {
                    current = self.vertex_map[current];
                }
                *v = current;
            }

            // Check for degenerate triangle (has duplicate vertices)
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
                triangles_to_remove.push(tri_idx);
            } else {
                // Collect new edges for this triangle
                for i in 0..3 {
                    let a = tri[i];
                    let b = tri[(i + 1) % 3];
                    let edge = if a < b { (a, b) } else { (b, a) };
                    new_edges.push(edge);
                }
            }
        }

        // Mark degenerate triangles as removed
        for tri_idx in triangles_to_remove {
            self.removed_triangles.insert(tri_idx);
        }

        // Add new edges to queue (may have changed costs)
        for (a, b) in new_edges {
            if self.config.preserve_boundary
                && (self.boundary_vertices.contains(&a) || self.boundary_vertices.contains(&b))
            {
                continue;
            }

            let cost = self.positions[a].distance(self.positions[b]);
            if cost <= self.config.max_error {
                self.edge_queue.push(EdgeEntry {
                    edge: (a, b),
                    neg_cost: std::cmp::Reverse(ordered_float::OrderedFloat(cost)),
                });
            }
        }
    }

    /// Converts state back to a Mesh.
    fn to_mesh(&self) -> Mesh {
        // Build vertex remapping (old canonical -> new sequential)
        let mut used_vertices: HashSet<usize> = HashSet::new();
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            if self.removed_triangles.contains(&tri_idx) {
                continue;
            }
            for &v in tri {
                used_vertices.insert(self.canonical_vertex(v));
            }
        }

        let mut vertex_remap: HashMap<usize, u32> = HashMap::new();
        let mut new_positions: Vec<Vec3> = Vec::new();
        let mut new_normals: Vec<Vec3> = Vec::new();
        let mut new_uvs: Vec<glam::Vec2> = Vec::new();

        for &v in &used_vertices {
            vertex_remap.insert(v, new_positions.len() as u32);
            new_positions.push(self.positions[v]);
            if v < self.normals.len() {
                new_normals.push(self.normals[v]);
            }
            if v < self.uvs.len() {
                new_uvs.push(self.uvs[v]);
            }
        }

        // Build new indices
        let mut new_indices: Vec<u32> = Vec::new();
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            if self.removed_triangles.contains(&tri_idx) {
                continue;
            }

            for &v in tri {
                let cv = self.canonical_vertex(v);
                new_indices.push(vertex_remap[&cv]);
            }
        }

        let mut mesh = Mesh::new();
        mesh.positions = new_positions;
        mesh.indices = new_indices;

        if new_normals.len() == mesh.positions.len() {
            mesh.normals = new_normals;
        } else {
            mesh.compute_smooth_normals();
        }

        if new_uvs.len() == mesh.positions.len() {
            mesh.uvs = new_uvs;
        }

        mesh
    }
}

/// Finds vertices that are on boundary edges (edges belonging to only one triangle).
fn find_boundary_vertices(triangles: &[[usize; 3]], vertex_count: usize) -> HashSet<usize> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for tri in triangles {
        for i in 0..3 {
            let v0 = tri[i];
            let v1 = tri[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    let mut boundary = HashSet::new();
    for ((v0, v1), count) in edge_count {
        if count == 1 {
            boundary.insert(v0);
            boundary.insert(v1);
        }
    }

    // Unused parameter but kept for future extensions
    let _ = vertex_count;

    boundary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{box_mesh, uv_sphere};

    #[test]
    fn test_decimate_basic() {
        let mesh = uv_sphere(16, 8);
        let original_tris = mesh.triangle_count();

        let mut config = DecimateConfig::target_ratio(0.5);
        config.preserve_boundary = false;
        let decimated = decimate(&mesh, config);
        let new_tris = decimated.triangle_count();

        // Should have significantly fewer triangles
        assert!(new_tris < original_tris);
        // Should have at least some triangles
        assert!(new_tris > 0);
    }

    #[test]
    fn test_decimate_target_triangles() {
        let mesh = uv_sphere(16, 8);
        let original = mesh.triangle_count();

        // Don't preserve boundary so we can actually decimate
        let mut config = DecimateConfig::target_triangles(50);
        config.preserve_boundary = false;
        let decimated = decimate(&mesh, config);

        // Should have significantly fewer triangles than original
        assert!(
            decimated.triangle_count() < original,
            "Should reduce triangles"
        );
    }

    #[test]
    fn test_decimate_preserves_normals() {
        let mesh = uv_sphere(16, 8);
        assert!(mesh.has_normals());

        let decimated = decimate(&mesh, DecimateConfig::target_ratio(0.3));

        // Should still have normals
        assert!(decimated.has_normals());
        assert_eq!(decimated.normals.len(), decimated.positions.len());
    }

    #[test]
    fn test_decimate_empty_mesh() {
        let mesh = Mesh::new();
        let decimated = decimate(&mesh, DecimateConfig::target_ratio(0.5));

        assert_eq!(decimated.positions.len(), 0);
    }

    #[test]
    fn test_decimate_already_small() {
        // Box mesh has only 12 triangles
        let mesh = box_mesh();
        let original_tris = mesh.triangle_count();

        // Try to decimate to more triangles than we have
        let decimated = decimate(&mesh, DecimateConfig::target_triangles(100));

        // Should keep all triangles
        assert_eq!(decimated.triangle_count(), original_tris);
    }

    #[test]
    fn test_decimate_config_max_error() {
        let mesh = uv_sphere(16, 8);

        // Very small max_error should prevent most collapses
        let mut config = DecimateConfig::target_ratio(0.1);
        config.max_error = 0.001;
        let decimated = decimate(&mesh, config);

        // Should have more triangles than with default max_error
        let config2 = DecimateConfig::target_ratio(0.1);
        let decimated2 = decimate(&mesh, config2);

        assert!(decimated.triangle_count() >= decimated2.triangle_count());
    }

    #[test]
    fn test_decimate_valid_mesh() {
        let mesh = uv_sphere(16, 8);
        let decimated = decimate(&mesh, DecimateConfig::target_ratio(0.3));

        // All indices should be valid
        for &idx in &decimated.indices {
            assert!((idx as usize) < decimated.positions.len());
        }

        // Triangle count should match indices
        assert_eq!(decimated.triangle_count(), decimated.indices.len() / 3);
    }
}
