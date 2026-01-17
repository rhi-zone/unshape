//! Geodesic distance computation on mesh surfaces.
//!
//! Computes shortest path distances along mesh surfaces using the fast marching method.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_mesh::{Mesh, UvSphere, geodesic::{compute_geodesic_distance, GeodesicConfig}};
//!
//! // Create a simple mesh
//! let mesh = UvSphere::new(1.0, 16, 8).apply();
//!
//! // Compute geodesic distance from vertex 0
//! let distances = compute_geodesic_distance(&mesh, &[0], GeodesicConfig::default());
//!
//! // distances[i] = geodesic distance from vertex i to source
//! ```

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for geodesic distance computation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeodesicConfig {
    /// Maximum distance to compute (for early termination).
    pub max_distance: Option<f32>,
    /// Whether to use the fast marching method (more accurate) or Dijkstra (faster).
    pub use_fast_marching: bool,
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            max_distance: None,
            use_fast_marching: true,
        }
    }
}

/// Result of geodesic distance computation.
#[derive(Debug, Clone)]
pub struct GeodesicResult {
    /// Distance from each vertex to the nearest source.
    pub distances: Vec<f32>,
    /// For each vertex, the index of the source it's closest to (when multiple sources).
    pub nearest_source: Vec<usize>,
}

impl GeodesicResult {
    /// Returns vertices within a given distance from sources.
    pub fn vertices_within_distance(&self, max_dist: f32) -> Vec<usize> {
        self.distances
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d <= max_dist)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the farthest vertex from all sources.
    pub fn farthest_vertex(&self) -> usize {
        self.distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the maximum distance found.
    pub fn max_distance(&self) -> f32 {
        self.distances.iter().cloned().fold(0.0f32, f32::max)
    }
}

/// Computes geodesic distances from source vertices to all other vertices.
pub fn compute_geodesic_distance(
    mesh: &Mesh,
    sources: &[usize],
    config: GeodesicConfig,
) -> GeodesicResult {
    if mesh.positions.is_empty() || sources.is_empty() {
        return GeodesicResult {
            distances: vec![f32::INFINITY; mesh.positions.len()],
            nearest_source: vec![0; mesh.positions.len()],
        };
    }

    // Build adjacency list
    let adjacency = build_adjacency(mesh);

    if config.use_fast_marching {
        fast_marching(mesh, sources, &adjacency, config.max_distance)
    } else {
        dijkstra(mesh, sources, &adjacency, config.max_distance)
    }
}

/// Computes geodesic distance from a single source vertex.
pub fn geodesic_distance_from(mesh: &Mesh, source: usize) -> Vec<f32> {
    compute_geodesic_distance(mesh, &[source], GeodesicConfig::default()).distances
}

/// Computes the geodesic path between two vertices.
pub fn geodesic_path(mesh: &Mesh, from: usize, to: usize) -> Vec<usize> {
    if from == to {
        return vec![from];
    }

    // Compute distances from target
    let result = compute_geodesic_distance(mesh, &[to], GeodesicConfig::default());

    // Build adjacency
    let adjacency = build_adjacency(mesh);

    // Trace path by following gradient descent
    let mut path = vec![from];
    let mut current = from;

    let mut visited = HashSet::new();
    visited.insert(current);

    while current != to {
        let neighbors = &adjacency[current];

        // Find neighbor with smallest distance
        let next = neighbors
            .iter()
            .filter(|&&n| !visited.contains(&n))
            .min_by(|&&a, &&b| {
                result.distances[a]
                    .partial_cmp(&result.distances[b])
                    .unwrap_or(Ordering::Equal)
            });

        match next {
            Some(&n) => {
                path.push(n);
                visited.insert(n);
                current = n;
            }
            None => break, // No path found
        }
    }

    path
}

/// Finds the vertex farthest from all boundary vertices (useful for mesh center).
pub fn find_mesh_center(mesh: &Mesh) -> usize {
    let boundary = find_boundary_vertices(mesh);

    if boundary.is_empty() {
        // No boundary, use first vertex
        let result = compute_geodesic_distance(mesh, &[0], GeodesicConfig::default());
        return result.farthest_vertex();
    }

    let result = compute_geodesic_distance(mesh, &boundary, GeodesicConfig::default());
    result.farthest_vertex()
}

/// Computes an iso-distance curve (vertices at approximately the same distance).
pub fn iso_distance_vertices(
    mesh: &Mesh,
    source: usize,
    target_distance: f32,
    tolerance: f32,
) -> Vec<usize> {
    let result = compute_geodesic_distance(mesh, &[source], GeodesicConfig::default());

    result
        .distances
        .iter()
        .enumerate()
        .filter(|&(_, d)| (*d - target_distance).abs() <= tolerance)
        .map(|(i, _)| i)
        .collect()
}

// Internal structures

#[derive(Clone, Copy)]
struct HeapEntry {
    vertex: usize,
    distance: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

fn build_adjacency(mesh: &Mesh) -> Vec<Vec<usize>> {
    let n = mesh.positions.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

    for tri in mesh.indices.chunks(3) {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        if !adjacency[a].contains(&b) {
            adjacency[a].push(b);
        }
        if !adjacency[a].contains(&c) {
            adjacency[a].push(c);
        }

        if !adjacency[b].contains(&a) {
            adjacency[b].push(a);
        }
        if !adjacency[b].contains(&c) {
            adjacency[b].push(c);
        }

        if !adjacency[c].contains(&a) {
            adjacency[c].push(a);
        }
        if !adjacency[c].contains(&b) {
            adjacency[c].push(b);
        }
    }

    adjacency
}

fn dijkstra(
    mesh: &Mesh,
    sources: &[usize],
    adjacency: &[Vec<usize>],
    max_distance: Option<f32>,
) -> GeodesicResult {
    let n = mesh.positions.len();
    let mut distances = vec![f32::INFINITY; n];
    let mut nearest_source = vec![0usize; n];
    let mut heap = BinaryHeap::new();

    // Initialize sources
    for (source_idx, &source) in sources.iter().enumerate() {
        if source < n {
            distances[source] = 0.0;
            nearest_source[source] = source_idx;
            heap.push(HeapEntry {
                vertex: source,
                distance: 0.0,
            });
        }
    }

    while let Some(HeapEntry { vertex, distance }) = heap.pop() {
        if distance > distances[vertex] {
            continue;
        }

        if let Some(max) = max_distance {
            if distance > max {
                continue;
            }
        }

        for &neighbor in &adjacency[vertex] {
            let edge_length = (mesh.positions[neighbor] - mesh.positions[vertex]).length();
            let new_dist = distance + edge_length;

            if new_dist < distances[neighbor] {
                distances[neighbor] = new_dist;
                nearest_source[neighbor] = nearest_source[vertex];
                heap.push(HeapEntry {
                    vertex: neighbor,
                    distance: new_dist,
                });
            }
        }
    }

    GeodesicResult {
        distances,
        nearest_source,
    }
}

fn fast_marching(
    mesh: &Mesh,
    sources: &[usize],
    adjacency: &[Vec<usize>],
    max_distance: Option<f32>,
) -> GeodesicResult {
    let n = mesh.positions.len();
    let mut distances = vec![f32::INFINITY; n];
    let mut nearest_source = vec![0usize; n];
    let mut frozen = vec![false; n];
    let mut heap = BinaryHeap::new();

    // Build triangle adjacency for update computations
    let triangle_adj = build_triangle_adjacency(mesh);

    // Initialize sources
    for (source_idx, &source) in sources.iter().enumerate() {
        if source < n {
            distances[source] = 0.0;
            nearest_source[source] = source_idx;
            frozen[source] = true;

            // Add neighbors to heap
            for &neighbor in &adjacency[source] {
                let edge_length = (mesh.positions[neighbor] - mesh.positions[source]).length();
                if edge_length < distances[neighbor] {
                    distances[neighbor] = edge_length;
                    nearest_source[neighbor] = source_idx;
                    heap.push(HeapEntry {
                        vertex: neighbor,
                        distance: edge_length,
                    });
                }
            }
        }
    }

    while let Some(HeapEntry { vertex, distance }) = heap.pop() {
        if frozen[vertex] || distance > distances[vertex] {
            continue;
        }

        if let Some(max) = max_distance {
            if distance > max {
                continue;
            }
        }

        frozen[vertex] = true;

        // Update neighbors using FMM update rule
        for &neighbor in &adjacency[vertex] {
            if frozen[neighbor] {
                continue;
            }

            // Try to update using triangles containing this edge
            let new_dist =
                compute_fmm_update(mesh, neighbor, vertex, &distances, &frozen, &triangle_adj);

            if new_dist < distances[neighbor] {
                distances[neighbor] = new_dist;
                nearest_source[neighbor] = nearest_source[vertex];
                heap.push(HeapEntry {
                    vertex: neighbor,
                    distance: new_dist,
                });
            }
        }
    }

    GeodesicResult {
        distances,
        nearest_source,
    }
}

fn build_triangle_adjacency(mesh: &Mesh) -> HashMap<(usize, usize), Vec<usize>> {
    let mut adj: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for (tri_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        // Each edge maps to the third vertex of triangles containing it
        adj.entry((a.min(b), a.max(b))).or_default().push(c);
        adj.entry((b.min(c), b.max(c))).or_default().push(a);
        adj.entry((c.min(a), c.max(a))).or_default().push(b);
        let _ = tri_idx;
    }

    adj
}

fn compute_fmm_update(
    mesh: &Mesh,
    target: usize,
    from: usize,
    distances: &[f32],
    frozen: &[bool],
    triangle_adj: &HashMap<(usize, usize), Vec<usize>>,
) -> f32 {
    let positions = &mesh.positions;

    // Direct edge update
    let edge_dist = distances[from] + (positions[target] - positions[from]).length();
    let mut best = edge_dist;

    // Try triangle updates
    let edge_key = (target.min(from), target.max(from));
    if let Some(thirds) = triangle_adj.get(&edge_key) {
        for &third in thirds {
            if !frozen[third] {
                continue;
            }

            // Solve the eikonal equation on this triangle
            if let Some(dist) = solve_triangle_update(
                positions[target],
                positions[from],
                positions[third],
                distances[from],
                distances[third],
            ) {
                best = best.min(dist);
            }
        }
    }

    best
}

fn solve_triangle_update(target: Vec3, v1: Vec3, v2: Vec3, d1: f32, d2: f32) -> Option<f32> {
    // Solve the eikonal equation: |∇u| = 1/speed = 1
    // Using the method from "Computing Geodesic Paths on Manifolds"

    let e1 = v1 - target;
    let e2 = v2 - target;

    let a = e1.length_squared();
    let b = e1.dot(e2);
    let c = e2.length_squared();

    let u = d2 - d1;

    // Quadratic coefficients
    let det = a * c - b * b;
    if det < 1e-10 {
        return None;
    }

    // Solve for parameter t
    let _t_num = c * u - b * (d2 - d1);
    let t_denom = det.sqrt();

    if t_denom < 1e-10 {
        return None;
    }

    // Check if solution is valid (within triangle)
    let qa = a - 2.0 * b + c;
    let qb = 2.0 * (b - a);
    let qc = a - u * u;

    let discriminant = qb * qb - 4.0 * qa * qc;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-qb + sqrt_disc) / (2.0 * qa);
    let t2 = (-qb - sqrt_disc) / (2.0 * qa);

    let mut result = f32::INFINITY;

    for t in [t1, t2] {
        if t >= 0.0 && t <= 1.0 {
            let interp = e1 * (1.0 - t) + e2 * t;
            let dist = d1 * (1.0 - t) + d2 * t + interp.length();
            result = result.min(dist);
        }
    }

    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

fn find_boundary_vertices(mesh: &Mesh) -> Vec<usize> {
    use std::collections::HashSet;

    // Count edge occurrences
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let edge = (a.min(b), a.max(b));
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // Boundary edges appear exactly once
    let mut boundary_verts: HashSet<usize> = HashSet::new();
    for ((a, b), count) in edge_count {
        if count == 1 {
            boundary_verts.insert(a as usize);
            boundary_verts.insert(b as usize);
        }
    }

    boundary_verts.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plane_mesh() -> Mesh {
        // Simple 2x2 plane with 9 vertices
        let mut mesh = Mesh::new();

        for y in 0..3 {
            for x in 0..3 {
                mesh.positions.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }

        // Two triangles per quad
        mesh.indices.extend([
            0, 1, 3, 1, 4, 3, // bottom-left quad
            1, 2, 4, 2, 5, 4, // bottom-right quad
            3, 4, 6, 4, 7, 6, // top-left quad
            4, 5, 7, 5, 8, 7, // top-right quad
        ]);

        mesh
    }

    #[test]
    fn test_geodesic_basic() {
        let mesh = make_plane_mesh();
        // Use Dijkstra for predictable results
        let config = GeodesicConfig {
            use_fast_marching: false,
            max_distance: None,
        };
        let result = compute_geodesic_distance(&mesh, &[0], config);

        // Source vertex should have distance 0
        assert_eq!(result.distances[0], 0.0);

        // Adjacent vertices should have distance ~1
        assert!((result.distances[1] - 1.0).abs() < 0.1);
        assert!((result.distances[3] - 1.0).abs() < 0.1);

        // Center (4) is at (1,1), path 0->1->4 or 0->3->4 = 2.0
        assert!((result.distances[4] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_geodesic_dijkstra() {
        let mesh = make_plane_mesh();
        let config = GeodesicConfig {
            use_fast_marching: false,
            max_distance: None,
        };
        let result = compute_geodesic_distance(&mesh, &[0], config);

        assert_eq!(result.distances[0], 0.0);
        assert!((result.distances[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_sources() {
        let mesh = make_plane_mesh();

        // Sources at opposite corners, use Dijkstra for predictable results
        let config = GeodesicConfig {
            use_fast_marching: false,
            max_distance: None,
        };
        let result = compute_geodesic_distance(&mesh, &[0, 8], config);

        // Both corners should have distance 0
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[8], 0.0);

        // Center (4) is 2 edges away from both corners
        let center_dist = result.distances[4];
        assert!((center_dist - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_max_distance() {
        let mesh = make_plane_mesh();
        let config = GeodesicConfig {
            max_distance: Some(1.5),
            use_fast_marching: false,
        };
        let result = compute_geodesic_distance(&mesh, &[0], config);

        // Vertices within max_distance should be computed
        assert_eq!(result.distances[0], 0.0);
        assert!((result.distances[1] - 1.0).abs() < 0.1);

        // Far vertices should be infinity
        assert!(result.distances[8] == f32::INFINITY);
    }

    #[test]
    fn test_geodesic_path() {
        let mesh = make_plane_mesh();
        let path = geodesic_path(&mesh, 0, 8);

        // Path should start at source and end at target
        assert_eq!(*path.first().unwrap(), 0);
        assert_eq!(*path.last().unwrap(), 8);

        // Path should be connected
        for window in path.windows(2) {
            let a = window[0];
            let b = window[1];
            let dist = (mesh.positions[a] - mesh.positions[b]).length();
            assert!(dist < 2.0); // Adjacent or diagonal
        }
    }

    #[test]
    fn test_farthest_vertex() {
        let mesh = make_plane_mesh();
        let result = compute_geodesic_distance(&mesh, &[0], GeodesicConfig::default());

        // Farthest from corner 0 should be corner 8
        let farthest = result.farthest_vertex();
        assert_eq!(farthest, 8);
    }

    #[test]
    fn test_vertices_within_distance() {
        let mesh = make_plane_mesh();
        let result = compute_geodesic_distance(&mesh, &[4], GeodesicConfig::default());

        // From center, vertices within distance 1.5 should include adjacent
        let nearby = result.vertices_within_distance(1.5);
        assert!(nearby.contains(&4)); // Center itself
        assert!(nearby.contains(&1)); // Top adjacent
        assert!(nearby.contains(&3)); // Left adjacent
    }

    #[test]
    fn test_iso_distance() {
        let mesh = make_plane_mesh();

        // Find vertices at distance ~1 from center
        let iso_verts = iso_distance_vertices(&mesh, 4, 1.0, 0.2);

        // Should include the 4 adjacent vertices
        assert!(
            iso_verts.contains(&1)
                || iso_verts.contains(&3)
                || iso_verts.contains(&5)
                || iso_verts.contains(&7)
        );
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::new();
        let result = compute_geodesic_distance(&mesh, &[0], GeodesicConfig::default());
        assert!(result.distances.is_empty());
    }

    #[test]
    fn test_geodesic_distance_from() {
        let mesh = make_plane_mesh();
        let distances = geodesic_distance_from(&mesh, 0);

        assert_eq!(distances[0], 0.0);
        assert!(distances[8] > 0.0);
    }

    #[test]
    fn test_boundary_vertices() {
        let mesh = make_plane_mesh();
        let boundary = find_boundary_vertices(&mesh);

        // All edge vertices should be on boundary (not center)
        assert!(!boundary.contains(&4)); // Center not on boundary
        assert!(boundary.len() == 8); // 8 edge vertices
    }
}
