//! Mesh topology analysis.
//!
//! Provides tools for analyzing mesh topology:
//! - Euler characteristic and genus
//! - Manifold testing
//! - Boundary detection
//! - Connected components
//!
//! # Example
//!
//! ```
//! use rhizome_resin_mesh::{UvSphere, analyze_topology};
//!
//! let sphere = UvSphere::new(1.0, 16, 8).apply();
//! let topo = analyze_topology(&sphere);
//!
//! assert!(topo.is_closed);
//! assert!(topo.is_manifold);
//! assert_eq!(topo.genus, 0); // Sphere has genus 0
//! ```

use crate::Mesh;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Result of topology analysis.
#[derive(Debug, Clone)]
pub struct TopologyInfo {
    /// Number of vertices.
    pub vertex_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Number of faces (triangles).
    pub face_count: usize,
    /// Euler characteristic: V - E + F.
    pub euler_characteristic: i32,
    /// Genus of the surface (for closed orientable manifolds).
    /// Genus 0 = sphere, 1 = torus, etc.
    pub genus: i32,
    /// Whether the mesh is a manifold (each edge has exactly 1 or 2 faces).
    pub is_manifold: bool,
    /// Whether the mesh is closed (no boundary edges).
    pub is_closed: bool,
    /// Whether the mesh is orientable.
    pub is_orientable: bool,
    /// Number of boundary loops.
    pub boundary_loop_count: usize,
    /// Number of connected components.
    pub connected_components: usize,
    /// Edges shared by more than 2 faces (non-manifold edges).
    pub non_manifold_edges: Vec<(u32, u32)>,
    /// Boundary edges (shared by exactly 1 face).
    pub boundary_edges: Vec<(u32, u32)>,
}

impl TopologyInfo {
    /// Returns true if the mesh is watertight (closed manifold).
    pub fn is_watertight(&self) -> bool {
        self.is_closed && self.is_manifold
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        format!(
            "V={} E={} F={} χ={} g={} manifold={} closed={} components={}",
            self.vertex_count,
            self.edge_count,
            self.face_count,
            self.euler_characteristic,
            self.genus,
            self.is_manifold,
            self.is_closed,
            self.connected_components
        )
    }
}

/// Canonical edge representation (smaller index first).
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Quantizes a position for spatial hashing.
fn quantize_position(p: Vec3, epsilon: f32) -> (i32, i32, i32) {
    let scale = 1.0 / epsilon;
    (
        (p.x * scale).round() as i32,
        (p.y * scale).round() as i32,
        (p.z * scale).round() as i32,
    )
}

/// Builds a mapping from vertex indices to welded vertex indices.
/// Vertices at the same position (within epsilon) get the same index.
fn build_weld_map(positions: &[Vec3], epsilon: f32) -> Vec<u32> {
    let mut position_to_index: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut weld_map = Vec::with_capacity(positions.len());
    let mut next_index = 0u32;

    for pos in positions {
        let key = quantize_position(*pos, epsilon);
        let index = *position_to_index.entry(key).or_insert_with(|| {
            let idx = next_index;
            next_index += 1;
            idx
        });
        weld_map.push(index);
    }

    weld_map
}

/// Analyzes the topology of a mesh.
///
/// This function welds vertices at the same position (within a small epsilon)
/// before analyzing topology. This is important for meshes where vertices
/// are duplicated for UV seams or sharp normals.
pub fn analyze_topology(mesh: &Mesh) -> TopologyInfo {
    analyze_topology_with_epsilon(mesh, 1e-5)
}

/// Analyzes the topology of a mesh with a custom weld epsilon.
pub fn analyze_topology_with_epsilon(mesh: &Mesh, epsilon: f32) -> TopologyInfo {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return TopologyInfo {
            vertex_count: mesh.positions.len(),
            edge_count: 0,
            face_count: 0,
            euler_characteristic: mesh.positions.len() as i32,
            genus: 0,
            is_manifold: true,
            is_closed: true,
            is_orientable: true,
            boundary_loop_count: 0,
            connected_components: if mesh.positions.is_empty() { 0 } else { 1 },
            non_manifold_edges: vec![],
            boundary_edges: vec![],
        };
    }

    // Build weld map to handle duplicate vertices
    let weld_map = build_weld_map(&mesh.positions, epsilon);
    let welded_vertex_count = weld_map.iter().max().map(|&m| m + 1).unwrap_or(0) as usize;

    let face_count = mesh.indices.len() / 3;

    // Build edge -> face count mapping using welded indices
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let v0 = weld_map[tri[0] as usize];
        let v1 = weld_map[tri[1] as usize];
        let v2 = weld_map[tri[2] as usize];

        // Skip degenerate triangles
        if v0 == v1 || v1 == v2 || v2 == v0 {
            continue;
        }

        edge_faces
            .entry(edge_key(v0, v1))
            .or_default()
            .push(face_idx);
        edge_faces
            .entry(edge_key(v1, v2))
            .or_default()
            .push(face_idx);
        edge_faces
            .entry(edge_key(v2, v0))
            .or_default()
            .push(face_idx);
    }

    let edge_count = edge_faces.len();

    // Classify edges
    let mut non_manifold_edges = Vec::new();
    let mut boundary_edges = Vec::new();

    for (&edge, faces) in &edge_faces {
        match faces.len() {
            1 => boundary_edges.push(edge),
            2 => {} // Normal manifold edge
            _ => non_manifold_edges.push(edge),
        }
    }

    let is_manifold = non_manifold_edges.is_empty();
    let is_closed = boundary_edges.is_empty();

    // Check orientability using directed edges (with welded indices)
    let is_orientable = check_orientability_welded(mesh, &weld_map, &edge_faces);

    // Count boundary loops
    let boundary_loop_count = count_boundary_loops(&boundary_edges, welded_vertex_count);

    // Count connected components using welded mesh
    let connected_components = count_connected_components_welded(mesh, &weld_map);

    // Euler characteristic (using welded vertex count)
    let euler_characteristic = welded_vertex_count as i32 - edge_count as i32 + face_count as i32;

    // Genus calculation (for closed orientable manifolds)
    // χ = 2 - 2g for a closed surface, so g = (2 - χ) / 2
    let genus = if is_closed && is_orientable && is_manifold {
        (2 - euler_characteristic) / 2
    } else {
        // For surfaces with boundary: χ = 2 - 2g - b, so g = (2 - χ - b) / 2
        (2 - euler_characteristic - boundary_loop_count as i32) / 2
    };

    TopologyInfo {
        vertex_count: welded_vertex_count,
        edge_count,
        face_count,
        euler_characteristic,
        genus: genus.max(0),
        is_manifold,
        is_closed,
        is_orientable,
        boundary_loop_count,
        connected_components,
        non_manifold_edges,
        boundary_edges,
    }
}

/// Counts the number of boundary loops.
fn count_boundary_loops(boundary_edges: &[(u32, u32)], _vertex_count: usize) -> usize {
    if boundary_edges.is_empty() {
        return 0;
    }

    // Build adjacency for boundary vertices
    let mut boundary_adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in boundary_edges {
        boundary_adj.entry(a).or_default().push(b);
        boundary_adj.entry(b).or_default().push(a);
    }

    // Count loops by traversing
    let mut visited: HashSet<u32> = HashSet::new();
    let mut loop_count = 0;

    for &start in boundary_adj.keys() {
        if visited.contains(&start) {
            continue;
        }

        // BFS/DFS to mark all vertices in this boundary component
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            if visited.insert(v) {
                if let Some(neighbors) = boundary_adj.get(&v) {
                    for &n in neighbors {
                        if !visited.contains(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
        }
        loop_count += 1;
    }

    loop_count
}

#[allow(dead_code)]
/// Counts connected components using union-find (for unwelded meshes).
fn count_connected_components(mesh: &Mesh) -> usize {
    if mesh.positions.is_empty() {
        return 0;
    }

    let n = mesh.positions.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            if rank[px] < rank[py] {
                parent[px] = py;
            } else if rank[px] > rank[py] {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px] += 1;
            }
        }
    }

    // Union vertices that share an edge
    for tri in mesh.indices.chunks(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        union(&mut parent, &mut rank, i0, i1);
        union(&mut parent, &mut rank, i1, i2);
    }

    // Count unique roots among vertices that are actually used
    let used_vertices: HashSet<usize> = mesh.indices.iter().map(|&i| i as usize).collect();
    let roots: HashSet<usize> = used_vertices
        .iter()
        .map(|&v| find(&mut parent, v))
        .collect();

    roots.len()
}

/// Check orientability using welded vertex indices.
fn check_orientability_welded(
    mesh: &Mesh,
    weld_map: &[u32],
    edge_faces: &HashMap<(u32, u32), Vec<usize>>,
) -> bool {
    if mesh.indices.len() < 3 {
        return true;
    }

    // Build directed edge -> face mapping using welded indices
    let mut directed_edges: HashMap<(u32, u32), usize> = HashMap::new();
    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let v0 = weld_map[tri[0] as usize];
        let v1 = weld_map[tri[1] as usize];
        let v2 = weld_map[tri[2] as usize];

        // Skip degenerate triangles
        if v0 == v1 || v1 == v2 || v2 == v0 {
            continue;
        }

        directed_edges.insert((v0, v1), face_idx);
        directed_edges.insert((v1, v2), face_idx);
        directed_edges.insert((v2, v0), face_idx);
    }

    // For consistent orientation, when two faces share an edge,
    // one should have it as (a,b) and the other as (b,a)
    for (&edge, faces) in edge_faces {
        if faces.len() == 2 {
            let (a, b) = edge;
            // Check if one face has (a,b) and other has (b,a)
            let f0_has_ab = directed_edges
                .get(&(a, b))
                .map(|&f| f == faces[0])
                .unwrap_or(false);
            let f0_has_ba = directed_edges
                .get(&(b, a))
                .map(|&f| f == faces[0])
                .unwrap_or(false);
            let f1_has_ab = directed_edges
                .get(&(a, b))
                .map(|&f| f == faces[1])
                .unwrap_or(false);
            let f1_has_ba = directed_edges
                .get(&(b, a))
                .map(|&f| f == faces[1])
                .unwrap_or(false);

            // Consistent: f0 has (a,b) and f1 has (b,a), or vice versa
            let consistent = (f0_has_ab && f1_has_ba) || (f0_has_ba && f1_has_ab);
            if !consistent {
                return false;
            }
        }
    }

    true
}

/// Counts connected components using union-find with welded indices.
fn count_connected_components_welded(mesh: &Mesh, weld_map: &[u32]) -> usize {
    if mesh.positions.is_empty() {
        return 0;
    }

    let n = weld_map.iter().max().map(|&m| m + 1).unwrap_or(0) as usize;
    if n == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            if rank[px] < rank[py] {
                parent[px] = py;
            } else if rank[px] > rank[py] {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px] += 1;
            }
        }
    }

    // Union vertices that share an edge (using welded indices)
    for tri in mesh.indices.chunks(3) {
        let i0 = weld_map[tri[0] as usize] as usize;
        let i1 = weld_map[tri[1] as usize] as usize;
        let i2 = weld_map[tri[2] as usize] as usize;
        union(&mut parent, &mut rank, i0, i1);
        union(&mut parent, &mut rank, i1, i2);
    }

    // Count unique roots among vertices that are actually used
    let used_vertices: HashSet<usize> = mesh
        .indices
        .iter()
        .map(|&i| weld_map[i as usize] as usize)
        .collect();
    let roots: HashSet<usize> = used_vertices
        .iter()
        .map(|&v| find(&mut parent, v))
        .collect();

    roots.len()
}

/// Checks if a mesh is manifold.
pub fn is_manifold(mesh: &Mesh) -> bool {
    analyze_topology(mesh).is_manifold
}

/// Checks if a mesh is closed (watertight, no boundary).
pub fn is_closed(mesh: &Mesh) -> bool {
    analyze_topology(mesh).is_closed
}

/// Computes the Euler characteristic of a mesh.
pub fn euler_characteristic(mesh: &Mesh) -> i32 {
    analyze_topology(mesh).euler_characteristic
}

/// Computes the genus of a mesh.
pub fn genus(mesh: &Mesh) -> i32 {
    analyze_topology(mesh).genus
}

/// Finds all boundary edges of a mesh.
pub fn find_boundary_edges(mesh: &Mesh) -> Vec<(u32, u32)> {
    analyze_topology(mesh).boundary_edges
}

/// Finds all non-manifold edges of a mesh.
pub fn find_non_manifold_edges(mesh: &Mesh) -> Vec<(u32, u32)> {
    analyze_topology(mesh).non_manifold_edges
}

/// Finds boundary vertices (vertices on at least one boundary edge).
pub fn find_boundary_vertices(mesh: &Mesh) -> Vec<u32> {
    let boundary_edges = find_boundary_edges(mesh);
    let mut boundary_verts: HashSet<u32> = HashSet::new();

    for (a, b) in boundary_edges {
        boundary_verts.insert(a);
        boundary_verts.insert(b);
    }

    let mut result: Vec<u32> = boundary_verts.into_iter().collect();
    result.sort();
    result
}

/// Counts connected components in a mesh.
pub fn connected_components(mesh: &Mesh) -> usize {
    analyze_topology(mesh).connected_components
}

/// Extracts boundary loops as ordered vertex sequences.
pub fn extract_boundary_loops(mesh: &Mesh) -> Vec<Vec<u32>> {
    let topo = analyze_topology(mesh);
    if topo.boundary_edges.is_empty() {
        return vec![];
    }

    // Build adjacency for boundary
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &topo.boundary_edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    let mut visited_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut loops = Vec::new();

    for &start in adj.keys() {
        // Find an unvisited edge from this vertex
        let neighbors = match adj.get(&start) {
            Some(n) => n,
            None => continue,
        };

        for &next in neighbors {
            let edge = edge_key(start, next);
            if visited_edges.contains(&edge) {
                continue;
            }

            // Trace the loop
            let mut loop_verts = vec![start];
            let mut current = next;
            let mut prev = start;
            visited_edges.insert(edge);

            loop {
                loop_verts.push(current);

                if current == start {
                    // Completed the loop
                    loop_verts.pop(); // Remove duplicate start
                    break;
                }

                // Find next vertex (not the one we came from)
                let current_neighbors = match adj.get(&current) {
                    Some(n) => n,
                    None => break,
                };

                let mut found_next = false;
                for &n in current_neighbors {
                    if n != prev {
                        let edge = edge_key(current, n);
                        if !visited_edges.contains(&edge) {
                            visited_edges.insert(edge);
                            prev = current;
                            current = n;
                            found_next = true;
                            break;
                        }
                    }
                }

                if !found_next {
                    break;
                }
            }

            if loop_verts.len() >= 3 {
                loops.push(loop_verts);
            }
        }
    }

    loops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cuboid, UvSphere};

    #[test]
    fn test_sphere_topology() {
        let sphere = UvSphere::new(1.0, 16, 8).apply();
        let topo = analyze_topology(&sphere);

        // Sphere should be closed manifold with genus 0
        assert!(topo.is_closed);
        assert!(topo.is_manifold);
        assert_eq!(topo.genus, 0);
        assert!(topo.is_orientable);
        assert_eq!(topo.connected_components, 1);
    }

    #[test]
    fn test_box_topology() {
        let cube = Cuboid::default().apply();
        let topo = analyze_topology(&cube);

        // Box should be closed manifold with genus 0
        assert!(topo.is_closed);
        assert!(topo.is_manifold);
        assert_eq!(topo.genus, 0);
        assert_eq!(topo.connected_components, 1);
    }

    #[test]
    fn test_euler_characteristic_sphere() {
        let sphere = UvSphere::new(1.0, 16, 8).apply();
        let chi = euler_characteristic(&sphere);

        // Euler characteristic of sphere is 2
        assert_eq!(chi, 2);
    }

    #[test]
    fn test_open_mesh() {
        // Create a single triangle (open mesh)
        let mesh = Mesh {
            positions: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2],
        };

        let topo = analyze_topology(&mesh);

        assert!(!topo.is_closed);
        assert!(topo.is_manifold);
        assert_eq!(topo.boundary_edges.len(), 3);
        assert_eq!(topo.boundary_loop_count, 1);
    }

    #[test]
    fn test_quad_with_hole() {
        // Create a quad with a triangular hole (like a frame)
        // This creates a surface with one boundary loop
        let mesh = Mesh {
            positions: vec![
                // Outer square
                glam::Vec3::new(0.0, 0.0, 0.0), // 0
                glam::Vec3::new(2.0, 0.0, 0.0), // 1
                glam::Vec3::new(2.0, 2.0, 0.0), // 2
                glam::Vec3::new(0.0, 2.0, 0.0), // 3
                // Inner triangle
                glam::Vec3::new(0.5, 0.5, 0.0), // 4
                glam::Vec3::new(1.5, 0.5, 0.0), // 5
                glam::Vec3::new(1.0, 1.5, 0.0), // 6
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![
                // Connect outer to inner with triangles
                0, 1, 5, 0, 5, 4, 1, 2, 5, 2, 6, 5, 2, 3, 6, 3, 4, 6, 3, 0, 4,
            ],
        };

        let topo = analyze_topology(&mesh);

        // Should have 2 boundary loops (outer and inner)
        assert!(!topo.is_closed);
        assert!(topo.is_manifold);
        assert_eq!(topo.boundary_loop_count, 2);
    }

    #[test]
    fn test_non_manifold_edge() {
        // Create three triangles sharing one edge
        let mesh = Mesh {
            positions: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),  // 0
                glam::Vec3::new(1.0, 0.0, 0.0),  // 1
                glam::Vec3::new(0.5, 1.0, 0.0),  // 2
                glam::Vec3::new(0.5, -1.0, 0.0), // 3
                glam::Vec3::new(0.5, 0.0, 1.0),  // 4
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![
                0, 1, 2, // First triangle
                0, 1, 3, // Second triangle (shares edge 0-1)
                0, 1, 4, // Third triangle (also shares edge 0-1)
            ],
        };

        let topo = analyze_topology(&mesh);

        assert!(!topo.is_manifold);
        assert!(topo.non_manifold_edges.contains(&(0, 1)));
    }

    #[test]
    fn test_connected_components() {
        // Two separate triangles
        let mesh = Mesh {
            positions: vec![
                // First triangle
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.5, 1.0, 0.0),
                // Second triangle (disconnected)
                glam::Vec3::new(5.0, 0.0, 0.0),
                glam::Vec3::new(6.0, 0.0, 0.0),
                glam::Vec3::new(5.5, 1.0, 0.0),
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let components = connected_components(&mesh);
        assert_eq!(components, 2);
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::new();
        let topo = analyze_topology(&mesh);

        assert_eq!(topo.vertex_count, 0);
        assert_eq!(topo.connected_components, 0);
        assert!(topo.is_manifold);
        assert!(topo.is_closed);
    }

    #[test]
    fn test_boundary_vertices() {
        // Single triangle
        let mesh = Mesh {
            positions: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2],
        };

        let boundary_verts = find_boundary_vertices(&mesh);
        assert_eq!(boundary_verts.len(), 3);
    }

    #[test]
    fn test_extract_boundary_loops() {
        // Single triangle (one boundary loop of 3 vertices)
        let mesh = Mesh {
            positions: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2],
        };

        let loops = extract_boundary_loops(&mesh);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 3);
    }

    #[test]
    fn test_topology_summary() {
        let sphere = UvSphere::new(1.0, 8, 4).apply();
        let topo = analyze_topology(&sphere);
        let summary = topo.summary();

        assert!(summary.contains("manifold=true"));
        assert!(summary.contains("closed=true"));
    }

    #[test]
    fn test_is_watertight() {
        let sphere = UvSphere::new(1.0, 8, 4).apply();
        let topo = analyze_topology(&sphere);
        assert!(topo.is_watertight());

        // Single triangle is not watertight
        let tri = Mesh {
            positions: vec![
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![],
            uvs: vec![],
            indices: vec![0, 1, 2],
        };
        let topo = analyze_topology(&tri);
        assert!(!topo.is_watertight());
    }

    #[test]
    fn test_convenience_functions() {
        let sphere = UvSphere::new(1.0, 8, 4).apply();

        assert!(is_manifold(&sphere));
        assert!(is_closed(&sphere));
        assert_eq!(euler_characteristic(&sphere), 2);
        assert_eq!(genus(&sphere), 0);
        assert!(find_boundary_edges(&sphere).is_empty());
        assert!(find_non_manifold_edges(&sphere).is_empty());
        assert!(find_boundary_vertices(&sphere).is_empty());
        assert_eq!(connected_components(&sphere), 1);
    }
}
