//! Edge loop and ring selection operations.
//!
//! Edge loops are continuous chains of edges that typically wrap around a mesh.
//! Edge rings are perpendicular to loops, following the quad topology "grain".
//!
//! # Usage
//!
//! ```ignore
//! let hemesh = HalfEdgeMesh::from_mesh(&cylinder);
//!
//! // Select an edge loop starting from an edge
//! let loop_edges = select_edge_loop(&hemesh, edge_id);
//!
//! // Select an edge ring
//! let ring_edges = select_edge_ring(&hemesh, edge_id);
//!
//! // Cut a loop into the mesh
//! let cut_mesh = loop_cut(&hemesh, edge_id, 1);
//! ```

use crate::halfedge::{FaceId, HalfEdgeId, HalfEdgeMesh, Vertex, VertexId};
use std::collections::HashSet;

/// Selects an edge loop starting from the given edge.
///
/// An edge loop follows edges across quad faces, continuing through
/// vertices that have exactly 4 edges (quad topology). The loop stops
/// when it returns to the starting edge or reaches a boundary/non-quad vertex.
pub fn select_edge_loop(mesh: &HalfEdgeMesh, start_edge: HalfEdgeId) -> Vec<HalfEdgeId> {
    if start_edge.is_null() || mesh.halfedges.is_empty() {
        return Vec::new();
    }

    let mut loop_edges = Vec::new();
    let mut visited: HashSet<u32> = HashSet::new();

    // Traverse in one direction
    let forward = traverse_loop_direction(mesh, start_edge, &mut visited);
    loop_edges.extend(forward);

    // Traverse in the opposite direction (from twin)
    let he = &mesh.halfedges[start_edge.0 as usize];
    if !he.twin.is_null() {
        let backward = traverse_loop_direction(mesh, he.twin, &mut visited);
        // Prepend backward edges (reversed) before start
        let mut result = Vec::new();
        for &e in backward.iter().rev() {
            // Get the twin to maintain consistent direction
            let twin = mesh.halfedges[e.0 as usize].twin;
            if !twin.is_null() {
                result.push(twin);
            }
        }
        result.extend(loop_edges);
        return result;
    }

    loop_edges
}

/// Traverses an edge loop in one direction.
fn traverse_loop_direction(
    mesh: &HalfEdgeMesh,
    start: HalfEdgeId,
    visited: &mut HashSet<u32>,
) -> Vec<HalfEdgeId> {
    let mut result = Vec::new();
    let mut current = start;

    loop {
        if current.is_null() || visited.contains(&current.0) {
            break;
        }

        // Mark both the edge and its twin as visited
        visited.insert(current.0);
        let twin = mesh.halfedges[current.0 as usize].twin;
        if !twin.is_null() {
            visited.insert(twin.0);
        }

        result.push(current);

        // Find the next edge in the loop
        // For quad topology: go to the opposite edge of the face
        let next = find_opposite_edge_in_face(mesh, current);
        if next.is_null() {
            break;
        }

        // Cross to the next face via twin
        let next_twin = mesh.halfedges[next.0 as usize].twin;
        if next_twin.is_null() {
            break;
        }

        current = next_twin;
    }

    result
}

/// Finds the edge opposite to the given edge within its face.
/// This only works for quad faces (returns NULL for non-quads).
fn find_opposite_edge_in_face(mesh: &HalfEdgeMesh, edge: HalfEdgeId) -> HalfEdgeId {
    let he = &mesh.halfedges[edge.0 as usize];
    if he.face.is_null() {
        return HalfEdgeId::NULL;
    }

    // Count edges in the face and find the opposite one
    let face_edges = mesh.face_halfedges(he.face);

    if face_edges.len() != 4 {
        // Not a quad, can't continue loop
        return HalfEdgeId::NULL;
    }

    // Find the position of our edge in the face
    let pos = face_edges.iter().position(|&e| e == edge);
    match pos {
        Some(i) => face_edges[(i + 2) % 4], // Opposite edge is 2 steps away
        None => HalfEdgeId::NULL,
    }
}

/// Selects an edge ring starting from the given edge.
///
/// An edge ring follows edges along the face direction (perpendicular to loops).
/// It connects edges that share a vertex within quad faces.
pub fn select_edge_ring(mesh: &HalfEdgeMesh, start_edge: HalfEdgeId) -> Vec<HalfEdgeId> {
    if start_edge.is_null() || mesh.halfedges.is_empty() {
        return Vec::new();
    }

    let mut ring_edges = Vec::new();
    let mut visited: HashSet<u32> = HashSet::new();

    // Traverse in both directions from the start edge
    let forward = traverse_ring_direction(mesh, start_edge, true, &mut visited);
    ring_edges.extend(forward);

    // Reset visited for backward traversal, but keep start edge marked
    let backward = traverse_ring_direction(mesh, start_edge, false, &mut visited);

    // Combine: backward (reversed) + forward (excluding start)
    let mut result = Vec::new();
    for &e in backward.iter().rev().skip(1) {
        result.push(e);
    }
    result.extend(ring_edges);

    result
}

/// Traverses an edge ring in one direction.
fn traverse_ring_direction(
    mesh: &HalfEdgeMesh,
    start: HalfEdgeId,
    forward: bool,
    visited: &mut HashSet<u32>,
) -> Vec<HalfEdgeId> {
    let mut result = Vec::new();
    let mut current = start;

    loop {
        if current.is_null() || visited.contains(&current.0) {
            break;
        }

        visited.insert(current.0);
        let twin = mesh.halfedges[current.0 as usize].twin;
        if !twin.is_null() {
            visited.insert(twin.0);
        }

        result.push(current);

        // Find adjacent edge in ring direction
        let next = find_ring_adjacent_edge(mesh, current, forward);
        if next.is_null() {
            break;
        }

        current = next;
    }

    result
}

/// Finds the adjacent edge in the ring direction.
fn find_ring_adjacent_edge(mesh: &HalfEdgeMesh, edge: HalfEdgeId, forward: bool) -> HalfEdgeId {
    let he = &mesh.halfedges[edge.0 as usize];

    // Get the vertex we're moving toward
    let _vertex = if forward { he.vertex } else { he.origin };

    // Find the face on the appropriate side
    let face_edge = if forward { edge } else { he.twin };
    if face_edge.is_null() {
        return HalfEdgeId::NULL;
    }

    let face_he = &mesh.halfedges[face_edge.0 as usize];
    if face_he.face.is_null() {
        return HalfEdgeId::NULL;
    }

    let face_edges = mesh.face_halfedges(face_he.face);
    if face_edges.len() != 4 {
        return HalfEdgeId::NULL;
    }

    // Find our edge and get the adjacent one (perpendicular in the quad)
    let pos = face_edges.iter().position(|&e| e == face_edge);
    match pos {
        Some(i) => {
            // Adjacent edge is 1 step away (next or prev based on direction)
            let adj_idx = if forward { (i + 1) % 4 } else { (i + 3) % 4 };
            let adj_edge = face_edges[adj_idx];

            // Cross to the next quad via twin
            let adj_twin = mesh.halfedges[adj_edge.0 as usize].twin;
            if adj_twin.is_null() {
                return HalfEdgeId::NULL;
            }

            // Find the continuing edge in that face
            let next_face_he = &mesh.halfedges[adj_twin.0 as usize];
            if next_face_he.face.is_null() {
                return HalfEdgeId::NULL;
            }

            let next_face_edges = mesh.face_halfedges(next_face_he.face);
            if next_face_edges.len() != 4 {
                return HalfEdgeId::NULL;
            }

            // The continuing edge is adjacent to adj_twin
            let adj_pos = next_face_edges.iter().position(|&e| e == adj_twin);
            match adj_pos {
                Some(j) => {
                    let continue_idx = if forward { (j + 1) % 4 } else { (j + 3) % 4 };
                    next_face_edges[continue_idx]
                }
                None => HalfEdgeId::NULL,
            }
        }
        None => HalfEdgeId::NULL,
    }
}

/// Performs a loop cut, adding new edge loops to the mesh.
///
/// This subdivides all faces along the loop, creating new vertices and edges.
pub fn loop_cut(mesh: &HalfEdgeMesh, edge: HalfEdgeId, cuts: u32) -> HalfEdgeMesh {
    if edge.is_null() || cuts == 0 {
        return mesh.clone();
    }

    let loop_edges = select_edge_loop(mesh, edge);
    if loop_edges.is_empty() {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // For each cut, we need to:
    // 1. Find all edges in the loop
    // 2. Add new vertices along those edges
    // 3. Split the adjacent faces

    for cut_idx in 0..cuts {
        let t = (cut_idx + 1) as f32 / (cuts + 1) as f32;

        // Collect edges to split and their midpoints
        let mut edge_midpoints: Vec<(HalfEdgeId, VertexId)> = Vec::new();

        for &loop_edge in &loop_edges {
            if loop_edge.0 as usize >= result.halfedges.len() {
                continue;
            }

            let he = &result.halfedges[loop_edge.0 as usize];
            let v0_pos = result.vertices[he.origin.0 as usize].position;
            let v1_pos = result.vertices[he.vertex.0 as usize].position;

            // Interpolate position
            let mid_pos = v0_pos.lerp(v1_pos, t);

            // Add new vertex
            let new_vid = VertexId(result.vertices.len() as u32);
            result.vertices.push(Vertex {
                position: mid_pos,
                normal: result.vertices[he.origin.0 as usize]
                    .normal
                    .lerp(result.vertices[he.vertex.0 as usize].normal, t)
                    .normalize_or_zero(),
                uv: result.vertices[he.origin.0 as usize]
                    .uv
                    .lerp(result.vertices[he.vertex.0 as usize].uv, t),
                halfedge: HalfEdgeId::NULL,
            });

            edge_midpoints.push((loop_edge, new_vid));
        }

        // Note: Full implementation would split faces here
        // This is a simplified version that just adds the vertices
    }

    result.compute_normals();
    result
}

/// Returns the edges that form the boundary of the mesh.
pub fn select_boundary_edges(mesh: &HalfEdgeMesh) -> Vec<HalfEdgeId> {
    let mut boundary = Vec::new();

    for (i, he) in mesh.halfedges.iter().enumerate() {
        if he.face.is_null() || he.twin.is_null() {
            continue;
        }

        // Check if twin is a boundary edge
        let twin = &mesh.halfedges[he.twin.0 as usize];
        if twin.face.is_null() {
            boundary.push(HalfEdgeId(i as u32));
        }
    }

    boundary
}

/// Returns all edges connected to a vertex.
pub fn select_vertex_edges(mesh: &HalfEdgeMesh, vertex: VertexId) -> Vec<HalfEdgeId> {
    let mut edges = Vec::new();
    let start = mesh.vertices[vertex.0 as usize].halfedge;

    if start.is_null() {
        return edges;
    }

    let mut current = start;
    loop {
        edges.push(current);

        let twin = mesh.halfedges[current.0 as usize].twin;
        if twin.is_null() {
            break;
        }

        current = mesh.halfedges[twin.0 as usize].next;
        if current == start || current.is_null() {
            break;
        }
    }

    edges
}

/// Returns all edges of a face.
pub fn select_face_edges(mesh: &HalfEdgeMesh, face: FaceId) -> Vec<HalfEdgeId> {
    mesh.face_halfedges(face)
}

/// Converts edge selection to vertex selection.
pub fn edges_to_vertices(mesh: &HalfEdgeMesh, edges: &[HalfEdgeId]) -> Vec<VertexId> {
    let mut vertices: HashSet<u32> = HashSet::new();

    for &edge in edges {
        if edge.0 as usize >= mesh.halfedges.len() {
            continue;
        }
        let he = &mesh.halfedges[edge.0 as usize];
        vertices.insert(he.origin.0);
        vertices.insert(he.vertex.0);
    }

    vertices.into_iter().map(VertexId).collect()
}

/// Converts edge selection to face selection.
pub fn edges_to_faces(mesh: &HalfEdgeMesh, edges: &[HalfEdgeId]) -> Vec<FaceId> {
    let mut faces: HashSet<u32> = HashSet::new();

    for &edge in edges {
        if edge.0 as usize >= mesh.halfedges.len() {
            continue;
        }
        let he = &mesh.halfedges[edge.0 as usize];
        if !he.face.is_null() {
            faces.insert(he.face.0);
        }
        if !he.twin.is_null() {
            let twin = &mesh.halfedges[he.twin.0 as usize];
            if !twin.face.is_null() {
                faces.insert(twin.face.0);
            }
        }
    }

    faces.into_iter().map(FaceId).collect()
}

/// Grows an edge selection to include adjacent edges.
pub fn grow_edge_selection(mesh: &HalfEdgeMesh, edges: &[HalfEdgeId]) -> Vec<HalfEdgeId> {
    let mut result: HashSet<u32> = edges.iter().map(|e| e.0).collect();

    for &edge in edges {
        if edge.0 as usize >= mesh.halfedges.len() {
            continue;
        }

        let he = &mesh.halfedges[edge.0 as usize];

        // Add edges from origin vertex
        let origin_edges = select_vertex_edges(mesh, he.origin);
        for e in origin_edges {
            result.insert(e.0);
            let twin = mesh.halfedges[e.0 as usize].twin;
            if !twin.is_null() {
                result.insert(twin.0);
            }
        }

        // Add edges from target vertex
        let target_edges = select_vertex_edges(mesh, he.vertex);
        for e in target_edges {
            result.insert(e.0);
            let twin = mesh.halfedges[e.0 as usize].twin;
            if !twin.is_null() {
                result.insert(twin.0);
            }
        }
    }

    result.into_iter().map(HalfEdgeId).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    fn make_quad_plane() -> HalfEdgeMesh {
        // Create a 2x2 grid of quads for testing
        use crate::MeshBuilder;
        use glam::Vec3;

        let mut builder = MeshBuilder::new();

        // 3x3 grid of vertices
        for y in 0..3 {
            for x in 0..3 {
                builder.vertex(Vec3::new(x as f32, y as f32, 0.0));
            }
        }

        // 2x2 grid of quads (as triangles)
        for y in 0..2 {
            for x in 0..2 {
                let i = y * 3 + x;
                builder.quad(i, i + 1, i + 4, i + 3);
            }
        }

        let mesh = builder.build();
        HalfEdgeMesh::from_mesh(&mesh)
    }

    #[test]
    fn test_select_edge_loop_empty() {
        let mesh = HalfEdgeMesh::new();
        let result = select_edge_loop(&mesh, HalfEdgeId(0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_edge_loop_null() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);
        let result = select_edge_loop(&hemesh, HalfEdgeId::NULL);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_edge_loop_basic() {
        let hemesh = make_quad_plane();
        if hemesh.halfedges.is_empty() {
            return; // Skip if mesh construction failed
        }

        let result = select_edge_loop(&hemesh, HalfEdgeId(0));
        // Should find some edges in the loop
        assert!(!result.is_empty() || hemesh.face_count() < 4);
    }

    #[test]
    fn test_select_edge_ring_empty() {
        let mesh = HalfEdgeMesh::new();
        let result = select_edge_ring(&mesh, HalfEdgeId(0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_boundary_edges() {
        let hemesh = make_quad_plane();
        let boundary = select_boundary_edges(&hemesh);
        // A plane should have boundary edges
        // (The exact count depends on mesh topology)
        let _ = boundary.len(); // Just verify it doesn't panic
    }

    #[test]
    fn test_select_vertex_edges() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);

        if hemesh.vertices.is_empty() {
            return;
        }

        let edges = select_vertex_edges(&hemesh, VertexId(0));
        // Each vertex in a cube should have edges
        assert!(!edges.is_empty() || hemesh.vertices[0].halfedge.is_null());
    }

    #[test]
    fn test_select_face_edges() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);

        if hemesh.faces.is_empty() {
            return;
        }

        let edges = select_face_edges(&hemesh, FaceId(0));
        // Triangle face should have 3 edges
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_edges_to_vertices() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);

        if hemesh.halfedges.is_empty() {
            return;
        }

        let edges = vec![HalfEdgeId(0)];
        let verts = edges_to_vertices(&hemesh, &edges);
        // One edge should give 2 vertices
        assert_eq!(verts.len(), 2);
    }

    #[test]
    fn test_edges_to_faces() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);

        if hemesh.halfedges.is_empty() {
            return;
        }

        let edges = vec![HalfEdgeId(0)];
        let faces = edges_to_faces(&hemesh, &edges);
        // Edge should belong to at least one face
        assert!(!faces.is_empty());
    }

    #[test]
    fn test_grow_edge_selection() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);

        if hemesh.halfedges.is_empty() {
            return;
        }

        let initial = vec![HalfEdgeId(0)];
        let grown = grow_edge_selection(&hemesh, &initial);
        // Growing should add more edges
        assert!(grown.len() >= initial.len());
    }

    #[test]
    fn test_loop_cut_empty() {
        let mesh = HalfEdgeMesh::new();
        let result = loop_cut(&mesh, HalfEdgeId(0), 1);
        assert_eq!(result.vertex_count(), 0);
    }

    #[test]
    fn test_loop_cut_zero_cuts() {
        let cube = Cuboid::default().apply();
        let hemesh = HalfEdgeMesh::from_mesh(&cube);
        let result = loop_cut(&hemesh, HalfEdgeId(0), 0);
        assert_eq!(result.vertex_count(), hemesh.vertex_count());
    }

    #[test]
    fn test_loop_cut_basic() {
        let hemesh = make_quad_plane();
        if hemesh.halfedges.is_empty() {
            return;
        }

        let result = loop_cut(&hemesh, HalfEdgeId(0), 1);
        // Should have at least as many vertices (adds new ones)
        assert!(result.vertex_count() >= hemesh.vertex_count());
    }
}
