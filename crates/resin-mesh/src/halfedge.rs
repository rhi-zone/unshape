//! Half-edge mesh representation for topology operations.
//!
//! The half-edge data structure enables efficient traversal and manipulation
//! of mesh topology. It's the internal representation used for operations
//! like Catmull-Clark subdivision and beveling.
//!
//! # Structure
//!
//! - Each edge is represented as two half-edges pointing in opposite directions
//! - Half-edges store: next (around face), twin (opposite direction), vertex (target), face
//! - Vertices store: one outgoing half-edge
//! - Faces store: one half-edge on its boundary
//!
//! # Usage
//!
//! ```ignore
//! // Convert from indexed mesh
//! let hemesh = HalfEdgeMesh::from_mesh(&mesh);
//!
//! // Perform operations
//! let subdivided = hemesh.catmull_clark();
//!
//! // Convert back to indexed mesh
//! let mesh = subdivided.to_mesh();
//! ```

use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// Index for a half-edge in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HalfEdgeId(pub u32);

/// Index for a vertex in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VertexId(pub u32);

/// Index for a face in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FaceId(pub u32);

/// Sentinel value for null references.
const NULL_ID: u32 = u32::MAX;

impl HalfEdgeId {
    /// Sentinel value representing no half-edge.
    pub const NULL: HalfEdgeId = HalfEdgeId(NULL_ID);

    /// Returns true if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

impl VertexId {
    /// Sentinel value representing no vertex.
    pub const NULL: VertexId = VertexId(NULL_ID);

    /// Returns true if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

impl FaceId {
    /// Sentinel value representing no face.
    pub const NULL: FaceId = FaceId(NULL_ID);

    /// Returns true if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

/// A half-edge in the mesh.
///
/// Each edge is represented as two half-edges pointing in opposite directions.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfEdge {
    /// Next half-edge around the face (counter-clockwise).
    pub next: HalfEdgeId,
    /// Previous half-edge around the face.
    pub prev: HalfEdgeId,
    /// Twin half-edge (opposite direction).
    pub twin: HalfEdgeId,
    /// Vertex this half-edge points to (target).
    pub vertex: VertexId,
    /// Vertex this half-edge comes from (source).
    pub origin: VertexId,
    /// Face this half-edge belongs to (NULL for boundary edges).
    pub face: FaceId,
}

/// A vertex in the mesh.
#[derive(Debug, Clone, Default)]
pub struct Vertex {
    /// Position in 3D space.
    pub position: Vec3,
    /// Normal vector.
    pub normal: Vec3,
    /// Texture coordinates.
    pub uv: Vec2,
    /// One outgoing half-edge from this vertex.
    pub halfedge: HalfEdgeId,
}

/// A face in the mesh.
#[derive(Debug, Clone, Default)]
pub struct Face {
    /// One half-edge on this face's boundary.
    pub halfedge: HalfEdgeId,
}

/// Half-edge mesh data structure.
///
/// Enables efficient topology traversal and modification for operations
/// like subdivision and beveling.
#[derive(Debug, Clone, Default)]
pub struct HalfEdgeMesh {
    /// All half-edges.
    pub halfedges: Vec<HalfEdge>,
    /// All vertices.
    pub vertices: Vec<Vertex>,
    /// All faces.
    pub faces: Vec<Face>,
}

impl HalfEdgeMesh {
    /// Creates an empty half-edge mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs a half-edge mesh from an indexed triangle mesh.
    pub fn from_mesh(mesh: &crate::Mesh) -> Self {
        let mut hemesh = HalfEdgeMesh::new();

        // Create vertices
        for i in 0..mesh.positions.len() {
            hemesh.vertices.push(Vertex {
                position: mesh.positions[i],
                normal: mesh.normals.get(i).copied().unwrap_or(Vec3::ZERO),
                uv: mesh.uvs.get(i).copied().unwrap_or(Vec2::ZERO),
                halfedge: HalfEdgeId::NULL,
            });
        }

        // Map from directed edge (from, to) -> half-edge index
        let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();

        // Create faces and half-edges
        for tri in mesh.indices.chunks(3) {
            let face_id = FaceId(hemesh.faces.len() as u32);
            hemesh.faces.push(Face {
                halfedge: HalfEdgeId::NULL,
            });

            let indices = [tri[0], tri[1], tri[2]];
            let mut face_halfedges = [HalfEdgeId::NULL; 3];

            // Create half-edges for this face
            for i in 0..3 {
                let from = indices[i];
                let to = indices[(i + 1) % 3];

                let he_id = HalfEdgeId(hemesh.halfedges.len() as u32);
                face_halfedges[i] = he_id;

                hemesh.halfedges.push(HalfEdge {
                    next: HalfEdgeId::NULL,
                    prev: HalfEdgeId::NULL,
                    twin: HalfEdgeId::NULL,
                    vertex: VertexId(to),
                    origin: VertexId(from),
                    face: face_id,
                });

                edge_map.insert((from, to), he_id);

                // Set vertex outgoing half-edge if not set
                if hemesh.vertices[from as usize].halfedge.is_null() {
                    hemesh.vertices[from as usize].halfedge = he_id;
                }
            }

            // Link next/prev pointers within face
            for i in 0..3 {
                let he_id = face_halfedges[i];
                let next_id = face_halfedges[(i + 1) % 3];
                let prev_id = face_halfedges[(i + 2) % 3];

                hemesh.halfedges[he_id.0 as usize].next = next_id;
                hemesh.halfedges[he_id.0 as usize].prev = prev_id;
            }

            // Set face's half-edge
            hemesh.faces[face_id.0 as usize].halfedge = face_halfedges[0];
        }

        // Link twin pointers
        for (&(from, to), &he_id) in &edge_map {
            if let Some(&twin_id) = edge_map.get(&(to, from)) {
                hemesh.halfedges[he_id.0 as usize].twin = twin_id;
            }
        }

        // Create boundary half-edges for edges without twins
        let boundary_edges: Vec<_> = edge_map
            .iter()
            .filter(|&(&(from, to), &he_id)| {
                hemesh.halfedges[he_id.0 as usize].twin.is_null()
                    && !edge_map.contains_key(&(to, from))
            })
            .map(|(&(from, to), &he_id)| (from, to, he_id))
            .collect();

        for (from, to, he_id) in boundary_edges {
            let boundary_he_id = HalfEdgeId(hemesh.halfedges.len() as u32);

            // Boundary half-edge goes in opposite direction: to -> from
            hemesh.halfedges.push(HalfEdge {
                next: HalfEdgeId::NULL,
                prev: HalfEdgeId::NULL,
                twin: he_id,
                vertex: VertexId(from),
                origin: VertexId(to),
                face: FaceId::NULL,
            });

            hemesh.halfedges[he_id.0 as usize].twin = boundary_he_id;
        }

        // Link boundary half-edges
        hemesh.link_boundary_edges();

        hemesh
    }

    /// Links boundary half-edges into loops.
    fn link_boundary_edges(&mut self) {
        // Collect boundary half-edges by their source vertex
        let mut boundary_by_source: HashMap<u32, HalfEdgeId> = HashMap::new();

        for (i, he) in self.halfedges.iter().enumerate() {
            if he.face.is_null() {
                // Boundary edge: source is the twin's vertex
                let source = self.halfedges[he.twin.0 as usize].vertex.0;
                boundary_by_source.insert(source, HalfEdgeId(i as u32));
            }
        }

        // Collect links to apply
        let mut links: Vec<(usize, HalfEdgeId)> = Vec::new();
        for (i, he) in self.halfedges.iter().enumerate() {
            if he.face.is_null() {
                let target = he.vertex.0;
                if let Some(&next_id) = boundary_by_source.get(&target) {
                    links.push((i, next_id));
                }
            }
        }

        // Apply links
        for (i, next_id) in links {
            self.halfedges[i].next = next_id;
            self.halfedges[next_id.0 as usize].prev = HalfEdgeId(i as u32);
        }
    }

    /// Converts the half-edge mesh back to an indexed triangle mesh.
    ///
    /// Triangulates any non-triangle faces using fan triangulation.
    pub fn to_mesh(&self) -> crate::Mesh {
        use crate::Mesh;

        let mut mesh = Mesh::new();

        // Copy vertices
        for v in &self.vertices {
            mesh.positions.push(v.position);
            mesh.normals.push(v.normal);
            mesh.uvs.push(v.uv);
        }

        // Triangulate each face
        for i in 0..self.faces.len() {
            let verts = self.face_vertices(FaceId(i as u32));

            if verts.len() < 3 {
                continue;
            }

            // Fan triangulation
            let v0 = verts[0].0;
            for j in 1..verts.len() - 1 {
                mesh.indices.push(v0);
                mesh.indices.push(verts[j].0);
                mesh.indices.push(verts[j + 1].0);
            }
        }

        mesh
    }

    // ==================== Topology Queries ====================

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of faces.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Returns the number of half-edges.
    pub fn halfedge_count(&self) -> usize {
        self.halfedges.len()
    }

    /// Returns the number of edges (half-edges / 2).
    pub fn edge_count(&self) -> usize {
        self.halfedges.len() / 2
    }

    /// Returns the vertex at the start of a half-edge.
    pub fn halfedge_source(&self, he: HalfEdgeId) -> VertexId {
        self.halfedges[he.0 as usize].origin
    }

    /// Returns the vertex at the end of a half-edge.
    pub fn halfedge_target(&self, he: HalfEdgeId) -> VertexId {
        self.halfedges[he.0 as usize].vertex
    }

    /// Returns all vertices of a face in order.
    pub fn face_vertices(&self, face: FaceId) -> Vec<VertexId> {
        let mut result = Vec::new();
        let start = self.faces[face.0 as usize].halfedge;

        if start.is_null() {
            return result;
        }

        let mut current = start;
        loop {
            result.push(self.halfedges[current.0 as usize].vertex);
            current = self.halfedges[current.0 as usize].next;
            if current == start || current.is_null() {
                break;
            }
        }

        result
    }

    /// Returns all half-edges around a face.
    pub fn face_halfedges(&self, face: FaceId) -> Vec<HalfEdgeId> {
        let mut result = Vec::new();
        let start = self.faces[face.0 as usize].halfedge;

        if start.is_null() {
            return result;
        }

        let mut current = start;
        loop {
            result.push(current);
            current = self.halfedges[current.0 as usize].next;
            if current == start || current.is_null() {
                break;
            }
        }

        result
    }

    /// Returns all faces adjacent to a vertex.
    pub fn vertex_faces(&self, vertex: VertexId) -> Vec<FaceId> {
        let mut result = Vec::new();
        let start = self.vertices[vertex.0 as usize].halfedge;

        if start.is_null() {
            return result;
        }

        let mut current = start;
        loop {
            let face = self.halfedges[current.0 as usize].face;
            if !face.is_null() {
                result.push(face);
            }
            // Move to the next outgoing half-edge via twin.next
            let twin = self.halfedges[current.0 as usize].twin;
            if twin.is_null() {
                break;
            }
            current = self.halfedges[twin.0 as usize].next;
            if current == start || current.is_null() {
                break;
            }
        }

        result
    }

    /// Returns all vertices adjacent to a vertex (one-ring neighborhood).
    pub fn vertex_neighbors(&self, vertex: VertexId) -> Vec<VertexId> {
        let mut result = Vec::new();
        let start = self.vertices[vertex.0 as usize].halfedge;

        if start.is_null() {
            return result;
        }

        let mut current = start;
        loop {
            result.push(self.halfedges[current.0 as usize].vertex);
            let twin = self.halfedges[current.0 as usize].twin;
            if twin.is_null() {
                break;
            }
            current = self.halfedges[twin.0 as usize].next;
            if current == start || current.is_null() {
                break;
            }
        }

        result
    }

    /// Returns the valence (number of edges) of a vertex.
    pub fn vertex_valence(&self, vertex: VertexId) -> usize {
        self.vertex_neighbors(vertex).len()
    }

    /// Returns the number of sides of a face.
    pub fn face_sides(&self, face: FaceId) -> usize {
        self.face_vertices(face).len()
    }

    /// Returns true if the vertex is on a boundary.
    pub fn is_boundary_vertex(&self, vertex: VertexId) -> bool {
        let start = self.vertices[vertex.0 as usize].halfedge;
        if start.is_null() {
            return false;
        }

        let mut current = start;
        loop {
            let twin = self.halfedges[current.0 as usize].twin;
            if twin.is_null() || self.halfedges[twin.0 as usize].face.is_null() {
                return true;
            }
            current = self.halfedges[twin.0 as usize].next;
            if current == start || current.is_null() {
                break;
            }
        }

        false
    }

    /// Returns true if the half-edge is on a boundary.
    pub fn is_boundary_edge(&self, he: HalfEdgeId) -> bool {
        let edge = &self.halfedges[he.0 as usize];
        edge.face.is_null()
            || edge.twin.is_null()
            || self.halfedges[edge.twin.0 as usize].face.is_null()
    }

    /// Computes the centroid of a face.
    pub fn face_centroid(&self, face: FaceId) -> Vec3 {
        let verts = self.face_vertices(face);
        if verts.is_empty() {
            return Vec3::ZERO;
        }

        let sum: Vec3 = verts
            .iter()
            .map(|v| self.vertices[v.0 as usize].position)
            .sum();
        sum / verts.len() as f32
    }

    /// Computes the midpoint of an edge.
    pub fn edge_midpoint(&self, he: HalfEdgeId) -> Vec3 {
        let v0 = self.halfedge_source(he);
        let v1 = self.halfedge_target(he);
        (self.vertices[v0.0 as usize].position + self.vertices[v1.0 as usize].position) * 0.5
    }

    // ==================== Catmull-Clark Subdivision ====================

    /// Performs Catmull-Clark subdivision.
    ///
    /// This subdivision scheme works on arbitrary polygon meshes and produces
    /// a smooth limit surface. After one iteration, all faces become quads.
    pub fn catmull_clark(&self) -> HalfEdgeMesh {
        // Step 1: Compute face points (centroid of each face)
        let face_points: Vec<Vec3> = (0..self.faces.len())
            .map(|i| self.face_centroid(FaceId(i as u32)))
            .collect();

        // Step 2: Compute edge points
        // For interior edges: average of edge midpoint and adjacent face centroids
        // For boundary edges: just the midpoint
        let mut edge_points: HashMap<(u32, u32), Vec3> = HashMap::new();
        let mut processed_edges: HashMap<(u32, u32), bool> = HashMap::new();

        for (i, he) in self.halfedges.iter().enumerate() {
            if he.face.is_null() {
                continue;
            }

            let v0 = self.halfedge_source(HalfEdgeId(i as u32));
            let v1 = he.vertex;
            let key = if v0.0 < v1.0 {
                (v0.0, v1.0)
            } else {
                (v1.0, v0.0)
            };

            if processed_edges.contains_key(&key) {
                continue;
            }
            processed_edges.insert(key, true);

            let midpoint = self.edge_midpoint(HalfEdgeId(i as u32));

            if self.is_boundary_edge(HalfEdgeId(i as u32)) {
                edge_points.insert(key, midpoint);
            } else {
                // Average of midpoint and two adjacent face centroids
                let f0 = he.face;
                let f1 = self.halfedges[he.twin.0 as usize].face;
                let fp0 = face_points[f0.0 as usize];
                let fp1 = face_points[f1.0 as usize];
                edge_points.insert(key, (midpoint + fp0 + fp1) / 3.0);
            }
        }

        // Step 3: Compute new vertex positions
        let new_vertex_positions: Vec<Vec3> = (0..self.vertices.len())
            .map(|i| {
                let vid = VertexId(i as u32);
                let v = &self.vertices[i];

                if self.is_boundary_vertex(vid) {
                    // Boundary vertex: average with adjacent boundary edge midpoints
                    let neighbors = self.vertex_neighbors(vid);
                    let mut boundary_mids = Vec::new();

                    for n in &neighbors {
                        let key = if vid.0 < n.0 {
                            (vid.0, n.0)
                        } else {
                            (n.0, vid.0)
                        };
                        if let Some(&ep) = edge_points.get(&key) {
                            // Check if this is a boundary edge
                            let start = self.vertices[vid.0 as usize].halfedge;
                            let mut current = start;
                            let mut is_boundary = false;
                            loop {
                                if self.halfedges[current.0 as usize].vertex == *n {
                                    is_boundary = self.is_boundary_edge(current);
                                    break;
                                }
                                let twin = self.halfedges[current.0 as usize].twin;
                                if twin.is_null() {
                                    break;
                                }
                                current = self.halfedges[twin.0 as usize].next;
                                if current == start || current.is_null() {
                                    break;
                                }
                            }
                            if is_boundary {
                                boundary_mids.push(ep);
                            }
                        }
                    }

                    if boundary_mids.len() == 2 {
                        (boundary_mids[0] + boundary_mids[1] + v.position * 2.0) / 4.0
                    } else {
                        v.position
                    }
                } else {
                    // Interior vertex
                    let faces = self.vertex_faces(vid);
                    let neighbors = self.vertex_neighbors(vid);
                    let n = neighbors.len() as f32;

                    if n < 3.0 {
                        return v.position;
                    }

                    // F = average of face points
                    let f: Vec3 = faces
                        .iter()
                        .map(|f| face_points[f.0 as usize])
                        .sum::<Vec3>()
                        / faces.len() as f32;

                    // R = average of edge midpoints
                    let r: Vec3 = neighbors
                        .iter()
                        .map(|neighbor| {
                            let mid =
                                (v.position + self.vertices[neighbor.0 as usize].position) * 0.5;
                            mid
                        })
                        .sum::<Vec3>()
                        / n;

                    // New position = (F + 2R + (n-3)P) / n
                    (f + r * 2.0 + v.position * (n - 3.0)) / n
                }
            })
            .collect();

        // Step 4: Build the new mesh
        let mut result = HalfEdgeMesh::new();

        // Add original vertices with new positions
        for (i, new_pos) in new_vertex_positions.iter().enumerate() {
            result.vertices.push(Vertex {
                position: *new_pos,
                normal: Vec3::ZERO,
                uv: self.vertices[i].uv,
                halfedge: HalfEdgeId::NULL,
            });
        }

        // Add face points as vertices
        let face_vertex_start = result.vertices.len();
        for fp in &face_points {
            result.vertices.push(Vertex {
                position: *fp,
                normal: Vec3::ZERO,
                uv: Vec2::ZERO,
                halfedge: HalfEdgeId::NULL,
            });
        }

        // Add edge points as vertices
        let mut edge_vertex_map: HashMap<(u32, u32), u32> = HashMap::new();
        for (key, ep) in &edge_points {
            edge_vertex_map.insert(*key, result.vertices.len() as u32);
            result.vertices.push(Vertex {
                position: *ep,
                normal: Vec3::ZERO,
                uv: Vec2::ZERO,
                halfedge: HalfEdgeId::NULL,
            });
        }

        // Create quads for each original face
        // Each face vertex connects to edge points and original vertices
        for (face_idx, _face) in self.faces.iter().enumerate() {
            let face_vertex = (face_vertex_start + face_idx) as u32;
            let halfedges = self.face_halfedges(FaceId(face_idx as u32));

            for he_id in halfedges {
                let he = &self.halfedges[he_id.0 as usize];
                let v_curr = he.vertex;
                let v_prev = self.halfedge_source(he_id);

                let edge_key1 = if v_prev.0 < v_curr.0 {
                    (v_prev.0, v_curr.0)
                } else {
                    (v_curr.0, v_prev.0)
                };

                let next_he = &self.halfedges[he.next.0 as usize];
                let v_next = next_he.vertex;
                let edge_key2 = if v_curr.0 < v_next.0 {
                    (v_curr.0, v_next.0)
                } else {
                    (v_next.0, v_curr.0)
                };

                let ep1 = *edge_vertex_map.get(&edge_key1).unwrap();
                let ep2 = *edge_vertex_map.get(&edge_key2).unwrap();

                // Create quad: face_point -> edge_point1 -> vertex -> edge_point2
                let quad_verts = [face_vertex, ep1, v_curr.0, ep2];
                let new_face_id = FaceId(result.faces.len() as u32);
                result.faces.push(Face {
                    halfedge: HalfEdgeId::NULL,
                });

                let mut quad_hes = [HalfEdgeId::NULL; 4];
                for i in 0..4 {
                    let he_idx = HalfEdgeId(result.halfedges.len() as u32);
                    quad_hes[i] = he_idx;
                    result.halfedges.push(HalfEdge {
                        next: HalfEdgeId::NULL,
                        prev: HalfEdgeId::NULL,
                        twin: HalfEdgeId::NULL,
                        vertex: VertexId(quad_verts[(i + 1) % 4]),
                        origin: VertexId(quad_verts[i]),
                        face: new_face_id,
                    });
                }

                // Link within quad
                for i in 0..4 {
                    result.halfedges[quad_hes[i].0 as usize].next = quad_hes[(i + 1) % 4];
                    result.halfedges[quad_hes[i].0 as usize].prev = quad_hes[(i + 3) % 4];
                }

                result.faces[new_face_id.0 as usize].halfedge = quad_hes[0];

                // Set vertex halfedges
                for i in 0..4 {
                    let vid = VertexId(quad_verts[i]);
                    if result.vertices[vid.0 as usize].halfedge.is_null() {
                        result.vertices[vid.0 as usize].halfedge = quad_hes[i];
                    }
                }
            }
        }

        // Link twin pointers
        result.link_twins();

        // Compute normals
        result.compute_normals();

        result
    }

    /// Links twin pointers by finding matching half-edges.
    fn link_twins(&mut self) {
        let mut edge_map: HashMap<(u32, u32), HalfEdgeId> = HashMap::new();

        for (i, he) in self.halfedges.iter().enumerate() {
            let source = self.halfedge_source(HalfEdgeId(i as u32));
            let target = he.vertex;
            edge_map.insert((source.0, target.0), HalfEdgeId(i as u32));
        }

        for i in 0..self.halfedges.len() {
            if !self.halfedges[i].twin.is_null() {
                continue;
            }

            let source = self.halfedge_source(HalfEdgeId(i as u32));
            let target = self.halfedges[i].vertex;

            if let Some(&twin) = edge_map.get(&(target.0, source.0)) {
                self.halfedges[i].twin = twin;
                self.halfedges[twin.0 as usize].twin = HalfEdgeId(i as u32);
            }
        }
    }

    /// Computes vertex normals from face geometry.
    pub fn compute_normals(&mut self) {
        // Reset normals
        for v in &mut self.vertices {
            v.normal = Vec3::ZERO;
        }

        // Accumulate face normals
        for i in 0..self.faces.len() {
            let verts = self.face_vertices(FaceId(i as u32));
            if verts.len() < 3 {
                continue;
            }

            // Compute face normal using first three vertices
            let p0 = self.vertices[verts[0].0 as usize].position;
            let p1 = self.vertices[verts[1].0 as usize].position;
            let p2 = self.vertices[verts[2].0 as usize].position;
            let normal = (p1 - p0).cross(p2 - p0);

            // Add to each vertex
            for v in verts {
                self.vertices[v.0 as usize].normal += normal;
            }
        }

        // Normalize
        for v in &mut self.vertices {
            v.normal = v.normal.normalize_or_zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mesh, MeshBuilder};
    use glam::Vec3;

    fn triangle_mesh() -> Mesh {
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(0.5, 1.0, 0.0));
        builder.triangle(v0, v1, v2);
        builder.build()
    }

    fn quad_mesh() -> Mesh {
        let mut builder = MeshBuilder::new();
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));
        builder.quad(v0, v1, v2, v3);
        builder.build()
    }

    fn cube_mesh() -> Mesh {
        crate::Cuboid::default().apply()
    }

    #[test]
    fn test_from_mesh_triangle() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        assert_eq!(hemesh.vertex_count(), 3);
        assert_eq!(hemesh.face_count(), 1);
        // Triangle has 3 interior + 3 boundary = 6 half-edges
        assert_eq!(hemesh.halfedge_count(), 6);
    }

    #[test]
    fn test_from_mesh_quad() {
        let mesh = quad_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        assert_eq!(hemesh.vertex_count(), 4);
        assert_eq!(hemesh.face_count(), 2); // Quad becomes 2 triangles
    }

    #[test]
    fn test_face_vertices() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        let verts = hemesh.face_vertices(FaceId(0));
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn test_vertex_neighbors() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        // Each vertex in a triangle has 2 neighbors
        for i in 0..3 {
            let neighbors = hemesh.vertex_neighbors(VertexId(i));
            assert_eq!(neighbors.len(), 2);
        }
    }

    #[test]
    fn test_vertex_valence() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        for i in 0..3 {
            assert_eq!(hemesh.vertex_valence(VertexId(i)), 2);
        }
    }

    #[test]
    fn test_face_centroid() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        let centroid = hemesh.face_centroid(FaceId(0));
        assert!((centroid.x - 0.5).abs() < 0.01);
        assert!((centroid.y - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_is_boundary_vertex() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        // All vertices in a single triangle are boundary vertices
        for i in 0..3 {
            assert!(hemesh.is_boundary_vertex(VertexId(i)));
        }
    }

    #[test]
    fn test_cube_interior_vertices() {
        let mesh = cube_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);

        // Cube vertices depend on how box_mesh constructs it
        // With proper shared vertices, interior vertices would exist
        // but box_mesh creates separate vertices per face for sharp edges
        assert!(hemesh.vertex_count() > 0);
        assert!(hemesh.face_count() > 0);
    }

    #[test]
    fn test_catmull_clark_triangle() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);
        let subdivided = hemesh.catmull_clark();

        // After Catmull-Clark, all faces should be quads
        // Original triangle becomes 3 quads (one per original vertex)
        assert_eq!(subdivided.face_count(), 3);

        // Each face should have 4 vertices
        for i in 0..subdivided.face_count() {
            assert_eq!(subdivided.face_sides(FaceId(i as u32)), 4);
        }
    }

    #[test]
    fn test_catmull_clark_quad() {
        let mesh = quad_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);
        let subdivided = hemesh.catmull_clark();

        // Each original triangle becomes 3 quads, so 2 triangles -> 6 quads
        assert_eq!(subdivided.face_count(), 6);
    }

    #[test]
    fn test_to_mesh() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);
        let back = hemesh.to_mesh();

        assert_eq!(back.vertex_count(), 3);
        assert_eq!(back.triangle_count(), 1);
    }

    #[test]
    fn test_catmull_clark_roundtrip() {
        let mesh = triangle_mesh();
        let hemesh = HalfEdgeMesh::from_mesh(&mesh);
        let subdivided = hemesh.catmull_clark();
        let final_mesh = subdivided.to_mesh();

        // Should have vertices and triangles
        assert!(final_mesh.vertex_count() > 0);
        assert!(final_mesh.triangle_count() > 0);
    }
}
