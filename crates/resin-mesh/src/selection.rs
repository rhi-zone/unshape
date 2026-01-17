//! Mesh selection system for constructive modeling.
//!
//! Provides selection of vertices, edges, and faces for use with editing operations.
//! Selections are stored as index sets and can be manipulated with various operations.
//!
//! # Usage
//!
//! ```ignore
//! use resin_mesh::{Mesh, MeshSelection, SelectionMode};
//!
//! let mesh = Mesh::cube(1.0);
//! let mut selection = MeshSelection::new();
//!
//! // Select all faces
//! selection.select_all_faces(&mesh);
//!
//! // Or select by trait
//! selection.select_faces_by_normal(&mesh, Vec3::Y, 0.9);
//!
//! // Expand selection to adjacent elements
//! selection.grow_faces(&mesh);
//! ```

use crate::Mesh;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Represents an edge as a sorted pair of vertex indices.
///
/// Edges are stored with the smaller index first to ensure consistent hashing
/// regardless of traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge(pub u32, pub u32);

impl Edge {
    /// Creates an edge from two vertex indices.
    ///
    /// The indices are automatically sorted to ensure consistent representation.
    pub fn new(a: u32, b: u32) -> Self {
        if a < b { Edge(a, b) } else { Edge(b, a) }
    }

    /// Returns the vertex indices as a tuple.
    pub fn vertices(&self) -> (u32, u32) {
        (self.0, self.1)
    }
}

/// Selection mode hint for UI/editing context.
///
/// This indicates the "active" selection mode for user interaction,
/// but doesn't restrict what can be selected - you can have vertices,
/// edges, and faces selected simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMode {
    /// Vertex selection mode.
    #[default]
    Vertex,
    /// Edge selection mode.
    Edge,
    /// Face selection mode.
    Face,
}

/// Mesh selection storing selected vertices, edges, and faces.
///
/// Selections are stored as index sets. Vertices and faces use simple u32 indices,
/// while edges use sorted vertex pairs for consistent identification.
#[derive(Debug, Clone, Default)]
pub struct MeshSelection {
    /// Selected vertex indices.
    pub vertices: HashSet<u32>,
    /// Selected edges (as sorted vertex pairs).
    pub edges: HashSet<Edge>,
    /// Selected face indices (triangle index / 3).
    pub faces: HashSet<u32>,
    /// Current selection mode hint.
    pub mode: SelectionMode,
}

impl MeshSelection {
    /// Creates an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a selection with the given mode.
    pub fn with_mode(mode: SelectionMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Returns true if the selection is empty (no vertices, edges, or faces selected).
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.edges.is_empty() && self.faces.is_empty()
    }

    /// Clears all selections.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
        self.faces.clear();
    }

    /// Clears vertex selection.
    pub fn clear_vertices(&mut self) {
        self.vertices.clear();
    }

    /// Clears edge selection.
    pub fn clear_edges(&mut self) {
        self.edges.clear();
    }

    /// Clears face selection.
    pub fn clear_faces(&mut self) {
        self.faces.clear();
    }

    // ========================================================================
    // Vertex selection
    // ========================================================================

    /// Selects a single vertex.
    pub fn select_vertex(&mut self, index: u32) {
        self.vertices.insert(index);
    }

    /// Deselects a single vertex.
    pub fn deselect_vertex(&mut self, index: u32) {
        self.vertices.remove(&index);
    }

    /// Toggles selection of a single vertex.
    pub fn toggle_vertex(&mut self, index: u32) {
        if self.vertices.contains(&index) {
            self.vertices.remove(&index);
        } else {
            self.vertices.insert(index);
        }
    }

    /// Returns true if a vertex is selected.
    pub fn is_vertex_selected(&self, index: u32) -> bool {
        self.vertices.contains(&index)
    }

    /// Selects all vertices in the mesh.
    pub fn select_all_vertices(&mut self, mesh: &Mesh) {
        self.vertices = (0..mesh.vertex_count() as u32).collect();
    }

    /// Inverts vertex selection.
    pub fn invert_vertices(&mut self, mesh: &Mesh) {
        let all: HashSet<u32> = (0..mesh.vertex_count() as u32).collect();
        self.vertices = all.difference(&self.vertices).copied().collect();
    }

    // ========================================================================
    // Edge selection
    // ========================================================================

    /// Selects an edge by its vertex indices.
    pub fn select_edge(&mut self, a: u32, b: u32) {
        self.edges.insert(Edge::new(a, b));
    }

    /// Deselects an edge by its vertex indices.
    pub fn deselect_edge(&mut self, a: u32, b: u32) {
        self.edges.remove(&Edge::new(a, b));
    }

    /// Toggles selection of an edge.
    pub fn toggle_edge(&mut self, a: u32, b: u32) {
        let edge = Edge::new(a, b);
        if self.edges.contains(&edge) {
            self.edges.remove(&edge);
        } else {
            self.edges.insert(edge);
        }
    }

    /// Returns true if an edge is selected.
    pub fn is_edge_selected(&self, a: u32, b: u32) -> bool {
        self.edges.contains(&Edge::new(a, b))
    }

    /// Selects all edges in the mesh.
    pub fn select_all_edges(&mut self, mesh: &Mesh) {
        self.edges.clear();
        for face_idx in 0..mesh.triangle_count() {
            let base = face_idx * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];

            self.edges.insert(Edge::new(i0, i1));
            self.edges.insert(Edge::new(i1, i2));
            self.edges.insert(Edge::new(i2, i0));
        }
    }

    /// Inverts edge selection.
    pub fn invert_edges(&mut self, mesh: &Mesh) {
        let mut all_edges = HashSet::new();
        for face_idx in 0..mesh.triangle_count() {
            let base = face_idx * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];

            all_edges.insert(Edge::new(i0, i1));
            all_edges.insert(Edge::new(i1, i2));
            all_edges.insert(Edge::new(i2, i0));
        }
        self.edges = all_edges.difference(&self.edges).copied().collect();
    }

    // ========================================================================
    // Face selection
    // ========================================================================

    /// Selects a face by its index.
    pub fn select_face(&mut self, index: u32) {
        self.faces.insert(index);
    }

    /// Deselects a face by its index.
    pub fn deselect_face(&mut self, index: u32) {
        self.faces.remove(&index);
    }

    /// Toggles selection of a face.
    pub fn toggle_face(&mut self, index: u32) {
        if self.faces.contains(&index) {
            self.faces.remove(&index);
        } else {
            self.faces.insert(index);
        }
    }

    /// Returns true if a face is selected.
    pub fn is_face_selected(&self, index: u32) -> bool {
        self.faces.contains(&index)
    }

    /// Selects all faces in the mesh.
    pub fn select_all_faces(&mut self, mesh: &Mesh) {
        self.faces = (0..mesh.triangle_count() as u32).collect();
    }

    /// Inverts face selection.
    pub fn invert_faces(&mut self, mesh: &Mesh) {
        let all: HashSet<u32> = (0..mesh.triangle_count() as u32).collect();
        self.faces = all.difference(&self.faces).copied().collect();
    }

    // ========================================================================
    // Selection expansion/shrinking
    // ========================================================================

    /// Grows vertex selection to include adjacent vertices.
    ///
    /// A vertex is added if it shares an edge with any currently selected vertex.
    pub fn grow_vertices(&mut self, mesh: &Mesh) {
        let adjacency = build_vertex_adjacency(mesh);
        let mut to_add: HashSet<u32> = HashSet::new();

        for &v in &self.vertices {
            if let Some(neighbors) = adjacency.get(&v) {
                to_add.extend(neighbors);
            }
        }

        self.vertices.extend(to_add);
    }

    /// Shrinks vertex selection by removing boundary vertices.
    ///
    /// A vertex is removed if any of its neighbors is not selected.
    pub fn shrink_vertices(&mut self, mesh: &Mesh) {
        let adjacency = build_vertex_adjacency(mesh);
        let mut to_remove = HashSet::new();

        for &v in &self.vertices {
            if let Some(neighbors) = adjacency.get(&v) {
                // If any neighbor is not selected, this is a boundary vertex
                if neighbors.iter().any(|n| !self.vertices.contains(n)) {
                    to_remove.insert(v);
                }
            }
        }

        for v in to_remove {
            self.vertices.remove(&v);
        }
    }

    /// Selects all vertices connected to currently selected vertices.
    ///
    /// Floods through edge connections until no new vertices are found.
    pub fn select_linked_vertices(&mut self, mesh: &Mesh) {
        if self.vertices.is_empty() {
            return;
        }

        let adjacency = build_vertex_adjacency(mesh);
        let mut stack: Vec<u32> = self.vertices.iter().copied().collect();

        while let Some(v) = stack.pop() {
            if let Some(neighbors) = adjacency.get(&v) {
                for &n in neighbors {
                    if self.vertices.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
    }

    /// Grows face selection to include adjacent faces.
    ///
    /// A face is added if it shares an edge with any currently selected face.
    pub fn grow_faces(&mut self, mesh: &Mesh) {
        let adjacency = build_face_adjacency(mesh);
        let mut to_add: HashSet<u32> = HashSet::new();

        for &f in &self.faces {
            if let Some(neighbors) = adjacency.get(&f) {
                to_add.extend(neighbors);
            }
        }

        self.faces.extend(to_add);
    }

    /// Shrinks face selection by removing boundary faces.
    ///
    /// A face is removed if any of its edge-adjacent neighbors is not selected.
    pub fn shrink_faces(&mut self, mesh: &Mesh) {
        let adjacency = build_face_adjacency(mesh);
        let mut to_remove = HashSet::new();

        for &f in &self.faces {
            if let Some(neighbors) = adjacency.get(&f) {
                // A face at the boundary has fewer than 3 selected neighbors
                // or has any unselected neighbor
                if neighbors.iter().any(|n| !self.faces.contains(n)) {
                    to_remove.insert(f);
                }
            } else {
                // No adjacency info = isolated face, remove it
                to_remove.insert(f);
            }
        }

        for f in to_remove {
            self.faces.remove(&f);
        }
    }

    /// Selects all faces connected to currently selected faces.
    ///
    /// Floods through edge connections until no new faces are found.
    pub fn select_linked_faces(&mut self, mesh: &Mesh) {
        if self.faces.is_empty() {
            return;
        }

        let adjacency = build_face_adjacency(mesh);
        let mut stack: Vec<u32> = self.faces.iter().copied().collect();

        while let Some(f) = stack.pop() {
            if let Some(neighbors) = adjacency.get(&f) {
                for &n in neighbors {
                    if self.faces.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
    }

    /// Grows edge selection to include adjacent edges.
    ///
    /// An edge is added if it shares a vertex with any currently selected edge.
    pub fn grow_edges(&mut self, mesh: &Mesh) {
        let edge_adjacency = build_edge_adjacency(mesh);
        let mut to_add: HashSet<Edge> = HashSet::new();

        for edge in &self.edges {
            if let Some(neighbors) = edge_adjacency.get(edge) {
                to_add.extend(neighbors.iter().copied());
            }
        }

        self.edges.extend(to_add);
    }

    /// Shrinks edge selection by removing boundary edges.
    pub fn shrink_edges(&mut self, mesh: &Mesh) {
        let edge_adjacency = build_edge_adjacency(mesh);
        let mut to_remove = HashSet::new();

        for edge in &self.edges {
            if let Some(neighbors) = edge_adjacency.get(edge) {
                if neighbors.iter().any(|n| !self.edges.contains(n)) {
                    to_remove.insert(*edge);
                }
            } else {
                to_remove.insert(*edge);
            }
        }

        for e in to_remove {
            self.edges.remove(&e);
        }
    }

    // ========================================================================
    // Selection by trait
    // ========================================================================

    /// Selects faces whose normal is similar to the given direction.
    ///
    /// `threshold` is the minimum dot product (0.0 = perpendicular, 1.0 = exact match).
    pub fn select_faces_by_normal(&mut self, mesh: &Mesh, direction: Vec3, threshold: f32) {
        let direction = direction.normalize_or_zero();

        for face_idx in 0..mesh.triangle_count() {
            let normal = compute_face_normal(mesh, face_idx);
            if normal.dot(direction) >= threshold {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Selects faces larger than the given area threshold.
    pub fn select_faces_by_area_min(&mut self, mesh: &Mesh, min_area: f32) {
        for face_idx in 0..mesh.triangle_count() {
            let area = compute_face_area(mesh, face_idx);
            if area >= min_area {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Selects faces smaller than the given area threshold.
    pub fn select_faces_by_area_max(&mut self, mesh: &Mesh, max_area: f32) {
        for face_idx in 0..mesh.triangle_count() {
            let area = compute_face_area(mesh, face_idx);
            if area <= max_area {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Randomly selects faces with the given probability.
    ///
    /// Uses a simple LCG for deterministic results based on the seed.
    pub fn select_faces_random(&mut self, mesh: &Mesh, probability: f32, seed: u64) {
        let mut rng = SimpleLcg::new(seed);

        for face_idx in 0..mesh.triangle_count() {
            if rng.next_f32() < probability {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Randomly selects vertices with the given probability.
    pub fn select_vertices_random(&mut self, mesh: &Mesh, probability: f32, seed: u64) {
        let mut rng = SimpleLcg::new(seed);

        for v in 0..mesh.vertex_count() {
            if rng.next_f32() < probability {
                self.vertices.insert(v as u32);
            }
        }
    }

    /// Selects boundary edges (edges that belong to only one face).
    pub fn select_boundary_edges(&mut self, mesh: &Mesh) {
        let edge_faces = build_edge_to_faces(mesh);

        for (edge, faces) in edge_faces {
            if faces.len() == 1 {
                self.edges.insert(edge);
            }
        }
    }

    /// Selects boundary vertices (vertices on boundary edges).
    pub fn select_boundary_vertices(&mut self, mesh: &Mesh) {
        let edge_faces = build_edge_to_faces(mesh);

        for (edge, faces) in edge_faces {
            if faces.len() == 1 {
                self.vertices.insert(edge.0);
                self.vertices.insert(edge.1);
            }
        }
    }

    // ========================================================================
    // Conversion between selection types
    // ========================================================================

    /// Converts face selection to vertex selection.
    ///
    /// Selects all vertices that belong to selected faces.
    pub fn faces_to_vertices(&mut self, mesh: &Mesh) {
        for &face_idx in &self.faces {
            let base = face_idx as usize * 3;
            if base + 2 < mesh.indices.len() {
                self.vertices.insert(mesh.indices[base]);
                self.vertices.insert(mesh.indices[base + 1]);
                self.vertices.insert(mesh.indices[base + 2]);
            }
        }
    }

    /// Converts face selection to edge selection.
    ///
    /// Selects all edges that belong to selected faces.
    pub fn faces_to_edges(&mut self, mesh: &Mesh) {
        for &face_idx in &self.faces {
            let base = face_idx as usize * 3;
            if base + 2 < mesh.indices.len() {
                let i0 = mesh.indices[base];
                let i1 = mesh.indices[base + 1];
                let i2 = mesh.indices[base + 2];

                self.edges.insert(Edge::new(i0, i1));
                self.edges.insert(Edge::new(i1, i2));
                self.edges.insert(Edge::new(i2, i0));
            }
        }
    }

    /// Converts edge selection to vertex selection.
    ///
    /// Selects all vertices that belong to selected edges.
    pub fn edges_to_vertices(&mut self) {
        for edge in &self.edges {
            self.vertices.insert(edge.0);
            self.vertices.insert(edge.1);
        }
    }

    /// Converts vertex selection to face selection.
    ///
    /// Selects faces where all vertices are selected.
    pub fn vertices_to_faces_all(&mut self, mesh: &Mesh) {
        for face_idx in 0..mesh.triangle_count() {
            let base = face_idx * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];

            if self.vertices.contains(&i0)
                && self.vertices.contains(&i1)
                && self.vertices.contains(&i2)
            {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Converts vertex selection to face selection (any).
    ///
    /// Selects faces where any vertex is selected.
    pub fn vertices_to_faces_any(&mut self, mesh: &Mesh) {
        for face_idx in 0..mesh.triangle_count() {
            let base = face_idx * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];

            if self.vertices.contains(&i0)
                || self.vertices.contains(&i1)
                || self.vertices.contains(&i2)
            {
                self.faces.insert(face_idx as u32);
            }
        }
    }

    /// Converts vertex selection to edge selection.
    ///
    /// Selects edges where both vertices are selected.
    pub fn vertices_to_edges(&mut self, mesh: &Mesh) {
        for face_idx in 0..mesh.triangle_count() {
            let base = face_idx * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];

            if self.vertices.contains(&i0) && self.vertices.contains(&i1) {
                self.edges.insert(Edge::new(i0, i1));
            }
            if self.vertices.contains(&i1) && self.vertices.contains(&i2) {
                self.edges.insert(Edge::new(i1, i2));
            }
            if self.vertices.contains(&i2) && self.vertices.contains(&i0) {
                self.edges.insert(Edge::new(i2, i0));
            }
        }
    }
}

// ============================================================================
// Soft Selection (Proportional Editing)
// ============================================================================

/// Falloff function for soft selection.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Falloff {
    /// Linear falloff: weight = 1 - (distance / radius).
    Linear,
    /// Smooth falloff: weight = smoothstep(1 - distance / radius).
    #[default]
    Smooth,
    /// Sharp falloff: weight = (1 - distance / radius)^2.
    Sharp,
    /// Root falloff: weight = sqrt(1 - distance / radius).
    Root,
    /// Constant falloff: weight = 1 within radius.
    Constant,
    /// Sphere falloff: weight = sqrt(1 - (distance / radius)^2).
    Sphere,
}

impl Falloff {
    /// Computes the weight for a given normalized distance (0 = center, 1 = edge).
    pub fn weight(&self, normalized_distance: f32) -> f32 {
        if normalized_distance >= 1.0 {
            return 0.0;
        }
        if normalized_distance <= 0.0 {
            return 1.0;
        }

        let t = 1.0 - normalized_distance;
        match self {
            Falloff::Linear => t,
            Falloff::Smooth => t * t * (3.0 - 2.0 * t),
            Falloff::Sharp => t * t,
            Falloff::Root => t.sqrt(),
            Falloff::Constant => 1.0,
            Falloff::Sphere => (1.0 - normalized_distance * normalized_distance).sqrt(),
        }
    }
}

/// Soft selection with falloff weights for smooth transformations.
///
/// Each vertex has a weight from 0.0 (not affected) to 1.0 (fully affected).
/// This enables proportional editing where transformations smoothly fade out.
#[derive(Debug, Clone, Default)]
pub struct SoftSelection {
    /// Weight per vertex (only non-zero weights are stored).
    pub weights: HashMap<u32, f32>,
    /// Falloff radius.
    pub radius: f32,
    /// Falloff function.
    pub falloff: Falloff,
}

impl SoftSelection {
    /// Creates a new soft selection with the given radius.
    pub fn new(radius: f32) -> Self {
        Self {
            weights: HashMap::new(),
            radius,
            falloff: Falloff::default(),
        }
    }

    /// Creates a soft selection with custom falloff.
    pub fn with_falloff(radius: f32, falloff: Falloff) -> Self {
        Self {
            weights: HashMap::new(),
            radius,
            falloff,
        }
    }

    /// Returns true if the soft selection is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Clears all weights.
    pub fn clear(&mut self) {
        self.weights.clear();
    }

    /// Gets the weight for a vertex (0.0 if not in selection).
    pub fn weight(&self, vertex: u32) -> f32 {
        self.weights.get(&vertex).copied().unwrap_or(0.0)
    }

    /// Sets the weight for a vertex.
    pub fn set_weight(&mut self, vertex: u32, weight: f32) {
        if weight > 0.0 {
            self.weights.insert(vertex, weight.clamp(0.0, 1.0));
        } else {
            self.weights.remove(&vertex);
        }
    }

    /// Builds soft selection from a hard selection with distance-based falloff.
    ///
    /// Selected vertices get weight 1.0, and nearby vertices get weights
    /// based on their distance from the nearest selected vertex.
    pub fn from_selection(
        selection: &MeshSelection,
        mesh: &Mesh,
        radius: f32,
        falloff: Falloff,
    ) -> Self {
        let mut soft = SoftSelection::with_falloff(radius, falloff);

        if selection.vertices.is_empty() {
            return soft;
        }

        // Selected vertices get full weight
        for &v in &selection.vertices {
            soft.weights.insert(v, 1.0);
        }

        // Find distances to nearest selected vertex for all other vertices
        for v in 0..mesh.vertex_count() as u32 {
            if selection.vertices.contains(&v) {
                continue;
            }

            let pos = mesh.positions[v as usize];
            let mut min_dist = f32::MAX;

            for &sv in &selection.vertices {
                let dist = pos.distance(mesh.positions[sv as usize]);
                min_dist = min_dist.min(dist);
            }

            if min_dist < radius {
                let weight = falloff.weight(min_dist / radius);
                if weight > 0.0 {
                    soft.weights.insert(v, weight);
                }
            }
        }

        soft
    }

    /// Builds soft selection from a point with distance-based falloff.
    pub fn from_point(point: Vec3, mesh: &Mesh, radius: f32, falloff: Falloff) -> Self {
        let mut soft = SoftSelection::with_falloff(radius, falloff);

        for v in 0..mesh.vertex_count() as u32 {
            let dist = mesh.positions[v as usize].distance(point);
            if dist < radius {
                let weight = falloff.weight(dist / radius);
                if weight > 0.0 {
                    soft.weights.insert(v, weight);
                }
            }
        }

        soft
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Builds vertex adjacency map (vertex -> set of connected vertices).
fn build_vertex_adjacency(mesh: &Mesh) -> HashMap<u32, HashSet<u32>> {
    let mut adjacency: HashMap<u32, HashSet<u32>> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0], tri[1], tri[2]];

        adjacency.entry(i0).or_default().insert(i1);
        adjacency.entry(i0).or_default().insert(i2);
        adjacency.entry(i1).or_default().insert(i0);
        adjacency.entry(i1).or_default().insert(i2);
        adjacency.entry(i2).or_default().insert(i0);
        adjacency.entry(i2).or_default().insert(i1);
    }

    adjacency
}

/// Builds face adjacency map (face -> set of edge-adjacent faces).
fn build_face_adjacency(mesh: &Mesh) -> HashMap<u32, HashSet<u32>> {
    // First, build edge -> faces map
    let edge_faces = build_edge_to_faces(mesh);

    // Then build face adjacency from shared edges
    let mut adjacency: HashMap<u32, HashSet<u32>> = HashMap::new();

    for (_, faces) in edge_faces {
        if faces.len() == 2 {
            let f0 = faces[0];
            let f1 = faces[1];
            adjacency.entry(f0).or_default().insert(f1);
            adjacency.entry(f1).or_default().insert(f0);
        }
    }

    adjacency
}

/// Builds edge -> faces map.
fn build_edge_to_faces(mesh: &Mesh) -> HashMap<Edge, Vec<u32>> {
    let mut edge_faces: HashMap<Edge, Vec<u32>> = HashMap::new();

    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        edge_faces
            .entry(Edge::new(i0, i1))
            .or_default()
            .push(face_idx as u32);
        edge_faces
            .entry(Edge::new(i1, i2))
            .or_default()
            .push(face_idx as u32);
        edge_faces
            .entry(Edge::new(i2, i0))
            .or_default()
            .push(face_idx as u32);
    }

    edge_faces
}

/// Builds edge adjacency map (edge -> set of edges sharing a vertex).
fn build_edge_adjacency(mesh: &Mesh) -> HashMap<Edge, HashSet<Edge>> {
    // First, collect all edges
    let mut all_edges = HashSet::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        all_edges.insert(Edge::new(i0, i1));
        all_edges.insert(Edge::new(i1, i2));
        all_edges.insert(Edge::new(i2, i0));
    }

    // Build vertex -> edges map
    let mut vertex_edges: HashMap<u32, HashSet<Edge>> = HashMap::new();
    for &edge in &all_edges {
        vertex_edges.entry(edge.0).or_default().insert(edge);
        vertex_edges.entry(edge.1).or_default().insert(edge);
    }

    // Build edge adjacency
    let mut adjacency: HashMap<Edge, HashSet<Edge>> = HashMap::new();
    for &edge in &all_edges {
        let mut neighbors = HashSet::new();

        if let Some(edges) = vertex_edges.get(&edge.0) {
            neighbors.extend(edges.iter().filter(|&&e| e != edge));
        }
        if let Some(edges) = vertex_edges.get(&edge.1) {
            neighbors.extend(edges.iter().filter(|&&e| e != edge));
        }

        adjacency.insert(edge, neighbors);
    }

    adjacency
}

/// Computes the normal of a face (triangle).
fn compute_face_normal(mesh: &Mesh, face_idx: usize) -> Vec3 {
    let base = face_idx * 3;
    if base + 2 >= mesh.indices.len() {
        return Vec3::ZERO;
    }

    let i0 = mesh.indices[base] as usize;
    let i1 = mesh.indices[base + 1] as usize;
    let i2 = mesh.indices[base + 2] as usize;

    let v0 = mesh.positions[i0];
    let v1 = mesh.positions[i1];
    let v2 = mesh.positions[i2];

    (v1 - v0).cross(v2 - v0).normalize_or_zero()
}

/// Computes the area of a face (triangle).
fn compute_face_area(mesh: &Mesh, face_idx: usize) -> f32 {
    let base = face_idx * 3;
    if base + 2 >= mesh.indices.len() {
        return 0.0;
    }

    let i0 = mesh.indices[base] as usize;
    let i1 = mesh.indices[base + 1] as usize;
    let i2 = mesh.indices[base + 2] as usize;

    let v0 = mesh.positions[i0];
    let v1 = mesh.positions[i1];
    let v2 = mesh.positions[i2];

    (v1 - v0).cross(v2 - v0).length() * 0.5
}

/// Simple LCG random number generator for deterministic selection.
struct SimpleLcg {
    state: u64,
}

impl SimpleLcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn make_single_triangle() -> Mesh {
        Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: vec![Vec2::ZERO; 3],
            indices: vec![0, 1, 2],
        }
    }

    fn make_two_triangles() -> Mesh {
        // Two triangles sharing an edge (0-1)
        Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(0.5, -1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 3, 1],
        }
    }

    #[test]
    fn test_edge_ordering() {
        let e1 = Edge::new(5, 3);
        let e2 = Edge::new(3, 5);
        assert_eq!(e1, e2);
        assert_eq!(e1.0, 3);
        assert_eq!(e1.1, 5);
    }

    #[test]
    fn test_select_all_vertices() {
        let mesh = make_single_triangle();
        let mut sel = MeshSelection::new();

        sel.select_all_vertices(&mesh);
        assert_eq!(sel.vertices.len(), 3);
        assert!(sel.is_vertex_selected(0));
        assert!(sel.is_vertex_selected(1));
        assert!(sel.is_vertex_selected(2));
    }

    #[test]
    fn test_invert_vertices() {
        let mesh = make_single_triangle();
        let mut sel = MeshSelection::new();

        sel.select_vertex(0);
        sel.invert_vertices(&mesh);

        assert!(!sel.is_vertex_selected(0));
        assert!(sel.is_vertex_selected(1));
        assert!(sel.is_vertex_selected(2));
    }

    #[test]
    fn test_select_all_edges() {
        let mesh = make_single_triangle();
        let mut sel = MeshSelection::new();

        sel.select_all_edges(&mesh);
        assert_eq!(sel.edges.len(), 3);
        assert!(sel.is_edge_selected(0, 1));
        assert!(sel.is_edge_selected(1, 2));
        assert!(sel.is_edge_selected(2, 0));
    }

    #[test]
    fn test_select_all_faces() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        sel.select_all_faces(&mesh);
        assert_eq!(sel.faces.len(), 2);
        assert!(sel.is_face_selected(0));
        assert!(sel.is_face_selected(1));
    }

    #[test]
    fn test_grow_vertices() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        // Select just vertex 0
        sel.select_vertex(0);
        assert_eq!(sel.vertices.len(), 1);

        // Grow to include neighbors
        sel.grow_vertices(&mesh);
        assert!(sel.vertices.len() > 1);
        // Vertex 0 is connected to 1, 2, and 3
        assert!(sel.is_vertex_selected(1));
        assert!(sel.is_vertex_selected(2));
        assert!(sel.is_vertex_selected(3));
    }

    #[test]
    fn test_shrink_vertices() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        // Select all vertices
        sel.select_all_vertices(&mesh);

        // Shrink - when all vertices are selected, there's no selection boundary
        // so nothing gets removed (all neighbors are selected)
        sel.shrink_vertices(&mesh);
        assert_eq!(sel.vertices.len(), 4);

        // Now test actual shrinking: select vertices 0, 1, 2 (not 3)
        sel.clear();
        sel.select_vertex(0);
        sel.select_vertex(1);
        sel.select_vertex(2);

        // Vertices 0 and 1 have neighbor 3 which is not selected, so they're at the boundary
        sel.shrink_vertices(&mesh);
        // Only vertex 2 remains (its neighbors 0, 1 were selected before)
        assert!(sel.is_vertex_selected(2));
        // Actually vertex 2 also has neighbor 0 and 1 which were deselected...
        // Let's reconsider: after checking adjacency in the two-triangle mesh:
        // 0 connects to 1, 2, 3
        // 1 connects to 0, 2, 3
        // 2 connects to 0, 1
        // 3 connects to 0, 1
        // If we select 0, 1, 2:
        // - vertex 0: neighbors are 1 (selected), 2 (selected), 3 (not selected) -> boundary
        // - vertex 1: neighbors are 0 (selected), 2 (selected), 3 (not selected) -> boundary
        // - vertex 2: neighbors are 0 (selected), 1 (selected) -> not boundary!
        // So vertex 2 should remain
        assert_eq!(sel.vertices.len(), 1);
        assert!(sel.is_vertex_selected(2));
    }

    #[test]
    fn test_grow_faces() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        sel.select_face(0);
        sel.grow_faces(&mesh);

        // Should now include face 1 (shares edge 0-1)
        assert!(sel.is_face_selected(0));
        assert!(sel.is_face_selected(1));
    }

    #[test]
    fn test_select_linked_vertices() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        // Start with one vertex
        sel.select_vertex(2);
        sel.select_linked_vertices(&mesh);

        // Should select all vertices (connected mesh)
        assert_eq!(sel.vertices.len(), 4);
    }

    #[test]
    fn test_faces_to_vertices() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        sel.select_face(0);
        sel.faces_to_vertices(&mesh);

        assert!(sel.is_vertex_selected(0));
        assert!(sel.is_vertex_selected(1));
        assert!(sel.is_vertex_selected(2));
        assert!(!sel.is_vertex_selected(3)); // Not in face 0
    }

    #[test]
    fn test_vertices_to_faces_all() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        // Select vertices of face 0 only
        sel.select_vertex(0);
        sel.select_vertex(1);
        sel.select_vertex(2);

        sel.vertices_to_faces_all(&mesh);

        assert!(sel.is_face_selected(0));
        assert!(!sel.is_face_selected(1)); // Not all vertices selected
    }

    #[test]
    fn test_select_boundary_edges() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        sel.select_boundary_edges(&mesh);

        // Edge 0-1 is shared, others are boundary
        assert!(!sel.is_edge_selected(0, 1));
        assert!(sel.is_edge_selected(1, 2));
        assert!(sel.is_edge_selected(2, 0));
        assert!(sel.is_edge_selected(0, 3));
        assert!(sel.is_edge_selected(3, 1));
    }

    #[test]
    fn test_select_faces_by_normal() {
        let mesh = make_single_triangle();
        let mut sel = MeshSelection::new();

        // Triangle faces +Z
        sel.select_faces_by_normal(&mesh, Vec3::Z, 0.9);
        assert!(sel.is_face_selected(0));

        sel.clear();
        sel.select_faces_by_normal(&mesh, Vec3::NEG_Z, 0.9);
        assert!(!sel.is_face_selected(0));
    }

    #[test]
    fn test_select_faces_random() {
        let mesh = make_two_triangles();
        let mut sel1 = MeshSelection::new();
        let mut sel2 = MeshSelection::new();

        // Same seed should give same result
        sel1.select_faces_random(&mesh, 0.5, 42);
        sel2.select_faces_random(&mesh, 0.5, 42);

        assert_eq!(sel1.faces, sel2.faces);

        // Different seed should (likely) give different result
        let mut sel3 = MeshSelection::new();
        sel3.select_faces_random(&mesh, 0.5, 123);
        // Can't guarantee different results, but check it runs
    }

    #[test]
    fn test_soft_selection_from_point() {
        let mesh = make_single_triangle();
        let soft = SoftSelection::from_point(Vec3::ZERO, &mesh, 2.0, Falloff::Linear);

        // Vertex 0 is at origin, should have weight 1.0
        assert!((soft.weight(0) - 1.0).abs() < 0.001);

        // Other vertices should have weight based on distance
        assert!(soft.weight(1) > 0.0);
        assert!(soft.weight(1) < 1.0);
    }

    #[test]
    fn test_falloff_functions() {
        // At center (0), all falloffs should return 1.0
        for falloff in [
            Falloff::Linear,
            Falloff::Smooth,
            Falloff::Sharp,
            Falloff::Root,
            Falloff::Constant,
            Falloff::Sphere,
        ] {
            assert!((falloff.weight(0.0) - 1.0).abs() < 0.001);
        }

        // At edge (1), all should return 0.0
        for falloff in [
            Falloff::Linear,
            Falloff::Smooth,
            Falloff::Sharp,
            Falloff::Root,
            Falloff::Sphere,
        ] {
            assert!(falloff.weight(1.0).abs() < 0.001);
        }

        // Constant returns 1.0 even at edge
        assert!((Falloff::Constant.weight(0.99) - 1.0).abs() < 0.001);

        // Beyond edge (>1), all return 0.0
        for falloff in [
            Falloff::Linear,
            Falloff::Smooth,
            Falloff::Sharp,
            Falloff::Root,
            Falloff::Constant,
            Falloff::Sphere,
        ] {
            assert!(falloff.weight(1.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_toggle_vertex() {
        let mut sel = MeshSelection::new();

        sel.toggle_vertex(0);
        assert!(sel.is_vertex_selected(0));

        sel.toggle_vertex(0);
        assert!(!sel.is_vertex_selected(0));
    }

    #[test]
    fn test_clear() {
        let mesh = make_two_triangles();
        let mut sel = MeshSelection::new();

        sel.select_all_vertices(&mesh);
        sel.select_all_edges(&mesh);
        sel.select_all_faces(&mesh);

        assert!(!sel.is_empty());

        sel.clear();
        assert!(sel.is_empty());
    }
}
