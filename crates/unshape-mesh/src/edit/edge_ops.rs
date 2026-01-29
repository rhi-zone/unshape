use std::collections::HashMap;

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;
use crate::selection::{Edge, MeshSelection};

/// Splits selected edges to create hard edges.
///
/// Duplicates vertices along selected edges so adjacent faces
/// no longer share vertices, allowing distinct normals.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct SplitEdges;

impl SplitEdges {
    /// Creates a split edges operation.
    pub fn new() -> Self {
        Self
    }

    /// Applies this operation to selected edges.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        split_edges(mesh, selection)
    }
}

/// Splits selected edges by duplicating vertices.
pub fn split_edges(mesh: &Mesh, selection: &MeshSelection) -> Mesh {
    if selection.edges.is_empty() {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Find which faces use each selected edge
    let mut edge_faces: HashMap<Edge, Vec<usize>> = HashMap::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        for edge in [Edge::new(i0, i1), Edge::new(i1, i2), Edge::new(i2, i0)] {
            if selection.edges.contains(&edge) {
                edge_faces.entry(edge).or_default().push(face_idx);
            }
        }
    }

    // For each selected edge with multiple faces, duplicate vertices for all but first face
    for (edge, faces) in edge_faces {
        if faces.len() < 2 {
            continue;
        }

        // Keep first face as-is, create new vertices for subsequent faces
        for &face_idx in &faces[1..] {
            let base = face_idx * 3;

            for i in 0..3 {
                let idx = result.indices[base + i];
                if idx == edge.0 || idx == edge.1 {
                    // Duplicate this vertex
                    let new_idx = result.positions.len() as u32;
                    result.positions.push(result.positions[idx as usize]);
                    if idx < result.normals.len() as u32 {
                        result.normals.push(result.normals[idx as usize]);
                    }
                    if idx < result.uvs.len() as u32 {
                        result.uvs.push(result.uvs[idx as usize]);
                    }
                    result.indices[base + i] = new_idx;
                }
            }
        }
    }

    // Recompute normals for split faces
    result.compute_smooth_normals();
    result
}

/// Edge crease data for subdivision control.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeCreases {
    /// Crease weight per edge (0.0 = smooth, 1.0 = sharp).
    pub weights: HashMap<Edge, f32>,
}

impl EdgeCreases {
    /// Creates empty edge creases.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets crease weight for an edge.
    pub fn set(&mut self, edge: Edge, weight: f32) {
        if weight > 0.0 {
            self.weights.insert(edge, weight.clamp(0.0, 1.0));
        } else {
            self.weights.remove(&edge);
        }
    }

    /// Gets crease weight for an edge.
    pub fn get(&self, edge: &Edge) -> f32 {
        self.weights.get(edge).copied().unwrap_or(0.0)
    }

    /// Returns true if the edge is creased.
    pub fn is_creased(&self, edge: &Edge) -> bool {
        self.weights.get(edge).copied().unwrap_or(0.0) > 0.0
    }
}

/// Marks selected edges with crease weights.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CreaseEdges {
    /// Crease weight to apply (0.0 = smooth, 1.0 = sharp).
    pub weight: f32,
}

impl Default for CreaseEdges {
    fn default() -> Self {
        Self { weight: 1.0 }
    }
}

impl CreaseEdges {
    /// Creates a crease edges operation with the given weight.
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }

    /// Applies crease weights to selected edges.
    pub fn apply(&self, creases: &mut EdgeCreases, selection: &MeshSelection) {
        for &edge in &selection.edges {
            creases.set(edge, self.weight);
        }
    }
}

/// Slides selected edges along their adjacent faces.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct SlideEdges {
    /// Slide factor (-1.0 to 1.0, direction along adjacent edges).
    pub factor: f32,
}

impl Default for SlideEdges {
    fn default() -> Self {
        Self { factor: 0.0 }
    }
}

impl SlideEdges {
    /// Creates an edge slide operation.
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }

    /// Applies this operation to selected edges.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        slide_edges(mesh, selection, self.factor)
    }
}

/// Slides selected edges along adjacent faces.
pub fn slide_edges(mesh: &Mesh, selection: &MeshSelection, factor: f32) -> Mesh {
    if selection.edges.is_empty() || factor.abs() < 1e-6 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // For each selected edge, find adjacent faces and compute slide directions
    let mut edge_to_faces: HashMap<Edge, Vec<usize>> = HashMap::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        for edge in [Edge::new(i0, i1), Edge::new(i1, i2), Edge::new(i2, i0)] {
            edge_to_faces.entry(edge).or_default().push(face_idx);
        }
    }

    // Compute slide vectors for each vertex on selected edges
    let mut vertex_slide: HashMap<u32, Vec3> = HashMap::new();

    for edge in &selection.edges {
        if let Some(faces) = edge_to_faces.get(edge) {
            for &face_idx in faces {
                let base = face_idx * 3;
                let verts = [
                    mesh.indices[base],
                    mesh.indices[base + 1],
                    mesh.indices[base + 2],
                ];

                // Find the third vertex (not on the edge)
                let third = verts.iter().find(|&&v| v != edge.0 && v != edge.1);

                if let Some(&third_v) = third {
                    // Compute slide direction for each edge vertex toward third vertex
                    let p_third = mesh.positions[third_v as usize];

                    for &edge_v in &[edge.0, edge.1] {
                        let p_edge = mesh.positions[edge_v as usize];
                        let slide_dir = (p_third - p_edge).normalize_or_zero();
                        *vertex_slide.entry(edge_v).or_insert(Vec3::ZERO) += slide_dir;
                    }
                }
            }
        }
    }

    // Apply slide
    for (&v, &slide_dir) in &vertex_slide {
        let slide_vec = slide_dir.normalize_or_zero() * factor;
        result.positions[v as usize] += slide_vec;
    }

    result.compute_smooth_normals();
    result
}
