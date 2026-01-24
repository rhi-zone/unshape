//! Selection-aware mesh editing operations.
//!
//! These operations work with [`MeshSelection`] to modify only selected elements,
//! enabling Blender-style constructive modeling workflows.
//!
//! # Operations
//!
//! ## Simple
//! - [`DeleteFaces`] - Remove selected faces from mesh
//! - [`TransformVertices`] - Apply transformation to selected vertices
//! - [`TriangulateFaces`] - Convert selected quads/ngons to triangles
//! - [`PokeFaces`] - Add vertex at face center, create triangle fan
//!
//! ## Medium
//! - [`ScaleFaces`] - Scale faces around their centers
//! - [`SmoothVertices`] - Laplacian smoothing on selected vertices
//! - [`MergeVertices`] - Collapse vertices to single point
//! - [`SplitEdges`] - Duplicate edges for hard edges
//! - [`CreaseEdges`] - Mark edges for subdivision weighting
//!
//! ## Complex
//! - [`ExtrudeFaces`] - Extrude selected faces along normals
//! - [`InsetFaces`] - Inset selected faces toward centers
//! - [`SubdivideFaces`] - Subdivide only selected faces
//! - [`SlideEdges`] - Slide edges along adjacent faces
//! - [`RipVertices`] - Disconnect vertices from adjacent faces
//!
//! ## Advanced
//! - [`BridgeEdgeLoops`] - Connect two edge loops with faces
//! - [`KnifeCut`] - Cut mesh along arbitrary path

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;
use crate::selection::{Edge, MeshSelection, SoftSelection};

// ============================================================================
// Simple Operations
// ============================================================================

/// Deletes selected faces from the mesh.
///
/// Can optionally remove orphaned vertices (vertices no longer used by any face)
/// and orphaned edges after deletion.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct DeleteFaces {
    /// Whether to remove vertices that are no longer used by any face.
    pub remove_orphaned_vertices: bool,
}

impl Default for DeleteFaces {
    fn default() -> Self {
        Self {
            remove_orphaned_vertices: true,
        }
    }
}

impl DeleteFaces {
    /// Creates a new delete faces operation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies this operation to a mesh with the given selection.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        delete_faces(mesh, selection, self.remove_orphaned_vertices)
    }
}

/// Deletes selected faces from a mesh.
///
/// If `remove_orphaned_vertices` is true, vertices that are no longer
/// referenced by any face will be removed from the mesh.
pub fn delete_faces(
    mesh: &Mesh,
    selection: &MeshSelection,
    remove_orphaned_vertices: bool,
) -> Mesh {
    if selection.faces.is_empty() {
        return mesh.clone();
    }

    // Collect faces to keep
    let mut new_indices = Vec::new();
    for face_idx in 0..mesh.triangle_count() {
        if !selection.faces.contains(&(face_idx as u32)) {
            let base = face_idx * 3;
            new_indices.push(mesh.indices[base]);
            new_indices.push(mesh.indices[base + 1]);
            new_indices.push(mesh.indices[base + 2]);
        }
    }

    if !remove_orphaned_vertices {
        return Mesh {
            positions: mesh.positions.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            indices: new_indices,
        };
    }

    // Find used vertices
    let used_vertices: HashSet<u32> = new_indices.iter().copied().collect();

    // Create vertex remapping
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut new_positions = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();

    for old_idx in 0..mesh.positions.len() as u32 {
        if used_vertices.contains(&old_idx) {
            let new_idx = new_positions.len() as u32;
            vertex_map.insert(old_idx, new_idx);

            new_positions.push(mesh.positions[old_idx as usize]);
            if old_idx < mesh.normals.len() as u32 {
                new_normals.push(mesh.normals[old_idx as usize]);
            }
            if old_idx < mesh.uvs.len() as u32 {
                new_uvs.push(mesh.uvs[old_idx as usize]);
            }
        }
    }

    // Remap indices
    let remapped_indices: Vec<u32> = new_indices.iter().map(|&i| vertex_map[&i]).collect();

    Mesh {
        positions: new_positions,
        normals: new_normals,
        uvs: new_uvs,
        indices: remapped_indices,
    }
}

/// Transforms selected vertices by a matrix.
///
/// Supports soft selection for smooth falloff transformations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct TransformVertices {
    /// The transformation matrix to apply.
    pub matrix: Mat4,
}

impl Default for TransformVertices {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
        }
    }
}

impl TransformVertices {
    /// Creates a translation transform.
    pub fn translate(offset: Vec3) -> Self {
        Self {
            matrix: Mat4::from_translation(offset),
        }
    }

    /// Creates a rotation transform around an axis.
    pub fn rotate(axis: Vec3, angle: f32) -> Self {
        Self {
            matrix: Mat4::from_axis_angle(axis, angle),
        }
    }

    /// Creates a uniform scale transform.
    pub fn scale(factor: f32) -> Self {
        Self {
            matrix: Mat4::from_scale(Vec3::splat(factor)),
        }
    }

    /// Creates a non-uniform scale transform.
    pub fn scale_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            matrix: Mat4::from_scale(Vec3::new(x, y, z)),
        }
    }

    /// Creates a transform from a matrix.
    pub fn from_matrix(matrix: Mat4) -> Self {
        Self { matrix }
    }

    /// Applies this operation to a mesh with hard selection.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        transform_vertices(mesh, selection, self.matrix)
    }

    /// Applies this operation with soft selection for smooth falloff.
    pub fn apply_soft(&self, mesh: &Mesh, soft_selection: &SoftSelection) -> Mesh {
        transform_vertices_soft(mesh, soft_selection, self.matrix)
    }
}

/// Transforms selected vertices by a matrix.
pub fn transform_vertices(mesh: &Mesh, selection: &MeshSelection, matrix: Mat4) -> Mesh {
    if selection.vertices.is_empty() {
        return mesh.clone();
    }

    let mut result = mesh.clone();
    let normal_matrix = matrix.inverse().transpose();

    for &v in &selection.vertices {
        let i = v as usize;
        if i < result.positions.len() {
            result.positions[i] = matrix.transform_point3(result.positions[i]);
        }
        if i < result.normals.len() {
            result.normals[i] = normal_matrix
                .transform_vector3(result.normals[i])
                .normalize_or_zero();
        }
    }

    result
}

/// Transforms vertices with soft selection weights for smooth falloff.
pub fn transform_vertices_soft(mesh: &Mesh, soft_selection: &SoftSelection, matrix: Mat4) -> Mesh {
    if soft_selection.is_empty() {
        return mesh.clone();
    }

    let mut result = mesh.clone();
    let normal_matrix = matrix.inverse().transpose();

    for (&v, &weight) in &soft_selection.weights {
        let i = v as usize;
        if i < result.positions.len() {
            let original = result.positions[i];
            let transformed = matrix.transform_point3(original);
            result.positions[i] = original.lerp(transformed, weight);
        }
        if i < result.normals.len() {
            let original = result.normals[i];
            let transformed = normal_matrix
                .transform_vector3(original)
                .normalize_or_zero();
            result.normals[i] = original.lerp(transformed, weight).normalize_or_zero();
        }
    }

    result
}

/// Triangulates selected faces (for meshes with quads/ngons).
///
/// Note: Since `Mesh` uses indexed triangles, this operation is primarily useful
/// when working with HalfEdgeMesh that supports quads. For indexed meshes,
/// this is effectively a no-op.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct TriangulateFaces;

impl TriangulateFaces {
    /// Creates a new triangulate faces operation.
    pub fn new() -> Self {
        Self
    }

    /// Applies this operation - for indexed triangle mesh, this is a no-op.
    pub fn apply(&self, mesh: &Mesh, _selection: &MeshSelection) -> Mesh {
        // IndexedMesh is already triangulated
        mesh.clone()
    }
}

/// Pokes selected faces by adding a vertex at their center.
///
/// Each selected triangle is replaced by 3 triangles connecting
/// the original edges to the new center vertex.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct PokeFaces {
    /// Offset along face normal for the poked vertex (0 = at face center).
    pub offset: f32,
}

impl Default for PokeFaces {
    fn default() -> Self {
        Self { offset: 0.0 }
    }
}

impl PokeFaces {
    /// Creates a poke operation with no offset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a poke operation with the given offset along face normals.
    pub fn with_offset(offset: f32) -> Self {
        Self { offset }
    }

    /// Applies this operation to a mesh with the given selection.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        poke_faces(mesh, selection, self.offset)
    }
}

/// Pokes selected faces by adding center vertices.
pub fn poke_faces(mesh: &Mesh, selection: &MeshSelection, offset: f32) -> Mesh {
    if selection.faces.is_empty() {
        return mesh.clone();
    }

    // Count new geometry
    let poked_count = selection.faces.len();
    let unpoked_count = mesh.triangle_count() - poked_count;

    // Each poked face becomes 3 triangles, unpoked stay as 1
    let new_triangle_count = unpoked_count + poked_count * 3;
    let new_vertex_count = mesh.vertex_count() + poked_count;

    let mut result = Mesh::with_capacity(new_vertex_count, new_triangle_count);

    // Copy all original vertices
    result.positions.extend_from_slice(&mesh.positions);
    result.normals.extend_from_slice(&mesh.normals);
    result.uvs.extend_from_slice(&mesh.uvs);

    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        if selection.faces.contains(&(face_idx as u32)) {
            // Poke this face
            let p0 = mesh.positions[i0 as usize];
            let p1 = mesh.positions[i1 as usize];
            let p2 = mesh.positions[i2 as usize];

            let center = (p0 + p1 + p2) / 3.0;
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            let center_pos = center + normal * offset;

            let center_idx = result.positions.len() as u32;
            result.positions.push(center_pos);
            result.normals.push(normal);

            // Interpolate UVs if present
            if !mesh.uvs.is_empty() {
                let uv0 = mesh.uvs[i0 as usize];
                let uv1 = mesh.uvs[i1 as usize];
                let uv2 = mesh.uvs[i2 as usize];
                result.uvs.push((uv0 + uv1 + uv2) / 3.0);
            }

            // Create 3 triangles from center to each edge
            result.indices.extend_from_slice(&[i0, i1, center_idx]);
            result.indices.extend_from_slice(&[i1, i2, center_idx]);
            result.indices.extend_from_slice(&[i2, i0, center_idx]);
        } else {
            // Keep original face
            result.indices.extend_from_slice(&[i0, i1, i2]);
        }
    }

    result
}

// ============================================================================
// Medium Operations
// ============================================================================

/// Scales selected faces around their individual centers.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct ScaleFaces {
    /// Scale factor (1.0 = no change, 0.5 = half size, 2.0 = double).
    pub factor: f32,
}

impl Default for ScaleFaces {
    fn default() -> Self {
        Self { factor: 1.0 }
    }
}

impl ScaleFaces {
    /// Creates a scale faces operation with the given factor.
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }

    /// Applies this operation to a mesh with the given selection.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        scale_faces(mesh, selection, self.factor)
    }
}

/// Scales selected faces around their centers.
pub fn scale_faces(mesh: &Mesh, selection: &MeshSelection, factor: f32) -> Mesh {
    if selection.faces.is_empty() || (factor - 1.0).abs() < 1e-6 {
        return mesh.clone();
    }

    // For indexed mesh with shared vertices, we need to split faces first
    // to avoid moving shared vertices multiple times with different scales.
    // We'll create new vertices for selected faces.

    let mut result = mesh.clone();

    // Track which vertices need to be duplicated for selected faces
    let mut vertex_to_new: HashMap<(u32, u32), u32> = HashMap::new(); // (original_idx, face_idx) -> new_idx

    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 >= mesh.indices.len() {
            continue;
        }

        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        let p0 = mesh.positions[i0 as usize];
        let p1 = mesh.positions[i1 as usize];
        let p2 = mesh.positions[i2 as usize];

        let center = (p0 + p1 + p2) / 3.0;

        // Create new vertices for this face
        for (local_idx, &orig_idx) in [i0, i1, i2].iter().enumerate() {
            let new_idx = result.positions.len() as u32;
            vertex_to_new.insert((orig_idx, face_idx), new_idx);

            let orig_pos = mesh.positions[orig_idx as usize];
            let scaled_pos = center + (orig_pos - center) * factor;
            result.positions.push(scaled_pos);

            if orig_idx < mesh.normals.len() as u32 {
                result.normals.push(mesh.normals[orig_idx as usize]);
            }
            if orig_idx < mesh.uvs.len() as u32 {
                result.uvs.push(mesh.uvs[orig_idx as usize]);
            }

            // Update the face index
            result.indices[base + local_idx] = new_idx;
        }
    }

    result
}

/// Applies Laplacian smoothing to selected vertices.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct SmoothVertices {
    /// Smoothing factor per iteration (0.0 = no change, 1.0 = move to average).
    pub lambda: f32,
    /// Number of smoothing iterations.
    pub iterations: usize,
}

impl Default for SmoothVertices {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            iterations: 1,
        }
    }
}

impl SmoothVertices {
    /// Creates a smooth vertices operation.
    pub fn new(lambda: f32, iterations: usize) -> Self {
        Self { lambda, iterations }
    }

    /// Applies this operation to selected vertices.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        smooth_vertices(mesh, selection, self.lambda, self.iterations)
    }
}

/// Smooths selected vertices using Laplacian smoothing.
pub fn smooth_vertices(
    mesh: &Mesh,
    selection: &MeshSelection,
    lambda: f32,
    iterations: usize,
) -> Mesh {
    if selection.vertices.is_empty() || iterations == 0 || lambda == 0.0 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Build adjacency for selected vertices only
    let adjacency = build_vertex_adjacency(mesh);

    for _ in 0..iterations {
        let mut new_positions = result.positions.clone();

        for &v in &selection.vertices {
            let i = v as usize;
            if i >= result.positions.len() {
                continue;
            }

            if let Some(neighbors) = adjacency.get(&v) {
                if neighbors.is_empty() {
                    continue;
                }

                // Compute centroid of neighbors
                let mut centroid = Vec3::ZERO;
                let mut count = 0;
                for &neighbor in neighbors {
                    if (neighbor as usize) < result.positions.len() {
                        centroid += result.positions[neighbor as usize];
                        count += 1;
                    }
                }

                if count > 0 {
                    centroid /= count as f32;
                    new_positions[i] = result.positions[i].lerp(centroid, lambda);
                }
            }
        }

        result.positions = new_positions;
    }

    result.compute_smooth_normals();
    result
}

/// Mode for merging vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MergeMode {
    /// Merge at the center of selected vertices.
    #[default]
    Center,
    /// Merge at a specific position.
    AtPosition,
    /// Merge at the first selected vertex.
    First,
    /// Merge at the last selected vertex.
    Last,
}

/// Merges selected vertices into a single vertex.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct MergeVertices {
    /// How to determine the merge position.
    pub mode: MergeMode,
    /// Target position when mode is AtPosition.
    pub position: Vec3,
}

impl Default for MergeVertices {
    fn default() -> Self {
        Self {
            mode: MergeMode::Center,
            position: Vec3::ZERO,
        }
    }
}

impl MergeVertices {
    /// Creates a merge at center operation.
    pub fn at_center() -> Self {
        Self {
            mode: MergeMode::Center,
            position: Vec3::ZERO,
        }
    }

    /// Creates a merge at position operation.
    pub fn at_position(position: Vec3) -> Self {
        Self {
            mode: MergeMode::AtPosition,
            position,
        }
    }

    /// Applies this operation to selected vertices.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        merge_vertices(mesh, selection, self.mode, self.position)
    }
}

/// Merges selected vertices into a single vertex.
pub fn merge_vertices(
    mesh: &Mesh,
    selection: &MeshSelection,
    mode: MergeMode,
    target_position: Vec3,
) -> Mesh {
    if selection.vertices.len() < 2 {
        return mesh.clone();
    }

    let selected: Vec<u32> = selection.vertices.iter().copied().collect();

    // Determine merge position
    let merge_pos = match mode {
        MergeMode::Center => {
            let sum: Vec3 = selected
                .iter()
                .filter_map(|&v| mesh.positions.get(v as usize).copied())
                .sum();
            sum / selected.len() as f32
        }
        MergeMode::AtPosition => target_position,
        MergeMode::First => mesh
            .positions
            .get(selected[0] as usize)
            .copied()
            .unwrap_or(Vec3::ZERO),
        MergeMode::Last => mesh
            .positions
            .get(*selected.last().unwrap() as usize)
            .copied()
            .unwrap_or(Vec3::ZERO),
    };

    // The first selected vertex becomes the merge target
    let merge_target = selected[0];

    // Build vertex remapping: all selected -> merge_target, others stay same
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    for &v in &selected {
        vertex_map.insert(v, merge_target);
    }

    // Find which vertices are actually used after merge
    let mut used_vertices: HashSet<u32> = HashSet::new();
    for idx in &mesh.indices {
        let mapped = vertex_map.get(idx).copied().unwrap_or(*idx);
        used_vertices.insert(mapped);
    }

    // Create new vertex list (keeping only used vertices)
    let mut new_vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut new_positions = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();

    for old_idx in 0..mesh.positions.len() as u32 {
        // Skip selected vertices except the merge target
        if selection.vertices.contains(&old_idx) && old_idx != merge_target {
            continue;
        }

        if !used_vertices.contains(&old_idx) && old_idx != merge_target {
            continue;
        }

        let new_idx = new_positions.len() as u32;
        new_vertex_map.insert(old_idx, new_idx);

        if old_idx == merge_target {
            new_positions.push(merge_pos);
            // Average normals of merged vertices
            let avg_normal: Vec3 = selected
                .iter()
                .filter_map(|&v| mesh.normals.get(v as usize).copied())
                .sum();
            new_normals.push(avg_normal.normalize_or_zero());
            // Use merge target's UV
            if let Some(&uv) = mesh.uvs.get(old_idx as usize) {
                new_uvs.push(uv);
            }
        } else {
            new_positions.push(mesh.positions[old_idx as usize]);
            if let Some(&n) = mesh.normals.get(old_idx as usize) {
                new_normals.push(n);
            }
            if let Some(&uv) = mesh.uvs.get(old_idx as usize) {
                new_uvs.push(uv);
            }
        }
    }

    // Remap indices
    let mut new_indices = Vec::new();
    for tri in mesh.indices.chunks(3) {
        let i0 = vertex_map.get(&tri[0]).copied().unwrap_or(tri[0]);
        let i1 = vertex_map.get(&tri[1]).copied().unwrap_or(tri[1]);
        let i2 = vertex_map.get(&tri[2]).copied().unwrap_or(tri[2]);

        let n0 = new_vertex_map.get(&i0).copied().unwrap_or(0);
        let n1 = new_vertex_map.get(&i1).copied().unwrap_or(0);
        let n2 = new_vertex_map.get(&i2).copied().unwrap_or(0);

        // Skip degenerate triangles
        if n0 != n1 && n1 != n2 && n2 != n0 {
            new_indices.extend_from_slice(&[n0, n1, n2]);
        }
    }

    Mesh {
        positions: new_positions,
        normals: new_normals,
        uvs: new_uvs,
        indices: new_indices,
    }
}

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

// ============================================================================
// Complex Operations
// ============================================================================

/// Extrudes selected faces along their normals.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct ExtrudeFaces {
    /// Distance to extrude (positive = outward).
    pub amount: f32,
}

impl Default for ExtrudeFaces {
    fn default() -> Self {
        Self { amount: 1.0 }
    }
}

impl ExtrudeFaces {
    /// Creates an extrude operation with the given amount.
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }

    /// Applies this operation to selected faces.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        extrude_faces(mesh, selection, self.amount)
    }
}

/// Extrudes selected faces along their normals.
pub fn extrude_faces(mesh: &Mesh, selection: &MeshSelection, amount: f32) -> Mesh {
    if selection.faces.is_empty() {
        return mesh.clone();
    }

    // We need to:
    // 1. Duplicate vertices of selected faces
    // 2. Move duplicated vertices along face normals
    // 3. Create side faces connecting original and extruded edges
    // 4. Replace original faces with extruded faces

    let mut result = mesh.clone();

    // Collect edges that are on the boundary of the selection (shared with unselected faces)
    let mut edge_to_faces: HashMap<Edge, Vec<u32>> = HashMap::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        for edge in [Edge::new(i0, i1), Edge::new(i1, i2), Edge::new(i2, i0)] {
            edge_to_faces.entry(edge).or_default().push(face_idx as u32);
        }
    }

    // Find boundary edges of selection (edges where one face is selected and one is not, or edge is on mesh boundary)
    let mut boundary_edges: HashSet<Edge> = HashSet::new();
    for (edge, faces) in &edge_to_faces {
        let selected_count = faces.iter().filter(|f| selection.faces.contains(f)).count();
        if selected_count == 1 {
            // Edge is on selection boundary
            boundary_edges.insert(*edge);
        }
    }

    // Map original vertices to new extruded vertices
    let mut vertex_to_extruded: HashMap<u32, u32> = HashMap::new();

    // Collect all vertices from selected faces
    let mut selected_vertices: HashSet<u32> = HashSet::new();
    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 < mesh.indices.len() {
            selected_vertices.insert(mesh.indices[base]);
            selected_vertices.insert(mesh.indices[base + 1]);
            selected_vertices.insert(mesh.indices[base + 2]);
        }
    }

    // Compute average normal per vertex from selected faces
    let mut vertex_normals: HashMap<u32, Vec3> = HashMap::new();
    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 >= mesh.indices.len() {
            continue;
        }
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        let p0 = mesh.positions[i0 as usize];
        let p1 = mesh.positions[i1 as usize];
        let p2 = mesh.positions[i2 as usize];

        let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();

        for &idx in &[i0, i1, i2] {
            *vertex_normals.entry(idx).or_insert(Vec3::ZERO) += face_normal;
        }
    }

    // Create extruded vertices
    for &v in &selected_vertices {
        let new_idx = result.positions.len() as u32;
        vertex_to_extruded.insert(v, new_idx);

        let pos = mesh.positions[v as usize];
        let normal = vertex_normals
            .get(&v)
            .copied()
            .unwrap_or(Vec3::Y)
            .normalize_or_zero();
        let extruded_pos = pos + normal * amount;

        result.positions.push(extruded_pos);
        if v < mesh.normals.len() as u32 {
            result.normals.push(normal);
        }
        if v < mesh.uvs.len() as u32 {
            result.uvs.push(mesh.uvs[v as usize]);
        }
    }

    // Update selected faces to use extruded vertices
    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 >= result.indices.len() {
            continue;
        }
        for i in 0..3 {
            let orig_idx = mesh.indices[base + i];
            if let Some(&new_idx) = vertex_to_extruded.get(&orig_idx) {
                result.indices[base + i] = new_idx;
            }
        }
    }

    // Create side faces for boundary edges
    for edge in boundary_edges {
        let orig_0 = edge.0;
        let orig_1 = edge.1;

        let extruded_0 = vertex_to_extruded.get(&orig_0).copied();
        let extruded_1 = vertex_to_extruded.get(&orig_1).copied();

        if let (Some(ext_0), Some(ext_1)) = (extruded_0, extruded_1) {
            // Create quad as two triangles
            // Quad: orig_0, orig_1, ext_1, ext_0
            // We need to determine winding order - use the face normal direction

            // Triangle 1: orig_0, orig_1, ext_1
            result.indices.extend_from_slice(&[orig_0, orig_1, ext_1]);
            // Triangle 2: orig_0, ext_1, ext_0
            result.indices.extend_from_slice(&[orig_0, ext_1, ext_0]);
        }
    }

    result.compute_smooth_normals();
    result
}

/// Insets selected faces toward their centers.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct InsetFaces {
    /// Inset amount (0.0 = no change, 1.0 = shrink to center).
    pub amount: f32,
    /// Whether to inset faces individually or as a region.
    pub individual: bool,
}

impl Default for InsetFaces {
    fn default() -> Self {
        Self {
            amount: 0.2,
            individual: true,
        }
    }
}

impl InsetFaces {
    /// Creates an inset operation with the given amount.
    pub fn new(amount: f32) -> Self {
        Self {
            amount,
            individual: true,
        }
    }

    /// Creates an inset operation that treats selection as a region.
    pub fn region(amount: f32) -> Self {
        Self {
            amount,
            individual: false,
        }
    }

    /// Applies this operation to selected faces.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        if self.individual {
            inset_faces_individual(mesh, selection, self.amount)
        } else {
            inset_faces_region(mesh, selection, self.amount)
        }
    }
}

/// Insets selected faces individually toward their centers.
pub fn inset_faces_individual(mesh: &Mesh, selection: &MeshSelection, amount: f32) -> Mesh {
    if selection.faces.is_empty() {
        return mesh.clone();
    }

    let selected_count = selection.faces.len();
    let unselected_count = mesh.triangle_count() - selected_count;

    // Each inset face creates: 1 inner triangle + 3 bridge quads (6 triangles)
    let new_triangle_count = unselected_count + selected_count * 7;
    let new_vertex_count = mesh.vertex_count() + selected_count * 6; // 3 orig + 3 inset per face

    let mut result = Mesh::with_capacity(new_vertex_count, new_triangle_count);

    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        if selection.faces.contains(&(face_idx as u32)) {
            // Inset this face
            let p0 = mesh.positions[i0 as usize];
            let p1 = mesh.positions[i1 as usize];
            let p2 = mesh.positions[i2 as usize];

            let center = (p0 + p1 + p2) / 3.0;
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();

            // Inset positions
            let ip0 = p0.lerp(center, amount);
            let ip1 = p1.lerp(center, amount);
            let ip2 = p2.lerp(center, amount);

            // Add original vertices for this face
            let base_idx = result.positions.len() as u32;
            result.positions.extend_from_slice(&[p0, p1, p2]);
            result.normals.extend_from_slice(&[normal, normal, normal]);

            // Add inset vertices
            result.positions.extend_from_slice(&[ip0, ip1, ip2]);
            result.normals.extend_from_slice(&[normal, normal, normal]);

            // UVs
            if !mesh.uvs.is_empty() {
                let uv0 = mesh.uvs[i0 as usize];
                let uv1 = mesh.uvs[i1 as usize];
                let uv2 = mesh.uvs[i2 as usize];
                let uv_center = (uv0 + uv1 + uv2) / 3.0;

                result.uvs.extend_from_slice(&[uv0, uv1, uv2]);
                result.uvs.extend_from_slice(&[
                    uv0.lerp(uv_center, amount),
                    uv1.lerp(uv_center, amount),
                    uv2.lerp(uv_center, amount),
                ]);
            }

            // Indices: base+0,1,2 = original, base+3,4,5 = inset
            // Inner triangle
            result
                .indices
                .extend_from_slice(&[base_idx + 3, base_idx + 4, base_idx + 5]);

            // Bridge quads
            // Edge 0-1
            result
                .indices
                .extend_from_slice(&[base_idx, base_idx + 1, base_idx + 4]);
            result
                .indices
                .extend_from_slice(&[base_idx, base_idx + 4, base_idx + 3]);
            // Edge 1-2
            result
                .indices
                .extend_from_slice(&[base_idx + 1, base_idx + 2, base_idx + 5]);
            result
                .indices
                .extend_from_slice(&[base_idx + 1, base_idx + 5, base_idx + 4]);
            // Edge 2-0
            result
                .indices
                .extend_from_slice(&[base_idx + 2, base_idx, base_idx + 3]);
            result
                .indices
                .extend_from_slice(&[base_idx + 2, base_idx + 3, base_idx + 5]);
        } else {
            // Keep original face - need to copy vertices
            let base_idx = result.positions.len() as u32;

            result.positions.push(mesh.positions[i0 as usize]);
            result.positions.push(mesh.positions[i1 as usize]);
            result.positions.push(mesh.positions[i2 as usize]);

            if !mesh.normals.is_empty() {
                result
                    .normals
                    .push(mesh.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y));
                result
                    .normals
                    .push(mesh.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y));
                result
                    .normals
                    .push(mesh.normals.get(i2 as usize).copied().unwrap_or(Vec3::Y));
            }

            if !mesh.uvs.is_empty() {
                result
                    .uvs
                    .push(mesh.uvs.get(i0 as usize).copied().unwrap_or_default());
                result
                    .uvs
                    .push(mesh.uvs.get(i1 as usize).copied().unwrap_or_default());
                result
                    .uvs
                    .push(mesh.uvs.get(i2 as usize).copied().unwrap_or_default());
            }

            result
                .indices
                .extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);
        }
    }

    result
}

/// Insets selected faces as a connected region.
pub fn inset_faces_region(mesh: &Mesh, selection: &MeshSelection, amount: f32) -> Mesh {
    if selection.faces.is_empty() {
        return mesh.clone();
    }

    // For region inset, we need to find the boundary of the selection
    // and inset vertices along that boundary

    // Find boundary vertices (vertices on edges between selected and unselected faces)
    let mut edge_to_faces: HashMap<Edge, Vec<u32>> = HashMap::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        for edge in [Edge::new(i0, i1), Edge::new(i1, i2), Edge::new(i2, i0)] {
            edge_to_faces.entry(edge).or_default().push(face_idx as u32);
        }
    }

    // Find boundary edges
    let mut boundary_edges: HashSet<Edge> = HashSet::new();
    for (edge, faces) in &edge_to_faces {
        let selected_count = faces.iter().filter(|f| selection.faces.contains(f)).count();
        if selected_count == 1 || (selected_count == faces.len() && faces.len() == 1) {
            boundary_edges.insert(*edge);
        }
    }

    // Find boundary vertices
    let mut boundary_vertices: HashSet<u32> = HashSet::new();
    for edge in &boundary_edges {
        boundary_vertices.insert(edge.0);
        boundary_vertices.insert(edge.1);
    }

    // Compute region center
    let mut region_center = Vec3::ZERO;
    let mut region_vertex_count = 0;
    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 < mesh.indices.len() {
            for i in 0..3 {
                let v = mesh.indices[base + i] as usize;
                region_center += mesh.positions[v];
                region_vertex_count += 1;
            }
        }
    }
    if region_vertex_count > 0 {
        region_center /= region_vertex_count as f32;
    }

    let mut result = mesh.clone();

    // Create inset vertices for boundary
    let mut vertex_to_inset: HashMap<u32, u32> = HashMap::new();
    for &v in &boundary_vertices {
        let new_idx = result.positions.len() as u32;
        vertex_to_inset.insert(v, new_idx);

        let pos = mesh.positions[v as usize];
        let inset_pos = pos.lerp(region_center, amount);

        result.positions.push(inset_pos);
        if v < mesh.normals.len() as u32 {
            result.normals.push(mesh.normals[v as usize]);
        }
        if v < mesh.uvs.len() as u32 {
            result.uvs.push(mesh.uvs[v as usize]);
        }
    }

    // Update selected faces to use inset vertices on boundary
    for &face_idx in &selection.faces {
        let base = face_idx as usize * 3;
        if base + 2 >= result.indices.len() {
            continue;
        }
        for i in 0..3 {
            let orig_idx = mesh.indices[base + i];
            if let Some(&inset_idx) = vertex_to_inset.get(&orig_idx) {
                result.indices[base + i] = inset_idx;
            }
        }
    }

    // Create bridge faces along boundary edges
    for edge in &boundary_edges {
        if let (Some(&inset_0), Some(&inset_1)) =
            (vertex_to_inset.get(&edge.0), vertex_to_inset.get(&edge.1))
        {
            // Create quad: edge.0, edge.1, inset_1, inset_0
            result.indices.extend_from_slice(&[edge.0, edge.1, inset_1]);
            result
                .indices
                .extend_from_slice(&[edge.0, inset_1, inset_0]);
        }
    }

    result.compute_smooth_normals();
    result
}

/// Subdivides selected faces.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct SubdivideFaces {
    /// Number of subdivision levels.
    pub levels: u32,
}

impl Default for SubdivideFaces {
    fn default() -> Self {
        Self { levels: 1 }
    }
}

impl SubdivideFaces {
    /// Creates a subdivide operation with the given levels.
    pub fn new(levels: u32) -> Self {
        Self { levels }
    }

    /// Applies this operation to selected faces.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        subdivide_faces(mesh, selection, self.levels)
    }
}

/// Subdivides selected faces by adding midpoint vertices.
pub fn subdivide_faces(mesh: &Mesh, selection: &MeshSelection, levels: u32) -> Mesh {
    if selection.faces.is_empty() || levels == 0 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    for _ in 0..levels {
        let mut new_mesh = Mesh::new();
        new_mesh.positions = result.positions.clone();
        new_mesh.normals = result.normals.clone();
        new_mesh.uvs = result.uvs.clone();

        // Track edge midpoints
        let mut edge_midpoints: HashMap<Edge, u32> = HashMap::new();

        // Rebuild face index set for current result
        let current_selection: HashSet<u32> = if levels == 1 {
            selection.faces.clone()
        } else {
            // After first subdivision, all faces derived from selected faces are selected
            // For simplicity, track by marking all new faces
            (0..result.triangle_count() as u32).collect()
        };

        for face_idx in 0..result.triangle_count() {
            let base = face_idx * 3;
            let i0 = result.indices[base];
            let i1 = result.indices[base + 1];
            let i2 = result.indices[base + 2];

            // Only subdivide faces that were originally selected (for first level)
            // or derived from selected faces (subsequent levels)
            let should_subdivide = if levels == 1 {
                selection.faces.contains(&(face_idx as u32))
            } else {
                current_selection.contains(&(face_idx as u32))
            };

            if should_subdivide {
                // Get or create midpoints
                let m01 =
                    get_or_create_midpoint(&mut new_mesh, &mut edge_midpoints, &result, i0, i1);
                let m12 =
                    get_or_create_midpoint(&mut new_mesh, &mut edge_midpoints, &result, i1, i2);
                let m20 =
                    get_or_create_midpoint(&mut new_mesh, &mut edge_midpoints, &result, i2, i0);

                // Create 4 triangles
                new_mesh.indices.extend_from_slice(&[i0, m01, m20]);
                new_mesh.indices.extend_from_slice(&[m01, i1, m12]);
                new_mesh.indices.extend_from_slice(&[m20, m12, i2]);
                new_mesh.indices.extend_from_slice(&[m01, m12, m20]);
            } else {
                // Keep original triangle
                new_mesh.indices.extend_from_slice(&[i0, i1, i2]);
            }
        }

        result = new_mesh;
    }

    result.compute_smooth_normals();
    result
}

fn get_or_create_midpoint(
    mesh: &mut Mesh,
    edge_midpoints: &mut HashMap<Edge, u32>,
    source: &Mesh,
    i0: u32,
    i1: u32,
) -> u32 {
    let edge = Edge::new(i0, i1);

    if let Some(&midpoint) = edge_midpoints.get(&edge) {
        return midpoint;
    }

    let p0 = source.positions[i0 as usize];
    let p1 = source.positions[i1 as usize];
    let midpoint_pos = (p0 + p1) * 0.5;

    let new_idx = mesh.positions.len() as u32;
    mesh.positions.push(midpoint_pos);

    if !source.normals.is_empty() {
        let n0 = source.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y);
        let n1 = source.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y);
        mesh.normals.push(((n0 + n1) * 0.5).normalize_or_zero());
    }

    if !source.uvs.is_empty() {
        let uv0 = source.uvs.get(i0 as usize).copied().unwrap_or_default();
        let uv1 = source.uvs.get(i1 as usize).copied().unwrap_or_default();
        mesh.uvs.push((uv0 + uv1) * 0.5);
    }

    edge_midpoints.insert(edge, new_idx);
    new_idx
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

/// Rips selected vertices, disconnecting them from adjacent faces.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct RipVertices;

impl RipVertices {
    /// Creates a rip vertices operation.
    pub fn new() -> Self {
        Self
    }

    /// Applies this operation to selected vertices.
    pub fn apply(&self, mesh: &Mesh, selection: &MeshSelection) -> Mesh {
        rip_vertices(mesh, selection)
    }
}

/// Rips selected vertices by duplicating them for each adjacent face.
pub fn rip_vertices(mesh: &Mesh, selection: &MeshSelection) -> Mesh {
    if selection.vertices.is_empty() {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // For each selected vertex, find all faces using it
    let mut vertex_to_faces: HashMap<u32, Vec<usize>> = HashMap::new();
    for face_idx in 0..mesh.triangle_count() {
        let base = face_idx * 3;
        for i in 0..3 {
            let v = mesh.indices[base + i];
            vertex_to_faces.entry(v).or_default().push(face_idx);
        }
    }

    // For each selected vertex with multiple faces, create duplicates
    for &v in &selection.vertices {
        if let Some(faces) = vertex_to_faces.get(&v) {
            if faces.len() < 2 {
                continue;
            }

            // Keep first face using original vertex, create new vertex for others
            for &face_idx in &faces[1..] {
                let new_idx = result.positions.len() as u32;

                result.positions.push(mesh.positions[v as usize]);
                if v < mesh.normals.len() as u32 {
                    result.normals.push(mesh.normals[v as usize]);
                }
                if v < mesh.uvs.len() as u32 {
                    result.uvs.push(mesh.uvs[v as usize]);
                }

                // Update face to use new vertex
                let base = face_idx * 3;
                for i in 0..3 {
                    if result.indices[base + i] == v {
                        result.indices[base + i] = new_idx;
                    }
                }
            }
        }
    }

    result
}

// ============================================================================
// Advanced Operations
// ============================================================================

/// Bridges two edge loops by creating connecting faces.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BridgeEdgeLoops {
    /// Number of segments in the bridge.
    pub segments: u32,
    /// Twist amount (in edge loop positions).
    pub twist: i32,
}

impl Default for BridgeEdgeLoops {
    fn default() -> Self {
        Self {
            segments: 1,
            twist: 0,
        }
    }
}

impl BridgeEdgeLoops {
    /// Creates a bridge operation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a bridge with the given segment count.
    pub fn with_segments(segments: u32) -> Self {
        Self { segments, twist: 0 }
    }

    /// Applies this operation to two edge loops.
    ///
    /// Takes two lists of vertices representing edge loops.
    pub fn apply(&self, mesh: &Mesh, loop1: &[u32], loop2: &[u32]) -> Mesh {
        bridge_edge_loops(mesh, loop1, loop2, self.segments, self.twist)
    }
}

/// Bridges two edge loops by creating connecting faces.
pub fn bridge_edge_loops(
    mesh: &Mesh,
    loop1: &[u32],
    loop2: &[u32],
    segments: u32,
    twist: i32,
) -> Mesh {
    if loop1.is_empty() || loop2.is_empty() || loop1.len() != loop2.len() {
        return mesh.clone();
    }

    let segments = segments.max(1);
    let n = loop1.len();

    let mut result = mesh.clone();

    // Apply twist to loop2
    let twist = ((twist % n as i32) + n as i32) as usize % n;
    let twisted_loop2: Vec<u32> = (0..n).map(|i| loop2[(i + twist) % n]).collect();

    if segments == 1 {
        // Direct bridge
        for i in 0..n {
            let i_next = (i + 1) % n;

            let v0 = loop1[i];
            let v1 = loop1[i_next];
            let v2 = twisted_loop2[i_next];
            let v3 = twisted_loop2[i];

            // Create quad as two triangles
            result.indices.extend_from_slice(&[v0, v1, v2]);
            result.indices.extend_from_slice(&[v0, v2, v3]);
        }
    } else {
        // Multi-segment bridge - create intermediate vertices
        let mut prev_ring = loop1.to_vec();

        for seg in 1..=segments {
            let t = seg as f32 / segments as f32;

            // Create interpolated ring
            let mut new_ring = Vec::with_capacity(n);

            for i in 0..n {
                let p1 = mesh.positions[loop1[i] as usize];
                let p2 = mesh.positions[twisted_loop2[i] as usize];
                let interp_pos = p1.lerp(p2, t);

                let new_idx = result.positions.len() as u32;
                result.positions.push(interp_pos);

                // Interpolate normals
                if !mesh.normals.is_empty() {
                    let n1 = mesh
                        .normals
                        .get(loop1[i] as usize)
                        .copied()
                        .unwrap_or(Vec3::Y);
                    let n2 = mesh
                        .normals
                        .get(twisted_loop2[i] as usize)
                        .copied()
                        .unwrap_or(Vec3::Y);
                    result.normals.push(n1.lerp(n2, t).normalize_or_zero());
                }

                // Interpolate UVs
                if !mesh.uvs.is_empty() {
                    let uv1 = mesh.uvs.get(loop1[i] as usize).copied().unwrap_or_default();
                    let uv2 = mesh
                        .uvs
                        .get(twisted_loop2[i] as usize)
                        .copied()
                        .unwrap_or_default();
                    result.uvs.push(uv1.lerp(uv2, t));
                }

                new_ring.push(if seg < segments {
                    new_idx
                } else {
                    twisted_loop2[i]
                });
            }

            // Connect previous ring to current ring
            let current_ring = if seg < segments {
                &new_ring
            } else {
                &twisted_loop2
            };

            for i in 0..n {
                let i_next = (i + 1) % n;

                let v0 = prev_ring[i];
                let v1 = prev_ring[i_next];
                let v2 = current_ring[i_next];
                let v3 = current_ring[i];

                result.indices.extend_from_slice(&[v0, v1, v2]);
                result.indices.extend_from_slice(&[v0, v2, v3]);
            }

            if seg < segments {
                prev_ring = new_ring;
            }
        }
    }

    result.compute_smooth_normals();
    result
}

/// Knife cut point on the mesh.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KnifePoint {
    /// Face index where the cut occurs.
    pub face: u32,
    /// Barycentric coordinates within the face.
    pub barycentric: [f32; 3],
}

/// Cuts the mesh along a path.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KnifeCut {
    /// Points defining the cut path.
    pub points: Vec<KnifePoint>,
}

impl Default for KnifeCut {
    fn default() -> Self {
        Self { points: Vec::new() }
    }
}

impl KnifeCut {
    /// Creates a new knife cut operation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a cut point.
    pub fn add_point(&mut self, face: u32, barycentric: [f32; 3]) {
        self.points.push(KnifePoint { face, barycentric });
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        knife_cut(mesh, &self.points)
    }
}

/// Cuts the mesh along a path defined by points on faces.
pub fn knife_cut(mesh: &Mesh, points: &[KnifePoint]) -> Mesh {
    if points.len() < 2 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Simple implementation: add vertices at cut points and split affected faces
    let mut face_cuts: HashMap<u32, Vec<(u32, [f32; 3])>> = HashMap::new();

    for (idx, point) in points.iter().enumerate() {
        face_cuts
            .entry(point.face)
            .or_default()
            .push((idx as u32, point.barycentric));
    }

    // Process each face with cuts
    let mut new_indices = Vec::new();
    let mut processed_faces: HashSet<u32> = HashSet::new();

    for (&face_idx, cuts) in &face_cuts {
        processed_faces.insert(face_idx);

        let base = face_idx as usize * 3;
        if base + 2 >= mesh.indices.len() {
            continue;
        }

        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        let p0 = mesh.positions[i0 as usize];
        let p1 = mesh.positions[i1 as usize];
        let p2 = mesh.positions[i2 as usize];

        // Create vertices for cut points
        let mut cut_vertex_indices = Vec::new();
        for &(_, bary) in cuts {
            let cut_pos = p0 * bary[0] + p1 * bary[1] + p2 * bary[2];
            let cut_idx = result.positions.len() as u32;
            result.positions.push(cut_pos);

            // Interpolate normal
            if !mesh.normals.is_empty() {
                let n0 = mesh.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y);
                let n1 = mesh.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y);
                let n2 = mesh.normals.get(i2 as usize).copied().unwrap_or(Vec3::Y);
                result
                    .normals
                    .push((n0 * bary[0] + n1 * bary[1] + n2 * bary[2]).normalize_or_zero());
            }

            // Interpolate UV
            if !mesh.uvs.is_empty() {
                let uv0 = mesh.uvs.get(i0 as usize).copied().unwrap_or_default();
                let uv1 = mesh.uvs.get(i1 as usize).copied().unwrap_or_default();
                let uv2 = mesh.uvs.get(i2 as usize).copied().unwrap_or_default();
                result
                    .uvs
                    .push(uv0 * bary[0] + uv1 * bary[1] + uv2 * bary[2]);
            }

            cut_vertex_indices.push(cut_idx);
        }

        // Simple triangulation: fan from first vertex through cut points
        if cut_vertex_indices.len() == 1 {
            // Single cut point - create 3 triangles
            let cut = cut_vertex_indices[0];
            new_indices.extend_from_slice(&[i0, i1, cut]);
            new_indices.extend_from_slice(&[i1, i2, cut]);
            new_indices.extend_from_slice(&[i2, i0, cut]);
        } else {
            // Multiple cut points - more complex triangulation needed
            // For now, just keep original face (proper implementation would be complex)
            new_indices.extend_from_slice(&[i0, i1, i2]);
        }
    }

    // Copy unaffected faces
    for face_idx in 0..mesh.triangle_count() {
        if !processed_faces.contains(&(face_idx as u32)) {
            let base = face_idx * 3;
            new_indices.extend_from_slice(&mesh.indices[base..base + 3]);
        }
    }

    result.indices = new_indices;
    result
}

// ============================================================================
// Helper Functions
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;
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
    fn test_delete_faces_single() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = delete_faces(&mesh, &selection, true);

        assert_eq!(result.triangle_count(), 1);
        // Should only have vertices needed for remaining face
        assert!(result.vertex_count() <= 4);
    }

    #[test]
    fn test_delete_faces_all() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_all_faces(&mesh);

        let result = delete_faces(&mesh, &selection, true);

        assert_eq!(result.triangle_count(), 0);
        assert_eq!(result.vertex_count(), 0);
    }

    #[test]
    fn test_transform_vertices() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);

        let transform = TransformVertices::translate(Vec3::new(1.0, 0.0, 0.0));
        let result = transform.apply(&mesh, &selection);

        assert_eq!(result.positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(result.positions[1], mesh.positions[1]); // Unchanged
    }

    #[test]
    fn test_poke_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = poke_faces(&mesh, &selection, 0.0);

        // 1 triangle becomes 3 triangles
        assert_eq!(result.triangle_count(), 3);
        // Added 1 center vertex
        assert_eq!(result.vertex_count(), 4);
    }

    #[test]
    fn test_scale_faces() {
        let cube = Cuboid::default().apply();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = scale_faces(&cube, &selection, 0.5);

        // Should have more vertices due to face duplication
        assert!(result.vertex_count() >= cube.vertex_count());
    }

    #[test]
    fn test_smooth_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);

        let result = smooth_vertices(&mesh, &selection, 0.5, 1);

        // Position should have moved toward neighbors
        assert_ne!(result.positions[0], mesh.positions[0]);
        // Other vertices unchanged
        assert_eq!(result.positions[2], mesh.positions[2]);
    }

    #[test]
    fn test_merge_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);
        selection.select_vertex(1);

        let result = merge_vertices(&mesh, &selection, MergeMode::Center, Vec3::ZERO);

        // Should have fewer vertices
        assert!(result.vertex_count() < mesh.vertex_count());
    }

    #[test]
    fn test_extrude_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = extrude_faces(&mesh, &selection, 1.0);

        // Should have more vertices and triangles
        assert!(result.vertex_count() > mesh.vertex_count());
        assert!(result.triangle_count() > mesh.triangle_count());
    }

    #[test]
    fn test_inset_faces_individual() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = inset_faces_individual(&mesh, &selection, 0.3);

        // 1 inner + 3 bridge quads (6 triangles) = 7 triangles
        assert_eq!(result.triangle_count(), 7);
    }

    #[test]
    fn test_subdivide_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = subdivide_faces(&mesh, &selection, 1);

        // 1 triangle becomes 4 triangles
        assert_eq!(result.triangle_count(), 4);
    }

    #[test]
    fn test_rip_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0); // Shared vertex

        let result = rip_vertices(&mesh, &selection);

        // Vertex 0 should be duplicated
        assert!(result.vertex_count() > mesh.vertex_count());
    }

    #[test]
    fn test_bridge_edge_loops() {
        // Create two simple triangular loops (3 vertices each)
        let test_mesh = Mesh {
            positions: vec![
                // Loop 1 (at z=0)
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                // Loop 2 (at z=1)
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.5, 1.0, 1.0),
            ],
            normals: vec![Vec3::Z; 6],
            uvs: vec![Vec2::ZERO; 6],
            indices: vec![],
        };

        let loop1 = vec![0, 1, 2];
        let loop2 = vec![3, 4, 5];

        let result = bridge_edge_loops(&test_mesh, &loop1, &loop2, 1, 0);

        // Should create 3 quads = 6 triangles
        assert_eq!(result.triangle_count(), 6);
    }
}
