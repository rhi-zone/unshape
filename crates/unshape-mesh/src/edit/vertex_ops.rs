use std::collections::{HashMap, HashSet};

use glam::{Mat4, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;
use crate::selection::{MeshSelection, SoftSelection};

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
