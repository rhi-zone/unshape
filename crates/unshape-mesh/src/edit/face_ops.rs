use std::collections::{HashMap, HashSet};

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;
use crate::selection::{Edge, MeshSelection};

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
