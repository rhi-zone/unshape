//! Mesh operations for procedural geometry.
//!
//! Operations are serializable structs with `apply` methods. Free functions
//! and method sugar delegate to these ops. See `docs/design/ops-as-values.md`.

use std::collections::HashMap;

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Extrudes a mesh along vertex normals.
///
/// Creates a shell by duplicating all vertices and offsetting them along
/// their normals, then connecting old and new vertices with side faces.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct Extrude {
    /// Distance to extrude (positive = outward, negative = inward).
    pub amount: f32,
    /// Whether to create side faces connecting old and new vertices.
    pub create_sides: bool,
    /// Whether to keep the original faces.
    pub keep_original: bool,
}

impl Default for Extrude {
    fn default() -> Self {
        Self {
            amount: 1.0,
            create_sides: true,
            keep_original: false,
        }
    }
}

impl Extrude {
    /// Creates a new extrude operation with the given amount.
    pub fn new(amount: f32) -> Self {
        Self {
            amount,
            ..Default::default()
        }
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        extrude_with_config(mesh, *self)
    }
}

/// Backwards-compatible type alias.
pub type ExtrudeConfig = Extrude;

/// Extrudes the entire mesh along vertex normals.
///
/// Creates a shell by duplicating all vertices and offsetting them along
/// their normals, then connecting old and new vertices with side faces.
pub fn extrude(mesh: &Mesh, amount: f32) -> Mesh {
    extrude_with_config(
        mesh,
        ExtrudeConfig {
            amount,
            ..Default::default()
        },
    )
}

/// Extrudes with full configuration options.
pub fn extrude_with_config(mesh: &Mesh, config: ExtrudeConfig) -> Mesh {
    let vertex_count = mesh.positions.len();
    let mut result = Mesh::with_capacity(vertex_count * 2, mesh.triangle_count() * 3);

    // Compute normals if not present
    let normals: Vec<Vec3> = if mesh.normals.len() == vertex_count {
        mesh.normals.clone()
    } else {
        compute_vertex_normals(mesh)
    };

    // Copy original vertices
    result.positions.extend_from_slice(&mesh.positions);
    result.normals.extend_from_slice(&normals);

    // Create extruded vertices
    for i in 0..vertex_count {
        let extruded_pos = mesh.positions[i] + normals[i] * config.amount;
        result.positions.push(extruded_pos);
        result.normals.push(normals[i]);
    }

    // Copy UVs for both original and extruded vertices
    if !mesh.uvs.is_empty() {
        result.uvs.extend_from_slice(&mesh.uvs);
        result.uvs.extend_from_slice(&mesh.uvs);
    }

    // Add original faces (with reversed winding if not keeping original)
    if config.keep_original {
        // Keep original faces pointing inward
        for tri in mesh.indices.chunks(3) {
            result.indices.push(tri[2]);
            result.indices.push(tri[1]);
            result.indices.push(tri[0]);
        }
    }

    // Add extruded faces (with same winding as original, offset by vertex_count)
    for tri in mesh.indices.chunks(3) {
        let offset = vertex_count as u32;
        result.indices.push(tri[0] + offset);
        result.indices.push(tri[1] + offset);
        result.indices.push(tri[2] + offset);
    }

    // Create side faces if requested
    if config.create_sides {
        // Build edge map: edge -> [triangle indices using it]
        let edges = collect_boundary_edges(mesh);

        for (i0, i1) in edges {
            let offset = vertex_count as u32;

            // Create quad connecting original and extruded edges
            // i0 -> i1 (original edge)
            // i0+offset -> i1+offset (extruded edge)
            // Quad: (i0, i1, i1+offset, i0+offset)

            // First triangle
            result.indices.push(i0);
            result.indices.push(i1);
            result.indices.push(i1 + offset);

            // Second triangle
            result.indices.push(i0);
            result.indices.push(i1 + offset);
            result.indices.push(i0 + offset);
        }
    }

    result
}

/// Solidifies a mesh by adding thickness.
///
/// Creates a solid shell from a surface mesh by extruding inward and
/// creating side walls at boundaries.
pub fn solidify(mesh: &Mesh, thickness: f32) -> Mesh {
    extrude_with_config(
        mesh,
        ExtrudeConfig {
            amount: -thickness,
            create_sides: true,
            keep_original: true,
        },
    )
}

/// Insets all faces toward their centers.
///
/// Creates a new inner face for each triangle, connected by bridge faces.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct Inset {
    /// Inset amount (0.0 = no change, 1.0 = shrink to center).
    pub amount: f32,
    /// Optional depth (positive = extrude inward after inset).
    pub depth: f32,
    /// Whether to create the connecting faces.
    pub create_bridge: bool,
}

impl Default for Inset {
    fn default() -> Self {
        Self {
            amount: 0.2,
            depth: 0.0,
            create_bridge: true,
        }
    }
}

impl Inset {
    /// Creates a new inset operation with the given amount.
    pub fn new(amount: f32) -> Self {
        Self {
            amount,
            ..Default::default()
        }
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        inset_with_config(mesh, *self)
    }
}

/// Backwards-compatible type alias.
pub type InsetConfig = Inset;

/// Insets all faces toward their centers.
///
/// Creates a new inner face for each triangle, connected by bridge faces.
pub fn inset(mesh: &Mesh, amount: f32) -> Mesh {
    inset_with_config(
        mesh,
        InsetConfig {
            amount,
            ..Default::default()
        },
    )
}

/// Insets faces with full configuration.
pub fn inset_with_config(mesh: &Mesh, config: InsetConfig) -> Mesh {
    let triangle_count = mesh.triangle_count();
    // Each triangle becomes: 1 center triangle + 3 bridge quads (6 triangles) = 7 triangles
    let estimated_triangles = triangle_count * 7;
    let estimated_vertices = triangle_count * 6; // 3 original + 3 inset per face

    let mut result = Mesh::with_capacity(estimated_vertices, estimated_triangles);

    // Compute face normals for depth extrusion
    let face_normals = compute_face_normals(mesh);

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Face center
        let center = (p0 + p1 + p2) / 3.0;

        // Inset positions (lerp toward center)
        let depth_offset = face_normals[face_idx] * -config.depth;
        let ip0 = p0.lerp(center, config.amount) + depth_offset;
        let ip1 = p1.lerp(center, config.amount) + depth_offset;
        let ip2 = p2.lerp(center, config.amount) + depth_offset;

        // Get normals
        let normal = face_normals[face_idx];

        // Add original vertices
        let base = result.positions.len() as u32;
        result.positions.push(p0);
        result.positions.push(p1);
        result.positions.push(p2);
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        // Add inset vertices
        result.positions.push(ip0);
        result.positions.push(ip1);
        result.positions.push(ip2);
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        // Indices: base+0,1,2 = original, base+3,4,5 = inset

        // Inner triangle (inset face)
        result.indices.push(base + 3);
        result.indices.push(base + 4);
        result.indices.push(base + 5);

        if config.create_bridge {
            // Bridge quads connecting original to inset edges
            // Edge 0-1: (0, 1, 4, 3)
            result.indices.push(base);
            result.indices.push(base + 1);
            result.indices.push(base + 4);

            result.indices.push(base);
            result.indices.push(base + 4);
            result.indices.push(base + 3);

            // Edge 1-2: (1, 2, 5, 4)
            result.indices.push(base + 1);
            result.indices.push(base + 2);
            result.indices.push(base + 5);

            result.indices.push(base + 1);
            result.indices.push(base + 5);
            result.indices.push(base + 4);

            // Edge 2-0: (2, 0, 3, 5)
            result.indices.push(base + 2);
            result.indices.push(base);
            result.indices.push(base + 3);

            result.indices.push(base + 2);
            result.indices.push(base + 3);
            result.indices.push(base + 5);
        }
    }

    // Copy UVs if present (simple copy for now, could interpolate)
    if mesh.has_uvs() {
        for tri in mesh.indices.chunks(3) {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            // Original vertices
            result.uvs.push(mesh.uvs[i0]);
            result.uvs.push(mesh.uvs[i1]);
            result.uvs.push(mesh.uvs[i2]);
            // Inset vertices (lerp toward center)
            let uv_center = (mesh.uvs[i0] + mesh.uvs[i1] + mesh.uvs[i2]) / 3.0;
            result.uvs.push(mesh.uvs[i0].lerp(uv_center, config.amount));
            result.uvs.push(mesh.uvs[i1].lerp(uv_center, config.amount));
            result.uvs.push(mesh.uvs[i2].lerp(uv_center, config.amount));
        }
    }

    result
}

/// Flips the winding order of all faces (reverses normals).
pub fn flip_normals(mesh: &Mesh) -> Mesh {
    let mut result = mesh.clone();

    // Reverse each triangle's winding
    for tri in result.indices.chunks_mut(3) {
        tri.swap(0, 2);
    }

    // Flip normals
    for normal in &mut result.normals {
        *normal = -*normal;
    }

    result
}

/// Duplicates the mesh with reversed normals for double-sided rendering.
pub fn make_double_sided(mesh: &Mesh) -> Mesh {
    let mut result = mesh.clone();
    let flipped = flip_normals(mesh);
    result.merge(&flipped);
    result
}

/// Welds vertices that are within the given distance.
///
/// Reduces vertex count by merging coincident vertices.
pub fn weld_vertices(mesh: &Mesh, tolerance: f32) -> Mesh {
    let tol_sq = tolerance * tolerance;

    // Map old vertex indices to new (welded) indices
    let mut vertex_map: Vec<usize> = Vec::with_capacity(mesh.positions.len());
    let mut new_positions: Vec<Vec3> = Vec::new();
    let mut new_normals: Vec<Vec3> = Vec::new();
    let mut new_uvs: Vec<glam::Vec2> = Vec::new();

    // Accumulated normals for averaging
    let mut normal_accum: Vec<Vec3> = Vec::new();
    let mut normal_counts: Vec<usize> = Vec::new();

    for (i, pos) in mesh.positions.iter().enumerate() {
        // Find existing vertex within tolerance
        let mut found = None;
        for (j, existing) in new_positions.iter().enumerate() {
            if pos.distance_squared(*existing) <= tol_sq {
                found = Some(j);
                break;
            }
        }

        if let Some(j) = found {
            vertex_map.push(j);
            // Accumulate normal for averaging
            if i < mesh.normals.len() {
                normal_accum[j] += mesh.normals[i];
                normal_counts[j] += 1;
            }
        } else {
            vertex_map.push(new_positions.len());
            new_positions.push(*pos);

            if i < mesh.normals.len() {
                normal_accum.push(mesh.normals[i]);
                new_normals.push(mesh.normals[i]);
                normal_counts.push(1);
            }

            if i < mesh.uvs.len() {
                new_uvs.push(mesh.uvs[i]);
            }
        }
    }

    // Average accumulated normals
    for (i, normal) in new_normals.iter_mut().enumerate() {
        if normal_counts[i] > 1 {
            *normal = (normal_accum[i] / normal_counts[i] as f32).normalize_or_zero();
        }
    }

    // Remap indices
    let new_indices: Vec<u32> = mesh
        .indices
        .iter()
        .map(|&i| vertex_map[i as usize] as u32)
        .collect();

    Mesh {
        positions: new_positions,
        normals: new_normals,
        uvs: new_uvs,
        indices: new_indices,
    }
}

/// Splits all faces so each face has its own vertices (no sharing).
///
/// Useful for flat shading or per-face attributes.
pub fn split_faces(mesh: &Mesh) -> Mesh {
    let triangle_count = mesh.triangle_count();
    let mut result = Mesh::with_capacity(triangle_count * 3, triangle_count);

    let face_normals = compute_face_normals(mesh);

    for (face_idx, tri) in mesh.indices.chunks(3).enumerate() {
        let base = result.positions.len() as u32;

        for &idx in tri {
            let i = idx as usize;
            result.positions.push(mesh.positions[i]);

            if !mesh.uvs.is_empty() {
                result.uvs.push(mesh.uvs[i]);
            }
        }

        // Use face normal for all three vertices
        let normal = face_normals[face_idx];
        result.normals.push(normal);
        result.normals.push(normal);
        result.normals.push(normal);

        result.indices.push(base);
        result.indices.push(base + 1);
        result.indices.push(base + 2);
    }

    result
}

/// Recalculates normals using the specified mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalMode {
    /// Flat shading - each face has a uniform normal.
    Flat,
    /// Smooth shading - normals are averaged at shared vertices.
    Smooth,
    /// Area-weighted smooth shading.
    AreaWeighted,
}

/// Recalculates mesh normals.
pub fn recalculate_normals(mesh: &Mesh, mode: NormalMode) -> Mesh {
    let mut result = mesh.clone();

    match mode {
        NormalMode::Flat => {
            // Split faces first for proper flat normals
            result = split_faces(&result);
        }
        NormalMode::Smooth | NormalMode::AreaWeighted => {
            result.compute_smooth_normals();
        }
    }

    result
}

/// Applies Laplacian smoothing to a mesh.
///
/// Moves each vertex towards the centroid of its neighbors, creating
/// a smoother surface. Multiple iterations increase the smoothing effect.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Mesh))]
pub struct Smooth {
    /// Smoothing factor per iteration (0.0 = no change, 1.0 = move to average).
    /// Values between 0.3-0.5 are typical.
    pub lambda: f32,
    /// Number of smoothing iterations.
    pub iterations: usize,
    /// Whether to preserve boundary vertices (don't move them).
    pub preserve_boundary: bool,
}

impl Default for Smooth {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            iterations: 1,
            preserve_boundary: true,
        }
    }
}

impl Smooth {
    /// Creates a new smooth operation with the given parameters.
    pub fn new(lambda: f32, iterations: usize) -> Self {
        Self {
            lambda,
            iterations,
            ..Default::default()
        }
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        smooth_with_config(mesh, *self)
    }
}

/// Backwards-compatible type alias.
pub type SmoothConfig = Smooth;

/// Applies Laplacian smoothing to a mesh.
///
/// Moves each vertex towards the centroid of its neighbors, creating
/// a smoother surface. Multiple iterations increase the smoothing effect.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_mesh::{Cuboid, smooth};
///
/// let cube = Cuboid::default().apply();
/// let smoothed = smooth(&cube, 0.5, 3);
/// ```
pub fn smooth(mesh: &Mesh, lambda: f32, iterations: usize) -> Mesh {
    smooth_with_config(
        mesh,
        SmoothConfig {
            lambda,
            iterations,
            ..Default::default()
        },
    )
}

/// Applies Laplacian smoothing with full configuration.
pub fn smooth_with_config(mesh: &Mesh, config: SmoothConfig) -> Mesh {
    if config.iterations == 0 || config.lambda == 0.0 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Build adjacency list: for each vertex, which vertices are connected to it
    let adjacency = build_adjacency(&result);

    // Find boundary vertices if we need to preserve them
    let boundary = if config.preserve_boundary {
        find_boundary_vertices(&result)
    } else {
        vec![false; result.positions.len()]
    };

    // Apply smoothing iterations
    for _ in 0..config.iterations {
        let mut new_positions = result.positions.clone();

        for (i, neighbors) in adjacency.iter().enumerate() {
            // Skip boundary vertices if preserving them
            if config.preserve_boundary && boundary[i] {
                continue;
            }

            // Skip vertices with no neighbors
            if neighbors.is_empty() {
                continue;
            }

            // Compute centroid of neighbors
            let mut centroid = Vec3::ZERO;
            for &neighbor in neighbors {
                centroid += result.positions[neighbor];
            }
            centroid /= neighbors.len() as f32;

            // Move vertex towards centroid
            new_positions[i] = result.positions[i].lerp(centroid, config.lambda);
        }

        result.positions = new_positions;
    }

    // Recalculate normals after smoothing
    result.compute_smooth_normals();

    result
}

/// Applies Taubin smoothing (avoids shrinkage).
///
/// Taubin smoothing alternates between positive and negative lambda values
/// to smooth the mesh while minimizing volume loss.
pub fn smooth_taubin(mesh: &Mesh, lambda: f32, mu: f32, iterations: usize) -> Mesh {
    if iterations == 0 {
        return mesh.clone();
    }

    let mut result = mesh.clone();
    let adjacency = build_adjacency(&result);
    let boundary = find_boundary_vertices(&result);

    for _ in 0..iterations {
        // Forward pass with positive lambda
        result = smooth_pass(&result, &adjacency, &boundary, lambda);
        // Backward pass with negative mu (typically mu = -lambda - small_value)
        result = smooth_pass(&result, &adjacency, &boundary, mu);
    }

    result.compute_smooth_normals();
    result
}

/// Single smoothing pass (used by Taubin smoothing).
fn smooth_pass(mesh: &Mesh, adjacency: &[Vec<usize>], boundary: &[bool], lambda: f32) -> Mesh {
    let mut result = mesh.clone();
    let mut new_positions = mesh.positions.clone();

    for (i, neighbors) in adjacency.iter().enumerate() {
        if boundary[i] || neighbors.is_empty() {
            continue;
        }

        let mut centroid = Vec3::ZERO;
        for &neighbor in neighbors {
            centroid += mesh.positions[neighbor];
        }
        centroid /= neighbors.len() as f32;

        new_positions[i] = mesh.positions[i].lerp(centroid, lambda);
    }

    result.positions = new_positions;
    result
}

/// Builds an adjacency list for the mesh vertices.
fn build_adjacency(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        // Add bidirectional connections
        if !adjacency[i0].contains(&i1) {
            adjacency[i0].push(i1);
        }
        if !adjacency[i1].contains(&i0) {
            adjacency[i1].push(i0);
        }

        if !adjacency[i1].contains(&i2) {
            adjacency[i1].push(i2);
        }
        if !adjacency[i2].contains(&i1) {
            adjacency[i2].push(i1);
        }

        if !adjacency[i2].contains(&i0) {
            adjacency[i2].push(i0);
        }
        if !adjacency[i0].contains(&i2) {
            adjacency[i0].push(i2);
        }
    }

    adjacency
}

/// Finds boundary vertices (vertices on edges that only belong to one face).
fn find_boundary_vertices(mesh: &Mesh) -> Vec<bool> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    // Count edge occurrences
    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let edges = [(i0, i1), (i1, i2), (i2, i0)];

        for (a, b) in edges {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // Mark vertices on boundary edges
    let mut boundary = vec![false; mesh.positions.len()];
    for ((a, b), count) in edge_count {
        if count == 1 {
            boundary[a] = true;
            boundary[b] = true;
        }
    }

    boundary
}

// ============================================================================
// Helper functions
// ============================================================================

/// Computes smooth vertex normals from face normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let normal = (v1 - v0).cross(v2 - v0); // unnormalized = area-weighted
        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    normals
}

/// Computes face normals (one per triangle).
fn compute_face_normals(mesh: &Mesh) -> Vec<Vec3> {
    mesh.indices
        .chunks(3)
        .map(|tri| {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];
            (v1 - v0).cross(v2 - v0).normalize_or_zero()
        })
        .collect()
}

/// Collects all edges from the mesh.
///
/// Returns edges as (lower_index, higher_index) pairs for deduplication.
fn collect_boundary_edges(mesh: &Mesh) -> Vec<(u32, u32)> {
    // Count how many times each edge appears
    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

        for (a, b) in edges {
            // Normalize edge direction for counting
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // Return all edges (for extrusion we want all edges, not just boundary)
    // Keep original winding order for correct face generation
    let mut result = Vec::new();
    for tri in mesh.indices.chunks(3) {
        result.push((tri[0], tri[1]));
        result.push((tri[1], tri[2]));
        result.push((tri[2], tri[0]));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_extrude_basic() {
        let cube = Cuboid::default().apply();
        let extruded = extrude(&cube, 0.5);

        // Should have more vertices and triangles
        assert!(extruded.vertex_count() > cube.vertex_count());
        assert!(extruded.triangle_count() > cube.triangle_count());
    }

    #[test]
    fn test_solidify() {
        let cube = Cuboid::default().apply();
        let solid = solidify(&cube, 0.1);

        // Solidify creates inner and outer shell plus sides
        assert!(solid.vertex_count() > cube.vertex_count());
    }

    #[test]
    fn test_inset() {
        let cube = Cuboid::default().apply();
        let inseted = inset(&cube, 0.3);

        // Inset creates many more triangles
        assert!(inseted.triangle_count() > cube.triangle_count());
    }

    #[test]
    fn test_flip_normals() {
        let cube = Cuboid::default().apply();
        let flipped = flip_normals(&cube);

        // Same vertex/triangle count
        assert_eq!(flipped.vertex_count(), cube.vertex_count());
        assert_eq!(flipped.triangle_count(), cube.triangle_count());

        // Normals should be reversed
        for (orig, flip) in cube.normals.iter().zip(flipped.normals.iter()) {
            assert!((orig + flip).length() < 0.001);
        }
    }

    #[test]
    fn test_double_sided() {
        let cube = Cuboid::default().apply();
        let doubled = make_double_sided(&cube);

        // Should have exactly double the geometry
        assert_eq!(doubled.vertex_count(), cube.vertex_count() * 2);
        assert_eq!(doubled.triangle_count(), cube.triangle_count() * 2);
    }

    #[test]
    fn test_weld_vertices() {
        // Create a mesh with duplicate vertices
        let mut mesh = Mesh::new();
        mesh.positions.push(Vec3::new(0.0, 0.0, 0.0));
        mesh.positions.push(Vec3::new(1.0, 0.0, 0.0));
        mesh.positions.push(Vec3::new(0.5, 1.0, 0.0));
        mesh.positions.push(Vec3::new(0.0001, 0.0, 0.0)); // Near duplicate of first
        mesh.positions.push(Vec3::new(1.0001, 0.0, 0.0)); // Near duplicate of second
        mesh.positions.push(Vec3::new(0.5, 1.0, 0.5));

        mesh.indices = vec![0, 1, 2, 3, 4, 5];

        let welded = weld_vertices(&mesh, 0.01);

        // Should have fewer vertices due to merging
        assert!(welded.vertex_count() < mesh.vertex_count());
    }

    #[test]
    fn test_split_faces() {
        let cube = Cuboid::default().apply();
        let split = split_faces(&cube);

        // Each triangle gets 3 unique vertices
        assert_eq!(split.vertex_count(), cube.triangle_count() * 3);
    }

    #[test]
    fn test_inset_with_depth() {
        let cube = Cuboid::default().apply();
        let inseted = inset_with_config(
            &cube,
            InsetConfig {
                amount: 0.3,
                depth: 0.2,
                create_bridge: true,
            },
        );

        // Should have the expected structure
        assert!(inseted.triangle_count() > cube.triangle_count());
        assert!(inseted.has_normals());
    }

    #[test]
    fn test_smooth_basic() {
        let cube = Cuboid::default().apply();
        // Use preserve_boundary: false since box_mesh has split vertices
        let smoothed = smooth_with_config(
            &cube,
            SmoothConfig {
                lambda: 0.5,
                iterations: 1,
                preserve_boundary: false,
            },
        );

        // Same vertex count, positions changed
        assert_eq!(smoothed.vertex_count(), cube.vertex_count());

        // At least some vertices should have moved
        let mut any_moved = false;
        for (orig, new) in cube.positions.iter().zip(smoothed.positions.iter()) {
            if orig.distance(*new) > 0.001 {
                any_moved = true;
                break;
            }
        }
        assert!(any_moved, "Smoothing should move some vertices");
    }

    #[test]
    fn test_smooth_zero_iterations() {
        let cube = Cuboid::default().apply();
        let smoothed = smooth(&cube, 0.5, 0);

        // Zero iterations should return identical mesh
        for (orig, new) in cube.positions.iter().zip(smoothed.positions.iter()) {
            assert_eq!(*orig, *new);
        }
    }

    #[test]
    fn test_smooth_zero_lambda() {
        let cube = Cuboid::default().apply();
        let smoothed = smooth(&cube, 0.0, 5);

        // Zero lambda should return identical mesh
        for (orig, new) in cube.positions.iter().zip(smoothed.positions.iter()) {
            assert_eq!(*orig, *new);
        }
    }

    #[test]
    fn test_smooth_with_config() {
        let cube = Cuboid::default().apply();
        let smoothed = smooth_with_config(
            &cube,
            SmoothConfig {
                lambda: 0.3,
                iterations: 2,
                preserve_boundary: false,
            },
        );

        // More iterations = more smoothing
        assert_eq!(smoothed.vertex_count(), cube.vertex_count());
        assert!(smoothed.has_normals());
    }

    #[test]
    fn test_smooth_taubin() {
        let cube = Cuboid::default().apply();
        // Typical Taubin parameters: lambda = 0.5, mu = -0.53
        let smoothed = smooth_taubin(&cube, 0.5, -0.53, 3);

        // Same vertex count
        assert_eq!(smoothed.vertex_count(), cube.vertex_count());
        assert!(smoothed.has_normals());
    }

    #[test]
    fn test_smooth_preserves_normals() {
        let cube = Cuboid::default().apply();
        let smoothed = smooth(&cube, 0.5, 2);

        // Should still have valid normals
        assert_eq!(smoothed.normals.len(), smoothed.positions.len());
        for normal in &smoothed.normals {
            let len = normal.length();
            assert!(
                (len - 1.0).abs() < 0.001 || len < 0.001,
                "Normal should be normalized or zero"
            );
        }
    }
}

#[cfg(all(test, feature = "dynop"))]
mod dynop_tests {
    use super::*;
    use crate::Cuboid;
    use rhizome_resin_op::{DynOp, OpRegistry, OpType, OpValue};

    #[test]
    fn test_smooth_dynop_roundtrip() {
        // Create an op
        let op = Smooth::new(0.5, 3);

        // Serialize to JSON
        let params = op.params();
        let type_name = op.type_name();

        // Deserialize via registry
        let mut registry = OpRegistry::new();
        crate::register_ops(&mut registry);

        let deserialized = registry.deserialize(type_name, params).unwrap();
        assert_eq!(deserialized.type_name(), "resin::Smooth");

        // Execute both and compare
        let cube = Cuboid::default().apply();
        let direct_result = op.apply(&cube);

        let input = OpValue::new(OpType::of::<Mesh>("Mesh"), cube);
        let dyn_result = deserialized.apply_dyn(input).unwrap();
        let dyn_mesh: Mesh = dyn_result.downcast().unwrap();

        assert_eq!(direct_result.positions.len(), dyn_mesh.positions.len());
        assert_eq!(direct_result.indices.len(), dyn_mesh.indices.len());
    }

    #[test]
    fn test_extrude_dynop_roundtrip() {
        let op = Extrude::new(0.2);
        let params = op.params();

        let mut registry = OpRegistry::new();
        crate::register_ops(&mut registry);

        let deserialized = registry.deserialize("resin::Extrude", params).unwrap();

        let cube = Cuboid::default().apply();
        let direct_result = op.apply(&cube);

        let input = OpValue::new(OpType::of::<Mesh>("Mesh"), cube);
        let dyn_result = deserialized.apply_dyn(input).unwrap();
        let dyn_mesh: Mesh = dyn_result.downcast().unwrap();

        assert_eq!(direct_result.positions.len(), dyn_mesh.positions.len());
    }

    #[test]
    fn test_inset_dynop_roundtrip() {
        let op = Inset::new(0.3);
        let params = op.params();

        let mut registry = OpRegistry::new();
        crate::register_ops(&mut registry);

        let deserialized = registry.deserialize("resin::Inset", params).unwrap();

        let cube = Cuboid::default().apply();
        let direct_result = op.apply(&cube);

        let input = OpValue::new(OpType::of::<Mesh>("Mesh"), cube);
        let dyn_result = deserialized.apply_dyn(input).unwrap();
        let dyn_mesh: Mesh = dyn_result.downcast().unwrap();

        assert_eq!(direct_result.positions.len(), dyn_mesh.positions.len());
    }

    #[test]
    fn test_op_type_info() {
        let smooth = Smooth::new(0.5, 3);
        assert_eq!(smooth.type_name(), "resin::Smooth");
        assert_eq!(smooth.input_type().name, "Mesh");
        assert_eq!(smooth.output_type().name, "Mesh");

        let extrude = Extrude::new(0.2);
        assert_eq!(extrude.type_name(), "resin::Extrude");
        assert_eq!(extrude.input_type().name, "Mesh");
        assert_eq!(extrude.output_type().name, "Mesh");
    }

    #[test]
    fn test_pipeline_execution() {
        use rhizome_resin_op::Pipeline;

        let mut registry = OpRegistry::new();
        crate::register_ops(&mut registry);

        // Build a pipeline
        let mut pipeline = Pipeline::new();
        pipeline.push(&Smooth::new(0.3, 2));
        pipeline.push(&Extrude::new(0.1));

        // Validate
        let (input_type, output_type) = pipeline.validate(&registry).unwrap();
        assert_eq!(input_type.name, "Mesh");
        assert_eq!(output_type.name, "Mesh");

        // Execute
        let cube = Cuboid::default().apply();
        let input = OpValue::new(OpType::of::<Mesh>("Mesh"), cube.clone());
        let output = pipeline.execute(input, &registry).unwrap();
        let result: Mesh = output.downcast().unwrap();

        // Compare with direct execution
        let direct = Extrude::new(0.1).apply(&Smooth::new(0.3, 2).apply(&cube));
        assert_eq!(result.positions.len(), direct.positions.len());
    }
}
