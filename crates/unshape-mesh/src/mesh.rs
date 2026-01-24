//! Core mesh types.

use glam::{Vec2, Vec3};
use unshape_geometry::{HasIndices, HasNormals, HasPositions, HasUVs};

/// A 3D mesh with indexed triangle topology.
///
/// Currently uses indexed representation for simplicity.
/// Half-edge representation will be added for topology operations.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals (per-vertex, not per-face).
    pub normals: Vec<Vec3>,
    /// Texture coordinates.
    pub uvs: Vec<Vec2>,
    /// Triangle indices (every 3 indices form a triangle).
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Creates an empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mesh with pre-allocated capacity.
    pub fn with_capacity(vertices: usize, triangles: usize) -> Self {
        Self {
            positions: Vec::with_capacity(vertices),
            normals: Vec::with_capacity(vertices),
            uvs: Vec::with_capacity(vertices),
            indices: Vec::with_capacity(triangles * 3),
        }
    }

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns the number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns true if the mesh has normals.
    pub fn has_normals(&self) -> bool {
        self.normals.len() == self.positions.len()
    }

    /// Returns true if the mesh has UVs.
    pub fn has_uvs(&self) -> bool {
        self.uvs.len() == self.positions.len()
    }

    /// Estimates the memory usage of this mesh in bytes.
    pub fn memory_estimate(&self) -> usize {
        use std::mem::size_of;
        self.positions.len() * size_of::<Vec3>()
            + self.normals.len() * size_of::<Vec3>()
            + self.uvs.len() * size_of::<Vec2>()
            + self.indices.len() * size_of::<u32>()
    }

    /// Computes flat normals from triangle geometry.
    ///
    /// Each vertex gets the normal of its triangle. Vertices shared
    /// between triangles will have the normal of the last triangle.
    /// For smooth normals, use `compute_smooth_normals`.
    pub fn compute_flat_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);

        for tri in self.indices.chunks(3) {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let v0 = self.positions[i0];
            let v1 = self.positions[i1];
            let v2 = self.positions[i2];

            let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();

            self.normals[i0] = normal;
            self.normals[i1] = normal;
            self.normals[i2] = normal;
        }
    }

    /// Computes smooth normals by averaging adjacent face normals.
    pub fn compute_smooth_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);

        // Accumulate face normals at each vertex
        for tri in self.indices.chunks(3) {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let v0 = self.positions[i0];
            let v1 = self.positions[i1];
            let v2 = self.positions[i2];

            let normal = (v1 - v0).cross(v2 - v0); // unnormalized = area-weighted

            self.normals[i0] += normal;
            self.normals[i1] += normal;
            self.normals[i2] += normal;
        }

        // Normalize accumulated normals
        for normal in &mut self.normals {
            *normal = normal.normalize_or_zero();
        }
    }

    /// Transforms all positions by a matrix.
    pub fn transform(&mut self, matrix: glam::Mat4) {
        let normal_matrix = matrix.inverse().transpose();

        for pos in &mut self.positions {
            *pos = matrix.transform_point3(*pos);
        }

        for normal in &mut self.normals {
            *normal = normal_matrix.transform_vector3(*normal).normalize_or_zero();
        }
    }

    /// Merges another mesh into this one.
    pub fn merge(&mut self, other: &Mesh) {
        let base_index = self.positions.len() as u32;

        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);

        self.indices
            .extend(other.indices.iter().map(|i| i + base_index));
    }
}

/// Builder for constructing meshes vertex by vertex.
#[derive(Debug, Clone, Default)]
pub struct MeshBuilder {
    mesh: Mesh,
}

impl MeshBuilder {
    /// Creates a new mesh builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a vertex with position only.
    pub fn vertex(&mut self, position: Vec3) -> u32 {
        let index = self.mesh.positions.len() as u32;
        self.mesh.positions.push(position);
        index
    }

    /// Adds a vertex with position and normal.
    pub fn vertex_with_normal(&mut self, position: Vec3, normal: Vec3) -> u32 {
        let index = self.mesh.positions.len() as u32;
        self.mesh.positions.push(position);
        self.mesh.normals.push(normal);
        index
    }

    /// Adds a vertex with position, normal, and UV.
    pub fn vertex_with_normal_uv(&mut self, position: Vec3, normal: Vec3, uv: Vec2) -> u32 {
        let index = self.mesh.positions.len() as u32;
        self.mesh.positions.push(position);
        self.mesh.normals.push(normal);
        self.mesh.uvs.push(uv);
        index
    }

    /// Adds a triangle from three vertex indices.
    pub fn triangle(&mut self, i0: u32, i1: u32, i2: u32) {
        self.mesh.indices.push(i0);
        self.mesh.indices.push(i1);
        self.mesh.indices.push(i2);
    }

    /// Adds a quad from four vertex indices (converted to two triangles).
    pub fn quad(&mut self, i0: u32, i1: u32, i2: u32, i3: u32) {
        // First triangle
        self.mesh.indices.push(i0);
        self.mesh.indices.push(i1);
        self.mesh.indices.push(i2);
        // Second triangle
        self.mesh.indices.push(i0);
        self.mesh.indices.push(i2);
        self.mesh.indices.push(i3);
    }

    /// Builds the final mesh.
    pub fn build(self) -> Mesh {
        self.mesh
    }
}

// Trait implementations for Mesh

impl HasPositions for Mesh {
    fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    fn positions_mut(&mut self) -> &mut [Vec3] {
        &mut self.positions
    }
}

impl HasNormals for Mesh {
    fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    fn normals_mut(&mut self) -> &mut [Vec3] {
        &mut self.normals
    }
}

impl HasUVs for Mesh {
    fn uvs(&self) -> &[Vec2] {
        &self.uvs
    }

    fn uvs_mut(&mut self) -> &mut [Vec2] {
        &mut self.uvs
    }
}

impl HasIndices for Mesh {
    fn indices(&self) -> &[u32] {
        &self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_builder() {
        let mut builder = MeshBuilder::new();

        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(0.0, 1.0, 0.0));

        builder.triangle(v0, v1, v2);

        let mesh = builder.build();

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_smooth_normals() {
        let mut builder = MeshBuilder::new();

        // Two triangles sharing an edge
        let v0 = builder.vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = builder.vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = builder.vertex(Vec3::new(0.5, 1.0, 0.0));
        let v3 = builder.vertex(Vec3::new(0.5, 0.0, 1.0));

        builder.triangle(v0, v1, v2);
        builder.triangle(v0, v3, v1);

        let mut mesh = builder.build();
        mesh.compute_smooth_normals();

        assert!(mesh.has_normals());
        // Shared vertices should have averaged normals
        assert!(mesh.normals[v0 as usize].length() > 0.99);
    }
}
