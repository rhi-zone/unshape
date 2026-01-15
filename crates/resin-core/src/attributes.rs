//! Attribute traits for geometry types.
//!
//! These traits provide generic access to common geometry attributes,
//! enabling generic algorithms over different geometry types.

use glam::{Vec2, Vec3, Vec4};

/// Geometry with vertex positions.
///
/// This is the fundamental trait for all geometry types.
/// Used as a bound for generic algorithms like rigging, deformation, etc.
pub trait HasPositions {
    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize;

    /// Returns all vertex positions.
    fn positions(&self) -> &[Vec3];

    /// Returns mutable access to vertex positions.
    fn positions_mut(&mut self) -> &mut [Vec3];

    /// Gets a single vertex position.
    fn get_position(&self, index: usize) -> Option<Vec3> {
        self.positions().get(index).copied()
    }

    /// Sets a single vertex position.
    fn set_position(&mut self, index: usize, position: Vec3) -> bool {
        if let Some(p) = self.positions_mut().get_mut(index) {
            *p = position;
            true
        } else {
            false
        }
    }
}

/// 2D geometry with vertex positions.
///
/// This is the 2D equivalent of [`HasPositions`] for 2D geometry types.
/// Used for vector graphics, gradient meshes, and other 2D shapes.
pub trait HasPositions2D {
    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize;

    /// Returns all vertex positions.
    fn positions(&self) -> &[Vec2];

    /// Returns mutable access to vertex positions.
    fn positions_mut(&mut self) -> &mut [Vec2];

    /// Gets a single vertex position.
    fn get_position(&self, index: usize) -> Option<Vec2> {
        self.positions().get(index).copied()
    }

    /// Sets a single vertex position.
    fn set_position(&mut self, index: usize, position: Vec2) -> bool {
        if let Some(p) = self.positions_mut().get_mut(index) {
            *p = position;
            true
        } else {
            false
        }
    }
}

/// Geometry with vertex normals.
pub trait HasNormals {
    /// Returns all vertex normals.
    fn normals(&self) -> &[Vec3];

    /// Returns mutable access to vertex normals.
    fn normals_mut(&mut self) -> &mut [Vec3];
}

/// Geometry with texture coordinates.
pub trait HasUVs {
    /// Returns all UV coordinates.
    fn uvs(&self) -> &[Vec2];

    /// Returns mutable access to UV coordinates.
    fn uvs_mut(&mut self) -> &mut [Vec2];
}

/// Geometry with vertex colors.
pub trait HasColors {
    /// Returns all vertex colors (RGBA).
    fn colors(&self) -> &[Vec4];

    /// Returns mutable access to vertex colors.
    fn colors_mut(&mut self) -> &mut [Vec4];
}

/// Geometry with indexed triangles.
pub trait HasIndices {
    /// Returns triangle indices (every 3 form a triangle).
    fn indices(&self) -> &[u32];

    /// Returns the number of triangles.
    fn triangle_count(&self) -> usize {
        self.indices().len() / 3
    }
}

/// Bounds for common geometry operations.
pub trait Geometry: HasPositions + HasIndices {}
impl<T: HasPositions + HasIndices> Geometry for T {}

/// Bounds for geometry with full vertex attributes.
pub trait FullGeometry: HasPositions + HasNormals + HasUVs + HasIndices {}
impl<T: HasPositions + HasNormals + HasUVs + HasIndices> FullGeometry for T {}
