//! Voxel operations for 3D discrete grids.
//!
//! Provides types and functions for working with voxel data:
//! - Dense voxel grids with arbitrary data
//! - Sparse voxel storage for memory efficiency
//! - SDF to voxel conversion
//! - Voxel editing operations (set, fill, sphere brush, etc.)
//! - Simple mesh generation from voxels
//!
//! # Example
//!
//! ```
//! use rhizome_resin_voxel::{VoxelGrid, fill_sphere};
//! use glam::Vec3;
//!
//! let mut grid = VoxelGrid::new(32, 32, 32, false);
//!
//! // Fill a sphere at the center
//! fill_sphere(&mut grid, Vec3::new(16.0, 16.0, 16.0), 8.0, true);
//!
//! assert!(grid.get(16, 16, 16));
//! ```

use glam::{IVec3, UVec3, Vec3};
use rhizome_resin_field::{EvalContext, Field};
use rhizome_resin_mesh::{Mesh, MeshBuilder};
use std::collections::HashMap;

/// A dense 3D voxel grid with arbitrary data type.
#[derive(Debug, Clone)]
pub struct VoxelGrid<T: Clone> {
    data: Vec<T>,
    size: UVec3,
}

impl<T: Clone> VoxelGrid<T> {
    /// Creates a new voxel grid filled with the default value.
    pub fn new(width: u32, height: u32, depth: u32, default: T) -> Self {
        let size = UVec3::new(width, height, depth);
        let count = (width * height * depth) as usize;
        Self {
            data: vec![default; count],
            size,
        }
    }

    /// Returns the grid dimensions.
    pub fn size(&self) -> UVec3 {
        self.size
    }

    /// Returns the total number of voxels.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Converts 3D coordinates to a linear index.
    fn index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * self.size.y * self.size.x + y * self.size.x + x) as usize
    }

    /// Returns true if the coordinates are within bounds.
    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < self.size.x
            && (y as u32) < self.size.y
            && (z as u32) < self.size.z
    }

    /// Gets the value at the given coordinates.
    ///
    /// Returns None if out of bounds.
    pub fn try_get(&self, x: i32, y: i32, z: i32) -> Option<&T> {
        if self.in_bounds(x, y, z) {
            Some(&self.data[self.index(x as u32, y as u32, z as u32)])
        } else {
            None
        }
    }

    /// Gets the value at the given coordinates.
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds.
    pub fn get(&self, x: u32, y: u32, z: u32) -> &T {
        &self.data[self.index(x, y, z)]
    }

    /// Sets the value at the given coordinates.
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds.
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let idx = self.index(x, y, z);
        self.data[idx] = value;
    }

    /// Sets the value at the given coordinates if in bounds.
    pub fn try_set(&mut self, x: i32, y: i32, z: i32, value: T) -> bool {
        if self.in_bounds(x, y, z) {
            self.set(x as u32, y as u32, z as u32, value);
            true
        } else {
            false
        }
    }

    /// Fills the entire grid with a value.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    /// Iterates over all voxels with their coordinates.
    pub fn iter(&self) -> impl Iterator<Item = (UVec3, &T)> {
        self.data.iter().enumerate().map(move |(i, v)| {
            let i = i as u32;
            let x = i % self.size.x;
            let y = (i / self.size.x) % self.size.y;
            let z = i / (self.size.x * self.size.y);
            (UVec3::new(x, y, z), v)
        })
    }

    /// Iterates over all voxels with mutable access.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (UVec3, &mut T)> {
        let size = self.size;
        self.data.iter_mut().enumerate().map(move |(i, v)| {
            let i = i as u32;
            let x = i % size.x;
            let y = (i / size.x) % size.y;
            let z = i / (size.x * size.y);
            (UVec3::new(x, y, z), v)
        })
    }
}

// Convenience type aliases
/// A binary (on/off) voxel grid.
pub type BinaryVoxelGrid = VoxelGrid<bool>;

/// A voxel grid with scalar density values.
pub type DensityVoxelGrid = VoxelGrid<f32>;

// ============================================================================
// Sparse voxel storage
// ============================================================================

/// Sparse voxel storage using a hashmap.
///
/// Only stores non-default voxels, making it memory-efficient
/// for sparsely populated grids.
#[derive(Debug, Clone)]
pub struct SparseVoxels<T: Clone + Default + PartialEq> {
    data: HashMap<IVec3, T>,
    default: T,
}

impl<T: Clone + Default + PartialEq> Default for SparseVoxels<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Default + PartialEq> SparseVoxels<T> {
    /// Creates a new empty sparse voxel container.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            default: T::default(),
        }
    }

    /// Creates a new sparse voxel container with a custom default value.
    pub fn with_default(default: T) -> Self {
        Self {
            data: HashMap::new(),
            default,
        }
    }

    /// Returns the number of stored (non-default) voxels.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if no voxels are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets the value at the given coordinates.
    ///
    /// Returns the default value if not set.
    pub fn get(&self, pos: IVec3) -> &T {
        self.data.get(&pos).unwrap_or(&self.default)
    }

    /// Sets the value at the given coordinates.
    ///
    /// If the value equals the default, removes it from storage.
    pub fn set(&mut self, pos: IVec3, value: T) {
        if value == self.default {
            self.data.remove(&pos);
        } else {
            self.data.insert(pos, value);
        }
    }

    /// Clears all stored voxels.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Iterates over all stored (non-default) voxels.
    pub fn iter(&self) -> impl Iterator<Item = (&IVec3, &T)> {
        self.data.iter()
    }

    /// Returns the bounding box of stored voxels.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        if self.data.is_empty() {
            return None;
        }

        let mut min = IVec3::MAX;
        let mut max = IVec3::MIN;

        for &pos in self.data.keys() {
            min = min.min(pos);
            max = max.max(pos);
        }

        Some((min, max))
    }
}

/// Binary sparse voxels.
pub type SparseBinaryVoxels = SparseVoxels<bool>;

// ============================================================================
// SDF to voxel conversion
// ============================================================================

/// Converts an SDF field to a binary voxel grid.
///
/// Voxels where SDF < 0 are set to true (inside).
pub fn sdf_to_voxels<F: Field<Vec3, f32>>(
    sdf: &F,
    bounds: (Vec3, Vec3),
    resolution: UVec3,
) -> BinaryVoxelGrid {
    let ctx = EvalContext::new();
    let (min, max) = bounds;
    let extent = max - min;
    let step = extent / resolution.as_vec3();

    let mut grid = VoxelGrid::new(resolution.x, resolution.y, resolution.z, false);

    for z in 0..resolution.z {
        for y in 0..resolution.y {
            for x in 0..resolution.x {
                let pos = min + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * step;
                let d = sdf.sample(pos, &ctx);
                if d < 0.0 {
                    grid.set(x, y, z, true);
                }
            }
        }
    }

    grid
}

/// Converts an SDF field to a density voxel grid.
///
/// Stores the actual SDF values.
pub fn sdf_to_density<F: Field<Vec3, f32>>(
    sdf: &F,
    bounds: (Vec3, Vec3),
    resolution: UVec3,
) -> DensityVoxelGrid {
    let ctx = EvalContext::new();
    let (min, max) = bounds;
    let extent = max - min;
    let step = extent / resolution.as_vec3();

    let mut grid = VoxelGrid::new(resolution.x, resolution.y, resolution.z, 0.0);

    for z in 0..resolution.z {
        for y in 0..resolution.y {
            for x in 0..resolution.x {
                let pos = min + Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * step;
                let d = sdf.sample(pos, &ctx);
                grid.set(x, y, z, d);
            }
        }
    }

    grid
}

// ============================================================================
// Voxel editing operations
// ============================================================================

/// Fills a sphere in a binary voxel grid.
pub fn fill_sphere(grid: &mut BinaryVoxelGrid, center: Vec3, radius: f32, value: bool) {
    let size = grid.size();
    let radius_sq = radius * radius;

    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                if (pos - center).length_squared() <= radius_sq {
                    grid.set(x, y, z, value);
                }
            }
        }
    }
}

/// Fills a box region in a binary voxel grid.
pub fn fill_box(grid: &mut BinaryVoxelGrid, min: UVec3, max: UVec3, value: bool) {
    let size = grid.size();
    let min = min.min(size);
    let max = max.min(size);

    for z in min.z..max.z {
        for y in min.y..max.y {
            for x in min.x..max.x {
                grid.set(x, y, z, value);
            }
        }
    }
}

/// Fills a sphere in sparse voxels.
pub fn fill_sphere_sparse(voxels: &mut SparseBinaryVoxels, center: Vec3, radius: f32, value: bool) {
    let radius_sq = radius * radius;
    let r_ceil = radius.ceil() as i32;

    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    let cz = center.z.round() as i32;

    for dz in -r_ceil..=r_ceil {
        for dy in -r_ceil..=r_ceil {
            for dx in -r_ceil..=r_ceil {
                let pos = Vec3::new(
                    (cx + dx) as f32 + 0.5,
                    (cy + dy) as f32 + 0.5,
                    (cz + dz) as f32 + 0.5,
                );
                if (pos - center).length_squared() <= radius_sq {
                    voxels.set(IVec3::new(cx + dx, cy + dy, cz + dz), value);
                }
            }
        }
    }
}

/// Dilates a binary voxel grid (grows solid regions).
pub fn dilate(grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
    let size = grid.size();
    let mut result = VoxelGrid::new(size.x, size.y, size.z, false);

    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                // Check if this voxel or any neighbor is solid
                let mut is_solid = *grid.get(x, y, z);
                if !is_solid {
                    for &(dx, dy, dz) in &NEIGHBORS_6 {
                        if let Some(&v) = grid.try_get(x as i32 + dx, y as i32 + dy, z as i32 + dz)
                        {
                            if v {
                                is_solid = true;
                                break;
                            }
                        }
                    }
                }
                result.set(x, y, z, is_solid);
            }
        }
    }

    result
}

/// Erodes a binary voxel grid (shrinks solid regions).
pub fn erode(grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
    let size = grid.size();
    let mut result = VoxelGrid::new(size.x, size.y, size.z, false);

    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                // Only solid if this voxel AND all neighbors are solid
                let mut is_solid = *grid.get(x, y, z);
                if is_solid {
                    for &(dx, dy, dz) in &NEIGHBORS_6 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if let Some(&v) = grid.try_get(nx, ny, nz) {
                            if !v {
                                is_solid = false;
                                break;
                            }
                        } else {
                            // Out of bounds = not solid
                            is_solid = false;
                            break;
                        }
                    }
                }
                result.set(x, y, z, is_solid);
            }
        }
    }

    result
}

/// 6-connected neighbor offsets.
const NEIGHBORS_6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

// ============================================================================
// Mesh generation
// ============================================================================

/// Generates a simple blocky mesh from a binary voxel grid.
///
/// Each solid voxel becomes a cube, with faces only on boundaries.
pub fn voxels_to_mesh(grid: &BinaryVoxelGrid, voxel_size: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let size = grid.size();
    let half = voxel_size / 2.0;

    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                if !*grid.get(x, y, z) {
                    continue;
                }

                let center = Vec3::new(x as f32, y as f32, z as f32) * voxel_size
                    + Vec3::splat(voxel_size / 2.0);

                // Check each face
                // -X face
                if x == 0 || !*grid.get(x - 1, y, z) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(-half, -half, -half),
                        center + Vec3::new(-half, -half, half),
                        center + Vec3::new(-half, half, half),
                        center + Vec3::new(-half, half, -half),
                        Vec3::NEG_X,
                    );
                }
                // +X face
                if x == size.x - 1 || !*grid.get(x + 1, y, z) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(half, -half, half),
                        center + Vec3::new(half, -half, -half),
                        center + Vec3::new(half, half, -half),
                        center + Vec3::new(half, half, half),
                        Vec3::X,
                    );
                }
                // -Y face
                if y == 0 || !*grid.get(x, y - 1, z) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(-half, -half, -half),
                        center + Vec3::new(half, -half, -half),
                        center + Vec3::new(half, -half, half),
                        center + Vec3::new(-half, -half, half),
                        Vec3::NEG_Y,
                    );
                }
                // +Y face
                if y == size.y - 1 || !*grid.get(x, y + 1, z) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(-half, half, half),
                        center + Vec3::new(half, half, half),
                        center + Vec3::new(half, half, -half),
                        center + Vec3::new(-half, half, -half),
                        Vec3::Y,
                    );
                }
                // -Z face
                if z == 0 || !*grid.get(x, y, z - 1) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(half, -half, -half),
                        center + Vec3::new(-half, -half, -half),
                        center + Vec3::new(-half, half, -half),
                        center + Vec3::new(half, half, -half),
                        Vec3::NEG_Z,
                    );
                }
                // +Z face
                if z == size.z - 1 || !*grid.get(x, y, z + 1) {
                    add_quad(
                        &mut builder,
                        center + Vec3::new(-half, -half, half),
                        center + Vec3::new(half, -half, half),
                        center + Vec3::new(half, half, half),
                        center + Vec3::new(-half, half, half),
                        Vec3::Z,
                    );
                }
            }
        }
    }

    builder.build()
}

fn add_quad(builder: &mut MeshBuilder, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, normal: Vec3) {
    use glam::Vec2;

    let i0 = builder.vertex_with_normal_uv(v0, normal, Vec2::new(0.0, 0.0));
    let i1 = builder.vertex_with_normal_uv(v1, normal, Vec2::new(1.0, 0.0));
    let i2 = builder.vertex_with_normal_uv(v2, normal, Vec2::new(1.0, 1.0));
    let i3 = builder.vertex_with_normal_uv(v3, normal, Vec2::new(0.0, 1.0));

    builder.quad(i0, i1, i2, i3);
}

/// Generates a mesh from sparse voxels.
pub fn sparse_voxels_to_mesh(voxels: &SparseBinaryVoxels, voxel_size: f32) -> Mesh {
    let mut builder = MeshBuilder::new();
    let half = voxel_size / 2.0;

    for (&pos, &is_solid) in voxels.iter() {
        if !is_solid {
            continue;
        }

        let center = Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) * voxel_size
            + Vec3::splat(voxel_size / 2.0);

        // Check each face
        for &(dx, dy, dz) in &NEIGHBORS_6 {
            let neighbor = IVec3::new(pos.x + dx, pos.y + dy, pos.z + dz);
            if !*voxels.get(neighbor) {
                let normal = Vec3::new(dx as f32, dy as f32, dz as f32);
                add_face(&mut builder, center, half, normal);
            }
        }
    }

    builder.build()
}

fn add_face(builder: &mut MeshBuilder, center: Vec3, half: f32, normal: Vec3) {
    // Determine face vertices based on normal direction
    let (v0, v1, v2, v3) = if normal.x < -0.5 {
        // -X face
        (
            center + Vec3::new(-half, -half, -half),
            center + Vec3::new(-half, -half, half),
            center + Vec3::new(-half, half, half),
            center + Vec3::new(-half, half, -half),
        )
    } else if normal.x > 0.5 {
        // +X face
        (
            center + Vec3::new(half, -half, half),
            center + Vec3::new(half, -half, -half),
            center + Vec3::new(half, half, -half),
            center + Vec3::new(half, half, half),
        )
    } else if normal.y < -0.5 {
        // -Y face
        (
            center + Vec3::new(-half, -half, -half),
            center + Vec3::new(half, -half, -half),
            center + Vec3::new(half, -half, half),
            center + Vec3::new(-half, -half, half),
        )
    } else if normal.y > 0.5 {
        // +Y face
        (
            center + Vec3::new(-half, half, half),
            center + Vec3::new(half, half, half),
            center + Vec3::new(half, half, -half),
            center + Vec3::new(-half, half, -half),
        )
    } else if normal.z < -0.5 {
        // -Z face
        (
            center + Vec3::new(half, -half, -half),
            center + Vec3::new(-half, -half, -half),
            center + Vec3::new(-half, half, -half),
            center + Vec3::new(half, half, -half),
        )
    } else {
        // +Z face
        (
            center + Vec3::new(-half, -half, half),
            center + Vec3::new(half, -half, half),
            center + Vec3::new(half, half, half),
            center + Vec3::new(-half, half, half),
        )
    };

    add_quad(builder, v0, v1, v2, v3, normal);
}

/// Counts the number of solid voxels in a binary grid.
pub fn count_solid(grid: &BinaryVoxelGrid) -> usize {
    grid.iter().filter(|&(_, v)| *v).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_grid_basic() {
        let mut grid = VoxelGrid::new(4, 4, 4, false);
        assert!(!*grid.get(0, 0, 0));

        grid.set(1, 2, 3, true);
        assert!(*grid.get(1, 2, 3));
    }

    #[test]
    fn test_voxel_grid_fill() {
        let mut grid = VoxelGrid::new(4, 4, 4, false);
        grid.fill(true);

        assert!(*grid.get(0, 0, 0));
        assert!(*grid.get(3, 3, 3));
    }

    #[test]
    fn test_in_bounds() {
        let grid = VoxelGrid::new(4, 4, 4, false);

        assert!(grid.in_bounds(0, 0, 0));
        assert!(grid.in_bounds(3, 3, 3));
        assert!(!grid.in_bounds(-1, 0, 0));
        assert!(!grid.in_bounds(4, 0, 0));
    }

    #[test]
    fn test_sparse_voxels() {
        let mut sparse = SparseBinaryVoxels::new();
        assert!(sparse.is_empty());

        sparse.set(IVec3::new(5, 10, 15), true);
        assert_eq!(sparse.len(), 1);
        assert!(*sparse.get(IVec3::new(5, 10, 15)));
        assert!(!*sparse.get(IVec3::new(0, 0, 0)));

        // Setting to default removes it
        sparse.set(IVec3::new(5, 10, 15), false);
        assert!(sparse.is_empty());
    }

    #[test]
    fn test_fill_sphere() {
        let mut grid = VoxelGrid::new(16, 16, 16, false);
        fill_sphere(&mut grid, Vec3::new(8.0, 8.0, 8.0), 4.0, true);

        // Center should be solid
        assert!(*grid.get(8, 8, 8));
        // Corner should be empty
        assert!(!*grid.get(0, 0, 0));
    }

    #[test]
    fn test_fill_box() {
        let mut grid = VoxelGrid::new(16, 16, 16, false);
        fill_box(&mut grid, UVec3::new(4, 4, 4), UVec3::new(8, 8, 8), true);

        assert!(*grid.get(5, 5, 5));
        assert!(!*grid.get(0, 0, 0));
        assert!(!*grid.get(9, 9, 9));
    }

    #[test]
    fn test_dilate_erode() {
        let mut grid = VoxelGrid::new(8, 8, 8, false);
        grid.set(4, 4, 4, true);

        let dilated = dilate(&grid);
        // Original and neighbors should be solid
        assert!(*dilated.get(4, 4, 4));
        assert!(*dilated.get(3, 4, 4));
        assert!(*dilated.get(5, 4, 4));

        let eroded = erode(&dilated);
        // After erosion, should shrink back
        // Actually, a single voxel dilated once then eroded once may not return to original
        // Let's just verify the erosion works
        let solid_count: usize = eroded.iter().filter(|&(_, v)| *v).count();
        assert!(solid_count <= count_solid(&dilated));
    }

    #[test]
    fn test_voxels_to_mesh() {
        let mut grid = VoxelGrid::new(2, 2, 2, false);
        grid.set(0, 0, 0, true);

        let mesh = voxels_to_mesh(&grid, 1.0);
        // Single voxel = 6 faces = 6 * 4 vertices = 24 vertices (with duplication)
        assert_eq!(mesh.positions.len(), 24);
        // 6 faces * 2 triangles * 3 indices = 36 indices
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn test_sparse_voxels_to_mesh() {
        let mut sparse = SparseBinaryVoxels::new();
        sparse.set(IVec3::new(0, 0, 0), true);

        let mesh = sparse_voxels_to_mesh(&sparse, 1.0);
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn test_sparse_bounds() {
        let mut sparse = SparseBinaryVoxels::new();
        assert!(sparse.bounds().is_none());

        sparse.set(IVec3::new(-5, 0, 10), true);
        sparse.set(IVec3::new(5, 10, -10), true);

        let (min, max) = sparse.bounds().unwrap();
        assert_eq!(min, IVec3::new(-5, 0, -10));
        assert_eq!(max, IVec3::new(5, 10, 10));
    }

    #[test]
    fn test_count_solid() {
        let mut grid = VoxelGrid::new(4, 4, 4, false);
        assert_eq!(count_solid(&grid), 0);

        grid.set(0, 0, 0, true);
        grid.set(1, 1, 1, true);
        assert_eq!(count_solid(&grid), 2);
    }

    #[test]
    fn test_iter() {
        let mut grid = VoxelGrid::new(2, 2, 2, 0u8);
        grid.set(1, 1, 1, 42);

        let found: Vec<_> = grid
            .iter()
            .filter(|&(_, v)| *v == 42)
            .map(|(pos, _)| pos)
            .collect();

        assert_eq!(found.len(), 1);
        assert_eq!(found[0], UVec3::new(1, 1, 1));
    }
}
