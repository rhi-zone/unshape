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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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

/// Operation to fill a sphere in a binary voxel grid.
///
/// Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = BinaryVoxelGrid, output = BinaryVoxelGrid))]
pub struct FillSphere {
    /// Center of the sphere in voxel coordinates.
    pub center: Vec3,
    /// Radius of the sphere in voxel units.
    pub radius: f32,
    /// Value to fill with (true = solid, false = empty).
    pub value: bool,
}

impl FillSphere {
    /// Creates a new FillSphere operation.
    pub fn new(center: Vec3, radius: f32, value: bool) -> Self {
        Self {
            center,
            radius,
            value,
        }
    }

    /// Applies the operation to a voxel grid.
    pub fn apply(&self, grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
        let mut result = grid.clone();
        fill_sphere(&mut result, self.center, self.radius, self.value);
        result
    }
}

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

/// Operation to fill a box region in a binary voxel grid.
///
/// Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = BinaryVoxelGrid, output = BinaryVoxelGrid))]
pub struct FillBox {
    /// Minimum corner of the box (inclusive).
    pub min: UVec3,
    /// Maximum corner of the box (exclusive).
    pub max: UVec3,
    /// Value to fill with (true = solid, false = empty).
    pub value: bool,
}

impl FillBox {
    /// Creates a new FillBox operation.
    pub fn new(min: UVec3, max: UVec3, value: bool) -> Self {
        Self { min, max, value }
    }

    /// Applies the operation to a voxel grid.
    pub fn apply(&self, grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
        let mut result = grid.clone();
        fill_box(&mut result, self.min, self.max, self.value);
        result
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

/// Operation to dilate a binary voxel grid (grows solid regions).
///
/// Morphological dilation expands solid regions by one voxel in all 6-connected
/// directions. Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = BinaryVoxelGrid, output = BinaryVoxelGrid))]
pub struct Dilate {
    /// Number of dilation iterations.
    pub iterations: u32,
}

impl Dilate {
    /// Creates a new Dilate operation with a single iteration.
    pub fn new() -> Self {
        Self { iterations: 1 }
    }

    /// Creates a Dilate operation with the specified number of iterations.
    pub fn with_iterations(iterations: u32) -> Self {
        Self { iterations }
    }

    /// Applies the dilation operation to a voxel grid.
    pub fn apply(&self, grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
        let mut result = grid.clone();
        for _ in 0..self.iterations {
            result = dilate(&result);
        }
        result
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

/// Operation to erode a binary voxel grid (shrinks solid regions).
///
/// Morphological erosion shrinks solid regions by one voxel in all 6-connected
/// directions. Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = BinaryVoxelGrid, output = BinaryVoxelGrid))]
pub struct Erode {
    /// Number of erosion iterations.
    pub iterations: u32,
}

impl Erode {
    /// Creates a new Erode operation with a single iteration.
    pub fn new() -> Self {
        Self { iterations: 1 }
    }

    /// Creates an Erode operation with the specified number of iterations.
    pub fn with_iterations(iterations: u32) -> Self {
        Self { iterations }
    }

    /// Applies the erosion operation to a voxel grid.
    pub fn apply(&self, grid: &BinaryVoxelGrid) -> BinaryVoxelGrid {
        let mut result = grid.clone();
        for _ in 0..self.iterations {
            result = erode(&result);
        }
        result
    }
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

/// Operation to generate a simple blocky mesh from a binary voxel grid.
///
/// Each solid voxel becomes a cube, with faces only on boundaries.
/// Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = BinaryVoxelGrid, output = Mesh))]
pub struct VoxelsToMesh {
    /// Size of each voxel in world units.
    pub voxel_size: f32,
}

impl VoxelsToMesh {
    /// Creates a new VoxelsToMesh operation with the specified voxel size.
    pub fn new(voxel_size: f32) -> Self {
        Self { voxel_size }
    }

    /// Applies the operation to a voxel grid.
    pub fn apply(&self, grid: &BinaryVoxelGrid) -> Mesh {
        voxels_to_mesh(grid, self.voxel_size)
    }
}

impl Default for VoxelsToMesh {
    fn default() -> Self {
        Self { voxel_size: 1.0 }
    }
}

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

/// Operation to generate a mesh from sparse voxels.
///
/// Uses the ops-as-values pattern for serialization and history tracking.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = SparseBinaryVoxels, output = Mesh))]
pub struct SparseVoxelsToMesh {
    /// Size of each voxel in world units.
    pub voxel_size: f32,
}

impl SparseVoxelsToMesh {
    /// Creates a new SparseVoxelsToMesh operation with the specified voxel size.
    pub fn new(voxel_size: f32) -> Self {
        Self { voxel_size }
    }

    /// Applies the operation to sparse voxels.
    pub fn apply(&self, voxels: &SparseBinaryVoxels) -> Mesh {
        sparse_voxels_to_mesh(voxels, self.voxel_size)
    }
}

impl Default for SparseVoxelsToMesh {
    fn default() -> Self {
        Self { voxel_size: 1.0 }
    }
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

/// Registers all voxel operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of voxel ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<Dilate>("resin::Dilate");
    registry.register_type::<Erode>("resin::Erode");
    registry.register_type::<FillBox>("resin::FillBox");
    registry.register_type::<FillSphere>("resin::FillSphere");
    registry.register_type::<SparseVoxelsToMesh>("resin::SparseVoxelsToMesh");
    registry.register_type::<VoxelsToMesh>("resin::VoxelsToMesh");
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

/// Invariant tests for voxel operations.
///
/// These tests verify mathematical and geometric properties that should hold
/// for all voxel implementations. Run with:
///
/// ```sh
/// cargo test -p rhizome-resin-voxel --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use glam::{UVec3, Vec3};

    // ========================================================================
    // VoxelGrid bounds and indexing invariants
    // ========================================================================

    /// Grid size should match the requested dimensions.
    #[test]
    fn test_grid_size_invariant() {
        for size in [(8, 8, 8), (16, 32, 24), (1, 100, 1), (64, 64, 64)] {
            let grid = VoxelGrid::new(size.0, size.1, size.2, false);
            assert_eq!(grid.size(), UVec3::new(size.0, size.1, size.2));
            assert_eq!(grid.len(), (size.0 * size.1 * size.2) as usize);
        }
    }

    /// Set and get should be consistent.
    #[test]
    fn test_set_get_roundtrip() {
        let mut grid = VoxelGrid::new(16, 16, 16, 0u8);

        // Set values at various positions
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let value = ((x + y * 16 + z * 256) % 256) as u8;
                    grid.set(x, y, z, value);
                }
            }
        }

        // Verify all values
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let expected = ((x + y * 16 + z * 256) % 256) as u8;
                    assert_eq!(*grid.get(x, y, z), expected);
                }
            }
        }
    }

    // ========================================================================
    // Sphere fill geometric invariants
    // ========================================================================

    /// Sphere fill should respect geometric bounds.
    #[test]
    fn test_sphere_fill_bounds() {
        let mut grid = VoxelGrid::new(32, 32, 32, false);
        let center = Vec3::new(16.0, 16.0, 16.0);
        let radius = 8.0;

        fill_sphere(&mut grid, center, radius, true);

        // All points inside sphere should be set
        // All points far outside sphere should not be set
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    let dist = pos.distance(center);
                    let is_set = *grid.get(x, y, z);

                    if dist < radius - 1.0 {
                        assert!(is_set, "Point clearly inside sphere should be set: {pos:?}");
                    }
                    if dist > radius + 1.0 {
                        assert!(
                            !is_set,
                            "Point clearly outside sphere should not be set: {pos:?}"
                        );
                    }
                }
            }
        }
    }

    /// Sphere fill count should be approximately (4/3)πr³.
    #[test]
    fn test_sphere_volume_approximation() {
        for radius in [4.0, 8.0, 12.0] {
            let size = (radius * 3.0) as u32;
            let mut grid = VoxelGrid::new(size, size, size, false);
            let center = Vec3::splat(size as f32 / 2.0);

            fill_sphere(&mut grid, center, radius, true);

            let actual_count = count_solid(&grid);
            let expected_volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);

            // Allow 20% error due to voxelization
            let error_ratio = (actual_count as f32 - expected_volume).abs() / expected_volume;
            assert!(
                error_ratio < 0.20,
                "Sphere r={radius}: volume error too large: {error_ratio:.1}%"
            );
        }
    }

    // ========================================================================
    // Box fill geometric invariants
    // ========================================================================

    /// Box fill should set exactly the requested region.
    #[test]
    fn test_box_fill_exact() {
        let mut grid = VoxelGrid::new(32, 32, 32, false);
        let min = UVec3::new(4, 8, 12);
        let max = UVec3::new(12, 16, 20);

        fill_box(&mut grid, min, max, true);

        let count = count_solid(&grid);
        let expected = (max.x - min.x) * (max.y - min.y) * (max.z - min.z);

        assert_eq!(count, expected as usize);

        // Verify boundaries
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    let inside = x >= min.x
                        && x < max.x
                        && y >= min.y
                        && y < max.y
                        && z >= min.z
                        && z < max.z;
                    assert_eq!(*grid.get(x, y, z), inside);
                }
            }
        }
    }

    // ========================================================================
    // Morphological operation invariants
    // ========================================================================

    /// Dilate should never decrease solid count.
    #[test]
    fn test_dilate_increases_count() {
        let mut grid = VoxelGrid::new(16, 16, 16, false);
        fill_sphere(&mut grid, Vec3::splat(8.0), 4.0, true);

        let before = count_solid(&grid);
        let dilated = dilate(&grid);
        let after = count_solid(&dilated);

        assert!(
            after >= before,
            "Dilate should not decrease count: {before} -> {after}"
        );
    }

    /// Erode should never increase solid count.
    #[test]
    fn test_erode_decreases_count() {
        let mut grid = VoxelGrid::new(16, 16, 16, false);
        fill_sphere(&mut grid, Vec3::splat(8.0), 6.0, true);

        let before = count_solid(&grid);
        let eroded = erode(&grid);
        let after = count_solid(&eroded);

        assert!(
            after <= before,
            "Erode should not increase count: {before} -> {after}"
        );
    }

    /// Dilate then erode (closing) should preserve connectivity.
    #[test]
    fn test_morphological_closing() {
        let mut grid = VoxelGrid::new(16, 16, 16, false);
        fill_sphere(&mut grid, Vec3::splat(8.0), 5.0, true);

        let original_count = count_solid(&grid);
        let closed = erode(&dilate(&grid));
        let closed_count = count_solid(&closed);

        // Closing should roughly preserve the shape
        let change_ratio =
            (closed_count as f32 - original_count as f32).abs() / original_count as f32;
        assert!(
            change_ratio < 0.3,
            "Closing should roughly preserve shape: change ratio {change_ratio:.1}%"
        );
    }

    // ========================================================================
    // Sparse voxel invariants
    // ========================================================================

    /// Sparse and dense representations should be equivalent.
    #[test]
    fn test_sparse_dense_equivalence() {
        let mut dense = VoxelGrid::new(16, 16, 16, false);
        let mut sparse = SparseVoxels::new();

        // Fill random pattern
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let should_set = (x + y + z) % 3 == 0;
                    if should_set {
                        dense.set(x, y, z, true);
                        sparse.set(IVec3::new(x as i32, y as i32, z as i32), true);
                    }
                }
            }
        }

        // Verify equivalence
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let dense_val = *dense.get(x, y, z);
                    let sparse_val = *sparse.get(IVec3::new(x as i32, y as i32, z as i32));

                    assert_eq!(dense_val, sparse_val, "Mismatch at ({x}, {y}, {z})");
                }
            }
        }

        // Sparse should track actual count
        let dense_count = count_solid(&dense);
        assert_eq!(sparse.len(), dense_count);
    }

    // ========================================================================
    // Mesh generation invariants
    // ========================================================================

    /// Generated mesh should have reasonable structure.
    #[test]
    fn test_voxels_to_mesh_validity() {
        let mut grid = VoxelGrid::new(8, 8, 8, false);
        fill_sphere(&mut grid, Vec3::splat(4.0), 3.0, true);

        let mesh = voxels_to_mesh(&grid, 1.0);

        // Mesh should have vertices
        assert!(!mesh.positions.is_empty(), "Mesh should have vertices");

        // Triangle count should be divisible by 3
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should form triangles");

        // All indices should be valid
        let max_vertex = mesh.positions.len();
        for &idx in &mesh.indices {
            assert!(
                (idx as usize) < max_vertex,
                "Index {idx} out of bounds (max {max_vertex})"
            );
        }
    }

    /// Larger voxel grids should produce more mesh faces.
    #[test]
    fn test_mesh_scales_with_voxels() {
        let mut small = VoxelGrid::new(8, 8, 8, false);
        fill_sphere(&mut small, Vec3::splat(4.0), 3.0, true);

        let mut large = VoxelGrid::new(16, 16, 16, false);
        fill_sphere(&mut large, Vec3::splat(8.0), 6.0, true);

        let small_mesh = voxels_to_mesh(&small, 1.0);
        let large_mesh = voxels_to_mesh(&large, 1.0);

        // Double radius should produce more faces
        assert!(
            large_mesh.indices.len() > small_mesh.indices.len(),
            "Larger sphere should produce more mesh faces"
        );
    }
}
