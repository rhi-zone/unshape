//! Point cloud operations for 3D point data.
//!
//! Provides types and functions for working with point clouds:
//! - Sampling from meshes and SDFs
//! - Normal estimation
//! - Filtering and downsampling
//! - K-nearest neighbor queries
//!
//! # Example
//!
//! ```
//! use unshape_pointcloud::{PointCloud, sample_mesh_uniform};
//! use unshape_mesh::Cuboid;
//!
//! let mesh = Cuboid::unit().apply();
//! let cloud = sample_mesh_uniform(&mesh, 1000);
//!
//! assert!(cloud.len() >= 100); // Some points generated
//! ```

use glam::{UVec3, Vec3, Vec4};
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_field::{EvalContext, Field};
use unshape_geometry::{HasColors, HasNormals, HasPositions};
use unshape_mesh::Mesh;

/// Registers all pointcloud operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of pointcloud ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Poisson>("resin::Poisson");
    registry.register_type::<RemoveOutliers>("resin::RemoveOutliers");
    registry.register_type::<VoxelDownsample>("resin::VoxelDownsample");
    registry.register_type::<CropBounds>("resin::CropBounds");
    registry.register_type::<EstimateNormals>("resin::EstimateNormals");
    registry.register_type::<UniformSampling>("resin::UniformSampling");
    registry.register_type::<SphereSurface>("resin::SphereSurface");
    registry.register_type::<SphereVolume>("resin::SphereVolume");
    registry.register_type::<GridPoints>("resin::GridPoints");
    registry.register_type::<BoxVolume>("resin::BoxVolume");
}

/// A point cloud with positions, optional normals, and optional colors.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// Point positions.
    pub positions: Vec<Vec3>,
    /// Point normals (same length as positions, or empty).
    pub normals: Vec<Vec3>,
    /// Point colors as RGBA in [0, 1] (same length as positions, or empty).
    pub colors: Vec<Vec4>,
}

impl PointCloud {
    /// Creates an empty point cloud.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a point cloud from positions only.
    pub fn from_positions(positions: Vec<Vec3>) -> Self {
        Self {
            positions,
            normals: Vec::new(),
            colors: Vec::new(),
        }
    }

    /// Creates a point cloud from positions and normals.
    pub fn from_positions_normals(positions: Vec<Vec3>, normals: Vec<Vec3>) -> Self {
        assert_eq!(
            positions.len(),
            normals.len(),
            "Positions and normals must have same length"
        );
        Self {
            positions,
            normals,
            colors: Vec::new(),
        }
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if the point cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns true if the point cloud has normals.
    pub fn has_normals(&self) -> bool {
        !self.normals.is_empty()
    }

    /// Returns true if the point cloud has colors.
    pub fn has_colors(&self) -> bool {
        !self.colors.is_empty()
    }

    /// Adds a point to the cloud.
    pub fn add_point(&mut self, position: Vec3) {
        self.positions.push(position);
        if self.has_normals() {
            self.normals.push(Vec3::ZERO);
        }
        if self.has_colors() {
            self.colors.push(Vec4::ONE);
        }
    }

    /// Adds a point with normal to the cloud.
    pub fn add_point_with_normal(&mut self, position: Vec3, normal: Vec3) {
        self.positions.push(position);
        if self.normals.is_empty() && !self.positions.is_empty() {
            // Initialize normals array if needed
            self.normals = vec![Vec3::ZERO; self.positions.len() - 1];
        }
        self.normals.push(normal.normalize_or_zero());
        if self.has_colors() {
            self.colors.push(Vec4::ONE);
        }
    }

    /// Computes the axis-aligned bounding box.
    pub fn bounding_box(&self) -> Option<(Vec3, Vec3)> {
        if self.positions.is_empty() {
            return None;
        }

        let mut min = self.positions[0];
        let mut max = self.positions[0];

        for &p in &self.positions[1..] {
            min = min.min(p);
            max = max.max(p);
        }

        Some((min, max))
    }

    /// Computes the centroid of the point cloud.
    pub fn centroid(&self) -> Option<Vec3> {
        if self.positions.is_empty() {
            return None;
        }

        let sum: Vec3 = self.positions.iter().copied().sum();
        Some(sum / self.positions.len() as f32)
    }

    /// Merges another point cloud into this one.
    pub fn merge(&mut self, other: &PointCloud) {
        self.positions.extend_from_slice(&other.positions);

        // Handle normals
        if self.has_normals() && other.has_normals() {
            self.normals.extend_from_slice(&other.normals);
        } else if self.has_normals() {
            self.normals
                .extend(std::iter::repeat_n(Vec3::ZERO, other.len()));
        }

        // Handle colors
        if self.has_colors() && other.has_colors() {
            self.colors.extend_from_slice(&other.colors);
        } else if self.has_colors() {
            self.colors
                .extend(std::iter::repeat_n(Vec4::ONE, other.len()));
        }
    }
}

// ============================================================================
// Attribute trait implementations
// ============================================================================

impl HasPositions for PointCloud {
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

impl HasNormals for PointCloud {
    fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    fn normals_mut(&mut self) -> &mut [Vec3] {
        &mut self.normals
    }
}

impl HasColors for PointCloud {
    fn colors(&self) -> &[Vec4] {
        &self.colors
    }

    fn colors_mut(&mut self) -> &mut [Vec4] {
        &mut self.colors
    }
}

// ============================================================================
// Sampling
// ============================================================================

/// Samples points uniformly from a mesh surface.
///
/// Uses random barycentric coordinates on each triangle,
/// with probability proportional to triangle area.
pub fn sample_mesh_uniform(mesh: &Mesh, count: usize) -> PointCloud {
    let mut rng = rand::rng();
    sample_mesh_uniform_with_rng(mesh, count, &mut rng)
}

/// Samples points uniformly from a mesh surface with a custom RNG.
pub fn sample_mesh_uniform_with_rng<R: Rng>(mesh: &Mesh, count: usize, rng: &mut R) -> PointCloud {
    if mesh.indices.is_empty() || mesh.positions.is_empty() {
        return PointCloud::new();
    }

    // Compute triangle areas and build CDF
    let triangles: Vec<[usize; 3]> = mesh
        .indices
        .chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let areas: Vec<f32> = triangles
        .iter()
        .map(|&[i0, i1, i2]| {
            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];
            (v1 - v0).cross(v2 - v0).length() * 0.5
        })
        .collect();

    let total_area: f32 = areas.iter().sum();
    if total_area <= 0.0 {
        return PointCloud::new();
    }

    // Build cumulative distribution
    let mut cdf = Vec::with_capacity(triangles.len());
    let mut cumulative = 0.0;
    for area in &areas {
        cumulative += area / total_area;
        cdf.push(cumulative);
    }

    // Sample points
    let mut positions = Vec::with_capacity(count);
    let mut normals = Vec::with_capacity(count);

    let has_normals = mesh.normals.len() == mesh.positions.len();

    for _ in 0..count {
        // Select triangle by area
        let r: f32 = rng.random();
        let tri_idx = cdf.partition_point(|&c| c < r).min(triangles.len() - 1);
        let [i0, i1, i2] = triangles[tri_idx];

        // Random barycentric coordinates
        let u: f32 = rng.random();
        let v: f32 = rng.random();
        let (u, v) = if u + v > 1.0 {
            (1.0 - u, 1.0 - v)
        } else {
            (u, v)
        };
        let w = 1.0 - u - v;

        // Interpolate position
        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];
        let pos = p0 * w + p1 * u + p2 * v;
        positions.push(pos);

        // Interpolate or compute normal
        if has_normals {
            let n0 = mesh.normals[i0];
            let n1 = mesh.normals[i1];
            let n2 = mesh.normals[i2];
            let normal = (n0 * w + n1 * u + n2 * v).normalize_or_zero();
            normals.push(normal);
        } else {
            // Compute face normal
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            normals.push(normal);
        }
    }

    PointCloud {
        positions,
        normals,
        colors: Vec::new(),
    }
}

/// Samples points from an SDF field using rejection sampling.
///
/// Samples points where the SDF value is within `threshold` of zero.
/// The `bounds` parameter specifies the sampling region (min, max).
pub fn sample_sdf<F: Field<Vec3, f32>>(
    sdf: &F,
    bounds: (Vec3, Vec3),
    count: usize,
    threshold: f32,
) -> PointCloud {
    let mut rng = rand::rng();
    sample_sdf_with_rng(sdf, bounds, count, threshold, &mut rng)
}

/// Samples points from an SDF field with a custom RNG.
pub fn sample_sdf_with_rng<F: Field<Vec3, f32>, R: Rng>(
    sdf: &F,
    bounds: (Vec3, Vec3),
    count: usize,
    threshold: f32,
    rng: &mut R,
) -> PointCloud {
    let ctx = EvalContext::new();
    let (min, max) = bounds;
    let extent = max - min;

    let mut positions = Vec::with_capacity(count);
    let max_attempts = count * 100; // Avoid infinite loops

    for _ in 0..max_attempts {
        if positions.len() >= count {
            break;
        }

        // Random point in bounds
        let u: f32 = rng.random();
        let v: f32 = rng.random();
        let w: f32 = rng.random();
        let p = min + extent * Vec3::new(u, v, w);

        // Check if near surface
        let d = sdf.sample(p, &ctx).abs();
        if d <= threshold {
            positions.push(p);
        }
    }

    PointCloud::from_positions(positions)
}

/// Poisson disk sampling operation for mesh surfaces.
///
/// Creates a more uniform distribution than pure random sampling by
/// ensuring minimum distance between points.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = PointCloud))]
pub struct Poisson {
    /// Minimum distance between points.
    pub min_distance: f32,
    /// Maximum attempts to place each point.
    pub max_attempts: u32,
}

impl Default for Poisson {
    fn default() -> Self {
        Self {
            min_distance: 0.1,
            max_attempts: 30,
        }
    }
}

impl Poisson {
    /// Creates a new Poisson disk sampler with the given minimum distance.
    pub fn new(min_distance: f32) -> Self {
        Self {
            min_distance,
            ..Default::default()
        }
    }

    /// Applies this Poisson disk sampling operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> PointCloud {
        sample_mesh_poisson(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type PoissonConfig = Poisson;

/// Samples points from a mesh surface using Poisson disk sampling.
///
/// This creates a more uniform distribution than pure random sampling.
pub fn sample_mesh_poisson(mesh: &Mesh, config: &Poisson) -> PointCloud {
    // First sample many points uniformly
    let oversampled = sample_mesh_uniform(mesh, 10000);
    if oversampled.is_empty() {
        return PointCloud::new();
    }

    // Then filter using Poisson disk constraint
    let mut accepted = Vec::new();
    let mut accepted_positions = Vec::new();
    let min_dist_sq = config.min_distance * config.min_distance;

    for (i, &pos) in oversampled.positions.iter().enumerate() {
        let is_valid = accepted_positions
            .iter()
            .all(|&p: &Vec3| (p - pos).length_squared() >= min_dist_sq);

        if is_valid {
            accepted.push(i);
            accepted_positions.push(pos);
        }
    }

    PointCloud {
        positions: accepted_positions,
        normals: if oversampled.has_normals() {
            accepted.iter().map(|&i| oversampled.normals[i]).collect()
        } else {
            Vec::new()
        },
        colors: Vec::new(),
    }
}

// ============================================================================
// Normal estimation
// ============================================================================

/// Estimates normals for a point cloud using local PCA.
///
/// For each point, finds the `k` nearest neighbors and fits a plane
/// using principal component analysis.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> PointCloud {
    if cloud.is_empty() {
        return cloud.clone();
    }

    let k = k.min(cloud.len() - 1).max(3);
    let mut normals = Vec::with_capacity(cloud.len());

    for i in 0..cloud.len() {
        // Find k nearest neighbors (brute force for simplicity)
        let mut distances: Vec<(usize, f32)> = cloud
            .positions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, &p)| (j, (p - cloud.positions[i]).length_squared()))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        // Compute centroid of neighborhood
        let neighbors: Vec<Vec3> = distances.iter().map(|&(j, _)| cloud.positions[j]).collect();

        let centroid: Vec3 = neighbors.iter().copied().sum::<Vec3>() / neighbors.len() as f32;

        // Compute covariance matrix
        let mut cov = [[0.0f32; 3]; 3];
        for &p in &neighbors {
            let d = p - centroid;
            for row in 0..3 {
                for col in 0..3 {
                    cov[row][col] += d[row] * d[col];
                }
            }
        }

        // Find smallest eigenvector using power iteration on inverse
        // (simplified: use cross product of two principal directions)
        let normal = estimate_normal_from_covariance(cov);
        normals.push(normal);
    }

    // Try to orient normals consistently
    orient_normals(&cloud.positions, &mut normals);

    PointCloud {
        positions: cloud.positions.clone(),
        normals,
        colors: cloud.colors.clone(),
    }
}

/// Estimates normal from covariance matrix using simplified eigenvector computation.
fn estimate_normal_from_covariance(cov: [[f32; 3]; 3]) -> Vec3 {
    // Power iteration to find dominant eigenvector
    let mut v = Vec3::new(1.0, 0.0, 0.0);

    for _ in 0..20 {
        let new_v = Vec3::new(
            cov[0][0] * v.x + cov[0][1] * v.y + cov[0][2] * v.z,
            cov[1][0] * v.x + cov[1][1] * v.y + cov[1][2] * v.z,
            cov[2][0] * v.x + cov[2][1] * v.y + cov[2][2] * v.z,
        );
        let len = new_v.length();
        if len > 0.0001 {
            v = new_v / len;
        }
    }

    // The normal is perpendicular to the dominant eigenvector
    // Find a vector not parallel to v
    let arbitrary = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };

    let tangent1 = v.cross(arbitrary).normalize_or_zero();
    let _tangent2 = v.cross(tangent1).normalize_or_zero();

    // The normal is the cross product of the two tangent vectors
    // But actually we want the smallest eigenvector, so we need to compute more carefully
    // For simplicity, compute the eigenvector with smallest eigenvalue by deflation

    // Compute second eigenvector
    let mut v2 = tangent1;
    for _ in 0..20 {
        let new_v = Vec3::new(
            cov[0][0] * v2.x + cov[0][1] * v2.y + cov[0][2] * v2.z,
            cov[1][0] * v2.x + cov[1][1] * v2.y + cov[1][2] * v2.z,
            cov[2][0] * v2.x + cov[2][1] * v2.y + cov[2][2] * v2.z,
        );
        // Remove component along v
        let proj = new_v.dot(v) * v;
        let orthogonal = new_v - proj;
        let len = orthogonal.length();
        if len > 0.0001 {
            v2 = orthogonal / len;
        }
    }

    // Normal is perpendicular to both v and v2
    v.cross(v2).normalize_or_zero()
}

/// Attempts to orient normals consistently (all pointing outward).
fn orient_normals(positions: &[Vec3], normals: &mut [Vec3]) {
    if positions.is_empty() {
        return;
    }

    // Simple heuristic: orient normals away from centroid
    let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / positions.len() as f32;

    for (pos, normal) in positions.iter().zip(normals.iter_mut()) {
        let to_centroid = centroid - *pos;
        if normal.dot(to_centroid) > 0.0 {
            *normal = -*normal;
        }
    }
}

// ============================================================================
// Filtering
// ============================================================================

/// Statistical outlier removal operation for point clouds.
///
/// Removes points whose mean distance to k nearest neighbors exceeds
/// the global mean + std_ratio * std_dev.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PointCloud, output = PointCloud))]
pub struct RemoveOutliers {
    /// Number of neighbors to consider.
    pub k: usize,
    /// Standard deviation multiplier for outlier threshold.
    pub std_ratio: f32,
}

impl Default for RemoveOutliers {
    fn default() -> Self {
        Self {
            k: 10,
            std_ratio: 2.0,
        }
    }
}

impl RemoveOutliers {
    /// Creates a new outlier removal operation with the given neighbor count.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    /// Applies this outlier removal operation to a point cloud.
    pub fn apply(&self, cloud: &PointCloud) -> PointCloud {
        remove_outliers(cloud, self)
    }
}

/// Backwards-compatible type alias.
pub type OutlierConfig = RemoveOutliers;

/// Voxel grid downsampling operation for point clouds.
///
/// Points within the same voxel are averaged into a single point.
/// This reduces point count while preserving overall shape.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PointCloud, output = PointCloud))]
pub struct VoxelDownsample {
    /// Size of voxel grid cells.
    pub voxel_size: f32,
}

impl Default for VoxelDownsample {
    fn default() -> Self {
        Self { voxel_size: 0.1 }
    }
}

impl VoxelDownsample {
    /// Creates a new voxel downsampling operation with the given voxel size.
    pub fn new(voxel_size: f32) -> Self {
        Self { voxel_size }
    }

    /// Applies this voxel downsampling operation to a point cloud.
    pub fn apply(&self, cloud: &PointCloud) -> PointCloud {
        voxel_downsample(cloud, self.voxel_size)
    }
}

/// Crop operation for point clouds.
///
/// Filters points to only those within a bounding box.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PointCloud, output = PointCloud))]
pub struct CropBounds {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
}

impl Default for CropBounds {
    fn default() -> Self {
        Self {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        }
    }
}

impl CropBounds {
    /// Creates a new crop operation with the given bounds.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Applies this crop operation to a point cloud.
    pub fn apply(&self, cloud: &PointCloud) -> PointCloud {
        crop_to_bounds(cloud, self.min, self.max)
    }
}

/// Normal estimation operation for point clouds.
///
/// Estimates normals using local PCA on k nearest neighbors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PointCloud, output = PointCloud))]
pub struct EstimateNormals {
    /// Number of neighbors for local PCA.
    pub k: usize,
}

impl Default for EstimateNormals {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl EstimateNormals {
    /// Creates a new normal estimation operation with the given neighbor count.
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Applies this normal estimation operation to a point cloud.
    pub fn apply(&self, cloud: &PointCloud) -> PointCloud {
        estimate_normals(cloud, self.k)
    }
}

/// Uniform sampling operation for mesh surfaces.
///
/// Samples points uniformly distributed across the mesh surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = PointCloud))]
pub struct UniformSampling {
    /// Number of points to sample.
    pub count: usize,
}

impl Default for UniformSampling {
    fn default() -> Self {
        Self { count: 1000 }
    }
}

impl UniformSampling {
    /// Creates a new uniform sampling operation with the given point count.
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// Applies this uniform sampling operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> PointCloud {
        sample_mesh_uniform(mesh, self.count)
    }
}

/// Removes statistical outliers from a point cloud.
///
/// Points whose mean distance to k nearest neighbors exceeds
/// the global mean + std_ratio * std_dev are removed.
pub fn remove_outliers(cloud: &PointCloud, config: &RemoveOutliers) -> PointCloud {
    if cloud.len() <= config.k {
        return cloud.clone();
    }

    // Compute mean distance to k nearest neighbors for each point
    let mut mean_distances = Vec::with_capacity(cloud.len());

    for i in 0..cloud.len() {
        let mut distances: Vec<f32> = cloud
            .positions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &p)| (p - cloud.positions[i]).length())
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k_nearest: f32 = distances.iter().take(config.k).sum::<f32>() / config.k as f32;
        mean_distances.push(k_nearest);
    }

    // Compute global statistics
    let global_mean: f32 = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
    let variance: f32 = mean_distances
        .iter()
        .map(|&d| (d - global_mean).powi(2))
        .sum::<f32>()
        / mean_distances.len() as f32;
    let std_dev = variance.sqrt();

    let threshold = global_mean + config.std_ratio * std_dev;

    // Filter points
    let indices: Vec<usize> = mean_distances
        .iter()
        .enumerate()
        .filter(|&(_, d)| *d <= threshold)
        .map(|(i, _)| i)
        .collect();

    PointCloud {
        positions: indices.iter().map(|&i| cloud.positions[i]).collect(),
        normals: if cloud.has_normals() {
            indices.iter().map(|&i| cloud.normals[i]).collect()
        } else {
            Vec::new()
        },
        colors: if cloud.has_colors() {
            indices.iter().map(|&i| cloud.colors[i]).collect()
        } else {
            Vec::new()
        },
    }
}

/// Downsamples a point cloud using voxel grid filtering.
///
/// Points within the same voxel are averaged into a single point.
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f32) -> PointCloud {
    use std::collections::HashMap;

    if cloud.is_empty() || voxel_size <= 0.0 {
        return cloud.clone();
    }

    type VoxelKey = (i32, i32, i32);
    type VoxelAccum = (Vec3, Vec3, Vec4, u32);
    // Map voxel coordinates to accumulated points (position, normal, color, count)
    let mut voxels: HashMap<VoxelKey, VoxelAccum> = HashMap::new();

    for i in 0..cloud.len() {
        let pos = cloud.positions[i];
        let voxel = (
            (pos.x / voxel_size).floor() as i32,
            (pos.y / voxel_size).floor() as i32,
            (pos.z / voxel_size).floor() as i32,
        );

        let normal = if cloud.has_normals() {
            cloud.normals[i]
        } else {
            Vec3::ZERO
        };
        let color = if cloud.has_colors() {
            cloud.colors[i]
        } else {
            Vec4::ZERO
        };

        let entry = voxels
            .entry(voxel)
            .or_insert((Vec3::ZERO, Vec3::ZERO, Vec4::ZERO, 0));
        entry.0 += pos;
        entry.1 += normal;
        entry.2 += color;
        entry.3 += 1;
    }

    // Compute averages
    let mut positions = Vec::with_capacity(voxels.len());
    let mut normals = Vec::with_capacity(voxels.len());
    let mut colors = Vec::with_capacity(voxels.len());

    for (pos_sum, normal_sum, color_sum, count) in voxels.values() {
        let n = *count as f32;
        positions.push(*pos_sum / n);

        if cloud.has_normals() {
            normals.push((*normal_sum / n).normalize_or_zero());
        }
        if cloud.has_colors() {
            colors.push(*color_sum / n);
        }
    }

    PointCloud {
        positions,
        normals,
        colors,
    }
}

/// Crops a point cloud to points within a bounding box.
pub fn crop_to_bounds(cloud: &PointCloud, min: Vec3, max: Vec3) -> PointCloud {
    let indices: Vec<usize> = cloud
        .positions
        .iter()
        .enumerate()
        .filter(|&(_, p)| {
            p.x >= min.x
                && p.x <= max.x
                && p.y >= min.y
                && p.y <= max.y
                && p.z >= min.z
                && p.z <= max.z
        })
        .map(|(i, _)| i)
        .collect();

    PointCloud {
        positions: indices.iter().map(|&i| cloud.positions[i]).collect(),
        normals: if cloud.has_normals() {
            indices.iter().map(|&i| cloud.normals[i]).collect()
        } else {
            Vec::new()
        },
        colors: if cloud.has_colors() {
            indices.iter().map(|&i| cloud.colors[i]).collect()
        } else {
            Vec::new()
        },
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Transforms all points by a matrix.
pub fn transform(cloud: &PointCloud, matrix: glam::Mat4) -> PointCloud {
    let normal_matrix = matrix.inverse().transpose();

    PointCloud {
        positions: cloud
            .positions
            .iter()
            .map(|&p| matrix.transform_point3(p))
            .collect(),
        normals: if cloud.has_normals() {
            cloud
                .normals
                .iter()
                .map(|&n| normal_matrix.transform_vector3(n).normalize_or_zero())
                .collect()
        } else {
            Vec::new()
        },
        colors: cloud.colors.clone(),
    }
}

// ============================================================================
// Seeded RNG (splitmix64, no external dependency)
// ============================================================================

/// Simple splitmix64 bijection for u64 → u64.
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Minimal seeded RNG backed by splitmix64.
struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(splitmix64(seed.wrapping_add(1)))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }

    /// Returns a value in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Returns a value in [-1, 1).
    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ============================================================================
// Generator ops — produce a PointCloud with no input
// ============================================================================

/// Generates random points on the surface of a sphere.
///
/// Uses rejection sampling in the unit cube to achieve a uniform distribution
/// on the sphere surface. Each point's normal equals the outward direction from
/// the center.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = PointCloud))]
pub struct SphereSurface {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
    /// Number of points to generate.
    pub count: usize,
    /// Seed for the RNG. Same seed → same result.
    pub seed: u64,
}

impl Default for SphereSurface {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 1.0,
            count: 1000,
            seed: 0,
        }
    }
}

impl SphereSurface {
    /// Generates the point cloud.
    pub fn apply(&self) -> PointCloud {
        let mut rng = SeededRng::new(self.seed);
        let mut positions = Vec::with_capacity(self.count);
        let mut normals = Vec::with_capacity(self.count);

        while positions.len() < self.count {
            // Rejection sample a unit direction.
            let v = Vec3::new(
                rng.next_f32_signed(),
                rng.next_f32_signed(),
                rng.next_f32_signed(),
            );
            let len = v.length();
            if !(1e-6..=1.0).contains(&len) {
                continue;
            }
            let normal = v / len;
            positions.push(self.center + normal * self.radius);
            normals.push(normal);
        }

        PointCloud {
            positions,
            normals,
            colors: Vec::new(),
        }
    }
}

/// Generates random points uniformly distributed inside a sphere volume.
///
/// Each point's normal equals the outward direction from the center (useful
/// for downstream processing; set to zero if you need interior points without normals).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = PointCloud))]
pub struct SphereVolume {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
    /// Number of points to generate.
    pub count: usize,
    /// Seed for the RNG.
    pub seed: u64,
}

impl Default for SphereVolume {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 1.0,
            count: 1000,
            seed: 0,
        }
    }
}

impl SphereVolume {
    /// Generates the point cloud.
    pub fn apply(&self) -> PointCloud {
        let mut rng = SeededRng::new(self.seed);
        let mut positions = Vec::with_capacity(self.count);
        let mut normals = Vec::with_capacity(self.count);

        while positions.len() < self.count {
            let v = Vec3::new(
                rng.next_f32_signed(),
                rng.next_f32_signed(),
                rng.next_f32_signed(),
            );
            if v.length_squared() > 1.0 {
                continue;
            }
            let point = self.center + v * self.radius;
            let normal = (point - self.center).normalize_or_zero();
            positions.push(point);
            normals.push(normal);
        }

        PointCloud {
            positions,
            normals,
            colors: Vec::new(),
        }
    }
}

/// Generates a regular grid of points inside a box.
///
/// Points are placed at uniform intervals along each axis. The total number of
/// points equals `resolution.x * resolution.y * resolution.z`.
/// Normals are set to `Vec3::Z`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = PointCloud))]
pub struct GridPoints {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
    /// Number of points along each axis.
    pub resolution: UVec3,
}

impl Default for GridPoints {
    fn default() -> Self {
        Self {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            resolution: UVec3::splat(10),
        }
    }
}

impl GridPoints {
    /// Generates the point cloud.
    pub fn apply(&self) -> PointCloud {
        let rx = self.resolution.x as usize;
        let ry = self.resolution.y as usize;
        let rz = self.resolution.z as usize;
        let total = rx * ry * rz;

        let mut positions = Vec::with_capacity(total);
        let extent = self.max - self.min;

        for iz in 0..rz {
            for iy in 0..ry {
                for ix in 0..rx {
                    // Place on grid; for resolution=1 put at 0.5 (center of cell).
                    let tx = if rx > 1 {
                        ix as f32 / (rx - 1) as f32
                    } else {
                        0.5
                    };
                    let ty = if ry > 1 {
                        iy as f32 / (ry - 1) as f32
                    } else {
                        0.5
                    };
                    let tz = if rz > 1 {
                        iz as f32 / (rz - 1) as f32
                    } else {
                        0.5
                    };
                    positions.push(self.min + extent * Vec3::new(tx, ty, tz));
                }
            }
        }

        PointCloud {
            normals: vec![Vec3::Z; positions.len()],
            positions,
            colors: Vec::new(),
        }
    }
}

/// Generates random points uniformly distributed inside a box volume.
///
/// Normals are set to `Vec3::Z` (no meaningful outward direction for axis-aligned boxes).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = PointCloud))]
pub struct BoxVolume {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
    /// Number of points to generate.
    pub count: usize,
    /// Seed for the RNG.
    pub seed: u64,
}

impl Default for BoxVolume {
    fn default() -> Self {
        Self {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            count: 1000,
            seed: 0,
        }
    }
}

impl BoxVolume {
    /// Generates the point cloud.
    pub fn apply(&self) -> PointCloud {
        let mut rng = SeededRng::new(self.seed);
        let extent = self.max - self.min;
        let mut positions = Vec::with_capacity(self.count);

        for _ in 0..self.count {
            let t = Vec3::new(rng.next_f32(), rng.next_f32(), rng.next_f32());
            positions.push(self.min + extent * t);
        }

        PointCloud {
            normals: vec![Vec3::Z; positions.len()],
            positions,
            colors: Vec::new(),
        }
    }
}

/// Samples points from an SDF field where the SDF value is close to `iso_value`.
///
/// Probes a regular grid at the given `resolution` and accepts candidate
/// points where `|sdf(p) - iso_value| < thickness`. From those candidates,
/// `count` points are chosen at random. Normals are estimated from the SDF
/// gradient via central finite differences.
///
/// Note: This op is not registered with `DynOp` because the SDF field is a
/// generic parameter that cannot be type-erased in the op system. Call
/// [`SdfSurface::apply_sdf`] directly with a concrete field.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SdfSurface {
    /// Minimum corner of the sampling region.
    pub min: Vec3,
    /// Maximum corner of the sampling region.
    pub max: Vec3,
    /// Grid resolution for candidate generation (resolution³ probes).
    pub resolution: usize,
    /// Maximum number of output points.
    pub count: usize,
    /// Surface iso-value (0 for a standard zero-level-set SDF).
    pub iso_value: f32,
    /// Half-width of the accepted band: `|sdf(p) - iso_value| < thickness`.
    pub thickness: f32,
    /// Seed for the RNG (used when sub-sampling candidates).
    pub seed: u64,
    /// Step size for gradient finite differences (relative to cell size).
    pub gradient_eps: f32,
}

impl Default for SdfSurface {
    fn default() -> Self {
        Self {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            resolution: 32,
            count: 1000,
            iso_value: 0.0,
            thickness: 0.05,
            seed: 0,
            gradient_eps: 0.01,
        }
    }
}

impl SdfSurface {
    /// Generates the point cloud from an SDF field.
    ///
    /// The field is evaluated at each grid point; candidates within the
    /// iso-band are collected, shuffled with the seeded RNG, then truncated
    /// to `self.count`.
    pub fn apply_sdf<F: Field<Vec3, f32>>(&self, sdf: &F) -> PointCloud {
        let ctx = EvalContext::new();
        let extent = self.max - self.min;
        let res = self.resolution.max(1);
        let step = extent / res as f32;
        let eps = self.gradient_eps;

        let mut candidates: Vec<Vec3> = Vec::new();

        for iz in 0..res {
            for iy in 0..res {
                for ix in 0..res {
                    let t = Vec3::new(
                        (ix as f32 + 0.5) / res as f32,
                        (iy as f32 + 0.5) / res as f32,
                        (iz as f32 + 0.5) / res as f32,
                    );
                    let p = self.min + extent * t;
                    let d = sdf.sample(p, &ctx);
                    if (d - self.iso_value).abs() < self.thickness {
                        candidates.push(p);
                    }
                }
            }
        }

        // Shuffle candidates with seeded RNG, then take `count`.
        let mut rng = SeededRng::new(self.seed);
        // Fisher-Yates shuffle
        for i in (1..candidates.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            candidates.swap(i, j);
        }
        candidates.truncate(self.count);

        // Estimate normals via central finite differences.
        let half_step = step * 0.5;
        let fd_eps = Vec3::new(
            eps.max(half_step.x * 0.5),
            eps.max(half_step.y * 0.5),
            eps.max(half_step.z * 0.5),
        );

        let mut positions = Vec::with_capacity(candidates.len());
        let mut normals = Vec::with_capacity(candidates.len());

        for p in candidates {
            let dx = sdf.sample(p + Vec3::new(fd_eps.x, 0.0, 0.0), &ctx)
                - sdf.sample(p - Vec3::new(fd_eps.x, 0.0, 0.0), &ctx);
            let dy = sdf.sample(p + Vec3::new(0.0, fd_eps.y, 0.0), &ctx)
                - sdf.sample(p - Vec3::new(0.0, fd_eps.y, 0.0), &ctx);
            let dz = sdf.sample(p + Vec3::new(0.0, 0.0, fd_eps.z), &ctx)
                - sdf.sample(p - Vec3::new(0.0, 0.0, fd_eps.z), &ctx);
            let grad = Vec3::new(dx, dy, dz).normalize_or_zero();
            positions.push(p);
            normals.push(grad);
        }

        PointCloud {
            positions,
            normals,
            colors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_mesh::Cuboid;

    #[test]
    fn test_point_cloud_basic() {
        let mut cloud = PointCloud::new();
        assert!(cloud.is_empty());

        cloud.add_point(Vec3::ZERO);
        cloud.add_point(Vec3::ONE);
        assert_eq!(cloud.len(), 2);
        assert!(!cloud.has_normals());
    }

    #[test]
    fn test_point_cloud_with_normals() {
        let cloud =
            PointCloud::from_positions_normals(vec![Vec3::ZERO, Vec3::ONE], vec![Vec3::Y, Vec3::Y]);
        assert_eq!(cloud.len(), 2);
        assert!(cloud.has_normals());
    }

    #[test]
    fn test_bounding_box() {
        let cloud =
            PointCloud::from_positions(vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0)]);

        let (min, max) = cloud.bounding_box().unwrap();
        assert_eq!(min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_centroid() {
        let cloud =
            PointCloud::from_positions(vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0)]);

        let c = cloud.centroid().unwrap();
        assert!((c - Vec3::ONE).length() < 0.001);
    }

    #[test]
    fn test_sample_mesh_uniform() {
        let mesh = Cuboid::unit().apply();
        let cloud = sample_mesh_uniform(&mesh, 100);

        assert!(cloud.len() >= 50); // Should get most of the requested points
        assert!(cloud.has_normals());

        // All points should be on or near the surface
        for &p in &cloud.positions {
            let max_coord = p.x.abs().max(p.y.abs()).max(p.z.abs());
            assert!(max_coord <= 0.6); // Within cube bounds
        }
    }

    #[test]
    fn test_voxel_downsample() {
        let cloud = PointCloud::from_positions(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.01, 0.01, 0.01),
            Vec3::new(1.0, 1.0, 1.0),
        ]);

        let downsampled = voxel_downsample(&cloud, 0.5);

        // First two points should be merged
        assert_eq!(downsampled.len(), 2);
    }

    #[test]
    fn test_crop_to_bounds() {
        let cloud = PointCloud::from_positions(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(2.0, 2.0, 2.0),
        ]);

        let cropped = crop_to_bounds(&cloud, Vec3::ZERO, Vec3::ONE);
        assert_eq!(cropped.len(), 2);
    }

    #[test]
    fn test_merge() {
        let mut cloud1 = PointCloud::from_positions(vec![Vec3::ZERO]);
        let cloud2 = PointCloud::from_positions(vec![Vec3::ONE]);

        cloud1.merge(&cloud2);
        assert_eq!(cloud1.len(), 2);
    }

    #[test]
    fn test_transform() {
        let cloud = PointCloud::from_positions(vec![Vec3::ONE]);
        let matrix = glam::Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));

        let transformed = transform(&cloud, matrix);
        assert!((transformed.positions[0] - Vec3::new(2.0, 1.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn test_estimate_normals() {
        let mesh = Cuboid::unit().apply();
        let cloud = sample_mesh_uniform(&mesh, 100);

        // Remove existing normals to test estimation
        let cloud_no_normals = PointCloud::from_positions(cloud.positions.clone());
        let with_normals = estimate_normals(&cloud_no_normals, 10);

        assert!(with_normals.has_normals());
        assert_eq!(with_normals.len(), cloud_no_normals.len());
    }

    #[test]
    fn test_sample_mesh_poisson() {
        let mesh = Cuboid::unit().apply();
        let config = PoissonConfig {
            min_distance: 0.2,
            max_attempts: 30,
        };

        let cloud = sample_mesh_poisson(&mesh, &config);

        // Check minimum distance constraint
        for i in 0..cloud.len() {
            for j in (i + 1)..cloud.len() {
                let dist = (cloud.positions[i] - cloud.positions[j]).length();
                assert!(dist >= config.min_distance * 0.9); // Small tolerance
            }
        }
    }

    #[test]
    fn test_has_positions_trait() {
        use unshape_geometry::HasPositions;

        let mut cloud =
            PointCloud::from_positions(vec![Vec3::ZERO, Vec3::ONE, Vec3::new(2.0, 0.0, 0.0)]);

        assert_eq!(cloud.vertex_count(), 3);
        assert_eq!(cloud.positions().len(), 3);
        assert_eq!(cloud.get_position(1), Some(Vec3::ONE));

        // Test mutation through trait
        cloud.set_position(0, Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(cloud.positions[0], Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_has_normals_trait() {
        use unshape_geometry::HasNormals;

        let mut cloud =
            PointCloud::from_positions_normals(vec![Vec3::ZERO, Vec3::ONE], vec![Vec3::Y, Vec3::X]);

        assert_eq!(cloud.normals().len(), 2);
        assert_eq!(cloud.normals()[0], Vec3::Y);

        // Test mutation through trait
        cloud.normals_mut()[0] = Vec3::Z;
        assert_eq!(cloud.normals[0], Vec3::Z);
    }

    #[test]
    fn test_has_colors_trait() {
        use unshape_geometry::HasColors;

        let mut cloud = PointCloud {
            positions: vec![Vec3::ZERO, Vec3::ONE],
            normals: Vec::new(),
            colors: vec![Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)],
        };

        assert_eq!(cloud.colors().len(), 2);
        assert_eq!(cloud.colors()[0], Vec4::new(1.0, 0.0, 0.0, 1.0));

        // Test mutation through trait
        cloud.colors_mut()[0] = Vec4::new(0.0, 0.0, 1.0, 1.0);
        assert_eq!(cloud.colors[0], Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    // ========================================================================
    // Generator op tests
    // ========================================================================

    #[test]
    fn test_sphere_surface_count() {
        let cloud = SphereSurface {
            center: Vec3::ZERO,
            radius: 2.0,
            count: 200,
            seed: 42,
        }
        .apply();
        assert_eq!(cloud.len(), 200);
        assert!(cloud.has_normals());
    }

    #[test]
    fn test_sphere_surface_on_surface() {
        let radius = 3.0;
        let cloud = SphereSurface {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius,
            count: 100,
            seed: 7,
        }
        .apply();
        for &p in &cloud.positions {
            let dist = (p - Vec3::new(1.0, 2.0, 3.0)).length();
            assert!(
                (dist - radius).abs() < 1e-4,
                "point {p:?} not on sphere surface: dist={dist}"
            );
        }
    }

    #[test]
    fn test_sphere_surface_normals_unit() {
        let cloud = SphereSurface::default().apply();
        for &n in &cloud.normals {
            assert!((n.length() - 1.0).abs() < 1e-5, "normal not unit: {:?}", n);
        }
    }

    #[test]
    fn test_sphere_volume_count() {
        let cloud = SphereVolume {
            count: 150,
            seed: 1,
            ..Default::default()
        }
        .apply();
        assert_eq!(cloud.len(), 150);
        assert!(cloud.has_normals());
    }

    #[test]
    fn test_sphere_volume_inside_sphere() {
        let radius = 1.5;
        let cloud = SphereVolume {
            radius,
            count: 200,
            seed: 3,
            ..Default::default()
        }
        .apply();
        for &p in &cloud.positions {
            assert!(p.length() <= radius + 1e-4, "point outside sphere: {:?}", p);
        }
    }

    #[test]
    fn test_grid_points_count() {
        let cloud = GridPoints {
            resolution: UVec3::new(3, 4, 5),
            ..Default::default()
        }
        .apply();
        assert_eq!(cloud.len(), 3 * 4 * 5);
    }

    #[test]
    fn test_grid_points_within_bounds() {
        let min = Vec3::new(-2.0, -1.0, 0.0);
        let max = Vec3::new(2.0, 1.0, 4.0);
        let cloud = GridPoints {
            min,
            max,
            resolution: UVec3::splat(5),
        }
        .apply();
        for &p in &cloud.positions {
            assert!(p.x >= min.x - 1e-5 && p.x <= max.x + 1e-5);
            assert!(p.y >= min.y - 1e-5 && p.y <= max.y + 1e-5);
            assert!(p.z >= min.z - 1e-5 && p.z <= max.z + 1e-5);
        }
    }

    #[test]
    fn test_box_volume_count() {
        let cloud = BoxVolume {
            count: 300,
            seed: 99,
            ..Default::default()
        }
        .apply();
        assert_eq!(cloud.len(), 300);
    }

    #[test]
    fn test_box_volume_within_bounds() {
        let min = Vec3::new(0.0, 0.0, 0.0);
        let max = Vec3::new(5.0, 3.0, 2.0);
        let cloud = BoxVolume {
            min,
            max,
            count: 500,
            seed: 0,
        }
        .apply();
        for &p in &cloud.positions {
            assert!(p.x >= min.x - 1e-5 && p.x <= max.x + 1e-5);
            assert!(p.y >= min.y - 1e-5 && p.y <= max.y + 1e-5);
            assert!(p.z >= min.z - 1e-5 && p.z <= max.z + 1e-5);
        }
    }

    #[test]
    fn test_sdf_surface_sphere() {
        // SDF of a unit sphere: |p| - 1
        struct UnitSphere;
        impl Field<Vec3, f32> for UnitSphere {
            fn sample(&self, p: Vec3, _ctx: &EvalContext) -> f32 {
                p.length() - 1.0
            }
        }

        let op = SdfSurface {
            min: Vec3::splat(-1.5),
            max: Vec3::splat(1.5),
            resolution: 16,
            count: 100,
            iso_value: 0.0,
            thickness: 0.15,
            seed: 0,
            gradient_eps: 0.01,
        };

        let cloud = op.apply_sdf(&UnitSphere);
        assert!(!cloud.is_empty(), "expected points near SDF surface");
        assert!(cloud.len() <= 100);
        for &p in &cloud.positions {
            let d = (p.length() - 1.0).abs();
            assert!(d < 0.15 + 1e-4, "point not near surface: dist={d}");
        }
        for &n in &cloud.normals {
            let len = n.length();
            assert!(len > 0.9 && len <= 1.0 + 1e-5, "normal not unit: {len}");
        }
    }

    #[test]
    fn test_sdf_surface_count_capped() {
        struct PlaneSdf;
        impl Field<Vec3, f32> for PlaneSdf {
            fn sample(&self, p: Vec3, _ctx: &EvalContext) -> f32 {
                p.y
            }
        }

        let cloud = SdfSurface {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            resolution: 10,
            count: 5,
            iso_value: 0.0,
            thickness: 0.5,
            seed: 1,
            gradient_eps: 0.01,
        }
        .apply_sdf(&PlaneSdf);

        assert!(cloud.len() <= 5);
    }
}
