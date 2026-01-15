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
//! use rhizome_resin_pointcloud::{PointCloud, sample_mesh_uniform};
//! use rhizome_resin_mesh::box_mesh;
//!
//! let mesh = box_mesh();
//! let cloud = sample_mesh_uniform(&mesh, 1000);
//!
//! assert!(cloud.len() >= 100); // Some points generated
//! ```

use glam::Vec3;
use rand::Rng;
use rhizome_resin_field::{EvalContext, Field};
use rhizome_resin_mesh::Mesh;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Registers all pointcloud operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of pointcloud ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<Poisson>("resin::Poisson");
    registry.register_type::<RemoveOutliers>("resin::RemoveOutliers");
}

/// A point cloud with positions, optional normals, and optional colors.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// Point positions.
    pub positions: Vec<Vec3>,
    /// Point normals (same length as positions, or empty).
    pub normals: Vec<Vec3>,
    /// Point colors as RGB in [0, 1] (same length as positions, or empty).
    pub colors: Vec<Vec3>,
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
            self.colors.push(Vec3::ONE);
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
            self.colors.push(Vec3::ONE);
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
                .extend(std::iter::repeat(Vec3::ZERO).take(other.len()));
        }

        // Handle colors
        if self.has_colors() && other.has_colors() {
            self.colors.extend_from_slice(&other.colors);
        } else if self.has_colors() {
            self.colors
                .extend(std::iter::repeat(Vec3::ONE).take(other.len()));
        }
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
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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

    // Map voxel coordinates to accumulated points
    let mut voxels: HashMap<(i32, i32, i32), (Vec3, Vec3, Vec3, u32)> = HashMap::new();

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
            Vec3::ZERO
        };

        let entry = voxels
            .entry(voxel)
            .or_insert((Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0));
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

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_mesh::box_mesh;

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
        let mesh = box_mesh();
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
        let mesh = box_mesh();
        let cloud = sample_mesh_uniform(&mesh, 100);

        // Remove existing normals to test estimation
        let cloud_no_normals = PointCloud::from_positions(cloud.positions.clone());
        let with_normals = estimate_normals(&cloud_no_normals, 10);

        assert!(with_normals.has_normals());
        assert_eq!(with_normals.len(), cloud_no_normals.len());
    }

    #[test]
    fn test_sample_mesh_poisson() {
        let mesh = box_mesh();
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
}
