//! Level of Detail (LOD) generation.
//!
//! Automatically generates multiple mesh detail levels for efficient rendering
//! at different viewing distances.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_mesh::{UvSphere, LodConfig, generate_lod_chain};
//!
//! let high_poly = UvSphere::new(1.0, 32, 16).apply();
//! let lod_chain = LodConfig::default().apply(&high_poly);
//!
//! // Use different LODs based on distance
//! let current_lod = lod_chain.select_by_screen_size(0.1); // 10% of screen
//! ```

use crate::{DecimateConfig, Mesh, decimate};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Generates a LOD chain from a high-poly mesh.
///
/// Creates a sequence of meshes at decreasing detail levels,
/// suitable for distance-based rendering optimization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = LodChain))]
pub struct GenerateLodChain {
    /// Number of LOD levels to generate (including original).
    pub levels: usize,
    /// Reduction ratio between consecutive levels (0.0 - 1.0).
    /// 0.5 means each level has half the triangles of the previous.
    pub reduction_ratio: f32,
    /// Minimum triangles to keep in the lowest LOD.
    pub min_triangles: usize,
    /// Whether to preserve mesh boundaries during decimation.
    pub preserve_boundary: bool,
    /// Maximum geometric error allowed during decimation.
    pub max_error: f32,
    /// Screen size thresholds for each LOD (as fraction of screen height).
    /// If None, computed automatically.
    pub screen_thresholds: Option<Vec<f32>>,
}

impl Default for GenerateLodChain {
    fn default() -> Self {
        Self {
            levels: 4,
            reduction_ratio: 0.5,
            min_triangles: 32,
            preserve_boundary: true,
            max_error: f32::MAX,
            screen_thresholds: None,
        }
    }
}

impl GenerateLodChain {
    /// Creates a config with a specific number of levels.
    pub fn with_levels(levels: usize) -> Self {
        Self {
            levels,
            ..Default::default()
        }
    }

    /// Sets the reduction ratio between levels.
    pub fn with_reduction_ratio(mut self, ratio: f32) -> Self {
        self.reduction_ratio = ratio.clamp(0.1, 0.9);
        self
    }

    /// Applies this operation to generate a LOD chain.
    pub fn apply(&self, mesh: &Mesh) -> LodChain {
        generate_lod_chain(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type LodConfig = GenerateLodChain;

/// A chain of meshes at different detail levels.
#[derive(Debug, Clone)]
pub struct LodChain {
    /// The meshes from highest to lowest detail.
    pub levels: Vec<LodLevel>,
}

/// A single LOD level.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// The mesh at this detail level.
    pub mesh: Mesh,
    /// Triangle count at this level.
    pub triangle_count: usize,
    /// Screen size threshold (switch to lower LOD when below this).
    pub screen_threshold: f32,
    /// Reduction ratio from original (1.0 for LOD 0).
    pub ratio_from_original: f32,
}

impl LodChain {
    /// Returns the number of LOD levels.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Returns true if there are no LOD levels.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Gets a specific LOD level.
    pub fn get(&self, level: usize) -> Option<&LodLevel> {
        self.levels.get(level)
    }

    /// Gets the mesh at a specific LOD level.
    pub fn mesh(&self, level: usize) -> Option<&Mesh> {
        self.levels.get(level).map(|l| &l.mesh)
    }

    /// Gets the highest detail mesh (LOD 0).
    pub fn highest_detail(&self) -> Option<&Mesh> {
        self.mesh(0)
    }

    /// Gets the lowest detail mesh.
    pub fn lowest_detail(&self) -> Option<&Mesh> {
        self.levels.last().map(|l| &l.mesh)
    }

    /// Selects the appropriate LOD based on screen size coverage.
    ///
    /// `screen_size` is the fraction of screen height covered by the object (0.0 - 1.0).
    /// Returns the LOD level index (0 = highest detail).
    pub fn select_by_screen_size(&self, screen_size: f32) -> usize {
        for (i, level) in self.levels.iter().enumerate() {
            if screen_size >= level.screen_threshold {
                return i;
            }
        }
        // Return lowest LOD if below all thresholds
        self.levels.len().saturating_sub(1)
    }

    /// Selects the appropriate LOD based on distance.
    ///
    /// `distance` is the distance from camera to object.
    /// `object_size` is the approximate world-space size of the object.
    /// `fov` is the vertical field of view in radians.
    pub fn select_by_distance(&self, distance: f32, object_size: f32, fov: f32) -> usize {
        // Calculate approximate screen size
        let screen_size = if distance > 0.0 {
            (object_size / (2.0 * distance * (fov / 2.0).tan())).clamp(0.0, 1.0)
        } else {
            1.0 // Very close, use highest detail
        };
        self.select_by_screen_size(screen_size)
    }

    /// Returns total memory estimate in bytes for all LOD levels.
    pub fn memory_estimate(&self) -> usize {
        self.levels.iter().map(|l| l.mesh.memory_estimate()).sum()
    }
}

/// Generates a LOD chain from a high-poly mesh.
pub fn generate_lod_chain(mesh: &Mesh, config: &GenerateLodChain) -> LodChain {
    if mesh.positions.is_empty() || config.levels == 0 {
        return LodChain { levels: vec![] };
    }

    let original_triangles = mesh.triangle_count();
    let mut levels = Vec::with_capacity(config.levels);

    // Compute screen thresholds if not provided
    let thresholds = config.screen_thresholds.clone().unwrap_or_else(|| {
        // Default thresholds: use geometric progression
        // LOD 0: > 0.25 screen height, LOD 1: > 0.125, etc.
        (0..config.levels)
            .map(|i| 0.25 * (0.5f32).powi(i as i32))
            .collect()
    });

    // LOD 0 is the original mesh
    levels.push(LodLevel {
        mesh: mesh.clone(),
        triangle_count: original_triangles,
        screen_threshold: thresholds.first().copied().unwrap_or(0.25),
        ratio_from_original: 1.0,
    });

    // Generate subsequent LOD levels
    let mut current_mesh = mesh.clone();
    let mut cumulative_ratio = 1.0;

    for i in 1..config.levels {
        cumulative_ratio *= config.reduction_ratio;
        let target_triangles = ((original_triangles as f32) * cumulative_ratio) as usize;

        // Stop if we've hit minimum triangles
        if target_triangles < config.min_triangles {
            break;
        }

        // Decimate from current (not original) for better quality
        let decimate_config = DecimateConfig {
            target_triangles: Some(target_triangles),
            target_ratio: None,
            max_error: config.max_error,
            preserve_boundary: config.preserve_boundary,
        };

        let decimated = decimate(&current_mesh, decimate_config);
        let actual_triangles = decimated.triangle_count();

        // Skip if decimation didn't reduce triangles significantly
        if actual_triangles >= current_mesh.triangle_count() * 9 / 10 {
            break;
        }

        levels.push(LodLevel {
            mesh: decimated.clone(),
            triangle_count: actual_triangles,
            screen_threshold: thresholds.get(i).copied().unwrap_or(0.0),
            ratio_from_original: actual_triangles as f32 / original_triangles as f32,
        });

        current_mesh = decimated;
    }

    LodChain { levels }
}

/// Generates LODs with specific triangle counts.
///
/// This gives more control over the exact triangle budget for each level.
pub fn generate_lod_chain_with_targets(
    mesh: &Mesh,
    targets: &[usize],
    preserve_boundary: bool,
) -> LodChain {
    if mesh.positions.is_empty() || targets.is_empty() {
        return LodChain { levels: vec![] };
    }

    let original_triangles = mesh.triangle_count();
    let mut levels = Vec::with_capacity(targets.len() + 1);

    // LOD 0 is the original mesh
    levels.push(LodLevel {
        mesh: mesh.clone(),
        triangle_count: original_triangles,
        screen_threshold: 0.5,
        ratio_from_original: 1.0,
    });

    // Generate each target LOD
    let mut current_mesh = mesh.clone();

    for (i, &target) in targets.iter().enumerate() {
        if target >= current_mesh.triangle_count() {
            continue;
        }

        let decimate_config = DecimateConfig {
            target_triangles: Some(target),
            target_ratio: None,
            max_error: f32::MAX,
            preserve_boundary,
        };

        let decimated = decimate(&current_mesh, decimate_config);
        let actual_triangles = decimated.triangle_count();

        levels.push(LodLevel {
            mesh: decimated.clone(),
            triangle_count: actual_triangles,
            screen_threshold: 0.5 * (0.5f32).powi((i + 1) as i32),
            ratio_from_original: actual_triangles as f32 / original_triangles as f32,
        });

        current_mesh = decimated;
    }

    LodChain { levels }
}

/// Estimates the bounding sphere radius of a mesh for LOD calculations.
pub fn estimate_bounding_radius(mesh: &Mesh) -> f32 {
    if mesh.positions.is_empty() {
        return 0.0;
    }

    // Find centroid
    let centroid: glam::Vec3 =
        mesh.positions.iter().copied().sum::<glam::Vec3>() / mesh.positions.len() as f32;

    // Find max distance from centroid
    mesh.positions
        .iter()
        .map(|p| p.distance(centroid))
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UvSphere;

    #[test]
    fn test_generate_lod_chain() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let config = LodConfig::default();
        let chain = generate_lod_chain(&mesh, &config);

        // Should have multiple levels
        assert!(chain.len() > 1);

        // First level should be original
        assert_eq!(chain.levels[0].triangle_count, mesh.triangle_count());

        // Each subsequent level should have fewer triangles
        for i in 1..chain.len() {
            assert!(
                chain.levels[i].triangle_count < chain.levels[i - 1].triangle_count,
                "LOD {} should have fewer triangles than LOD {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_lod_chain_selection() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let config = LodConfig::with_levels(4);
        let chain = generate_lod_chain(&mesh, &config);

        // Large screen size should select LOD 0
        assert_eq!(chain.select_by_screen_size(1.0), 0);
        assert_eq!(chain.select_by_screen_size(0.5), 0);

        // Very small screen size should select lowest LOD
        let lowest = chain.select_by_screen_size(0.001);
        assert!(lowest > 0);
    }

    #[test]
    fn test_lod_chain_by_distance() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let chain = generate_lod_chain(&mesh, &LodConfig::default());

        let fov = std::f32::consts::FRAC_PI_2; // 90 degrees
        let object_size = 1.0;

        // Very close should use high detail
        let close = chain.select_by_distance(1.0, object_size, fov);

        // Far away should use lower detail
        let far = chain.select_by_distance(100.0, object_size, fov);

        assert!(far >= close);
    }

    #[test]
    fn test_lod_config_builder() {
        let mut config = GenerateLodChain::with_levels(6).with_reduction_ratio(0.6);
        config.min_triangles = 100;
        config.preserve_boundary = false;
        config.max_error = 0.1;

        assert_eq!(config.levels, 6);
        assert!((config.reduction_ratio - 0.6).abs() < 0.001);
        assert_eq!(config.min_triangles, 100);
        assert!(!config.preserve_boundary);
        assert!((config.max_error - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_lod_chain_with_targets() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let original_tris = mesh.triangle_count();

        let targets = vec![original_tris / 2, original_tris / 4, original_tris / 8];
        let chain = generate_lod_chain_with_targets(&mesh, &targets, false);

        // Should have original + target levels (may be fewer if decimation stops early)
        assert!(chain.len() > 1);

        // First should be original
        assert_eq!(chain.levels[0].triangle_count, original_tris);
    }

    #[test]
    fn test_estimate_bounding_radius() {
        let mesh = UvSphere::new(1.0, 16, 8).apply();
        let radius = estimate_bounding_radius(&mesh);

        // UV sphere with default params should have radius ~1.0
        assert!(radius > 0.9 && radius < 1.1);
    }

    #[test]
    fn test_lod_chain_memory() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let chain = generate_lod_chain(&mesh, &LodConfig::default());

        let total_memory = chain.memory_estimate();

        // Should be less than num_levels * original mesh memory
        let original_memory = mesh.memory_estimate();
        assert!(total_memory < chain.len() * original_memory * 2);
    }

    #[test]
    fn test_empty_mesh_lod() {
        let mesh = Mesh::new();
        let chain = generate_lod_chain(&mesh, &LodConfig::default());

        assert!(chain.is_empty());
    }

    #[test]
    fn test_lod_accessors() {
        let mesh = UvSphere::new(1.0, 32, 16).apply();
        let chain = generate_lod_chain(&mesh, &LodConfig::default());

        assert!(chain.highest_detail().is_some());
        assert!(chain.lowest_detail().is_some());
        assert!(chain.get(0).is_some());
        assert!(chain.mesh(0).is_some());
    }
}
