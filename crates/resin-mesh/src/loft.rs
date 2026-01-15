//! Curve lofting - surface generation from profile curves.
//!
//! Creates meshes by interpolating between multiple cross-sectional profiles.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_mesh::{loft, Loft};
//! use glam::Vec3;
//!
//! // Create three circular profiles at different heights
//! let profiles = vec![
//!     circle_points(1.0, 0.0, 16),  // radius 1 at y=0
//!     circle_points(0.5, 1.0, 16),  // radius 0.5 at y=1
//!     circle_points(0.8, 2.0, 16),  // radius 0.8 at y=2
//! ];
//!
//! let mesh = Loft::default().apply(&profiles);
//! ```

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Lofts between profile curves to create a mesh surface.
///
/// Creates meshes by interpolating between multiple cross-sectional profiles.
/// All profiles should have the same number of points.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec<Vec3>>, output = Mesh))]
pub struct Loft {
    /// Whether to close the surface at the start profile.
    pub cap_start: bool,
    /// Whether to close the surface at the end profile.
    pub cap_end: bool,
    /// Number of interpolated profiles between each input profile (0 = no interpolation).
    pub interpolation_steps: usize,
    /// Whether the profiles should be closed loops.
    pub closed_profiles: bool,
}

impl Default for Loft {
    fn default() -> Self {
        Self {
            cap_start: false,
            cap_end: false,
            interpolation_steps: 0,
            closed_profiles: true,
        }
    }
}

impl Loft {
    /// Creates config with closed caps at both ends.
    pub fn with_caps() -> Self {
        Self {
            cap_start: true,
            cap_end: true,
            ..Default::default()
        }
    }

    /// Applies this loft operation to the given profiles.
    pub fn apply(&self, profiles: &[Vec<Vec3>]) -> Mesh {
        loft(profiles, self.clone())
    }
}

/// Backwards-compatible type alias.
pub type LoftConfig = Loft;

/// Creates a mesh by lofting between profile curves.
///
/// Each profile is a list of points forming a cross-section.
/// All profiles should have the same number of points.
pub fn loft(profiles: &[Vec<Vec3>], config: LoftConfig) -> Mesh {
    if profiles.len() < 2 {
        return Mesh::new();
    }

    // Ensure all profiles have the same point count
    let point_count = profiles[0].len();
    if point_count < 3 {
        return Mesh::new();
    }

    for profile in profiles {
        if profile.len() != point_count {
            // Profiles must have same number of points
            return Mesh::new();
        }
    }

    // Optionally interpolate between profiles
    let working_profiles = if config.interpolation_steps > 0 {
        interpolate_profiles(profiles, config.interpolation_steps)
    } else {
        profiles.to_vec()
    };

    let profile_count = working_profiles.len();

    // Build vertex list
    let mut positions: Vec<Vec3> = Vec::new();
    for profile in &working_profiles {
        positions.extend(profile);
    }

    // Build indices for the lofted surface
    let mut indices: Vec<u32> = Vec::new();

    for p in 0..profile_count - 1 {
        let base_current = (p * point_count) as u32;
        let base_next = ((p + 1) * point_count) as u32;

        let loop_end = if config.closed_profiles {
            point_count
        } else {
            point_count - 1
        };

        for i in 0..loop_end {
            let i_next = (i + 1) % point_count;

            let v0 = base_current + i as u32;
            let v1 = base_current + i_next as u32;
            let v2 = base_next + i_next as u32;
            let v3 = base_next + i as u32;

            // Create two triangles for each quad
            indices.push(v0);
            indices.push(v1);
            indices.push(v2);

            indices.push(v0);
            indices.push(v2);
            indices.push(v3);
        }
    }

    // Add caps if requested
    if config.cap_start && config.closed_profiles {
        add_cap(&mut indices, 0, point_count, true);
    }
    if config.cap_end && config.closed_profiles {
        let last_base = (profile_count - 1) * point_count;
        add_cap(&mut indices, last_base, point_count, false);
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.compute_smooth_normals();

    mesh
}

/// Creates a lofted mesh from a single profile extruded along a path.
///
/// The profile is placed at each point along the path, oriented to follow the path direction.
pub fn loft_along_path(profile: &[Vec3], path: &[Vec3], config: LoftConfig) -> Mesh {
    if path.len() < 2 || profile.len() < 3 {
        return Mesh::new();
    }

    // Generate profiles by transforming the base profile to each path point
    let profiles: Vec<Vec<Vec3>> = path
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            // Compute tangent direction
            let tangent = if i == 0 {
                (path[1] - path[0]).normalize_or_zero()
            } else if i == path.len() - 1 {
                (path[i] - path[i - 1]).normalize_or_zero()
            } else {
                ((path[i + 1] - path[i - 1]) * 0.5).normalize_or_zero()
            };

            // Build local coordinate frame
            let (right, up) = build_frame(tangent);

            // Transform profile points
            profile
                .iter()
                .map(|&p| pos + right * p.x + up * p.y + tangent * p.z)
                .collect()
        })
        .collect();

    loft(&profiles, config)
}

/// Builds an orthonormal frame from a forward direction.
fn build_frame(forward: Vec3) -> (Vec3, Vec3) {
    // Choose an up hint that's not parallel to forward
    let up_hint = if forward.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };

    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();

    (right, up)
}

/// Interpolates between profiles using linear interpolation.
fn interpolate_profiles(profiles: &[Vec<Vec3>], steps: usize) -> Vec<Vec<Vec3>> {
    let mut result = Vec::new();

    for i in 0..profiles.len() - 1 {
        let p0 = &profiles[i];
        let p1 = &profiles[i + 1];

        result.push(p0.clone());

        // Add interpolated profiles
        for s in 1..=steps {
            let t = s as f32 / (steps + 1) as f32;
            let interpolated: Vec<Vec3> = p0
                .iter()
                .zip(p1.iter())
                .map(|(&a, &b)| a.lerp(b, t))
                .collect();
            result.push(interpolated);
        }
    }

    // Add the last profile
    result.push(profiles.last().unwrap().clone());

    result
}

/// Adds a cap (fan triangulation) to close a profile.
fn add_cap(indices: &mut Vec<u32>, base: usize, point_count: usize, invert: bool) {
    // Create center point (we'll compute it from existing vertices)
    // For now, use fan triangulation from first vertex

    for i in 1..point_count - 1 {
        let v0 = base as u32;
        let v1 = (base + i) as u32;
        let v2 = (base + i + 1) as u32;

        if invert {
            indices.push(v0);
            indices.push(v2);
            indices.push(v1);
        } else {
            indices.push(v0);
            indices.push(v1);
            indices.push(v2);
        }
    }
}

/// Helper: creates a circular profile at a given height.
pub fn circle_profile(radius: f32, y: f32, segments: usize) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(segments);
    let step = std::f32::consts::TAU / segments as f32;

    for i in 0..segments {
        let angle = i as f32 * step;
        points.push(Vec3::new(radius * angle.cos(), y, radius * angle.sin()));
    }

    points
}

/// Helper: creates a rectangular profile at a given height.
pub fn rect_profile(width: f32, height: f32, y: f32) -> Vec<Vec3> {
    let hw = width / 2.0;
    let hh = height / 2.0;
    vec![
        Vec3::new(-hw, y, -hh),
        Vec3::new(hw, y, -hh),
        Vec3::new(hw, y, hh),
        Vec3::new(-hw, y, hh),
    ]
}

/// Helper: creates a star-shaped profile at a given height.
pub fn star_profile(outer_radius: f32, inner_radius: f32, y: f32, points: usize) -> Vec<Vec3> {
    let mut vertices = Vec::with_capacity(points * 2);
    let step = std::f32::consts::TAU / (points * 2) as f32;

    for i in 0..(points * 2) {
        let angle = i as f32 * step;
        let r = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        vertices.push(Vec3::new(r * angle.cos(), y, r * angle.sin()));
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loft_basic() {
        let profiles = vec![circle_profile(1.0, 0.0, 8), circle_profile(1.0, 1.0, 8)];

        let mesh = loft(&profiles, LoftConfig::default());

        // Should have 16 vertices (8 per profile)
        assert_eq!(mesh.positions.len(), 16);
        // Should have 16 quads = 32 triangles = 96 indices
        assert!(mesh.indices.len() > 0);
        assert!(mesh.has_normals());
    }

    #[test]
    fn test_loft_with_caps() {
        let profiles = vec![circle_profile(1.0, 0.0, 8), circle_profile(1.0, 1.0, 8)];

        let config = LoftConfig::with_caps();
        let mesh = loft(&profiles, config);

        // Should have more triangles due to caps
        assert!(mesh.triangle_count() > 16);
    }

    #[test]
    fn test_loft_varying_radius() {
        let profiles = vec![
            circle_profile(1.0, 0.0, 16),
            circle_profile(0.5, 0.5, 16),
            circle_profile(1.5, 1.0, 16),
        ];

        let mesh = loft(&profiles, LoftConfig::default());

        // Should have 48 vertices
        assert_eq!(mesh.positions.len(), 48);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_loft_with_interpolation() {
        let profiles = vec![circle_profile(1.0, 0.0, 8), circle_profile(0.5, 1.0, 8)];

        let mut config = LoftConfig::default();
        config.interpolation_steps = 2;
        let mesh = loft(&profiles, config);

        // Should have 4 profiles (original 2 + 2 interpolated)
        assert_eq!(mesh.positions.len(), 8 * 4);
    }

    #[test]
    fn test_loft_along_path() {
        let profile = vec![
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ];

        let path = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];

        let mesh = loft_along_path(&profile, &path, LoftConfig::default());

        assert_eq!(mesh.positions.len(), 12); // 4 points * 3 path positions
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_circle_profile() {
        let profile = circle_profile(1.0, 0.0, 16);

        assert_eq!(profile.len(), 16);
        // All points should be at y=0
        for p in &profile {
            assert!((p.y - 0.0).abs() < 0.001);
        }
        // All points should be at distance 1 from origin (in xz plane)
        for p in &profile {
            let dist = (p.x * p.x + p.z * p.z).sqrt();
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_rect_profile() {
        let profile = rect_profile(2.0, 1.0, 0.5);

        assert_eq!(profile.len(), 4);
        // All points should be at y=0.5
        for p in &profile {
            assert!((p.y - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_star_profile() {
        let profile = star_profile(1.0, 0.5, 0.0, 5);

        assert_eq!(profile.len(), 10); // 5 points * 2 (outer + inner)
    }

    #[test]
    fn test_loft_empty() {
        let profiles: Vec<Vec<Vec3>> = vec![];
        let mesh = loft(&profiles, LoftConfig::default());
        assert!(mesh.positions.is_empty());
    }

    #[test]
    fn test_loft_single_profile() {
        let profiles = vec![circle_profile(1.0, 0.0, 8)];
        let mesh = loft(&profiles, LoftConfig::default());
        assert!(mesh.positions.is_empty());
    }
}
