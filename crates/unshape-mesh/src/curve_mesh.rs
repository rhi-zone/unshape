//! Mesh generation from curves - extrude, revolve, and sweep operations.
//!
//! # Example
//!
//! ```ignore
//! use unshape_mesh::{extrude_profile, revolve_profile, sweep_profile};
//! use glam::{Vec2, Vec3};
//!
//! // Create a profile (2D outline)
//! let profile = vec![
//!     Vec2::new(0.0, 0.0),
//!     Vec2::new(1.0, 0.0),
//!     Vec2::new(1.0, 1.0),
//!     Vec2::new(0.0, 1.0),
//! ];
//!
//! // Extrude along Z axis
//! let extruded = extrude_profile(&profile, Vec3::Z * 2.0);
//!
//! // Revolve around Y axis
//! let revolved = revolve_profile(&profile, 32);
//!
//! // Sweep along a path
//! let path = vec![Vec3::ZERO, Vec3::Y, Vec3::new(1.0, 2.0, 0.0)];
//! let swept = sweep_profile(&profile, &path, 16);
//! ```

use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Extrudes a 2D profile along a direction to create a 3D mesh.
///
/// The profile is placed on the XY plane at the origin, then extruded along the given direction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct ExtrudeProfile {
    /// The 2D profile points to extrude.
    pub profile: Vec<Vec2>,
    /// Direction and distance to extrude.
    pub direction: Vec3,
    /// Whether to cap the start of the extrusion.
    pub cap_start: bool,
    /// Whether to cap the end of the extrusion.
    pub cap_end: bool,
    /// Number of segments along the extrusion direction.
    pub segments: usize,
}

impl Default for ExtrudeProfile {
    fn default() -> Self {
        Self {
            profile: Vec::new(),
            direction: Vec3::Z,
            cap_start: true,
            cap_end: true,
            segments: 1,
        }
    }
}

impl ExtrudeProfile {
    /// Creates a new extrude profile operation.
    pub fn new(profile: Vec<Vec2>, direction: Vec3) -> Self {
        Self {
            profile,
            direction,
            ..Default::default()
        }
    }

    /// Applies this operation to generate a mesh.
    pub fn apply(&self) -> Mesh {
        extrude_profile_impl(&self.profile, self.direction, self)
    }
}

/// Backwards-compatible type alias.
pub type ExtrudeProfileConfig = ExtrudeProfile;

/// Revolves a 2D profile around an axis to create a surface of revolution.
///
/// The profile should be in the XY plane with Y being the axis of revolution.
/// X values represent distance from the axis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Revolve {
    /// The 2D profile points to revolve.
    pub profile: Vec<Vec2>,
    /// Number of segments around the revolution.
    pub segments: usize,
    /// Axis to revolve around (default: Y axis).
    pub axis: Vec3,
    /// Angle to revolve in radians (default: full rotation TAU).
    pub angle: f32,
    /// Whether to close the revolve (only if angle < TAU).
    pub close: bool,
    /// Whether to cap ends if not a full rotation.
    pub cap_ends: bool,
}

impl Default for Revolve {
    fn default() -> Self {
        Self {
            profile: Vec::new(),
            segments: 32,
            axis: Vec3::Y,
            angle: std::f32::consts::TAU,
            close: true,
            cap_ends: true,
        }
    }
}

impl Revolve {
    /// Creates a new revolve operation.
    pub fn new(profile: Vec<Vec2>, segments: usize) -> Self {
        Self {
            profile,
            segments,
            ..Default::default()
        }
    }

    /// Applies this operation to generate a mesh.
    pub fn apply(&self) -> Mesh {
        revolve_profile_impl(&self.profile, self.segments, self)
    }
}

/// Backwards-compatible type alias.
pub type RevolveConfig = Revolve;

/// Extrudes a 2D profile along a direction to create a 3D mesh.
///
/// The profile is placed on the XY plane at the origin, then extruded along the given direction.
pub fn extrude_profile(profile: &[Vec2], direction: Vec3) -> Mesh {
    extrude_profile_impl(profile, direction, &ExtrudeProfile::default())
}

/// Extrudes a 2D profile with full configuration.
pub fn extrude_profile_with_config(
    profile: &[Vec2],
    direction: Vec3,
    config: ExtrudeProfileConfig,
) -> Mesh {
    extrude_profile_impl(profile, direction, &config)
}

/// Internal implementation for extrude profile.
fn extrude_profile_impl(profile: &[Vec2], direction: Vec3, config: &ExtrudeProfile) -> Mesh {
    if profile.len() < 3 {
        return Mesh::new();
    }

    let n = profile.len();
    let segments = config.segments.max(1);

    // Build vertices for all segments
    let mut positions: Vec<Vec3> = Vec::new();

    for s in 0..=segments {
        let t = s as f32 / segments as f32;
        let offset = direction * t;

        for p in profile {
            positions.push(Vec3::new(p.x, p.y, 0.0) + offset);
        }
    }

    // Build side faces
    let mut indices: Vec<u32> = Vec::new();

    for s in 0..segments {
        let base_current = (s * n) as u32;
        let base_next = ((s + 1) * n) as u32;

        for i in 0..n {
            let i_next = (i + 1) % n;

            let v0 = base_current + i as u32;
            let v1 = base_current + i_next as u32;
            let v2 = base_next + i_next as u32;
            let v3 = base_next + i as u32;

            // Two triangles per quad
            indices.push(v0);
            indices.push(v1);
            indices.push(v2);

            indices.push(v0);
            indices.push(v2);
            indices.push(v3);
        }
    }

    // Add caps
    if config.cap_start {
        add_cap_2d(&mut indices, 0, n, true);
    }
    if config.cap_end {
        let last_base = segments * n;
        add_cap_2d(&mut indices, last_base, n, false);
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.compute_smooth_normals();

    mesh
}

/// Revolves a 2D profile around an axis to create a surface of revolution.
///
/// The profile should be in the XY plane with Y being the axis of revolution.
/// X values represent distance from the axis.
pub fn revolve_profile(profile: &[Vec2], segments: usize) -> Mesh {
    revolve_profile_impl(profile, segments, &Revolve::default())
}

/// Revolves a 2D profile with full configuration.
pub fn revolve_profile_with_config(
    profile: &[Vec2],
    segments: usize,
    config: RevolveConfig,
) -> Mesh {
    revolve_profile_impl(profile, segments, &config)
}

/// Internal implementation for revolve profile.
fn revolve_profile_impl(profile: &[Vec2], segments: usize, config: &Revolve) -> Mesh {
    if profile.len() < 2 || segments < 3 {
        return Mesh::new();
    }

    let n = profile.len();
    let angle_step = config.angle / segments as f32;
    let is_full_rotation = (config.angle - std::f32::consts::TAU).abs() < 0.001;

    // Build axis rotation
    let axis = config.axis.normalize_or_zero();
    if axis.length() < 0.001 {
        return Mesh::new();
    }

    // Build vertices by rotating profile around axis
    let mut positions: Vec<Vec3> = Vec::new();

    let segment_count = if is_full_rotation {
        segments
    } else {
        segments + 1
    };

    for s in 0..segment_count {
        let angle = s as f32 * angle_step;
        let rotation = glam::Quat::from_axis_angle(axis, angle);

        for p in profile {
            // Profile point: x = distance from axis, y = height along axis
            let base_point = Vec3::new(p.x, 0.0, 0.0);
            let rotated = rotation * base_point;
            let point = rotated + axis * p.y;
            positions.push(point);
        }
    }

    // Build faces
    let mut indices: Vec<u32> = Vec::new();

    let loop_segments = if is_full_rotation { segments } else { segments };

    for s in 0..loop_segments {
        let base_current = (s * n) as u32;
        let base_next = if is_full_rotation && s == segments - 1 {
            0
        } else {
            ((s + 1) * n) as u32
        };

        for i in 0..n - 1 {
            let v0 = base_current + i as u32;
            let v1 = base_current + (i + 1) as u32;
            let v2 = base_next + (i + 1) as u32;
            let v3 = base_next + i as u32;

            indices.push(v0);
            indices.push(v1);
            indices.push(v2);

            indices.push(v0);
            indices.push(v2);
            indices.push(v3);
        }
    }

    // Add end caps for partial rotation
    if !is_full_rotation && config.cap_ends {
        // Start cap
        add_profile_cap(&mut indices, &positions, 0, n, true);
        // End cap
        let last_base = (segment_count - 1) * n;
        add_profile_cap(&mut indices, &positions, last_base, n, false);
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.compute_smooth_normals();

    mesh
}

/// Sweeps a 2D profile along a 3D path.
///
/// The profile is oriented perpendicular to the path at each point.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Sweep {
    /// The 2D profile points to sweep.
    pub profile: Vec<Vec2>,
    /// The 3D path to sweep along.
    pub path: Vec<Vec3>,
    /// Segments per unit length (currently unused, reserved for future interpolation).
    pub segments_per_unit: usize,
    /// Whether to cap the start.
    pub cap_start: bool,
    /// Whether to cap the end.
    pub cap_end: bool,
    /// Scale factor along the path (1.0 = uniform).
    pub scale_along_path: f32,
}

impl Default for Sweep {
    fn default() -> Self {
        Self {
            profile: Vec::new(),
            path: Vec::new(),
            segments_per_unit: 10,
            cap_start: true,
            cap_end: true,
            scale_along_path: 1.0,
        }
    }
}

impl Sweep {
    /// Creates a new sweep operation.
    pub fn new(profile: Vec<Vec2>, path: Vec<Vec3>) -> Self {
        Self {
            profile,
            path,
            ..Default::default()
        }
    }

    /// Applies this operation to generate a mesh.
    pub fn apply(&self) -> Mesh {
        sweep_profile_impl(&self.profile, &self.path, self.segments_per_unit, self)
    }
}

/// Backwards-compatible type alias.
pub type SweepConfig = Sweep;

/// Sweeps a 2D profile along a 3D path.
///
/// The profile is oriented perpendicular to the path at each point.
pub fn sweep_profile(profile: &[Vec2], path: &[Vec3], segments_per_unit: usize) -> Mesh {
    sweep_profile_impl(profile, path, segments_per_unit, &Sweep::default())
}

/// Sweeps a 2D profile along a path with configuration.
pub fn sweep_profile_with_config(
    profile: &[Vec2],
    path: &[Vec3],
    segments_per_unit: usize,
    config: SweepConfig,
) -> Mesh {
    sweep_profile_impl(profile, path, segments_per_unit, &config)
}

/// Internal implementation for sweep profile.
fn sweep_profile_impl(
    profile: &[Vec2],
    path: &[Vec3],
    _segments_per_unit: usize,
    config: &Sweep,
) -> Mesh {
    if profile.len() < 3 || path.len() < 2 {
        return Mesh::new();
    }

    let n = profile.len();

    // Build vertices by placing profile at each path point
    let mut positions: Vec<Vec3> = Vec::new();

    for (i, &pos) in path.iter().enumerate() {
        // Compute tangent
        let tangent = if i == 0 {
            (path[1] - path[0]).normalize_or_zero()
        } else if i == path.len() - 1 {
            (path[i] - path[i - 1]).normalize_or_zero()
        } else {
            ((path[i + 1] - path[i - 1]) * 0.5).normalize_or_zero()
        };

        // Build local frame
        let (right, up) = build_frame(tangent);

        // Compute scale along path
        let t = i as f32 / (path.len() - 1) as f32;
        let scale = 1.0 + (config.scale_along_path - 1.0) * t;

        // Transform profile points
        for p in profile {
            let point = pos + right * (p.x * scale) + up * (p.y * scale);
            positions.push(point);
        }
    }

    // Build side faces
    let mut indices: Vec<u32> = Vec::new();

    for s in 0..path.len() - 1 {
        let base_current = (s * n) as u32;
        let base_next = ((s + 1) * n) as u32;

        for i in 0..n {
            let i_next = (i + 1) % n;

            let v0 = base_current + i as u32;
            let v1 = base_current + i_next as u32;
            let v2 = base_next + i_next as u32;
            let v3 = base_next + i as u32;

            indices.push(v0);
            indices.push(v1);
            indices.push(v2);

            indices.push(v0);
            indices.push(v2);
            indices.push(v3);
        }
    }

    // Add caps
    if config.cap_start {
        add_cap_2d(&mut indices, 0, n, true);
    }
    if config.cap_end {
        let last_base = (path.len() - 1) * n;
        add_cap_2d(&mut indices, last_base, n, false);
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.compute_smooth_normals();

    mesh
}

/// Builds an orthonormal frame from a forward direction.
fn build_frame(forward: Vec3) -> (Vec3, Vec3) {
    let up_hint = if forward.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };

    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();

    (right, up)
}

/// Adds a cap using fan triangulation (for 2D profile on XY plane).
fn add_cap_2d(indices: &mut Vec<u32>, base: usize, point_count: usize, invert: bool) {
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

/// Adds a cap for revolve (handles 3D positions).
fn add_profile_cap(
    indices: &mut Vec<u32>,
    _positions: &[Vec3],
    base: usize,
    point_count: usize,
    invert: bool,
) {
    // Use fan triangulation
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

#[cfg(test)]
mod tests {
    use super::*;

    fn square_profile() -> Vec<Vec2> {
        vec![
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(-0.5, 0.5),
        ]
    }

    fn semicircle_profile(segments: usize) -> Vec<Vec2> {
        let mut points = Vec::new();
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let angle = t * std::f32::consts::PI;
            points.push(Vec2::new(angle.cos() * 0.5 + 0.5, angle.sin() * 0.5));
        }
        points
    }

    #[test]
    fn test_extrude_square() {
        let profile = square_profile();
        let mesh = extrude_profile(&profile, Vec3::Z * 2.0);

        // 4 vertices per layer, 2 layers = 8 vertices
        assert_eq!(mesh.positions.len(), 8);
        // 4 sides * 2 triangles + 2 caps * 2 triangles = 12 triangles
        assert!(mesh.triangle_count() >= 8);
        assert!(mesh.has_normals());
    }

    #[test]
    fn test_extrude_with_segments() {
        let profile = square_profile();
        let config = ExtrudeProfileConfig {
            segments: 3,
            ..Default::default()
        };
        let mesh = extrude_profile_with_config(&profile, Vec3::Z * 2.0, config);

        // 4 vertices per layer, 4 layers = 16 vertices
        assert_eq!(mesh.positions.len(), 16);
    }

    #[test]
    fn test_extrude_no_caps() {
        let profile = square_profile();
        let config = ExtrudeProfile {
            cap_start: false,
            cap_end: false,
            segments: 1,
            ..Default::default()
        };
        let mesh = extrude_profile_with_config(&profile, Vec3::Z * 2.0, config);

        // Only side faces: 4 sides * 2 triangles = 8 triangles
        assert_eq!(mesh.triangle_count(), 8);
    }

    #[test]
    fn test_revolve_semicircle() {
        let profile = semicircle_profile(8);
        let mesh = revolve_profile(&profile, 16);

        // Should create a sphere-like shape
        assert!(mesh.positions.len() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
    }

    #[test]
    fn test_revolve_partial() {
        let profile = vec![Vec2::new(0.5, 0.0), Vec2::new(0.5, 1.0)];

        let config = RevolveConfig {
            angle: std::f32::consts::PI, // Half rotation
            cap_ends: true,
            ..Default::default()
        };
        let mesh = revolve_profile_with_config(&profile, 16, config);

        assert!(mesh.positions.len() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_sweep_along_path() {
        let profile = square_profile();
        let path = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ];

        let mesh = sweep_profile(&profile, &path, 10);

        // 4 vertices per path point, 4 path points = 16 vertices
        assert_eq!(mesh.positions.len(), 16);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
    }

    #[test]
    fn test_sweep_with_scale() {
        let profile = square_profile();
        let path = vec![Vec3::ZERO, Vec3::Y, Vec3::Y * 2.0];

        let config = SweepConfig {
            scale_along_path: 0.5, // Taper to 50%
            ..Default::default()
        };
        let mesh = sweep_profile_with_config(&profile, &path, 10, config);

        assert_eq!(mesh.positions.len(), 12);
    }

    #[test]
    fn test_extrude_empty() {
        let profile: Vec<Vec2> = vec![];
        let mesh = extrude_profile(&profile, Vec3::Z);
        assert!(mesh.positions.is_empty());
    }

    #[test]
    fn test_revolve_empty() {
        let profile: Vec<Vec2> = vec![];
        let mesh = revolve_profile(&profile, 16);
        assert!(mesh.positions.is_empty());
    }

    #[test]
    fn test_sweep_empty() {
        let profile: Vec<Vec2> = vec![];
        let path = vec![Vec3::ZERO, Vec3::Y];
        let mesh = sweep_profile(&profile, &path, 10);
        assert!(mesh.positions.is_empty());
    }
}
