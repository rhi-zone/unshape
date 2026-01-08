//! UV projection and mapping operations.
//!
//! Provides various methods for generating texture coordinates on meshes.

use glam::{Mat4, Vec2, Vec3};

use crate::Mesh;

/// Projects UVs onto a mesh using planar projection.
///
/// Projects vertices onto the XY plane (by default) and uses the result as UVs.
/// The projection can be transformed using the provided matrix.
pub fn project_planar(mesh: &mut Mesh, transform: Mat4, scale: Vec2) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    for pos in &mesh.positions {
        // Transform position
        let p = transform.transform_point3(*pos);

        // Project onto XY plane
        let uv = Vec2::new(p.x * scale.x, p.y * scale.y);
        mesh.uvs.push(uv);
    }
}

/// Projects UVs using planar projection along a specific axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionAxis {
    /// Project along X axis (use YZ plane).
    X,
    /// Project along Y axis (use XZ plane).
    #[default]
    Y,
    /// Project along Z axis (use XY plane).
    Z,
}

/// Projects UVs along the specified axis.
pub fn project_planar_axis(mesh: &mut Mesh, axis: ProjectionAxis, scale: Vec2) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    for pos in &mesh.positions {
        let uv = match axis {
            ProjectionAxis::X => Vec2::new(pos.y * scale.x, pos.z * scale.y),
            ProjectionAxis::Y => Vec2::new(pos.x * scale.x, pos.z * scale.y),
            ProjectionAxis::Z => Vec2::new(pos.x * scale.x, pos.y * scale.y),
        };
        mesh.uvs.push(uv);
    }
}

/// Configuration for cylindrical UV projection.
#[derive(Debug, Clone)]
pub struct CylindricalConfig {
    /// Center of the cylinder.
    pub center: Vec3,
    /// Axis of the cylinder (default Y).
    pub axis: Vec3,
    /// UV scale.
    pub scale: Vec2,
    /// Whether to use the mesh bounds for V coordinate.
    pub use_bounds: bool,
}

impl Default for CylindricalConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            scale: Vec2::ONE,
            use_bounds: true,
        }
    }
}

/// Projects UVs using cylindrical projection.
///
/// Wraps UV coordinates around a cylinder aligned with the specified axis.
/// U = angle around axis, V = distance along axis.
pub fn project_cylindrical(mesh: &mut Mesh, config: &CylindricalConfig) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    let axis = config.axis.normalize();

    // Compute bounds along axis if needed
    let (min_v, max_v) = if config.use_bounds {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for pos in &mesh.positions {
            let v = (*pos - config.center).dot(axis);
            min = min.min(v);
            max = max.max(v);
        }
        (min, max)
    } else {
        (0.0, 1.0)
    };

    let v_range = (max_v - min_v).max(0.001);

    // Build orthonormal basis
    let (tangent, bitangent) = orthonormal_basis(axis);

    for pos in &mesh.positions {
        let local = *pos - config.center;

        // Project onto plane perpendicular to axis
        let x = local.dot(tangent);
        let y = local.dot(bitangent);

        // Angle around axis (0 to 1)
        let u = (y.atan2(x) / std::f32::consts::TAU + 0.5).fract();

        // Distance along axis (normalized to bounds)
        let v = if config.use_bounds {
            (local.dot(axis) - min_v) / v_range
        } else {
            local.dot(axis)
        };

        mesh.uvs
            .push(Vec2::new(u * config.scale.x, v * config.scale.y));
    }
}

/// Configuration for spherical UV projection.
#[derive(Debug, Clone)]
pub struct SphericalConfig {
    /// Center of the sphere.
    pub center: Vec3,
    /// Up axis for the sphere (default Y).
    pub up: Vec3,
    /// UV scale.
    pub scale: Vec2,
}

impl Default for SphericalConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            up: Vec3::Y,
            scale: Vec2::ONE,
        }
    }
}

/// Projects UVs using spherical projection.
///
/// Maps positions to UV coordinates on a sphere.
/// U = longitude (angle around up axis), V = latitude (angle from up axis).
pub fn project_spherical(mesh: &mut Mesh, config: &SphericalConfig) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    let up = config.up.normalize();
    let (tangent, bitangent) = orthonormal_basis(up);

    for pos in &mesh.positions {
        let dir = (*pos - config.center).normalize_or_zero();

        // Longitude (U): angle around up axis
        let x = dir.dot(tangent);
        let y = dir.dot(bitangent);
        let u = (y.atan2(x) / std::f32::consts::TAU + 0.5).fract();

        // Latitude (V): angle from up axis (0 at bottom, 1 at top)
        let v = (dir.dot(up) * 0.5 + 0.5).clamp(0.0, 1.0);

        mesh.uvs
            .push(Vec2::new(u * config.scale.x, v * config.scale.y));
    }
}

/// Configuration for box/triplanar UV projection.
#[derive(Debug, Clone)]
pub struct BoxConfig {
    /// Center of the projection.
    pub center: Vec3,
    /// UV scale.
    pub scale: Vec2,
    /// Blend sharpness for triplanar (higher = sharper transitions).
    pub blend_sharpness: f32,
}

impl Default for BoxConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            scale: Vec2::ONE,
            blend_sharpness: 1.0,
        }
    }
}

/// Projects UVs using box projection (per-face dominant axis).
///
/// Each triangle is projected along its dominant normal axis.
/// This creates clean projections for box-like or architectural geometry.
pub fn project_box(mesh: &mut Mesh, config: &BoxConfig) {
    // We need to split faces for proper box projection since
    // vertices shared between faces with different dominant axes
    // would have conflicting UVs.

    // For now, project based on vertex normal (approximate)
    // For proper box projection, use project_box_per_face

    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    // Ensure we have normals
    let normals: Vec<Vec3> = if mesh.normals.len() == mesh.positions.len() {
        mesh.normals.clone()
    } else {
        compute_vertex_normals(mesh)
    };

    for (pos, normal) in mesh.positions.iter().zip(normals.iter()) {
        let local = *pos - config.center;
        let abs_normal = normal.abs();

        // Choose projection based on dominant normal axis
        let uv = if abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z {
            // Project along X (use YZ)
            Vec2::new(local.y, local.z)
        } else if abs_normal.y >= abs_normal.z {
            // Project along Y (use XZ)
            Vec2::new(local.x, local.z)
        } else {
            // Project along Z (use XY)
            Vec2::new(local.x, local.y)
        };

        mesh.uvs.push(uv * config.scale);
    }
}

/// Projects UVs using box projection on a per-face basis.
///
/// This creates a new mesh where each face has its own vertices
/// with UVs projected along the face's dominant axis.
pub fn project_box_per_face(mesh: &Mesh, config: &BoxConfig) -> Mesh {
    let triangle_count = mesh.triangle_count();
    let mut result = Mesh::with_capacity(triangle_count * 3, triangle_count);

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Compute face normal
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let face_normal = edge1.cross(edge2).normalize_or_zero();
        let abs_normal = face_normal.abs();

        // Determine dominant axis
        let (u_axis, v_axis) = if abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z {
            (1, 2) // YZ plane
        } else if abs_normal.y >= abs_normal.z {
            (0, 2) // XZ plane
        } else {
            (0, 1) // XY plane
        };

        let base = result.positions.len() as u32;

        for pos in [p0, p1, p2] {
            let local = pos - config.center;
            let uv = Vec2::new(
                [local.x, local.y, local.z][u_axis],
                [local.x, local.y, local.z][v_axis],
            ) * config.scale;

            result.positions.push(pos);
            result.normals.push(face_normal);
            result.uvs.push(uv);
        }

        result.indices.push(base);
        result.indices.push(base + 1);
        result.indices.push(base + 2);
    }

    result
}

/// Scales existing UVs.
pub fn scale_uvs(mesh: &mut Mesh, scale: Vec2) {
    for uv in &mut mesh.uvs {
        *uv *= scale;
    }
}

/// Translates existing UVs.
pub fn translate_uvs(mesh: &mut Mesh, offset: Vec2) {
    for uv in &mut mesh.uvs {
        *uv += offset;
    }
}

/// Rotates existing UVs around a pivot point.
pub fn rotate_uvs(mesh: &mut Mesh, angle: f32, pivot: Vec2) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    for uv in &mut mesh.uvs {
        let local = *uv - pivot;
        let rotated = Vec2::new(
            local.x * cos_a - local.y * sin_a,
            local.x * sin_a + local.y * cos_a,
        );
        *uv = rotated + pivot;
    }
}

/// Transforms UVs by a 2D matrix.
pub fn transform_uvs(mesh: &mut Mesh, matrix: glam::Mat3) {
    for uv in &mut mesh.uvs {
        let v3 = matrix * glam::Vec3::new(uv.x, uv.y, 1.0);
        *uv = Vec2::new(v3.x, v3.y);
    }
}

/// Normalizes UVs to fit within [0, 1] range based on current bounds.
pub fn normalize_uvs(mesh: &mut Mesh) {
    if mesh.uvs.is_empty() {
        return;
    }

    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);

    for uv in &mesh.uvs {
        min = min.min(*uv);
        max = max.max(*uv);
    }

    let range = max - min;
    let scale = Vec2::new(
        if range.x > 0.001 { 1.0 / range.x } else { 1.0 },
        if range.y > 0.001 { 1.0 / range.y } else { 1.0 },
    );

    for uv in &mut mesh.uvs {
        *uv = (*uv - min) * scale;
    }
}

/// Flips UVs along the U axis.
pub fn flip_u(mesh: &mut Mesh) {
    for uv in &mut mesh.uvs {
        uv.x = 1.0 - uv.x;
    }
}

/// Flips UVs along the V axis.
pub fn flip_v(mesh: &mut Mesh) {
    for uv in &mut mesh.uvs {
        uv.y = 1.0 - uv.y;
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Computes smooth vertex normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let normal = (v1 - v0).cross(v2 - v0);
        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    normals
}

/// Builds an orthonormal basis from a single vector.
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;

    let tangent = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = Vec3::new(b, sign + n.y * n.y * a, -n.y);

    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_mesh;

    #[test]
    fn test_planar_projection() {
        let mut mesh = box_mesh();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_planar_axis_projection() {
        let mut mesh = box_mesh();
        project_planar_axis(&mut mesh, ProjectionAxis::Z, Vec2::ONE);

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_cylindrical_projection() {
        let mut mesh = box_mesh();
        project_cylindrical(&mut mesh, &CylindricalConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        // All UVs should be in valid range
        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0);
            assert!(uv.y >= 0.0 && uv.y <= 1.0);
        }
    }

    #[test]
    fn test_spherical_projection() {
        let mut mesh = box_mesh();
        project_spherical(&mut mesh, &SphericalConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        // All UVs should be in valid range
        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0);
            assert!(uv.y >= 0.0 && uv.y <= 1.0);
        }
    }

    #[test]
    fn test_box_projection() {
        let mut mesh = box_mesh();
        project_box(&mut mesh, &BoxConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_box_per_face_projection() {
        let mesh = box_mesh();
        let projected = project_box_per_face(&mesh, &BoxConfig::default());

        // Each triangle gets 3 unique vertices
        assert_eq!(projected.vertex_count(), mesh.triangle_count() * 3);
        assert_eq!(projected.uvs.len(), projected.positions.len());
    }

    #[test]
    fn test_scale_uvs() {
        let mut mesh = box_mesh();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        let original_uvs = mesh.uvs.clone();
        scale_uvs(&mut mesh, Vec2::new(2.0, 0.5));

        for (orig, scaled) in original_uvs.iter().zip(mesh.uvs.iter()) {
            assert!((scaled.x - orig.x * 2.0).abs() < 0.001);
            assert!((scaled.y - orig.y * 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_translate_uvs() {
        let mut mesh = box_mesh();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        let original_uvs = mesh.uvs.clone();
        translate_uvs(&mut mesh, Vec2::new(0.5, -0.25));

        for (orig, translated) in original_uvs.iter().zip(mesh.uvs.iter()) {
            assert!((translated.x - (orig.x + 0.5)).abs() < 0.001);
            assert!((translated.y - (orig.y - 0.25)).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_uvs() {
        let mut mesh = box_mesh();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        normalize_uvs(&mut mesh);

        // All UVs should be in [0, 1] range
        for uv in &mesh.uvs {
            assert!(uv.x >= -0.001 && uv.x <= 1.001);
            assert!(uv.y >= -0.001 && uv.y <= 1.001);
        }
    }

    #[test]
    fn test_rotate_uvs() {
        let mut mesh = box_mesh();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);
        normalize_uvs(&mut mesh);

        // Rotate 90 degrees around center
        rotate_uvs(&mut mesh, std::f32::consts::FRAC_PI_2, Vec2::splat(0.5));

        // UVs should still exist
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }
}
