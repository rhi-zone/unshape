//! UV projection and transform operations as serializable Op structs.
//!
//! Each struct wraps a free function from [`crate::uv`] and can be serialized,
//! stored in a pipeline, or used as a graph node.
//!
//! # Example
//!
//! ```ignore
//! let mesh = ProjectBox::default().apply(&cuboid);
//! let mesh = ProjectSphere::default().apply(&cuboid);
//! let mesh = PackUVCharts::default().apply(&mesh);
//! ```

use glam::{Mat4, Vec2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    AtlasPackConfig, BoxConfig, CylindricalConfig, Mesh, ProjectionAxis, SphericalConfig, flip_u,
    flip_v, normalize_uvs, pack_mesh_uvs, project_box, project_box_per_face, project_cylindrical,
    project_planar, project_planar_axis, project_spherical, rotate_uvs, scale_uvs, translate_uvs,
};

// ============================================================================
// Projection ops
// ============================================================================

/// Projects UVs using planar projection.
///
/// Vertices are projected onto the XY plane (by default) and the result is
/// used as UV coordinates. The projection can be transformed via `transform`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectPlanar {
    /// Transform applied to positions before projection onto XY.
    pub transform: Mat4,
    /// Scale applied to projected UVs.
    pub scale: Vec2,
}

impl Default for ProjectPlanar {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            scale: Vec2::ONE,
        }
    }
}

impl ProjectPlanar {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        project_planar(&mut result, self.transform, self.scale);
        result
    }
}

/// Projects UVs using planar projection along a fixed world axis.
///
/// Equivalent to [`ProjectPlanar`] with a pre-built transform, but simpler
/// to express for axis-aligned projections.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectPlanarAxis {
    /// The axis to project along.
    pub axis: ProjectionAxis,
    /// UV scale.
    pub scale: Vec2,
}

impl Default for ProjectPlanarAxis {
    fn default() -> Self {
        Self {
            axis: ProjectionAxis::Y,
            scale: Vec2::ONE,
        }
    }
}

impl ProjectPlanarAxis {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        project_planar_axis(&mut result, self.axis, self.scale);
        result
    }
}

/// Projects UVs using cylindrical projection.
///
/// Wraps UV coordinates around a cylinder. U = angle around axis, V =
/// distance along axis.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectCylinder {
    /// Cylinder configuration (center, axis, scale, use_bounds).
    pub config: CylindricalConfig,
}

impl ProjectCylinder {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        project_cylindrical(&mut result, &self.config);
        result
    }
}

/// Projects UVs using spherical projection.
///
/// Maps positions to UV coordinates on a sphere. U = longitude (angle around
/// up axis), V = latitude (angle from up axis).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectSphere {
    /// Sphere configuration (center, up axis, scale).
    pub config: SphericalConfig,
}

impl ProjectSphere {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        project_spherical(&mut result, &self.config);
        result
    }
}

/// Projects UVs using box/triplanar projection (per-vertex dominant axis).
///
/// Each vertex is projected along its dominant normal axis. This creates clean
/// projections for box-like or architectural geometry.
///
/// For a per-face variant that avoids seams at axis boundaries,
/// use [`ProjectBoxPerFace`].
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectBox {
    /// Box projection configuration (center, scale, blend sharpness).
    pub config: BoxConfig,
}

impl ProjectBox {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        project_box(&mut result, &self.config);
        result
    }
}

/// Projects UVs using box projection on a per-face basis.
///
/// Creates a new mesh where each face has its own vertices with UVs projected
/// along the face's dominant axis. Avoids seams but increases vertex count.
///
/// Use [`ProjectBox`] when vertex sharing is important.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProjectBoxPerFace {
    /// Box projection configuration (center, scale, blend sharpness).
    pub config: BoxConfig,
}

impl ProjectBoxPerFace {
    /// Applies this operation to a mesh, returning a new mesh with UVs set.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        project_box_per_face(mesh, &self.config)
    }
}

// ============================================================================
// Atlas packing op
// ============================================================================

/// Packs a mesh's UV islands into an atlas.
///
/// Finds UV islands (connected components), packs them into the [0,1]²
/// space using the maxrects algorithm, and applies the result to the mesh.
///
/// The mesh must already have UVs (e.g. from one of the `Project*` ops).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PackUVCharts {
    /// Atlas packing configuration (padding, rotation, target aspect).
    pub config: AtlasPackConfig,
}

impl PackUVCharts {
    /// Applies this operation to a mesh, returning a new mesh with packed UVs.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        pack_mesh_uvs(&mut result, &self.config);
        result
    }
}

// ============================================================================
// UV transform ops
// ============================================================================

/// Scales existing UV coordinates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScaleUVs {
    /// Scale factor in UV space.
    pub scale: Vec2,
}

impl Default for ScaleUVs {
    fn default() -> Self {
        Self { scale: Vec2::ONE }
    }
}

impl ScaleUVs {
    /// Applies this operation to a mesh, returning a new mesh with scaled UVs.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        scale_uvs(&mut result, self.scale);
        result
    }
}

/// Translates existing UV coordinates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TranslateUVs {
    /// Translation offset in UV space.
    pub offset: Vec2,
}

impl Default for TranslateUVs {
    fn default() -> Self {
        Self { offset: Vec2::ZERO }
    }
}

impl TranslateUVs {
    /// Applies this operation to a mesh, returning a new mesh with translated UVs.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        translate_uvs(&mut result, self.offset);
        result
    }
}

/// Rotates existing UV coordinates around a pivot point.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RotateUVs {
    /// Rotation angle in radians.
    pub angle: f32,
    /// Pivot point for rotation.
    pub pivot: Vec2,
}

impl Default for RotateUVs {
    fn default() -> Self {
        Self {
            angle: 0.0,
            pivot: Vec2::splat(0.5),
        }
    }
}

impl RotateUVs {
    /// Applies this operation to a mesh, returning a new mesh with rotated UVs.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        rotate_uvs(&mut result, self.angle, self.pivot);
        result
    }
}

/// Normalizes UV coordinates to fit within [0, 1] based on current bounds.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NormalizeUVs;

impl NormalizeUVs {
    /// Applies this operation to a mesh, returning a new mesh with normalized UVs.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        normalize_uvs(&mut result);
        result
    }
}

/// Flips UV coordinates along the U axis (mirrors horizontally).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlipU;

impl FlipU {
    /// Applies this operation to a mesh, returning a new mesh with flipped U coordinates.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        flip_u(&mut result);
        result
    }
}

/// Flips UV coordinates along the V axis (mirrors vertically).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlipV;

impl FlipV {
    /// Applies this operation to a mesh, returning a new mesh with flipped V coordinates.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        let mut result = mesh.clone();
        flip_v(&mut result);
        result
    }
}

// ============================================================================
// Mesh sugar methods
// ============================================================================

impl Mesh {
    /// Projects UVs using planar projection onto the XY plane.
    pub fn project_planar(&self, transform: Mat4, scale: Vec2) -> Mesh {
        ProjectPlanar { transform, scale }.apply(self)
    }

    /// Projects UVs using planar projection along a fixed world axis.
    pub fn project_planar_axis(&self, axis: ProjectionAxis, scale: Vec2) -> Mesh {
        ProjectPlanarAxis { axis, scale }.apply(self)
    }

    /// Projects UVs using cylindrical projection.
    pub fn project_cylindrical(&self, config: CylindricalConfig) -> Mesh {
        ProjectCylinder { config }.apply(self)
    }

    /// Projects UVs using spherical projection.
    pub fn project_spherical(&self, config: SphericalConfig) -> Mesh {
        ProjectSphere { config }.apply(self)
    }

    /// Projects UVs using box/triplanar projection (per-vertex dominant axis).
    pub fn project_box(&self, config: BoxConfig) -> Mesh {
        ProjectBox { config }.apply(self)
    }

    /// Projects UVs using box projection on a per-face basis.
    pub fn project_box_per_face(&self, config: BoxConfig) -> Mesh {
        ProjectBoxPerFace { config }.apply(self)
    }

    /// Packs UV islands into an atlas.
    pub fn pack_uv_charts(&self, config: AtlasPackConfig) -> Mesh {
        PackUVCharts { config }.apply(self)
    }

    /// Scales existing UV coordinates.
    pub fn scale_uvs(&self, scale: Vec2) -> Mesh {
        ScaleUVs { scale }.apply(self)
    }

    /// Translates existing UV coordinates.
    pub fn translate_uvs(&self, offset: Vec2) -> Mesh {
        TranslateUVs { offset }.apply(self)
    }

    /// Rotates existing UV coordinates around a pivot point.
    pub fn rotate_uvs(&self, angle: f32, pivot: Vec2) -> Mesh {
        RotateUVs { angle, pivot }.apply(self)
    }

    /// Normalizes UV coordinates to fit within [0, 1].
    pub fn normalize_uvs(&self) -> Mesh {
        NormalizeUVs.apply(self)
    }

    /// Flips UV coordinates along the U axis (mirrors horizontally).
    pub fn flip_u(&self) -> Mesh {
        FlipU.apply(self)
    }

    /// Flips UV coordinates along the V axis (mirrors vertically).
    pub fn flip_v(&self) -> Mesh {
        FlipV.apply(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    fn make_cube() -> Mesh {
        Cuboid::default().apply()
    }

    #[test]
    fn test_project_planar_op() {
        let mesh = make_cube();
        let result = ProjectPlanar::default().apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
        assert!(!result.uvs.is_empty());
    }

    #[test]
    fn test_project_planar_axis_op() {
        let mesh = make_cube();
        let result = ProjectPlanarAxis {
            axis: ProjectionAxis::Z,
            scale: Vec2::ONE,
        }
        .apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
    }

    #[test]
    fn test_project_cylinder_op() {
        let mesh = make_cube();
        let result = ProjectCylinder::default().apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
        // Cylindrical UVs are always in [0, 1]
        for uv in &result.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0, "U out of range: {}", uv.x);
            assert!(uv.y >= 0.0 && uv.y <= 1.0, "V out of range: {}", uv.y);
        }
    }

    #[test]
    fn test_project_sphere_op() {
        let mesh = make_cube();
        let result = ProjectSphere::default().apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
    }

    #[test]
    fn test_project_box_op() {
        let mesh = make_cube();
        let original_uvs = mesh.uvs.clone();
        let result = ProjectBox::default().apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
        // UVs should differ from original (which were empty)
        assert_ne!(result.uvs, original_uvs);
    }

    #[test]
    fn test_project_box_per_face_op() {
        let mesh = make_cube();
        let result = ProjectBoxPerFace::default().apply(&mesh);
        // Each triangle gets 3 unique vertices
        assert_eq!(result.vertex_count(), mesh.triangle_count() * 3);
        assert_eq!(result.uvs.len(), result.positions.len());
    }

    #[test]
    fn test_pack_uv_charts_op() {
        let mesh = ProjectBox::default().apply(&make_cube());
        let result = PackUVCharts::default().apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
        // After packing, UVs should still be valid
        assert!(!result.uvs.is_empty());
    }

    #[test]
    fn test_scale_uvs_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube());
        let original_uvs = mesh.uvs.clone();
        let result = ScaleUVs {
            scale: Vec2::new(2.0, 0.5),
        }
        .apply(&mesh);
        assert_eq!(result.uvs.len(), result.positions.len());
        // At least some UVs should be different
        let changed = original_uvs
            .iter()
            .zip(result.uvs.iter())
            .any(|(a, b)| (*a - *b).length() > 0.001);
        assert!(changed, "ScaleUVs should change UV values");
    }

    #[test]
    fn test_translate_uvs_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube());
        let original_uvs = mesh.uvs.clone();
        let result = TranslateUVs {
            offset: Vec2::new(0.5, -0.25),
        }
        .apply(&mesh);
        let changed = original_uvs
            .iter()
            .zip(result.uvs.iter())
            .any(|(a, b)| (*a - *b).length() > 0.001);
        assert!(changed, "TranslateUVs should change UV values");
    }

    #[test]
    fn test_rotate_uvs_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube()).normalize_uvs();
        let original_uvs = mesh.uvs.clone();
        let result = RotateUVs {
            angle: std::f32::consts::FRAC_PI_2,
            pivot: Vec2::splat(0.5),
        }
        .apply(&mesh);
        let changed = original_uvs
            .iter()
            .zip(result.uvs.iter())
            .any(|(a, b)| (*a - *b).length() > 0.001);
        assert!(changed, "RotateUVs should change UV values");
    }

    #[test]
    fn test_normalize_uvs_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube());
        let result = NormalizeUVs.apply(&mesh);
        // After normalization, all UVs should be in [0, 1]
        for uv in &result.uvs {
            assert!(uv.x >= -0.001 && uv.x <= 1.001, "U out of [0,1]: {}", uv.x);
            assert!(uv.y >= -0.001 && uv.y <= 1.001, "V out of [0,1]: {}", uv.y);
        }
    }

    #[test]
    fn test_flip_u_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube()).normalize_uvs();
        let original_uvs = mesh.uvs.clone();
        let result = FlipU.apply(&mesh);
        let changed = original_uvs
            .iter()
            .zip(result.uvs.iter())
            .any(|(a, b)| (a.x - b.x).abs() > 0.001);
        assert!(changed, "FlipU should change U values");
    }

    #[test]
    fn test_flip_v_op() {
        let mesh = ProjectPlanar::default().apply(&make_cube()).normalize_uvs();
        let original_uvs = mesh.uvs.clone();
        let result = FlipV.apply(&mesh);
        let changed = original_uvs
            .iter()
            .zip(result.uvs.iter())
            .any(|(a, b)| (a.y - b.y).abs() > 0.001);
        assert!(changed, "FlipV should change V values");
    }

    #[test]
    fn test_mesh_sugar_project_planar() {
        let mesh = make_cube();
        let result = mesh.project_planar(Mat4::IDENTITY, Vec2::ONE);
        assert_eq!(result.uvs.len(), result.positions.len());
    }

    #[test]
    fn test_mesh_sugar_project_box() {
        let mesh = make_cube();
        let result = mesh.project_box(BoxConfig::default());
        assert_eq!(result.uvs.len(), result.positions.len());
    }

    #[test]
    fn test_mesh_sugar_pack_uv_charts() {
        let mesh = make_cube().project_box(BoxConfig::default());
        let result = mesh.pack_uv_charts(AtlasPackConfig::default());
        assert_eq!(result.uvs.len(), result.positions.len());
    }
}
