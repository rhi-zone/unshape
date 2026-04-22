//! Mesh-to-field bridge: query a mesh's per-vertex curvature as a spatial field.
//!
//! Requires the `mesh` feature.

use std::sync::Arc;

use glam::Vec3;
use unshape_mesh::{Mesh, compute_curvature};

use crate::{EvalContext, Field};

/// Which curvature quantity to expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureKind {
    /// Gaussian curvature (K): product of principal curvatures.
    Gaussian,
    /// Mean curvature (H): average of principal curvatures.
    Mean,
    /// Maximum principal curvature (k1).
    Max,
    /// Minimum principal curvature (k2).
    Min,
}

/// A `Field<Vec3, f32>` that returns precomputed per-vertex curvature values.
///
/// For a query point, the field finds the nearest mesh vertex (by Euclidean
/// distance) and returns that vertex's curvature value. Curvature values are
/// precomputed at construction time.
///
/// # Example
///
/// ```
/// use unshape_field::{MeshCurvatureField, CurvatureKind, Field, EvalContext};
/// use unshape_mesh::UvSphere;
/// use glam::Vec3;
///
/// let mesh = UvSphere::new(1.0, 16, 16).apply();
/// let field = MeshCurvatureField::new(mesh, CurvatureKind::Mean);
/// let ctx = EvalContext::new();
///
/// // Query near the top of the sphere
/// let h = field.sample(Vec3::new(0.0, 0.9, 0.0), &ctx);
/// assert!(h > 0.0, "sphere should have positive mean curvature");
/// ```
pub struct MeshCurvatureField {
    /// Vertex positions for nearest-vertex lookup.
    positions: Arc<Vec<Vec3>>,
    /// Precomputed curvature values, one per vertex.
    curvatures: Arc<Vec<f32>>,
    /// Which curvature kind is stored.
    pub kind: CurvatureKind,
}

impl MeshCurvatureField {
    /// Precomputes curvature for all vertices and stores the mesh's vertex positions.
    pub fn new(mesh: Mesh, kind: CurvatureKind) -> Self {
        let result = compute_curvature(&mesh);
        let curvatures = match kind {
            CurvatureKind::Gaussian => result.gaussian,
            CurvatureKind::Mean => result.mean,
            CurvatureKind::Max => result.k1,
            CurvatureKind::Min => result.k2,
        };
        Self {
            positions: Arc::new(mesh.positions),
            curvatures: Arc::new(curvatures),
            kind,
        }
    }
}

impl std::fmt::Debug for MeshCurvatureField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshCurvatureField")
            .field("vertices", &self.positions.len())
            .field("kind", &self.kind)
            .finish()
    }
}

impl Clone for MeshCurvatureField {
    fn clone(&self) -> Self {
        Self {
            positions: Arc::clone(&self.positions),
            curvatures: Arc::clone(&self.curvatures),
            kind: self.kind,
        }
    }
}

impl Field<Vec3, f32> for MeshCurvatureField {
    /// Returns the curvature at the nearest mesh vertex to `input`.
    ///
    /// If the mesh has no vertices, returns `0.0`.
    fn sample(&self, input: Vec3, _ctx: &EvalContext) -> f32 {
        if self.positions.is_empty() {
            return 0.0;
        }

        let nearest_idx = self
            .positions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (*a - input).length_squared();
                let db = (*b - input).length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.curvatures[nearest_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_mesh::UvSphere;

    #[test]
    fn test_sphere_mean_curvature_positive() {
        let mesh = UvSphere::new(1.0, 16, 16).apply();
        let field = MeshCurvatureField::new(mesh, CurvatureKind::Mean);
        let ctx = EvalContext::new();
        let h = field.sample(Vec3::new(0.0, 0.9, 0.0), &ctx);
        assert!(
            h > 0.0,
            "sphere should have positive mean curvature, got {h}"
        );
    }

    #[test]
    fn test_sphere_gaussian_curvature_positive() {
        let mesh = UvSphere::new(1.0, 16, 16).apply();
        let field = MeshCurvatureField::new(mesh, CurvatureKind::Gaussian);
        let ctx = EvalContext::new();
        let k = field.sample(Vec3::new(1.0, 0.0, 0.0), &ctx);
        assert!(
            k > 0.0,
            "sphere should have positive Gaussian curvature, got {k}"
        );
    }

    #[test]
    fn test_empty_mesh_returns_zero() {
        let field = MeshCurvatureField::new(Mesh::default(), CurvatureKind::Mean);
        let ctx = EvalContext::new();
        let result = field.sample(Vec3::ZERO, &ctx);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_clone_shares_data() {
        let mesh = UvSphere::new(1.0, 8, 8).apply();
        let field = MeshCurvatureField::new(mesh, CurvatureKind::Max);
        let cloned = field.clone();
        assert!(Arc::ptr_eq(&field.positions, &cloned.positions));
        assert!(Arc::ptr_eq(&field.curvatures, &cloned.curvatures));
    }
}
