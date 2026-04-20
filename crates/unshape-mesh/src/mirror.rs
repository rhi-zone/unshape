//! Mirror mesh across an axis plane.
//!
//! Reflects a mesh across the XY, XZ, or YZ plane and merges it with the
//! original, welding seam vertices within a configurable threshold.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use glam::Vec3;

use crate::{Mesh, weld_vertices};

// ============================================================================
// Types
// ============================================================================

/// The axis plane to mirror across.
///
/// - `X` mirrors across the YZ plane (negates X coordinates).
/// - `Y` mirrors across the XZ plane (negates Y coordinates).
/// - `Z` mirrors across the XY plane (negates Z coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MirrorAxis {
    /// Mirror across the YZ plane (negate X).
    X,
    /// Mirror across the XZ plane (negate Y).
    Y,
    /// Mirror across the XY plane (negate Z).
    Z,
}

/// Mirrors a mesh across an axis plane, merging both halves at the seam.
///
/// The mirrored half is appended to the original, and vertices on or very near
/// the mirror plane are welded if they are within `merge_threshold`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mirror {
    /// The axis plane to mirror across.
    pub axis: MirrorAxis,
    /// Distance threshold for welding vertices at the mirror seam.
    /// Set to 0.0 to skip welding.
    pub merge_threshold: f32,
}

impl Default for Mirror {
    fn default() -> Self {
        Self {
            axis: MirrorAxis::X,
            merge_threshold: 1e-4,
        }
    }
}

impl Mirror {
    /// Creates a new mirror op for the given axis with default threshold.
    pub fn new(axis: MirrorAxis) -> Self {
        Self {
            axis,
            ..Default::default()
        }
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        mirror(mesh, self)
    }
}

// ============================================================================
// Implementation
// ============================================================================

/// Applies a [`Mirror`] op to a mesh.
pub fn mirror(mesh: &Mesh, op: &Mirror) -> Mesh {
    // Build the mirrored copy: reflect positions and flip normals.
    let mut mirrored = Mesh {
        positions: mesh
            .positions
            .iter()
            .map(|&p| reflect(p, op.axis))
            .collect(),
        normals: mesh.normals.iter().map(|&n| reflect(n, op.axis)).collect(),
        uvs: mesh.uvs.clone(),
        // Reverse triangle winding to keep outward-facing normals correct.
        indices: mesh
            .indices
            .chunks(3)
            .flat_map(|tri| [tri[0], tri[2], tri[1]])
            .collect(),
    };

    // Merge original + mirrored.
    let mut result = mesh.clone();
    result.merge(&mirrored);

    // Weld seam vertices if a threshold is provided.
    if op.merge_threshold > 0.0 {
        result = weld_vertices(&result, op.merge_threshold);
    }

    result
}

/// Reflects a vector across the given axis plane.
fn reflect(v: Vec3, axis: MirrorAxis) -> Vec3 {
    match axis {
        MirrorAxis::X => Vec3::new(-v.x, v.y, v.z),
        MirrorAxis::Y => Vec3::new(v.x, -v.y, v.z),
        MirrorAxis::Z => Vec3::new(v.x, v.y, -v.z),
    }
}

// ============================================================================
// Method sugar on Mesh
// ============================================================================

impl Mesh {
    /// Mirrors this mesh across the given axis plane, welding the seam.
    ///
    /// Sugar for `Mirror::new(axis).apply(self)`.
    pub fn mirror(&self, axis: MirrorAxis) -> Mesh {
        Mirror::new(axis).apply(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cuboid, Plane};

    #[test]
    fn test_mirror_doubles_geometry() {
        // A plane (open mesh with no symmetry) mirrored should roughly double triangles.
        let plane = Plane::new(1.0, 1.0, 1, 1).apply();
        let initial_tris = plane.triangle_count();

        let mirrored = Mirror {
            axis: MirrorAxis::Y,
            merge_threshold: 0.0, // no welding, just count
        }
        .apply(&plane);

        assert_eq!(
            mirrored.triangle_count(),
            initial_tris * 2,
            "mirroring should double triangle count before welding"
        );
    }

    #[test]
    fn test_mirror_axis_x_reflects_position() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![Vec3::new(1.0, 0.0, 0.0)];
        mesh.normals = vec![Vec3::X];
        mesh.uvs = vec![];
        mesh.indices = vec![];

        let mirrored_pos = reflect(mesh.positions[0], MirrorAxis::X);
        assert_eq!(mirrored_pos, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_mirror_winding_is_flipped() {
        // After mirroring, the mirrored half should have reversed winding.
        // A single triangle: indices [0, 1, 2] → mirrored becomes [0, 2, 1].
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.normals = vec![Vec3::Z; 3];
        mesh.uvs = vec![];
        mesh.indices = vec![0, 1, 2];

        let result = Mirror {
            axis: MirrorAxis::X,
            merge_threshold: 0.0,
        }
        .apply(&mesh);

        // Original triangle: indices 0,1,2
        assert_eq!(result.indices[0], 0);
        assert_eq!(result.indices[1], 1);
        assert_eq!(result.indices[2], 2);
        // Mirrored triangle: base offset=3, winding [0,2,1] → [3, 5, 4]
        assert_eq!(result.indices[3], 3);
        assert_eq!(result.indices[4], 5);
        assert_eq!(result.indices[5], 4);
    }

    #[test]
    fn test_mirror_method_sugar() {
        let mesh = Cuboid::default().apply();
        let _result = mesh.mirror(MirrorAxis::Z);
        // Just ensure it compiles and returns a mesh.
    }
}
