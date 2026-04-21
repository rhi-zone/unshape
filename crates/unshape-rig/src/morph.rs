//! Morph target (blend shape) deformation.
//!
//! Morph targets store per-vertex position and normal deltas that are blended
//! by scalar weights to produce pose-space mesh deformations (e.g. facial
//! expressions, corrective shapes).
//!
//! # Usage
//! ```rust,ignore
//! let targets = vec![my_smile_target, my_frown_target];
//! let weights = vec![0.5, 0.0];
//! let op = ApplyMorphTargets { targets, weights };
//! let deformed_positions = op.apply(&rest_positions, &rest_normals);
//! ```

use glam::Vec3;

// ---------------------------------------------------------------------------
// MorphTarget
// ---------------------------------------------------------------------------

/// Per-vertex position and normal deltas for a single blend shape.
///
/// Uses a sparse representation: only the affected vertices are stored, with
/// `indices` mapping each entry to its vertex index in the mesh.
#[derive(Debug, Clone)]
pub struct MorphTarget {
    /// Human-readable name (e.g. `"smile"`, `"brow_up_L"`).
    pub name: String,
    /// Position offset for each affected vertex (world-space delta).
    pub position_deltas: Vec<Vec3>,
    /// Normal offset for each affected vertex (unnormalized delta, applied
    /// before renormalization).
    pub normal_deltas: Vec<Vec3>,
    /// Vertex indices corresponding to each entry in `position_deltas` /
    /// `normal_deltas`.  Must have the same length as `position_deltas`.
    pub indices: Vec<usize>,
}

impl MorphTarget {
    /// Creates a morph target that affects all vertices (dense).
    ///
    /// `position_deltas` and `normal_deltas` must have the same length.
    pub fn dense(
        name: impl Into<String>,
        position_deltas: Vec<Vec3>,
        normal_deltas: Vec<Vec3>,
    ) -> Self {
        let len = position_deltas.len();
        Self {
            name: name.into(),
            indices: (0..len).collect(),
            position_deltas,
            normal_deltas,
        }
    }

    /// Creates a morph target with explicit sparse indices.
    pub fn sparse(
        name: impl Into<String>,
        indices: Vec<usize>,
        position_deltas: Vec<Vec3>,
        normal_deltas: Vec<Vec3>,
    ) -> Self {
        Self {
            name: name.into(),
            indices,
            position_deltas,
            normal_deltas,
        }
    }
}

// ---------------------------------------------------------------------------
// DeformedMesh — output of ApplyMorphTargets
// ---------------------------------------------------------------------------

/// Result of applying morph targets to a mesh's positions and normals.
#[derive(Debug, Clone)]
pub struct DeformedMesh {
    /// Deformed vertex positions.
    pub positions: Vec<Vec3>,
    /// Deformed vertex normals (unit length).
    pub normals: Vec<Vec3>,
}

// ---------------------------------------------------------------------------
// ApplyMorphTargets — op struct
// ---------------------------------------------------------------------------

/// Op: blend a set of morph targets into mesh positions and normals.
///
/// Each target contributes `delta * weight[i]` to every affected vertex.
/// Normals are renormalized after all targets are applied.
///
/// # Ops-as-values
/// This type is the canonical representation of the operation.  The
/// convenience methods on mesh types delegate here.
#[derive(Debug, Clone)]
pub struct ApplyMorphTargets {
    /// Blend shapes to apply.
    pub targets: Vec<MorphTarget>,
    /// Per-target weight (same order as `targets`).  Values outside `[0, 1]`
    /// are allowed for over-drive / subtractive blending.
    pub weights: Vec<f32>,
}

impl ApplyMorphTargets {
    /// Creates the op from parallel slices.
    pub fn new(targets: Vec<MorphTarget>, weights: Vec<f32>) -> Self {
        Self { targets, weights }
    }

    /// Applies the morph targets to the given positions and normals.
    ///
    /// Returns a [`DeformedMesh`] with the blended positions and renormalized
    /// normals.  The input slices are not modified.
    ///
    /// # Panics
    /// Does not panic — out-of-bounds vertex indices in a `MorphTarget` are
    /// silently ignored.
    pub fn apply(&self, positions: &[Vec3], normals: &[Vec3]) -> DeformedMesh {
        let mut out_positions = positions.to_vec();
        let mut out_normals = normals.to_vec();

        for (target, &weight) in self.targets.iter().zip(self.weights.iter()) {
            if weight == 0.0 {
                continue;
            }
            for (entry_idx, &vertex_idx) in target.indices.iter().enumerate() {
                if vertex_idx >= out_positions.len() {
                    continue;
                }
                if let Some(&delta) = target.position_deltas.get(entry_idx) {
                    out_positions[vertex_idx] += delta * weight;
                }
                if let Some(&delta) = target.normal_deltas.get(entry_idx) {
                    out_normals[vertex_idx] += delta * weight;
                }
            }
        }

        // Renormalize normals
        for n in &mut out_normals {
            *n = n.normalize_or_zero();
        }

        DeformedMesh {
            positions: out_positions,
            normals: out_normals,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_half_weight_adds_half_delta() {
        // Position delta: move each vertex by Y=1
        let target = MorphTarget {
            name: "up".into(),
            indices: vec![0, 1, 2],
            position_deltas: vec![Vec3::Y, Vec3::Y, Vec3::Y],
            normal_deltas: vec![Vec3::ZERO; 3],
        };

        let rest = vec![Vec3::ZERO; 3];
        let normals = vec![Vec3::Z; 3];
        let op = ApplyMorphTargets::new(vec![target], vec![0.5]);
        let result = op.apply(&rest, &normals);

        for pos in &result.positions {
            assert!((pos.y - 0.5).abs() < 1e-6, "Expected y=0.5, got {pos:?}");
        }
    }

    #[test]
    fn test_morph_zero_weight_is_noop() {
        let target = MorphTarget {
            name: "noop".into(),
            indices: vec![0],
            position_deltas: vec![Vec3::splat(100.0)],
            normal_deltas: vec![Vec3::splat(100.0)],
        };

        let rest = vec![Vec3::new(1.0, 2.0, 3.0)];
        let normals = vec![Vec3::Y];
        let op = ApplyMorphTargets::new(vec![target], vec![0.0]);
        let result = op.apply(&rest, &normals);

        assert!((result.positions[0] - rest[0]).length() < 1e-6);
    }

    #[test]
    fn test_morph_normals_renormalized() {
        let target = MorphTarget {
            name: "tilt".into(),
            indices: vec![0],
            position_deltas: vec![Vec3::ZERO],
            // Large normal delta that would make length >> 1 if not renormalized
            normal_deltas: vec![Vec3::new(10.0, 0.0, 0.0)],
        };

        let rest = vec![Vec3::ZERO];
        let normals = vec![Vec3::Y];
        let op = ApplyMorphTargets::new(vec![target], vec![1.0]);
        let result = op.apply(&rest, &normals);

        let len = result.normals[0].length();
        assert!(
            (len - 1.0).abs() < 1e-6,
            "Normal not renormalized; length={len}"
        );
    }

    #[test]
    fn test_morph_out_of_bounds_index_ignored() {
        let target = MorphTarget {
            name: "oob".into(),
            indices: vec![999], // only 1 vertex in mesh
            position_deltas: vec![Vec3::ONE],
            normal_deltas: vec![Vec3::ONE],
        };

        let rest = vec![Vec3::ZERO];
        let normals = vec![Vec3::Y];
        let op = ApplyMorphTargets::new(vec![target], vec![1.0]);
        // Should not panic
        let result = op.apply(&rest, &normals);
        assert_eq!(result.positions[0], Vec3::ZERO);
    }

    #[test]
    fn test_morph_multiple_targets() {
        let t0 = MorphTarget {
            name: "x".into(),
            indices: vec![0],
            position_deltas: vec![Vec3::X],
            normal_deltas: vec![Vec3::ZERO],
        };
        let t1 = MorphTarget {
            name: "y".into(),
            indices: vec![0],
            position_deltas: vec![Vec3::Y],
            normal_deltas: vec![Vec3::ZERO],
        };

        let rest = vec![Vec3::ZERO];
        let normals = vec![Vec3::Z];
        let op = ApplyMorphTargets::new(vec![t0, t1], vec![1.0, 0.5]);
        let result = op.apply(&rest, &normals);

        let expected = Vec3::new(1.0, 0.5, 0.0);
        assert!(
            (result.positions[0] - expected).length() < 1e-6,
            "Expected {expected:?}, got {:?}",
            result.positions[0]
        );
    }
}
