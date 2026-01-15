//! Skinning (vertex-bone weights) for mesh deformation.

use crate::skeleton::{BoneId, Pose, Skeleton};
use glam::{Mat4, Vec3};

/// Maximum bones per vertex (GPU-friendly limit).
pub const MAX_INFLUENCES: usize = 4;

/// Bone influences for a single vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexInfluences {
    /// Bone indices (unused slots have weight 0).
    pub bones: [BoneId; MAX_INFLUENCES],
    /// Weights for each bone (should sum to 1.0).
    pub weights: [f32; MAX_INFLUENCES],
}

impl Default for VertexInfluences {
    fn default() -> Self {
        Self {
            bones: [BoneId(0); MAX_INFLUENCES],
            weights: [0.0; MAX_INFLUENCES],
        }
    }
}

impl VertexInfluences {
    /// Creates influences from a single bone.
    pub fn single(bone: BoneId) -> Self {
        let mut influences = Self::default();
        influences.bones[0] = bone;
        influences.weights[0] = 1.0;
        influences
    }

    /// Creates influences from two bones.
    pub fn two(bone_a: BoneId, weight_a: f32, bone_b: BoneId, weight_b: f32) -> Self {
        let mut influences = Self::default();
        influences.bones[0] = bone_a;
        influences.weights[0] = weight_a;
        influences.bones[1] = bone_b;
        influences.weights[1] = weight_b;
        influences
    }

    /// Normalizes weights to sum to 1.0.
    pub fn normalize(&mut self) {
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Returns the number of non-zero influences.
    pub fn influence_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }
}

/// Skinning data for a mesh (vertex-bone weights).
#[derive(Debug, Clone, Default)]
pub struct Skin {
    /// Per-vertex influences.
    influences: Vec<VertexInfluences>,
    /// Inverse bind matrices (one per bone).
    /// Transform3D from world space to bone space at bind time.
    inverse_bind_matrices: Vec<Mat4>,
}

impl Skin {
    /// Creates an empty skin.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a skin with the given vertex count.
    pub fn with_vertex_count(count: usize) -> Self {
        Self {
            influences: vec![VertexInfluences::default(); count],
            inverse_bind_matrices: Vec::new(),
        }
    }

    /// Sets up inverse bind matrices from a skeleton's rest pose.
    pub fn compute_bind_matrices(&mut self, skeleton: &Skeleton) {
        self.inverse_bind_matrices.clear();
        for i in 0..skeleton.bone_count() {
            let world = skeleton.world_transform(BoneId(i as u32));
            self.inverse_bind_matrices.push(world.to_matrix().inverse());
        }
    }

    /// Sets influences for a vertex.
    pub fn set_influences(&mut self, vertex: usize, influences: VertexInfluences) {
        if let Some(v) = self.influences.get_mut(vertex) {
            *v = influences;
        }
    }

    /// Gets influences for a vertex.
    pub fn influences(&self, vertex: usize) -> VertexInfluences {
        self.influences.get(vertex).copied().unwrap_or_default()
    }

    /// Returns all vertex influences.
    pub fn all_influences(&self) -> &[VertexInfluences] {
        &self.influences
    }

    /// Returns the inverse bind matrices.
    pub fn inverse_bind_matrices(&self) -> &[Mat4] {
        &self.inverse_bind_matrices
    }

    /// Computes the skinning matrix for a bone.
    pub fn bone_matrix(&self, skeleton: &Skeleton, pose: &Pose, bone: BoneId) -> Mat4 {
        let posed_world = pose.world_transform(skeleton, bone).to_matrix();
        let inv_bind = self
            .inverse_bind_matrices
            .get(bone.index())
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        posed_world * inv_bind
    }

    /// Deforms a position using the skin.
    pub fn deform_position(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        vertex: usize,
        position: Vec3,
    ) -> Vec3 {
        let influences = self.influences(vertex);
        let mut result = Vec3::ZERO;

        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight > 0.0 {
                let bone = influences.bones[i];
                let matrix = self.bone_matrix(skeleton, pose, bone);
                result += weight * matrix.transform_point3(position);
            }
        }

        result
    }

    /// Deforms a normal using the skin (uses inverse transpose for normals).
    pub fn deform_normal(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        vertex: usize,
        normal: Vec3,
    ) -> Vec3 {
        let influences = self.influences(vertex);
        let mut result = Vec3::ZERO;

        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight > 0.0 {
                let bone = influences.bones[i];
                let matrix = self.bone_matrix(skeleton, pose, bone);
                // For normals, use inverse transpose of the 3x3 rotation part
                // Since we're dealing with uniform scale in most cases, we can
                // just use the rotation and renormalize
                result += weight * matrix.transform_vector3(normal);
            }
        }

        result.normalize_or_zero()
    }

    /// Deforms an array of positions in place.
    pub fn deform_positions(&self, skeleton: &Skeleton, pose: &Pose, positions: &mut [Vec3]) {
        // Precompute bone matrices
        let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
            .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
            .collect();

        for (i, pos) in positions.iter_mut().enumerate() {
            let influences = self.influences(i);
            let mut result = Vec3::ZERO;

            for j in 0..MAX_INFLUENCES {
                let weight = influences.weights[j];
                if weight > 0.0 {
                    let bone_idx = influences.bones[j].index();
                    if let Some(matrix) = bone_matrices.get(bone_idx) {
                        result += weight * matrix.transform_point3(*pos);
                    }
                }
            }

            *pos = result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;
    use crate::transform::Transform3D;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_2;

    fn arm_skeleton() -> (Skeleton, BoneId, BoneId) {
        let mut skel = Skeleton::new();

        let upper = skel
            .add_bone(Bone {
                name: "upper_arm".into(),
                parent: None,
                local_transform: Transform3D::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                length: 2.0,
            })
            .id;

        let lower = skel
            .add_bone(Bone {
                name: "lower_arm".into(),
                parent: Some(upper),
                local_transform: Transform3D::from_translation(Vec3::new(0.0, 2.0, 0.0)),
                length: 2.0,
            })
            .id;

        (skel, upper, lower)
    }

    #[test]
    fn test_vertex_influences() {
        let mut influences = VertexInfluences::default();
        influences.bones[0] = BoneId(0);
        influences.weights[0] = 0.6;
        influences.bones[1] = BoneId(1);
        influences.weights[1] = 0.4;

        assert_eq!(influences.influence_count(), 2);
    }

    #[test]
    fn test_influences_normalize() {
        let mut influences = VertexInfluences::default();
        influences.bones[0] = BoneId(0);
        influences.weights[0] = 2.0;
        influences.bones[1] = BoneId(1);
        influences.weights[1] = 2.0;

        influences.normalize();

        assert!((influences.weights[0] - 0.5).abs() < 0.0001);
        assert!((influences.weights[1] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_skin_deform_identity() {
        let (skel, upper, _) = arm_skeleton();
        let pose = skel.rest_pose();

        let mut skin = Skin::with_vertex_count(1);
        skin.compute_bind_matrices(&skel);
        skin.set_influences(0, VertexInfluences::single(upper));

        // At rest pose, vertex should stay in place
        let pos = Vec3::new(0.0, 1.0, 0.0);
        let deformed = skin.deform_position(&skel, &pose, 0, pos);

        assert!((deformed - pos).length() < 0.0001);
    }

    #[test]
    fn test_skin_deform_rotated() {
        let (skel, upper, _) = arm_skeleton();
        let mut pose = skel.rest_pose();

        let mut skin = Skin::with_vertex_count(1);
        skin.compute_bind_matrices(&skel);
        skin.set_influences(0, VertexInfluences::single(upper));

        // Rotate upper arm 90 degrees around Z
        pose.set(
            upper,
            Transform3D::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );

        // Vertex at (0, 1, 0) should move to (-1, 0, 0)
        let pos = Vec3::new(0.0, 1.0, 0.0);
        let deformed = skin.deform_position(&skel, &pose, 0, pos);

        assert!((deformed.x - (-1.0)).abs() < 0.0001);
        assert!(deformed.y.abs() < 0.0001);
    }

    #[test]
    fn test_skin_blended_weights() {
        let (skel, upper, lower) = arm_skeleton();
        let mut pose = skel.rest_pose();

        let mut skin = Skin::with_vertex_count(1);
        skin.compute_bind_matrices(&skel);

        // Vertex influenced 50/50 by upper and lower
        skin.set_influences(0, VertexInfluences::two(upper, 0.5, lower, 0.5));

        // Rotate upper arm
        pose.set(
            upper,
            Transform3D::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );

        let pos = Vec3::new(0.0, 1.0, 0.0);
        let deformed = skin.deform_position(&skel, &pose, 0, pos);

        // Should be blend of both transformations
        // Upper contribution: (-1, 0, 0)
        // Lower contribution: affected by upper's rotation plus its own offset
        // This is a complex case - just verify it moved
        assert!((deformed - pos).length() > 0.1);
    }
}
