//! Skinning (vertex-bone weights) for mesh deformation.
//!
//! Supports both linear blend skinning (LBS) and dual-quaternion skinning (DQS).
//! DQS eliminates the "candy-wrapper" collapse artifact that LBS produces on
//! twisting joints by blending dual quaternions instead of matrices.

use crate::skeleton::{BoneId, Pose, Skeleton};
use glam::{Mat4, Quat, Vec3};

/// Maximum bones per vertex (GPU-friendly limit).
pub const MAX_INFLUENCES: usize = 4;

/// Selects the skinning algorithm used when deforming a mesh.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SkinningMethod {
    /// Classic linear blend skinning (LBS). Fast but produces candy-wrapper
    /// collapse artifacts when joints twist more than ~90°.
    #[default]
    LinearBlend,
    /// Dual-quaternion skinning (DQS). Preserves volume on twisting joints at
    /// the cost of a small amount of extra computation.
    DualQuaternion,
    /// Weighted blend of LBS and DQS.
    ///
    /// `blend = 0.0` is pure LBS; `blend = 1.0` is pure DQS.
    BlendedDq {
        /// Interpolation factor between LBS (0) and DQS (1).
        blend: f32,
    },
}

// ---------------------------------------------------------------------------
// Dual-quaternion math
// ---------------------------------------------------------------------------

/// A dual quaternion representing a rigid body transform (rotation + translation).
///
/// A dual quaternion `q = real + ε·dual` where ε² = 0.  The real part is the
/// rotation quaternion; the dual part encodes the translation:
/// `dual = 0.5 * t * real` where `t` is the translation as a pure quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuat {
    /// Non-dual (rotation) part.
    pub real: Quat,
    /// Dual (translation) part.
    pub dual: Quat,
}

impl DualQuat {
    /// The identity dual quaternion (no rotation, no translation).
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };

    /// Constructs a dual quaternion from a rotation and translation.
    pub fn from_rotation_translation(rot: Quat, trans: Vec3) -> Self {
        // dual = 0.5 * pure_translation_quat * rot
        let t = Quat::from_xyzw(trans.x, trans.y, trans.z, 0.0);
        let dual = (t * rot) * 0.5;
        Self { real: rot, dual }
    }

    /// Constructs a dual quaternion from a 4×4 rigid-body matrix.
    ///
    /// Scale is ignored — only the rotation and translation components are used.
    pub fn from_matrix(mat: Mat4) -> Self {
        let (_, rot, trans) = mat.to_scale_rotation_translation();
        Self::from_rotation_translation(rot, trans)
    }

    /// Scales both parts by a scalar (used for weighted blending).
    #[must_use]
    pub fn scale(self, f: f32) -> Self {
        Self {
            real: Quat::from_xyzw(
                self.real.x * f,
                self.real.y * f,
                self.real.z * f,
                self.real.w * f,
            ),
            dual: Quat::from_xyzw(
                self.dual.x * f,
                self.dual.y * f,
                self.dual.z * f,
                self.dual.w * f,
            ),
        }
    }

    /// Adds two dual quaternions component-wise (for blending).
    #[must_use]
    pub fn add_dq(self, other: Self) -> Self {
        Self {
            real: Quat::from_xyzw(
                self.real.x + other.real.x,
                self.real.y + other.real.y,
                self.real.z + other.real.z,
                self.real.w + other.real.w,
            ),
            dual: Quat::from_xyzw(
                self.dual.x + other.dual.x,
                self.dual.y + other.dual.y,
                self.dual.z + other.dual.z,
                self.dual.w + other.dual.w,
            ),
        }
    }

    /// Normalizes the dual quaternion so that the real part has unit length,
    /// then rescales the dual part consistently.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = (self.real.x * self.real.x
            + self.real.y * self.real.y
            + self.real.z * self.real.z
            + self.real.w * self.real.w)
            .sqrt();

        if len < 1e-10 {
            return Self::IDENTITY;
        }

        let inv = 1.0 / len;
        Self {
            real: Quat::from_xyzw(
                self.real.x * inv,
                self.real.y * inv,
                self.real.z * inv,
                self.real.w * inv,
            ),
            dual: Quat::from_xyzw(
                self.dual.x * inv,
                self.dual.y * inv,
                self.dual.z * inv,
                self.dual.w * inv,
            ),
        }
    }

    /// Transforms a point by this dual quaternion.
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        // Extract rotation (real part as a proper normalized Quat for glam ops)
        let r = self.real;
        // Extract translation: t = 2 * dual * conjugate(real)
        // conjugate(real) = (-rx, -ry, -rz, rw)
        let rc = Quat::from_xyzw(-r.x, -r.y, -r.z, r.w);
        let d = self.dual;
        // dual_quat product (2 * d * rc) gives a pure quaternion whose xyz = translation
        let t2 = quat_mul_raw(
            (d.x * 2.0, d.y * 2.0, d.z * 2.0, d.w * 2.0),
            (rc.x, rc.y, rc.z, rc.w),
        );
        let translation = Vec3::new(t2.0, t2.1, t2.2);

        // Apply: rotate then translate
        r * p + translation
    }

    /// Transforms a direction vector (no translation applied).
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.real * v
    }
}

/// Raw quaternion multiply returning `(x, y, z, w)`.
#[inline]
fn quat_mul_raw(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> (f32, f32, f32, f32) {
    let (ax, ay, az, aw) = a;
    let (bx, by, bz, bw) = b;
    (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
}

// ---------------------------------------------------------------------------
// VertexInfluences
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Skin
// ---------------------------------------------------------------------------

/// Skinning data for a mesh (vertex-bone weights).
#[derive(Debug, Clone, Default)]
pub struct Skin {
    /// Per-vertex influences.
    influences: Vec<VertexInfluences>,
    /// Inverse bind matrices (one per bone).
    /// Transform3D from world space to bone space at bind time.
    inverse_bind_matrices: Vec<Mat4>,
    /// Skinning algorithm to use when deforming vertices.
    pub method: SkinningMethod,
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
            method: SkinningMethod::default(),
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

    /// Computes the skinning matrix for a bone (LBS path).
    pub fn bone_matrix(&self, skeleton: &Skeleton, pose: &Pose, bone: BoneId) -> Mat4 {
        let posed_world = pose.world_transform(skeleton, bone).to_matrix();
        let inv_bind = self
            .inverse_bind_matrices
            .get(bone.index())
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        posed_world * inv_bind
    }

    /// Computes the skinning dual quaternion for a bone (DQS path).
    pub fn bone_dual_quat(&self, skeleton: &Skeleton, pose: &Pose, bone: BoneId) -> DualQuat {
        DualQuat::from_matrix(self.bone_matrix(skeleton, pose, bone))
    }

    // -----------------------------------------------------------------------
    // LBS helpers
    // -----------------------------------------------------------------------

    fn deform_position_lbs(
        &self,
        bone_matrices: &[Mat4],
        influences: VertexInfluences,
        position: Vec3,
    ) -> Vec3 {
        let mut result = Vec3::ZERO;
        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight > 0.0 {
                let bone_idx = influences.bones[i].index();
                if let Some(matrix) = bone_matrices.get(bone_idx) {
                    result += weight * matrix.transform_point3(position);
                }
            }
        }
        result
    }

    fn deform_normal_lbs(
        &self,
        bone_matrices: &[Mat4],
        influences: VertexInfluences,
        normal: Vec3,
    ) -> Vec3 {
        let mut result = Vec3::ZERO;
        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight > 0.0 {
                let bone_idx = influences.bones[i].index();
                if let Some(matrix) = bone_matrices.get(bone_idx) {
                    result += weight * matrix.transform_vector3(normal);
                }
            }
        }
        result.normalize_or_zero()
    }

    // -----------------------------------------------------------------------
    // DQS helpers
    // -----------------------------------------------------------------------

    fn deform_position_dqs(
        &self,
        bone_dqs: &[DualQuat],
        influences: VertexInfluences,
        position: Vec3,
    ) -> Vec3 {
        // Antipodality fix: ensure all DQs are on the same hemisphere as the
        // first non-zero-weight DQ before summing.
        let mut first_real: Option<Quat> = None;
        let mut blended = DualQuat::IDENTITY.scale(0.0); // zero DQ for accumulation

        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight <= 0.0 {
                continue;
            }
            let bone_idx = influences.bones[i].index();
            let dq = match bone_dqs.get(bone_idx) {
                Some(&dq) => dq,
                None => continue,
            };

            // Antipodality: flip if dot(real, first_real) < 0
            let dq = if let Some(fr) = first_real {
                let dot = dq.real.x * fr.x + dq.real.y * fr.y + dq.real.z * fr.z + dq.real.w * fr.w;
                if dot < 0.0 { dq.scale(-1.0) } else { dq }
            } else {
                first_real = Some(dq.real);
                dq
            };

            blended = blended.add_dq(dq.scale(weight));
        }

        blended.normalize().transform_point(position)
    }

    fn deform_normal_dqs(
        &self,
        bone_dqs: &[DualQuat],
        influences: VertexInfluences,
        normal: Vec3,
    ) -> Vec3 {
        let mut first_real: Option<Quat> = None;
        let mut blended = DualQuat::IDENTITY.scale(0.0);

        for i in 0..MAX_INFLUENCES {
            let weight = influences.weights[i];
            if weight <= 0.0 {
                continue;
            }
            let bone_idx = influences.bones[i].index();
            let dq = match bone_dqs.get(bone_idx) {
                Some(&dq) => dq,
                None => continue,
            };

            let dq = if let Some(fr) = first_real {
                let dot = dq.real.x * fr.x + dq.real.y * fr.y + dq.real.z * fr.z + dq.real.w * fr.w;
                if dot < 0.0 { dq.scale(-1.0) } else { dq }
            } else {
                first_real = Some(dq.real);
                dq
            };

            blended = blended.add_dq(dq.scale(weight));
        }

        blended
            .normalize()
            .transform_vector(normal)
            .normalize_or_zero()
    }

    // -----------------------------------------------------------------------
    // Public deformation API
    // -----------------------------------------------------------------------

    /// Deforms a position using the skin (respects `self.method`).
    pub fn deform_position(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        vertex: usize,
        position: Vec3,
    ) -> Vec3 {
        let influences = self.influences(vertex);
        match self.method {
            SkinningMethod::LinearBlend => {
                let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
                    .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
                    .collect();
                self.deform_position_lbs(&bone_matrices, influences, position)
            }
            SkinningMethod::DualQuaternion => {
                let bone_dqs: Vec<DualQuat> = (0..skeleton.bone_count())
                    .map(|i| self.bone_dual_quat(skeleton, pose, BoneId(i as u32)))
                    .collect();
                self.deform_position_dqs(&bone_dqs, influences, position)
            }
            SkinningMethod::BlendedDq { blend } => {
                let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
                    .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
                    .collect();
                let bone_dqs: Vec<DualQuat> = bone_matrices
                    .iter()
                    .map(|m| DualQuat::from_matrix(*m))
                    .collect();
                let lbs = self.deform_position_lbs(&bone_matrices, influences, position);
                let dqs = self.deform_position_dqs(&bone_dqs, influences, position);
                lbs.lerp(dqs, blend)
            }
        }
    }

    /// Deforms a normal using the skin (respects `self.method`).
    pub fn deform_normal(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        vertex: usize,
        normal: Vec3,
    ) -> Vec3 {
        let influences = self.influences(vertex);
        match self.method {
            SkinningMethod::LinearBlend => {
                let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
                    .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
                    .collect();
                self.deform_normal_lbs(&bone_matrices, influences, normal)
            }
            SkinningMethod::DualQuaternion => {
                let bone_dqs: Vec<DualQuat> = (0..skeleton.bone_count())
                    .map(|i| self.bone_dual_quat(skeleton, pose, BoneId(i as u32)))
                    .collect();
                self.deform_normal_dqs(&bone_dqs, influences, normal)
            }
            SkinningMethod::BlendedDq { blend } => {
                let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
                    .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
                    .collect();
                let bone_dqs: Vec<DualQuat> = bone_matrices
                    .iter()
                    .map(|m| DualQuat::from_matrix(*m))
                    .collect();
                let lbs = self.deform_normal_lbs(&bone_matrices, influences, normal);
                let dqs = self.deform_normal_dqs(&bone_dqs, influences, normal);
                lbs.lerp(dqs, blend).normalize_or_zero()
            }
        }
    }

    /// Deforms an array of positions in place.
    pub fn deform_positions(&self, skeleton: &Skeleton, pose: &Pose, positions: &mut [Vec3]) {
        // Precompute bone matrices
        let bone_matrices: Vec<Mat4> = (0..skeleton.bone_count())
            .map(|i| self.bone_matrix(skeleton, pose, BoneId(i as u32)))
            .collect();

        match self.method {
            SkinningMethod::LinearBlend => {
                for (i, pos) in positions.iter_mut().enumerate() {
                    let influences = self.influences(i);
                    *pos = self.deform_position_lbs(&bone_matrices, influences, *pos);
                }
            }
            SkinningMethod::DualQuaternion => {
                let bone_dqs: Vec<DualQuat> = bone_matrices
                    .iter()
                    .map(|m| DualQuat::from_matrix(*m))
                    .collect();
                for (i, pos) in positions.iter_mut().enumerate() {
                    let influences = self.influences(i);
                    *pos = self.deform_position_dqs(&bone_dqs, influences, *pos);
                }
            }
            SkinningMethod::BlendedDq { blend } => {
                let bone_dqs: Vec<DualQuat> = bone_matrices
                    .iter()
                    .map(|m| DualQuat::from_matrix(*m))
                    .collect();
                for (i, pos) in positions.iter_mut().enumerate() {
                    let influences = self.influences(i);
                    let lbs = self.deform_position_lbs(&bone_matrices, influences, *pos);
                    let dqs = self.deform_position_dqs(&bone_dqs, influences, *pos);
                    *pos = lbs.lerp(dqs, blend);
                }
            }
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

    // -----------------------------------------------------------------------
    // Dual-quaternion skinning tests
    // -----------------------------------------------------------------------

    /// Classic candy-wrapper test: two root-level bones twist ±90° around Y in
    /// opposite directions and the vertex is influenced 50/50.
    ///
    /// Under LBS the opposing rotation matrices cancel and the X component
    /// collapses to zero (candy-wrapper).  Under DQS the rotations are blended
    /// in quaternion space and the result stays near `(1, 0, 0)`.
    #[test]
    fn test_dqs_avoids_candy_wrapper() {
        // Two independent root bones at the origin twisting opposite ways.
        let mut skel = Skeleton::new();
        let b0 = skel
            .add_bone(Bone {
                name: "bone_pos".into(),
                parent: None,
                local_transform: Transform3D::IDENTITY,
                length: 1.0,
            })
            .id;
        let b1 = skel
            .add_bone(Bone {
                name: "bone_neg".into(),
                parent: None,
                local_transform: Transform3D::IDENTITY,
                length: 1.0,
            })
            .id;

        let mut pose = skel.rest_pose();
        // b0: +90° around Y  |  b1: -90° around Y
        pose.set(
            b0,
            Transform3D::from_rotation(Quat::from_rotation_y(FRAC_PI_2)),
        );
        pose.set(
            b1,
            Transform3D::from_rotation(Quat::from_rotation_y(-FRAC_PI_2)),
        );

        let mut skin_lbs = Skin::with_vertex_count(1);
        skin_lbs.compute_bind_matrices(&skel);
        skin_lbs.set_influences(0, VertexInfluences::two(b0, 0.5, b1, 0.5));
        skin_lbs.method = SkinningMethod::LinearBlend;

        let mut skin_dqs = Skin::with_vertex_count(1);
        skin_dqs.compute_bind_matrices(&skel);
        skin_dqs.set_influences(0, VertexInfluences::two(b0, 0.5, b1, 0.5));
        skin_dqs.method = SkinningMethod::DualQuaternion;

        // +90° Y rotates X→Z; -90° Y rotates X→-Z.
        // LBS: 0.5*(0,0,1) + 0.5*(0,0,-1) = (0,0,0) — full candy-wrapper collapse.
        // DQS: blended quaternion ≈ identity, result ≈ (1,0,0) — no collapse.
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let lbs_result = skin_lbs.deform_position(&skel, &pose, 0, pos);
        let dqs_result = skin_dqs.deform_position(&skel, &pose, 0, pos);

        let lbs_dist = lbs_result.length();
        let dqs_dist = dqs_result.length();

        assert!(
            lbs_dist < 0.01,
            "LBS should collapse near zero for opposite twists; dist={lbs_dist}, pos={lbs_result:?}"
        );
        assert!(
            dqs_dist > 0.9,
            "DQS should preserve distance for opposite twists; dist={dqs_dist}, pos={dqs_result:?}"
        );
    }

    #[test]
    fn test_dqs_identity_pose_matches_lbs() {
        let (skel, upper, _lower) = arm_skeleton();
        let pose = skel.rest_pose();

        let mut skin_lbs = Skin::with_vertex_count(1);
        skin_lbs.compute_bind_matrices(&skel);
        skin_lbs.set_influences(0, VertexInfluences::single(upper));
        skin_lbs.method = SkinningMethod::LinearBlend;

        let mut skin_dqs = Skin::with_vertex_count(1);
        skin_dqs.compute_bind_matrices(&skel);
        skin_dqs.set_influences(0, VertexInfluences::single(upper));
        skin_dqs.method = SkinningMethod::DualQuaternion;

        let pos = Vec3::new(0.5, 1.0, 0.3);
        let lbs = skin_lbs.deform_position(&skel, &pose, 0, pos);
        let dqs = skin_dqs.deform_position(&skel, &pose, 0, pos);

        assert!(
            (lbs - dqs).length() < 0.001,
            "LBS and DQS should agree at rest pose; lbs={lbs:?} dqs={dqs:?}"
        );
    }

    #[test]
    fn test_blended_dq_limits() {
        let (skel, upper, _lower) = arm_skeleton();
        let mut pose = skel.rest_pose();
        pose.set(
            upper,
            Transform3D::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );

        let make_skin = |method: SkinningMethod| -> Skin {
            let mut s = Skin::with_vertex_count(1);
            s.compute_bind_matrices(&skel);
            s.set_influences(0, VertexInfluences::single(upper));
            s.method = method;
            s
        };

        let skin_lbs = make_skin(SkinningMethod::LinearBlend);
        let skin_dqs = make_skin(SkinningMethod::DualQuaternion);
        let skin_blend0 = make_skin(SkinningMethod::BlendedDq { blend: 0.0 });
        let skin_blend1 = make_skin(SkinningMethod::BlendedDq { blend: 1.0 });

        let pos = Vec3::new(0.0, 1.0, 0.0);
        let lbs = skin_lbs.deform_position(&skel, &pose, 0, pos);
        let dqs = skin_dqs.deform_position(&skel, &pose, 0, pos);
        let b0 = skin_blend0.deform_position(&skel, &pose, 0, pos);
        let b1 = skin_blend1.deform_position(&skel, &pose, 0, pos);

        assert!(
            (b0 - lbs).length() < 0.001,
            "BlendedDq(0) should match LBS; got {b0:?} vs lbs={lbs:?}"
        );
        assert!(
            (b1 - dqs).length() < 0.001,
            "BlendedDq(1) should match DQS; got {b1:?} vs dqs={dqs:?}"
        );
    }

    #[test]
    fn test_dual_quat_identity_roundtrip() {
        let dq = DualQuat::IDENTITY;
        let p = Vec3::new(1.0, 2.0, 3.0);
        let result = dq.transform_point(p);
        assert!(
            (result - p).length() < 0.0001,
            "Identity DQ should not move point"
        );
    }

    #[test]
    fn test_dual_quat_pure_translation() {
        let trans = Vec3::new(5.0, 0.0, 0.0);
        let dq = DualQuat::from_rotation_translation(Quat::IDENTITY, trans);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = dq.transform_point(p);
        assert!(
            (result - Vec3::new(6.0, 0.0, 0.0)).length() < 0.001,
            "Pure translation DQ failed: {result:?}"
        );
    }

    #[test]
    fn test_dual_quat_pure_rotation() {
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        let dq = DualQuat::from_rotation_translation(rot, Vec3::ZERO);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = dq.transform_point(p);
        assert!(
            (result - Vec3::new(0.0, 1.0, 0.0)).length() < 0.001,
            "Pure rotation DQ failed: {result:?}"
        );
    }
}
