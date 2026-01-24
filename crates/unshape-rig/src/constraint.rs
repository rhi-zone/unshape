//! Constraint evaluation system for skeletal animation.
//!
//! Constraints modify a pose after it has been computed, applying
//! effects like path following, IK, look-at, etc.

use crate::{BoneId, Path3D, Path3DExt, Pose, Skeleton, Transform3D};
use glam::{Quat, Vec3};

/// A constraint that modifies bone transforms.
pub trait Constraint: Send + Sync {
    /// Applies the constraint to the pose.
    fn apply(&self, skeleton: &Skeleton, pose: &mut Pose);

    /// Returns the bones affected by this constraint.
    fn affected_bones(&self) -> &[BoneId];
}

/// A path constraint that positions a bone along a 3D path.
#[derive(Debug, Clone)]
pub struct PathConstraint {
    /// The bone to constrain.
    pub bone: BoneId,
    /// The path to follow.
    pub path: Path3D,
    /// Position along the path (0.0 to 1.0).
    pub offset: f32,
    /// Whether to orient the bone to the path tangent.
    pub follow_tangent: bool,
    /// The local axis that points along the bone (default: Y+).
    pub forward_axis: Vec3,
    /// The local axis that points "up" for banking (default: Z+).
    pub up_axis: Vec3,
}

impl PathConstraint {
    /// Creates a new path constraint.
    pub fn new(bone: BoneId, path: Path3D) -> Self {
        Self {
            bone,
            path,
            offset: 0.0,
            follow_tangent: true,
            forward_axis: Vec3::Y,
            up_axis: Vec3::Z,
        }
    }

    /// Sets the forward axis.
    pub fn with_forward_axis(mut self, axis: Vec3) -> Self {
        self.forward_axis = axis.normalize_or_zero();
        self
    }

    /// Sets the up axis.
    pub fn with_up_axis(mut self, axis: Vec3) -> Self {
        self.up_axis = axis.normalize_or_zero();
        self
    }
}

impl Constraint for PathConstraint {
    fn apply(&self, skeleton: &Skeleton, pose: &mut Pose) {
        let sample = self.path.sample_at(self.offset);

        // Get parent world transform to convert to local space
        let parent_world = skeleton
            .bone(self.bone)
            .and_then(|b| b.parent)
            .map(|p| pose.world_transform(skeleton, p))
            .unwrap_or(Transform3D::IDENTITY);

        let parent_inv = parent_world.inverse();

        // Position in local space
        let local_pos = parent_inv.transform_point(sample.position);

        // Rotation
        let rotation = if self.follow_tangent {
            // Transform3D tangent to local space
            let local_tangent = parent_inv.transform_vector(sample.tangent);

            // Compute rotation from forward axis to tangent
            rotation_from_to_with_up(self.forward_axis, local_tangent, self.up_axis)
        } else {
            Quat::IDENTITY
        };

        let transform = Transform3D {
            translation: local_pos,
            rotation,
            scale: Vec3::ONE,
        };

        pose.set(self.bone, transform);
    }

    fn affected_bones(&self) -> &[BoneId] {
        std::slice::from_ref(&self.bone)
    }
}

/// Computes a rotation that aligns `from` to `to` with an up hint.
fn rotation_from_to_with_up(from: Vec3, to: Vec3, up: Vec3) -> Quat {
    let to = to.normalize_or_zero();
    if to.length_squared() < 0.001 {
        return Quat::IDENTITY;
    }

    // Build orthonormal basis
    let forward = to;
    let right = up.cross(forward).normalize_or_zero();
    let actual_up = forward.cross(right);

    // Build rotation from basis
    let from_forward = from.normalize_or_zero();
    let from_right = up.cross(from_forward).normalize_or_zero();
    let from_up = from_forward.cross(from_right);

    // Rotation from 'from' basis to 'to' basis
    let from_mat = glam::Mat3::from_cols(from_right, from_forward, from_up);
    let to_mat = glam::Mat3::from_cols(right, forward, actual_up);

    Quat::from_mat3(&(to_mat * from_mat.transpose()))
}

/// A set of constraints to evaluate.
#[derive(Default)]
pub struct ConstraintStack {
    constraints: Vec<Box<dyn Constraint>>,
}

impl ConstraintStack {
    /// Creates an empty constraint stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a constraint to the stack.
    pub fn push<C: Constraint + 'static>(&mut self, constraint: C) {
        self.constraints.push(Box::new(constraint));
    }

    /// Removes all constraints.
    pub fn clear(&mut self) {
        self.constraints.clear();
    }

    /// Returns the number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Evaluates all constraints in order.
    pub fn evaluate(&self, skeleton: &Skeleton, pose: &mut Pose) {
        for constraint in &self.constraints {
            constraint.apply(skeleton, pose);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bone, path3d::line3d};

    fn single_bone_skeleton() -> (Skeleton, BoneId) {
        let mut skel = Skeleton::new();
        let bone = skel.add_bone(Bone::new("test")).id;
        (skel, bone)
    }

    #[test]
    fn test_path_constraint_position() {
        let (skel, bone) = single_bone_skeleton();
        let mut pose = skel.rest_pose();

        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let mut constraint = PathConstraint::new(bone, path);
        constraint.offset = 0.5;
        constraint.follow_tangent = false;

        constraint.apply(&skel, &mut pose);

        let world = pose.world_transform(&skel, bone);
        assert!((world.translation.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_path_constraint_tangent() {
        let (skel, bone) = single_bone_skeleton();
        let mut pose = skel.rest_pose();

        // Path going in +X direction
        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let mut constraint = PathConstraint::new(bone, path).with_forward_axis(Vec3::Y);
        constraint.offset = 0.5;
        constraint.follow_tangent = true;

        constraint.apply(&skel, &mut pose);

        // The Y axis should now point in +X direction
        let world = pose.world_transform(&skel, bone);
        let rotated_y = world.rotation * Vec3::Y;
        assert!((rotated_y.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_constraint_stack() {
        let (skel, bone) = single_bone_skeleton();
        let mut pose = skel.rest_pose();

        let path = line3d(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let mut constraint = PathConstraint::new(bone, path);
        constraint.offset = 1.0;

        let mut stack = ConstraintStack::new();
        stack.push(constraint);

        stack.evaluate(&skel, &mut pose);

        let world = pose.world_transform(&skel, bone);
        assert!((world.translation.x - 10.0).abs() < 0.001);
    }
}
