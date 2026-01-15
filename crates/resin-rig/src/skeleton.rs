//! Skeleton, bone, and pose types.

use crate::transform::Transform;
use glam::Vec3;

/// A bone identifier (index into skeleton).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoneId(pub u32);

impl BoneId {
    /// Creates a new bone ID.
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A bone in a skeleton.
#[derive(Debug, Clone)]
pub struct Bone {
    /// Human-readable name.
    pub name: String,
    /// Parent bone (None for root).
    pub parent: Option<BoneId>,
    /// Local transform in parent space (rest/bind pose).
    pub local_transform: Transform,
    /// Length of the bone (for visualization, IK).
    pub length: f32,
}

impl Default for Bone {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: None,
            local_transform: Transform::IDENTITY,
            length: 1.0,
        }
    }
}

impl Bone {
    /// Creates a new bone.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Returns the tail position in local space.
    pub fn tail_local(&self) -> Vec3 {
        Vec3::new(0.0, self.length, 0.0)
    }
}

/// A skeleton (hierarchy of bones).
#[derive(Debug, Clone, Default)]
pub struct Skeleton {
    bones: Vec<Bone>,
}

/// Result of adding a bone to a skeleton.
#[derive(Debug, Clone)]
pub struct AddBoneResult {
    /// The ID of the added bone.
    pub id: BoneId,
}

impl Skeleton {
    /// Creates an empty skeleton.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a bone to the skeleton.
    pub fn add_bone(&mut self, bone: Bone) -> AddBoneResult {
        let id = BoneId(self.bones.len() as u32);
        self.bones.push(bone);
        AddBoneResult { id }
    }

    /// Returns the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Returns a bone by ID.
    pub fn bone(&self, id: BoneId) -> Option<&Bone> {
        self.bones.get(id.index())
    }

    /// Returns a mutable bone by ID.
    pub fn bone_mut(&mut self, id: BoneId) -> Option<&mut Bone> {
        self.bones.get_mut(id.index())
    }

    /// Returns all bones.
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// Finds a bone by name.
    pub fn find_bone(&self, name: &str) -> Option<BoneId> {
        self.bones
            .iter()
            .position(|b| b.name == name)
            .map(|i| BoneId(i as u32))
    }

    /// Returns root bones (bones without parents).
    pub fn root_bones(&self) -> Vec<BoneId> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent.is_none())
            .map(|(i, _)| BoneId(i as u32))
            .collect()
    }

    /// Returns children of a bone.
    pub fn children(&self, parent: BoneId) -> Vec<BoneId> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent == Some(parent))
            .map(|(i, _)| BoneId(i as u32))
            .collect()
    }

    /// Computes the world transform for a bone in rest pose.
    pub fn world_transform(&self, id: BoneId) -> Transform {
        let mut transform = Transform::IDENTITY;
        let mut current = Some(id);

        // Collect chain from root to bone
        let mut chain = Vec::new();
        while let Some(bone_id) = current {
            chain.push(bone_id);
            current = self.bones.get(bone_id.index()).and_then(|b| b.parent);
        }

        // Apply transforms from root to bone
        for bone_id in chain.into_iter().rev() {
            if let Some(bone) = self.bones.get(bone_id.index()) {
                transform = transform.then(&bone.local_transform);
            }
        }

        transform
    }

    /// Creates a rest pose for this skeleton.
    pub fn rest_pose(&self) -> Pose {
        Pose::rest(self.bone_count())
    }
}

/// A pose (animated bone transforms).
///
/// Transforms are relative to the bone's rest pose.
#[derive(Debug, Clone)]
pub struct Pose {
    /// Per-bone transforms (delta from rest pose).
    transforms: Vec<Transform>,
}

impl Pose {
    /// Creates a rest pose (all identity transforms).
    pub fn rest(bone_count: usize) -> Self {
        Self {
            transforms: vec![Transform::IDENTITY; bone_count],
        }
    }

    /// Gets the transform for a bone.
    pub fn get(&self, id: BoneId) -> Transform {
        self.transforms
            .get(id.index())
            .copied()
            .unwrap_or(Transform::IDENTITY)
    }

    /// Sets the transform for a bone.
    pub fn set(&mut self, id: BoneId, transform: Transform) {
        if let Some(t) = self.transforms.get_mut(id.index()) {
            *t = transform;
        }
    }

    /// Returns the number of bone transforms.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns true if the pose is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Returns all transforms.
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// Returns mutable access to all transforms.
    pub fn transforms_mut(&mut self) -> &mut [Transform] {
        &mut self.transforms
    }

    /// Blends two poses.
    pub fn blend(&self, other: &Pose, t: f32) -> Pose {
        let len = self.transforms.len().max(other.transforms.len());
        let mut result = Pose::rest(len);

        for i in 0..len {
            let a = self
                .transforms
                .get(i)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            let b = other
                .transforms
                .get(i)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            result.transforms[i] = a.lerp(&b, t);
        }

        result
    }

    /// Computes the final world transform for a bone.
    pub fn world_transform(&self, skeleton: &Skeleton, id: BoneId) -> Transform {
        let mut transform = Transform::IDENTITY;
        let mut current = Some(id);

        // Collect chain from root to bone
        let mut chain = Vec::new();
        while let Some(bone_id) = current {
            chain.push(bone_id);
            current = skeleton.bone(bone_id).and_then(|b| b.parent);
        }

        // Apply transforms from root to bone
        for bone_id in chain.into_iter().rev() {
            if let Some(bone) = skeleton.bone(bone_id) {
                // local_transform * pose_transform
                let posed = bone.local_transform.then(&self.get(bone_id));
                transform = transform.then(&posed);
            }
        }

        transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_2;

    fn simple_skeleton() -> (Skeleton, BoneId, BoneId, BoneId) {
        let mut skel = Skeleton::new();

        let root = skel
            .add_bone(Bone {
                name: "root".into(),
                parent: None,
                local_transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                length: 1.0,
            })
            .id;

        let upper = skel
            .add_bone(Bone {
                name: "upper".into(),
                parent: Some(root),
                local_transform: Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                length: 1.0,
            })
            .id;

        let lower = skel
            .add_bone(Bone {
                name: "lower".into(),
                parent: Some(upper),
                local_transform: Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                length: 1.0,
            })
            .id;

        (skel, root, upper, lower)
    }

    #[test]
    fn test_skeleton_creation() {
        let (skel, root, upper, lower) = simple_skeleton();

        assert_eq!(skel.bone_count(), 3);
        assert_eq!(skel.bone(root).unwrap().name, "root");
        assert_eq!(skel.bone(upper).unwrap().parent, Some(root));
        assert_eq!(skel.bone(lower).unwrap().parent, Some(upper));
    }

    #[test]
    fn test_find_bone() {
        let (skel, _, upper, _) = simple_skeleton();

        assert_eq!(skel.find_bone("upper"), Some(upper));
        assert_eq!(skel.find_bone("nonexistent"), None);
    }

    #[test]
    fn test_root_bones() {
        let (skel, root, _, _) = simple_skeleton();
        let roots = skel.root_bones();
        assert_eq!(roots, vec![root]);
    }

    #[test]
    fn test_children() {
        let (skel, root, upper, lower) = simple_skeleton();

        assert_eq!(skel.children(root), vec![upper]);
        assert_eq!(skel.children(upper), vec![lower]);
        assert_eq!(skel.children(lower), Vec::<BoneId>::new());
    }

    #[test]
    fn test_world_transform() {
        let (skel, root, upper, lower) = simple_skeleton();

        let root_world = skel.world_transform(root);
        assert_eq!(root_world.translation, Vec3::ZERO);

        let upper_world = skel.world_transform(upper);
        assert_eq!(upper_world.translation, Vec3::new(0.0, 1.0, 0.0));

        let lower_world = skel.world_transform(lower);
        assert_eq!(lower_world.translation, Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn test_rest_pose() {
        let (skel, _, _, _) = simple_skeleton();
        let pose = skel.rest_pose();

        assert_eq!(pose.len(), 3);
        assert_eq!(pose.get(BoneId(0)), Transform::IDENTITY);
    }

    #[test]
    fn test_posed_transform() {
        let (skel, _, upper, lower) = simple_skeleton();
        let mut pose = skel.rest_pose();

        // Rotate upper bone 90 degrees around Z
        pose.set(
            upper,
            Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );

        let lower_world = pose.world_transform(&skel, lower);
        // With upper rotated 90° CCW, lower's local Y offset becomes -X offset
        assert!((lower_world.translation.x - (-1.0)).abs() < 0.0001);
        assert!((lower_world.translation.y - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_pose_blend() {
        let (skel, _, upper, _) = simple_skeleton();
        let pose_a = skel.rest_pose();
        let mut pose_b = skel.rest_pose();

        pose_b.set(
            upper,
            Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );

        let blended = pose_a.blend(&pose_b, 0.5);
        assert_eq!(blended.get(upper).translation, Vec3::new(5.0, 0.0, 0.0));
    }
}
