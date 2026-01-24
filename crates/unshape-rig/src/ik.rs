//! Inverse Kinematics solvers.
//!
//! Provides CCD and FABRIK algorithms for positioning bone chains.
//!
//! # Ops-as-Values
//!
//! The [`SolveCcd`] and [`SolveFabrik`] structs wrap the IK algorithms as
//! serializable operations, enabling:
//! - Project file storage
//! - Node graph integration
//! - Animation pipeline composition
//!
//! ```ignore
//! use unshape_rig::{SolveCcd, IkConfig};
//!
//! // Create a reusable solver
//! let solver = SolveCcd::new(IkConfig::default());
//!
//! // Use in animation loop
//! let result = solver.apply(&skeleton, &mut pose, &chain, target);
//! ```

use crate::{BoneId, Pose, Skeleton, Transform3D};
use glam::{Quat, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for IK solving.
///
/// Controls iteration limits and convergence thresholds for IK solvers.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Ik))]
pub struct Ik {
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Distance threshold for success.
    pub tolerance: f32,
}

/// Backwards-compatible type alias.
pub type IkConfig = Ik;

impl Ik {
    /// Applies this generator, returning the configuration.
    pub fn apply(&self) -> Ik {
        *self
    }
}

impl Default for Ik {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 0.001,
        }
    }
}

/// Result of an IK solve.
#[derive(Debug, Clone, Copy)]
pub struct IkResult {
    /// Whether the target was reached within tolerance.
    pub reached: bool,
    /// Final distance to target.
    pub distance: f32,
    /// Number of iterations used.
    pub iterations: u32,
}

/// A chain of bones for IK solving.
#[derive(Debug, Clone)]
pub struct IkChain {
    /// Bones in the chain, from root to end effector.
    pub bones: Vec<BoneId>,
}

impl IkChain {
    /// Creates a new IK chain.
    pub fn new(bones: Vec<BoneId>) -> Self {
        Self { bones }
    }

    /// Creates a chain from a skeleton, walking from end bone to root.
    pub fn from_end_bone(skeleton: &Skeleton, end_bone: BoneId, length: usize) -> Self {
        let mut bones = Vec::with_capacity(length);
        let mut current = Some(end_bone);

        while let Some(bone_id) = current {
            bones.push(bone_id);
            if bones.len() >= length {
                break;
            }
            current = skeleton.bone(bone_id).and_then(|b| b.parent);
        }

        bones.reverse(); // Root to end
        Self { bones }
    }

    /// Returns the end effector bone.
    pub fn end_bone(&self) -> Option<BoneId> {
        self.bones.last().copied()
    }

    /// Returns the root bone of the chain.
    pub fn root_bone(&self) -> Option<BoneId> {
        self.bones.first().copied()
    }

    /// Returns the chain length.
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }
}

/// Gets the world position of a bone's tip (end point).
fn bone_tip_world(skeleton: &Skeleton, pose: &Pose, bone_id: BoneId) -> Vec3 {
    let world = pose.world_transform(skeleton, bone_id);
    let bone = skeleton.bone(bone_id).unwrap();
    world.transform_point(bone.tail_local())
}

/// Gets the world position of a bone's head (start point).
fn bone_head_world(skeleton: &Skeleton, pose: &Pose, bone_id: BoneId) -> Vec3 {
    pose.world_transform(skeleton, bone_id).translation
}

// ============================================================================
// CCD (Cyclic Coordinate Descent)
// ============================================================================

/// Solves IK using Cyclic Coordinate Descent.
///
/// CCD works by iterating through the chain from end to root,
/// rotating each bone to point toward the target.
pub fn solve_ccd(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: &IkChain,
    target: Vec3,
    config: &IkConfig,
) -> IkResult {
    if chain.is_empty() {
        return IkResult {
            reached: false,
            distance: f32::MAX,
            iterations: 0,
        };
    }

    let end_bone = chain.end_bone().unwrap();
    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Work backward through the chain
        for &bone_id in chain.bones.iter().rev() {
            let end_pos = bone_tip_world(skeleton, pose, end_bone);
            let bone_pos = bone_head_world(skeleton, pose, bone_id);

            // Vector from bone to end effector
            let to_end = (end_pos - bone_pos).normalize_or_zero();
            // Vector from bone to target
            let to_target = (target - bone_pos).normalize_or_zero();

            if to_end.length_squared() < 0.0001 || to_target.length_squared() < 0.0001 {
                continue;
            }

            // Rotation to align end effector toward target
            let rotation = Quat::from_rotation_arc(to_end, to_target);

            // Apply rotation in local space
            let current = pose.get(bone_id);
            let world = pose.world_transform(skeleton, bone_id);
            let parent_world = skeleton
                .bone(bone_id)
                .and_then(|b| b.parent)
                .map(|p| pose.world_transform(skeleton, p))
                .unwrap_or(Transform3D::IDENTITY);

            // Convert world rotation to local
            let new_world_rot = rotation * world.rotation;
            let local_rot = parent_world.rotation.inverse() * new_world_rot;

            pose.set(
                bone_id,
                Transform3D {
                    rotation: local_rot,
                    ..current
                },
            );
        }

        // Check convergence
        let end_pos = bone_tip_world(skeleton, pose, end_bone);
        let distance = (end_pos - target).length();
        if distance < config.tolerance {
            return IkResult {
                reached: true,
                distance,
                iterations,
            };
        }
    }

    let end_pos = bone_tip_world(skeleton, pose, end_bone);
    IkResult {
        reached: false,
        distance: (end_pos - target).length(),
        iterations,
    }
}

// ============================================================================
// FABRIK (Forward And Backward Reaching Inverse Kinematics)
// ============================================================================

/// Solves IK using FABRIK algorithm.
///
/// FABRIK is a heuristic iterative method that uses two passes:
/// 1. Forward: Pull chain toward target
/// 2. Backward: Pull chain back to root
pub fn solve_fabrik(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: &IkChain,
    target: Vec3,
    config: &IkConfig,
) -> IkResult {
    if chain.is_empty() {
        return IkResult {
            reached: false,
            distance: f32::MAX,
            iterations: 0,
        };
    }

    // Get current joint positions
    let mut positions: Vec<Vec3> = chain
        .bones
        .iter()
        .map(|&b| bone_head_world(skeleton, pose, b))
        .collect();

    // Add end effector position
    let end_bone = chain.end_bone().unwrap();
    positions.push(bone_tip_world(skeleton, pose, end_bone));

    // Calculate bone lengths
    let lengths: Vec<f32> = positions
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .collect();

    // Check if target is reachable
    let total_length: f32 = lengths.iter().sum();
    let root_pos = positions[0];
    let root_to_target = (target - root_pos).length();

    if root_to_target > total_length {
        // Target unreachable, stretch toward it
        let dir = (target - root_pos).normalize_or_zero();
        let mut pos = root_pos;
        for (i, &len) in lengths.iter().enumerate() {
            positions[i] = pos;
            pos += dir * len;
        }
        positions[lengths.len()] = pos;

        apply_positions_to_pose(skeleton, pose, chain, &positions);

        return IkResult {
            reached: false,
            distance: (pos - target).length(),
            iterations: 1,
        };
    }

    let mut iterations = 0;

    let last_idx = positions.len() - 1;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Forward reaching (from end to root)
        positions[last_idx] = target;
        for i in (0..positions.len() - 1).rev() {
            let dir = (positions[i] - positions[i + 1]).normalize_or_zero();
            positions[i] = positions[i + 1] + dir * lengths[i];
        }

        // Backward reaching (from root to end)
        positions[0] = root_pos;
        for i in 0..positions.len() - 1 {
            let dir = (positions[i + 1] - positions[i]).normalize_or_zero();
            positions[i + 1] = positions[i] + dir * lengths[i];
        }

        // Check convergence
        let distance = (positions[last_idx] - target).length();
        if distance < config.tolerance {
            apply_positions_to_pose(skeleton, pose, chain, &positions);
            return IkResult {
                reached: true,
                distance,
                iterations,
            };
        }
    }

    apply_positions_to_pose(skeleton, pose, chain, &positions);

    let distance = (positions[last_idx] - target).length();
    IkResult {
        reached: distance < config.tolerance,
        distance,
        iterations,
    }
}

/// Applies solved positions back to pose rotations.
fn apply_positions_to_pose(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: &IkChain,
    positions: &[Vec3],
) {
    for (i, &bone_id) in chain.bones.iter().enumerate() {
        let bone = skeleton.bone(bone_id).unwrap();
        let current_pos = positions[i];
        let target_pos = positions[i + 1];

        // Direction bone should point
        let desired_dir = (target_pos - current_pos).normalize_or_zero();

        // Current bone direction in world space
        let parent_world = bone
            .parent
            .map(|p| pose.world_transform(skeleton, p))
            .unwrap_or(Transform3D::IDENTITY);

        let current_transform = pose.get(bone_id);
        let world_rot = parent_world.rotation * current_transform.rotation;

        // Bone's rest direction (typically Y+)
        let bone_dir = world_rot * Vec3::Y;

        // Rotation from current to desired
        let rotation = Quat::from_rotation_arc(bone_dir, desired_dir);
        let new_world_rot = rotation * world_rot;

        // Convert to local space
        let local_rot = parent_world.rotation.inverse() * new_world_rot;

        pose.set(
            bone_id,
            Transform3D {
                rotation: local_rot,
                ..current_transform
            },
        );
    }
}

// ============================================================================
// IK Solver Op Structs
// ============================================================================

/// CCD (Cyclic Coordinate Descent) IK solver as an op struct.
///
/// This wraps the CCD algorithm in a serializable struct for use in
/// animation pipelines and node graphs.
///
/// # Example
///
/// ```ignore
/// use unshape_rig::{SolveCcd, IkConfig, IkChain, Skeleton, Pose};
/// use glam::Vec3;
///
/// let solver = SolveCcd::new(IkConfig::default());
/// let result = solver.apply(&skeleton, &mut pose, &chain, Vec3::new(1.0, 1.0, 0.0));
/// println!("Reached target: {}", result.reached);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveCcd {
    /// IK solver configuration.
    pub config: IkConfig,
}

impl SolveCcd {
    /// Creates a new CCD solver with the given configuration.
    pub fn new(config: IkConfig) -> Self {
        Self { config }
    }

    /// Creates a CCD solver with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: IkConfig::default(),
        }
    }

    /// Solves IK for the given chain to reach the target position.
    ///
    /// This mutates the pose in place and returns the solve result.
    pub fn apply(
        &self,
        skeleton: &Skeleton,
        pose: &mut Pose,
        chain: &IkChain,
        target: Vec3,
    ) -> IkResult {
        solve_ccd(skeleton, pose, chain, target, &self.config)
    }
}

impl Default for SolveCcd {
    fn default() -> Self {
        Self::default_config()
    }
}

/// FABRIK (Forward And Backward Reaching Inverse Kinematics) solver as an op struct.
///
/// FABRIK is often faster than CCD and produces smoother results for
/// longer chains.
///
/// # Example
///
/// ```ignore
/// use unshape_rig::{SolveFabrik, IkConfig, IkChain, Skeleton, Pose};
/// use glam::Vec3;
///
/// let solver = SolveFabrik::with_tolerance(0.01);
/// let result = solver.apply(&skeleton, &mut pose, &chain, Vec3::new(2.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveFabrik {
    /// IK solver configuration.
    pub config: IkConfig,
}

impl SolveFabrik {
    /// Creates a new FABRIK solver with the given configuration.
    pub fn new(config: IkConfig) -> Self {
        Self { config }
    }

    /// Creates a FABRIK solver with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: IkConfig::default(),
        }
    }

    /// Creates a FABRIK solver with custom tolerance.
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self {
            config: IkConfig {
                tolerance,
                ..IkConfig::default()
            },
        }
    }

    /// Creates a FABRIK solver with custom iteration limit.
    pub fn with_max_iterations(max_iterations: u32) -> Self {
        Self {
            config: IkConfig {
                max_iterations,
                ..IkConfig::default()
            },
        }
    }

    /// Solves IK for the given chain to reach the target position.
    ///
    /// This mutates the pose in place and returns the solve result.
    pub fn apply(
        &self,
        skeleton: &Skeleton,
        pose: &mut Pose,
        chain: &IkChain,
        target: Vec3,
    ) -> IkResult {
        solve_fabrik(skeleton, pose, chain, target, &self.config)
    }
}

impl Default for SolveFabrik {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bone;

    fn two_bone_chain() -> (Skeleton, IkChain, Pose) {
        let mut skel = Skeleton::new();

        let root = skel
            .add_bone(Bone {
                name: "root".into(),
                parent: None,
                local_transform: Transform3D::IDENTITY,
                length: 1.0,
            })
            .id;

        let end = skel
            .add_bone(Bone {
                name: "end".into(),
                parent: Some(root),
                local_transform: Transform3D::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                length: 1.0,
            })
            .id;

        let chain = IkChain::new(vec![root, end]);
        let pose = skel.rest_pose();

        (skel, chain, pose)
    }

    #[test]
    fn test_chain_from_end_bone() {
        let (skel, _, _) = two_bone_chain();
        let chain = IkChain::from_end_bone(&skel, BoneId(1), 2);

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.root_bone(), Some(BoneId(0)));
        assert_eq!(chain.end_bone(), Some(BoneId(1)));
    }

    #[test]
    fn test_ccd_reachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        // Target within reach
        let target = Vec3::new(1.5, 1.0, 0.0);
        let result = solve_ccd(&skel, &mut pose, &chain, target, &IkConfig::default());

        assert!(result.distance < 0.1);
    }

    #[test]
    fn test_ccd_unreachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        // Target way out of reach
        let target = Vec3::new(100.0, 0.0, 0.0);
        let result = solve_ccd(&skel, &mut pose, &chain, target, &IkConfig::default());

        // Should stretch toward target but not reach
        assert!(!result.reached);
    }

    #[test]
    fn test_fabrik_reachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        let target = Vec3::new(1.0, 1.0, 0.0);
        let result = solve_fabrik(&skel, &mut pose, &chain, target, &IkConfig::default());

        assert!(result.distance < 0.1);
    }

    #[test]
    fn test_fabrik_unreachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        // Target beyond max reach
        let target = Vec3::new(10.0, 0.0, 0.0);
        let result = solve_fabrik(&skel, &mut pose, &chain, target, &IkConfig::default());

        assert!(!result.reached);
    }
}
