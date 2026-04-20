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
/// When `pole_target` is set, after FABRIK converges the chain is twisted
/// around the root→tip axis so the first interior joint points toward the
/// pole target.
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
    /// Optional pole vector to control the elbow/knee direction.
    pub pole_target: Option<Vec3>,
}

impl SolveFabrik {
    /// Creates a new FABRIK solver with the given configuration.
    pub fn new(config: IkConfig) -> Self {
        Self {
            config,
            pole_target: None,
        }
    }

    /// Creates a FABRIK solver with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: IkConfig::default(),
            pole_target: None,
        }
    }

    /// Creates a FABRIK solver with custom tolerance.
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self {
            config: IkConfig {
                tolerance,
                ..IkConfig::default()
            },
            pole_target: None,
        }
    }

    /// Creates a FABRIK solver with custom iteration limit.
    pub fn with_max_iterations(max_iterations: u32) -> Self {
        Self {
            config: IkConfig {
                max_iterations,
                ..IkConfig::default()
            },
            pole_target: None,
        }
    }

    /// Sets the pole target hint.
    pub fn with_pole_target(mut self, pole: Vec3) -> Self {
        self.pole_target = Some(pole);
        self
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
        let result = solve_fabrik(skeleton, pose, chain, target, &self.config);

        if let Some(pole) = self.pole_target {
            apply_pole_twist(skeleton, pose, chain, pole);
        }

        result
    }
}

impl Default for SolveFabrik {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Two-Bone Analytical IK Solver
// ============================================================================

/// Analytical (closed-form) IK solver for exactly 2-bone chains.
///
/// Faster and more predictable than iterative methods like CCD or FABRIK.
/// Suitable for arms (upper arm + forearm) and legs (thigh + shin).
///
/// The optional `pole_target` controls which direction the elbow/knee
/// bends: it is projected onto the plane perpendicular to the root→target
/// axis and used to orient the mid-joint in that plane.
///
/// # Example
///
/// ```ignore
/// use unshape_rig::{SolveTwoBone, IkChain, Skeleton, Pose};
/// use glam::Vec3;
///
/// let solver = SolveTwoBone {
///     target: Vec3::new(1.5, -1.0, 0.0),
///     pole_target: Some(Vec3::new(0.0, 0.0, 1.0)),
/// };
/// let result = solver.solve(&chain, &skeleton, &mut pose);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveTwoBone {
    /// World-space target position for the end effector.
    pub target: Vec3,
    /// Optional pole target (elbow/knee hint) in world space.
    pub pole_target: Option<Vec3>,
}

impl SolveTwoBone {
    /// Solves the two-bone IK chain analytically.
    ///
    /// `chain` must contain exactly 2 bones; returns early with `reached: false`
    /// if the chain length differs.
    pub fn solve(&self, chain: &IkChain, skeleton: &Skeleton, pose: &mut Pose) -> IkResult {
        if chain.len() != 2 {
            return IkResult {
                reached: false,
                distance: f32::MAX,
                iterations: 0,
            };
        }

        let bone0_id = chain.bones[0];
        let bone1_id = chain.bones[1];

        let len0 = skeleton.bone(bone0_id).map(|b| b.length).unwrap_or(1.0);
        let len1 = skeleton.bone(bone1_id).map(|b| b.length).unwrap_or(1.0);
        let total_len = len0 + len1;

        let root_pos = bone_head_world(skeleton, pose, bone0_id);
        let to_target = self.target - root_pos;
        let dist_raw = to_target.length();

        // Clamp distance to reachable range; at least epsilon to avoid NaNs.
        let dist = dist_raw.clamp(f32::EPSILON, total_len - f32::EPSILON);
        let reached = dist_raw <= total_len;

        // Direction from root toward target.
        let root_to_target = to_target.normalize_or_zero();

        // Law of cosines: angle at bone0 (elbow angle on the root side).
        // dist² = len0² + len1² - 2·len0·len1·cos(elbow_angle)
        // cos(elbow_angle) = (len0² + dist² - len1²) / (2·len0·dist)
        let cos_angle0 =
            ((len0 * len0 + dist * dist - len1 * len1) / (2.0 * len0 * dist)).clamp(-1.0, 1.0);
        let angle0 = cos_angle0.acos();

        // Bend plane normal: default to a plane containing root_to_target.
        // If a pole target is given, use it to define the plane.
        let bend_normal = if let Some(pole) = self.pole_target {
            let to_pole = (pole - root_pos).normalize_or_zero();
            // Project pole onto plane perpendicular to root→target.
            let proj = to_pole - root_to_target * to_pole.dot(root_to_target);
            if proj.length_squared() > 0.0001 {
                proj.normalize()
            } else {
                best_perp(root_to_target)
            }
        } else {
            best_perp(root_to_target)
        };

        // Mid-joint position: rotate root_to_target by angle0 around bend_normal.
        let rot0 = Quat::from_axis_angle(bend_normal, angle0);
        let mid_dir = rot0 * root_to_target;
        let mid_pos = root_pos + mid_dir * len0;
        let end_pos = root_pos + root_to_target * dist;

        // Apply positions back to pose.
        let positions = [root_pos, mid_pos, end_pos];
        apply_positions_to_pose(skeleton, pose, chain, &positions);

        let final_distance = (end_pos - self.target).length();
        IkResult {
            reached,
            distance: final_distance,
            iterations: 1,
        }
    }
}

/// Returns an arbitrary unit vector perpendicular to `v`.
fn best_perp(v: Vec3) -> Vec3 {
    let candidate = if v.abs().dot(Vec3::Y) < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    v.cross(candidate).normalize_or_zero()
}

/// Applies a pole-vector twist to the chain after a FABRIK solve.
///
/// Rotates all bones around the root→tip axis so the first interior joint
/// points toward `pole`.
fn apply_pole_twist(skeleton: &Skeleton, pose: &mut Pose, chain: &IkChain, pole: Vec3) {
    if chain.len() < 2 {
        return;
    }

    let root_pos = bone_head_world(skeleton, pose, chain.bones[0]);
    let tip_pos = {
        let end = chain.end_bone().unwrap();
        bone_tip_world(skeleton, pose, end)
    };

    let axis = (tip_pos - root_pos).normalize_or_zero();
    if axis.length_squared() < 0.0001 {
        return;
    }

    // Current mid-joint direction (projected onto the plane perp to axis).
    let mid_pos = bone_head_world(skeleton, pose, chain.bones[1]);
    let mid_vec = mid_pos - root_pos;
    let mid_proj = (mid_vec - axis * mid_vec.dot(axis)).normalize_or_zero();

    // Desired direction: pole projected onto same plane.
    let pole_vec = pole - root_pos;
    let pole_proj = (pole_vec - axis * pole_vec.dot(axis)).normalize_or_zero();

    if mid_proj.length_squared() < 0.0001 || pole_proj.length_squared() < 0.0001 {
        return;
    }

    let twist = Quat::from_rotation_arc(mid_proj, pole_proj);
    if twist.is_nan() {
        return;
    }

    // Apply twist rotation to all bones in the chain (world-to-local conversion).
    for &bone_id in &chain.bones {
        let current = pose.get(bone_id);
        let world = pose.world_transform(skeleton, bone_id);
        let parent_world = skeleton
            .bone(bone_id)
            .and_then(|b| b.parent)
            .map(|p| pose.world_transform(skeleton, p))
            .unwrap_or(Transform3D::IDENTITY);

        let new_world_rot = twist * world.rotation;
        let local_rot = parent_world.rotation.inverse() * new_world_rot;

        pose.set(
            bone_id,
            Transform3D {
                rotation: local_rot,
                ..current
            },
        );
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

    #[test]
    fn test_two_bone_reachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        // Target within reach (total chain length = 2.0)
        let solver = SolveTwoBone {
            target: Vec3::new(1.0, 1.0, 0.0),
            pole_target: None,
        };
        let result = solver.solve(&chain, &skel, &mut pose);

        assert!(result.reached);
        assert!(result.distance < 0.01, "distance: {}", result.distance);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_two_bone_unreachable() {
        let (skel, chain, mut pose) = two_bone_chain();

        // Target beyond reach
        let solver = SolveTwoBone {
            target: Vec3::new(10.0, 0.0, 0.0),
            pole_target: None,
        };
        let result = solver.solve(&chain, &skel, &mut pose);

        assert!(!result.reached);
    }

    #[test]
    fn test_two_bone_requires_exactly_two_bones() {
        let (skel, _, _) = two_bone_chain();
        let chain = IkChain::new(vec![BoneId(0)]);
        let mut pose = skel.rest_pose();

        let solver = SolveTwoBone {
            target: Vec3::new(1.0, 0.0, 0.0),
            pole_target: None,
        };
        let result = solver.solve(&chain, &skel, &mut pose);

        assert!(!result.reached);
        assert_eq!(result.distance, f32::MAX);
    }

    #[test]
    fn test_fabrik_with_pole_target() {
        let (skel, chain, mut pose) = two_bone_chain();

        let solver = SolveFabrik {
            config: IkConfig::default(),
            pole_target: Some(Vec3::new(0.0, 0.0, 1.0)),
        };
        let result = solver.apply(&skel, &mut pose, &chain, Vec3::new(1.0, 1.0, 0.0));

        // Should still reach target
        assert!(result.distance < 0.1, "distance: {}", result.distance);
    }
}
