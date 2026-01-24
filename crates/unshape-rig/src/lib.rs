//! Skeletal animation and rigging for resin.
//!
//! Provides types for bones, skeletons, poses, mesh skinning, constraints, animation, and IK.

/// Registers all rig operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of rig ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Gait>("resin::Gait");
    registry.register_type::<Ik>("resin::Ik");
    registry.register_type::<MotionMatching>("resin::MotionMatching");
    registry.register_type::<Secondary>("resin::Secondary");
}

mod animation;
mod blend;
mod constraint;
mod ik;
mod locomotion;
mod motion_matching;
mod path3d;
pub mod secondary;
mod skeleton;
mod skin;
mod transform;

pub use animation::{
    AnimationClip, AnimationPlayer, AnimationTarget, Interpolate, Interpolation, Keyframe, Lerp,
    Track,
};
pub use blend::{AnimationLayer, AnimationPose, AnimationStack, BlendMode, BlendNode, Crossfade};
pub use constraint::{Constraint, ConstraintStack, PathConstraint};
pub use ik::{Ik, IkChain, IkConfig, IkResult, solve_ccd, solve_fabrik};
pub use locomotion::{
    FootPlacement, Gait, GaitConfig, GaitPattern, LegState, ProceduralHop, ProceduralWalk,
    WalkAnimator,
};
pub use motion_matching::{
    FrameRef, MatchResult, MotionClip, MotionDatabase, MotionFrame, MotionFrameBuilder,
    MotionMatcher, MotionMatching, MotionMatchingConfig, MotionQuery, apply_frame_to_pose,
    blend_frames, compute_match_cost, find_best_match,
};
pub use path3d::{Path3D, Path3DBuilder, Path3DExt, PathSample, line3d, polyline3d};
pub use unshape_transform::SpatialTransform;
pub use secondary::{
    Drag, FollowThrough, JiggleBone, JiggleChain, JiggleMesh, OverlappingAction,
    RotationFollowThrough, Secondary, SecondaryConfig, SecondaryMotion, WindForce,
    apply_wind_to_bone, apply_wind_to_chain,
};
pub use skeleton::{AddBoneResult, Bone, BoneId, Pose, Skeleton};
pub use skin::{MAX_INFLUENCES, Skin, VertexInfluences};
pub use transform::Transform3D;
