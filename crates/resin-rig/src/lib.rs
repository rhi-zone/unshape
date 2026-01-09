//! Skeletal animation and rigging for resin.
//!
//! Provides types for bones, skeletons, poses, mesh skinning, constraints, animation, and IK.

mod animation;
mod blend;
mod constraint;
mod ik;
mod locomotion;
mod path3d;
mod skeleton;
mod skin;
mod transform;

pub use animation::{
    AnimationClip, AnimationPlayer, AnimationTarget, Interpolate, Interpolation, Keyframe, Track,
};
pub use blend::{AnimationLayer, AnimationPose, AnimationStack, BlendMode, BlendNode, Crossfade};
pub use constraint::{Constraint, ConstraintStack, PathConstraint};
pub use ik::{IkChain, IkConfig, IkResult, solve_ccd, solve_fabrik};
pub use locomotion::{
    FootPlacement, GaitConfig, GaitPattern, LegState, ProceduralHop, ProceduralWalk, WalkAnimator,
};
pub use path3d::{Path3D, Path3DBuilder, PathCommand3D, PathSample, line3d, polyline3d};
pub use skeleton::{AddBoneResult, Bone, BoneId, Pose, Skeleton};
pub use skin::{MAX_INFLUENCES, Skin, VertexInfluences};
pub use transform::Transform;
