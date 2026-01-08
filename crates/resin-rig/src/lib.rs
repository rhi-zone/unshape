//! Skeletal animation and rigging for resin.
//!
//! Provides types for bones, skeletons, poses, and mesh skinning.

mod skeleton;
mod skin;
mod transform;

pub use skeleton::{AddBoneResult, Bone, BoneId, Pose, Skeleton};
pub use skin::{MAX_INFLUENCES, Skin, VertexInfluences};
pub use transform::Transform;
