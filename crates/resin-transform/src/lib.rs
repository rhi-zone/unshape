//! Spatial transform trait for unified 2D/3D transforms.
//!
//! This crate provides [`SpatialTransform`], a common interface for transforms
//! across dimensions, enabling generic algorithms that work with both 2D and 3D.
//!
//! # Implementors
//!
//! - `Transform` (resin-rig) - 3D transform with `Vec3`, `Quat`, `Mat4`
//! - `Transform2D` (resin-motion) - 2D transform with `Vec2`, `f32`, `Mat3`
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_transform::SpatialTransform;
//!
//! fn apply_transform<T: SpatialTransform>(transform: &T, points: &[T::Vector]) -> Vec<T::Vector>
//! where
//!     T::Vector: Copy,
//! {
//!     points.iter().map(|&p| transform.transform_point(p)).collect()
//! }
//! ```

/// Unified interface for spatial transforms (2D and 3D).
///
/// This trait abstracts over the dimensionality of transforms, allowing
/// generic code to work with both `Transform` (3D) and `Transform2D` (2D) types.
///
/// # Associated Types
///
/// - `Vector`: The position/translation type (`Vec2` or `Vec3`)
/// - `Rotation`: The rotation representation (`f32` radians for 2D, `Quat` for 3D)
/// - `Matrix`: The transformation matrix type (`Mat3` or `Mat4`)
pub trait SpatialTransform {
    /// The vector type for positions and translations.
    ///
    /// `Vec2` for 2D transforms, `Vec3` for 3D transforms.
    type Vector: Copy;

    /// The rotation representation.
    ///
    /// `f32` (radians) for 2D transforms, `Quat` for 3D transforms.
    type Rotation: Copy;

    /// The matrix type for the full transformation.
    ///
    /// `Mat3` for 2D transforms, `Mat4` for 3D transforms.
    type Matrix: Copy;

    /// Returns the translation component.
    fn translation(&self) -> Self::Vector;

    /// Returns the rotation component.
    fn rotation(&self) -> Self::Rotation;

    /// Returns the scale component.
    fn scale(&self) -> Self::Vector;

    /// Converts to a transformation matrix.
    ///
    /// The matrix transforms points from local space to parent/world space.
    fn to_matrix(&self) -> Self::Matrix;

    /// Transforms a point from local space to parent/world space.
    fn transform_point(&self, point: Self::Vector) -> Self::Vector;
}
