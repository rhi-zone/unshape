//! Transform type for skeletal animation.

use glam::{Mat4, Quat, Vec3};
use rhizome_resin_easing::Lerp;
use rhizome_resin_transform::SpatialTransform;

/// A 3D transform (translation, rotation, scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Position offset.
    pub translation: Vec3,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Scale factors per axis.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// Identity transform (no translation, rotation, or scale).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a new transform.
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Creates a transform with only translation.
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only rotation.
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only scale.
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with uniform scale.
    pub fn from_uniform_scale(scale: f32) -> Self {
        Self::from_scale(Vec3::splat(scale))
    }

    /// Converts to a 4x4 matrix (TRS order).
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Creates a transform from a 4x4 matrix.
    ///
    /// Note: This assumes the matrix contains only TRS transformations (no shear).
    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Combines two transforms (self then other).
    ///
    /// This is equivalent to multiplying their matrices.
    pub fn then(&self, other: &Transform) -> Transform {
        // For proper TRS composition:
        // T' = T1 + R1 * S1 * T2
        // R' = R1 * R2
        // S' = S1 * S2 (component-wise, assuming no shear)
        Transform {
            translation: self.translation + self.rotation * (self.scale * other.translation),
            rotation: self.rotation * other.rotation,
            scale: self.scale * other.scale,
        }
    }

    /// Returns the inverse transform.
    pub fn inverse(&self) -> Transform {
        let inv_rotation = self.rotation.inverse();
        let inv_scale = Vec3::ONE / self.scale;
        let inv_translation = inv_rotation * (-self.translation * inv_scale);
        Transform {
            translation: inv_translation,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }

    /// Transforms a point.
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * point)
    }

    /// Transforms a direction (ignores translation, normalizes result).
    pub fn transform_direction(&self, direction: Vec3) -> Vec3 {
        (self.rotation * direction).normalize()
    }

    /// Transforms a vector (ignores translation, applies scale).
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * (self.scale * vector)
    }

    /// Linearly interpolates between two transforms.
    pub fn lerp(&self, other: &Transform, t: f32) -> Transform {
        Transform {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

impl Lerp for Transform {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }
}

impl From<Transform> for Mat4 {
    fn from(t: Transform) -> Self {
        t.to_matrix()
    }
}

impl From<Mat4> for Transform {
    fn from(m: Mat4) -> Self {
        Transform::from_matrix(m)
    }
}

impl SpatialTransform for Transform {
    type Vector = Vec3;
    type Rotation = Quat;
    type Matrix = Mat4;

    fn translation(&self) -> Vec3 {
        self.translation
    }

    fn rotation(&self) -> Quat {
        self.rotation
    }

    fn scale(&self) -> Vec3 {
        self.scale
    }

    fn to_matrix(&self) -> Mat4 {
        self.to_matrix()
    }

    fn transform_point(&self, point: Vec3) -> Vec3 {
        self.transform_point(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_identity() {
        let t = Transform::IDENTITY;
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(p), p);
    }

    #[test]
    fn test_translation() {
        let t = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(t.transform_point(p), Vec3::new(11.0, 2.0, 3.0));
    }

    #[test]
    fn test_rotation() {
        let t = Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2));
        let p = Vec3::new(1.0, 0.0, 0.0);
        let result = t.transform_point(p);
        assert!((result.x).abs() < 0.0001);
        assert!((result.y - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_scale() {
        let t = Transform::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(t.transform_point(p), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_inverse() {
        let t = Transform::new(
            Vec3::new(5.0, 3.0, 1.0),
            Quat::from_rotation_y(0.5),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let inv = t.inverse();
        let combined = t.then(&inv);

        assert!((combined.translation - Vec3::ZERO).length() < 0.0001);
        assert!((combined.rotation.w - 1.0).abs() < 0.0001);
        assert!((combined.scale - Vec3::ONE).length() < 0.0001);
    }

    #[test]
    fn test_lerp() {
        let a = Transform::from_translation(Vec3::ZERO);
        let b = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.translation, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn test_matrix_roundtrip() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_x(0.5),
            Vec3::new(1.5, 1.5, 1.5),
        );
        let m = t.to_matrix();
        let t2 = Transform::from_matrix(m);

        assert!((t.translation - t2.translation).length() < 0.0001);
        assert!((t.rotation.w - t2.rotation.w).abs() < 0.0001);
        assert!((t.scale - t2.scale).length() < 0.0001);
    }

    #[test]
    fn test_spatial_transform_trait() {
        // Test using the trait generically
        fn transform_points<T: SpatialTransform<Vector = Vec3>>(
            transform: &T,
            points: &[Vec3],
        ) -> Vec<Vec3> {
            points
                .iter()
                .map(|&p| transform.transform_point(p))
                .collect()
        }

        let t = Transform::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);

        let points = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let transformed = transform_points(&t, &points);

        assert_eq!(transformed[0], Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(transformed[1], Vec3::new(11.0, 0.0, 0.0));
        assert_eq!(transformed[2], Vec3::new(10.0, 1.0, 0.0));

        // Verify trait accessors match struct fields
        assert_eq!(SpatialTransform::translation(&t), t.translation);
        assert_eq!(SpatialTransform::rotation(&t), t.rotation);
        assert_eq!(SpatialTransform::scale(&t), t.scale);
    }
}
