//! Dynamic value type for graph execution.

use glam::{Vec2, Vec3, Vec4};
use std::any::TypeId;
use std::fmt;

use crate::error::TypeError;

/// Runtime value type for dynamic graph execution.
///
/// This enum represents all possible values that can flow through a graph.
/// Type safety is enforced at graph construction time (via typed slots) or
/// at load time (via TypeId validation). At execution time, we trust the
/// graph is valid.
#[derive(Debug, Clone)]
pub enum Value {
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// 32-bit signed integer
    I32(i32),
    /// Boolean
    Bool(bool),
    /// 2D vector
    Vec2(Vec2),
    /// 3D vector
    Vec3(Vec3),
    /// 4D vector
    Vec4(Vec4),
    // TODO: Add Image, Mesh, Field, etc. as we implement them
}

/// Type identifier for values in the graph system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 32-bit signed integer.
    I32,
    /// Boolean.
    Bool,
    /// 2D vector.
    Vec2,
    /// 3D vector.
    Vec3,
    /// 4D vector.
    Vec4,
}

impl Value {
    /// Returns the type of this value.
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::I32(_) => ValueType::I32,
            Value::Bool(_) => ValueType::Bool,
            Value::Vec2(_) => ValueType::Vec2,
            Value::Vec3(_) => ValueType::Vec3,
            Value::Vec4(_) => ValueType::Vec4,
        }
    }

    /// Attempts to extract an f32 value.
    pub fn as_f32(&self) -> Result<f32, TypeError> {
        match self {
            Value::F32(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::F32, other.value_type())),
        }
    }

    /// Attempts to extract an f64 value.
    pub fn as_f64(&self) -> Result<f64, TypeError> {
        match self {
            Value::F64(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::F64, other.value_type())),
        }
    }

    /// Attempts to extract an i32 value.
    pub fn as_i32(&self) -> Result<i32, TypeError> {
        match self {
            Value::I32(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::I32, other.value_type())),
        }
    }

    /// Attempts to extract a bool value.
    pub fn as_bool(&self) -> Result<bool, TypeError> {
        match self {
            Value::Bool(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::Bool, other.value_type())),
        }
    }

    /// Attempts to extract a Vec2 value.
    pub fn as_vec2(&self) -> Result<Vec2, TypeError> {
        match self {
            Value::Vec2(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::Vec2, other.value_type())),
        }
    }

    /// Attempts to extract a Vec3 value.
    pub fn as_vec3(&self) -> Result<Vec3, TypeError> {
        match self {
            Value::Vec3(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::Vec3, other.value_type())),
        }
    }

    /// Attempts to extract a Vec4 value.
    pub fn as_vec4(&self) -> Result<Vec4, TypeError> {
        match self {
            Value::Vec4(v) => Ok(*v),
            other => Err(TypeError::expected(ValueType::Vec4, other.value_type())),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::F32 => write!(f, "f32"),
            ValueType::F64 => write!(f, "f64"),
            ValueType::I32 => write!(f, "i32"),
            ValueType::Bool => write!(f, "bool"),
            ValueType::Vec2 => write!(f, "Vec2"),
            ValueType::Vec3 => write!(f, "Vec3"),
            ValueType::Vec4 => write!(f, "Vec4"),
        }
    }
}

impl ValueType {
    /// Returns the TypeId for this value type.
    pub fn type_id(&self) -> TypeId {
        match self {
            ValueType::F32 => TypeId::of::<f32>(),
            ValueType::F64 => TypeId::of::<f64>(),
            ValueType::I32 => TypeId::of::<i32>(),
            ValueType::Bool => TypeId::of::<bool>(),
            ValueType::Vec2 => TypeId::of::<Vec2>(),
            ValueType::Vec3 => TypeId::of::<Vec3>(),
            ValueType::Vec4 => TypeId::of::<Vec4>(),
        }
    }
}

// Convenience From impls
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F32(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I32(v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<Vec2> for Value {
    fn from(v: Vec2) -> Self {
        Value::Vec2(v)
    }
}

impl From<Vec3> for Value {
    fn from(v: Vec3) -> Self {
        Value::Vec3(v)
    }
}

impl From<Vec4> for Value {
    fn from(v: Vec4) -> Self {
        Value::Vec4(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type() {
        assert_eq!(Value::F32(1.0).value_type(), ValueType::F32);
        assert_eq!(Value::F64(1.0).value_type(), ValueType::F64);
        assert_eq!(Value::I32(1).value_type(), ValueType::I32);
        assert_eq!(Value::Bool(true).value_type(), ValueType::Bool);
        assert_eq!(Value::Vec2(Vec2::ZERO).value_type(), ValueType::Vec2);
        assert_eq!(Value::Vec3(Vec3::ZERO).value_type(), ValueType::Vec3);
        assert_eq!(Value::Vec4(Vec4::ZERO).value_type(), ValueType::Vec4);
    }

    #[test]
    fn test_as_f32_success() {
        let v = Value::F32(3.14);
        assert_eq!(v.as_f32().unwrap(), 3.14);
    }

    #[test]
    fn test_as_f32_failure() {
        let v = Value::I32(42);
        assert!(v.as_f32().is_err());
    }

    #[test]
    fn test_as_f64_success() {
        let v = Value::F64(3.14);
        assert_eq!(v.as_f64().unwrap(), 3.14);
    }

    #[test]
    fn test_as_i32_success() {
        let v = Value::I32(42);
        assert_eq!(v.as_i32().unwrap(), 42);
    }

    #[test]
    fn test_as_bool_success() {
        assert!(Value::Bool(true).as_bool().unwrap());
        assert!(!Value::Bool(false).as_bool().unwrap());
    }

    #[test]
    fn test_as_vec2_success() {
        let v = Value::Vec2(Vec2::new(1.0, 2.0));
        assert_eq!(v.as_vec2().unwrap(), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_as_vec3_success() {
        let v = Value::Vec3(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.as_vec3().unwrap(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_as_vec4_success() {
        let v = Value::Vec4(Vec4::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(v.as_vec4().unwrap(), Vec4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_type_error_message() {
        let v = Value::Bool(true);
        let err = v.as_f32().unwrap_err();
        assert!(err.to_string().contains("f32"));
        assert!(err.to_string().contains("bool"));
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", ValueType::F32), "f32");
        assert_eq!(format!("{}", ValueType::Vec3), "Vec3");
    }

    #[test]
    fn test_value_type_type_id() {
        assert_eq!(ValueType::F32.type_id(), TypeId::of::<f32>());
        assert_eq!(ValueType::Vec3.type_id(), TypeId::of::<Vec3>());
    }

    #[test]
    fn test_from_impls() {
        let _: Value = 1.0f32.into();
        let _: Value = 1.0f64.into();
        let _: Value = 1i32.into();
        let _: Value = true.into();
        let _: Value = Vec2::ZERO.into();
        let _: Value = Vec3::ZERO.into();
        let _: Value = Vec4::ZERO.into();
    }
}
