//! Dynamic value type for graph execution.
//!
//! The [`Value`] enum represents data flowing through graph wires.
//! Type checking happens at graph construction time via [`ValueType`].
//!
//! # Example
//!
//! ```
//! use unshape_core::{Value, ValueType};
//! use glam::Vec3;
//!
//! // Create values
//! let f = Value::F32(1.5);
//! let v = Value::Vec3(Vec3::new(1.0, 2.0, 3.0));
//!
//! // Check types
//! assert_eq!(f.value_type(), ValueType::F32);
//!
//! // Extract with type safety
//! let x: f32 = f.as_f32().unwrap();
//! let vec: Vec3 = v.as_vec3().unwrap();
//!
//! // From conversions for convenience
//! let f: Value = 3.14f32.into();
//! let v: Value = Vec3::X.into();
//! ```

use glam::{Vec2, Vec3, Vec4};
use std::any::{Any, TypeId};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::error::TypeError;

/// Where data currently resides.
///
/// Used for scheduling decisions — prefer backends that match data location
/// to minimize transfers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum DataLocation {
    /// Data is in CPU memory (heap).
    #[default]
    Cpu,
    /// Data is in GPU memory.
    Gpu {
        /// Device identifier (for multi-GPU systems).
        device_id: u32,
    },
}

/// Trait for complex values that can flow through graphs.
///
/// Implement this for types like `Image`, `Mesh`, `GpuTexture`, `AudioBuffer`, etc.
/// that need to be passed through graph wires but are too large or complex
/// for the fixed `Value` variants.
///
/// # Example
///
/// ```ignore
/// use unshape_core::{GraphValue, DataLocation};
/// use std::any::Any;
///
/// pub struct Image {
///     pixels: Vec<[f32; 4]>,
///     width: u32,
///     height: u32,
/// }
///
/// impl GraphValue for Image {
///     fn as_any(&self) -> &dyn Any { self }
///     fn type_name(&self) -> &'static str { "Image" }
/// }
/// ```
pub trait GraphValue: Any + Send + Sync + std::fmt::Debug {
    /// Returns self as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns a human-readable type name for debugging/display.
    fn type_name(&self) -> &'static str;

    /// Where this value currently resides.
    ///
    /// Override for GPU-resident types to return `DataLocation::Gpu`.
    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }

    /// Returns a stable identifier for this value's type.
    ///
    /// Used for hashing and equality. Override if your type has
    /// meaningful content-based identity.
    fn stable_id(&self) -> u64 {
        // Default: use pointer address (identity-based)
        self.as_any() as *const dyn Any as *const () as u64
    }
}

/// Runtime value type for dynamic graph execution.
///
/// This enum represents all possible values that can flow through a graph.
/// Type safety is enforced at graph construction time (via typed slots) or
/// at load time (via TypeId validation). At execution time, we trust the
/// graph is valid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Opaque value for complex/large types.
    ///
    /// Use for `Image`, `Mesh`, `GpuTexture`, `AudioBuffer`, etc.
    /// Stored as `Arc` for cheap cloning. Use [`Value::opaque`] to create
    /// and [`Value::downcast_ref`] to extract.
    #[cfg_attr(feature = "serde", serde(skip))]
    Opaque(Arc<dyn GraphValue>),
}

/// Type identifier for values in the graph system.
///
/// Note: The `Custom` variant cannot be serialized (TypeId is not serializable).
/// Use concrete type registration for serialization of custom types.
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
    /// Custom/opaque type (for Image, Mesh, GpuTexture, etc.)
    ///
    /// This variant cannot be serialized directly. For serialization,
    /// register custom types with a type registry that maps names to types.
    Custom {
        /// Rust TypeId for the concrete type.
        type_id: TypeId,
        /// Human-readable name for display/debugging.
        name: &'static str,
    },
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
            Value::Opaque(v) => ValueType::Custom {
                type_id: v.as_any().type_id(),
                name: v.type_name(),
            },
        }
    }

    /// Creates an opaque value from any type implementing [`GraphValue`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let image = Image::new(512, 512);
    /// let value = Value::opaque(image);
    /// ```
    pub fn opaque<T: GraphValue>(value: T) -> Self {
        Value::Opaque(Arc::new(value))
    }

    /// Creates an opaque value from an existing `Arc<dyn GraphValue>`.
    pub fn from_arc(value: Arc<dyn GraphValue>) -> Self {
        Value::Opaque(value)
    }

    /// Attempts to downcast an opaque value to a concrete type.
    ///
    /// Returns `None` if this is not an `Opaque` variant or if the
    /// concrete type doesn't match.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(image) = value.downcast_ref::<Image>() {
    ///     println!("Image size: {}x{}", image.width, image.height);
    /// }
    /// ```
    pub fn downcast_ref<T: GraphValue + 'static>(&self) -> Option<&T> {
        match self {
            Value::Opaque(v) => v.as_any().downcast_ref(),
            _ => None,
        }
    }

    /// Returns the data location of this value.
    ///
    /// Primitives are always on CPU. Opaque values report their location
    /// via [`GraphValue::location`].
    pub fn location(&self) -> DataLocation {
        match self {
            Value::Opaque(v) => v.location(),
            _ => DataLocation::Cpu,
        }
    }

    /// Returns `true` if this is an opaque value.
    pub fn is_opaque(&self) -> bool {
        matches!(self, Value::Opaque(_))
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
            ValueType::Custom { name, .. } => write!(f, "{}", name),
        }
    }
}

impl ValueType {
    /// Creates a `Custom` value type for a concrete type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let image_type = ValueType::of::<Image>("Image");
    /// ```
    pub fn of<T: 'static>(name: &'static str) -> Self {
        ValueType::Custom {
            type_id: TypeId::of::<T>(),
            name,
        }
    }

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
            ValueType::Custom { type_id, .. } => *type_id,
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

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Discriminant first for type safety
        std::mem::discriminant(self).hash(state);
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::F64(v) => v.to_bits().hash(state),
            Value::I32(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
            Value::Vec2(v) => {
                v.x.to_bits().hash(state);
                v.y.to_bits().hash(state);
            }
            Value::Vec3(v) => {
                v.x.to_bits().hash(state);
                v.y.to_bits().hash(state);
                v.z.to_bits().hash(state);
            }
            Value::Vec4(v) => {
                v.x.to_bits().hash(state);
                v.y.to_bits().hash(state);
                v.z.to_bits().hash(state);
                v.w.to_bits().hash(state);
            }
            Value::Opaque(v) => {
                // Hash by type + stable_id (default: pointer identity)
                v.as_any().type_id().hash(state);
                v.stable_id().hash(state);
            }
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::F32(a), Value::F32(b)) => a.to_bits() == b.to_bits(),
            (Value::F64(a), Value::F64(b)) => a.to_bits() == b.to_bits(),
            (Value::I32(a), Value::I32(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Vec2(a), Value::Vec2(b)) => {
                a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
            }
            (Value::Vec3(a), Value::Vec3(b)) => {
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.z.to_bits() == b.z.to_bits()
            }
            (Value::Vec4(a), Value::Vec4(b)) => {
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.z.to_bits() == b.z.to_bits()
                    && a.w.to_bits() == b.w.to_bits()
            }
            (Value::Opaque(a), Value::Opaque(b)) => {
                // Equal if same type and same stable_id
                a.as_any().type_id() == b.as_any().type_id() && a.stable_id() == b.stable_id()
            }
            _ => false,
        }
    }
}

impl Eq for Value {}

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

    #[test]
    fn test_value_hash_eq() {
        use std::collections::hash_map::DefaultHasher;

        fn hash(v: &Value) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        // Same values should hash equally
        assert_eq!(hash(&Value::F32(1.0)), hash(&Value::F32(1.0)));
        assert_eq!(hash(&Value::I32(42)), hash(&Value::I32(42)));
        assert_eq!(
            hash(&Value::Vec3(Vec3::new(1.0, 2.0, 3.0))),
            hash(&Value::Vec3(Vec3::new(1.0, 2.0, 3.0)))
        );

        // Different values should (usually) hash differently
        assert_ne!(hash(&Value::F32(1.0)), hash(&Value::F32(2.0)));
        assert_ne!(hash(&Value::F32(1.0)), hash(&Value::I32(1)));

        // PartialEq consistency
        assert_eq!(Value::F32(1.0), Value::F32(1.0));
        assert_ne!(Value::F32(1.0), Value::F32(2.0));
        assert_ne!(Value::F32(1.0), Value::I32(1));
    }

    #[test]
    fn test_value_hash_nan() {
        use std::collections::hash_map::DefaultHasher;

        fn hash(v: &Value) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        // NaN with same bit pattern should hash consistently
        let nan1 = Value::F32(f32::NAN);
        let nan2 = Value::F32(f32::NAN);
        assert_eq!(hash(&nan1), hash(&nan2));
        // Using to_bits means same bit pattern = equal
        assert_eq!(nan1, nan2);
    }

    #[test]
    fn test_value_usable_as_map_key() {
        use std::collections::HashMap;

        let mut map: HashMap<Value, &str> = HashMap::new();
        map.insert(Value::F32(1.0), "one");
        map.insert(Value::I32(2), "two");
        map.insert(Value::Vec3(Vec3::X), "x-axis");

        assert_eq!(map.get(&Value::F32(1.0)), Some(&"one"));
        assert_eq!(map.get(&Value::I32(2)), Some(&"two"));
        assert_eq!(map.get(&Value::Vec3(Vec3::X)), Some(&"x-axis"));
        assert_eq!(map.get(&Value::F32(2.0)), None);
    }

    // === Opaque value tests ===

    /// Test type for opaque values
    #[derive(Debug, Clone)]
    struct TestImage {
        width: u32,
        height: u32,
        data: Vec<f32>,
    }

    impl GraphValue for TestImage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn type_name(&self) -> &'static str {
            "TestImage"
        }
    }

    /// Test type with custom location
    #[derive(Debug)]
    struct GpuBuffer {
        size: usize,
        device_id: u32,
    }

    impl GraphValue for GpuBuffer {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn type_name(&self) -> &'static str {
            "GpuBuffer"
        }
        fn location(&self) -> DataLocation {
            DataLocation::Gpu {
                device_id: self.device_id,
            }
        }
    }

    #[test]
    fn test_opaque_value_creation() {
        let image = TestImage {
            width: 512,
            height: 512,
            data: vec![0.0; 512 * 512],
        };
        let value = Value::opaque(image);
        assert!(value.is_opaque());
    }

    #[test]
    fn test_opaque_downcast() {
        let image = TestImage {
            width: 256,
            height: 128,
            data: vec![1.0; 256 * 128],
        };
        let value = Value::opaque(image);

        // Successful downcast
        let extracted = value.downcast_ref::<TestImage>().unwrap();
        assert_eq!(extracted.width, 256);
        assert_eq!(extracted.height, 128);

        // Failed downcast (wrong type)
        assert!(value.downcast_ref::<GpuBuffer>().is_none());

        // Downcast on non-opaque value
        let f = Value::F32(1.0);
        assert!(f.downcast_ref::<TestImage>().is_none());
    }

    #[test]
    fn test_opaque_value_type() {
        let image = TestImage {
            width: 64,
            height: 64,
            data: vec![],
        };
        let value = Value::opaque(image);
        let vt = value.value_type();

        match vt {
            ValueType::Custom { name, type_id } => {
                assert_eq!(name, "TestImage");
                assert_eq!(type_id, TypeId::of::<TestImage>());
            }
            _ => panic!("Expected Custom variant"),
        }

        // Display works
        assert_eq!(format!("{}", vt), "TestImage");
    }

    #[test]
    fn test_opaque_location_cpu() {
        let image = TestImage {
            width: 32,
            height: 32,
            data: vec![],
        };
        let value = Value::opaque(image);
        assert_eq!(value.location(), DataLocation::Cpu);
    }

    #[test]
    fn test_opaque_location_gpu() {
        let buffer = GpuBuffer {
            size: 1024,
            device_id: 0,
        };
        let value = Value::opaque(buffer);
        assert_eq!(value.location(), DataLocation::Gpu { device_id: 0 });
    }

    #[test]
    fn test_primitive_location() {
        // All primitives should be on CPU
        assert_eq!(Value::F32(1.0).location(), DataLocation::Cpu);
        assert_eq!(Value::I32(42).location(), DataLocation::Cpu);
        assert_eq!(Value::Vec3(Vec3::ONE).location(), DataLocation::Cpu);
    }

    #[test]
    fn test_opaque_clone() {
        let image = TestImage {
            width: 100,
            height: 100,
            data: vec![0.5; 100 * 100],
        };
        let value1 = Value::opaque(image);
        let value2 = value1.clone();

        // Both should reference the same Arc
        let img1 = value1.downcast_ref::<TestImage>().unwrap();
        let img2 = value2.downcast_ref::<TestImage>().unwrap();
        assert_eq!(img1.width, img2.width);
        assert!(std::ptr::eq(img1, img2)); // Same pointer (Arc clone)
    }

    #[test]
    fn test_opaque_hash_eq() {
        use std::collections::hash_map::DefaultHasher;

        fn hash(v: &Value) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        let image = TestImage {
            width: 10,
            height: 10,
            data: vec![],
        };
        let value1 = Value::opaque(image);
        let value2 = value1.clone();

        // Cloned opaque values (same Arc) should be equal and hash same
        assert_eq!(value1, value2);
        assert_eq!(hash(&value1), hash(&value2));

        // Different opaque values should not be equal (identity-based)
        let image2 = TestImage {
            width: 10,
            height: 10,
            data: vec![],
        };
        let value3 = Value::opaque(image2);
        assert_ne!(value1, value3);
    }

    #[test]
    fn test_value_type_of() {
        let vt = ValueType::of::<TestImage>("TestImage");
        assert_eq!(vt.type_id(), TypeId::of::<TestImage>());
        assert_eq!(format!("{}", vt), "TestImage");
    }

    #[test]
    fn test_data_location_default() {
        let loc: DataLocation = Default::default();
        assert_eq!(loc, DataLocation::Cpu);
    }
}
