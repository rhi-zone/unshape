//! Dynamic operation infrastructure for resin pipelines.
//!
//! This crate provides runtime execution of serialized operation pipelines
//! with type validation. It's the foundation for:
//!
//! - Loading and executing saved pipelines
//! - Type checking at load time rather than execution time
//! - Recording and replaying operation history
//!
//! # Using the Derive Macro
//!
//! Use `#[derive(Op)]` to auto-generate `DynOp` implementations:
//!
//! ```ignore
//! use resin_op::Op;
//!
//! #[derive(Clone, Serialize, Deserialize, Op)]
//! #[op(input = Mesh, output = Mesh)]
//! pub struct Subdivide { pub levels: u32 }
//!
//! impl Subdivide {
//!     pub fn apply(&self, mesh: &Mesh) -> Mesh { ... }
//! }
//! ```
//!
//! # Pipelines
//!
//! Operations can be chained into pipelines that are validated and executed:
//!
//! ```ignore
//! let mut pipeline = Pipeline::new();
//! pipeline.push(&Subdivide { levels: 2 });
//! pipeline.push(&Smooth::new(0.5, 3));
//!
//! // Validate types match
//! let (input_type, output_type) = pipeline.validate(&registry)?;
//!
//! // Execute
//! let result = pipeline.execute(input_mesh, &registry)?;
//! ```

// Re-export the derive macro
pub use unshape_op_macros::Op;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Type identifier for operation inputs/outputs.
///
/// Uses `TypeId` for runtime identification and a string name for
/// serialization/debugging. This allows any type to be used in pipelines,
/// not just a fixed set.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OpType {
    /// Rust's TypeId for runtime type checking.
    pub type_id: TypeId,
    /// Human-readable name for serialization/debugging.
    pub name: &'static str,
}

impl OpType {
    /// Creates an OpType from a concrete type.
    pub fn of<T: 'static>(name: &'static str) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name,
        }
    }
}

impl fmt::Debug for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpType({})", self.name)
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Runtime value for operation pipelines.
///
/// Type-erased container for domain objects. Values are boxed to avoid
/// depending on concrete types in this crate.
pub struct OpValue {
    /// The type of this value.
    pub op_type: OpType,
    /// The actual value, type-erased.
    value: Box<dyn Any + Send + Sync>,
}

impl OpValue {
    /// Creates a new OpValue with an explicit type.
    pub fn new<T: Any + Send + Sync>(op_type: OpType, value: T) -> Self {
        Self {
            op_type,
            value: Box::new(value),
        }
    }

    /// Creates a new OpValue, inferring the OpType from the value.
    pub fn from<T: Any + Send + Sync>(type_name: &'static str, value: T) -> Self {
        Self {
            op_type: OpType::of::<T>(type_name),
            value: Box::new(value),
        }
    }

    /// Attempts to downcast to a concrete type, consuming self.
    pub fn downcast<T: Any>(self) -> Result<T, OpError> {
        self.value
            .downcast::<T>()
            .map(|b| *b)
            .map_err(|_| OpError::DowncastFailed {
                expected: std::any::type_name::<T>(),
            })
    }

    /// Attempts to downcast to a reference.
    pub fn downcast_ref<T: Any>(&self) -> Result<&T, OpError> {
        self.value
            .downcast_ref::<T>()
            .ok_or_else(|| OpError::DowncastFailed {
                expected: std::any::type_name::<T>(),
            })
    }

    /// Returns the inner value as a Box<dyn Any>.
    pub fn into_inner(self) -> Box<dyn Any + Send + Sync> {
        self.value
    }
}

impl fmt::Debug for OpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpValue")
            .field("op_type", &self.op_type)
            .finish_non_exhaustive()
    }
}

/// Errors from dynamic operation execution.
#[derive(Debug, Error)]
pub enum OpError {
    /// Type mismatch between expected and actual.
    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: OpType, got: OpType },

    /// Failed to downcast value to concrete type.
    #[error("failed to downcast to {expected}")]
    DowncastFailed { expected: &'static str },

    /// Unknown operation type.
    #[error("unknown operation type: {0}")]
    UnknownType(String),

    /// Deserialization failed.
    #[error("deserialization failed: {0}")]
    DeserializationFailed(String),

    /// Operation-specific error.
    #[error("operation failed: {0}")]
    ExecutionError(String),
}

/// Trait for operations that can be executed dynamically.
///
/// This is the runtime interface for operation pipelines. Op authors
/// write concrete `apply(&self, input: &T) -> T` methods; the `DynOp`
/// impl handles OpValue wrapping/unwrapping.
pub trait DynOp: Send + Sync {
    /// Returns the type name for serialization/debugging.
    fn type_name(&self) -> &'static str;

    /// Returns the expected input type.
    fn input_type(&self) -> OpType;

    /// Returns the output type.
    fn output_type(&self) -> OpType;

    /// Executes the operation with a dynamic value.
    fn apply_dyn(&self, input: OpValue) -> Result<OpValue, OpError>;

    /// Serializes the operation parameters to JSON.
    fn params(&self) -> serde_json::Value;
}

/// A boxed dynamic operation.
pub type BoxedOp = Box<dyn DynOp>;

/// Factory function for creating operations from JSON params.
type OpFactory = Box<dyn Fn(serde_json::Value) -> Result<BoxedOp, OpError> + Send + Sync>;

/// Registry for deserializing operations by type name.
pub struct OpRegistry {
    factories: HashMap<String, OpFactory>,
}

impl OpRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Registers a factory function for a type name.
    pub fn register<F>(&mut self, type_name: &str, factory: F)
    where
        F: Fn(serde_json::Value) -> Result<BoxedOp, OpError> + Send + Sync + 'static,
    {
        self.factories
            .insert(type_name.to_string(), Box::new(factory));
    }

    /// Registers an op type that implements serde Deserialize.
    ///
    /// This is a convenience for ops that can be deserialized directly.
    pub fn register_type<T>(&mut self, type_name: &str)
    where
        T: DynOp + for<'de> Deserialize<'de> + 'static,
    {
        self.register(type_name, |params| {
            let op: T = serde_json::from_value(params)
                .map_err(|e| OpError::DeserializationFailed(e.to_string()))?;
            Ok(Box::new(op))
        });
    }

    /// Deserializes an operation from type name and params.
    pub fn deserialize(
        &self,
        type_name: &str,
        params: serde_json::Value,
    ) -> Result<BoxedOp, OpError> {
        let factory = self
            .factories
            .get(type_name)
            .ok_or_else(|| OpError::UnknownType(type_name.to_string()))?;
        factory(params)
    }

    /// Returns true if a type name is registered.
    pub fn contains(&self, type_name: &str) -> bool {
        self.factories.contains_key(type_name)
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A serialized operation (type name + params).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialOp {
    /// Type name for registry lookup.
    pub type_name: String,
    /// Serialized parameters.
    pub params: serde_json::Value,
}

impl SerialOp {
    /// Creates a new SerialOp.
    pub fn new(type_name: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            type_name: type_name.into(),
            params,
        }
    }

    /// Creates from a DynOp.
    pub fn from_op(op: &dyn DynOp) -> Self {
        Self {
            type_name: op.type_name().to_string(),
            params: op.params(),
        }
    }
}

/// A pipeline of operations.
///
/// Pipelines are linear chains of operations. Each operation's output
/// type must match the next operation's input type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Pipeline {
    /// Operations in execution order.
    pub ops: Vec<SerialOp>,
}

impl Pipeline {
    /// Creates an empty pipeline.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Adds an operation to the pipeline.
    pub fn push(&mut self, op: &dyn DynOp) {
        self.ops.push(SerialOp::from_op(op));
    }

    /// Adds a serialized operation to the pipeline.
    pub fn push_serial(&mut self, op: SerialOp) {
        self.ops.push(op);
    }

    /// Returns true if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Returns the number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Validates type compatibility of the pipeline.
    ///
    /// Returns `(input_type, output_type)` if valid.
    pub fn validate(&self, registry: &OpRegistry) -> Result<(OpType, OpType), OpError> {
        if self.ops.is_empty() {
            return Err(OpError::ExecutionError("empty pipeline".to_string()));
        }

        let ops: Vec<BoxedOp> = self
            .ops
            .iter()
            .map(|s| registry.deserialize(&s.type_name, s.params.clone()))
            .collect::<Result<_, _>>()?;

        let input_type = ops[0].input_type();
        let mut current = ops[0].output_type();

        for op in ops.iter().skip(1) {
            if op.input_type() != current {
                return Err(OpError::TypeMismatch {
                    expected: op.input_type(),
                    got: current,
                });
            }
            current = op.output_type();
        }

        Ok((input_type, current))
    }

    /// Executes the pipeline on an input value.
    pub fn execute(&self, input: OpValue, registry: &OpRegistry) -> Result<OpValue, OpError> {
        let mut value = input;

        for serial in &self.ops {
            let op = registry.deserialize(&serial.type_name, serial.params.clone())?;

            if op.input_type() != value.op_type {
                return Err(OpError::TypeMismatch {
                    expected: op.input_type(),
                    got: value.op_type,
                });
            }

            value = op.apply_dyn(value)?;
        }

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Marker type for "Image" in tests
    struct TestImage;

    fn float_type() -> OpType {
        OpType::of::<f32>("f32")
    }

    fn image_type() -> OpType {
        OpType::of::<TestImage>("Image")
    }

    // Test op that doubles a float
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct DoubleOp;

    impl DynOp for DoubleOp {
        fn type_name(&self) -> &'static str {
            "test::Double"
        }

        fn input_type(&self) -> OpType {
            float_type()
        }

        fn output_type(&self) -> OpType {
            float_type()
        }

        fn apply_dyn(&self, input: OpValue) -> Result<OpValue, OpError> {
            let value: f32 = input.downcast()?;
            Ok(OpValue::from("f32", value * 2.0))
        }

        fn params(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    // Test op with parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct AddOp {
        amount: f32,
    }

    impl DynOp for AddOp {
        fn type_name(&self) -> &'static str {
            "test::Add"
        }

        fn input_type(&self) -> OpType {
            float_type()
        }

        fn output_type(&self) -> OpType {
            float_type()
        }

        fn apply_dyn(&self, input: OpValue) -> Result<OpValue, OpError> {
            let value: f32 = input.downcast()?;
            Ok(OpValue::from("f32", value + self.amount))
        }

        fn params(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }

    // Type-changing op for testing validation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ToImageOp;

    impl DynOp for ToImageOp {
        fn type_name(&self) -> &'static str {
            "test::ToImage"
        }

        fn input_type(&self) -> OpType {
            float_type()
        }

        fn output_type(&self) -> OpType {
            image_type()
        }

        fn apply_dyn(&self, input: OpValue) -> Result<OpValue, OpError> {
            // Just pass through for testing
            Ok(OpValue::new(image_type(), input.into_inner()))
        }

        fn params(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    fn test_registry() -> OpRegistry {
        let mut registry = OpRegistry::new();
        // Unit structs need custom factories since serde can't deserialize {} to them
        registry.register("test::Double", |_| Ok(Box::new(DoubleOp)));
        registry.register_type::<AddOp>("test::Add");
        registry.register("test::ToImage", |_| Ok(Box::new(ToImageOp)));
        registry
    }

    #[test]
    fn test_op_value_downcast() {
        let value = OpValue::from("f32", 42.0f32);
        assert_eq!(value.op_type.name, "f32");

        let extracted: f32 = value.downcast().unwrap();
        assert_eq!(extracted, 42.0);
    }

    #[test]
    fn test_op_value_downcast_ref() {
        let value = OpValue::from("f32", 42.0f32);
        let extracted: &f32 = value.downcast_ref().unwrap();
        assert_eq!(*extracted, 42.0);
    }

    #[test]
    fn test_op_value_downcast_fail() {
        let value = OpValue::from("f32", 42.0f32);
        let result: Result<i32, _> = value.downcast();
        assert!(result.is_err());
    }

    #[test]
    fn test_single_op_execution() {
        let op = DoubleOp;
        let input = OpValue::from("f32", 5.0f32);
        let output = op.apply_dyn(input).unwrap();
        let result: f32 = output.downcast().unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_registry_contains() {
        let registry = test_registry();
        assert!(registry.contains("test::Double"));
        assert!(!registry.contains("test::Unknown"));
    }

    #[test]
    fn test_registry_deserialize() {
        let registry = test_registry();

        let op = registry
            .deserialize("test::Add", serde_json::json!({"amount": 5.0}))
            .unwrap();

        assert_eq!(op.type_name(), "test::Add");
        assert_eq!(op.input_type().name, "f32");
    }

    #[test]
    fn test_pipeline_validate_success() {
        let registry = test_registry();

        let mut pipeline = Pipeline::new();
        pipeline.push(&DoubleOp);
        pipeline.push(&AddOp { amount: 3.0 });

        let (input, output) = pipeline.validate(&registry).unwrap();
        assert_eq!(input.name, "f32");
        assert_eq!(output.name, "f32");
    }

    #[test]
    fn test_pipeline_validate_type_mismatch() {
        let registry = test_registry();

        let mut pipeline = Pipeline::new();
        pipeline.push(&ToImageOp); // Float -> Image
        pipeline.push(&DoubleOp); // Expects Float, but gets Image

        let result = pipeline.validate(&registry);
        assert!(matches!(result, Err(OpError::TypeMismatch { .. })));
    }

    #[test]
    fn test_pipeline_execute() {
        let registry = test_registry();

        let mut pipeline = Pipeline::new();
        pipeline.push(&DoubleOp);
        pipeline.push(&AddOp { amount: 3.0 });

        let input = OpValue::from("f32", 5.0f32);
        let output = pipeline.execute(input, &registry).unwrap();
        let result: f32 = output.downcast().unwrap();
        // 5 * 2 + 3 = 13
        assert_eq!(result, 13.0);
    }

    #[test]
    fn test_pipeline_serialize() {
        let mut pipeline = Pipeline::new();
        pipeline.push(&DoubleOp);
        pipeline.push(&AddOp { amount: 3.0 });

        let json = serde_json::to_string_pretty(&pipeline).unwrap();
        assert!(json.contains("test::Double"));
        assert!(json.contains("test::Add"));

        let loaded: Pipeline = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.ops.len(), 2);
    }

    #[test]
    fn test_op_type_display() {
        let mesh_type = OpType::of::<()>("Mesh");
        assert_eq!(format!("{}", mesh_type), "Mesh");
    }

    #[test]
    fn test_serial_op_from_op() {
        let op = AddOp { amount: 5.0 };
        let serial = SerialOp::from_op(&op);

        assert_eq!(serial.type_name, "test::Add");
        assert_eq!(serial.params["amount"], 5.0);
    }
}
