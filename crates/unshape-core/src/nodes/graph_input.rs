//! `GraphInput` — reads a named value from `EvalContext` at execution time.

use std::any::Any;

use crate::error::GraphError;
use crate::eval::EvalContext;
use crate::node::{DynNode, PortDescriptor};
use crate::value::{Value, ValueType};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A source node that reads a named value injected by the host via [`EvalContext`].
///
/// Has zero inputs and one output port named `"value"`. The declared `value_type`
/// controls what type the host is expected to provide.
///
/// # Execution behaviour
///
/// 1. Look up `self.name` in `ctx.input()`.
/// 2. If not found, fall back to `self.default`.
/// 3. If still not found, return [`GraphError::MissingInput`].
/// 4. Verify the resolved value's type matches `self.value_type`; if not, return
///    [`GraphError::InputTypeMismatch`].
/// 5. Return the value.
///
/// # Serialization
///
/// Serialization is supported for nodes whose `value_type` is a primitive variant
/// (`F32`, `F64`, `I32`, `Bool`, `Vec2`, `Vec3`, `Vec4`). `ValueType::Custom`
/// cannot be round-tripped via serde.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphInput {
    /// The host-facing name used to look up the value in `EvalContext`.
    pub name: String,
    /// The expected type of the provided value.
    pub value_type: ValueType,
    /// Optional fallback when the host does not provide this input.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub default: Option<Value>,
}

impl GraphInput {
    /// Create a new graph input node.
    pub fn new(name: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            name: name.into(),
            value_type,
            default: None,
        }
    }

    /// Set the default value to use when the host does not provide this input.
    pub fn with_default(mut self, default: impl Into<Value>) -> Self {
        self.default = Some(default.into());
        self
    }
}

impl DynNode for GraphInput {
    fn type_name(&self) -> &'static str {
        "core::GraphInput"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("value", self.value_type)]
    }

    fn execute(&self, _inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let value = ctx
            .input(&self.name)
            .cloned()
            .or_else(|| self.default.clone())
            .ok_or_else(|| GraphError::MissingInput {
                name: self.name.clone(),
            })?;

        if value.value_type() != self.value_type {
            return Err(GraphError::InputTypeMismatch {
                name: self.name.clone(),
                expected: self.value_type,
                got: value.value_type(),
            });
        }

        Ok(vec![value])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_input_happy_path() {
        let node = GraphInput::new("x", ValueType::F32);
        let ctx = EvalContext::new().with_input("x", 1.5f32);
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs, vec![Value::F32(1.5)]);
    }

    #[test]
    fn graph_input_missing_no_default() {
        let node = GraphInput::new("x", ValueType::F32);
        let ctx = EvalContext::new();
        let err = node.execute(&[], &ctx).unwrap_err();
        assert!(matches!(err, GraphError::MissingInput { name } if name == "x"));
    }

    #[test]
    fn graph_input_missing_with_default() {
        let node = GraphInput::new("x", ValueType::F32).with_default(1.0f32);
        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs, vec![Value::F32(1.0)]);
    }

    #[test]
    fn graph_input_type_mismatch() {
        let node = GraphInput::new("x", ValueType::F32);
        let ctx = EvalContext::new().with_input("x", 42i32);
        let err = node.execute(&[], &ctx).unwrap_err();
        assert!(
            matches!(err, GraphError::InputTypeMismatch { name, expected, got }
                if name == "x" && expected == ValueType::F32 && got == ValueType::I32)
        );
    }

    #[test]
    fn graph_input_output_port() {
        let node = GraphInput::new("y", ValueType::Vec3);
        let outputs = node.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "value");
        assert_eq!(outputs[0].value_type, ValueType::Vec3);
    }

    #[test]
    fn graph_input_no_input_ports() {
        let node = GraphInput::new("z", ValueType::Bool);
        assert!(node.inputs().is_empty());
    }

    #[test]
    fn graph_input_type_name() {
        let node = GraphInput::new("w", ValueType::I32);
        assert_eq!(node.type_name(), "core::GraphInput");
    }
}
