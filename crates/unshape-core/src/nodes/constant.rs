//! `ConstantNode` — an authored value embedded in the graph.

use std::any::Any;

use crate::error::GraphError;
use crate::eval::EvalContext;
use crate::node::{DynNode, PortDescriptor};
use crate::value::Value;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A source node that outputs a single constant value embedded in the graph.
///
/// Has zero inputs and one output port named `"value"`. The output type
/// matches the runtime type of the stored value.
///
/// # Serialization
///
/// Serialization is supported only for primitive `Value` variants (`F32`, `F64`,
/// `I32`, `Bool`, `Vec2`, `Vec3`, `Vec4`). Attempting to serialize a
/// `ConstantNode` whose value is `Value::Opaque` will return an error because
/// `TypeId` cannot be serialized.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConstantNode {
    /// The value this node outputs.
    pub value: Value,
}

impl ConstantNode {
    /// Create a new constant node wrapping the given value.
    pub fn new(value: impl Into<Value>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl DynNode for ConstantNode {
    fn type_name(&self) -> &'static str {
        "core::Constant"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("value", self.value.value_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![self.value.clone()])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ValueType;

    #[test]
    fn constant_node_f32_execute() {
        let node = ConstantNode::new(42.0f32);
        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs, vec![Value::F32(42.0)]);
    }

    #[test]
    fn constant_node_output_port() {
        let node = ConstantNode::new(42.0f32);
        let outputs = node.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "value");
        assert_eq!(outputs[0].value_type, ValueType::F32);
    }

    #[test]
    fn constant_node_no_input_ports() {
        let node = ConstantNode::new(1i32);
        assert!(node.inputs().is_empty());
    }

    #[test]
    fn constant_node_type_name() {
        let node = ConstantNode::new(true);
        assert_eq!(node.type_name(), "core::Constant");
    }

    #[test]
    fn constant_node_i32() {
        let node = ConstantNode::new(7i32);
        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs, vec![Value::I32(7)]);
    }
}
