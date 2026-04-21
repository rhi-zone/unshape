//! `GraphOutput` — a named terminal sink that captures a value from the graph.

use std::any::Any;

use crate::error::GraphError;
use crate::eval::EvalContext;
use crate::node::{DynNode, PortDescriptor};
use crate::value::{Value, ValueType};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A sink node that receives a named output value from the graph.
///
/// Has one input port named `"value"` typed `self.value_type`, and zero output
/// ports. The `name` identifies this output in the graph's interface.
///
/// # Execution behaviour
///
/// 1. Receive `inputs[0]`.
/// 2. Validate its type matches `self.value_type`; if not, return
///    [`GraphError::InputTypeMismatch`].
/// 3. Return the value as a passthrough so callers can retrieve it.
///
/// # Passthrough output
///
/// Although `GraphOutput` is logically a sink, `execute` returns the value
/// as `Ok(vec![inputs[0].clone()])`. This allows [`Graph::execute_named_outputs`]
/// to call `execute_with_context` on each `GraphOutput` node and collect
/// `(name, result[0])` without needing special plumbing.
///
/// # Serialization
///
/// Serialization is supported for nodes whose `value_type` is a primitive
/// variant. `ValueType::Custom` cannot be round-tripped via serde.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphOutput {
    /// The host-facing name identifying this output in the graph interface.
    pub name: String,
    /// The expected type of the incoming value.
    pub value_type: ValueType,
}

impl GraphOutput {
    /// Create a new graph output node.
    pub fn new(name: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            name: name.into(),
            value_type,
        }
    }
}

impl DynNode for GraphOutput {
    fn type_name(&self) -> &'static str {
        "core::GraphOutput"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("value", self.value_type)]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let value = inputs[0].clone();

        if value.value_type() != self.value_type {
            return Err(GraphError::InputTypeMismatch {
                name: self.name.clone(),
                expected: self.value_type,
                got: value.value_type(),
            });
        }

        // Passthrough: return the value so execute_named_outputs can collect it.
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
    fn graph_output_happy_path() {
        let node = GraphOutput::new("result", ValueType::F32);
        let ctx = EvalContext::new();
        let outputs = node.execute(&[Value::F32(1.5)], &ctx).unwrap();
        assert_eq!(outputs, vec![Value::F32(1.5)]);
    }

    #[test]
    fn graph_output_type_mismatch() {
        let node = GraphOutput::new("result", ValueType::F32);
        let ctx = EvalContext::new();
        let err = node.execute(&[Value::I32(42)], &ctx).unwrap_err();
        assert!(
            matches!(err, GraphError::InputTypeMismatch { name, expected, got }
                if name == "result" && expected == ValueType::F32 && got == ValueType::I32)
        );
    }

    #[test]
    fn graph_output_input_port() {
        let node = GraphOutput::new("out", ValueType::Vec3);
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "value");
        assert_eq!(inputs[0].value_type, ValueType::Vec3);
    }

    #[test]
    fn graph_output_no_output_ports() {
        let node = GraphOutput::new("out", ValueType::Bool);
        assert!(node.outputs().is_empty());
    }

    #[test]
    fn graph_output_type_name() {
        let node = GraphOutput::new("out", ValueType::I32);
        assert_eq!(node.type_name(), "core::GraphOutput");
    }
}
