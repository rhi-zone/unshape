//! Node trait for graph execution.

use crate::error::GraphError;
use crate::eval::EvalContext;
use crate::value::{Value, ValueType};
use std::any::Any;

/// Port descriptor for a node input or output.
#[derive(Debug, Clone)]
pub struct PortDescriptor {
    /// Port name for display/debugging.
    pub name: &'static str,
    /// Type of values this port accepts/produces.
    pub value_type: ValueType,
}

impl PortDescriptor {
    /// Create a new port descriptor.
    pub fn new(name: &'static str, value_type: ValueType) -> Self {
        Self { name, value_type }
    }
}

/// Trait for nodes that can be executed dynamically.
///
/// This is the runtime interface for graph execution. Node authors
/// typically don't implement this directly - instead they use derive
/// macros that generate the implementation from concrete types.
pub trait DynNode: Send + Sync + Any {
    /// Returns the type name for serialization/debugging.
    fn type_name(&self) -> &'static str;

    /// Returns descriptors for all input ports.
    fn inputs(&self) -> Vec<PortDescriptor>;

    /// Returns descriptors for all output ports.
    fn outputs(&self) -> Vec<PortDescriptor>;

    /// Executes the node with the given inputs and evaluation context.
    ///
    /// # Arguments
    /// * `inputs` - Input values, one per input port in order.
    /// * `ctx` - Evaluation context (time, cancellation, quality hints, etc.)
    ///
    /// # Returns
    /// Output values, one per output port in order.
    ///
    /// # Cancellation
    /// Long-running nodes should periodically check `ctx.is_cancelled()` and
    /// return `Err(GraphError::Cancelled)` if true.
    fn execute(&self, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError>;

    /// Returns `self` as `&dyn Any` for downcasting and type identification.
    ///
    /// Used by compute backends to look up registered kernels by node type.
    fn as_any(&self) -> &dyn Any;
}

/// A boxed dynamic node.
pub type BoxedNode = Box<dyn DynNode>;

/// Helper macro for creating port descriptors.
#[macro_export]
macro_rules! ports {
    ($($name:literal : $ty:ident),* $(,)?) => {
        &[
            $(PortDescriptor::new($name, ValueType::$ty)),*
        ]
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_descriptor_new() {
        let port = PortDescriptor::new("input", ValueType::F32);
        assert_eq!(port.name, "input");
        assert_eq!(port.value_type, ValueType::F32);
    }

    #[test]
    fn test_port_descriptor_clone() {
        let port = PortDescriptor::new("output", ValueType::Vec3);
        let cloned = port.clone();
        assert_eq!(cloned.name, "output");
        assert_eq!(cloned.value_type, ValueType::Vec3);
    }

    #[test]
    fn test_ports_macro() {
        let ports: &[PortDescriptor] = ports!["x": F32, "y": F32, "result": Vec2];
        assert_eq!(ports.len(), 3);
        assert_eq!(ports[0].name, "x");
        assert_eq!(ports[0].value_type, ValueType::F32);
        assert_eq!(ports[2].name, "result");
        assert_eq!(ports[2].value_type, ValueType::Vec2);
    }
}
