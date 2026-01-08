//! Node trait for graph execution.

use crate::context::EvalContext;
use crate::error::GraphError;
use crate::value::{Value, ValueType};

/// Port descriptor for a node input or output.
#[derive(Debug, Clone)]
pub struct PortDescriptor {
    /// Port name for display/debugging.
    pub name: &'static str,
    /// Type of values this port accepts/produces.
    pub value_type: ValueType,
}

impl PortDescriptor {
    pub fn new(name: &'static str, value_type: ValueType) -> Self {
        Self { name, value_type }
    }
}

/// Trait for nodes that can be executed dynamically.
///
/// This is the runtime interface for graph execution. Node authors
/// typically don't implement this directly - instead they use derive
/// macros that generate the implementation from concrete types.
pub trait DynNode: Send + Sync {
    /// Returns the type name for serialization/debugging.
    fn type_name(&self) -> &'static str;

    /// Returns descriptors for all input ports.
    fn inputs(&self) -> Vec<PortDescriptor>;

    /// Returns descriptors for all output ports.
    fn outputs(&self) -> Vec<PortDescriptor>;

    /// Executes the node with the given inputs.
    ///
    /// # Arguments
    /// * `inputs` - Input values, one per input port in order.
    /// * `ctx` - Evaluation context with time, resolution, etc.
    ///
    /// # Returns
    /// Output values, one per output port in order.
    fn execute(&self, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError>;
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
