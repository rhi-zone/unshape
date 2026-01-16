//! CPU backend implementation.

use crate::backend::{BackendCapabilities, BackendKind, ComputeBackend, Cost, WorkloadHint};
use crate::error::BackendError;
use rhizome_resin_core::{DynNode, EvalContext, Value};

/// Default CPU backend.
///
/// This backend uses the node's native `execute()` method directly.
/// It's always available and supports all nodes (since all nodes
/// implement `DynNode::execute`).
///
/// # Cost Model
///
/// The CPU backend uses a simple cost model:
/// - Compute cost scales linearly with element count
/// - No transfer cost (data is already on CPU)
///
/// # Example
///
/// ```
/// use rhizome_resin_backend::{CpuBackend, ComputeBackend, BackendKind};
///
/// let backend = CpuBackend;
/// assert_eq!(backend.name(), "cpu");
/// assert_eq!(backend.capabilities().kind, BackendKind::Cpu);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            bulk_efficient: false,
            streaming_efficient: true,
        }
    }

    fn supports_node(&self, _node: &dyn DynNode) -> bool {
        // CPU can execute any node via DynNode::execute
        true
    }

    fn estimate_cost(&self, _node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
        // Simple linear cost model
        // CPU has no transfer overhead but scales linearly with work
        Some(Cost {
            compute: workload.element_count as f64,
            transfer: 0.0,
        })
    }

    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, BackendError> {
        node.execute(inputs, ctx).map_err(BackendError::GraphError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_name() {
        assert_eq!(CpuBackend.name(), "cpu");
    }

    #[test]
    fn test_cpu_backend_capabilities() {
        let caps = CpuBackend.capabilities();
        assert_eq!(caps.kind, BackendKind::Cpu);
        assert!(!caps.bulk_efficient);
        assert!(caps.streaming_efficient);
    }

    #[test]
    fn test_cpu_backend_cost() {
        let workload = WorkloadHint::bulk(1000, 16);
        let cost = CpuBackend.estimate_cost(&DummyNode, &workload).unwrap();
        assert_eq!(cost.compute, 1000.0);
        assert_eq!(cost.transfer, 0.0);
    }

    // Dummy node for testing
    struct DummyNode;

    impl DynNode for DummyNode {
        fn type_name(&self) -> &'static str {
            "DummyNode"
        }

        fn inputs(&self) -> Vec<rhizome_resin_core::PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<rhizome_resin_core::PortDescriptor> {
            vec![rhizome_resin_core::PortDescriptor::new(
                "out",
                rhizome_resin_core::ValueType::F32,
            )]
        }

        fn execute(
            &self,
            _inputs: &[Value],
            _ctx: &EvalContext,
        ) -> Result<Vec<Value>, rhizome_resin_core::GraphError> {
            Ok(vec![Value::F32(42.0)])
        }
    }

    #[test]
    fn test_cpu_backend_execute() {
        let ctx = EvalContext::new();
        let result = CpuBackend.execute(&DummyNode, &[], &ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_f32().unwrap(), 42.0);
    }

    #[test]
    fn test_cpu_backend_supports_all_nodes() {
        assert!(CpuBackend.supports_node(&DummyNode));
    }
}
