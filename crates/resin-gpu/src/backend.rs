//! GPU compute backend implementation.

use crate::{GpuContext, GpuError};
use rhizome_resin_backend::{
    BackendCapabilities, BackendError, BackendKind, ComputeBackend, Cost, WorkloadHint,
};
use rhizome_resin_core::{DynNode, EvalContext, Value};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// GPU compute backend using wgpu.
///
/// This backend executes nodes on the GPU when a kernel is registered
/// for that node type. Falls back to CPU for unsupported nodes.
///
/// # Registering Kernels
///
/// ```ignore
/// use rhizome_resin_gpu::{GpuComputeBackend, GpuKernel};
///
/// struct MyNoiseKernel;
///
/// impl GpuKernel for MyNoiseKernel {
///     fn execute(
///         &self,
///         ctx: &GpuContext,
///         inputs: &[Value],
///         eval_ctx: &EvalContext,
///     ) -> Result<Vec<Value>, GpuError> {
///         // Execute compute shader...
///         todo!()
///     }
/// }
///
/// let mut backend = GpuComputeBackend::new()?;
/// backend.register_kernel::<MyNoiseNode>(Arc::new(MyNoiseKernel));
/// ```
pub struct GpuComputeBackend {
    ctx: GpuContext,
    kernels: RwLock<HashMap<TypeId, Arc<dyn GpuKernel>>>,
    /// Device ID for this backend (default 0).
    device_id: u32,
}

impl GpuComputeBackend {
    /// Creates a new GPU backend with the default adapter.
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            ctx: GpuContext::new()?,
            kernels: RwLock::new(HashMap::new()),
            device_id: 0,
        })
    }

    /// Creates a GPU backend with an existing context.
    pub fn with_context(ctx: GpuContext) -> Self {
        Self {
            ctx,
            kernels: RwLock::new(HashMap::new()),
            device_id: 0,
        }
    }

    /// Returns a reference to the GPU context.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Returns the device ID for this backend.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Registers a GPU kernel for a node type.
    ///
    /// The kernel will be used to execute nodes of type `N` on the GPU.
    pub fn register_kernel<N: DynNode + 'static>(&self, kernel: Arc<dyn GpuKernel>) {
        let type_id = TypeId::of::<N>();
        self.kernels.write().unwrap().insert(type_id, kernel);
    }

    /// Checks if a kernel is registered for the given node type.
    fn has_kernel_for(&self, node: &dyn DynNode) -> bool {
        let type_id = node.as_any().type_id();
        self.kernels.read().unwrap().contains_key(&type_id)
    }

    /// Gets the kernel for a node type, if registered.
    fn get_kernel(&self, node: &dyn DynNode) -> Option<Arc<dyn GpuKernel>> {
        let type_id = node.as_any().type_id();
        self.kernels.read().unwrap().get(&type_id).cloned()
    }
}

impl std::fmt::Debug for GpuComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuComputeBackend")
            .field("device_id", &self.device_id)
            .field("registered_kernels", &self.kernels.read().unwrap().len())
            .finish()
    }
}

impl ComputeBackend for GpuComputeBackend {
    fn name(&self) -> &str {
        "gpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Gpu,
            bulk_efficient: true,
            streaming_efficient: false,
        }
    }

    fn supports_node(&self, node: &dyn DynNode) -> bool {
        self.has_kernel_for(node)
    }

    fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
        if !self.supports_node(node) {
            return None;
        }

        // GPU has high fixed overhead but low per-element cost
        // Transfer cost is significant for data movement
        let overhead = 100.0; // Fixed GPU dispatch overhead
        let compute_per_element = 0.001; // GPU is ~1000x faster per element for parallel work
        let transfer_per_byte = 0.01; // PCIe transfer cost

        Some(Cost {
            compute: overhead + workload.element_count as f64 * compute_per_element,
            transfer: workload.total_bytes() as f64 * transfer_per_byte,
        })
    }

    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, BackendError> {
        let kernel = self.get_kernel(node).ok_or(BackendError::Unsupported)?;

        kernel
            .execute(&self.ctx, inputs, ctx)
            .map_err(|e| BackendError::ExecutionFailed(e.to_string()))
    }
}

/// Trait for GPU kernel implementations.
///
/// Each node type that can run on GPU needs a kernel implementation
/// that knows how to execute that operation using compute shaders.
pub trait GpuKernel: Send + Sync {
    /// Executes the kernel on the GPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The GPU context with device and queue
    /// * `inputs` - Input values for the operation
    /// * `eval_ctx` - Evaluation context (time, etc.)
    ///
    /// # Returns
    ///
    /// Output values from the kernel, or an error.
    fn execute(
        &self,
        ctx: &GpuContext,
        inputs: &[Value],
        eval_ctx: &EvalContext,
    ) -> Result<Vec<Value>, GpuError>;

    /// Returns an optional custom cost estimate for this kernel.
    ///
    /// If `None`, the default GPU cost model is used.
    fn estimate_cost(&self, _workload: &WorkloadHint) -> Option<Cost> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_core::{PortDescriptor, ValueType};

    // Test node for kernel registration
    struct TestGpuNode;

    impl DynNode for TestGpuNode {
        fn type_name(&self) -> &'static str {
            "TestGpuNode"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("out", ValueType::F32)]
        }

        fn execute(
            &self,
            _inputs: &[Value],
            _ctx: &EvalContext,
        ) -> Result<Vec<Value>, rhizome_resin_core::GraphError> {
            Ok(vec![Value::F32(1.0)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // Test kernel
    struct TestKernel;

    impl GpuKernel for TestKernel {
        fn execute(
            &self,
            _ctx: &GpuContext,
            _inputs: &[Value],
            _eval_ctx: &EvalContext,
        ) -> Result<Vec<Value>, GpuError> {
            Ok(vec![Value::F32(42.0)])
        }
    }

    #[test]
    fn test_gpu_backend_name() {
        // Skip if no GPU available
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return, // No GPU, skip test
        };
        assert_eq!(backend.name(), "gpu");
    }

    #[test]
    fn test_gpu_backend_capabilities() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        let caps = backend.capabilities();
        assert_eq!(caps.kind, BackendKind::Gpu);
        assert!(caps.bulk_efficient);
        assert!(!caps.streaming_efficient);
    }

    #[test]
    fn test_gpu_backend_unsupported_without_kernel() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        let node = TestGpuNode;
        assert!(!backend.supports_node(&node));
    }

    #[test]
    fn test_gpu_backend_supported_with_kernel() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        backend.register_kernel::<TestGpuNode>(Arc::new(TestKernel));
        let node = TestGpuNode;
        assert!(backend.supports_node(&node));
    }

    #[test]
    fn test_gpu_backend_cost_estimate() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        backend.register_kernel::<TestGpuNode>(Arc::new(TestKernel));

        let node = TestGpuNode;
        let workload = WorkloadHint::bulk(10000, 16);
        let cost = backend.estimate_cost(&node, &workload).unwrap();

        // GPU should have overhead but low per-element cost
        assert!(cost.compute > 0.0);
        assert!(cost.transfer > 0.0);
    }

    #[test]
    fn test_gpu_backend_execute() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        backend.register_kernel::<TestGpuNode>(Arc::new(TestKernel));

        let node = TestGpuNode;
        let ctx = EvalContext::new();
        let result = backend.execute(&node, &[], &ctx).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_f32().unwrap(), 42.0);
    }

    #[test]
    fn test_gpu_backend_execute_unsupported() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let node = TestGpuNode;
        let ctx = EvalContext::new();
        let result = backend.execute(&node, &[], &ctx);

        assert!(matches!(result, Err(BackendError::Unsupported)));
    }
}
