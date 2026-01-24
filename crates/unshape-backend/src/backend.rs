//! Core backend trait and types.

use crate::error::BackendError;
use unshape_core::{DynNode, EvalContext, Value};

/// A compute backend that can execute nodes.
///
/// Backends register themselves and advertise capabilities.
/// The scheduler selects backends based on node requirements
/// and execution policy.
///
/// # Implementing a Backend
///
/// ```ignore
/// use unshape_backend::*;
///
/// pub struct MyGpuBackend {
///     // GPU-specific state
/// }
///
/// impl ComputeBackend for MyGpuBackend {
///     fn name(&self) -> &str { "my-gpu" }
///
///     fn capabilities(&self) -> BackendCapabilities {
///         BackendCapabilities {
///             kind: BackendKind::Gpu,
///             bulk_efficient: true,
///             streaming_efficient: false,
///         }
///     }
///
///     fn supports_node(&self, node: &dyn DynNode) -> bool {
///         // Check if we have a GPU implementation for this node type
///         false
///     }
///
///     fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
///         Some(Cost {
///             compute: 1.0 + workload.element_count as f64 * 0.001,
///             transfer: workload.total_bytes() as f64 * 0.01,
///         })
///     }
///
///     fn execute(
///         &self,
///         node: &dyn DynNode,
///         inputs: &[Value],
///         ctx: &EvalContext,
///     ) -> Result<Vec<Value>, BackendError> {
///         // Execute on GPU
///         Err(BackendError::Unsupported)
///     }
/// }
/// ```
pub trait ComputeBackend: Send + Sync {
    /// Returns the unique name of this backend.
    ///
    /// Used for logging, debugging, and explicit backend selection.
    fn name(&self) -> &str;

    /// Returns the capabilities of this backend.
    fn capabilities(&self) -> BackendCapabilities;

    /// Returns `true` if this backend can execute the given node.
    ///
    /// A backend might not support a node if:
    /// - No GPU kernel is registered for the node type
    /// - The node requires features the backend doesn't have
    /// - Input/output types aren't compatible
    fn supports_node(&self, node: &dyn DynNode) -> bool;

    /// Estimates the cost of executing a node with the given workload.
    ///
    /// Returns `None` if the node is not supported.
    /// The scheduler uses this to choose between backends.
    fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost>;

    /// Executes a node on this backend.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to execute
    /// * `inputs` - Input values for the node
    /// * `ctx` - Evaluation context (time, cancellation, etc.)
    ///
    /// # Returns
    ///
    /// Output values from the node, or an error.
    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, BackendError>;
}

/// Capabilities of a compute backend.
#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    /// Broad category of the backend.
    pub kind: BackendKind,
    /// Whether this backend is efficient for bulk operations (many elements).
    pub bulk_efficient: bool,
    /// Whether this backend is efficient for streaming (low latency).
    pub streaming_efficient: bool,
}

/// Broad category of a backend.
///
/// Used by [`ExecutionPolicy::PreferKind`] to express preferences
/// without naming specific backends.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Standard CPU execution.
    Cpu,
    /// SIMD-optimized CPU execution.
    CpuSimd,
    /// GPU compute (wgpu, CUDA, etc.).
    Gpu,
    /// Custom backend type.
    Custom(String),
}

/// Hints about workload size for scheduling decisions.
///
/// Backends use this to estimate costs and decide whether
/// GPU overhead is worth it.
#[derive(Clone, Debug, Default)]
pub struct WorkloadHint {
    /// Number of elements to process (pixels, vertices, samples, etc.).
    pub element_count: usize,
    /// Approximate input data size in bytes.
    pub input_bytes: usize,
    /// Approximate output data size in bytes.
    pub output_bytes: usize,
}

impl WorkloadHint {
    /// Creates a hint for a single element.
    pub fn single() -> Self {
        Self {
            element_count: 1,
            input_bytes: 64,
            output_bytes: 64,
        }
    }

    /// Creates a hint for bulk processing.
    pub fn bulk(count: usize, bytes_per_element: usize) -> Self {
        Self {
            element_count: count,
            input_bytes: count * bytes_per_element,
            output_bytes: count * bytes_per_element,
        }
    }

    /// Creates a hint for texture processing.
    pub fn texture(width: u32, height: u32, channels: u32) -> Self {
        let pixels = (width * height) as usize;
        let bytes = pixels * (channels as usize) * 4; // assuming f32
        Self {
            element_count: pixels,
            input_bytes: bytes,
            output_bytes: bytes,
        }
    }

    /// Total bytes (input + output).
    pub fn total_bytes(&self) -> usize {
        self.input_bytes + self.output_bytes
    }
}

/// Estimated execution cost.
///
/// Used by the scheduler to choose between backends.
/// Values are relative — only comparisons matter.
#[derive(Clone, Debug, Default)]
pub struct Cost {
    /// Estimated compute time (relative units).
    pub compute: f64,
    /// Estimated data transfer time (relative units).
    pub transfer: f64,
}

impl Cost {
    /// Total cost (compute + transfer).
    pub fn total(&self) -> f64 {
        self.compute + self.transfer
    }

    /// Creates a zero cost.
    pub fn zero() -> Self {
        Self {
            compute: 0.0,
            transfer: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_hint_single() {
        let hint = WorkloadHint::single();
        assert_eq!(hint.element_count, 1);
    }

    #[test]
    fn test_workload_hint_bulk() {
        let hint = WorkloadHint::bulk(1000, 16);
        assert_eq!(hint.element_count, 1000);
        assert_eq!(hint.input_bytes, 16000);
        assert_eq!(hint.total_bytes(), 32000);
    }

    #[test]
    fn test_workload_hint_texture() {
        let hint = WorkloadHint::texture(512, 512, 4);
        assert_eq!(hint.element_count, 512 * 512);
        assert_eq!(hint.input_bytes, 512 * 512 * 4 * 4);
    }

    #[test]
    fn test_cost_total() {
        let cost = Cost {
            compute: 10.0,
            transfer: 5.0,
        };
        assert_eq!(cost.total(), 15.0);
    }

    #[test]
    fn test_backend_kind_equality() {
        assert_eq!(BackendKind::Cpu, BackendKind::Cpu);
        assert_ne!(BackendKind::Cpu, BackendKind::Gpu);
        assert_eq!(
            BackendKind::Custom("foo".into()),
            BackendKind::Custom("foo".into())
        );
    }
}
