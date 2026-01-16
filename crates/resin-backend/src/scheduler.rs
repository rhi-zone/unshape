//! Backend-aware execution scheduling.
//!
//! This module provides the [`BackendAwareEvaluator`] which wraps any
//! [`Evaluator`](rhizome_resin_core::eval::Evaluator) and routes node
//! execution through appropriate compute backends.

use crate::backend::{ComputeBackend, Cost, WorkloadHint};
use crate::error::BackendError;
use crate::policy::ExecutionPolicy;
use crate::registry::BackendRegistry;
use rhizome_resin_core::{DynNode, EvalContext, NodeId, Value};
use std::sync::Arc;

/// Scheduler that selects backends for node execution.
///
/// The scheduler uses the [`ExecutionPolicy`] to choose which backend
/// should execute each node, considering:
/// - Backend capabilities and support for the node type
/// - Estimated execution cost
/// - Data locality (where inputs currently reside)
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_backend::{Scheduler, BackendRegistry, ExecutionPolicy};
///
/// let registry = BackendRegistry::with_cpu();
/// let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
///
/// // Get the best backend for a node
/// let backend = scheduler.select_backend(&node, &workload);
/// ```
pub struct Scheduler {
    registry: BackendRegistry,
    policy: ExecutionPolicy,
}

impl Scheduler {
    /// Creates a new scheduler with the given registry and policy.
    pub fn new(registry: BackendRegistry, policy: ExecutionPolicy) -> Self {
        Self { registry, policy }
    }

    /// Returns a reference to the backend registry.
    pub fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Returns a mutable reference to the backend registry.
    pub fn registry_mut(&mut self) -> &mut BackendRegistry {
        &mut self.registry
    }

    /// Returns the current execution policy.
    pub fn policy(&self) -> &ExecutionPolicy {
        &self.policy
    }

    /// Sets the execution policy.
    pub fn set_policy(&mut self, policy: ExecutionPolicy) {
        self.policy = policy;
    }

    /// Selects the best backend for executing a node.
    ///
    /// Returns `None` if no backend supports the node.
    pub fn select_backend(
        &self,
        node: &dyn DynNode,
        workload: &WorkloadHint,
    ) -> Option<&Arc<dyn ComputeBackend>> {
        match &self.policy {
            ExecutionPolicy::Auto => self.select_auto(node, workload),
            ExecutionPolicy::PreferKind(kind) => {
                // Try preferred kind first
                let preferred = self
                    .registry
                    .backends_of_kind(kind)
                    .into_iter()
                    .find(|b| b.supports_node(node));

                preferred.or_else(|| self.registry.first_supporting(node))
            }
            ExecutionPolicy::Named(name) => {
                let backend = self.registry.get(name)?;
                if backend.supports_node(node) {
                    Some(backend)
                } else {
                    None
                }
            }
            ExecutionPolicy::LocalFirst => {
                // For now, just use auto - LocalFirst would need data location info
                self.select_auto(node, workload)
            }
            ExecutionPolicy::MinimizeCost => self.select_min_cost(node, workload),
        }
    }

    /// Auto-select based on workload characteristics.
    fn select_auto(
        &self,
        node: &dyn DynNode,
        workload: &WorkloadHint,
    ) -> Option<&Arc<dyn ComputeBackend>> {
        let candidates = self.registry.backends_for_node(node);
        if candidates.is_empty() {
            return None;
        }

        // Simple heuristic: prefer bulk-efficient backends for large workloads
        let bulk_threshold = 10_000; // Elements threshold for GPU consideration

        if workload.element_count >= bulk_threshold {
            // Prefer bulk-efficient backends (typically GPU)
            if let Some(bulk) = candidates.iter().find(|b| b.capabilities().bulk_efficient) {
                return Some(*bulk);
            }
        }

        // For small workloads or if no bulk backend, prefer streaming-efficient (typically CPU)
        if let Some(streaming) = candidates
            .iter()
            .find(|b| b.capabilities().streaming_efficient)
        {
            return Some(*streaming);
        }

        // Fall back to first available
        candidates.first().copied()
    }

    /// Select the backend with minimum estimated cost.
    fn select_min_cost(
        &self,
        node: &dyn DynNode,
        workload: &WorkloadHint,
    ) -> Option<&Arc<dyn ComputeBackend>> {
        let candidates = self.registry.backends_for_node(node);

        candidates
            .into_iter()
            .filter_map(|b| {
                b.estimate_cost(node, workload)
                    .map(|cost| (b, cost.total()))
            })
            .min_by(|(_, cost_a), (_, cost_b)| {
                cost_a
                    .partial_cmp(cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(backend, _)| backend)
    }

    /// Execute a node using the selected backend.
    pub fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
        workload: &WorkloadHint,
    ) -> Result<Vec<Value>, BackendError> {
        let backend = self
            .select_backend(node, workload)
            .ok_or(BackendError::Unsupported)?;

        backend.execute(node, inputs, ctx)
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("registry", &self.registry)
            .field("policy", &self.policy)
            .finish()
    }
}

/// Result of backend-aware evaluation.
#[derive(Debug)]
pub struct BackendEvalResult {
    /// Output values for each requested node.
    pub outputs: Vec<Vec<Value>>,
    /// Which backend executed each node.
    pub backend_assignments: Vec<(NodeId, String)>,
    /// Total estimated cost.
    pub total_cost: Cost,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use rhizome_resin_core::{GraphError, PortDescriptor, ValueType};

    struct TestNode;

    impl DynNode for TestNode {
        fn type_name(&self) -> &'static str {
            "TestNode"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("out", ValueType::F32)]
        }

        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(1.0)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_scheduler_new() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);

        assert!(matches!(scheduler.policy(), ExecutionPolicy::Auto));
        assert_eq!(scheduler.registry().len(), 1);
    }

    #[test]
    fn test_scheduler_select_auto() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);

        let node = TestNode;
        let workload = WorkloadHint::single();

        let backend = scheduler.select_backend(&node, &workload);
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "cpu");
    }

    #[test]
    fn test_scheduler_select_named() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Named("cpu".into()));

        let node = TestNode;
        let workload = WorkloadHint::single();

        let backend = scheduler.select_backend(&node, &workload);
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "cpu");
    }

    #[test]
    fn test_scheduler_select_named_nonexistent() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Named("gpu".into()));

        let node = TestNode;
        let workload = WorkloadHint::single();

        let backend = scheduler.select_backend(&node, &workload);
        assert!(backend.is_none());
    }

    #[test]
    fn test_scheduler_select_prefer_kind() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::PreferKind(BackendKind::Cpu));

        let node = TestNode;
        let workload = WorkloadHint::single();

        let backend = scheduler.select_backend(&node, &workload);
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "cpu");
    }

    #[test]
    fn test_scheduler_select_minimize_cost() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::MinimizeCost);

        let node = TestNode;
        let workload = WorkloadHint::bulk(1000, 16);

        let backend = scheduler.select_backend(&node, &workload);
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "cpu");
    }

    #[test]
    fn test_scheduler_execute() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);

        let node = TestNode;
        let ctx = EvalContext::new();
        let workload = WorkloadHint::single();

        let result = scheduler.execute(&node, &[], &ctx, &workload).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_f32().unwrap(), 1.0);
    }

    #[test]
    fn test_scheduler_set_policy() {
        let registry = BackendRegistry::with_cpu();
        let mut scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);

        scheduler.set_policy(ExecutionPolicy::MinimizeCost);
        assert!(matches!(scheduler.policy(), ExecutionPolicy::MinimizeCost));
    }
}
