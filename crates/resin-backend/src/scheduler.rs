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

// ============================================================================
// Backend-Aware Evaluator
// ============================================================================

use rhizome_resin_core::{
    CacheKey, CachePolicy, EvalCache, EvalResult, Evaluator, Graph, GraphError, KeepAllPolicy,
};
use std::time::Instant;

/// Evaluator that routes node execution through compute backends.
///
/// This evaluator implements lazy evaluation with caching, similar to
/// [`LazyEvaluator`](rhizome_resin_core::LazyEvaluator), but routes each
/// node's execution through the [`Scheduler`] to select the best backend.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_backend::{
///     BackendAwareEvaluator, BackendRegistry, ExecutionPolicy, Scheduler,
/// };
/// use rhizome_resin_core::{Graph, EvalContext, Evaluator};
///
/// // Set up registry with CPU and optionally GPU
/// let registry = BackendRegistry::with_cpu();
/// let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
///
/// // Create evaluator
/// let mut evaluator = BackendAwareEvaluator::new(scheduler);
///
/// // Evaluate graph - backends selected automatically per node
/// let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;
/// ```
pub struct BackendAwareEvaluator {
    scheduler: Scheduler,
    cache: EvalCache,
    policy: Box<dyn CachePolicy>,
}

impl BackendAwareEvaluator {
    /// Creates a new backend-aware evaluator with the given scheduler.
    pub fn new(scheduler: Scheduler) -> Self {
        Self {
            scheduler,
            cache: EvalCache::new(),
            policy: Box::new(KeepAllPolicy),
        }
    }

    /// Creates an evaluator with a custom cache policy.
    pub fn with_cache_policy(scheduler: Scheduler, policy: impl CachePolicy + 'static) -> Self {
        Self {
            scheduler,
            cache: EvalCache::new(),
            policy: Box::new(policy),
        }
    }

    /// Returns a reference to the scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Returns a mutable reference to the scheduler.
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    /// Recursively evaluate a node using backends.
    fn evaluate_node(
        &mut self,
        graph: &Graph,
        node_id: NodeId,
        ctx: &EvalContext,
        computed: &mut Vec<NodeId>,
        cached: &mut Vec<NodeId>,
        backend_assignments: &mut Vec<(NodeId, String)>,
    ) -> Result<Vec<Value>, GraphError> {
        // Check for cancellation
        if ctx.is_cancelled() {
            return Err(GraphError::Cancelled);
        }

        let node = graph
            .get_node(node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;

        let inputs_desc = node.inputs();
        let num_inputs = inputs_desc.len();

        // Gather inputs by recursively evaluating upstream nodes
        let mut inputs = Vec::with_capacity(num_inputs);
        for port in 0..num_inputs {
            let wire = graph
                .wires()
                .iter()
                .find(|w| w.to_node == node_id && w.to_port == port);

            match wire {
                Some(w) => {
                    let upstream_outputs = self.evaluate_node(
                        graph,
                        w.from_node,
                        ctx,
                        computed,
                        cached,
                        backend_assignments,
                    )?;
                    let value = upstream_outputs.get(w.from_port).cloned().ok_or_else(|| {
                        GraphError::ExecutionError(format!(
                            "missing output port {} on node {}",
                            w.from_port, w.from_node
                        ))
                    })?;
                    inputs.push(value);
                }
                None => {
                    return Err(GraphError::UnconnectedInput {
                        node: node_id,
                        port,
                    });
                }
            }
        }

        // Check cache
        let cache_key = CacheKey::new(node_id, &inputs);
        if let Some(entry) = self.cache.get(&cache_key) {
            if self.policy.is_valid(&cache_key, entry) {
                cached.push(node_id);
                return Ok(entry.outputs.clone());
            }
        }

        // Estimate workload for backend selection
        let workload = estimate_workload(&inputs);

        // Execute through scheduler (selects best backend)
        let outputs = self
            .scheduler
            .execute(node.as_ref(), &inputs, ctx, &workload)
            .map_err(|e| GraphError::ExecutionError(e.to_string()))?;

        // Record which backend was used
        if let Some(backend) = self.scheduler.select_backend(node.as_ref(), &workload) {
            backend_assignments.push((node_id, backend.name().to_string()));
        }

        // Cache result if policy allows
        if self.policy.should_cache(node_id, &outputs) {
            self.cache.insert(cache_key, outputs.clone());
        }

        computed.push(node_id);
        Ok(outputs)
    }
}

impl Evaluator for BackendAwareEvaluator {
    fn evaluate(
        &mut self,
        graph: &Graph,
        outputs: &[NodeId],
        ctx: &EvalContext,
    ) -> Result<EvalResult, GraphError> {
        let start = Instant::now();
        let mut computed = Vec::new();
        let mut cached = Vec::new();
        let mut backend_assignments = Vec::new();
        let mut results = Vec::with_capacity(outputs.len());

        for &node_id in outputs {
            let node_outputs = self.evaluate_node(
                graph,
                node_id,
                ctx,
                &mut computed,
                &mut cached,
                &mut backend_assignments,
            )?;
            results.push(node_outputs);
        }

        Ok(EvalResult {
            outputs: results,
            computed_nodes: computed,
            cached_nodes: cached,
            elapsed: start.elapsed(),
        })
    }

    fn invalidate(&mut self, node: NodeId) {
        let keys_to_remove: Vec<_> = self
            .cache
            .keys()
            .filter(|k| k.node_id == node)
            .cloned()
            .collect();
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Estimate workload from input values.
fn estimate_workload(inputs: &[Value]) -> WorkloadHint {
    // Heuristic: sum up sizes of inputs
    let mut total_elements = 1usize;
    let mut total_bytes = 0usize;

    for input in inputs {
        match input {
            Value::F32(_) | Value::I32(_) | Value::Bool(_) => {
                total_bytes += 4;
            }
            Value::F64(_) => {
                total_bytes += 8;
            }
            Value::Vec2(_) => {
                total_bytes += 8;
            }
            Value::Vec3(_) => {
                total_bytes += 12;
            }
            Value::Vec4(_) => {
                total_bytes += 16;
            }
            Value::Opaque(v) => {
                // Try to estimate from type name
                let name = v.type_name();
                if name.contains("Texture") || name.contains("Image") {
                    // Assume medium-sized texture
                    total_elements = total_elements.max(256 * 256);
                    total_bytes += 256 * 256 * 4;
                } else if name.contains("Mesh") {
                    total_elements = total_elements.max(10_000);
                    total_bytes += 10_000 * 32; // vertices with positions + normals
                } else {
                    total_bytes += 1024; // default estimate
                }
            }
        }
    }

    WorkloadHint {
        element_count: total_elements,
        input_bytes: total_bytes,
        output_bytes: total_bytes, // assume similar output size
    }
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

    // ========================================================================
    // BackendAwareEvaluator tests
    // ========================================================================

    struct ConstNode(f32);

    impl DynNode for ConstNode {
        fn type_name(&self) -> &'static str {
            "ConstNode"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("value", ValueType::F32)]
        }

        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.0)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct AddNode;

    impl DynNode for AddNode {
        fn type_name(&self) -> &'static str {
            "AddNode"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![
                PortDescriptor::new("a", ValueType::F32),
                PortDescriptor::new("b", ValueType::F32),
            ]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("result", ValueType::F32)]
        }

        fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            let a = inputs[0]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let b = inputs[1]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(a + b)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_backend_aware_evaluator_simple() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
        let mut evaluator = BackendAwareEvaluator::new(scheduler);

        let mut graph = Graph::new();
        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect(const_a, 0, add, 0).unwrap();
        graph.connect(const_b, 0, add, 1).unwrap();

        let ctx = EvalContext::new();
        let result = evaluator.evaluate(&graph, &[add], &ctx).unwrap();

        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].len(), 1);
        assert_eq!(result.outputs[0][0].as_f32().unwrap(), 5.0);
    }

    #[test]
    fn test_backend_aware_evaluator_caching() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
        let mut evaluator = BackendAwareEvaluator::new(scheduler);

        let mut graph = Graph::new();
        let const_node = graph.add_node(ConstNode(42.0));

        let ctx = EvalContext::new();

        // First evaluation
        let result1 = evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();
        assert_eq!(result1.computed_nodes.len(), 1);
        assert_eq!(result1.cached_nodes.len(), 0);

        // Second evaluation should use cache
        let result2 = evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();
        assert_eq!(result2.computed_nodes.len(), 0);
        assert_eq!(result2.cached_nodes.len(), 1);

        // Same output value
        assert_eq!(result1.outputs[0][0], result2.outputs[0][0]);
    }

    #[test]
    fn test_backend_aware_evaluator_invalidate() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
        let mut evaluator = BackendAwareEvaluator::new(scheduler);

        let mut graph = Graph::new();
        let const_node = graph.add_node(ConstNode(42.0));

        let ctx = EvalContext::new();

        // First evaluation
        evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();

        // Invalidate
        evaluator.invalidate(const_node);

        // Should recompute
        let result = evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();
        assert_eq!(result.computed_nodes.len(), 1);
        assert_eq!(result.cached_nodes.len(), 0);
    }

    #[test]
    fn test_backend_aware_evaluator_clear_cache() {
        let registry = BackendRegistry::with_cpu();
        let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
        let mut evaluator = BackendAwareEvaluator::new(scheduler);

        let mut graph = Graph::new();
        let const_node = graph.add_node(ConstNode(42.0));

        let ctx = EvalContext::new();

        // First evaluation
        evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();

        // Clear all cache
        evaluator.clear_cache();

        // Should recompute
        let result = evaluator.evaluate(&graph, &[const_node], &ctx).unwrap();
        assert_eq!(result.computed_nodes.len(), 1);
        assert_eq!(result.cached_nodes.len(), 0);
    }

    #[test]
    fn test_estimate_workload() {
        // Empty inputs
        let workload = estimate_workload(&[]);
        assert_eq!(workload.element_count, 1);
        assert_eq!(workload.input_bytes, 0);

        // Primitive inputs
        let workload = estimate_workload(&[
            Value::F32(1.0),
            Value::Vec3(rhizome_resin_core::glam::Vec3::ZERO),
        ]);
        assert_eq!(workload.input_bytes, 4 + 12);
    }
}
