//! Compute backends demo.
//!
//! Demonstrates heterogeneous execution using the compute backend system.
//! Nodes are automatically routed to the best available backend (CPU/GPU).
//!
//! Run with: `cargo run --example compute_backends`

use rhizome_resin_backend::{
    BackendNodeExecutor, BackendRegistry, ExecutionPolicy, LazyEvaluator, Scheduler,
    backend_evaluator,
};
use rhizome_resin_core::{
    DynNode, EvalContext, Evaluator, Graph, GraphError, PortDescriptor, Value, ValueType,
};
use std::any::Any;

// ============================================================================
// Example Nodes
// ============================================================================

/// A simple constant node that outputs a float value.
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A node that multiplies two inputs.
struct MultiplyNode;

impl DynNode for MultiplyNode {
    fn type_name(&self) -> &'static str {
        "MultiplyNode"
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
        Ok(vec![Value::F32(a * b)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A node that adds two inputs.
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("Compute Backends Demo");
    println!("=====================\n");

    // Build a simple expression graph: (2 * 3) + (4 * 5)
    let mut graph = Graph::new();

    let const_2 = graph.add_node(ConstNode(2.0));
    let const_3 = graph.add_node(ConstNode(3.0));
    let const_4 = graph.add_node(ConstNode(4.0));
    let const_5 = graph.add_node(ConstNode(5.0));

    let mul_a = graph.add_node(MultiplyNode);
    let mul_b = graph.add_node(MultiplyNode);
    let add = graph.add_node(AddNode);

    // Wire up: mul_a = 2 * 3
    graph.connect(const_2, 0, mul_a, 0).unwrap();
    graph.connect(const_3, 0, mul_a, 1).unwrap();

    // Wire up: mul_b = 4 * 5
    graph.connect(const_4, 0, mul_b, 0).unwrap();
    graph.connect(const_5, 0, mul_b, 1).unwrap();

    // Wire up: add = mul_a + mul_b
    graph.connect(mul_a, 0, add, 0).unwrap();
    graph.connect(mul_b, 0, add, 1).unwrap();

    println!("Graph: (2 * 3) + (4 * 5)");
    println!(
        "Nodes: {} | Wires: {}\n",
        graph.nodes_iter().count(),
        graph.wires().len()
    );

    // Simple API: backend_evaluator() convenience function
    let mut evaluator = backend_evaluator(ExecutionPolicy::Auto);

    // Show available backends via the scheduler
    println!("Available backends:");
    for backend in evaluator.executor().scheduler().registry().iter() {
        let caps = backend.capabilities();
        println!(
            "  - {} ({:?}, bulk_efficient={}, streaming_efficient={})",
            backend.name(),
            caps.kind,
            caps.bulk_efficient,
            caps.streaming_efficient
        );
    }
    println!();

    // Evaluate
    let ctx = EvalContext::new();
    let result = evaluator.evaluate(&graph, &[add], &ctx).unwrap();

    println!("Evaluation result:");
    println!("  Output: {:?}", result.outputs[0][0]);
    println!("  Computed nodes: {:?}", result.computed_nodes);
    println!("  Cached nodes: {:?}", result.cached_nodes);
    println!("  Elapsed: {:?}\n", result.elapsed);

    // Verify the result
    let output = result.outputs[0][0].as_f32().unwrap();
    let expected = (2.0 * 3.0) + (4.0 * 5.0);
    println!(
        "Verification: {} == {} ? {}",
        output,
        expected,
        (output - expected).abs() < f32::EPSILON
    );

    // Demonstrate caching
    println!("\nRe-evaluating (should use cache)...");
    let result2 = evaluator.evaluate(&graph, &[add], &ctx).unwrap();
    println!("  Computed nodes: {:?}", result2.computed_nodes);
    println!("  Cached nodes: {:?}", result2.cached_nodes);

    // Demonstrate different policies
    println!("\nTrying different execution policies:");

    let policies = [
        ("Auto", ExecutionPolicy::Auto),
        (
            "PreferKind(Cpu)",
            ExecutionPolicy::PreferKind(rhizome_resin_backend::BackendKind::Cpu),
        ),
        ("MinimizeCost", ExecutionPolicy::MinimizeCost),
    ];

    for (name, policy) in policies {
        // Simple: use backend_evaluator()
        let mut evaluator = backend_evaluator(policy);
        let result = evaluator.evaluate(&graph, &[add], &ctx).unwrap();
        println!("  {}: output = {:?}", name, result.outputs[0][0]);
    }

    // Advanced API: manual setup for custom registries
    println!("\nAdvanced: manual setup with custom registry");
    let registry = BackendRegistry::with_cpu();
    // Could add GPU here: registry.register(Arc::new(gpu_backend));
    let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
    let mut evaluator = LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler));
    let result = evaluator.evaluate(&graph, &[add], &ctx).unwrap();
    println!("  Output: {:?}", result.outputs[0][0]);

    println!("\nDemo complete!");
}
