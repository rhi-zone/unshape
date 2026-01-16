# resin-core

The foundational crate for resin's node graph system. Provides the runtime graph container, node execution traits, value types, and evaluation strategies.

## Overview

**resin-core** provides:

- **Graph** - Container for nodes and wires, with eager execution
- **DynNode** - Trait for implementing custom nodes
- **Value** - Runtime value type for data flowing through the graph
- **EvalContext** - Execution environment (time, cancellation, quality hints)
- **Evaluator** trait and **LazyEvaluator** - Pluggable evaluation strategies with caching

## Related Crates

| Crate | Relationship |
|-------|--------------|
| **resin-serde** | Serialization for Graph using NodeRegistry |
| **resin-op** | DynOp trait for operations-as-values pattern |
| **resin-history** | Undo/redo and event sourcing built on graphs |
| **resin-macros** | `#[derive(DynNode)]` macro for node definitions |

## Core Concepts

### Graph & Nodes

A `Graph` contains nodes (computation units) connected by wires (data flow connections). Each node has typed input/output ports.

```rust
use rhizome_resin_core::{Graph, DynNode, EvalContext, Value, ValueType, PortDescriptor, GraphError};

// Define a custom node that adds two numbers
struct AddNode;

impl DynNode for AddNode {
    fn type_name(&self) -> &'static str {
        "math::Add"
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
        let a = inputs[0].as_f32().map_err(|e| GraphError::ExecutionError(e.to_string()))?;
        let b = inputs[1].as_f32().map_err(|e| GraphError::ExecutionError(e.to_string()))?;
        Ok(vec![Value::F32(a + b)])
    }
}
```

### Wiring and Execution

Connect nodes by their port indices, then execute from any output node:

```rust
// Build a simple graph: const(2.0) + const(3.0)
let mut graph = Graph::new();

// Add nodes (assume ConstNode outputs a constant f32)
let c1 = graph.add_node(ConstNode { value: 2.0 });
let c2 = graph.add_node(ConstNode { value: 3.0 });
let add = graph.add_node(AddNode);

// Wire: const1.output[0] -> add.input[0]
//       const2.output[0] -> add.input[1]
graph.connect(c1, 0, add, 0).unwrap();
graph.connect(c2, 0, add, 1).unwrap();

// Execute - computes all nodes in topological order
let result = graph.execute(add).unwrap();
assert_eq!(result[0].as_f32().unwrap(), 5.0);
```

### EvalContext

The evaluation context provides execution parameters beyond just input values:

```rust
use rhizome_resin_core::{EvalContext, CancellationToken};

// Basic context
let ctx = EvalContext::new();

// Context with time (for animation)
let ctx = EvalContext::new()
    .with_time(1.5, 90, 1.0 / 60.0)  // time=1.5s, frame=90, dt=16.6ms
    .with_seed(42);                   // deterministic randomness

// Context with cancellation support
let token = CancellationToken::new();
let ctx = EvalContext::new().with_cancel(token.clone());

// From another thread:
// token.cancel();

// In long-running nodes, check periodically:
if ctx.is_cancelled() {
    return Err(GraphError::Cancelled);
}
```

Context fields:

| Field | Purpose |
|-------|---------|
| `time` | Current time in seconds |
| `frame` | Current frame number |
| `dt` | Delta time since last evaluation |
| `preview_mode` | Hint that quality can be reduced |
| `target_resolution` | Resolution hint for LOD decisions |
| `seed` | Random seed for reproducibility |

## Evaluation Strategies

### Eager Execution (Default)

`Graph::execute()` runs all nodes in topological order:

```rust
// Executes entire graph, returns outputs of `output_node`
let result = graph.execute(output_node)?;

// With custom context
let ctx = EvalContext::new().with_time(1.0, 60, 1.0/60.0);
let result = graph.execute_with_context(output_node, &ctx)?;
```

### Lazy Evaluation

`LazyEvaluator` only computes nodes needed for requested outputs, with caching:

```rust
use rhizome_resin_core::{LazyEvaluator, Evaluator, EvalContext};

let mut evaluator = LazyEvaluator::new();
let ctx = EvalContext::new();

// Only nodes upstream of `output_node` are computed
let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;

// Result contains outputs and diagnostics
println!("Outputs: {:?}", result.outputs);
println!("Computed {} nodes", result.computed_nodes.len());
println!("Used {} cached nodes", result.cached_nodes.len());

// Second call uses cache - no recomputation
let result2 = evaluator.evaluate(&graph, &[output_node], &ctx)?;

// Invalidate a node when its params change
evaluator.invalidate(some_node);
```

### Custom Cache Policies

Implement `CachePolicy` to control caching behavior:

```rust
use rhizome_resin_core::{CachePolicy, CacheKey, CacheEntry, EvalCache, NodeId, Value};

struct TimeLimitedPolicy {
    max_age: std::time::Duration,
}

impl CachePolicy for TimeLimitedPolicy {
    fn should_cache(&self, _node: NodeId, _outputs: &[Value]) -> bool {
        true
    }

    fn is_valid(&self, _key: &CacheKey, entry: &CacheEntry) -> bool {
        entry.created_at.elapsed() < self.max_age
    }

    fn evict(&mut self, cache: &mut EvalCache) -> usize {
        // Custom eviction logic
        0
    }
}

let evaluator = LazyEvaluator::with_policy(TimeLimitedPolicy {
    max_age: std::time::Duration::from_secs(60),
});
```

## Value Types

The `Value` enum represents data flowing through wires:

```rust
use rhizome_resin_core::Value;
use glam::{Vec2, Vec3, Vec4};

// Scalars
let f = Value::F32(1.5);
let d = Value::F64(1.5);
let i = Value::I32(-42);
let b = Value::Bool(true);

// Vectors (using glam)
let v2 = Value::Vec2(Vec2::new(1.0, 2.0));
let v3 = Value::Vec3(Vec3::new(1.0, 2.0, 3.0));
let v4 = Value::Vec4(Vec4::new(1.0, 2.0, 3.0, 4.0));

// Type-safe extraction
let x: f32 = f.as_f32().unwrap();
```

## Cancellation

Long-running evaluations can be cancelled cooperatively:

```rust
use rhizome_resin_core::{CancellationToken, EvalContext, LazyEvaluator, Evaluator};
use std::thread;

let token = CancellationToken::new();
let ctx = EvalContext::new().with_cancel(token.clone());

// Spawn evaluation in background
let handle = thread::spawn(move || {
    let mut evaluator = LazyEvaluator::new();
    evaluator.evaluate(&graph, &[output], &ctx)
});

// Cancel from main thread
thread::sleep(std::time::Duration::from_millis(100));
token.cancel();

// Evaluation returns Err(GraphError::Cancelled)
let result = handle.join().unwrap();
```

For custom nodes with long inner loops:

```rust
fn execute(&self, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
    for i in 0..1_000_000 {
        // Check every N iterations
        if i % 1000 == 0 && ctx.is_cancelled() {
            return Err(GraphError::Cancelled);
        }
        // ... expensive work ...
    }
    Ok(vec![/* results */])
}
```

## Progress Reporting

Track evaluation progress via callbacks:

```rust
use rhizome_resin_core::{EvalContext, EvalProgress};

let ctx = EvalContext::new().with_progress(|progress: EvalProgress| {
    println!(
        "Progress: {}/{} nodes, elapsed: {:?}",
        progress.completed_nodes,
        progress.total_nodes,
        progress.elapsed
    );
});
```

## Using the Derive Macro

The `#[derive(DynNode)]` macro simplifies node definitions:

```rust
use rhizome_resin_core::{DynNodeDerive, EvalContext};

#[derive(DynNodeDerive, Clone, Default)]
struct MultiplyNode {
    #[input]
    a: f32,
    #[input]
    b: f32,
    #[output]
    result: f32,
}

impl MultiplyNode {
    fn compute(&mut self, _ctx: &EvalContext) {
        self.result = self.a * self.b;
    }
}
```

The macro generates the `DynNode` trait implementation, port descriptors, and value marshalling.

## Error Handling

```rust
use rhizome_resin_core::GraphError;

match graph.execute(node) {
    Ok(outputs) => { /* use outputs */ }
    Err(GraphError::NodeNotFound(id)) => { /* node doesn't exist */ }
    Err(GraphError::UnconnectedInput { node, port }) => { /* missing wire */ }
    Err(GraphError::TypeMismatch { expected, got }) => { /* wire type error */ }
    Err(GraphError::Cancelled) => { /* evaluation was cancelled */ }
    Err(GraphError::ExecutionError(msg)) => { /* node execution failed */ }
    Err(e) => { /* other errors */ }
}
```

## Example: Animation Graph

```rust
use rhizome_resin_core::{Graph, EvalContext, LazyEvaluator, Evaluator};

fn render_frame(
    graph: &Graph,
    evaluator: &mut LazyEvaluator,
    output: NodeId,
    frame: u64,
) -> Result<Vec<Value>, GraphError> {
    let time = frame as f64 / 60.0;
    let ctx = EvalContext::new()
        .with_time(time, frame, 1.0 / 60.0)
        .with_preview_mode(true)  // Lower quality for preview
        .with_resolution(1280, 720);

    let result = evaluator.evaluate(graph, &[output], &ctx)?;
    Ok(result.outputs.into_iter().next().unwrap())
}
```
