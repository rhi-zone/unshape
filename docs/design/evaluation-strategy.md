# Evaluation Strategy

How graphs execute: when nodes compute, in what order, and who initiates.

## The Question

Two orthogonal choices:

**Lazy vs Eager:** When is computation performed?
- Lazy: on demand (when output requested)
- Eager: immediately (when inputs change)

**Pull vs Push:** Who initiates computation?
- Pull: consumer requests from producer
- Push: producer sends to consumer

## Strategy Combinations

| Strategy | Description | Use case |
|----------|-------------|----------|
| Pull + Lazy | Compute only requested outputs, only when asked | Rendering, export |
| Push + Eager | Changes propagate immediately | Live preview, streaming |
| Push invalidate + Pull compute | Mark dirty on change, recompute on demand | Interactive editing |

## Per-Domain Fit

| Domain | Best fit | Reason |
|--------|----------|--------|
| Texture generation | Pull + Lazy | Generate at needed resolution, skip unused |
| Mesh generation | Pull + Lazy | Generate when exporting |
| Audio streaming | Push + Eager | Real-time, continuous sample flow |
| Audio rendering | Pull + Lazy | Render to file offline |
| Animation preview | Push + Eager | See changes immediately |
| Animation render | Pull + Lazy | Render specific frames |

**Key insight:** Same graph may need different strategies in different contexts.

## Design: Evaluator Trait

Separate graph structure from evaluation strategy.

```rust
/// Graph is just data - nodes and wires
pub struct Graph {
    nodes: Vec<Node>,
    wires: Vec<Wire>,
    // No evaluation logic here
}

/// Evaluator determines how to execute the graph
pub trait Evaluator {
    /// Evaluate graph, producing requested outputs
    fn evaluate(
        &mut self,
        graph: &Graph,
        request: &EvalRequest,
        ctx: &mut EvalContext,
    ) -> Result<EvalResult>;
}

/// What outputs are needed
pub struct EvalRequest {
    pub outputs: Vec<NodeId>,
    // Future: specific attributes, resolution, etc.
}

/// Results of evaluation
pub struct EvalResult {
    pub values: HashMap<NodeId, Value>,
}
```

### Built-in Evaluators

```rust
/// Lazy pull-based evaluation (default)
/// Only computes nodes needed for requested outputs
pub struct LazyEvaluator {
    cache: EvalCache,
}

impl Evaluator for LazyEvaluator {
    fn evaluate(&mut self, graph: &Graph, request: &EvalRequest, ctx: &mut EvalContext) -> Result<EvalResult> {
        let mut results = HashMap::new();

        for &output in &request.outputs {
            let value = self.evaluate_node(graph, output, ctx)?;
            results.insert(output, value);
        }

        Ok(EvalResult { values: results })
    }
}

impl LazyEvaluator {
    fn evaluate_node(&mut self, graph: &Graph, node: NodeId, ctx: &mut EvalContext) -> Result<Value> {
        // Check cache
        let cache_key = self.compute_cache_key(graph, node);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Evaluate dependencies first (recursive pull)
        let inputs = self.evaluate_inputs(graph, node, ctx)?;

        // Compute this node
        let value = graph.nodes[node].compute(&inputs, ctx)?;

        // Cache result
        self.cache.insert(cache_key, value.clone());

        Ok(value)
    }
}
```

```rust
/// Eager push-based evaluation
/// Computes all nodes in topological order
pub struct EagerEvaluator;

impl Evaluator for EagerEvaluator {
    fn evaluate(&mut self, graph: &Graph, request: &EvalRequest, ctx: &mut EvalContext) -> Result<EvalResult> {
        let order = topological_sort(graph);
        let mut values = HashMap::new();

        // Compute all nodes in order
        for node in order {
            let inputs = gather_inputs(&values, graph, node);
            let value = graph.nodes[node].compute(&inputs, ctx)?;
            values.insert(node, value);
        }

        // Return requested outputs
        Ok(EvalResult {
            values: request.outputs.iter()
                .filter_map(|&id| values.remove(&id).map(|v| (id, v)))
                .collect()
        })
    }
}
```

```rust
/// Streaming evaluator for real-time audio/video
/// Processes in blocks, maintains inter-block state
pub struct StreamingEvaluator {
    block_size: usize,
    state: GraphState,  // feedback wire values
}

impl StreamingEvaluator {
    /// Process one block
    pub fn process_block(
        &mut self,
        graph: &Graph,
        inputs: &BlockInputs,
        ctx: &mut EvalContext,
    ) -> Result<BlockOutputs> {
        // Set block inputs
        // Evaluate graph (eager within block)
        // Update feedback state for next block
        // Return block outputs
    }
}
```

### Hybrid: Dirty Tracking

```rust
/// Tracks which nodes need recomputation
pub struct DirtyTracker {
    dirty: HashSet<NodeId>,
}

impl DirtyTracker {
    /// Mark node and all downstream nodes as dirty
    pub fn mark_dirty(&mut self, graph: &Graph, node: NodeId) {
        self.dirty.insert(node);
        for downstream in graph.downstream(node) {
            self.mark_dirty(graph, downstream);
        }
    }

    /// Check if node needs recomputation
    pub fn is_dirty(&self, node: NodeId) -> bool {
        self.dirty.contains(&node)
    }

    /// Clear dirty flag after recomputation
    pub fn mark_clean(&mut self, node: NodeId) {
        self.dirty.remove(&node);
    }
}

/// Lazy evaluator with dirty tracking
/// Push invalidation + Pull computation
pub struct IncrementalEvaluator {
    cache: EvalCache,
    dirty: DirtyTracker,
}

impl IncrementalEvaluator {
    /// Called when input changes
    pub fn invalidate(&mut self, graph: &Graph, changed_node: NodeId) {
        self.dirty.mark_dirty(graph, changed_node);
    }

    /// Evaluate, recomputing only dirty nodes
    fn evaluate_node(&mut self, graph: &Graph, node: NodeId, ctx: &mut EvalContext) -> Result<Value> {
        if !self.dirty.is_dirty(node) {
            if let Some(cached) = self.cache.get(&node) {
                return Ok(cached.clone());
            }
        }

        // Recompute
        let inputs = self.evaluate_inputs(graph, node, ctx)?;
        let value = graph.nodes[node].compute(&inputs, ctx)?;

        self.cache.insert(node, value.clone());
        self.dirty.mark_clean(node);

        Ok(value)
    }
}
```

## Shared Utilities

All evaluators share core utilities:

```rust
/// Topological sort for evaluation order
pub fn topological_sort(graph: &Graph) -> Vec<NodeId> {
    // Kahn's algorithm or DFS-based
}

/// Find all nodes that feed into given node
pub fn find_dependencies(graph: &Graph, node: NodeId) -> Vec<NodeId> {
    // Transitive closure of input wires
}

/// Find all nodes that depend on given node
pub fn find_dependents(graph: &Graph, node: NodeId) -> Vec<NodeId> {
    // Transitive closure of output wires
}

/// Compute cache key from node + input values
pub fn compute_cache_key(graph: &Graph, node: NodeId, inputs: &[Value]) -> CacheKey {
    // Hash node id + input hashes
}
```

## API Design

```rust
impl Graph {
    /// Evaluate with default evaluator (lazy)
    pub fn evaluate(&self, outputs: &[NodeId]) -> Result<EvalResult> {
        LazyEvaluator::new().evaluate(self, &EvalRequest { outputs: outputs.to_vec() }, &mut EvalContext::new())
    }

    /// Evaluate with specific evaluator
    pub fn evaluate_with<E: Evaluator>(
        &self,
        evaluator: &mut E,
        outputs: &[NodeId],
    ) -> Result<EvalResult> {
        evaluator.evaluate(self, &EvalRequest { outputs: outputs.to_vec() }, &mut EvalContext::new())
    }
}

// Usage
let result = graph.evaluate(&[mesh_output])?;  // lazy default

let mut streaming = StreamingEvaluator::new(512);
for block in audio_blocks {
    let output = streaming.process_block(&graph, &block)?;
}
```

## Cost Analysis

| Aspect | Cost | Notes |
|--------|------|-------|
| Trait design | Low | Simple interface |
| LazyEvaluator | ~200 LOC | Recursive eval + caching |
| EagerEvaluator | ~100 LOC | Topo sort + loop |
| StreamingEvaluator | ~300 LOC | Block processing + state |
| IncrementalEvaluator | ~250 LOC | Dirty tracking + cache |
| Shared utilities | ~200 LOC | Topo sort, dependency finding |
| Total | ~1000 LOC | Spread across multiple files |

**Maintenance:** Evaluators are independent. Changes to one don't affect others.

**Testing:** Each evaluator tested independently against same graph fixtures.

## Implementation Order

1. **LazyEvaluator** - Default, covers generation use case
2. **Shared utilities** - Topo sort, caching
3. **IncrementalEvaluator** - When live preview needed
4. **StreamingEvaluator** - When audio implemented
5. **EagerEvaluator** - Maybe never needed (lazy covers most cases)

## Recurrent Graphs

Evaluators must handle cycles (feedback wires). See [recurrent-graphs.md](./recurrent-graphs.md).

For recurrent graphs:
- **LazyEvaluator:** Error on cycles, or require explicit iteration count
- **StreamingEvaluator:** Natural fit - feedback values carried between blocks
- **IncrementalEvaluator:** Cycles complicate dirty tracking

```rust
impl LazyEvaluator {
    /// Evaluate recurrent graph for N iterations
    pub fn evaluate_recurrent(
        &mut self,
        graph: &Graph,
        outputs: &[NodeId],
        iterations: usize,
        ctx: &mut EvalContext,
    ) -> Result<EvalResult> {
        for _ in 0..iterations {
            // Evaluate one iteration
            // Update feedback wire values in ctx.state
        }
        // Return final values
    }
}
```

## Open Questions

1. **Cache invalidation:** Time-based expiry? Memory pressure? Manual clear?

2. **Parallel evaluation:** Independent subgraphs can run in parallel. Worth the complexity?

3. **Partial results:** Can we return partial results if evaluation fails midway?

4. **Progress reporting:** Long evaluations should report progress. Callback? Channel?

5. **Cancellation:** How to cancel in-progress evaluation?

## Summary

| Decision | Choice |
|----------|--------|
| Architecture | Evaluator trait, graph is just data |
| Default | LazyEvaluator (pull + lazy) |
| Extensibility | Users can implement custom evaluators |
| Shared code | Utilities for topo sort, caching, dirty tracking |

This follows our modular philosophy - don't force one strategy, provide sensible default, allow alternatives.
