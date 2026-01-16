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

## EvalContext

Context passed to node execution - provides environment info beyond just inputs.

```rust
/// Context for node execution
pub struct EvalContext {
    // === Control ===
    /// Cancellation token (check with is_cancelled())
    cancel: Option<CancellationToken>,
    /// Progress reporter for sub-node progress
    progress: Option<ProgressReporter>,

    // === Time ===
    /// Current time in seconds
    pub time: f64,
    /// Current frame number
    pub frame: u64,
    /// Delta time since last evaluation
    pub dt: f64,

    // === Quality hints ===
    /// True if this is a preview render (nodes can reduce quality)
    pub preview_mode: bool,
    /// Target resolution hint (for LOD decisions)
    pub target_resolution: Option<(u32, u32)>,

    // === Recurrent graphs ===
    /// Feedback wire values from previous iteration
    feedback_state: Option<FeedbackState>,

    // === Determinism ===
    /// Random seed for reproducible procedural generation
    pub seed: u64,
}

impl EvalContext {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().map_or(false, |t| t.is_cancelled())
    }

    pub fn report_progress(&self, completed: usize, total: usize) {
        if let Some(ref p) = self.progress {
            p.report(completed, total);
        }
    }
}
```

**DynNode signature includes context:**

```rust
pub trait DynNode: Send + Sync {
    fn execute(&self, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError>;
    // ...
}
```

This is analogous to audio's `AudioContext`, shader uniforms, or animation evaluation context.

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

## Cache Invalidation Policy

Pluggable via trait - different use cases need different policies.

```rust
/// Policy for cache retention and eviction
pub trait CachePolicy {
    /// Called after node evaluation - should we cache this result?
    fn should_cache(&self, node: NodeId, value: &Value) -> bool;

    /// Called before lookup - is this entry still valid?
    fn is_valid(&self, key: &CacheKey, entry: &CacheEntry) -> bool;

    /// Called on memory pressure or explicit clear
    fn evict(&mut self, cache: &mut EvalCache) -> usize; // bytes freed
}
```

**Built-in policies:**

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `KeepAll` | Never evict | Short-lived evaluation, batch export |
| `Lru { max_bytes }` | Evict least-recently-used | Long-running editor session |
| `Generational` | Keep recent, drop old | Interactive preview |
| `Manual` | Only evict on explicit call | User-controlled memory |

Default: `KeepAll` for simple cases, `Lru` when memory-bounded evaluation needed.

## Parallel Evaluation

Pluggable, disabled by default. Independent subgraphs can evaluate in parallel.

```rust
/// Controls parallel execution of independent nodes
pub trait ParallelPolicy {
    /// Given independent nodes, how to execute them?
    fn execute_parallel(
        &self,
        nodes: Vec<NodeId>,
        graph: &Graph,
        ctx: &EvalContext,
    ) -> Vec<Result<Value>>;
}

/// Sequential execution (default)
pub struct Sequential;

/// Rayon-based parallel execution
pub struct Rayon { min_nodes: usize }  // only parallelize if >= N independent nodes
```

**Rationale:** Parallelism adds overhead (task spawning, synchronization). For small graphs or graphs dominated by one expensive node, sequential is faster. User/evaluator can opt-in when profiling shows benefit.

## Progress Reporting

Two options offered (not a trait - they're structurally different):

```rust
pub struct EvalOptions {
    /// Synchronous callback, called between nodes
    pub progress_callback: Option<Box<dyn Fn(EvalProgress) + Send>>,

    /// Async channel for polling from another thread
    pub progress_channel: Option<flume::Sender<EvalProgress>>,
}

pub struct EvalProgress {
    pub completed_nodes: usize,
    pub total_nodes: usize,
    pub current_node: NodeId,
    pub elapsed: Duration,
}
```

**Why not a trait?** Callbacks and channels have fundamentally different semantics:
- Callback: synchronous, evaluator blocks while calling
- Channel: async, evaluator sends and continues

Offering both covers the common cases:
- Callback: simple single-threaded progress bar
- Channel: async UI thread polling while compute runs in background

## Cancellation

Configurable granularity - from simple node-boundary checks to true preemption.

```rust
pub enum CancellationMode {
    /// Check only between nodes (zero overhead, coarse)
    NodeBoundary,
    /// Pass token to nodes via EvalContext, nodes check periodically (cooperative)
    Cooperative,
    /// Spawn nodes as abortable tasks (has overhead, true preemption)
    Preemptive,
}

pub struct EvalOptions {
    pub cancel: Option<CancellationToken>,
    pub cancellation_mode: CancellationMode,  // default: Cooperative
    // ...
}
```

**Mode tradeoffs:**

| Mode | Overhead | Granularity | Node cooperation required |
|------|----------|-------------|---------------------------|
| `NodeBoundary` | Zero | Coarse (waits for node to finish) | No |
| `Cooperative` | Minimal | Fine (if nodes check) | Yes |
| `Preemptive` | Task spawn/abort | Immediate | No |

Usage:
```rust
let token = CancellationToken::new();

// Spawn evaluation
let handle = spawn(|| graph.evaluate_with_options(&outputs, EvalOptions {
    cancel: Some(token.clone()),
    cancellation_mode: CancellationMode::Cooperative,
    ..default()
}));

// Cancel from another thread
token.cancel();

// Evaluation returns Err(EvalError::Cancelled)
```

For `Cooperative` mode, nodes check via `EvalContext`:
```rust
fn execute(&self, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
    for i in 0..expensive_iterations {
        if ctx.is_cancelled() {
            return Err(GraphError::Cancelled);
        }
        // ... do work ...
    }
}
```

## Error Handling

Per-request configuration - different contexts want different behavior.

```rust
pub struct EvalRequest {
    pub outputs: Vec<NodeId>,
    pub error_handling: ErrorHandling,
}

pub enum ErrorHandling {
    /// Stop on first error, return Err (default)
    FailFast,

    /// Propagate errors as Value::Error, continue evaluation
    Propagate,

    /// Use fallback values for failed nodes
    /// Function can return None to propagate error for that specific node
    Fallback { default_fn: Box<dyn Fn(NodeId, &NodeError) -> Option<Value> + Send> },
}

impl ErrorHandling {
    /// Convenience: use a static map of defaults
    pub fn fallback_map(defaults: HashMap<NodeId, Value>) -> Self {
        Self::Fallback {
            default_fn: Box::new(move |id, _| defaults.get(&id).cloned())
        }
    }
}
```

**Use case fit:**

| Context | Error Handling | Rationale |
|---------|----------------|-----------|
| Final export | `FailFast` | Don't write corrupt output |
| Live preview | `Propagate` | Show what works, indicate errors |
| Validation | `FailFast` | Error is the point |
| Batch processing | `Propagate` | Process what you can, report failures |

When using `Propagate`, downstream nodes receive `Value::Error` as input and can:
- Propagate it (default behavior)
- Handle it (use fallback, skip operation, etc.)

**Note:** `ErrorHandling` uses a closure for `Fallback`, which isn't serializable. This is fine - error handling is evaluation-time configuration, not part of the graph structure. The graph (nodes, wires, parameters) is what gets serialized; evaluation options are chosen fresh each time.

**Open:** Need to validate this design against real use cases before committing.

## Enums vs Traits

Some extension points use traits, others use enums. The decision:

**Use trait when:** Many implementations expected from different sources (users, plugins, domains).

**Use enum when:** Few well-known variants that we control, or extensibility is provided another way (e.g., via function parameter).

| Type | Choice | Rationale |
|------|--------|-----------|
| `Evaluator` | Trait | Core extension point - users may implement custom evaluation strategies |
| `CachePolicy` | Trait | Domain-specific eviction logic likely (e.g., "keep meshes, evict textures") |
| `ParallelPolicy` | Trait | May want custom thread pool, priority scheduling, etc. |
| `CancellationMode` | Enum | Three modes cover the space; no obvious custom variant |
| `ErrorHandling` | Enum | `Fallback` takes a function, so custom logic goes there |

If we're wrong, enums → traits is an easy refactor.

## Tradeoffs & Open Decisions

Decisions we're not 100% on, may revisit:

### Cancellation Mode

**Choice:** Configurable via `CancellationMode` - `NodeBoundary`, `Cooperative`, or `Preemptive`.

**Default:** `Cooperative` - nodes check `ctx.is_cancelled()` periodically.

**Tradeoff:**
- `NodeBoundary`: Zero overhead, but expensive nodes block cancellation
- `Cooperative`: Minimal overhead, but requires node cooperation
- `Preemptive`: True preemption via task abort, but has spawn overhead and potential cleanup issues

**Revisit if:** `Preemptive` mode causes resource leaks or cleanup problems. May need more sophisticated abort handling (e.g., catch_unwind, cleanup hooks).

### Cache Key Strategy

**Choice:** Hash `(node_id, input_value_hashes)` using float-to-bits for floats.

**Tradeoff:** When `Value` gains large types (Mesh, Image), hashing becomes expensive. Options:
- Hash-on-construction (store hash in value)
- Identity/pointer comparison (fast but doesn't dedupe equivalent values)
- Skip caching for large values

**Revisit when:** We add `Value::Mesh`, `Value::Image`, etc.

### Channel vs Callback for Progress

**Choice:** Offer both - callback for simple cases, channel for async polling.

**Tradeoff:** Two mechanisms instead of one. Could unify with a trait, but callback and channel have fundamentally different semantics (sync vs async), so a trait would be awkward.

**Revisit if:** A third progress mechanism is needed, or if the dual API causes confusion.

### Parallel Evaluation Granularity

**Choice:** Parallelize at node level - independent nodes can run concurrently.

**Tradeoff:** For graphs with many tiny nodes, task overhead may exceed benefit. For graphs with few large nodes, parallelism is limited. Could parallelize within nodes instead (or additionally), but that requires node cooperation.

**Revisit if:** Profiling shows node-level parallelism isn't enough, or overhead is too high.

### Simple Dependencies vs Feature-Gated

**Choice:** Start with `std::sync::mpsc` and `Arc<AtomicBool>` for channels/cancellation.

**Tradeoff:** Less ergonomic than `flume`/`tokio-util`, but zero additional dependencies in resin-core.

**Revisit if:** The simple versions cause pain points (e.g., need `select!` across multiple channels).

## Summary

| Decision | Choice |
|----------|--------|
| Architecture | Evaluator trait, graph is just data |
| Default | LazyEvaluator (pull + lazy) |
| Extensibility | Users can implement custom evaluators |
| Shared code | Utilities for topo sort, caching, dirty tracking |
| Cache policy | Pluggable trait, `KeepAll` default |
| Parallelism | Pluggable trait, sequential default, opt-in parallel |
| Progress | Callback + channel options (not a trait) |
| Cancellation | CancellationToken, async+select internal |
| Error handling | Per-request enum (`FailFast` / `Propagate` / `Fallback`) |

This follows our modular philosophy - don't force one strategy, provide sensible default, allow alternatives.
