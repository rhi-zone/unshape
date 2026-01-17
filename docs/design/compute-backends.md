# Compute Backends

Where computation executes: CPU, GPU, SIMD, future accelerators.

Orthogonal to [evaluation-strategy.md](./evaluation-strategy.md) (when/how graph is traversed).

## Status: Implemented ✅

The compute backends system is fully implemented:

| Crate | What's There |
|-------|--------------|
| `resin-core` | `Value::Opaque`, `GraphValue` trait, `DataLocation`, `DynNode::as_any()`, `NodeExecutor` trait, `LazyEvaluator<E>` |
| `resin-backend` | `ComputeBackend` trait, `CpuBackend`, `BackendRegistry`, `ExecutionPolicy`, `Scheduler`, `BackendNodeExecutor` |
| `resin-gpu` | `GpuComputeBackend`, `GpuKernel` trait, kernels for noise and image-expr |

**Quick start (simple):**
```rust
use rhizome_resin_backend::{backend_evaluator, ExecutionPolicy};

// One-liner: CPU backend with auto policy
let mut evaluator = backend_evaluator(ExecutionPolicy::Auto);
let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;
```

**Quick start (with GPU):**
```rust
use rhizome_resin_backend::{BackendRegistry, BackendNodeExecutor, Scheduler, ExecutionPolicy, LazyEvaluator};
use rhizome_resin_gpu::{GpuComputeBackend, register_kernels};

// Setup registry with CPU (always) and GPU (if available)
let mut registry = BackendRegistry::with_cpu();
if let Ok(gpu) = GpuComputeBackend::new() {
    register_kernels(&gpu);  // Register noise, image-expr kernels
    registry.register(Arc::new(gpu));
}

// Create scheduler and evaluator
let scheduler = Scheduler::new(registry, ExecutionPolicy::Auto);
let mut evaluator = LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler));

// Evaluate - backends selected automatically per node
let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;
```

## The Problem

Not all nodes can run everywhere. Not all backends are available. Users want control without hardcoding specific backends.

**Dimensions:**

| Dimension | Examples |
|-----------|----------|
| Compute target | CPU scalar, CPU SIMD, GPU compute, GPU fragment |
| Data residency | CPU heap, GPU buffer, GPU texture, shared memory |
| Workload size | Single point, bulk (texture), streaming (audio) |
| Transfer cost | CPU↔GPU copies are expensive |

**Constraints:**
- Some nodes only have CPU impl
- Some only GPU (shader-based)
- Some have both
- GPU overhead kills single-point evaluation
- Data transfers dominate small workloads

## Design: Extensible Backend Trait

Backends register themselves, advertise capabilities. No closed enum.

```rust
/// A compute backend that can execute nodes
pub trait ComputeBackend: Send + Sync {
    /// Unique name for this backend
    fn name(&self) -> &str;

    /// What this backend can do
    fn capabilities(&self) -> BackendCapabilities;

    /// Can this backend execute the given node?
    fn supports_node(&self, node: &dyn DynNode) -> bool;

    /// Estimate cost for executing node with given workload
    /// Returns None if unsupported
    fn estimate_cost(
        &self,
        node: &dyn DynNode,
        workload: &WorkloadHint,
    ) -> Option<Cost>;

    /// Execute a node
    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Value, BackendError>;
}

/// What a backend can do
#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    /// General category
    pub kind: BackendKind,
    /// Supported value types for input/output
    pub value_types: Vec<ValueTypeId>,
    /// Can handle bulk operations efficiently
    pub bulk_efficient: bool,
    /// Can handle streaming (low latency)
    pub streaming_efficient: bool,
}

/// Broad category (for policy hints, not exhaustive matching)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    CpuSimd,
    Gpu,
    Custom(String),
}
```

## Built-in Backends

```rust
/// Default CPU backend - calls node's execute() directly
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str { "cpu" }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            value_types: vec![/* all */],
            bulk_efficient: false,
            streaming_efficient: true,
        }
    }

    fn supports_node(&self, _node: &dyn DynNode) -> bool {
        true  // CPU can run anything with a DynNode impl
    }

    fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
        Some(Cost {
            compute: workload.element_count as f64 * node.estimated_cost_per_element(),
            transfer: 0.0,  // no transfer for CPU
        })
    }

    fn execute(&self, node: &dyn DynNode, inputs: &[Value], ctx: &EvalContext) -> Result<Value, BackendError> {
        node.execute(inputs, ctx).map_err(BackendError::from)
    }
}
```

```rust
/// GPU compute backend via wgpu
pub struct GpuComputeBackend {
    ctx: Arc<GpuContext>,
    /// Registry of node type -> GPU kernel
    kernels: HashMap<TypeId, Box<dyn GpuKernel>>,
}

impl ComputeBackend for GpuComputeBackend {
    fn name(&self) -> &str { "gpu-compute" }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Gpu,
            value_types: vec![/* types with GPU support */],
            bulk_efficient: true,
            streaming_efficient: false,  // GPU has latency
        }
    }

    fn supports_node(&self, node: &dyn DynNode) -> bool {
        self.kernels.contains_key(&node.type_id())
    }

    fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
        if !self.supports_node(node) {
            return None;
        }
        Some(Cost {
            // GPU is fast for bulk, but has fixed overhead
            compute: 1.0 + workload.element_count as f64 * 0.001,
            transfer: workload.input_bytes as f64 * 0.01 + workload.output_bytes as f64 * 0.01,
        })
    }

    fn execute(&self, node: &dyn DynNode, inputs: &[Value], ctx: &EvalContext) -> Result<Value, BackendError> {
        let kernel = self.kernels.get(&node.type_id())
            .ok_or(BackendError::Unsupported)?;
        kernel.dispatch(&self.ctx, node, inputs, ctx)
    }
}

/// GPU kernel for a specific node type
pub trait GpuKernel: Send + Sync {
    fn dispatch(
        &self,
        ctx: &GpuContext,
        node: &dyn DynNode,
        inputs: &[Value],
        eval_ctx: &EvalContext,
    ) -> Result<Value, BackendError>;
}
```

## Backend Registry

```rust
/// Collection of available backends
pub struct BackendRegistry {
    backends: Vec<Arc<dyn ComputeBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self { backends: vec![] };
        registry.register(Arc::new(CpuBackend));  // CPU always available
        registry
    }

    pub fn register(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backends.push(backend);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ComputeBackend>> {
        self.backends.iter().find(|b| b.name() == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn ComputeBackend>> {
        self.backends.iter()
    }

    /// Find backends that can execute a node
    pub fn backends_for_node(&self, node: &dyn DynNode) -> Vec<&Arc<dyn ComputeBackend>> {
        self.backends.iter().filter(|b| b.supports_node(node)).collect()
    }
}
```

## Execution Policy

Policy describes intent, not specific backends.

```rust
/// How to choose backends for execution
pub enum ExecutionPolicy {
    /// Let scheduler pick based on workload + capabilities
    Auto,

    /// Prefer backends of this kind
    PreferKind(BackendKind),

    /// Use specific backend by name (escape hatch)
    Named(String),

    /// Minimize data transfers - keep data where it is
    LocalFirst,

    /// Lowest estimated cost wins
    MinimizeCost,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self::Auto
    }
}
```

## Scheduler

Matches policy to backends, considering node capabilities and data location.

```rust
pub struct BackendScheduler {
    registry: BackendRegistry,
    policy: ExecutionPolicy,
}

impl BackendScheduler {
    /// Choose backend for a node given current data locations
    pub fn select_backend(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        workload: &WorkloadHint,
    ) -> Result<&Arc<dyn ComputeBackend>, SchedulerError> {
        let candidates = self.registry.backends_for_node(node);
        if candidates.is_empty() {
            return Err(SchedulerError::NoBackendAvailable);
        }

        match &self.policy {
            ExecutionPolicy::Auto => {
                // Heuristic: GPU for bulk, CPU for small/streaming
                if workload.element_count > 1000 {
                    candidates.iter()
                        .find(|b| b.capabilities().kind == BackendKind::Gpu)
                        .or(candidates.first())
                        .copied()
                        .ok_or(SchedulerError::NoBackendAvailable)
                } else {
                    candidates.first().copied().ok_or(SchedulerError::NoBackendAvailable)
                }
            }

            ExecutionPolicy::PreferKind(kind) => {
                candidates.iter()
                    .find(|b| &b.capabilities().kind == kind)
                    .or(candidates.first())
                    .copied()
                    .ok_or(SchedulerError::NoBackendAvailable)
            }

            ExecutionPolicy::Named(name) => {
                candidates.iter()
                    .find(|b| b.name() == name)
                    .copied()
                    .ok_or(SchedulerError::BackendNotFound(name.clone()))
            }

            ExecutionPolicy::LocalFirst => {
                // Check where input data lives, prefer matching backend
                let input_location = infer_data_location(inputs);
                candidates.iter()
                    .find(|b| location_matches(&b.capabilities().kind, &input_location))
                    .or(candidates.first())
                    .copied()
                    .ok_or(SchedulerError::NoBackendAvailable)
            }

            ExecutionPolicy::MinimizeCost => {
                candidates.iter()
                    .filter_map(|b| b.estimate_cost(node, workload).map(|c| (b, c)))
                    .min_by(|(_, a), (_, b)| a.total().partial_cmp(&b.total()).unwrap())
                    .map(|(b, _)| b)
                    .copied()
                    .ok_or(SchedulerError::NoBackendAvailable)
            }
        }
    }
}
```

## Integration with EvalContext

```rust
/// Extended EvalContext with backend info
pub struct EvalContext {
    // ... existing fields from evaluation-strategy.md ...

    /// Available backends
    pub backends: Arc<BackendRegistry>,

    /// Execution policy for this evaluation
    pub policy: ExecutionPolicy,
}
```

## Workload Hints

Nodes can provide hints about their workload for better scheduling.

```rust
/// Hints about workload size for scheduling decisions
pub struct WorkloadHint {
    /// Number of elements to process
    pub element_count: usize,
    /// Approximate input data size in bytes
    pub input_bytes: usize,
    /// Approximate output data size in bytes
    pub output_bytes: usize,
}

impl WorkloadHint {
    pub fn single() -> Self {
        Self { element_count: 1, input_bytes: 64, output_bytes: 64 }
    }

    pub fn bulk(count: usize, bytes_per_element: usize) -> Self {
        Self {
            element_count: count,
            input_bytes: count * bytes_per_element,
            output_bytes: count * bytes_per_element,
        }
    }
}
```

## Cost Model

```rust
/// Estimated execution cost
#[derive(Clone, Debug)]
pub struct Cost {
    /// Compute time (relative units)
    pub compute: f64,
    /// Data transfer time (relative units)
    pub transfer: f64,
}

impl Cost {
    pub fn total(&self) -> f64 {
        self.compute + self.transfer
    }
}
```

## Data Residency

Values track where they live for transfer optimization.

```rust
/// Where data currently resides
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataLocation {
    Cpu,
    Gpu { device_id: u32 },
    /// Data exists in multiple locations (synced)
    Mirrored(Vec<DataLocation>),
}

/// Extended Value with location tracking
pub enum Value {
    // ... existing variants ...

    /// GPU texture (lives on GPU)
    GpuTexture(GpuTextureHandle),

    /// GPU buffer (lives on GPU)
    GpuBuffer(GpuBufferHandle),
}

impl Value {
    pub fn location(&self) -> DataLocation {
        match self {
            Value::GpuTexture(_) | Value::GpuBuffer(_) => DataLocation::Gpu { device_id: 0 },
            _ => DataLocation::Cpu,
        }
    }
}
```

## Registering GPU Kernels

Domain crates register GPU implementations for their nodes.

```rust
// In resin-noise
impl NoiseNode {
    pub fn register_gpu_kernel(backend: &mut GpuComputeBackend) {
        backend.register_kernel::<Self>(NoiseGpuKernel::new());
    }
}

struct NoiseGpuKernel {
    pipeline: Option<wgpu::ComputePipeline>,
}

impl GpuKernel for NoiseGpuKernel {
    fn dispatch(
        &self,
        ctx: &GpuContext,
        node: &dyn DynNode,
        inputs: &[Value],
        eval_ctx: &EvalContext,
    ) -> Result<Value, BackendError> {
        // Compile shader if needed, dispatch compute, return GpuTexture
    }
}
```

## Example Usage

```rust
// Setup
let mut backends = BackendRegistry::new();
if let Ok(gpu_ctx) = GpuContext::new() {
    let mut gpu = GpuComputeBackend::new(Arc::new(gpu_ctx));
    NoiseNode::register_gpu_kernel(&mut gpu);
    backends.register(Arc::new(gpu));
}

// Evaluation with auto backend selection
let ctx = EvalContext {
    backends: Arc::new(backends),
    policy: ExecutionPolicy::Auto,
    ..default()
};

let result = graph.evaluate_with_context(&[output_node], &ctx)?;

// Or explicit GPU preference
let ctx = EvalContext {
    policy: ExecutionPolicy::PreferKind(BackendKind::Gpu),
    ..default()
};
```

## Future Extensions

**SIMD backend:**
```rust
pub struct SimdBackend;
// Uses wide crate or explicit SIMD for bulk ops
```

**JIT backend:**
```rust
pub struct JitBackend {
    compiler: JitCompiler,
}
// Compiles node graphs to native code
```

**Distributed backend:**
```rust
pub struct RemoteBackend {
    connection: NetworkConnection,
}
// Offloads to remote compute nodes
```

## Open Questions

### 1. Data Transfers: Automatic vs Explicit

**The question:** Should the scheduler automatically insert CPU↔GPU copies, or require explicit transfer nodes?

| Approach | Pros | Cons |
|----------|------|------|
| **Automatic** | User doesn't think about it; "just works" | Hidden costs; hard to optimize; magic |
| **Explicit nodes** | Full control; costs visible in graph | Tedious; user must understand GPU model |
| **Both** | Best of both worlds | More API surface; two ways to do things |

**Leaning:** Both. Automatic by default (scheduler inserts transfers), but expose `TransferToGpu`/`TransferToCpu` nodes for users who want control. Automatic transfers get logged/visualized so users can see what's happening.

**Revisit if:** Automatic transfers cause performance surprises that are hard to debug.

### 2. Pipeline Fusion

**The question:** Multiple consecutive GPU nodes could fuse into one dispatch. Where does this optimization live?

| Approach | Pros | Cons |
|----------|------|------|
| **In scheduler** | Transparent; automatic | Complex scheduling logic |
| **In GpuBackend** | Backend knows its capabilities | Tight coupling to wgpu details |
| **Separate pass** | Clean separation; composable | Extra graph traversal |
| **JIT/codegen** | Maximum optimization potential | Heavy machinery |

**Leaning:** Separate optimization pass that runs before scheduling. Identifies fusable sequences (e.g., consecutive image ops), replaces with fused node. Similar to how `fuse_affine_chains()` works for audio.

**Open sub-questions:**
- How to detect fusable sequences? Pattern matching? Trait marker?
- How to generate fused shader? Template composition? Full codegen?
- What's the fusion boundary? Same data type? Same workgroup size?

**Revisit if:** Fusion patterns are too varied for a single pass approach.

### 3. GPU Memory Management

**The question:** Who owns GPU buffers/textures? When are they freed? When can they be reused?

| Approach | Pros | Cons |
|----------|------|------|
| **Reference counting** (`Arc<GpuBuffer>`) | Simple; automatic cleanup | Prevent reuse while refs held; ref cycles |
| **Explicit free** | Full control | Manual; error-prone; use-after-free risk |
| **Pool-based** | Efficient reuse; bounded memory | Pool sizing; fragmentation |
| **Frame-based** | Simple lifetime; batch free | Can't persist across frames |
| **Arena per evaluation** | Clear ownership; bulk free | Can't share across evaluations |

**Leaning:** Pool-based with ref counting. Evaluator owns a `GpuMemoryPool`. Allocations return `Arc<GpuBuffer>`. Pool tracks outstanding refs; reclaims when dropped. Explicit `pool.retain(buffer)` / `pool.release(buffer)` for fine control.

```rust
pub struct GpuMemoryPool {
    device: Arc<wgpu::Device>,
    buffers: Vec<PoolEntry>,
    // ...
}

impl GpuMemoryPool {
    /// Allocate or reuse a buffer of given size
    pub fn allocate(&mut self, size: usize) -> Arc<GpuBuffer>;

    /// Hint that buffer can be reused (even if refs remain)
    pub fn release(&mut self, buffer: &Arc<GpuBuffer>);

    /// Prevent buffer from being reused until explicitly released
    pub fn retain(&mut self, buffer: &Arc<GpuBuffer>);
}
```

**Open sub-questions:**
- How to size the pool? Fixed? Growing? User-configured?
- How to handle fragmentation? (small allocs interspersed with large)
- Should textures and buffers share a pool or be separate?

**Revisit if:** Pool overhead is significant, or lifetime tracking proves too complex.

### 4. Async Execution Model

**The question:** GPU work is inherently async. How should `execute()` behave?

| Approach | Pros | Cons |
|----------|------|------|
| **Blocking** | Simple API; no async runtime | Serializes CPU↔GPU; wastes parallelism |
| **Future-based** (`async fn`) | Idiomatic Rust async | Requires async runtime; viral |
| **Handle/fence** | Explicit control; no runtime | Manual polling; more API surface |
| **Submit-then-sync** | Batch submissions; one wait | Internal complexity; timing assumptions |

**Leaning:** Non-blocking dispatch with lightweight handle. Evaluator batches submissions internally, syncs at graph boundary (or when CPU needs result).

```rust
/// GPU execution handle - represents in-flight work
pub struct GpuFuture<T> {
    // ...
}

impl<T> GpuFuture<T> {
    /// Block until result ready
    pub fn wait(self) -> T;

    /// Check if ready without blocking
    pub fn is_ready(&self) -> bool;

    /// Poll for result (returns None if not ready)
    pub fn try_get(&mut self) -> Option<T>;
}

impl ComputeBackend for GpuComputeBackend {
    fn execute(&self, ...) -> Result<Value, BackendError> {
        // Internally: submit to queue, return GpuFuture wrapped in Value
        // Scheduler handles sync points
    }
}
```

**User-facing API:** Sync by default (evaluator waits at boundary). Power users can access handles for manual pipelining.

**Open sub-questions:**
- How to express "I need this value on CPU now" vs "keep it on GPU for next op"?
- How to handle errors from async GPU work? (might not know until sync)
- Should we integrate with `async`/`await` or stay sync with manual handles?

**Revisit if:** Manual handle management proves too error-prone, or async Rust becomes more ergonomic.

### 5. Backend Selection Granularity

**The question:** At what level do we select backends?

| Granularity | Pros | Cons |
|-------------|------|------|
| **Per-graph** | Simple; one decision | Can't mix CPU/GPU nodes |
| **Per-node** | Maximum flexibility | Overhead; many decisions |
| **Per-subgraph** | Balance; batch similar nodes | Subgraph detection complexity |
| **Per-value-type** | Natural boundaries | Inflexible for mixed workloads |

**Leaning:** Per-node with subgraph optimization. Scheduler decides per-node, but optimization pass groups consecutive same-backend nodes to minimize transitions.

**Revisit if:** Per-node overhead is measurable, or subgraph grouping proves fragile.

## Tradeoffs & Tentative Decisions

Decisions we've made but aren't 100% committed to.

### Trait vs Enum for ComputeBackend

**Choice:** Trait (`dyn ComputeBackend`).

**Rationale:** Extensibility. New backends (CUDA, WebGPU, remote) can be added without modifying core. Follows "plugin/trait systems > hardcoded switches" principle.

**Alternative considered:** Enum. Simpler, no vtable overhead, exhaustive matching. But closed — adding a backend means changing the enum everywhere.

**Revisit if:** Dynamic dispatch overhead is measurable in hot paths, or we find that backends are actually a small fixed set.

### BackendKind as Enum

**Choice:** `BackendKind` is an enum with `Custom(String)` escape hatch.

**Rationale:** Policies need to express "prefer GPU" without naming specific backends. A few broad categories (Cpu, CpuSimd, Gpu) cover 90% of cases. `Custom` allows extension without being fully open.

**Alternative considered:** Capability flags instead of kinds. More flexible but harder to express simple preferences like "prefer GPU."

**Revisit if:** Categories prove too coarse, or `Custom` becomes overused.

### ExecutionPolicy as Enum

**Choice:** Enum with fixed variants (Auto, PreferKind, Named, LocalFirst, MinimizeCost).

**Rationale:** Policies are intent, not implementation. A small set of intents covers most use cases. Unlike backends (which are implementations), policies don't need extensibility.

**Alternative considered:** Policy trait. More flexible but overkill — what custom policy would someone write that isn't covered by MinimizeCost + custom Cost impl?

**Revisit if:** We discover common policy patterns that don't fit the enum.

### Cost Model Simplicity

**Choice:** Simple `Cost { compute: f64, transfer: f64 }`.

**Rationale:** Start simple. Actual cost modeling is hard; better to have something usable than something perfect.

**Alternative considered:** Richer model (latency vs throughput, memory bandwidth, queue depth). More accurate but complex to populate and reason about.

**Revisit if:** Simple model makes bad decisions in practice.

### Workload Hints from Caller

**Choice:** Caller provides `WorkloadHint` to scheduler.

**Rationale:** Caller knows workload size (e.g., texture dimensions). Node doesn't always know until execution.

**Alternative considered:** Nodes estimate their own workload from inputs. More automatic but requires nodes to understand input structure before executing.

**Revisit if:** Caller-provided hints are often wrong or tedious to provide.

### Scheduler in Core vs Separate Crate

**Choice:** Scheduler lives alongside `Evaluator` in core (or a thin `resin-backend` crate).

**Rationale:** Scheduling is part of evaluation. Separating would create awkward dependencies.

**Alternative considered:** Fully separate `resin-scheduler` crate. Cleaner separation but artificial — scheduler needs deep integration with evaluator.

**Revisit if:** Scheduler grows complex enough to warrant isolation.

## Crate Structure

### Dependency Graph

```
resin-core ─────────────────┐
     │                      │
     │ (optional)           │
     ▼                      │
resin-backend ◄─────────────┘
     │
     │ (optional, feature-gated)
     ▼
resin-gpu
```

- `resin-core` has Value, DynNode, Graph, Evaluator — no GPU deps
- `resin-backend` has ComputeBackend trait, BackendRegistry, ExecutionPolicy — abstractions only
- `resin-gpu` implements GpuComputeBackend, depends on wgpu

### resin-core changes

**Value with type-erased opaque data:**

```rust
// value.rs
use std::any::Any;
use std::sync::Arc;

/// Marker trait for values that can flow through graphs
pub trait GraphValue: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_boxed(&self) -> Box<dyn GraphValue>;

    /// Where this value currently resides
    fn location(&self) -> DataLocation { DataLocation::Cpu }
}

/// Where data lives
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum DataLocation {
    #[default]
    Cpu,
    Gpu { device_id: u32 },
}

#[derive(Debug, Clone)]
pub enum Value {
    // Primitives (inline, no allocation)
    F32(f32),
    F64(f64),
    I32(i32),
    Bool(bool),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),

    // Large/opaque values (heap allocated, type-erased)
    // Use for Image, Mesh, GpuTexture, AudioBuffer, etc.
    Opaque(Arc<dyn GraphValue>),
}

impl Value {
    /// Create an opaque value from any GraphValue
    pub fn opaque<T: GraphValue>(value: T) -> Self {
        Value::Opaque(Arc::new(value))
    }

    /// Try to downcast an opaque value
    pub fn downcast_ref<T: GraphValue + 'static>(&self) -> Option<&T> {
        match self {
            Value::Opaque(v) => v.as_any().downcast_ref(),
            _ => None,
        }
    }

    /// Where this value lives
    pub fn location(&self) -> DataLocation {
        match self {
            Value::Opaque(v) => v.location(),
            _ => DataLocation::Cpu,
        }
    }
}
```

**ValueType becomes extensible:**

```rust
// value.rs
use std::any::TypeId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueType {
    // Built-in primitives
    F32,
    F64,
    I32,
    Bool,
    Vec2,
    Vec3,
    Vec4,

    // Type-erased (registered types)
    Custom { type_id: TypeId, name: &'static str },
}

impl ValueType {
    pub fn of<T: 'static>(name: &'static str) -> Self {
        ValueType::Custom { type_id: TypeId::of::<T>(), name }
    }
}
```

### resin-backend (new crate)

Thin abstraction layer — no heavy dependencies.

```rust
// lib.rs
use resin_core::{Value, DynNode, EvalContext, DataLocation};
use std::sync::Arc;

/// A compute backend that can execute nodes
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;
    fn supports_node(&self, node: &dyn DynNode) -> bool;
    fn estimate_cost(&self, node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost>;
    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Value, BackendError>;
}

#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub bulk_efficient: bool,
    pub streaming_efficient: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    CpuSimd,
    Gpu,
    Custom(String),
}

#[derive(Clone, Debug)]
pub struct WorkloadHint {
    pub element_count: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct Cost {
    pub compute: f64,
    pub transfer: f64,
}

#[derive(Debug)]
pub enum BackendError {
    Unsupported,
    ExecutionFailed(String),
    TransferFailed(String),
}

/// Registry of available backends
pub struct BackendRegistry {
    backends: Vec<Arc<dyn ComputeBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self { backends: vec![] }
    }

    pub fn register(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backends.push(backend);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ComputeBackend>> {
        self.backends.iter().find(|b| b.name() == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn ComputeBackend>> {
        self.backends.iter()
    }
}

/// Execution policy
#[derive(Clone, Debug, Default)]
pub enum ExecutionPolicy {
    #[default]
    Auto,
    PreferKind(BackendKind),
    Named(String),
    LocalFirst,
    MinimizeCost,
}

/// CPU backend - always available, uses DynNode::execute directly
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str { "cpu" }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            bulk_efficient: false,
            streaming_efficient: true,
        }
    }

    fn supports_node(&self, _node: &dyn DynNode) -> bool {
        true // CPU can run any node
    }

    fn estimate_cost(&self, _node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
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
    ) -> Result<Value, BackendError> {
        node.execute(inputs, ctx)
            .map(|outputs| outputs.into_iter().next().unwrap_or(Value::Bool(false)))
            .map_err(|e| BackendError::ExecutionFailed(e.to_string()))
    }
}
```

### resin-gpu changes

Implements `ComputeBackend`, provides GPU-specific types.

```rust
// backend.rs
use resin_backend::{ComputeBackend, BackendCapabilities, BackendKind, WorkloadHint, Cost, BackendError};
use resin_core::{Value, DynNode, EvalContext, GraphValue, DataLocation};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU texture that implements GraphValue
pub struct GpuTextureValue {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
    device_id: u32,
}

impl GraphValue for GpuTextureValue {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_boxed(&self) -> Box<dyn GraphValue> {
        panic!("GPU textures cannot be cloned directly") // or implement copy
    }
    fn location(&self) -> DataLocation {
        DataLocation::Gpu { device_id: self.device_id }
    }
}

/// GPU kernel for a specific node type
pub trait GpuKernel: Send + Sync {
    fn execute(
        &self,
        ctx: &GpuContext,
        node: &dyn DynNode,
        inputs: &[Value],
        eval_ctx: &EvalContext,
    ) -> Result<Value, BackendError>;
}

/// GPU compute backend
pub struct GpuComputeBackend {
    ctx: Arc<GpuContext>,
    kernels: HashMap<TypeId, Box<dyn GpuKernel>>,
    device_id: u32,
}

impl GpuComputeBackend {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        Self {
            ctx,
            kernels: HashMap::new(),
            device_id: 0,
        }
    }

    pub fn register_kernel<N: DynNode + 'static>(&mut self, kernel: impl GpuKernel + 'static) {
        self.kernels.insert(TypeId::of::<N>(), Box::new(kernel));
    }
}

impl ComputeBackend for GpuComputeBackend {
    fn name(&self) -> &str { "gpu-compute" }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Gpu,
            bulk_efficient: true,
            streaming_efficient: false,
        }
    }

    fn supports_node(&self, node: &dyn DynNode) -> bool {
        // Check if we have a kernel registered for this node type
        // Note: needs node.type_id() or similar
        false // TODO: implement type checking
    }

    fn estimate_cost(&self, _node: &dyn DynNode, workload: &WorkloadHint) -> Option<Cost> {
        Some(Cost {
            compute: 1.0 + workload.element_count as f64 * 0.001,
            transfer: (workload.input_bytes + workload.output_bytes) as f64 * 0.01,
        })
    }

    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Value, BackendError> {
        // Look up kernel, dispatch
        Err(BackendError::Unsupported) // TODO
    }
}
```

### EvalContext integration

```rust
// In resin-core, EvalContext stays simple
pub struct EvalContext {
    pub time: f64,
    pub frame: u64,
    pub dt: f64,
    pub preview_mode: bool,
    pub target_resolution: Option<(u32, u32)>,
    pub seed: u64,
    // ... existing fields
}

// In resin-backend, extend with a wrapper or separate type
pub struct BackendEvalContext<'a> {
    pub eval: &'a EvalContext,
    pub backends: &'a BackendRegistry,
    pub policy: ExecutionPolicy,
}
```

### Scheduler integration with Evaluator

```rust
// In resin-backend
use resin_core::{Graph, Evaluator, EvalRequest, EvalResult};

/// Evaluator that uses backend scheduling
pub struct BackendAwareEvaluator {
    inner: Box<dyn Evaluator>,
    registry: Arc<BackendRegistry>,
    policy: ExecutionPolicy,
}

impl BackendAwareEvaluator {
    pub fn new(
        inner: impl Evaluator + 'static,
        registry: Arc<BackendRegistry>,
        policy: ExecutionPolicy,
    ) -> Self {
        Self {
            inner: Box::new(inner),
            registry,
            policy,
        }
    }
}

impl Evaluator for BackendAwareEvaluator {
    fn evaluate(
        &mut self,
        graph: &Graph,
        request: &EvalRequest,
        ctx: &mut EvalContext,
    ) -> Result<EvalResult, GraphError> {
        // For each node:
        // 1. Select backend based on policy
        // 2. Insert transfers if needed
        // 3. Execute via backend
        // 4. Return results
        todo!()
    }
}
```

### Usage example

```rust
use resin_core::{Graph, LazyEvaluator, EvalContext};
use resin_backend::{BackendRegistry, BackendAwareEvaluator, CpuBackend, ExecutionPolicy};
use resin_gpu::{GpuContext, GpuComputeBackend};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build graph
    let graph = Graph::new();
    // ... add nodes ...

    // Setup backends
    let mut registry = BackendRegistry::new();
    registry.register(Arc::new(CpuBackend));

    if let Ok(gpu_ctx) = GpuContext::new() {
        let gpu = GpuComputeBackend::new(Arc::new(gpu_ctx));
        registry.register(Arc::new(gpu));
    }

    // Create backend-aware evaluator
    let lazy = LazyEvaluator::new();
    let mut evaluator = BackendAwareEvaluator::new(
        lazy,
        Arc::new(registry),
        ExecutionPolicy::Auto,
    );

    // Evaluate - scheduler picks backends automatically
    let mut ctx = EvalContext::new();
    let result = evaluator.evaluate(&graph, &request, &mut ctx)?;

    Ok(())
}
```

### Migration path

1. **Phase 1:** Add `Value::Opaque` and `GraphValue` trait to resin-core ✅
2. **Phase 2:** Create `resin-backend` with traits and `CpuBackend` ✅
3. **Phase 3:** Move/refactor `resin-gpu` to implement `GpuComputeBackend` ✅
4. **Phase 4:** Add `Scheduler` for backend selection ✅
5. **Phase 5:** Add `BackendAwareEvaluator` that implements `Evaluator` ✅
6. **Phase 6:** Register GPU kernels for existing GPU ops (noise, image-expr) ✅

### Implementation Notes

**Actual `GpuKernel` signature:**
```rust
pub trait GpuKernel: Send + Sync {
    fn execute(
        &self,
        ctx: &GpuContext,
        inputs: &[Value],
        eval_ctx: &EvalContext,
    ) -> Result<Vec<Value>, GpuError>;
}
```
Note: Kernels receive `inputs` not `node`. This is a known limitation - kernels
can't access node parameters (like expressions stored in `RemapUvNode`).
See TODO.md for planned fix: pass `node: &dyn DynNode` to kernels.

**DynNode requirement:**
Nodes must implement `fn as_any(&self) -> &dyn Any` for kernel lookup via `TypeId`.
The `#[derive(Node)]` macro generates this automatically.

**CPU fallback pattern:**
Nodes like `NoiseTextureNode` have CPU fallback in `execute()` and GPU path via kernel.
Nodes like `RemapUvNode` require GPU (return error from `execute()`).

## Summary

| Aspect | Design |
|--------|--------|
| Backend abstraction | `ComputeBackend` trait |
| Extensibility | Registry, any impl can register |
| Policy | Intent-based enum, not backend names |
| Scheduling | `Scheduler` matches policy to capabilities |
| Evaluation | `LazyEvaluator<BackendNodeExecutor>` via `NodeExecutor` trait |
| Data location | Tracked via `GraphValue::location()` |
| Cost model | Estimate compute + transfer |

This keeps the core evaluation strategy unchanged while allowing heterogeneous execution across CPU/GPU/future backends.

## Architecture Notes

**Kernel node access:** `GpuKernel::execute()` receives `&dyn DynNode`, so kernels can
downcast to access node-specific parameters (e.g., `RemapUvNode::expr`).
