# Editor Integration

How unshape serves as a backend for an interactive editor.

Two distinct performance scenarios arise, with very different requirements.

## Scenario 1: Interactive Editing (the common case)

The user tweaks a parameter on op N. The editor must show a result as fast as possible.

**What needs to run:** Only op N. Everything upstream is unchanged — its output is already cached as op N's input buffer. Everything downstream may need to re-run too (for non-destructive stacked ops), but same principle applies: one op at a time.

**What matters:** Single-op throughput. GPU acceleration per op. Writing into a pre-allocated buffer without allocating.

### The `apply_into` Interface

Each op should support writing into an existing buffer alongside the allocating `apply()` path:

```rust
pub struct GaussianBlur { pub radius: f32, pub sigma: f32 }

impl GaussianBlur {
    // Allocating form — convenient, not perf-critical
    pub fn apply(&self, input: &ImageBuffer) -> ImageBuffer {
        let mut out = ImageBuffer::new(input.width(), input.height());
        self.apply_into(input, &mut out);
        out
    }

    // Zero-alloc form — used by editor's interactive loop
    pub fn apply_into(&self, input: &ImageBuffer, output: &mut ImageBuffer) {
        // write pixels directly into output
    }

    // GPU form — input and output are already on GPU
    pub fn apply_gpu(&self, ctx: &GpuContext, input: &GpuTexture, output: &mut GpuTexture) {
        // dispatch compute shader, no CPU involvement
    }
}
```

### The Interactive Loop

```
user changes param on op N
  → mark op N dirty (and all downstream)
  → for each dirty op, in order:
      → call op.apply_gpu(ctx, cached_input, &mut cached_output)
      → (or apply_into for CPU ops)
  → composite final buffer to display
```

The cached input for op N is the cached output of op N-1 — it hasn't changed and doesn't move. Upstream GPU textures remain resident on the GPU across frames.

### What Makes This Fast

- **No allocation** — all buffers pre-allocated at session start, sized to canvas dimensions
- **No CPU↔GPU transfer** — input texture already on GPU from last frame; output written to GPU texture directly; display composites from GPU
- **Single dispatch** — one compute shader invocation per dirty op
- **Undo/redo is free** — undo just swaps which cached buffer is "current output of op N"; no recomputation needed as long as the cache is warm

The pipeline-level optimization described in [compute-backends.md](./compute-backends.md) is not needed here. The editor's cache *is* the optimization: ops that didn't change don't run.

## Scenario 2: History Replay

The user switches to a different history branch, or the buffer cache was evicted due to memory pressure. The editor must reconstruct the canvas state by re-running some or all ops from scratch.

**What needs to run:** A sequence of ops, potentially the full pipeline from an ancestor state.

**What matters:** Pipeline throughput. Minimizing CPU↔GPU transfers across a multi-op sequence. Fusing consecutive GPU ops into a single dispatch.

This is where the machinery in [compute-backends.md](./compute-backends.md) applies:

- `ExecutionPolicy::LocalFirst` — keep data on GPU between consecutive GPU ops, avoid readback
- Scheduler inserts `TransferToCpu`/`TransferToGpu` only at CPU↔GPU boundaries
- Pipeline fusion (future): consecutive image-expr GPU ops merged into one shader

Replay is the slow/background case. The editor can show a spinner, replay asynchronously, and repaint when done. It does not need to be interactive-speed.

## Buffer Layout for an Editor Session

```rust
pub struct EditorSession {
    /// One pre-allocated GPU texture per slot in the op pipeline
    /// Indexed by op position; ping-ponged for in-place filters
    gpu_buffers: Vec<GpuTexture>,

    /// CPU mirror for ops that require CPU fallback
    cpu_buffers: Vec<ImageBuffer>,

    /// Which buffers are valid (not dirty)
    valid: BitVec,
}
```

Filters that cannot read and write the same buffer (convolutions, distortions) use ping-pong: slot N → slot N+1 → slot N (alternating). The editor pre-allocates two physical buffers per logical slot when needed.

Multi-input ops (composite, blend) require three buffers: source A (slot N-1), source B (from another branch), destination (slot N). The session graph determines how many physical buffers are needed before any user interaction begins.

## Relationship to Existing Evaluators

The `IncrementalEvaluator` with dirty tracking (see [evaluation-strategy.md](./evaluation-strategy.md)) handles *when* to re-run nodes. The `apply_into` / `apply_gpu` interface handles *how* to run them without allocation. Both are needed; they compose cleanly.

An editor would use `IncrementalEvaluator` to track dirty state and call `apply_gpu` on each re-evaluated node, bypassing the allocating `Value`-based path entirely for the hot interactive loop. The full `Value`/graph machinery is used for export, batch processing, and replay — where allocation cost is amortized.

## Summary

| Scenario | Frequency | Key Requirement | Relevant Design |
|----------|-----------|-----------------|-----------------|
| Interactive (one op changed) | Every frame | Single-op GPU throughput, no alloc | `apply_into` / `apply_gpu` + buffer cache |
| History replay / cache miss | Occasional | Pipeline throughput, minimize transfers | `compute-backends.md`, `ExecutionPolicy` |
| Undo/redo (cache warm) | Frequent | Near-zero cost | Buffer swap only, no recomputation |
| Export / batch | Infrequent | Correctness, memory | Standard `apply()` path |
