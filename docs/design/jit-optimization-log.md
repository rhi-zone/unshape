# JIT Optimization Log

Experiments with Cranelift JIT for audio graph compilation.

## Baseline Measurements (44100 samples = 1 second of audio)

| Benchmark | Time | Description |
|-----------|------|-------------|
| `gain_rust_1sec` | 15 µs | Pure Rust loop: `sample * gain` |
| `gain_block_1sec` | 2.6-3.1 µs | `BlockProcessor` trait on `Gain` node |
| `gain_jit_1sec` | 61-66 µs | Per-sample JIT (`compile_gain`) |

The native Rust `BlockProcessor` is fastest because LLVM can vectorize the simple multiply loop.

## Experiment 1: Block-based JIT with external context advancement

**Goal:** Amortize JIT function call overhead by processing blocks instead of per-sample.

**Implementation:**
- `compile_graph()` generates a loop that processes all samples
- External `advance_context()` function called per sample to update `ctx.time` and `ctx.sample_index`

**Result:** `gain_jit_block_1sec` = 71 µs

**Analysis:** Still slower than per-sample JIT (61 µs). The external function call per sample adds overhead, but at least it's in the same ballpark.

## Experiment 2: Inline context advancement

**Goal:** Eliminate external function call overhead by inlining the context update.

**Implementation:**
- Added `#[repr(C)]` to `AudioContext` and `GraphState` for predictable layout
- Moved `ctx` to first field in `GraphState` (offset 0)
- Generated inline load/add/store instructions for `time += dt` and `sample_index += 1`

**Result:** `gain_jit_block_1sec` = 120-121 µs (SLOWER!)

**Analysis:** Counterintuitively, the inline version is ~70% slower than the external call. Possible causes:
1. Memory access pattern preventing optimization
2. Increased code size in the hot loop affecting I-cache
3. Cranelift not optimizing the loads/stores well
4. The `dt` load could be hoisted outside the loop (it's constant) but isn't

## Experiment 3: Skip context advancement for pure-math graphs

**Goal:** Avoid unnecessary work when no nodes need the context.

**Implementation:**
- Check `has_stateful` flag from `analyze_graph()`
- Only emit context advancement code if there are stateful nodes

**Result:** `gain_jit_block_1sec` = 20 µs (down from 121 µs!)

**Analysis:** 6x improvement for pure-math graphs! But still 8x slower than native (2.6 µs).

**Important caveat:** This is not a fair comparison:
- Native `BlockProcessor` DOES advance context (`ctx.advance()`) every sample
- But LLVM can inline, keep values in registers, and **vectorize** the loop
- JIT does scalar operations without SIMD

## Why Native is Faster

Native Rust's 2.6 µs vs JIT's 20 µs:
- Native: LLVM can vectorize `output[i] = input[i] * gain` into SIMD (8 samples at once)
- JIT: Cranelift generates scalar code (1 sample at a time)
- Native: Context values stay in registers
- JIT: Every access is a memory load/store

## Next Steps to Try

1. **Hoist dt outside the loop** - Load once before loop, only update time/index inside
2. **Use registers for time tracking** - Keep time in a register, write back once at end
3. **Generate SIMD code** - Use Cranelift's SIMD instructions for pure-math chains
4. **Profile the generated code** - Look at actual x86 output to understand what's happening

## Observations

- For simple operations like Gain, native Rust with `BlockProcessor` beats JIT by 20-40x
- JIT value proposition is for:
  - Complex graph routing that can be compiled away
  - Dynamically-built graphs where compile-time optimization isn't possible
  - Mixed graphs where inlining stateful+pure nodes together helps
- The per-sample JIT functions (`compile_gain`, `compile_tremolo`) are simpler and faster than the graph compiler, but don't scale to arbitrary graphs
