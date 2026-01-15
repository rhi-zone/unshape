# Time Models

How time-dependent computation works across resin domains.

## The Problem

"Graph context provides time" is too simple. Different domains have fundamentally different relationships with time:

- Can you seek to arbitrary time? (scrubbing)
- Does output depend on history? (state)
- Is time implicit or explicit?
- Can you parallelize across time?

## Time Models

### 1. Stateless (Pure)

Output is a pure function of inputs + time. No history dependence.

```rust
fn eval(&self, inputs: &Inputs, time: f32) -> Output
```

**Properties:**
- Can seek to any time instantly
- Can evaluate times in any order
- Can parallelize across time (render frames in parallel)
- Cacheable (same inputs + time = same output)

**Examples:**
- Procedural noise: `noise(pos, time)`
- Oscillators: `sin(frequency * time + phase)`
- Blend shapes: `lerp(shape_a, shape_b, time)`
- Easing functions: `ease(t)`

### 2. Stateful (Sequential)

Output depends on previous state. Must process in order.

```rust
fn step(&mut self, inputs: &Inputs, dt: f32) -> Output
```

**Properties:**
- Cannot seek without computing all prior frames
- Must evaluate in order
- Cannot parallelize across time
- State must be stored/managed

**Examples:**
- Physics simulation: position depends on velocity depends on forces over time
- Audio filters: IIR filters have memory
- Particle systems: particle positions evolve
- Delays/reverbs: buffer of past samples

### 3. Implicit Time (Streaming)

Time is position in a stream. No explicit time parameter.

```rust
fn process(&mut self, block: &[Sample]) -> Vec<Sample>
```

**Properties:**
- Time = sample_index / sample_rate
- Naturally sequential (audio must play in order)
- Block-based processing for efficiency

**Examples:**
- Audio streams
- Video frame sequences

### 4. Baked (Cached Sequential)

Pre-computed stateful simulation stored as stateless data.

```rust
// Simulate once
let cache = physics.simulate(0.0..10.0, dt=1/60);

// Sample anywhere (stateless lookup)
let pose = cache.sample(time);  // interpolates cached frames
```

**Properties:**
- Pay simulation cost once
- Then seek freely
- Memory cost (store all frames)
- Lossy if sampling between cached frames

**Examples:**
- Animation caches (Alembic)
- Physics caches
- Fluid simulation caches

## Per-Domain Analysis

### Textures

**Dominant model:** Stateless

Time is just another dimension. 4D noise `noise(x, y, z, t)` can be sampled at any t.

```rust
trait AnimatedTexture {
    fn sample(&self, uv: Vec2, time: f32) -> Color;
}
```

**Exception:** Texture sequences (flipbook animation) are technically baked.

### Mesh Generation

**Dominant model:** Stateless

Procedural mesh is pure function of parameters.

```rust
let mesh = generate_terrain(seed, time);  // same inputs = same mesh
```

**Exception:** Erosion simulation is stateful (iterative process).

### Audio Synthesis

**Dominant model:** Stateless (surprisingly)

Oscillators, wavetables, FM synthesis are all pure functions of phase/time.

```rust
fn oscillator(frequency: f32, time: f32) -> f32 {
    sin(frequency * time * TAU)
}
```

Phase accumulation *looks* stateful but is really:
```rust
// "Stateful" version
self.phase += frequency * dt;
output = sin(self.phase);

// Equivalent stateless version
output = sin(frequency * time * TAU);
```

### Audio Effects

**Dominant model:** Stateful

Filters, delays, reverbs all have memory.

```rust
struct LowPassFilter {
    prev_output: f32,  // state!
}

impl LowPassFilter {
    fn process(&mut self, input: f32) -> f32 {
        self.prev_output = self.prev_output * 0.9 + input * 0.1;
        self.prev_output
    }
}
```

**Implication:** Audio effect chains cannot seek. Must process from start (or accept discontinuity).

### Rigging / Animation

**Dominant model:** Stateless

Pose is function of time + parameters.

```rust
fn evaluate_rig(skeleton: &Skeleton, time: f32) -> Pose {
    // blend animations, apply IK, etc.
    // no state, just computation
}
```

**Exception:** Procedural secondary motion (jiggle bones) is often stateful.

### Physics

**Dominant model:** Stateful

Cannot skip frames. State evolves over time.

```rust
struct PhysicsWorld {
    bodies: Vec<RigidBody>,  // positions, velocities
}

impl PhysicsWorld {
    fn step(&mut self, dt: f32) {
        // integrate velocities, resolve collisions
        // state changes!
    }
}
```

**Solution for seeking:** Bake to cache, or re-simulate from start.

## Mixing Models in Graphs

**Problem:** What happens when stateless and stateful nodes connect?

```
[Noise (stateless)] -> [Filter (stateful)] -> [Output]
```

The graph becomes stateful. Downstream of any stateful node inherits statefulness.

**Options:**

### A. Track statefulness in type system

```rust
trait StatelessNode {
    fn eval(&self, ctx: &EvalContext) -> Output;
}

trait StatefulNode {
    fn step(&mut self, ctx: &EvalContext, dt: f32) -> Output;
}

// Graph is stateless only if ALL nodes are stateless
```

Pros: Compile-time guarantees
Cons: Two parallel hierarchies, complex

### B. Runtime flag

```rust
trait Node {
    fn is_stateful(&self) -> bool;
    fn eval(&self, ctx: &mut EvalContext) -> Output;
}

// Context provides state storage
struct EvalContext {
    time: f32,
    dt: f32,
    state: StateStore,  // nodes store state here by ID
}
```

Pros: Simpler API
Cons: Runtime checks, can't statically prove seekability

### C. State is always external

```rust
// Nodes never hold state. State passed in/out explicitly.
trait Node {
    type State: Default;
    fn eval(&self, input: Input, state: &mut Self::State, dt: f32) -> Output;
}

// Stateless nodes just use `()` for state
impl Node for Noise {
    type State = ();
    fn eval(&self, input: Input, _state: &mut (), _dt: f32) -> Output { ... }
}
```

Pros: Explicit, state management is caller's problem
Cons: Verbose for simple stateless nodes

## How Time Reaches Fields

Two options considered:

**Option A: Time as extra dimension**
```rust
// Animated 2D = 3D field (x, y, t)
impl Field<Vec3, Color> for AnimatedNoise { ... }
```

Pros: Pure, mathematically clean, seekable by definition
Cons: Type changes for animated vs static, proliferates dimensions

**Option B: Time in EvalContext (chosen)**
```rust
trait Field<I, O> {
    fn sample(&self, input: I, ctx: &EvalContext) -> O;
}
```

Pros: Same type for static/animated, extensible context, proven pattern
Cons: Context parameter even when unused

**Decision: EvalContext (Option B)**

Shadertoy validates this pattern. Their shader inputs are essentially EvalContext:
- `iTime` - time in seconds
- `iTimeDelta` - dt
- `iFrame` - frame number
- `iResolution` - output resolution
- `iSampleRate` - for audio shaders

Time is context, not coordinate. Position (uv/fragCoord) is the input. Battle-tested in millions of shaders.

## EvalContext Design

```rust
struct EvalContext<'a> {
    // Time info (Shadertoy-style)
    time: f32,              // absolute time in seconds (iTime)
    dt: f32,                // delta time since last eval (iTimeDelta)
    frame: u64,             // frame number (iFrame)

    // Resolution (when materializing)
    resolution: UVec2,      // output resolution (iResolution)

    // Audio-specific
    sample_rate: f32,       // samples per second (iSampleRate)

    // For stateful nodes
    state: &'a mut StateStore,

    // For resolving references
    assets: &'a AssetStore,

    // For caching
    cache: &'a mut EvalCache,
}
```

Fields that don't need time simply ignore `ctx`. No overhead for static fields beyond the parameter.

## Decisions

1. **How time reaches fields**: EvalContext (Shadertoy pattern). See above.

2. **State serialization**: Solved by recurrent graphs - feedback edges ARE the state. `GraphSnapshot { graph, feedback_state }` captures everything.

3. **Seeking stateful graphs**: User choice via enum:
   ```rust
   enum SeekBehavior {
       Resimulate,    // correct, slow - replay from start
       Discontinuity, // fast, may glitch - jump directly
       Error,         // fail-safe - refuse to seek
   }
   ```
   Default: `Discontinuity` for interactive preview, `Resimulate` for final render.

4. **Delay granularity**: Per-edge, configurable (from recurrent-graphs):
   ```rust
   enum Delay {
       Samples(u32),     // audio: z⁻ⁿ
       Frames(u32),      // animation: previous N frames
       Duration(f32),    // explicit seconds
   }
   ```

## Audio Block Processing

Audio processes in blocks (128-1024 samples) for efficiency, not per-sample.

```rust
fn process_block(&mut self, input: &[f32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
    // SIMD-friendly, cache-friendly
}
```

**How this fits with recurrent graphs:**
- Block = iteration unit
- Feedback edges carry state between blocks
- Within a block: samples can be parallel (SIMD)
- Control-rate parameters: update once per block, not per-sample

```rust
struct AudioGraphState {
    feedback_values: HashMap<WireId, Value>,
    block_size: usize,
}

fn process_audio_block(
    graph: &Graph,
    input: &[f32],
    state: &mut AudioGraphState,
) -> Vec<f32> {
    // 1. Read feedback from previous block
    // 2. Process all samples in block (vectorized where possible)
    // 3. Write feedback for next block
}
```

**Block size considerations:**
- Smaller blocks = lower latency, more overhead
- Larger blocks = higher latency, better throughput
- Typical: 128-512 for live, 1024+ for offline render

## Mixed Sample Rates

Different domains run at different rates:
- Audio: 44.1kHz, 48kHz, 96kHz
- Video: 24fps, 30fps, 60fps
- Animation: 30-60Hz
- Physics: 60-120Hz (fixed timestep)

**Problem:** What happens when signals cross rate boundaries?

```
[LFO @ 60Hz] ──?──> [Filter @ 48kHz]
[Audio @ 48kHz] ──?──> [Envelope display @ 60fps]
```

**Options:**

### A: Explicit conversion nodes

```
[LFO] -> [Upsample 60->48000] -> [Filter]
[Audio] -> [Downsample 48000->60] -> [Display]
```

Pros: Clear, no magic, user controls quality
Cons: Verbose, easy to forget

### B: Automatic conversion on wires

```rust
// Wire knows source and dest rates, converts automatically
struct Wire {
    from: NodeId,
    to: NodeId,
    rate_conversion: Option<RateConversion>,
}

enum RateConversion {
    Upsample { method: Interpolation },
    Downsample { method: Decimation },
}
```

Pros: Less boilerplate
Cons: Hidden behavior, quality not obvious

### C: Rate as node property, graph validates

```rust
struct Node {
    // ...
    sample_rate: SampleRate,  // Hz as f64, not enum
}

// Graph checks edges, requires explicit conversion where rates differ
fn validate(&self) -> Result<(), RateMismatch> { ... }
```

Pros: Flexible (any rate, not just enum), explicit where needed
Cons: More validation logic

**Decision:** Option C - rates as numeric values, explicit conversion nodes, graph validates. No magic `enum Rate`, no hidden conversions.

## Decisions

1. **How time reaches fields**: EvalContext (Shadertoy pattern). See above.

2. **State serialization**: Solved by recurrent graphs - feedback edges ARE the state. `GraphSnapshot { graph, feedback_state }` captures everything.

3. **Seeking stateful graphs**: User choice via enum:
   ```rust
   enum SeekBehavior {
       Resimulate,    // correct, slow - replay from start
       Discontinuity, // fast, may glitch - jump directly
       Error,         // fail-safe - refuse to seek
   }
   ```
   Default: `Discontinuity` for interactive preview, `Resimulate` for final render.

4. **Delay granularity**: Per-edge, configurable (from recurrent-graphs):
   ```rust
   enum Delay {
       Samples(u32),     // audio: z⁻ⁿ
       Frames(u32),      // animation: previous N frames
       Duration(f32),    // explicit seconds
   }
   ```

5. **Baking API**: Explicit, not automatic.
   ```rust
   // User controls when to bake (expensive operation)
   let cache = graph.bake(0.0..10.0, dt: 1.0/60.0)?;
   let value = cache.sample(4.5);  // now seekable
   ```
   Automatic baking on seek would hide expensive operations.

6. **Audio blocks**: Block = iteration unit for audio graphs. Feedback edges carry inter-block state. SIMD within blocks.

7. **Mixed rates**: Numeric sample rates (not enum), explicit conversion nodes, graph validates mismatches. No hidden up/downsampling.

## Decisions (continued)

8. **Hybrid nodes**: Not a special case. "Mostly stateless with optional smoothing" = stateless node + feedback edge.

   ```rust
   // Smooth is stateless - takes current and previous as inputs
   struct Smooth { factor: f32 }
   impl Smooth {
       fn apply(&self, current: f32, prev: f32) -> f32 {
           lerp(prev, current, self.factor)
       }
   }

   // Graph topology provides state via feedback:
   // [Input] -> [Smooth] -> [Output]
   //              ↑   │
   //              └───┘  ← feedback edge
   ```

   No new concept - feedback edges already handle this.

9. **Determinism**: Best effort by default, optional strict mode.

   ```rust
   struct EvalOptions {
       deterministic: bool,  // sacrifice perf for reproducibility
   }
   ```

   - **Default (false)**: Fast, parallel, platform-optimized. Same inputs *usually* give same outputs.
   - **Strict (true)**: Single-threaded, no fast-math, explicit seeds. Reproducible across runs/platforms.

   Full determinism is expensive. Most use cases don't need it. Strict mode for those that do (automated testing, regression checks).

## Implementation Notes

- **Rate conversion quality**: Upsample/Downsample nodes should offer configurable interpolation (linear, cubic, sinc). Default to linear for low latency, sinc for quality.

## Summary

| Model | Seekable | Parallelizable | State | Domains |
|-------|----------|----------------|-------|---------|
| Stateless | Yes | Yes | None | Textures, mesh gen, synth, rigging |
| Stateful | No | No | Internal | Filters, physics, particles |
| Streaming | No | No | Position | Audio/video streams |
| Baked | Yes | Yes | Cached | Cached simulations |

**Key insight:** Most of resin's domains are naturally stateless. Statefulness appears mainly in:
- Audio effects (filters, delays)
- Physics
- Particle systems

These might warrant special handling rather than trying to unify everything.
