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
[Noise (stateless)] → [Filter (stateful)] → [Output]
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

## Open Questions

1. **Audio block boundaries**: Processing happens in blocks (128-1024 samples), but graph model is conceptually per-sample. How to reconcile? See [recurrent-graphs](./recurrent-graphs.md) for related discussion.

2. **Mixed rates**: Audio at 48kHz, control/animation at 60Hz. How do feedback edges work across rate boundaries? Interpolation? Sample-and-hold?

3. **Hybrid nodes**: Nodes that are "mostly stateless" with optional smoothing/filtering. Explicit state input, or implicit via context?

4. **Baking API**: How does user trigger simulation bake? Explicit `graph.bake(0.0..10.0, dt)` or automatic when seeking stateful graph?

5. **Determinism**: Floating point reproducibility across platforms. Threading order. Probably punt to "best effort" with optional strict mode.

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
