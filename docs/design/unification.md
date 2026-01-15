# Type Unification Analysis

This document analyzes opportunities for type unification across the resin codebase - cases where multiple types represent overlapping or subset concepts that could benefit from a unified abstraction.

## Summary

| Domain | Issue | Priority | Design Doc? |
|--------|-------|----------|-------------|
| **Curves/Paths** | ~~Fragmented 2D/3D, mixed function/struct APIs~~ | ~~HIGH~~ | ✅ Done - `resin-curve` crate |
| **Graphs** | ~~"Node"/"Edge" overloaded across domains~~ | ~~MEDIUM~~ | ✅ Done - terminology in `conventions.md` |
| **Transforms** | ~~Separate 2D/3D types~~ | ~~MEDIUM~~ | ✅ Done - `SpatialTransform` trait |
| **Vertex Data** | ~~Per-subsystem Vertex structs~~ | ~~LOW~~ | ✅ Partial - traits on SoA types |
| **Mesh** | Two representations | NONE | Already unified correctly |
| **Fields** | Trait + implementations | NONE | Well-designed |
| **Effects** | ~~Audio/Graphics divergent models~~ | ~~EXPLORE~~ | ✅ Audio primitives implemented |

---

## HIGH PRIORITY

### 1. Curve/Path Representations

**Current state (fragmented):**

| Crate | Type | Purpose |
|-------|------|---------|
| resin-vector | `Path` + `PathCommand` | SVG-like 2D paths (MoveTo, LineTo, CubicTo, etc.) |
| resin-vector | `bezier.rs` functions | `quadratic_point()`, `cubic_point()`, etc. |
| resin-spline | `CubicBezier<T>` | Generic typed cubic bezier struct |
| resin-spline | `BezierSpline<T>` | Sequence of beziers |
| resin-spline | `CatmullRom<T>`, `BSpline<T>`, `Nurbs<T>` | Other spline types |
| resin-rig | `Path3D` + `PathCommand3D` | 3D version of Path (nearly identical structure) |

**Problems:**

1. `Path` (2D) and `Path3D` are nearly identical but separate implementations
2. `CubicBezier<T>` in resin-spline is a struct, but resin-vector has function-based API
3. Bezier math is implemented twice (functions vs struct methods)
4. No unified interface between linear paths and curved splines
5. Operations that work on 2D don't automatically work on 3D

**Recommended solution:**

See `docs/design/curve-types.md` for detailed design. Summary:

```rust
// Core trait (in resin-vector or new resin-curve)
pub trait Curve: Clone {
    type Point;  // Vec2 or Vec3

    fn point_at(&self, t: f32) -> Self::Point;
    fn tangent_at(&self, t: f32) -> Self::Point;
    fn bounding_box(&self) -> Bounds<Self::Point>;
    fn to_cubics(&self) -> Vec<CubicBezier<Self::Point>>;

    // Default implementations
    fn length(&self) -> f32 { /* adaptive integration */ }
    fn flatten(&self, tolerance: f32) -> Vec<Self::Point> { /* subdivision */ }
}

// Concrete types implement trait
impl Curve for Line<Vec2> { ... }
impl Curve for CubicBezier<Vec2> { ... }
impl Curve for Arc { ... }

// Enum for mixed paths (single match point per method)
pub enum Segment<V> {
    Line(Line<V>),
    Quad(QuadBezier<V>),
    Cubic(CubicBezier<V>),
    Arc(Arc<V>),  // 2D only, or generalize
}

impl<V> Curve for Segment<V> where ... { ... }

// Path generic over segment type
pub struct Path<C: Curve = Segment<Vec2>> {
    segments: Vec<C>,
    closed: bool,
}
```

**Migration path:**

1. Create `Curve` trait in resin-vector (or new resin-curve crate)
2. Implement for existing types (`CubicBezier`, etc.)
3. Create `Segment<V>` enum with trait impl
4. Make `Path<C>` generic, default to `Segment<Vec2>`
5. Deprecate `Path3D`, replace with `Path<Segment<Vec3>>`
6. Move/consolidate bezier functions into `CubicBezier` impl

---

## MEDIUM PRIORITY

### 2. Graph/Node/Edge Terminology

**Status: ✅ Complete**

Established clear terminology across domains:

| Domain | Type | Meaning |
|--------|------|---------|
| Data Flow (resin-core) | `Node` | Processing unit with typed inputs/outputs |
| Data Flow (resin-core) | `Wire` | Port-to-port connection |
| Vector Graphics (resin-vector) | `Anchor` | 2D position where curves meet |
| Vector Graphics (resin-vector) | `Edge` | Bezier curve connecting anchors |
| Spatial Networks (resin-procgen) | `NetworkNode` | Position in roads/rivers |
| Topology (resin-mesh) | `Vertex` | 3D position with attributes |
| Topology (resin-mesh) | `HalfEdge` | Directional edge for traversal |
| Skeletal (resin-rig) | `Bone` | Joint in skeletal hierarchy |

See `docs/conventions.md` for the full terminology guide.

### 3. Transform Types

**Status: ✅ Complete**

Added `SpatialTransform` trait in `resin-transform` crate with implementations in `resin-rig` (Transform) and `resin-motion` (Transform2D).

```rust
pub trait SpatialTransform {
    type Vector: Copy;   // Vec2 or Vec3
    type Rotation: Copy; // f32 or Quat
    type Matrix: Copy;   // Mat3 or Mat4

    fn translation(&self) -> Self::Vector;
    fn rotation(&self) -> Self::Rotation;
    fn scale(&self) -> Self::Vector;
    fn to_matrix(&self) -> Self::Matrix;
    fn transform_point(&self, point: Self::Vector) -> Self::Vector;
}
```

**Implementation:**
- `Transform3D`: `Vector=Vec3`, `Rotation=Quat`, `Matrix=Mat4`
- `Transform2D`: `Vector=Vec2`, `Rotation=f32`, `Matrix=Mat3`

This enables generic algorithms over transforms while preserving domain-specific features (2D anchor/skew, 3D quaternion rotation).

---

## LOW PRIORITY

### 4. Vertex Attribute Types

**Status: ✅ Partially complete**

**Implemented:**

The attribute traits in `resin-core` are now implemented where data layout allows:

| Type | HasPositions | HasNormals | HasColors | HasIndices |
|------|--------------|-----------|-----------|-----------|
| `Mesh` | ✅ | ✅ | ❌ | ✅ |
| `PointCloud` | ✅ | ✅ | ✅ | ❌ |

Added `HasPositions2D` trait for 2D geometry types.

**Design Limitation: SoA vs AoS**

The traits require returning slices (`&[Vec3]`), which only works with **Struct-of-Arrays (SoA)** storage:

```rust
// SoA - CAN implement traits (returns &[Vec3] slice)
struct PointCloud {
    positions: Vec<Vec3>,  // ✅ Can return &self.positions
    normals: Vec<Vec3>,
}

// AoS - CANNOT implement traits (no contiguous slice)
struct SoftBody {
    vertices: Vec<SoftVertex>,  // ❌ Cannot return &[Vec3] from Vec<SoftVertex>
}
```

**Types that cannot implement traits:**

| Type | Reason |
|------|--------|
| `SoftBody` | Stores `Vec<SoftVertex>` (AoS) |
| `GradientMesh` | Stores `Vec<GradientVertex>` (AoS) |
| `HalfEdgeMesh` | Internal topology type, uses `Vertex` struct |

**Why this is acceptable:**

1. The main indexed types (`Mesh`, `PointCloud`) use SoA and implement traits
2. AoS types like `SoftBody` store additional per-vertex state (velocity, mass) that doesn't fit generic patterns
3. Converting these to SoA would require breaking changes for marginal benefit
4. The traits enable generic algorithms on the types that matter most for GPU/rendering

---

## ALREADY WELL-DESIGNED

### Mesh Representations

**Status: Good example of unification done right**

| Type | Purpose |
|------|---------|
| `HalfEdgeMesh` | Topology-rich, for editing operations |
| `Mesh` | Indexed arrays, for GPU rendering |

Clear conversions exist:
- `HalfEdgeMesh::from_mesh(&mesh)` - convert for editing
- `halfedge.to_mesh()` - convert for rendering

This follows the "general-internal-constrained-api" pattern documented in `docs/design/general-internal-constrained-api.md`.

### Field System

**Status: Well-designed composition**

| Crate | Purpose |
|-------|---------|
| resin-field | `Field<I, O>` trait, combinators |
| resin-noise | Noise function implementations |
| resin-expr-field | Expression-based field building |

The trait-based design allows composition without type proliferation.

---

## Implementation Priority

1. ~~**Curves** (HIGH)~~ - ✅ Complete - `resin-curve` crate with `Curve` trait
2. ~~**Graph terminology** (MEDIUM)~~ - ✅ Complete - renamed types and documented in `conventions.md`
3. ~~**Transforms** (MEDIUM)~~ - ✅ Complete - `resin-transform` crate with `SpatialTransform` trait
4. ~~**Vertex attributes** (LOW)~~ - ✅ Partial - traits implemented on SoA types (`Mesh`, `PointCloud`); AoS types documented as out-of-scope

---

## EXPLORATION NEEDED

### 5. Audio/Graphics Effects

**Status: Implemented (audio primitives)**

Both domains process signals through chains of effects, but use different execution models.

**Current state:**

| Aspect | Audio (`resin-audio`) | Graphics (`resin-field`) |
|--------|----------------------|--------------------------|
| Core trait | `AudioNode` | `Field<I, O>` |
| Execution | Eager (every sample) | Lazy (on demand) |
| State | Stateful (delay lines, filters) | Stateless (pure functions) |
| Domain | Time (sequential samples) | Space (parallel coordinates) |
| Composition | `Chain` (linear), `AudioGraph` (DAG) | Method chaining (`.map()`, `.add()`) |
| Context | `AudioContext` (sample_rate, time) | `EvalContext` |

**Already unified:**

- Both use `resin-op` for serialization (operations-as-values pattern)
- Both register with `OpRegistry` for dynamic pipelines
- Both follow general-internal-constrained-api pattern

**Fundamental differences:**

1. **Statefulness** - Audio effects *require* history buffers (reverb tails, delay lines, filter coefficients). Graphics fields are pure `sample(coord) -> value` functions.

2. **Execution model** - Audio must process every sample eagerly at audio rate (~44100Hz). Graphics can evaluate lazily at arbitrary coordinates.

3. **Parallelism** - Audio is inherently sequential (sample N depends on sample N-1 for stateful effects). Graphics is embarrassingly parallel (each pixel independent).

**Audio effect primitive decomposition:**

Analysis of `resin-audio/src/effects.rs` reveals that effects are *not* orthogonal - they're compositions of a small set of shared primitives:

| Primitive | Description | Used by |
|-----------|-------------|---------|
| **Delay line** | Circular buffer with read/write | Reverb, Chorus, Flanger, Limiter |
| **LFO** | Phase accumulator → waveform | Chorus, Flanger, Phaser, Tremolo |
| **Envelope follower** | Attack/release smoothing | Compressor, Limiter, NoiseGate |
| **Filter** | Biquad, allpass, comb | Reverb, Distortion, Phaser |
| **Waveshaper** | Transfer function | Distortion, Bitcrusher |
| **Feedback** | Output → input routing | Reverb, Chorus, Flanger, Phaser |
| **Mix** | Dry/wet blend | Almost all effects |

Effect decomposition into primitives:

```
Chorus   = Delay + (LFO → delay_time) + Feedback + Mix
Flanger  = Delay + (LFO → delay_time) + Feedback + Mix  // same as Chorus, different params!
Phaser   = Allpass[] + (LFO → filter_freq) + Feedback + Mix
Tremolo  = LFO → amplitude  // simplest modulation effect
Reverb   = Comb[] parallel + Allpass[] series + Mix
         where Comb = Delay + Feedback + LPF (damping)
Compressor = EnvelopeFollower → GainControl
Limiter    = Lookahead + EnvelopeFollower → GainControl
NoiseGate  = EnvelopeFollower → GateControl
Distortion = Waveshaper + Filter (tone)
Bitcrusher = SampleHold + Quantize
```

**Key insight:** Chorus and Flanger are *literally the same effect* with different default parameters. Most "modulation effects" follow the pattern `LFO → some_parameter`.

**Implications for unification:**

This suggests a primitive-based architecture could work for both domains:

| Audio Primitive | Graphics Equivalent |
|-----------------|---------------------|
| Delay line | Texture buffer / history |
| LFO | Animated parameter / time-varying field |
| Envelope follower | (no direct equivalent - stateless) |
| Filter (blur in frequency) | Convolution kernel / blur |
| Waveshaper | Color curve / transfer function |
| Feedback | Recursive field evaluation |
| Mix | Field blending (lerp, add, multiply) |

Graphics already has some of these as `Field` combinators (`.map()` = waveshaper, `.mix()` = mix). Missing: blur/convolution, feedback/recursion.

**Potential unification opportunities:**

1. **Modulation routing** - Audio has `ModSource` → parameter mapping (LFO, envelope, velocity to any parameter). Graphics could use same pattern for animating field parameters over time.

2. **Effect library parity** - Audio has rich effects (`Reverb`, `Chorus`, `Distortion`, `Compressor`, etc.). Graphics lacks equivalent high-level effects (`Blur`, `Bloom`, `ColorGrade`, `Sharpen`).

3. **Pipeline abstraction** - Both want to chain operations. Could there be a shared `Pipeline<T>` validation/serialization layer?

4. **Parameter presets** - Audio has `SynthPatch` / `PatchBank` for preset management. Graphics could benefit from similar.

**Questions to resolve:**

- Should we refactor audio effects to expose primitives (Delay, LFO, EnvelopeFollower) as composable building blocks?
- Can graphics adopt the same primitive vocabulary where applicable (blur = convolution, color curves = waveshaper)?
- Is the eager/lazy distinction fundamental, or could audio primitives be made lazy for offline processing?
- Should LFO / animated parameters be a shared abstraction for both domains?

**Performance considerations:**

Decomposing effects into primitives has potential overhead:

| Concern | Impact | Mitigation |
|---------|--------|------------|
| Function call overhead | Negligible | Rust inlines aggressively, especially generics |
| Cache locality | Moderate | Keep primitive state in contiguous struct |
| Virtual dispatch | Moderate | Use concrete types, not `Box<dyn Primitive>` |

Actual cost drivers in audio are `sin()` calls and buffer cache misses, not abstraction. Decomposition could *help* via control-rate LFO batching and SIMD.

**Approach:** Benchmark current implementation, save results, then refactor and compare. Avoid maintaining parallel implementations.

**Dynamic dispatch strategy:**

Current `resin-audio` uses dyn at the right level:

```
Chain level:     Vec<Box<dyn AudioNode>>  ← dyn (runtime flexible)
Within effects:  concrete fields          ← static (performance)
```

Use cases for dyn: runtime-configurable chains, serialization, plugin systems, node graphs.

NOT needed within effects - primitives should be concrete generic types that compose into effect structs, which implement `AudioNode` for chain-level dyn dispatch.

**Implementation results:**

Audio primitives implemented in `resin-audio/src/primitive.rs`:
- `DelayLine<INTERP>` - const generic for interpolation
- `PhaseOsc` - phase accumulator + waveforms
- `EnvelopeFollower` - attack/release smoothing
- `Allpass1` - first-order allpass for phasers
- `Smoother` - one-pole parameter smoothing
- `Mix` - dry/wet blending

Composition structs replace monolithic effects:

| Composition | Replaces | Performance |
|-------------|----------|-------------|
| `ModulatedDelay` | Chorus, Flanger | Same |
| `AmplitudeMod` | Tremolo | **2.4x faster** |
| `AllpassBank` | Phaser | **15% faster** |

Constructor functions (`chorus()`, `flanger()`, `tremolo()`, `phaser()`) provide familiar API.

**Remaining work:**
- DynamicsProcessor (Compressor/Limiter/NoiseGate) - different envelope targets, Limiter needs lookahead
- Reverb - complex internal CombFilter/AllpassFilter structure
- Graphics primitives (blur/convolution, feedback/recursion)

**Decision:** Primitive-based architecture validated. Effects are constructor functions returning composition structs.

---

## Related Documents

- `curve-types.md` - Detailed curve trait design
- `general-internal-constrained-api.md` - Pattern for internal vs public types
- `normalization.md` - Code style consistency (completed)
- `conventions.md` - Coordinate systems and naming conventions
- `ops-as-values.md` - Serializable operations pattern (used by both audio and graphics)
- `../domains/audio.md` - Audio domain overview
