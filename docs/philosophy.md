# Philosophy

Core design principles for Unshape.

## Generative Mindset

Everything in Unshape should be describable procedurally:

- **Parameters over presets** - expose knobs, don't bake decisions
- **Expressions over constants** - values can be computed, animated, or data-driven
- **Node graphs over imperative code** - composition of operations, not sequences of mutations
- **Lazy evaluation** - build descriptions, evaluate on demand
- **Operations as values** - ops are serializable structs, methods are sugar (see [ops-as-values](./design/ops-as-values.md))

## Unify, Don't Multiply

Fewer concepts = less mental load.

- One interface that handles multiple cases > separate interfaces per case
- Plugin/trait systems > hardcoded switches
- Extend existing abstractions > create parallel ones

When adding a new feature, first ask: can an existing concept handle this?

## Simplicity Over Cleverness

- Prefer stdlib over dependencies
- Functions over traits (until you need the trait)
- Explicit over implicit
- **No DSLs** - custom syntax is subjective, hard to maintain, and creates learning burden

If proposing a new dependency, ask: can existing code do this?

### Why No DSLs?

DSLs (domain-specific languages) seem appealing but carry hidden costs:

1. **Subjectivity** - syntax preferences vary wildly between users
2. **Maintenance burden** - parsers, error messages, tooling, documentation
3. **Learning curve** - users must learn new syntax on top of Rust
4. **Debugging difficulty** - DSL errors harder to trace than Rust compiler errors
5. **IDE support** - no autocomplete, no go-to-definition, no refactoring

Instead, use **Rust APIs**: builders, combinators, method chaining. These get full IDE support, type checking, and familiar syntax.

```rust
// Bad: DSL mini-notation
let pattern = parse_pattern("bd [sn cp] hh*2")?;

// Good: Rust combinator API
let pattern = cat(vec![
    pure("bd"),
    stack(vec![pure("sn"), pure("cp")]),
    pure("hh").fast(2.0),
]);
```

The Rust version is longer but: compiles with type safety, has IDE autocomplete, produces clear error messages, and requires no custom parser.

## General Internal, Constrained APIs

Store the general representation internally. Expose constrained APIs for common cases.

| Domain | General (internal) | Constrained API |
|--------|-------------------|-----------------|
| Vector | VectorNetwork | Path (degree ≤ 2) |
| Mesh | HalfEdgeMesh | IndexedMesh (no adjacency) |
| Audio | AudioGraph | Chain (linear) |
| Deformers | DeformerGraph | DeformerStack (linear) |
| Textures | TextureGraph | TextureExpr (fluent) |

Benefits:
- No loss of generality - full power available when needed
- Simpler common case - constrained APIs are easier to use
- Progressive disclosure - start simple, go general when needed

See [design/general-internal-constrained-api](./design/general-internal-constrained-api.md) for details.

## Plugin Crate Pattern

Optional, domain-specific, or heavyweight features go in plugin crates:

```
Core (always available)     Plugin (opt-in)
─────────────────────       ─────────────────────
Mesh primitives             rhi-unshape-instances (instancing)
Audio nodes                 rhi-unshape-poly (polyphony)
Rig primitives              rhi-unshape-autorig (procedural rigging)
Skeleton/skinning           rhi-unshape-anim (animation blending)
Expressions (dew)           rhi-unshape-expr-field (Field integration)
```

**Why plugins:**
- Core stays lean - don't pay for what you don't use
- Heavy dependencies isolated (ML for autorig, etc.)
- Domain-specific logic doesn't pollute core
- Users can swap implementations

**When to plugin:** If it's optional, domain-specific, or has heavy deps.

## Lazy vs Materialized

Two representations for the same concept - one lazy, one concrete:

| Lazy (description) | Materialized (data) |
|--------------------|---------------------|
| `Field<I, O>` | `Image`, `Mesh` |
| `Field<VertexData, bool>` | `SelectionSet` |
| `AudioGraph` | `AudioBuffer` |
| Expression AST | Compiled Cranelift/WGSL |

**Evaluation is explicit:**
```rust
// Lazy - describes computation
let noise: impl Field<Vec2, f32> = perlin().scale(4.0);

// Materialized - explicit call
let image: Image = noise.render(1024, 1024);
```

No hidden materializations. User controls when to pay the cost.

## Typed Build, Dynamic Execute

Compile-time safety for Rust code, runtime validation for loaded graphs:

```rust
// Building in Rust - compile-time type safety
let noise = graph.add(Perlin::new());        // Output<Field>
let render = graph.add(Render::new(1024));   // Input<Field> -> Output<Image>
graph.connect(noise.out, render.input);      // ✓ types match

// Loaded from file - runtime TypeId validation
let graph = load_graph("effect.json")?;      // validates at load time
let result = graph.execute(input)?;          // Value enum at runtime
```

Node authors write concrete types, derive macros generate dynamic wrappers.

## Graph is Artifact, Signals are Ephemeral

The graph encodes *structure and response* — how a system behaves given input. Signals (time, live audio, external data feeds, user interaction) are consumed at evaluation time but not stored in the graph. Replaying the same graph at a different time, with a different signal, produces different output: this is intentional.

**The graph is the artwork. Its expression is ephemeral.**

Implications:
- Graphs must be fully serializable — the artifact must be preservable and distributable
- Signal sources are injected by the host, not embedded in the graph
- Replay semantics are well-defined: same graph + different context = different expression, not a bug
- A graph's "content" at any moment belongs to the signal as much as the author

This principle is also a design test: if anything about Unshape makes signal sources feel like they belong *inside* the graph, or makes replay feel like a special case, the design has regressed.

## Signal-Driven Experiences

A key motivating use case for Unshape: experiences where the creator authors a *response system* — how media behaves given input — while the driving signal comes from elsewhere.

Think of a visualizer authored for a specific track. The peaks, cuts, and choreography are all responses to that music's structure. Now feed it a different track: the visualizer's latent expectations collide with input it wasn't designed for, and the result belongs to neither author. This kind of work requires genuine decoupling between the response system and the signal — something most generative tools assume away.

Unshape's node inputs are (or can be) live signals. Any input port can be driven by time, audio, sensor data, another graph's output, or anything the host provides. This isn't an add-on; it's what it means for a node graph to be fully general.

**Design test:** Can a response system authored for one signal be driven by a completely different one? If not, the decoupling isn't real.

## Realtime as a First-Class Concern

Unshape must hold under realtime constraints, not just offline batch use. There are two distinct realtime concerns with different optimization targets:

### Realtime Replay

The graph is stable; context streams in continuously. Optimize for **throughput and jitter**: can the graph keep up with the signal frame-to-frame or sample-to-sample?

- JIT compilation, SIMD, GPU offload, block processing
- Stateful nodes are fine — you're always advancing forward in time
- Seeking is generally irrelevant; the signal is live

This is the signal-driven experiences case. The `StreamingEvaluator` and block-based audio processing are architectural commitments here.

### Realtime Editing

The graph is changing; the author is tweaking parameters and expects to see the output update immediately. Optimize for **time from change to updated output**.

- Dirty tracking, incremental recomputation, aggressive caching of unchanged subgraphs
- Stateful nodes are tricky: changing a parameter upstream of a stateful node (e.g. a filter, a physics sim) may require re-simulating from the beginning to produce a correct result — or accepting a discontinuity. This is the `SeekBehavior` tradeoff in `time-models.md`.
- Seeking/scrubbing the timeline is essential

This is the media editing substrate case — the graph as project file, think `.psd` but 100× smaller and fully rewindable (at the cost of replaying from the last checkpoint).

---

These two concerns also differ in what's moving: during replay, the graph is fixed and signals flow through it; during editing, the graph itself is being mutated. Both must be designed for. Any gap where either feels like a second-class citizen is a design deficit.

Signal ingestion and output surfaces (display, speakers, network) are the host's responsibility. Unshape provides the graph substrate that consumes context and produces values.

## Host Controls Runtime

Graphs adapt to host environment, not vice versa:

```rust
struct EvalContext {
    time: f32,           // host provides
    sample_rate: u32,    // host provides
    resolution: UVec2,   // host provides
}

// Same graph works in different hosts
// - DAW at 96kHz
// - Game at 48kHz
// - Preview at 256x256
// - Final render at 4096x4096
```

Nodes query context, lazy-init buffers on first use. No rebuild needed for different contexts.

## Generic Traits Over Type Proliferation

One generic trait, not many specialized ones:

```rust
// Good: generic over geometry type
trait Rig<G: HasPositions> { ... }
trait Deformer<G> { ... }
trait Morph<G> { ... }
trait Field<I, O> { ... }

// Bad: separate traits per type
trait MeshRig { ... }
trait PathRig { ... }
trait Mesh2DRig { ... }
```

Implementations specialize; abstractions stay general.

## Core = Contract, Host = Loading

Core defines traits and serialization contracts. Plugin *loading* is the host's responsibility:

```rust
// Core provides
trait DynNode: Serialize + Deserialize { ... }
fn register_node<N: DynNode>(registry: &mut Registry);

// Host provides
fn load_plugins(path: &Path) -> Vec<Box<dyn DynNode>>;
// (wasm, dylib, statically linked - host's choice)
```

Optional adapters (`rhi-unshape-wasm-plugins`, etc.) for common loading patterns.

## Bevy Compatibility

Unshape is designed to work with the bevy ecosystem without requiring it:

- Core types use `glam` for math (same as bevy)
- Types should implement `From`/`Into` for bevy equivalents where sensible
- Individual bevy crates (e.g., `bevy_reflect`) can be used where valuable
- No hard dependency on `bevy` itself

## Workspace Structure

Implementation is split by domain, with plugin crates for optional features:

```
crates/
  # Core crates (always available)
  unshape/              # umbrella crate, re-exports
  rhi-unshape-core/         # shared primitives, Value enum, Graph
  rhi-unshape-mesh/         # 3D mesh generation, half-edge
  rhi-unshape-audio/        # audio synthesis, nodes
  rhi-unshape-texture/      # procedural textures, fields
  rhi-unshape-vector/       # 2D vector art, paths
  rhi-unshape-rig/          # rigging, bones, skinning

  # Expression integration
  rhi-unshape-expr-field/   # bridges dew expressions to Field system

  # External dependencies
  # dew (git)                 # expression AST, parsing, eval, backends

  # Plugin crates (opt-in)
  rhi-unshape-instances/    # mesh instancing
  rhi-unshape-poly/         # audio polyphony
  rhi-unshape-autorig/      # procedural rigging
  rhi-unshape-anim/         # animation blending
```

Each crate should be usable independently.
